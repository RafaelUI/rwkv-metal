"""
rwkv_metal.reranker.encode
==========================
Свёртка пар (документ, запрос) в состояния — и кэш этих состояний.

Зачем кэш
---------
База заморожена, значит отображение «текст пары → состояние» фиксировано.
Тогда обучение головы не обязано каждый шаг пересчитывать длинный проход по
базе: пары можно сосчитать ОДИН раз, а дальше учить голову на готовых
состояниях. Голова крошечная (один-два RWKV-блока на одном токене), поэтому
эпоха по десяткам тысяч пар занимает секунды вместо десятков минут — и
становится возможным то, ради чего это всё: большие батчи, много эпох,
честный подбор гиперпараметров на ноутбуке.

Хранится не всё состояние, а только слои, которые читает голова
(`RerankerHead.unique_sources`). Для умолчания (последний слой) это 1 слой
из 12: 98 КБ на пару в bf16 против 2.4 МБ на полное состояние в fp32 — в 25
раз меньше.

Как считается
-------------
Документов много, запросов на документ — мало, поэтому порядок работы такой:

    1. кодируем ПРЕФИКС «Instruct + Document» (дорого, O(L_doc)), пачками;
    2. продолжаем каждый префикс всеми нужными хвостами «Query: ...»
       (дёшево, O(L_query)) — префикс уже в состоянии, пересчитывать нечего;
    3. от финального состояния оставляем только читаемые головой слои.

Замер на M4 Air (0.1B): шаг 1 — ~73 мс на документ из 512 токенов, шаг 2 —
~3.5 мс на пару. То есть добавить запросу ещё восемь кандидатов почти
бесплатно, если документы уже в пуле.
"""
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

from ..model.state import RWKVState, build_mask
from .data import PairTemplate, RerankSample


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    return np.array(x.astype(mx.float32) if x.dtype == mx.bfloat16 else x)


@dataclass
class StateCache:
    """states:     [n_pairs, n_src, H, S, S] — состояния пар, numpy
    pair_index: [n_samples, n_cand] — индекс строки в states, numpy
    labels:     [n_samples] — позиция правильного кандидата, numpy

    Почему numpy, а не mx.array
    ---------------------------
    Кэш — это гигабайты, а трогаем мы за шаг сотни строк. Держать его
    mx.array означает, во-первых, лишнюю полную копию при создании (numpy-
    буфер → mx), во-вторых, отсутствие способа НЕ держать его в памяти
    целиком. numpy решает и то, и другое: батч набирается numpy-гейтерингом
    (дёшево, локально) и превращается в mx.array уже в своём размере, а сам
    массив может быть memmap — тогда кэш вообще не обязан влезать в RAM,
    страницы подтягиваются по мере обращения.

    Это не вкусовщина: на 16 ГБ машине кэш в паре гигабайт плюс база плюс
    транзиентная копия — это своп, а своп портит не только скорость, но и
    все замеры, снятые в это время.
    """
    states: np.ndarray
    pair_index: np.ndarray
    labels: np.ndarray

    def __post_init__(self):
        # приходящие mx.array (старые вызовы, тесты) приводим к numpy
        self.states = _to_numpy(self.states)
        self.pair_index = _to_numpy(self.pair_index).astype(np.int32)
        self.labels = _to_numpy(self.labels).astype(np.int32)

    @property
    def n_pairs(self) -> int:
        return self.states.shape[0]

    @property
    def n_samples(self) -> int:
        return self.pair_index.shape[0]

    @property
    def n_cand(self) -> int:
        return self.pair_index.shape[1]

    def nbytes(self) -> int:
        return int(self.states.nbytes)

    @property
    def mmapped(self) -> bool:
        return isinstance(self.states, np.memmap)

    def gather(self, rows: np.ndarray) -> mx.array:
        """Строки кэша → mx.array. Единственное место, где данные попадают
        в MLX, и попадают ровно в размере батча."""
        return mx.array(np.ascontiguousarray(self.states[rows]))

    def save(self, path: str):
        """`path` — .npy для состояний; рядом ляжет .idx.npz с индексами.

        Если состояния УЖЕ являются memmap'ом этого же файла (кэш строился с
        out_path), переписывать нечего: достаточно сбросить страницы. Наивный
        `np.save` в этом случае открыл бы целевой файл на запись, обрезав его
        под тем самым memmap'ом, из которого читает, — данные портятся, а
        numpy при этом успевает вытянуть весь массив в память.
        """
        path = str(path)
        base = path[:-4] if path.endswith(".npy") else path
        target = os.path.abspath(base + ".npy")
        src = os.path.abspath(getattr(self.states, "filename", "") or "")
        if isinstance(self.states, np.memmap) and src == target:
            self.states.flush()
        else:
            np.save(target, self.states)
        np.savez(base + ".idx.npz", pair_index=self.pair_index,
                 labels=self.labels)

    @staticmethod
    def load(path: str, mmap: bool = True) -> "StateCache":
        """mmap=True (умолчание): состояния читаются страницами с диска, в
        RAM живёт только то, что реально трогали."""
        path = str(path)
        base = path[:-4] if path.endswith(".npy") else path
        states = np.load(base + ".npy", mmap_mode="r" if mmap else None)
        idx = np.load(base + ".idx.npz")
        return StateCache(states, idx["pair_index"], idx["labels"])


def _encode_prefix_ids(tokenizer, template: PairTemplate, instruct: str,
                        doc: str, max_doc_tokens: int) -> List[int]:
    """Токены префикса. Куски токенизируются по отдельности, чтобы обрезка
    документа была ровно по токенам, а не по символам."""
    if template.doc_first:
        head = tokenizer.encode(f"Instruct: {instruct}\nDocument: ")
        body = tokenizer.encode(doc)[:max_doc_tokens]
        return head + body + tokenizer.encode("\n")
    head = tokenizer.encode(f"Instruct: {instruct}\n")
    return head


def _encode_suffix_ids(tokenizer, template: PairTemplate, doc: str, query: str,
                        max_query_tokens: int, max_doc_tokens: int,
                        terminator: Optional[int]) -> List[int]:
    if template.doc_first:
        ids = tokenizer.encode("Query: ") + tokenizer.encode(query)[:max_query_tokens]
    else:
        ids = (tokenizer.encode("Query: ") + tokenizer.encode(query)[:max_query_tokens]
               + tokenizer.encode("\nDocument: ") + tokenizer.encode(doc)[:max_doc_tokens])
    if terminator is not None:
        ids = ids + [terminator]
    return ids


def _batch_ids(seqs: Sequence[Sequence[int]], pad: int = 0):
    """Right-padding + маска + позиции последних реальных токенов."""
    lens = [len(s) for s in seqs]
    T = max(lens)
    arr = np.full((len(seqs), T), pad, dtype=np.int32)
    for i, s in enumerate(seqs):
        arr[i, :len(s)] = s
    return (mx.array(arr), build_mask(lens, T),
            mx.array(np.array([L - 1 for L in lens], dtype=np.int32)))


def encode_pairs(reranker, tokenizer, pool: Sequence[str],
                 samples: Sequence[RerankSample],
                 template: PairTemplate = None,
                 max_doc_tokens: int = 384,
                 max_query_tokens: int = 96,
                 terminator: Optional[int] = 0,
                 doc_batch: int = 8,
                 query_batch: int = 16,
                 dtype=mx.float16,
                 out_path: str = None,
                 verbose: bool = True) -> StateCache:
    """Свернуть все пары (кандидат, запрос) в кэш состояний.

    doc_batch / query_batch подобраны под 16 ГБ unified memory: префиксы
    длинные и держат много активаций, хвосты короткие.

    dtype: в чём хранить кэш. fp16 (умолчание) — вдвое меньше fp32 и точнее
    bf16 при том же размере: 10 бит мантиссы против 7. Узкий диапазон fp16
    здесь не мешает — состояния на порядки ниже потолка 65504, и выход за
    него проверяется явно. fp32 — если хочется без компромиссов.

    out_path: писать состояния сразу в .npy на диск (memmap). Тогда кэш не
    занимает RAM ни при построении, ни при обучении — страницы подтягиваются
    по обращению. Для кэшей, сравнимых с объёмом памяти, это разница между
    «работает» и «машина ушла в своп».
    """
    template = template or PairTemplate()
    t0 = time.time()

    # (instruct, doc_id) → строка префикса. Инструкций в корпусе единицы,
    # поэтому дедупликация по паре, а не по документу, почти ничего не стоит.
    prefix_key: Dict[Tuple[str, int], int] = {}
    prefix_jobs: List[Tuple[str, int]] = []
    # для каждого префикса — какие пары его ждут: (pair_row, query, doc_id).
    # doc_id хранится отдельно от ключа префикса: при doc_first=False документ
    # живёт в ХВОСТЕ, а префикс у всех пар один.
    pending: List[List[Tuple[int, str, int]]] = []

    n_cand = len(samples[0].doc_ids)
    if any(len(s.doc_ids) != n_cand for s in samples):
        raise ValueError("у всех примеров должно быть одинаковое число кандидатов")
    pair_index = np.zeros((len(samples), n_cand), dtype=np.int32)
    n_pairs = 0
    for si, s in enumerate(samples):
        for ci, did in enumerate(s.doc_ids):
            # при doc_first=False префикс от документа не зависит, поэтому
            # ключ по документу дал бы N одинаковых префиксов вместо одного
            key = (s.instruct, did if template.doc_first else -1)
            pi = prefix_key.get(key)
            if pi is None:
                pi = len(prefix_jobs)
                prefix_key[key] = pi
                prefix_jobs.append(key)
                pending.append([])
            pending[pi].append((n_pairs, s.query, did))
            pair_index[si, ci] = n_pairs
            n_pairs += 1

    n_src = len(reranker.head.unique_sources)
    cfg = reranker.base.config
    # Буфер — numpy, а не mx.array. Запись `out[rows] = sel` в MLX это
    # scatter, порождающий массив целиком заново: на кэше в пару гигабайт это
    # лишние гигабайты транзиентной памяти на КАЖДОЙ пачке. numpy пишет на
    # месте, а с out_path пишет сразу на диск и в RAM не живёт вообще.
    #
    # fp16 (а не bf16) выбран сознательно: 10 бит мантиссы против 7, а
    # диапазон здесь с запасом (состояния 0.1B доходят до ~50 при потолке
    # 65504). Проверка на выход за диапазон — ниже.
    np_dtype = {mx.float16: np.float16, mx.float32: np.float32}.get(dtype)
    if np_dtype is None:
        raise ValueError(
            f"dtype={dtype} не поддерживается кэшем: нужен mx.float16 "
            "(умолчание) или mx.float32. bf16 у numpy нет, а хранить кэш в "
            "MLX — значит держать его в памяти целиком."
        )
    shape = (n_pairs, n_src, cfg.n_head, cfg.head_size, cfg.head_size)
    if out_path:
        out_path = str(out_path)
        if not out_path.endswith(".npy"):
            out_path += ".npy"
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        out_np = np.lib.format.open_memmap(out_path, mode="w+",
                                           dtype=np_dtype, shape=shape)
    else:
        out_np = np.zeros(shape, dtype=np_dtype)
    max_abs = 0.0
    rows_done = 0

    if verbose:
        print(f"пар {n_pairs}, уникальных префиксов {len(prefix_jobs)} "
              f"(экономия {1 - len(prefix_jobs)/max(1,n_pairs):.0%}), "
              f"кэш {out_np.nbytes/1e9:.2f} ГБ"
              f"{' (memmap на диске)' if out_path else ''}")

    for start in range(0, len(prefix_jobs), doc_batch):
        chunk = prefix_jobs[start:start + doc_batch]
        seqs = [_encode_prefix_ids(tokenizer, template, instruct, pool[did],
                                   max_doc_tokens)
                for instruct, did in chunk]
        idx, mask, end_idx = _batch_ids(seqs)
        st = reranker.encode(idx, mask=mask, end_idx=end_idx)
        st.eval()

        # все хвосты для этой пачки префиксов
        jobs = [(local, pair_row, query, did)
                for local in range(len(chunk))
                for pair_row, query, did in pending[start + local]]

        for qs in range(0, len(jobs), query_batch):
            part = jobs[qs:qs + query_batch]
            locals_ = mx.array(np.array([p[0] for p in part], dtype=np.int32))
            sub = st[locals_]
            qseqs = [_encode_suffix_ids(tokenizer, template, pool[p[3]], p[2],
                                        max_query_tokens, max_doc_tokens,
                                        terminator)
                     for p in part]
            qidx, qmask, qend = _batch_ids(qseqs)
            st_pair = reranker.encode(qidx, mask=qmask, end_idx=qend, state=sub)
            sel = reranker.select(st_pair)                   # [b, n_src, H, S, S]
            arr = np.array(sel.astype(mx.float32))
            max_abs = max(max_abs, float(np.abs(arr).max()))
            out_np[[p[1] for p in part]] = arr.astype(np_dtype)
            rows_done += len(part)

        if verbose and (start // doc_batch) % 20 == 0:
            el = time.time() - t0
            frac = (start + len(chunk)) / len(prefix_jobs)
            print(f"  префиксы {start + len(chunk)}/{len(prefix_jobs)} | "
                  f"пары {rows_done}/{n_pairs} | {el:.0f}s | "
                  f"осталось ~{el/max(1e-9,frac)*(1-frac):.0f}s", flush=True)

    if np_dtype is np.float16 and max_abs > 60000:
        raise OverflowError(
            f"состояния доходят до {max_abs:.0f}, fp16 обрежется на 65504. "
            "Передай dtype=mx.float32 (вдвое больше памяти)."
        )
    if out_path:
        out_np.flush()

    if verbose:
        print(f"готово за {time.time()-t0:.0f}s, макс|состояние| {max_abs:.1f}")

    cache = StateCache(
        states=out_np,
        pair_index=pair_index,
        labels=np.array([s.label for s in samples], dtype=np.int32),
    )
    if out_path:
        cache.save(out_path)          # состояния уже на диске, пишутся индексы
    return cache


def encode_pairs_direct(reranker, tokenizer, pairs: Sequence[Tuple[str, str, str]],
                        template: PairTemplate = None,
                        max_doc_tokens: int = 384,
                        max_query_tokens: int = 96,
                        terminator: Optional[int] = 0,
                        batch_size: int = 8) -> mx.array:
    """Сплошной путь без кэша префиксов: (instruct, doc, query) → состояния.

    Нужен там, где кэшировать нечего — разовый скоринг, тесты, сравнение
    с `encode_pairs` на эквивалентность.
    """
    template = template or PairTemplate()
    outs = []
    for start in range(0, len(pairs), batch_size):
        part = pairs[start:start + batch_size]
        seqs = []
        for instruct, doc, query in part:
            ids = (_encode_prefix_ids(tokenizer, template, instruct, doc, max_doc_tokens)
                   + _encode_suffix_ids(tokenizer, template, doc, query,
                                        max_query_tokens, max_doc_tokens, terminator))
            seqs.append(ids)
        idx, mask, end_idx = _batch_ids(seqs)
        st = reranker.encode(idx, mask=mask, end_idx=end_idx)
        outs.append(reranker.select(st))
        mx.eval(outs[-1])
    return mx.concatenate(outs, axis=0)
