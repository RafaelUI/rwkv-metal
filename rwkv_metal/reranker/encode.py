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
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

from ..model.state import RWKVState, build_mask
from .data import PairTemplate, RerankSample


@dataclass
class StateCache:
    """states:     [n_pairs, n_src, H, S, S] — состояния пар
    pair_index: [n_samples, n_cand] — индекс строки в states
    labels:     [n_samples] — позиция правильного кандидата
    """
    states: mx.array
    pair_index: mx.array
    labels: mx.array

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
        return self.states.nbytes

    def save(self, path: str):
        mx.save_safetensors(path, {
            "states": self.states,
            "pair_index": self.pair_index,
            "labels": self.labels,
        })

    @staticmethod
    def load(path: str) -> "StateCache":
        d = mx.load(path)
        return StateCache(d["states"], d["pair_index"], d["labels"])


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
                 verbose: bool = True) -> StateCache:
    """Свернуть все пары (кандидат, запрос) в кэш состояний.

    doc_batch / query_batch подобраны под 16 ГБ unified memory: префиксы
    длинные и держат много активаций, хвосты короткие.

    dtype: в чём хранить кэш. fp16 (умолчание) — вдвое меньше fp32 и точнее
    bf16 при том же размере: 10 бит мантиссы против 7. Узкий диапазон fp16
    здесь не мешает — состояния на порядки ниже потолка 65504, и выход за
    него проверяется явно. fp32 — если хочется без компромиссов, bf16 — если
    состояния почему-то огромные.
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
    # Буфер — numpy на хосте, а не mx.array. Запись `out[rows] = sel` в MLX
    # это scatter, порождающий новый массив целиком: на кэше в пару гигабайт
    # это лишние гигабайты транзиентной памяти на КАЖДОЙ пачке. numpy пишет
    # на месте.
    # fp16 (а не bf16) выбран сознательно: 10 бит мантиссы против 7, а
    # диапазон здесь заведомо безопасен (состояния 0.1B доходят до ~50 при
    # потолке fp16 в 65504). Проверка на выход за диапазон — ниже.
    np_dtype = {mx.float16: np.float16, mx.float32: np.float32}.get(dtype)
    host = np_dtype is not None
    if host:
        out_np = np.zeros((n_pairs, n_src, cfg.n_head, cfg.head_size,
                           cfg.head_size), dtype=np_dtype)
        out = None
    else:
        out_np = None
        out = mx.zeros((n_pairs, n_src, cfg.n_head, cfg.head_size,
                        cfg.head_size), dtype=dtype)
    max_abs = 0.0
    rows_done = 0

    if verbose:
        print(f"пар {n_pairs}, уникальных префиксов {len(prefix_jobs)} "
              f"(экономия {1 - len(prefix_jobs)/max(1,n_pairs):.0%}), "
              f"кэш {out.nbytes/1e9:.2f} ГБ")

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
            rows = [p[1] for p in part]
            if host:
                arr = np.array(sel.astype(mx.float32))
                max_abs = max(max_abs, float(np.abs(arr).max()))
                out_np[rows] = arr.astype(np_dtype)
            else:
                mx.eval(sel)
                out[mx.array(np.array(rows, dtype=np.int32))] = sel.astype(dtype)
                mx.eval(out)
            rows_done += len(part)

        if verbose and (start // doc_batch) % 20 == 0:
            el = time.time() - t0
            frac = (start + len(chunk)) / len(prefix_jobs)
            print(f"  префиксы {start + len(chunk)}/{len(prefix_jobs)} | "
                  f"пары {rows_done}/{n_pairs} | {el:.0f}s | "
                  f"осталось ~{el/max(1e-9,frac)*(1-frac):.0f}s", flush=True)

    if host:
        if np_dtype is np.float16 and max_abs > 60000:
            raise OverflowError(
                f"состояния доходят до {max_abs:.0f}, fp16 обрежется на 65504. "
                "Передай dtype=mx.float32 (вдвое больше памяти) или "
                "dtype=mx.bfloat16 (шире диапазон, грубее мантисса)."
            )
        out = mx.array(out_np)

    if verbose:
        print(f"готово за {time.time()-t0:.0f}s, макс|состояние| {max_abs:.1f}")

    return StateCache(
        states=out,
        pair_index=mx.array(pair_index),
        labels=mx.array(np.array([s.label for s in samples], dtype=np.int32)),
    )


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
