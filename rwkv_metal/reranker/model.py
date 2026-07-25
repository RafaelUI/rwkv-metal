"""
rwkv_metal.reranker.model
=========================
Cross-encoder реранкер поверх RWKV-7.

Идея (по мотивам howard-hou/EmbeddingRWKV, но не построчный порт)
-----------------------------------------------------------------
Классический cross-encoder гоняет пару (запрос, документ) через трансформер
и снимает скор с [CLS]. У RWKV вся история пары уже свёрнута в рекуррентное
состояние фиксированного размера `h [H, S, S]` на слой. Значит, скор можно
взять НЕ из потокенных активаций, а прямо из состояния: запустить поверх
него несколько обучаемых токенов-зондов через маленький стек RWKV-блоков и
спроецировать выход в скаляр.

Что это даёт:

  * Голова читает состояние, а не последний hidden state, — то есть видит
    свёрнутую матрицу [S, S] на слой, а не вектор [D]. Считывание
    `y = h·r` — это, по сути, один шаг внимания к содержимому состояния.
  * Стоимость головы не зависит от длины пары: один-два токена на блок.
  * Состояние префикса кэшируется. При шаблоне «документ → запрос»
    состояние документа считается один раз, а на каждый запрос остаётся
    O(L_query) вместо O(L_doc + L_query). Замер на M4 Air (0.1B, 8 пар):
    полный проход 512 токенов — 584 мс, продолжение на 16 токенов с
    кэша — 28 мс.

База заморожена. Обучается только голова: несколько RWKV-блоков,
инициализированных весами выбранных слоёв базы, + MLP в скаляр.

Отличия от оригинала
--------------------
  * Right-padding с точной маской вместо left-padding: пад-токены сделаны
    no-op для рекуррентности, поэтому состояние строки не зависит ни от
    паддинга, ни от соседей по батчу (см. `RWKV_Tmix_x070.__call__`).
  * Состояние снимается на позиции последнего РЕАЛЬНОГО токена строки
    (`end_idx`), а не в конце паддинга.
  * Голова zero-init: на нулевом шаге все скоры равны нулю, стартовый лосс
    в точности ln(N+1) для listwise — удобно ловить сломанное обучение.
  * n_probe > 1: несколько токенов-зондов, если одного чтения состояния мало.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn

from ..model.rwkv7_x070 import RWKVBlock, RWKV7X070
from ..model.state import RWKVState


def resolve_layer_indices(layer_idx: Sequence[int], n_layer: int) -> List[int]:
    """Нормализует индексы слоёв базы (поддерживает отрицательные)."""
    out = []
    for i in layer_idx:
        j = i + n_layer if i < 0 else i
        if not 0 <= j < n_layer:
            raise ValueError(f"слой {i} вне диапазона для модели из {n_layer} слоёв")
        out.append(j)
    if not out:
        raise ValueError("layer_idx пуст")
    return out


@dataclass
class RerankerConfig:
    """layer_idx:     какие слои базы читает голова. Один блок на индекс.
                      Умолчание (-1,) — только последний слой: самое дешёвое
                      и обычно достаточное. Больше слоёв = больше сигнала с
                      разных уровней абстракции, но и больше параметров.
    shared_state:     все блоки головы читают состояние ПОСЛЕДНЕГО слоя базы,
                      а глубина стека остаётся len(layer_idx). Способ
                      углубить голову, не трогая, откуда она читает.
    n_probe:          сколько обучаемых токенов проходит через голову. Каждый
                      токен — ещё одно чтение состояния; скор снимается с
                      последнего. 1 — как в оригинале.
    head_hidden:      ширина скрытого слоя MLP-головы (по умолчанию n_embd).
    """
    layer_idx: tuple = (-1,)
    shared_state: bool = False
    n_probe: int = 1
    head_hidden: Optional[int] = None


class RerankerHead(nn.Module):
    """Стек RWKV-блоков поверх состояния базы + проекция в скаляр.

    Сиблинг базы, а не её подмодуль: `freeze()` базы (или add_lora поверх
    неё) не должен молча заморозить голову.
    """

    def __init__(self, config, rcfg: RerankerConfig, ranks: dict = None):
        super().__init__()
        self.rcfg = rcfg
        self.layer_idx = resolve_layer_indices(rcfg.layer_idx, config.n_layer)
        D = config.n_embd
        n = len(self.layer_idx)

        # Какой слой базы читает блок i, и какие слои вообще нужны.
        # Голове нужны НЕ все L состояний, а только эти — что решающе важно
        # для кэша: хранить [n_unique, H, S, S] вместо [L, H, S, S] дешевле
        # ровно в L/n_unique раз (для умолчания — в 12 раз).
        last = self.layer_idx[-1]
        self.sources = [last if rcfg.shared_state else self.layer_idx[i]
                        for i in range(n)]
        self.unique_sources = sorted(set(self.sources))
        self.source_slot = [self.unique_sources.index(s) for s in self.sources]

        # Токены-зонды. Масштаб не важен: ln0 нормирует каждый токен.
        self.probe = mx.random.normal((rcfg.n_probe, D)) * 0.02
        self.ln0 = nn.LayerNorm(D)
        self.blocks = [RWKVBlock(config, i, ranks) for i in range(n)]
        self.ln_out = nn.LayerNorm(D)

        hidden = rcfg.head_hidden or D
        self.score_fc1 = nn.Linear(D, hidden)
        self.score_fc2 = nn.Linear(hidden, 1, bias=False)
        # zero-init: стартовые скоры ровно нули (см. докстроку модуля)
        self.score_fc2.weight = mx.zeros_like(self.score_fc2.weight)

    def select(self, state: RWKVState) -> mx.array:
        """RWKVState → [B, n_unique, H, S, S]: только читаемые головой слои.

        Это единица кэширования при обучении на замороженной базе: пара
        (документ, запрос) сворачивается в такой тензор один раз, дальше
        обучение головы состояние не пересчитывает.
        """
        return mx.stack([state.wkv[s] for s in self.unique_sources], axis=1)

    def __call__(self, state) -> mx.array:
        """state: RWKVState базы либо уже отобранный тензор
        [B, n_unique, H, S, S] (см. `select`). Возвращает скоры [B]."""
        sel = self.select(state) if isinstance(state, RWKVState) else state
        B = sel.shape[0]
        D = self.probe.shape[1]

        x = mx.broadcast_to(self.probe[None], (B, self.probe.shape[0], D))
        x = self.ln0(x)

        v_first = None
        for i, block in enumerate(self.blocks):
            h_in = sel[:, self.source_slot[i]].astype(mx.float32)   # [B, H, S, S]
            x, v_first = block(x, v_first, h_in=h_in)

        x = self.ln_out(x[:, -1])                          # последний зонд, [B, D]
        return self.score_fc2(mx.tanh(self.score_fc1(x))).squeeze(-1)   # [B]

    # ── Инициализация из базы ────────────────────────────────────────────
    def init_from_base(self, base: RWKV7X070):
        """Копирует в блоки головы веса выбранных слоёв базы, а ln0/ln_out —
        одноимённые слои базы.

        Нумерация блоков головы своя (0..n-1), поэтому у блока 0 нет
        value-residual (`v_lora`), даже если исходный слой базы её имел, — как
        в оригинале. Обратный случай (блок головы i>0 инициализируется слоем
        базы 0, где v_lora нет) решается нейтрализацией: v_lora_B зануляется,
        а её bias уводится в -10, откуда sigmoid ≈ 0 и value-residual
        выключается вместо того, чтобы остаться случайной.
        """
        from mlx.utils import tree_flatten, tree_unflatten

        for i, src_idx in enumerate(self.layer_idx):
            src = dict(tree_flatten(base.blocks[src_idx].parameters()))
            dst_keys = set(k for k, _ in tree_flatten(self.blocks[i].parameters()))
            upd = {k: v for k, v in src.items() if k in dst_keys}
            self.blocks[i].update(tree_unflatten(list(upd.items())))

            missing_v = [k for k in dst_keys
                         if k.startswith("tmix.v_lora") and k not in src]
            if missing_v:
                tm = self.blocks[i].tmix
                tm.v_lora_B.weight = mx.zeros_like(tm.v_lora_B.weight)
                tm.v_lora_B.bias = mx.full(tm.v_lora_B.bias.shape, -10.0,
                                           dtype=tm.v_lora_B.bias.dtype)

        self.ln0.update(base.ln0.parameters())
        self.ln_out.update(base.ln_out.parameters())
        mx.eval(self.parameters())
        return self

    def set_dtype(self, dtype):
        """Привести ВСЕ параметры головы к одному типу.

        Нужно потому, что `init_from_base` копирует веса базы как есть, а
        официальные чекпоинты лежат в bf16. Без этого голова обучалась бы в
        bf16, у которого 8 бит мантиссы: при lr порядка 1e-4 и весах порядка
        0.05 шаг оптимизатора получается меньше кванта представления и
        просто теряется при округлении. База при этом остаётся в своём типе —
        она заморожена, и её точность здесь ни при чём.
        """
        from mlx.utils import tree_map
        if isinstance(dtype, str):
            dtype = {"bfloat16": mx.bfloat16, "bf16": mx.bfloat16,
                     "float32": mx.float32, "fp32": mx.float32}[dtype]
        self.update(tree_map(
            lambda x: x.astype(dtype) if isinstance(x, mx.array) else x,
            self.parameters()))
        mx.eval(self.parameters())
        return self


class Reranker(nn.Module):
    """Замороженная база + обучаемая голова.

    base:  RWKV7X070 (официальная архитектура). RWKV7 из pretrain-ветки не
           подходит: там межблочный перенос token-shift делает состояние
           некаузальным — см. докстроку `RWKV7.body`.
    """

    def __init__(self, base: RWKV7X070, rcfg: RerankerConfig = None,
                 freeze_base: bool = True, init_from_base: bool = True,
                 head_dtype=mx.float32):
        super().__init__()
        if not isinstance(base, RWKV7X070):
            raise TypeError(
                "Реранкер работает только с RWKV7X070 (официальная архитектура). "
                f"Получено: {type(base).__name__}."
            )
        self.base = base
        self.rcfg = rcfg or RerankerConfig()
        self.head = RerankerHead(base.config, self.rcfg, getattr(base, "ranks", None))
        if init_from_base:
            self.head.init_from_base(base)
        # официальные веса лежат в bf16, и голова унаследовала бы его от базы:
        # 8 бит мантиссы против 24 у fp32 — часть шагов оптимизатора просто
        # терялась бы при округлении. Голова маленькая (8-23 М), fp32 ей
        # ничего не стоит. head_dtype=mx.bfloat16 вернёт прежнее поведение.
        if head_dtype is not None:
            self.head.set_dtype(head_dtype)
        if freeze_base:
            self.base.freeze()

    # ── Состояние базы ───────────────────────────────────────────────────
    def encode(self, idx, mask=None, end_idx=None, state=None,
               detach: bool = True) -> RWKVState:
        """Свернуть последовательность в состояние базы.

        detach=True (умолчание) обрывает граф: база заморожена, и без
        stop_gradient MLX всё равно протаскивал бы backward через весь
        длинный проход ради ничего.
        """
        st = self.base.states(idx, mask=mask, end_idx=end_idx, state=state)
        return st.stop_gradient() if detach else st

    def select(self, state: RWKVState) -> mx.array:
        """Свернуть состояние базы до того, что реально читает голова."""
        return self.head.select(state)

    def score_states(self, state) -> mx.array:
        return self.head(state)

    def __call__(self, idx, mask=None, end_idx=None, state=None) -> mx.array:
        return self.head(self.encode(idx, mask=mask, end_idx=end_idx, state=state))

    # ── Сохранение / загрузка ────────────────────────────────────────────
    #
    # Конфигурация пишется в metadata чекпоинта не для красоты. Голова из
    # одного блока над слоем 5 и голова из одного блока над слоем 11 имеют
    # ОДИНАКОВЫЕ формы всех тензоров: перепутав их, `update()` отработает
    # молча, а модель будет читать не тот слой и выдавать правдоподобный
    # мусор. Здесь это ловится на загрузке.
    def _metadata(self, extra: dict = None) -> dict:
        md = {
            "format": "rwkv-metal-reranker-head-v1",
            "layer_idx": ",".join(str(i) for i in self.head.layer_idx),
            "shared_state": str(int(self.rcfg.shared_state)),
            "n_probe": str(self.rcfg.n_probe),
            "head_hidden": str(self.rcfg.head_hidden or ""),
            "base_n_layer": str(self.base.config.n_layer),
            "base_n_embd": str(self.base.config.n_embd),
            "base_n_head": str(self.base.config.n_head),
        }
        if extra:
            md.update({k: str(v) for k, v in extra.items()})
        return md

    def save_head(self, path: str, extra: dict = None):
        """extra: произвольные строки в metadata — сюда стоит класть контракт
        подачи текста (шаблон, обрезки, терминатор, инструкция). Модель о нём
        не знает, а расходится он так же молча, как и слои."""
        from mlx.utils import tree_flatten
        mx.save_safetensors(path, dict(tree_flatten(self.head.parameters())),
                            metadata=self._metadata(extra))

    def load_head(self, path: str, strict: bool = True):
        from mlx.utils import tree_unflatten
        weights, md = mx.load(path, return_metadata=True)
        if strict and md.get("format", "").startswith("rwkv-metal-reranker-head"):
            want = self._metadata()
            for key in ("layer_idx", "shared_state", "n_probe",
                        "base_n_layer", "base_n_embd", "base_n_head"):
                if key in md and md[key] != want[key]:
                    raise ValueError(
                        f"чекпоинт не соответствует модели: {key}="
                        f"{md[key]!r} в файле против {want[key]!r} здесь. "
                        f"Собери Reranker с той же конфигурацией — проще всего "
                        f"через Reranker.from_head(base, {path!r})."
                    )
        self.head.update(tree_unflatten(list(weights.items())))
        mx.eval(self.head.parameters())
        return self

    @staticmethod
    def read_head_metadata(path: str) -> dict:
        """Метаданные чекпоинта без загрузки весов в модель."""
        _, md = mx.load(path, return_metadata=True)
        return md

    @classmethod
    def from_head(cls, base: RWKV7X070, path: str, **kwargs) -> "Reranker":
        """Собрать реранкер по конфигурации, записанной в самом чекпоинте.

        Рекомендуемый способ загрузки: не нужно помнить, какие слои читала
        голова и сколько у неё зондов.
        """
        md = cls.read_head_metadata(path)
        if not md.get("format", "").startswith("rwkv-metal-reranker-head"):
            raise ValueError(
                f"{path}: нет метаданных реранкера. Это чекпоинт, сохранённый "
                "старой версией save_head — собери Reranker вручную с нужной "
                "конфигурацией и вызови load_head(..., strict=False)."
            )
        hidden = md.get("head_hidden") or ""
        rcfg = RerankerConfig(
            layer_idx=tuple(int(i) for i in md["layer_idx"].split(",")),
            shared_state=bool(int(md.get("shared_state", "0"))),
            n_probe=int(md.get("n_probe", "1")),
            head_hidden=int(hidden) if hidden else None,
        )
        model = cls(base, rcfg, **kwargs)
        model.load_head(path)
        return model
