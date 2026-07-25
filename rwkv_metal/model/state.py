"""
rwkv_metal.model.state
======================
Рекуррентное состояние RWKV-7: всё, что нужно, чтобы продолжить
последовательность с того места, где она оборвалась.

RWKV — RNN, поэтому «продолжить» стоит O(длины продолжения), а не
O(длины всего префикса). Это и есть та экономия, ради которой реранкер
кэширует состояние документа и досчитывает только запрос.

Состояние состоит из трёх частей на слой:

    wkv         [L, B, H, S, S]  матрица WKV-рекуррентности
    tmix_shift  [L, B, 1, D]     вход tmix (ln1(x)) на последней позиции
    cmix_shift  [L, B, 1, D]     вход cmix (ln2(x)) на последней позиции

Две последние — состояние token-shift. Их часто забывают: без них
продолжение отличается от сплошного прохода на первом токене каждого
слоя (там, где token-shift читает «предыдущий» токен и получает ноль).

`v_first` в состояние НЕ входит намеренно: в x070 это не бегущее
состояние, а величина, вычисляемая слоем 0 для каждой позиции заново и
потребляемая слоями выше на той же позиции. Продолжение считает свой
v_first из своих же токенов — переносить нечего.
"""
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx


@dataclass
class RWKVState:
    """Полное рекуррентное состояние на границе последовательности."""

    wkv: mx.array          # [L, B, H, S, S]
    tmix_shift: mx.array   # [L, B, 1, D]
    cmix_shift: mx.array   # [L, B, 1, D]

    # ── Конструкторы ─────────────────────────────────────────────────────
    @staticmethod
    def zeros(config, batch: int = 1, dtype=mx.float32) -> "RWKVState":
        L, H, S, D = config.n_layer, config.n_head, config.head_size, config.n_embd
        return RWKVState(
            wkv=mx.zeros((L, batch, H, S, S), dtype=mx.float32),
            tmix_shift=mx.zeros((L, batch, 1, D), dtype=dtype),
            cmix_shift=mx.zeros((L, batch, 1, D), dtype=dtype),
        )

    @staticmethod
    def stack(wkv_list, tmix_list, cmix_list) -> "RWKVState":
        return RWKVState(
            wkv=mx.stack(wkv_list, axis=0),
            tmix_shift=mx.stack(tmix_list, axis=0),
            cmix_shift=mx.stack(cmix_list, axis=0),
        )

    # ── Свойства ─────────────────────────────────────────────────────────
    @property
    def n_layer(self) -> int:
        return self.wkv.shape[0]

    @property
    def batch(self) -> int:
        return self.wkv.shape[1]

    def nbytes(self) -> int:
        return sum(x.nbytes for x in (self.wkv, self.tmix_shift, self.cmix_shift))

    # ── Манипуляции по батчу ─────────────────────────────────────────────
    def __getitem__(self, item) -> "RWKVState":
        """Срез/выборка по оси батча (ось 1 у всех трёх тензоров)."""
        if isinstance(item, int):
            item = slice(item, item + 1)
        if isinstance(item, mx.array):
            return RWKVState(
                wkv=mx.take(self.wkv, item, axis=1),
                tmix_shift=mx.take(self.tmix_shift, item, axis=1),
                cmix_shift=mx.take(self.cmix_shift, item, axis=1),
            )
        return RWKVState(
            wkv=self.wkv[:, item],
            tmix_shift=self.tmix_shift[:, item],
            cmix_shift=self.cmix_shift[:, item],
        )

    def repeat(self, n: int) -> "RWKVState":
        """Размножить состояние с batch=1 на n строк (один документ —
        много запросов). Без копирования данных до первой записи."""
        assert self.batch == 1, f"repeat() ждёт batch=1, получил {self.batch}"
        return RWKVState(
            wkv=mx.repeat(self.wkv, n, axis=1),
            tmix_shift=mx.repeat(self.tmix_shift, n, axis=1),
            cmix_shift=mx.repeat(self.cmix_shift, n, axis=1),
        )

    @staticmethod
    def concat(states) -> "RWKVState":
        """Склеить состояния по оси батча."""
        return RWKVState(
            wkv=mx.concatenate([s.wkv for s in states], axis=1),
            tmix_shift=mx.concatenate([s.tmix_shift for s in states], axis=1),
            cmix_shift=mx.concatenate([s.cmix_shift for s in states], axis=1),
        )

    # ── Прочее ───────────────────────────────────────────────────────────
    def astype(self, dtype) -> "RWKVState":
        """wkv всегда остаётся fp32: ядро считает рекуррентность в fp32, и
        именно её точность определяет, совпадёт ли продолжение со сплошным
        проходом. Приводится только token-shift."""
        return RWKVState(
            wkv=self.wkv,
            tmix_shift=self.tmix_shift.astype(dtype),
            cmix_shift=self.cmix_shift.astype(dtype),
        )

    def stop_gradient(self) -> "RWKVState":
        return RWKVState(
            wkv=mx.stop_gradient(self.wkv),
            tmix_shift=mx.stop_gradient(self.tmix_shift),
            cmix_shift=mx.stop_gradient(self.cmix_shift),
        )

    def eval(self) -> "RWKVState":
        mx.eval(self.wkv, self.tmix_shift, self.cmix_shift)
        return self


def build_mask(lengths, total: int, dtype=mx.float32) -> mx.array:
    """Маска реальных токенов [B, T] для right-padded батча.

    lengths: список/массив длин (число реальных токенов в строке).
    Пад-позиции получают 0 и становятся no-op для WKV-рекуррентности
    (см. `RWKV_Tmix_x070.__call__`), поэтому финальное состояние строки
    равно состоянию после её последнего настоящего токена.
    """
    lens = mx.array(lengths).reshape(-1, 1)
    pos = mx.arange(total).reshape(1, -1)
    return (pos < lens).astype(dtype)


def gather_last(x: mx.array, end_idx: Optional[mx.array] = None) -> mx.array:
    """[B, T, D] → [B, 1, D] на позиции end_idx (или последней, если None)."""
    if end_idx is None:
        return x[:, -1:]
    return mx.take_along_axis(x, end_idx.reshape(-1, 1, 1), axis=1)
