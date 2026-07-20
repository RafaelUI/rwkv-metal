"""
Гибрид: коды -- в родной битовой раскладке MLX (wq, как в rwkvq_native.py,
даёт скорость stock quantized_matmul), а scale/bias -- в компактном sb6
суперблочном виде (qsqm/ddm, как в rwkv_linear.py) и разворачиваются в
плоский [OUT,NB] fp32 КАЖДЫЙ forward -- дёшево, т.к. это маленькие
массивы (NB = IN/32 элементов на строку, а не IN*OUT).

Идея: коды весят ОДИНАКОВО что в sb6 (codes-ниббл 4бита + 2 битплоскости
= 0.75Б/значение), что в родной MLX-упаковке (6бит=0.75Б/значение) --
переключение упаковки кодов НИЧЕГО не стоит по памяти. Раздутие было
только от scale/bias: родной путь (rwkvq_native.py) хранит их fp32 ПОЛНЫМ
per-group (32 значения/группа) = 0.25Б/значение, а sb6-суперблок -- 6-бит
qs/qm против fp16 d/dm на 256 значений = ~0.012Б/значение. Держим
компактную версию, разворачиваем на лету -- разворачивание маленьких
[OUT,NB]-массивов на forward пренебрежимо дёшево относительно самого
matmul.

Ожидание: память ~= уровень fused-кернеля (rwkvq_linear.py), скорость ~=
уровень native (rwkvq_native.py, тот же mx.quantized_matmul).
"""
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .rwkvq_linear import RwkvqLinear
from .rwkvq_native import _pack_codes_mlx6, GROUP_SIZE, BITS


def _codes_only(lin: RwkvqLinear) -> np.ndarray:
    """Только коды (0..63 int), БЕЗ scale/bias -- те остаются компактными
    в qsqm/ddm, разворачиваются в __call__."""
    OUT, IN, NB = lin.out_features, lin.in_features, lin.NB
    blk = lin.qblk.reshape(OUT, NB, 16 + 4 * lin.xbits)
    cb = blk[:, :, :16]
    q = mx.concatenate([cb & 0xF, cb >> 4], axis=2).astype(mx.int32)
    if lin.xbits >= 1:
        qh = blk[:, :, 16:20].reshape(OUT, IN // 8)
        bits_ = (qh[..., None] >> mx.arange(8, dtype=mx.uint8)) & 1
        q = q + bits_.reshape(OUT, NB, 32).astype(mx.int32) * 16
    if lin.xbits >= 2:
        qh2 = blk[:, :, 20:24].reshape(OUT, IN // 8)
        bits2 = (qh2[..., None] >> mx.arange(8, dtype=mx.uint8)) & 1
        q = q + bits2.reshape(OUT, NB, 32).astype(mx.int32) * 32
    return np.array(q.reshape(OUT, NB, 32))


class RwkvqHybridLinear(nn.Module):
    """Frozen linear: коды в родной MLX-упаковке (wq) + компактные sb6
    scale/bias (qsqm/ddm), развёрнутые на лету перед quantized_matmul."""

    def __init__(self, lin: RwkvqLinear):
        super().__init__()
        assert lin.xbits == 2, "bits=6 (xbits=2) required"
        self.out_features, self.in_features = lin.out_features, lin.in_features
        self.NB, self.NSB, self._gw_sb = lin.NB, lin.NSB, lin._gw_sb

        codes = _codes_only(lin)
        OUT, NB, _ = codes.shape
        self.wq = mx.array(_pack_codes_mlx6(codes).reshape(OUT, NB * BITS))
        # компактные scale/bias -- те же буферы, что у fused-кернеля
        self.qsqm = lin.qsqm
        self.ddm = lin.ddm
        self.freeze()

    @classmethod
    def from_sidecar(cls, sidecar_path: str, key: str):
        return cls(RwkvqLinear.from_sidecar(sidecar_path, key))

    def _expand_scale_bias(self):
        OUT, NB, NSB, sb = self.out_features, self.NB, self.NSB, self._gw_sb
        sm = self.qsqm.reshape(OUT, NB, 2)
        qs = sm[:, :, 0].astype(mx.float32)
        qm = mx.view(sm[:, :, 1], mx.int8).astype(mx.float32)
        dd = self.ddm.reshape(OUT, NSB, 2)
        d = dd[:, :, 0].astype(mx.float32)
        dm = dd[:, :, 1].astype(mx.float32)
        d_c = mx.repeat(d, sb, axis=1)
        dm_c = mx.repeat(dm, sb, axis=1)
        scale = (qs * d_c).astype(mx.float16).astype(mx.float32)
        scale = mx.maximum(scale, 1e-8)
        bias = (qm * dm_c).astype(mx.float16).astype(mx.float32)
        return scale, bias

    def __call__(self, x):
        scale, bias = self._expand_scale_bias()
        return mx.quantized_matmul(x.astype(mx.float32), self.wq, scale, bias,
                                    transpose=True, group_size=GROUP_SIZE, bits=BITS
                                    ).astype(x.dtype)
