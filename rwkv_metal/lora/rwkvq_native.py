"""
"Родной" MLX-квант путь для .rwkvq (sb6, bits=6 -- REDUCTION-пресет):
вместо своего fused-кернеля деквантования (rwkvq_kernel.py) -- ОДНОКРАТНАЯ
(при загрузке модели) перепаковка НАШИХ точных кодов+scale+bias в
битовый контейнер mx.quantize/mx.quantized_matmul, дальше forward идёт
через штатный, полностью оптимизированный тайловый GEMM MLX -- тот же
кернель, что и у стокового QLoRA (nn.QuantizedLinear).

ВАЖНО: это НЕ mx.quantize(dense_weight) -- тот пересчитывает scale/bias
по min/max блока и искажает ~89% значений (round-trip re-quantization,
проверено tests/dev_check_requantize_roundtrip.py). Здесь коды/scale/bias
берутся ИЗ sb6 НАПРЯМУЮ (bit-в-bit те же значения, что и калибровка
REDUCTION/COMPRESSION посчитала), просто перекладываются в другой
битовый контейнер. Сверено бит-в-бит с rwkv_quant-референсом через
mx.dequantize(wq, scale, bias) -- 0 расхождений
(tests/dev_pack_native_mlx.py).

Битовая раскладка mx.quantize(bits=6, group_size=32) реверс-инжинирена
эмпирически (one-hot тесты, tests/dev_reverse_mlx_pack.py): LSB-first
битовый поток на группу из 32 кодов, поле позиции p начинается на
глобальном бите p*6, переходит через границы 32-битных слов без
выравнивания. НЕ проверено для bits != 6 (у quantized.h разные ветки
паковки для степеней двойки -- см. get_bytes_per_pack) -- не используйте
для COMPRESSION-пресета (там proj/cmix на 4/5 бит) без отдельной
проверки.
"""
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .rwkvq_linear import RwkvqLinear

GROUP_SIZE = 32
BITS = 6


def _codes_scale_bias(lin: RwkvqLinear):
    """Как RwkvqLinear._dequant_w_slow, но без финального combine --
    отдельно коды (0..63 int), scale[OUT,NB] f32, bias[OUT,NB] f32."""
    OUT, IN, NB, NSB = lin.out_features, lin.in_features, lin.NB, lin.NSB
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

    sm = lin.qsqm.reshape(OUT, NB, 2)
    qs = sm[:, :, 0].astype(mx.float32)
    qm = mx.view(sm[:, :, 1], mx.int8).astype(mx.float32)
    dd = lin.ddm.reshape(OUT, NSB, 2)
    d = dd[:, :, 0].astype(mx.float32)
    dm = dd[:, :, 1].astype(mx.float32)
    sb = NB // NSB
    d_c = mx.repeat(d, sb, axis=1)
    dm_c = mx.repeat(dm, sb, axis=1)
    scale = (qs * d_c).astype(mx.float16).astype(mx.float32)
    scale = mx.maximum(scale, 1e-8)
    bias = (qm * dm_c).astype(mx.float16).astype(mx.float32)

    return np.array(q.reshape(OUT, NB, 32)), np.array(scale), np.array(bias)


def _pack_codes_mlx6(codes32: np.ndarray) -> np.ndarray:
    """codes32: [..., 32] int (0..63) -> [..., 6] uint32, нативная
    упаковка mx.quantize(bits=6, group_size=32)."""
    lead = codes32.shape[:-1]
    words = np.zeros((*lead, 6), dtype=np.uint32)
    codes = codes32.astype(np.uint32)
    for p in range(32):
        bit_start = p * 6
        w0 = bit_start // 32
        off0 = bit_start % 32
        bits_in_w0 = min(6, 32 - off0)
        bits_in_w1 = 6 - bits_in_w0
        code_p = codes[..., p]
        words[..., w0] |= ((code_p & ((1 << bits_in_w0) - 1)) << off0).astype(np.uint32)
        if bits_in_w1 > 0:
            words[..., w0 + 1] |= ((code_p >> bits_in_w0) & ((1 << bits_in_w1) - 1)).astype(np.uint32)
    return words


class RwkvqNativeLinear(nn.Module):
    """Frozen linear на РОДНОМ MLX quantized_matmul поверх перепакованных
    sb6-данных. y = quantized_matmul(x, wq, scale, bias, transpose=True).
    Однократная перепаковка при конструировании (не на каждый forward)."""

    def __init__(self, lin: RwkvqLinear):
        super().__init__()
        assert lin.xbits == 2, "bits=6 (xbits=2) required -- см. докстринг модуля"
        self.out_features, self.in_features = lin.out_features, lin.in_features
        codes, scale, bias = _codes_scale_bias(lin)
        OUT, NB, _ = codes.shape
        wq_np = _pack_codes_mlx6(codes).reshape(OUT, NB * BITS)
        self.wq = mx.array(wq_np)
        self.scale = mx.array(scale)
        self.bias = mx.array(bias)
        self.freeze()

    @classmethod
    def from_sidecar(cls, sidecar_path: str, key: str):
        return cls(RwkvqLinear.from_sidecar(sidecar_path, key))

    def __call__(self, x):
        return mx.quantized_matmul(x.astype(mx.float32), self.wq, self.scale, self.bias,
                                    transpose=True, group_size=GROUP_SIZE, bits=BITS
                                    ).astype(x.dtype)
