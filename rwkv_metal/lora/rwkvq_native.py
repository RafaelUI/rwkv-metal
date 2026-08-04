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

Битовая раскладка mx.quantize(group_size=32) реверс-инжинирена
эмпирически (one-hot тесты, tests/dev_reverse_mlx_pack.py): LSB-first
битовый поток на группу из 32 кодов, поле позиции p начинается на
глобальном бите p*bits, переходит через границы 32-битных слов без
выравнивания.

ОГРАНИЧЕНИЕ «ТОЛЬКО bits=6» СНЯТО (05.08). Здесь раньше стояло
предупреждение не использовать это для COMPRESSION (proj@5, cmix@4),
потому что раскладка была проверена лишь на шести битах. Проверена на
всех: rwkv-quant/tests/probe_mlx_native_packing.py гоняет 4, 5, 6 и 8 --
правило одно и то же. Реверс-инжиниринг для этого не понадобился:
mx.dequantize есть обратная функция к упаковке, и упаковщик сверяется
прямо против неё. ВАЖНО сверять КОДЫ, а не значения: сравнение
q*scale + bias в fp16 даёт 75-81% совпадений на всех битностях сразу,
включая заведомо рабочую шестую, -- это разное округление в ядре, а не
разная раскладка, и по нему легко сделать вывод наоборот.

Сама перекладка теперь живёт в rwkv_quant.formats.codec
(sb6_to_mlx_affine, numpy, без torch) и берётся прямо из .rwkvq, минуя
сайдкар. Здешний _pack_codes_mlx6 -- прежняя реализация для bits=6.
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


def _pack_codes_mlx(codes32: np.ndarray, bits: int) -> np.ndarray:
    """codes32: [..., 32] -> [..., bits] uint32, раскладка
    mx.quantize(group_size=32).

    Реализация делегирована в rwkv_quant.formats.codec: там она одна на
    все три репозитория и покрыта гейтом на 4/5/6/8 бит. Прежняя копия
    здесь была захардкожена под шесть и породила ассерт `xbits == 2`,
    который жил дольше, чем ограничение, его вызвавшее."""
    from rwkv_quant.formats import codec
    return codec.pack_mlx_affine(codes32, bits)


class RwkvqNativeLinear(nn.Module):
    """Frozen linear на РОДНОМ MLX quantized_matmul поверх перепакованных
    sb6-данных. y = quantized_matmul(x, wq, scale, bias, transpose=True).
    Однократная перепаковка при конструировании (не на каждый forward).

    Работает на ЛЮБОЙ битности sb6 (4/5/6), то есть и на COMPRESSION.
    Прежде здесь стоял ассерт `xbits == 2`, потому что раскладка MLX
    была реверс-инжинирена только для шести бит; проверка на 4/5/6/8
    показала, что правило одно и то же."""

    def __init__(self, lin: RwkvqLinear):
        super().__init__()
        self.out_features, self.in_features = lin.out_features, lin.in_features
        self.bits = 4 + lin.xbits
        codes, scale, bias = _codes_scale_bias(lin)
        OUT, NB, _ = codes.shape
        wq_np = _pack_codes_mlx(codes, self.bits).reshape(OUT, NB * self.bits)
        self.wq = mx.array(wq_np)
        self.scale = mx.array(scale)
        self.bias = mx.array(bias)
        self.freeze()

    @classmethod
    def from_sidecar(cls, sidecar_path: str, key: str):
        return cls(RwkvqLinear.from_sidecar(sidecar_path, key))

    def __call__(self, x):
        return mx.quantized_matmul(x.astype(mx.float32), self.wq, self.scale, self.bias,
                                    transpose=True, group_size=GROUP_SIZE,
                                    bits=self.bits).astype(x.dtype)
