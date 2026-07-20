import sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-metal")
import numpy as np, mlx.core as mx
from rwkv_metal.lora.rwkvq_linear import RwkvqLinear

def codes_scale_bias(lin: RwkvqLinear):
    """Как _dequant_w_slow, но возвращает (codes[OUT,IN] int32 0..63,
    scale[OUT,NB] f32, bias[OUT,NB] f32) БЕЗ финального combine."""
    OUT, IN, NB, NSB = lin.out_features, lin.in_features, lin.NB, lin.NSB
    blk = lin.qblk.reshape(OUT, NB, 16 + 4 * lin.xbits)
    cb = blk[:, :, :16]
    q = mx.concatenate([cb & 0xF, cb >> 4], axis=2).astype(mx.int32)
    if lin.xbits >= 1:
        qh = blk[:, :, 16:20].reshape(OUT, IN // 8)
        bits = (qh[..., None] >> mx.arange(8, dtype=mx.uint8)) & 1
        q = q + bits.reshape(OUT, NB, 32).astype(mx.int32) * 16
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


def pack_codes_mlx6(codes32: np.ndarray) -> np.ndarray:
    """codes32: [..., 32] int, значения 0..63 -> [..., 6] uint32 (нативная
    упаковка mx.quantize(bits=6, group_size=32): LSB-first битовый поток,
    поле позиции p начинается на глобальном бите p*6)."""
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
        part0 = (code_p & ((1 << bits_in_w0) - 1)) << off0
        words[..., w0] |= part0.astype(np.uint32)
        if bits_in_w1 > 0:
            part1 = (code_p >> bits_in_w0) & ((1 << bits_in_w1) - 1)
            words[..., w0 + 1] |= part1.astype(np.uint32)
    return words


for key in ["blocks.0.att.key.weight", "blocks.5.ffn.value.weight", "blocks.10.att.output.weight"]:
    lin = RwkvqLinear.from_sidecar("/tmp/reduction_v2.rwkvq_mlx", key)
    codes, scale, bias = codes_scale_bias(lin)
    OUT, NB, _ = codes.shape
    wq_np = pack_codes_mlx6(codes).reshape(OUT, NB * 6)

    wq = mx.array(wq_np)
    scale_mx = mx.array(scale)
    bias_mx = mx.array(bias)

    w_native = mx.dequantize(wq, scale_mx, bias_mx, group_size=32, bits=6)
    mx.eval(w_native)

    w_ref = lin._dequant_w()  # наш проверенный fused-кернель, bf16
    mx.eval(w_ref)

    ref_bits = np.array(w_ref.view(mx.uint16))
    got_bits = np.array(w_native.astype(mx.bfloat16).view(mx.uint16))
    n_mis = int((ref_bits != got_bits).sum())
    print(f"{key:30s} mismatch={n_mis}/{w_ref.size}")
    if n_mis:
        diff = np.abs(np.array(w_ref.astype(mx.float32)) - np.array(w_native))
        print(f"   max_abs_diff={diff.max():.6g}")
