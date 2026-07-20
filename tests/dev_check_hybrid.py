import sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-metal")
import numpy as np, mlx.core as mx
from rwkv_metal.lora.rwkvq_linear import RwkvqLinear
from rwkv_metal.lora.rwkvq_hybrid import RwkvqHybridLinear

for key in ["blocks.0.att.key.weight", "blocks.5.ffn.value.weight", "emb.weight"]:
    lin_ref = RwkvqLinear.from_sidecar("/tmp/reduction_v2.rwkvq_mlx", key)
    w_ref = lin_ref._dequant_w(); mx.eval(w_ref)

    hy = RwkvqHybridLinear.from_sidecar("/tmp/reduction_v2.rwkvq_mlx", key)
    scale, bias = hy._expand_scale_bias()
    w_hy = mx.dequantize(hy.wq, scale, bias, group_size=32, bits=6).astype(mx.bfloat16)
    mx.eval(w_hy)

    ref_bits = np.array(w_ref.view(mx.uint16))
    got_bits = np.array(w_hy.view(mx.uint16))
    n_mis = int((ref_bits != got_bits).sum())
    print(f"{key:30s} correctness mismatch={n_mis}/{w_ref.size}")

    x = mx.random.normal((128, hy.in_features)).astype(mx.float32)
    def call():
        return hy(x)
    y = call(); mx.eval(y)
    N = 30
    t0 = time.time()
    for _ in range(N):
        y = call(); mx.eval(y)
    dt = (time.time() - t0) / N * 1000
    print(f"  hybrid __call__ warm: {dt:.3f} ms  out_shape={tuple(y.shape)}")
