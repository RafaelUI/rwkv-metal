import sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-metal")
import numpy as np, mlx.core as mx
from rwkv_metal.lora.rwkvq_linear import RwkvqLinear
from tests.dev_pack_native_mlx import codes_scale_bias, pack_codes_mlx6

B, T = 1, 128

for key in ["blocks.0.att.key.weight", "blocks.5.ffn.value.weight", "emb.weight"]:
    lin = RwkvqLinear.from_sidecar("/tmp/reduction_v2.rwkvq_mlx", key)
    codes, scale, bias = codes_scale_bias(lin)
    OUT, NB, _ = codes.shape
    wq = mx.array(pack_codes_mlx6(codes).reshape(OUT, NB * 6))
    scale_mx, bias_mx = mx.array(scale), mx.array(bias)

    xin = mx.random.normal((B * T, lin.in_features)).astype(mx.float32)

    def native():
        return mx.quantized_matmul(xin, wq, scale_mx, bias_mx, transpose=True,
                                     group_size=32, bits=6)

    def fused_current():
        w = lin._dequant_w()
        return xin.astype(mx.bfloat16) @ w.T

    for name, fn in (("native_qmm", native), ("fused+matmul (текущий)", fused_current)):
        y = fn(); mx.eval(y)
        N = 30
        t0 = time.time()
        for _ in range(N):
            y = fn(); mx.eval(y)
        dt = (time.time() - t0) / N * 1000
        print(f"{key:28s} {name:24s} {dt:7.3f} ms  out_shape={tuple(y.shape)}")
