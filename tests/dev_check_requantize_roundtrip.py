import sys
sys.path.insert(0, "/Users/s/Develop/rwkv-metal")
import numpy as np, mlx.core as mx
from rwkv_metal.lora.rwkvq_linear import RwkvqLinear

for key in ["blocks.0.att.key.weight", "blocks.5.ffn.value.weight", "blocks.10.att.output.weight"]:
    lin = RwkvqLinear.from_sidecar("/tmp/reduction_v2.rwkvq_mlx", key)
    w_sb6 = lin._dequant_w()  # bf16, наш точный fused-деквант
    mx.eval(w_sb6)

    w_f32 = w_sb6.astype(mx.float32)
    wq, scales, biases = mx.quantize(w_f32, group_size=32, bits=6)
    w_req = mx.dequantize(wq, scales, biases, group_size=32, bits=6).astype(mx.bfloat16)
    mx.eval(w_req)

    diff = np.abs(np.array(w_sb6.astype(mx.float32)) - np.array(w_req.astype(mx.float32)))
    bits_sb6 = np.array(w_sb6.view(mx.uint16))
    bits_req = np.array(w_req.view(mx.uint16))
    n_mis = int((bits_sb6 != bits_req).sum())
    print(f"{key:30s} max_abs_diff={diff.max():.6g} mean_abs_diff={diff.mean():.6g} "
          f"mismatch_bf16_bits={n_mis}/{w_sb6.size} ({100*n_mis/w_sb6.size:.2f}%)")
