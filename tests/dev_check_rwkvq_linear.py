import sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-metal")
import numpy as np
import mlx.core as mx
from rwkv_metal.lora.rwkvq_linear import RwkvqLinear

ref = mx.load("/tmp/ref_bits_check.safetensors")
keys = ["blocks.0.att.key.weight", "blocks.5.ffn.value.weight", "emb.weight"]
ref_keys = ["blocks_0_att_key_weight", "blocks_5_ffn_value_weight", "emb_weight"]

for k, rk in zip(keys, ref_keys):
    lin = RwkvqLinear.from_sidecar("/tmp/reduction_v2.rwkvq_mlx", k)
    w = lin._dequant_w()
    mx.eval(w)
    r = ref[rk]
    got_bits = np.array(w.view(mx.uint16))
    ref_bits = np.array(r.view(mx.uint16))
    n_mis = int((got_bits != ref_bits).sum())
    print(f"{k:30s} shape={tuple(w.shape)} mismatch={n_mis}/{w.size}")

# бенч сквозного forward одного слоя (proj-набор) + сравнение с "голым" matmul
lin = RwkvqLinear.from_sidecar("/tmp/reduction_v2.rwkvq_mlx", "blocks.0.att.key.weight")
x = mx.random.normal((1, 8, 2048)).astype(mx.bfloat16)
for _ in range(3):
    y = lin(x); mx.eval(y)
t0 = time.time()
for _ in range(20):
    y = lin(x); mx.eval(y)
print(f"RwkvqLinear.__call__ (dequant+matmul) warm: {(time.time()-t0)/20*1000:.3f} ms")
