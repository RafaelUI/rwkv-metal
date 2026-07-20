import sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-metal")
import numpy as np
import mlx.core as mx
from rwkv_metal.lora.rwkvq_linear import RwkvqLinear
from rwkv_metal.lora.rwkvq_kernel import dequant_dense

ref = mx.load("/tmp/ref_bits_check.safetensors")
keys = ["blocks.0.att.key.weight", "blocks.5.ffn.value.weight", "emb.weight"]
ref_keys = ["blocks_0_att_key_weight", "blocks_5_ffn_value_weight", "emb_weight"]

for k, rk in zip(keys, ref_keys):
    lin = RwkvqLinear.from_sidecar("/tmp/reduction_v2.rwkvq_mlx", k)
    w32 = dequant_dense(lin.qblk, lin.qsqm, lin.ddm, lin.out_features, lin.in_features,
                         gw_sb=8, xbits=lin.xbits)
    w = w32.astype(mx.bfloat16)
    mx.eval(w)
    r = ref[rk]
    got_bits = np.array(w.view(mx.uint16))
    ref_bits = np.array(r.view(mx.uint16))
    n_mis = int((got_bits != ref_bits).sum())
    print(f"{k:30s} shape={tuple(w.shape)} xbits={lin.xbits} mismatch={n_mis}/{w.size}")
    if n_mis:
        idx = np.argwhere(got_bits != ref_bits)[:5]
        for r0, c0 in idx:
            print(f"   sample mismatch at [{r0},{c0}]: got={got_bits[r0,c0]:#06x} ref={ref_bits[r0,c0]:#06x}")

print()
print("=== speed: fused kernel vs композитный MLX-порт (RwkvqLinear._dequant_w) ===")
for k in ["blocks.0.att.key.weight", "blocks.5.ffn.value.weight", "emb.weight"]:
    lin = RwkvqLinear.from_sidecar("/tmp/reduction_v2.rwkvq_mlx", k)

    def fused():
        w32 = dequant_dense(lin.qblk, lin.qsqm, lin.ddm, lin.out_features, lin.in_features,
                             gw_sb=8, xbits=lin.xbits)
        return w32.astype(mx.bfloat16)

    def composite():
        return lin._dequant_w()

    for name, fn in (("fused", fused), ("composite", composite)):
        w = fn(); mx.eval(w)
        N = 20
        t0 = time.time()
        for _ in range(N):
            w = fn(); mx.eval(w)
        dt = (time.time() - t0) / N * 1000
        print(f"  {k:30s} {name:10s} {dt:7.3f} ms  shape={tuple(w.shape)}")
