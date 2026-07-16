import os, sys
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
from rwkv_metal.kernel.wkv7 import wkv7_train, wkv7_train_py

mode = sys.argv[1] if len(sys.argv) > 1 else "check"
mx.random.seed(0)
B, T, H, D = 2, 64, 4, 64
def rnd(s=1.0): return mx.random.normal((B, T, H, D)) * s
def l2(x): return x / mx.sqrt((x * x).sum(-1, keepdims=True) + 1e-12)

# Реалистичный режим RWKV-7 (контрактивный):
r = rnd(0.5)
v = rnd(0.5)
k = rnd(0.5)
kk = l2(k)                                   # |kk| = 1
iclr = mx.sigmoid(rnd())                      # (0,1)
a = -kk                                       # как в модели
b = kk * iclr                                 # как в модели
w = mx.exp(-0.606531 * mx.sigmoid(rnd()))     # decay (~0.55, 1.0)
mx.eval(r, w, k, v, a, b)

out_metal = wkv7_train(r, w, k, v, a, b); mx.eval(out_metal)
out_ref   = wkv7_train_py(r, w, k, v, a, b); mx.eval(out_ref)
scale = float(mx.max(mx.abs(out_ref))) + 1e-9
d_ref = float(mx.max(mx.abs(out_metal - out_ref)))
print(f"  parity Metal vs Python einsum : max|d| = {d_ref:.3e}  (rel {d_ref/scale:.2e}, |out|~{scale:.2f})", flush=True)

om = np.array(out_metal, copy=True)
base = "/tmp/fwd_base.npy"
if mode == "save":
    np.save(base, om); print(f"  baseline сохранён -> {base}", flush=True)
elif os.path.exists(base):
    ob = np.load(base); d = float(np.max(np.abs(om - ob)))
    tag = "BIT-IDENTICAL" if d == 0 else ("OK (fp32 noise)" if d < 1e-4 else "!!! РАСХОЖДЕНИЕ")
    print(f"  до/после смены dispatch       : max|d| = {d:.3e}  {tag}", flush=True)
