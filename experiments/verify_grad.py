import os, sys
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
from rwkv_metal.kernel.wkv7 import wkv7_train, wkv7_train_py

mx.random.seed(0)
B, T, H, D = 2, 64, 4, 64
def rnd(s=1.0): return mx.random.normal((B, T, H, D)) * s
def l2(x): return x / mx.sqrt((x * x).sum(-1, keepdims=True) + 1e-12)
r = rnd(0.5); v = rnd(0.5); k = rnd(0.5)
kk = l2(k); iclr = mx.sigmoid(rnd())
a = -kk; b = kk * iclr
w = mx.exp(-0.606531 * mx.sigmoid(rnd()))
g = rnd()                              # фикс. котангента
mx.eval(r, w, k, v, a, b, g)

def L_metal(r, w, k, v, a, b): return (wkv7_train(r, w, k, v, a, b) * g).sum()
def L_ref(r, w, k, v, a, b):   return (wkv7_train_py(r, w, k, v, a, b) * g).sum()
gm = mx.grad(L_metal, argnums=(0, 1, 2, 3, 4, 5))(r, w, k, v, a, b)
gr = mx.grad(L_ref,   argnums=(0, 1, 2, 3, 4, 5))(r, w, k, v, a, b)
mx.eval(gm, gr)

print("  паритет градиентов Metal vjp vs Python autograd:", flush=True)
ok = True
for nm, x, y in zip(["dr", "dw", "dk", "dv", "da", "db"], gm, gr):
    d = float(mx.max(mx.abs(x - y))); sc = float(mx.max(mx.abs(y))) + 1e-9
    rel = d / sc; ok = ok and rel < 1e-3
    print(f"    {nm}: max|d|={d:.3e}  rel={rel:.2e}", flush=True)
print(f"  -> {'OK: backward совпадает с эталоном' if ok else '!!! РАСХОЖДЕНИЕ'}", flush=True)
