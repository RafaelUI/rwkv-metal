import os, sys, importlib
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
W  = importlib.import_module("rwkv_metal.kernel.wkv7")
WC = importlib.import_module("rwkv_metal.kernel.wkv7_checkpoint")

mx.random.seed(0)
B, T, H, D = 2, 64, 4, 64
def rnd(s=1.0): return mx.random.normal((B, T, H, D)) * s
def l2(x): return x / mx.sqrt((x * x).sum(-1, keepdims=True) + 1e-12)
r = rnd(0.5); v = rnd(0.5); k = rnd(0.5)
kk = l2(k); iclr = mx.sigmoid(rnd())
a = -kk; b = kk * iclr
w = mx.exp(-0.606531 * mx.sigmoid(rnd()))
g = rnd()
mx.eval(r, w, k, v, a, b, g)

def L_ref(r, w, k, v, a, b): return (W.wkv7_train_py(r, w, k, v, a, b) * g).sum()
gr = mx.grad(L_ref, argnums=(0,1,2,3,4,5))(r, w, k, v, a, b); mx.eval(gr)

def rel(x, y):
    return float(mx.max(mx.abs(x - y))) / (float(mx.max(mx.abs(y))) + 1e-9)

print(f"  CHUNK :   dr        dw        da        | dk(контроль)", flush=True)
for c in [64, 32, 16, 8]:
    W.CHUNK = c; WC.CHUNK = c; W._ckpt_cache.clear(); WC._fwd_cache.clear(); WC._bwd_cache.clear()
    def L_m(r, w, k, v, a, b): return (W.wkv7_train(r, w, k, v, a, b) * g).sum()
    gm = mx.grad(L_m, argnums=(0,1,2,3,4,5))(r, w, k, v, a, b); mx.eval(gm)
    dr, dw, dk, dv, da, db = (rel(gm[i], gr[i]) for i in range(6))
    print(f"   {c:3d}  : {dr:.2e}  {dw:.2e}  {da:.2e}  | {dk:.2e}", flush=True)
