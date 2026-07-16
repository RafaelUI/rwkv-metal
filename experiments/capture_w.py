import os, sys
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
from rwkv_metal.pretrain.config import PretrainConfig
from rwkv_metal.model.rwkv7 import RWKV7
import rwkv_metal.model.rwkv7 as M

CKPT = "/Users/s/Develop/rwkvNeuro/checkpoints/rwkv7_12l256d_best.npz"
cfg = PretrainConfig(n_layer=12, n_embd=256, head_size=64, vocab_size=32000,
                     ctx_len=512, batch_size=4, dtype="float32")
model = RWKV7(cfg)
try:
    model.load_weights(CKPT)
except Exception:
    model.load_weights(CKPT, strict=False)
model.set_dtype("float32")

orig = M.wkv7
buf = []
def cap(r, w, k, v, a, b, **kw):
    buf.append(np.array(w, copy=True).ravel())
    return orig(r, w, k, v, a, b, **kw)
M.wkv7 = cap

mx.random.seed(0)
x = mx.random.randint(1, cfg.vocab_size, (4, 512)); mx.eval(x)
out = model.body(x); mx.eval(out)

w_all = np.concatenate(buf)
print(f"  собрано w: {w_all.size:,} значений из {len(buf)} вызовов (слоёв)", flush=True)
qs = [0, 0.1, 1, 5, 50, 95, 99, 100]
vals = np.percentile(w_all, qs)
print("  перцентили w:", flush=True)
for q, val in zip(qs, vals):
    print(f"    p{q:<5}: {val:.4f}", flush=True)
print(f"  теоретический пол exp(-0.606531) = {np.exp(-0.606531):.4f}", flush=True)
for thr in [0.60, 0.65, 0.70, 0.80]:
    frac = float((w_all < thr).mean())
    print(f"  доля w < {thr:.2f} : {100*frac:6.2f}%", flush=True)
print(f"  min(w) = {w_all.min():.4f}   mean(w) = {w_all.mean():.4f}", flush=True)
