"""Регрессия для параметризованного wkv7_infer: произвольный T против
пошагового T=1 и против прежнего поведения (T=CHUNK)."""
import sys; sys.path.insert(0, ".")
import numpy as np, mlx.core as mx
from rwkv_metal.kernel.wkv7 import wkv7_infer, CHUNK, HEAD_SIZE

np.random.seed(0)
B, H, D = 2, 4, HEAD_SIZE
def rand(T):
    r,k,v,a,b = [mx.array(np.random.randn(B,T,H,D).astype(np.float32))*0.3 for _ in range(5)]
    w = mx.array(np.exp(-np.exp(np.random.randn(B,T,H,D).astype(np.float32))))
    return r,w,k,v,a,b

ok = True
for T in (1, 5, CHUNK, 33):
    r,w,k,v,a,b = rand(T)
    h0 = mx.array(np.random.randn(B,H,D,D).astype(np.float32))*0.1
    o_full, h_full = wkv7_infer(r,w,k,v,a,b,h0)
    outs, hh = [], h0
    for t in range(T):
        sl = lambda x: x[:, t:t+1]
        o, hh = wkv7_infer(sl(r),sl(w),sl(k),sl(v),sl(a),sl(b),hh)
        outs.append(o)
    e  = float(mx.abs(mx.concatenate(outs,axis=1)-o_full).max())
    eh = float(mx.abs(hh-h_full).max())
    print(f"T={T:>2}: out err {e:.2e}, state err {eh:.2e}")
    ok &= e == 0.0 and eh == 0.0
print("[OK]" if ok else "[FAIL]"); sys.exit(0 if ok else 1)
