import os, sys
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dplr_mlx import dplr_recurrence_mlx
from s3_dplr_kernel import dplr_bwd_metal_bh_batched, dplr_bwd_metal_bh_fast
from verify_s34b import rel, mk

wt = wb = 0.0
for BH, N in [(8, 2), (8, 4), (32, 4), (8, 16)]:
    for wl in [None, 0.270]:
        T = N * 16
        heads = [mk(h, T, 64, wl) for h in range(BH)]
        R, W, K, V, A, B = (mx.stack([h[i] for h in heads]) for i in range(6))
        do = mx.stack([h[6] for h in heads])
        gF = dplr_bwd_metal_bh_fast(R, W, K, V, A, B, do, 16)
        gB = dplr_bwd_metal_bh_batched(R, W, K, V, A, B, do, 16)
        mx.eval(*gF, *gB)
        eb = max(rel(gB[i], gF[i]) for i in range(6))
        et = 0.0
        for h in range(min(BH, 4)):
            rr, ww, kk, vv, aa, bb_, dd = (*heads[h][:6], do[h])
            def lT(r, w, k, v, a, b):
                x = lambda z: z[None, :, None, :]
                return mx.sum(dplr_recurrence_mlx(x(r), x(w), x(k), x(v), x(a), x(b))[0, :, 0] * dd)
            gT = mx.grad(lT, argnums=[0, 1, 2, 3, 4, 5])(rr, ww, kk, vv, aa, bb_)
            mx.eval(*gT)
            et = max(et, max(rel(gT[i], gF[i][h]) for i in range(6)))
        wt = max(wt, et); wb = max(wb, eb)
        lab = "model" if wl is None else f"w={wl}"
        print(f"BH={BH:>3} N={N:>2} {lab:>8}: vs batched={eb:.2e}  vs truth={et:.2e}")
print(f"WORST vs batched={wb:.2e}  vs truth={wt:.2e}  " + ("PASS" if max(wt, wb) < 1e-4 else "FAIL"))
