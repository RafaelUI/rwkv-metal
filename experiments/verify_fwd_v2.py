"""verify_fwd_v2.py — L2: расщеплённый форвард (WY+fscan) бит-близко к dplr_forward_metal_save."""
import os, sys
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s3_dplr_kernel import dplr_forward_metal_save, dplr_forward_metal_save_v2
from verify_s34b import rel, mk

C, D = 16, 64
worst = 0.0
for BH, N in [(8, 2), (8, 4), (32, 4), (8, 16)]:
    for wl in [None, 0.270]:
        T = N * C
        heads = [mk(h, T, D, wl) for h in range(BH)]
        R, W, K, V, A, B = (mx.stack([h[i] for h in heads]) for i in range(6))
        o1, c1 = dplr_forward_metal_save(R, W, K, V, A, B, C)
        o2, c2 = dplr_forward_metal_save_v2(R, W, K, V, A, B, C)
        eo = rel(o1, o2)
        ec = 0.0
        for key in ["S_in", "Am", "u", "wmat", "v2"]:
            for n in range(N):
                ec = max(ec, rel(c1[key][n], c2[key][n]))
        mx.eval(o1, o2)
        worst = max(worst, eo, ec)
        lab = "model" if wl is None else f"w={wl}"
        print(f"BH={BH:>3} N={N:>2} {lab:>7}: o={eo:.2e}  cache={ec:.2e}")
print(f"WORST={worst:.2e}  " + ("PASS" if worst < 1e-4 else "FAIL"))
