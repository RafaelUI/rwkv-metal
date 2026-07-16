"""verify_s34b_metal_bh.py — S3.4b-ii BH-батч: батч-драйвер vs поголовный single-head
(доказан vs истина) и vs ИСТИНА (autograd рекуррентности по головам). BH,N,w-свип."""
import os, sys
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dplr_mlx import dplr_recurrence_mlx
from s3_dplr_kernel import dplr_bwd_metal, dplr_bwd_metal_bh
from verify_s34b import rel, mk


def main(C=16, D=64, BH=6, seeds=(0, 1)):
    print(f"S3.4b-ii BH-батч (BH={BH}, C={C}, D={D}, fp32):")
    names = ["dr", "dw", "dk", "dv", "da", "db"]
    worst_loop = worst_truth = 0.0
    for N in (2, 3):
        T = N * C
        for wl in [None, 0.545, 0.270]:
            el = et = 0.0
            for seed in seeds:
                heads = [mk(seed * 100 + h, T, D, wl) for h in range(BH)]
                R = mx.stack([h[0] for h in heads]); W = mx.stack([h[1] for h in heads])
                K = mx.stack([h[2] for h in heads]); V = mx.stack([h[3] for h in heads])
                A = mx.stack([h[4] for h in heads]); B = mx.stack([h[5] for h in heads])
                do = mx.stack([h[6] for h in heads])
                gB = dplr_bwd_metal_bh(R, W, K, V, A, B, do, C)
                # поголовный single-head (proven)
                per = [dplr_bwd_metal(*heads[h][:6], do[h], C) for h in range(BH)]
                gL = [mx.stack([per[h][i] for h in range(BH)]) for i in range(6)]
                mx.eval(*gB, *gL)
                el = max(el, max(rel(gL[i], gB[i]) for i in range(6)))
                # vs истина (по головам)
                for h in range(BH):
                    rr, ww, kk, vv, aa, bb_, dd = (*heads[h][:6], do[h])

                    def lT(r, w, k, v, a, b):
                        x = lambda z: z[None, :, None, :]
                        return mx.sum(dplr_recurrence_mlx(x(r), x(w), x(k), x(v), x(a), x(b))[0, :, 0] * dd)
                    gT = mx.grad(lT, argnums=[0, 1, 2, 3, 4, 5])(rr, ww, kk, vv, aa, bb_)
                    mx.eval(*gT)
                    et = max(et, max(rel(gT[i], gB[i][h]) for i in range(6)))
            worst_loop = max(worst_loop, el); worst_truth = max(worst_truth, et)
            lab = "model" if wl is None else f"w={wl:.3f}"
            print(f"  N={N} {lab:>9}: batch vs per-head={el:.2e}  batch vs truth={et:.2e}")
    print(f"WORST vs per-head={worst_loop:.2e}  vs truth={worst_truth:.2e}  "
          + ("PASS" if max(worst_loop, worst_truth) < 1e-4 else "FAIL"))


if __name__ == "__main__":
    main()
