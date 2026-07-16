"""verify_s34b_saved.py — S3.4b-iii: полный Metal fwd(SAVE)+bwd БЕЗ MLX в тяжёлом пути,
vs ИСТИНА (autograd рек.) и vs recompute-MLX-драйвер (должны совпасть бит-близко)."""
import os, sys
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dplr_mlx import dplr_recurrence_mlx
from s3_dplr_kernel import dplr_bwd_metal_bh, dplr_bwd_metal_bh_saved
from verify_s34b import rel, mk


def main(C=16, D=64, BH=6, seeds=(0, 1)):
    print(f"S3.4b-iii Metal fwd(SAVE)+bwd, без MLX (BH={BH}, C={C}, D={D}):")
    wt = wr = 0.0
    for N in (2, 3, 4):
        T = N * C
        for wl in [None, 0.545, 0.270]:
            et = er = 0.0
            for seed in seeds:
                heads = [mk(seed * 100 + h, T, D, wl) for h in range(BH)]
                R, W, K, V, A, B = (mx.stack([h[i] for h in heads]) for i in range(6))
                do = mx.stack([h[6] for h in heads])
                gS = dplr_bwd_metal_bh_saved(R, W, K, V, A, B, do, C)
                gR = dplr_bwd_metal_bh(R, W, K, V, A, B, do, C)   # recompute-MLX путь
                mx.eval(*gS, *gR)
                er = max(er, max(rel(gR[i], gS[i]) for i in range(6)))
                for h in range(BH):
                    rr, ww, kk, vv, aa, bb_, dd = (*heads[h][:6], do[h])

                    def lT(r, w, k, v, a, b):
                        x = lambda z: z[None, :, None, :]
                        return mx.sum(dplr_recurrence_mlx(x(r), x(w), x(k), x(v), x(a), x(b))[0, :, 0] * dd)
                    gT = mx.grad(lT, argnums=[0, 1, 2, 3, 4, 5])(rr, ww, kk, vv, aa, bb_)
                    mx.eval(*gT)
                    et = max(et, max(rel(gT[i], gS[i][h]) for i in range(6)))
            wt = max(wt, et); wr = max(wr, er)
            lab = "model" if wl is None else f"w={wl:.3f}"
            print(f"  N={N} {lab:>9}: vs truth={et:.2e}  vs recompute-путь={er:.2e}")
    print(f"WORST vs truth={wt:.2e}  vs recompute={wr:.2e}  "
          + ("PASS" if max(wt, wr) < 1e-4 else "FAIL"))


if __name__ == "__main__":
    main()
