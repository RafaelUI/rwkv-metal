"""verify_s34b_metal.py — S3.4b-ii: полный Metal межчанк-backward vs ИСТИНА
(autograd рекуррентности) и vs MLX-оракул chunk_bwd_seq. Single head, N=2,3,4, w-свип."""
import os, sys
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dplr_bwd_chunk_mlx import chunk_bwd_seq
from dplr_mlx import dplr_recurrence_mlx
from s3_dplr_kernel import dplr_bwd_metal
from verify_s34b import bb, rel, mk


def main(C=16, D=64, seeds=(0, 1, 2)):
    print(f"S3.4b-ii Metal inter-chunk backward vs ИСТИНА & MLX-оракул (C={C}, D={D}):")
    names = ["dr", "dw", "dk", "dv", "da", "db"]
    worst_t = worst_o = 0.0
    for N in (2, 3, 4):
        T = N * C
        for wl in [None, 0.747, 0.545, 0.270]:
            et = eo = 0.0
            for seed in seeds:
                r, w, k, v, a, b, do = mk(seed, T, D, wl)
                gM = dplr_bwd_metal(r, w, k, v, a, b, do, C)
                gO = chunk_bwd_seq(r, w, k, v, a, b, do, C)

                def lT(r, w, k, v, a, b):
                    return mx.sum(dplr_recurrence_mlx(bb(r), bb(w), bb(k), bb(v),
                                                      bb(a), bb(b))[0, :, 0] * do)
                gT = mx.grad(lT, argnums=[0, 1, 2, 3, 4, 5])(r, w, k, v, a, b)
                mx.eval(*gM, *gO, *gT)
                et = max(et, max(rel(gT[i], gM[i]) for i in range(6)))
                eo = max(eo, max(rel(gO[i], gM[i]) for i in range(6)))
            worst_t = max(worst_t, et); worst_o = max(worst_o, eo)
            lab = "model" if wl is None else f"w={wl:.3f}"
            print(f"  N={N} {lab:>9}: vs truth={et:.2e}  vs MLX-оракул={eo:.2e}")
    print(f"WORST vs truth={worst_t:.2e}  vs MLX={worst_o:.2e}  "
          + ("PASS" if max(worst_t, worst_o) < 1e-4 else "FAIL"))


if __name__ == "__main__":
    main()
