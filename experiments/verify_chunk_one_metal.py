"""verify_chunk_one_metal.py — S3.4b-ii микрогард: Metal per-chunk bwd (KB+_bwd2+хост)
vs MLX-оракул chunk_bwd_one. Single head, случайные S_in/dS — изолирует KB-формы."""
import os, sys
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dplr_bwd_one_mlx import chunk_bwd_one
from s3_dplr_kernel import chunk_bwd_one_metal
from verify_chunk_one import mk


def rel(ref, got):
    return (mx.max(mx.abs(ref - got)) / (mx.max(mx.abs(ref)) + 1e-12)).item()


def main(C=16, D=64, seeds=(0, 1, 2, 3)):
    print(f"S3.4b-ii Metal per-chunk bwd vs MLX-оракул (C={C}, D={D}, fp32):")
    names = ["dr", "dw", "dk", "dv", "da", "db", "dS_in"]
    worst = 0.0
    for wl in [None, 0.747, 0.545, 0.270]:
        per = [0.0] * 7
        for seed in seeds:
            r, w, k, v, a, b, S_in, do, dS = mk(seed, C, D, wl)
            ref = chunk_bwd_one(r, w, k, v, a, b, S_in, dS, do, C)
            got = chunk_bwd_one_metal(r, w, k, v, a, b, S_in, dS, do)
            mx.eval(*ref, *got)
            for i in range(7):
                per[i] = max(per[i], rel(ref[i], got[i]))
        wr = max(per)
        worst = max(worst, wr)
        lab = "model" if wl is None else f"w={wl:.3f}"
        det = " ".join(f"{n}={e:.1e}" for n, e in zip(names, per))
        print(f"  {lab:>9}: max={wr:.2e} | {det}")
    print(f"WORST = {worst:.2e}  " + ("PASS" if worst < 1e-4 else "FAIL"))


if __name__ == "__main__":
    main()
