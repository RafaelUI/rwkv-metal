"""verify_chunk_one.py — S3.4b-ii микрогард уровня MLX:
hand per-chunk bwd (chunk_bwd_one) vs autograd(vjp) изолированного chunk_fwd_one.
Случайные S_in и dS — изолирует ВСЕ S-ветви и carry без многочанкового цикла."""
import os, sys
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dplr_bwd_one_mlx import chunk_fwd_one, chunk_bwd_one


def rel(ref, got):
    return (mx.max(mx.abs(ref - got)) / (mx.max(mx.abs(ref)) + 1e-12)).item()


def mk(seed, C, D, wl):
    mx.random.seed(seed)
    r = mx.random.normal((C, D)) * 0.5
    k = mx.random.normal((C, D)) * 0.5
    v = mx.random.normal((C, D)) * 0.5
    kk = mx.random.normal((C, D)); kk = kk / (mx.linalg.norm(kk, axis=-1, keepdims=True) + 1e-6)
    a = -kk; b = kk * 0.1
    w = (mx.exp(-0.606531 * mx.sigmoid(mx.random.normal((C, D))))
         if wl is None else mx.full((C, D), wl))
    S_in = mx.random.normal((D, D)) * 0.3
    do = mx.random.normal((C, D))
    dS = mx.random.normal((D, D)) * 0.3
    return r, w, k, v, a, b, S_in, do, dS


def main(C=16, D=64, seeds=(0, 1, 2, 3)):
    print(f"S3.4b-ii per-chunk MLX hand-bwd vs autograd (C={C}, D={D}, fp32):")
    names = ["dr", "dw", "dk", "dv", "da", "db", "dS_in"]
    worst = 0.0
    for wl in [None, 0.747, 0.545, 0.270]:
        wr = 0.0
        for seed in seeds:
            r, w, k, v, a, b, S_in, do, dS = mk(seed, C, D, wl)
            hand = chunk_bwd_one(r, w, k, v, a, b, S_in, dS, do, C)

            def f(r, w, k, v, a, b, S_in):
                o, S_out, _ = chunk_fwd_one(r, w, k, v, a, b, S_in, C)
                return o, S_out
            _, vjp = mx.vjp(f, (r, w, k, v, a, b, S_in), (do, dS))
            mx.eval(*hand, *vjp)
            for i in range(7):
                wr = max(wr, rel(vjp[i], hand[i]))
        worst = max(worst, wr)
        lab = "model" if wl is None else f"w={wl:.3f}"
        print(f"  {lab:>9}: max rel(hand vs autograd) over {names} = {wr:.2e}")
    print(f"WORST = {worst:.2e}  " + ("PASS" if worst < 1e-4 else "FAIL"))


if __name__ == "__main__":
    main()
