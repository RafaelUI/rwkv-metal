"""verify_s34b.py — S3.4b: межчанк MLX backward (carry dS) vs ИСТИНА (autograd рекуррентности)."""
import os, sys
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dplr_bwd_chunk_mlx import chunk_fwd_seq, chunk_bwd_seq
from dplr_mlx import dplr_recurrence_mlx


def bb(x):
    return x[None, :, None, :]


def rel(ref, got):
    return (mx.max(mx.abs(ref - got)) / (mx.max(mx.abs(ref)) + 1e-12)).item()


def mk(seed, T, D, wl):
    mx.random.seed(seed)
    r = mx.random.normal((T, D)) * 0.5
    k = mx.random.normal((T, D)) * 0.5
    v = mx.random.normal((T, D)) * 0.5
    kk = mx.random.normal((T, D)); kk = kk / (mx.linalg.norm(kk, axis=-1, keepdims=True) + 1e-6)
    a = -kk; b = kk * 0.1
    w = (mx.exp(-0.606531 * mx.sigmoid(mx.random.normal((T, D))))
         if wl is None else mx.full((T, D), wl))
    do = mx.random.normal((T, D))
    return r, w, k, v, a, b, do


def main(C=16, D=64, seeds=(0, 1, 2)):
    print(f"S3.4b inter-chunk MLX backward (carry dS) (C={C}, D={D}, fp32):")
    worst_fwd = worst_grad = 0.0
    for N in (2, 3, 4):
        T = N * C
        for wl in [None, 0.747, 0.545, 0.270]:
            ef = eg = 0.0
            for seed in seeds:
                r, w, k, v, a, b, do = mk(seed, T, D, wl)
                o_k, _ = chunk_fwd_seq(r, w, k, v, a, b, C)
                o_ref = dplr_recurrence_mlx(bb(r), bb(w), bb(k), bb(v), bb(a), bb(b))[0, :, 0]
                mx.eval(o_k, o_ref)
                ef = max(ef, rel(o_ref, o_k))

                gA = chunk_bwd_seq(r, w, k, v, a, b, do, C)

                def lT(r, w, k, v, a, b):
                    return mx.sum(dplr_recurrence_mlx(bb(r), bb(w), bb(k), bb(v),
                                                      bb(a), bb(b))[0, :, 0] * do)
                gT = mx.grad(lT, argnums=[0, 1, 2, 3, 4, 5])(r, w, k, v, a, b)
                mx.eval(*gA, *gT)
                eg = max(eg, max(rel(gT[i], gA[i]) for i in range(6)))

            worst_fwd = max(worst_fwd, ef); worst_grad = max(worst_grad, eg)
            lab = "model" if wl is None else f"w={wl:.3f}"
            print(f"  N={N} {lab:>9}: fwd rel={ef:.2e}  grad rel vs truth={eg:.2e}")
    print(f"WORST fwd={worst_fwd:.2e}  grad={worst_grad:.2e}  "
          + ("PASS" if max(worst_fwd, worst_grad) < 1e-4 else "FAIL"))


if __name__ == "__main__":
    main()
