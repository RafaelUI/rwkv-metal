"""verify_s3.py — S3.1: A-матрицы фьюзед-кернела vs MLX-оракул (стабильная форма)."""
import os, sys
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s3_dplr_kernel import compute_amats


def main(C=16, D=64, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((C, D)) * 0.5
    k = mx.random.normal((C, D)) * 0.5
    kk = mx.random.normal((C, D)); kk = kk / (mx.linalg.norm(kk, axis=-1, keepdims=True) + 1e-6)
    alpha = -kk; beta = kk * 0.1
    gk = -0.606531 * mx.sigmoid(mx.random.normal((C, D)))
    gc = mx.cumsum(gk, axis=0)

    ii = mx.arange(C)[:, None]; jj = mx.arange(C)[None, :]
    le = (jj <= ii); lt = (jj < ii)
    diff = gc[:, None, :] - gc[None, :, :]
    diff_s = gc[:, None, :] - gk[:, None, :] - gc[None, :, :]
    def ref(L, R, m, dd):
        return mx.where(m, mx.sum(L[:, None, :] * R[None, :, :] * mx.exp(dd), axis=-1), 0.0)
    refs = {"A_qk": ref(q, k, le, diff), "A_qb": ref(q, beta, le, diff),
            "A_ab": ref(alpha, beta, lt, diff_s), "A_ak": ref(alpha, k, lt, diff_s)}
    masks = {"A_qk": le, "A_qb": le, "A_ab": lt, "A_ak": lt}

    raw = compute_amats(q, k, alpha, beta, gc, gk); mx.eval(raw)
    print(f"S3.1 A-matrices vs oracle (C={C}, D={D}, fp32):")
    ok = True
    for name, rw in zip(["A_qk", "A_qb", "A_ab", "A_ak"], raw):
        g = mx.where(masks[name], rw, 0.0)
        e = mx.max(mx.abs(g - refs[name])).item()
        ok &= e < 1e-4
        print(f"  {name}: max_abs={e:.3e}")
    print("PASS" if ok else "FAIL")




def main_s32(C=16, D=64, seed=0):
    from s3_dplr_kernel import trisolve
    mx.random.seed(seed)
    v = mx.random.normal((C, D)) * 0.5
    k = mx.random.normal((C, D)) * 0.5
    kk = mx.random.normal((C, D)); kk = kk / (mx.linalg.norm(kk, axis=-1, keepdims=True) + 1e-6)
    alpha = -kk; beta = kk * 0.1
    gk = -0.606531 * mx.sigmoid(mx.random.normal((C, D))); gc = mx.cumsum(gk, axis=0)
    ii = mx.arange(C)[:, None]; jj = mx.arange(C)[None, :]; lt = (jj < ii)
    diff_s = gc[:, None, :] - gk[:, None, :] - gc[None, :, :]
    A_ab = mx.where(lt, mx.sum(alpha[:, None, :] * beta[None, :, :] * mx.exp(diff_s), -1), 0.0)
    A_ak = mx.where(lt, mx.sum(alpha[:, None, :] * k[None, :, :] * mx.exp(diff_s), -1), 0.0)
    A_inv = mx.eye(C); P = mx.eye(C)
    for _ in range(C - 1):
        P = P @ A_ab; A_inv = A_inv + P
    RHS_u = A_ak @ v; RHS_w = mx.exp(gc - gk) * alpha
    eu = mx.max(mx.abs(trisolve(A_ab, RHS_u) - A_inv @ RHS_u)).item()
    ew = mx.max(mx.abs(trisolve(A_ab, RHS_w) - A_inv @ RHS_w)).item()
    print(f"S3.2 trisolve vs A_inv@RHS (C={C}, D={D}, fp32):")
    print(f"  u   : max_abs={eu:.3e}")
    print(f"  wmat: max_abs={ew:.3e}")
    print("PASS" if max(eu, ew) < 1e-4 else "FAIL")



def main_s33a(C=16, D=64, seed=0):
    from s3_dplr_kernel import compute_amats_masked
    mx.random.seed(seed)
    q = mx.random.normal((C, D)) * 0.5; k = mx.random.normal((C, D)) * 0.5
    kk = mx.random.normal((C, D)); kk = kk / (mx.linalg.norm(kk, axis=-1, keepdims=True) + 1e-6)
    alpha = -kk; beta = kk * 0.1
    gk = -0.606531 * mx.sigmoid(mx.random.normal((C, D))); gc = mx.cumsum(gk, axis=0)
    ii = mx.arange(C)[:, None]; jj = mx.arange(C)[None, :]; le = (jj <= ii); lt = (jj < ii)
    diff = gc[:, None, :] - gc[None, :, :]; diff_s = gc[:, None, :] - gk[:, None, :] - gc[None, :, :]
    def ref(L, R, m, dd): return mx.where(m, mx.sum(L[:, None, :] * R[None, :, :] * mx.exp(dd), -1), 0.0)
    refs = [ref(q, k, le, diff), ref(q, beta, le, diff), ref(alpha, beta, lt, diff_s), ref(alpha, k, lt, diff_s)]
    r = compute_amats_masked(q, k, alpha, beta, gc, gk); mx.eval(r)
    print(f"S3.3a masked A-matrices in-kernel vs masked oracle (C={C}, D={D}):")
    ok = True
    for nm, g, rf in zip(["A_qk", "A_qb", "A_ab", "A_ak"], r, refs):
        e = mx.max(mx.abs(g - rf)).item(); ok &= e < 1e-4
        print(f"  {nm}: max_abs={e:.3e}")
    print("PASS" if ok else "FAIL")




def main_s33b(C=16, D=64, seed=0):
    """S3.3b: фьюзед forward одного чанка (S=0) vs wkv7_train_py и MLX-оракул."""
    import sys, os
    sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
    from s3_dplr_kernel import chunk_fwd_s0
    from rwkv_metal.kernel.wkv7 import wkv7_train_py
    from dplr_mlx import dplr_chunkwise_mlx

    mx.random.seed(seed)
    q = mx.random.normal((C, D)) * 0.5
    k = mx.random.normal((C, D)) * 0.5
    v = mx.random.normal((C, D)) * 0.5
    kk = mx.random.normal((C, D)); kk = kk / (mx.linalg.norm(kk, axis=-1, keepdims=True) + 1e-6)
    alpha = -kk; beta = kk * 0.1
    gk = -0.606531 * mx.sigmoid(mx.random.normal((C, D)))
    gc = mx.cumsum(gk, axis=0)
    w = mx.exp(gk)

    o_k = chunk_fwd_s0(q, k, v, alpha, beta, gc, gk); mx.eval(o_k)

    # эталон 1: боевой wkv7_train_py на [1,C,1,D] (S=0)
    def b(x): return x[None, :, None, :]
    o_ref = wkv7_train_py(b(q), b(w), b(k), b(v), b(alpha), b(beta))[0, :, 0]
    # эталон 2: MLX-оракул чанковый при N=1 (T=C)
    o_orc = dplr_chunkwise_mlx(b(q), b(w), b(k), b(v), b(alpha), b(beta), chunk_size=C)[0, :, 0]
    mx.eval(o_ref, o_orc)

    e_ref = mx.max(mx.abs(o_k - o_ref)).item()
    e_orc = mx.max(mx.abs(o_k - o_orc)).item()
    scale = mx.max(mx.abs(o_ref)).item()
    print(f"S3.3b fused chunk fwd (S=0) (C={C}, D={D}, fp32):")
    print(f"  vs wkv7_train_py : max_abs={e_ref:.3e}  (|o|max={scale:.3e})")
    print(f"  vs MLX oracle N=1: max_abs={e_orc:.3e}")
    print("PASS" if max(e_ref, e_orc) < 1e-4 else "FAIL")


def main_s33c(C=16, D=64, seed=0):
    """S3.3c: полный forward через step-кернел (межчанк S + B*H) vs wkv7_train_py и MLX-оракул."""
    import sys, os
    sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
    from s3_dplr_kernel import dplr_forward_metal
    from rwkv_metal.kernel.wkv7 import wkv7_train_py
    from dplr_mlx import dplr_chunkwise_mlx

    print(f"S3.3c full forward (step-kernel, B*H unroll) (C={C}, D={D}, fp32):")
    ok = True
    for B, H, N in [(1, 1, 2), (2, 3, 3), (2, 2, 4)]:
        mx.random.seed(seed)
        T = N * C
        r = mx.random.normal((B, T, H, D)) * 0.5
        k = mx.random.normal((B, T, H, D)) * 0.5
        v = mx.random.normal((B, T, H, D)) * 0.5
        kk = mx.random.normal((B, T, H, D)); kk = kk / (mx.linalg.norm(kk, axis=-1, keepdims=True) + 1e-6)
        a = -kk; b = kk * 0.1
        w = mx.exp(-0.606531 * mx.sigmoid(mx.random.normal((B, T, H, D))))

        o_m = dplr_forward_metal(r, w, k, v, a, b, chunk_size=C); mx.eval(o_m)
        o_ref = wkv7_train_py(r, w, k, v, a, b)
        o_orc = dplr_chunkwise_mlx(r, w, k, v, a, b, chunk_size=C)
        mx.eval(o_ref, o_orc)
        e_ref = mx.max(mx.abs(o_m - o_ref)).item()
        e_orc = mx.max(mx.abs(o_m - o_orc)).item()
        sc = mx.max(mx.abs(o_ref)).item()
        ok &= max(e_ref, e_orc) < 1e-4
        print(f"  B={B} H={H} N={N} (T={T}): vs train_py={e_ref:.3e}  vs oracle={e_orc:.3e}  (|o|max={sc:.2e})")
    print("PASS" if ok else "FAIL")

def main_s34a(C=16, D=64, seeds=(0, 1, 2)):
    """S3.4a-ii: Metal single-chunk backward (S=0) vs MLX-hand-bwd и vs ИСТИНА; свип w."""
    import sys, os
    sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
    from s3_dplr_kernel import chunk_bwd_s0_metal
    from dplr_bwd_mlx import chunk_bwd_s0_mlx
    from dplr_mlx import dplr_recurrence_mlx

    def mk(seed, wl):
        mx.random.seed(seed)
        r = mx.random.normal((C, D)) * 0.5; k = mx.random.normal((C, D)) * 0.5; v = mx.random.normal((C, D)) * 0.5
        kk = mx.random.normal((C, D)); kk = kk / (mx.linalg.norm(kk, axis=-1, keepdims=True) + 1e-6)
        a = -kk; b = kk * 0.1
        w = mx.exp(-0.606531 * mx.sigmoid(mx.random.normal((C, D)))) if wl is None else mx.full((C, D), wl)
        do = mx.random.normal((C, D)); return r, w, k, v, a, b, do

    def rel(ref, got): return (mx.max(mx.abs(ref - got)) / (mx.max(mx.abs(ref)) + 1e-12)).item()

    def bb(x): return x[None, :, None, :]
    print(f"S3.4a-ii Metal single-chunk backward (C={C}, D={D}, fp32):")
    worst_h = worst_t = 0.0
    for wl in [None, 0.747, 0.545, 0.270]:
        for seed in seeds:
            r, w, k, v, a, b, do = mk(seed, wl)
            gM = chunk_bwd_s0_metal(r, w, k, v, a, b, do); mx.eval(*gM)
            gH = chunk_bwd_s0_mlx(r, w, k, v, a, b, do)
            def lT(r, w, k, v, a, b): return mx.sum(dplr_recurrence_mlx(bb(r), bb(w), bb(k), bb(v), bb(a), bb(b))[0, :, 0] * do)
            gT = mx.grad(lT, argnums=[0, 1, 2, 3, 4, 5])(r, w, k, v, a, b); mx.eval(*gH, *gT)
            eh = max(rel(gH[i], gM[i]) for i in range(6))
            et = max(rel(gT[i], gM[i]) for i in range(6))
            worst_h = max(worst_h, eh); worst_t = max(worst_t, et)
        lab = "model" if wl is None else f"w={wl:.3f}"
        print(f"  {lab:>9}: max rel vs hand-bwd={worst_h:.2e}  vs truth={worst_t:.2e}")
    print("PASS" if max(worst_h, worst_t) < 1e-4 else "FAIL")


if __name__ == "__main__":
    main()
    print()
    main_s32()
    print()
    main_s33a()
    print()
    main_s33b()
    print()
    main_s33c()
    print()
    main_s34a()
