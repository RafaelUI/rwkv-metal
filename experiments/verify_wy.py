import os, sys
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s3_dplr_kernel import _wy_kernel, _fwd_intermediates_mlx_bh
from verify_s34b import rel, mk

C, D = 16, 64
for BH, N in [(8, 4), (32, 4)]:
    T = N * C
    heads = [mk(h, T, D, None) for h in range(BH)]
    R, W, K, V, A, B = (mx.stack([h[i] for h in heads]) for i in range(6))
    NB = BH * N
    f = lambda x: x.reshape(BH, N, C, D).reshape(NB, C, D)
    gk = mx.log(W); gc = mx.cumsum(gk.reshape(BH, N, C, D), axis=2)
    wy = _wy_kernel(C, D)
    Am, u, wmat, o_base, s_base = wy(
        inputs=[f(R), f(K), f(V), f(A), f(B), gc.reshape(NB, C, D), gk.reshape(NB, C, D)],
        grid=(32, NB, 1), threadgroup=(32, 1, 1),
        output_shapes=[(NB, 4, C, C), (NB, C, D), (NB, C, D), (NB, C, D), (NB, D, D)],
        output_dtypes=[mx.float32] * 5)
    # MLX-эталон (S_in=0: u,wmat не зависят от S; o_base=A_qk@v; Sbase=(k*dec)^T@v)
    Aqk, Aqb, Aab, Aak, uM, wmatM, _ = _fwd_intermediates_mlx_bh(
        R.reshape(NB, C, D), W.reshape(NB, C, D), K.reshape(NB, C, D), V.reshape(NB, C, D),
        A.reshape(NB, C, D), B.reshape(NB, C, D), mx.zeros((NB, D, D)), C)
    AmM = mx.stack([Aqk, Aqb, Aab, Aak], axis=1)
    gcl = gc.reshape(NB, C, D)[:, -1]
    dec = mx.exp(gcl[:, None, :] - gc.reshape(NB, C, D))
    obM = Aqk @ V.reshape(NB, C, D)
    sbM = mx.swapaxes(K.reshape(NB, C, D) * dec, 1, 2) @ V.reshape(NB, C, D)
    mx.eval(Am, u, wmat, o_base, s_base, AmM, uM, wmatM, obM, sbM)
    e = max(rel(AmM, Am), rel(uM, u), rel(wmatM, wmat), rel(obM, o_base), rel(sbM, s_base))
    print(f"BH={BH} N={N}: Am/u/wmat/o_base/Sbase vs MLX = {e:.2e}  " + ("OK" if e < 1e-4 else "FAIL"))
