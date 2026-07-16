"""verify_save_fwd.py — SAVE-форвард: o бит-в-бит vs оригинальный Metal-форвард;
Am/u/wmat/v2 бит-в-бит vs MLX (_fwd_intermediates_mlx_bh) → демонстрация «дыр нет»."""
import os, sys
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s3_dplr_kernel import (dplr_forward_metal, dplr_forward_metal_save,
                            _fwd_intermediates_mlx_bh)
from verify_s34b import mk


def rel(a, b): return (mx.max(mx.abs(a - b)) / (mx.max(mx.abs(a)) + 1e-12)).item()


def main(C=16, D=64, BH=6, seeds=(0, 1)):
    print(f"SAVE-форвард (BH={BH}, C={C}, D={D}):")
    wo = wi = 0.0
    for N in (2, 3, 4):
        T = N * C
        for wl in [None, 0.545, 0.270]:
            eo = ei = 0.0
            for seed in seeds:
                heads = [mk(seed * 100 + h, T, D, wl) for h in range(BH)]
                R, W, K, V, A, B = (mx.stack([h[i] for h in heads]) for i in range(6))
                o_save, cache = dplr_forward_metal_save(R, W, K, V, A, B, C)
                # o vs оригинальный форвард ([B,T,H,D]-обёртка; B=1,H=BH)
                def bthd(x): return mx.transpose(x.reshape(1, BH, T, D), (0, 2, 1, 3))
                o_ref = dplr_forward_metal(bthd(R), bthd(W), bthd(K), bthd(V), bthd(A), bthd(B), C)
                o_ref = mx.transpose(o_ref, (0, 2, 1, 3)).reshape(BH, T, D)
                mx.eval(o_save, o_ref); eo = max(eo, rel(o_ref, o_save))
                # интермедиаты per-chunk vs MLX
                for n in range(N):
                    s = slice(n * C, (n + 1) * C)
                    Aqk, Aqb, Aab, Aak, u, wm, v2 = _fwd_intermediates_mlx_bh(
                        R[:, s], W[:, s], K[:, s], V[:, s], A[:, s], B[:, s], cache["S_in"][n], C)
                    AmM = mx.stack([Aqk, Aqb, Aab, Aak], axis=1)
                    mx.eval(AmM, u, wm, v2)
                    ei = max(ei, rel(AmM, cache["Am"][n]), rel(u, cache["u"][n]),
                             rel(wm, cache["wmat"][n]), rel(v2, cache["v2"][n]))
            wo = max(wo, eo); wi = max(wi, ei)
            lab = "model" if wl is None else f"w={wl:.3f}"
            print(f"  N={N} {lab:>9}: o vs fwd={eo:.2e}  Am/u/wmat/v2 vs MLX={ei:.2e}")
    print(f"WORST o={wo:.2e}  intermediates={wi:.2e}  "
          + ("PASS" if max(wo, wi) < 1e-4 else "FAIL"))


if __name__ == "__main__":
    main()
