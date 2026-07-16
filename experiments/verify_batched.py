"""verify_batched.py — батч-драйвер бит-близко к dplr_bwd_metal_bh_saved + дельта скорости."""
import os, sys, time
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s3_dplr_kernel import dplr_bwd_metal_bh_saved, dplr_bwd_metal_bh_batched
from verify_s34b import rel, mk


def main(C=16, D=64):
    for BH, N in [(8, 4), (32, 4), (128, 4)]:
        T = N * C
        heads = [mk(h, T, D, None) for h in range(BH)]
        R, W, K, V, A, B = (mx.stack([h[i] for h in heads]) for i in range(6))
        do = mx.stack([h[6] for h in heads])
        gS = dplr_bwd_metal_bh_saved(R, W, K, V, A, B, do, C)
        gB = dplr_bwd_metal_bh_batched(R, W, K, V, A, B, do, C)
        mx.eval(*gS, *gB)
        err = max(rel(gS[i], gB[i]) for i in range(6))
        def tt(fn, it=40, wu=10):
            for _ in range(wu): mx.eval(*fn())
            ts = []
            for _ in range(it):
                t0 = time.perf_counter(); mx.eval(*fn()); ts.append((time.perf_counter()-t0)*1e3)
            ts.sort(); return ts[len(ts)//2]
        ts = tt(lambda: dplr_bwd_metal_bh_saved(R, W, K, V, A, B, do, C))
        tb = tt(lambda: dplr_bwd_metal_bh_batched(R, W, K, V, A, B, do, C))
        print(f"BH={BH:>3}: rel(saved vs batched)={err:.2e}  "
              f"saved fwd+bwd={ts:6.2f}мс  batched={tb:6.2f}мс  ({ts/tb:.2f}x)")


if __name__ == "__main__":
    main()
