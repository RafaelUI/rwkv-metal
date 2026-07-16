"""verify_parallel.py — Фаза 1: parallel-KB бит-в-бит vs batched (послед. цикл) + дельта."""
import os, sys, time
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s3_dplr_kernel import dplr_bwd_metal_bh_batched, dplr_bwd_metal_bh_parallel
from verify_s34b import rel, mk

for BH, N in [(8, 4), (32, 4), (8, 32)]:
    T = N * 16
    heads = [mk(h, T, 64, None) for h in range(BH)]
    R, W, K, V, A, B = (mx.stack([h[i] for h in heads]) for i in range(6))
    do = mx.stack([h[6] for h in heads])
    gB = dplr_bwd_metal_bh_batched(R, W, K, V, A, B, do, 16)
    gP = dplr_bwd_metal_bh_parallel(R, W, K, V, A, B, do, 16)
    mx.eval(*gB, *gP)
    err = max(rel(gB[i], gP[i]) for i in range(6))
    print(f"BH={BH:>3} N={N:>2}: rel(parallel vs batched)={err:.2e}  "
          + ("OK" if err < 1e-5 else "FAIL"))
