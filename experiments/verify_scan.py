"""verify_scan.py — Фаза 2: dS_scan_metal бит-близко vs _dS_carry_seq (throwaway producer)."""
import os, sys
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s3_dplr_kernel import dplr_forward_metal_save, _dS_carry_seq, dS_scan_metal
from verify_s34b import rel, mk

for BH, N in [(8, 2), (8, 4), (32, 4), (8, 16)]:
    T = N * 16
    heads = [mk(h, T, 64, None) for h in range(BH)]
    R, W, K, V, A, B = (mx.stack([h[i] for h in heads]) for i in range(6))
    do = mx.stack([h[6] for h in heads])
    _, cache = dplr_forward_metal_save(R, W, K, V, A, B, 16)
    ref = _dS_carry_seq(R, W, K, V, A, B, do, 16, cache)
    got = dS_scan_metal(R, W, K, V, A, B, do, 16, cache)
    mx.eval(ref, got)
    e = rel(ref, got)
    print(f"BH={BH:>3} N={N:>2}: rel(scan vs seq)={e:.2e}  " + ("OK" if e < 1e-5 else "FAIL"))
