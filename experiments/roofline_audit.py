"""
Per-op roofline audit at the REAL training shapes.

Context: full step for 12L x 448d, batch 8, ctx 512, bf16 = ~1176 ms, of which
WKV fwd+bwd is only ~32% and optimizing WKV 1.24x changed the step by 0%.
So the wall is in the other ~70%: projections, token-shift/lerp chains,
low-rank branches, cmix. Classic tuning hasn't moved it, so instead of
guessing, measure every primitive against its own roofline and look for the
one that is anomalously far off.

For each op we report achieved TFLOPS (compute-bound ops) or GB/s
(bandwidth-bound ops). An op sitting at a small fraction of what the same
hardware delivers on a *larger* version of the same op is the clue.

Key hypothesis to test: the r/k/v/o projections are four separate
[4096,448] x [448,448] GEMMs per layer. That K=448, N=448 shape may be too
small to saturate; fusing r/k/v into one [448,1344] GEMM would triple the
work per launch at identical FLOPs. If the fused GEMM's TFLOPS is much
higher, that's a real, non-obvious win worth ~3 GEMM launches per layer.

Usage:
    python experiments/roofline_audit.py
"""
import time

import mlx.core as mx

B, T, D = 8, 512, 448
H, S = 7, 64
DT = mx.bfloat16
ITEM = 2  # bytes per bf16 element


def sync():
    mx.eval(mx.array(0.0))


def timed(fn, n=20, warm=5):
    for _ in range(warm):
        r = fn(); mx.eval(r)
    t0 = time.perf_counter()
    for _ in range(n):
        r = fn(); mx.eval(r)
    return (time.perf_counter() - t0) / n


def gemm(M, K, N, label, n=20):
    a = mx.random.normal((M, K)).astype(DT)
    b = mx.random.normal((K, N)).astype(DT)
    mx.eval(a, b)
    dt = timed(lambda: a @ b, n=n)
    flops = 2 * M * K * N
    tflops = flops / dt / 1e12
    print(f"  {label:<34} {M:>6}x{K:<5}x{N:<6} {dt*1000:>8.2f} ms {tflops:>8.2f} TFLOPS")
    return tflops


def elemwise(fn, bytes_moved, label, n=30):
    dt = timed(fn, n=n)
    gbs = bytes_moved / dt / 1e9
    print(f"  {label:<34} {'':>19} {dt*1000:>8.2f} ms {gbs:>8.1f} GB/s")
    return gbs


def main():
    print(f"shapes: B={B} T={T} D={D} (tokens={B*T}), dtype=bf16\n")

    print("=== GEMMs: does a bigger N help? (projection fusion hypothesis) ===")
    print(f"  {'op':<34} {'M x K x N':<19} {'time':>11} {'throughput':>15}")
    M = B * T
    t1 = gemm(M, D, D, "single proj (r or k or v or o)")
    t3 = gemm(M, D, 3 * D, "FUSED r+k+v (one GEMM)")
    t4 = gemm(M, D, 4 * D, "FUSED r+k+v+o")
    gemm(M, D, 4 * D, "cmix key (448->1792)")
    gemm(M, 4 * D, D, "cmix value (1792->448)")
    gemm(M, D, 64, "low-rank A (448->64)")
    gemm(M, 64, D, "low-rank B (64->448)")
    print()
    gemm(M, D, D, "  same shape, fp32 check", n=10) if False else None

    print("=== bigger M: is M=4096 enough to saturate? ===")
    for m in (1024, 4096, 16384, 65536):
        gemm(m, D, D, f"proj at M={m}")
    print()

    print("=== elementwise / memory ops at [B,T,D] ===")
    print(f"  {'op':<34} {'':<19} {'time':>11} {'throughput':>15}")
    x = mx.random.normal((B, T, D)).astype(DT)
    xx = mx.random.normal((B, T, D)).astype(DT)
    p = mx.random.normal((1, 1, D)).astype(DT)
    mx.eval(x, xx, p)
    nbytes = B * T * D * ITEM

    elemwise(lambda: x + xx, 3 * nbytes, "add (2 read + 1 write)")
    elemwise(lambda: x + xx * p, 3 * nbytes, "lerp: x + xx*p")
    elemwise(lambda: mx.concatenate([x[:, :1], x[:, :-1]], axis=1),
             2 * nbytes, "token_shift (concatenate)")
    elemwise(lambda: mx.concatenate([x[:, :1], x[:, :-1]], axis=1) - x,
             3 * nbytes, "token_shift - x")
    elemwise(lambda: mx.sigmoid(x), 2 * nbytes, "sigmoid")
    elemwise(lambda: mx.tanh(x), 2 * nbytes, "tanh")
    elemwise(lambda: x.reshape(B, T, H, S), nbytes, "reshape to [B,T,H,S]")
    elemwise(lambda: x.astype(mx.float32), nbytes + 2 * nbytes, "bf16 -> fp32 cast")
    print()

    print("=== the whole tmix prologue, op by op ===")
    def prologue():
        sh = mx.concatenate([x[:, :1], x[:, :-1]], axis=1) - x
        return (x + sh * p, x + sh * p, x + sh * p,
                x + sh * p, x + sh * p, x + sh * p)
    dt = timed(prologue, n=20)
    # 1 concat(2) + 1 sub(3) + 6 lerps(3 each) = 23 tensor passes
    print(f"  token_shift + 6 lerps: {dt*1000:.2f} ms  "
          f"-> {23*nbytes/dt/1e9:.1f} GB/s effective, x12 layers = {dt*12*1000:.0f} ms")

    def prologue_compiled():
        sh = mx.concatenate([x[:, :1], x[:, :-1]], axis=1) - x
        return (x + sh * p, x + sh * p, x + sh * p,
                x + sh * p, x + sh * p, x + sh * p)
    cf = mx.compile(prologue_compiled)
    dt_c = timed(cf, n=20)
    print(f"  same under mx.compile:  {dt_c*1000:.2f} ms  "
          f"({dt/dt_c:.2f}x), x12 layers = {dt_c*12*1000:.0f} ms")
    print()

    print("=== summary ===")
    print(f"  single 448x448 proj : {t1:.2f} TFLOPS")
    print(f"  fused r+k+v (x3 N)  : {t3:.2f} TFLOPS  -> {t3/max(t1,1e-9):.2f}x better utilization")
    print(f"  fused r+k+v+o (x4 N): {t4:.2f} TFLOPS  -> {t4/max(t1,1e-9):.2f}x")


if __name__ == "__main__":
    main()
