"""
bench_s33c.py — скорость/память Metal step-кернела forward (S3.3c) vs боевое ядро.
Шейп handoff: B=20,T=512,H=4,D=64,fp32. ОДИН WKV-слой (в 12L-шаге зовётся 12x).
ВНИМАНИЕ: step-кернел forward = драйвер с per-chunk dispatch + хост S-update +
  mx.eval на чанк (серийная S-зависимость). Это in-situ-честная цифра forward,
  НЕ изолированное ядро. У step-кернела backward ПОКА НЕТ (S3.4) → меряем только fwd.
  Для контекста: battle fwd и fwd+bwd, оракул chunk16 fwd.
"""
import os, sys, time
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv_metal.kernel.wkv7_checkpoint import make_wkv7_checkpoint
from dplr_mlx import dplr_chunkwise_mlx
from s3_dplr_kernel import dplr_forward_metal, chunk_step

B, T, H, D = 20, 512, 4, 64
WARMUP, ITERS = 2, 5
GB = 1024**3


def inputs(seed=0):
    mx.random.seed(seed)
    r = mx.random.normal((B, T, H, D)) * 0.5
    k = mx.random.normal((B, T, H, D)) * 0.5
    v = mx.random.normal((B, T, H, D)) * 0.5
    kk = mx.random.normal((B, T, H, D)); kk = kk / (mx.linalg.norm(kk, axis=-1, keepdims=True) + 1e-6)
    a, b = -kk, kk * 0.1
    w = mx.full((B, T, H, D), 0.9)
    dy = mx.random.normal((B, T, H, D))
    return r, w, k, v, a, b, dy


def time_fwd(fn, args, **kw):
    r, w, k, v, a, b, dy = args
    for _ in range(WARMUP): mx.eval(fn(r, w, k, v, a, b, **kw))
    mx.reset_peak_memory(); t0 = time.perf_counter()
    for _ in range(ITERS): mx.eval(fn(r, w, k, v, a, b, **kw))
    return (time.perf_counter() - t0) / ITERS, mx.get_peak_memory() / GB


def time_fwdbwd(fn, args, **kw):
    r, w, k, v, a, b, dy = args
    def loss(r, w, k, v, a, b): return mx.sum(fn(r, w, k, v, a, b, **kw) * dy)
    gfn = mx.grad(loss, argnums=[0, 1, 2, 3, 4, 5])
    for _ in range(WARMUP): mx.eval(gfn(r, w, k, v, a, b))
    mx.reset_peak_memory(); t0 = time.perf_counter()
    for _ in range(ITERS): mx.eval(gfn(r, w, k, v, a, b))
    return (time.perf_counter() - t0) / ITERS, mx.get_peak_memory() / GB


def time_isolated_chunk(C=16):
    """Изолированный ОДИН chunk_step для всех BH (без драйвер-цикла) — сырая стоимость ядра."""
    BH, N = B * H, T // C
    mx.random.seed(1)
    mk = lambda: mx.random.normal((BH, C, D)) * 0.5
    q, k, v = mk(), mk(), mk()
    al, be = -mk(), mk() * 0.1
    gk = -0.606531 * mx.sigmoid(mx.random.normal((BH, C, D))); gc = mx.cumsum(gk, axis=1)
    S = mx.zeros((BH, D, D))
    for _ in range(WARMUP): mx.eval(*chunk_step(q, k, v, al, be, gc, gk, S))
    mx.reset_peak_memory(); t0 = time.perf_counter()
    R = 20
    for _ in range(R): mx.eval(*chunk_step(q, k, v, al, be, gc, gk, S))
    dt = (time.perf_counter() - t0) / R
    # пропускная: один chunk_step покрывает BH*C токенов; полный слой = N таких
    return dt, mx.get_peak_memory() / GB, BH * C / dt, N


def row(label, dt, peak, note=""):
    print(f"  {label:30s} {dt*1e3:9.2f} ms  {B*T/dt:10.0f} tok/s  {peak:7.2f} GB  {note}", flush=True)


def main():
    print(f"=== S3.3c bench  B={B} T={T} H={H} D={D} fp32  (warmup{WARMUP} iters{ITERS}) ===", flush=True)
    args = inputs()
    battle = make_wkv7_checkpoint(B=B, T=T, H=H, D=D)
    print("--- forward: tok/s + peak (полный слой) ---", flush=True)
    dt, pk = time_fwd(battle, args);                         row("battle fwd (Metal bwd-capable)", dt, pk)
    dt, pk = time_fwd(dplr_chunkwise_mlx, args, chunk_size=16); row("MLX oracle chunk16 fwd", dt, pk)
    dt, pk = time_fwd(dplr_forward_metal, args, chunk_size=16); row("S3.3c step-kernel fwd (driver)", dt, pk, "per-chunk dispatch+eval")
    print("--- reference fwd+bwd (battle only; step-kernel bwd = S3.4) ---", flush=True)
    dt, pk = time_fwdbwd(battle, args);                      row("battle fwd+bwd", dt, pk)
    print("--- isolated single chunk_step (raw kernel, no driver loop) ---", flush=True)
    dt, pk, tps, N = time_isolated_chunk(16)
    print(f"  one chunk_step (BH={B*H},C=16): {dt*1e3:7.3f} ms  {tps:10.0f} tok/s(chunk)  {pk:.2f} GB  | x{N} chunks/layer", flush=True)
    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
