"""
bench_dplr.py — S2: скорость/память нового MLX-чанка vs боевое ядро.
Конфиг handoff: B=20, T=512, H=4 (n_embd256/head64), D=64, fp32.
Логирует (приоритет): tok/s -> пик памяти -> grad-стабильность.
Memory-safe: лимит + детач (nohup) снаружи. Один WKV-слой (в 12L-шаге зовётся 12x).
"""
import os, sys, time, gc
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv_metal.kernel.wkv7_checkpoint import make_wkv7_checkpoint
from dplr_mlx import dplr_chunkwise_mlx

B, T, H, D = 20, 512, 4, 64
WARMUP, ITERS = 2, 5
GB = 1024**3


def inputs(w_level=0.9, seed=0):
    mx.random.seed(seed)
    r = mx.random.normal((B, T, H, D)) * 0.5
    k = mx.random.normal((B, T, H, D)) * 0.5
    v = mx.random.normal((B, T, H, D)) * 0.5
    kk = mx.random.normal((B, T, H, D))
    kk = kk / (mx.linalg.norm(kk, axis=-1, keepdims=True) + 1e-6)
    a, b = -kk, kk * 0.1
    w = mx.full((B, T, H, D), w_level)
    dy = mx.random.normal((B, T, H, D))
    return r, w, k, v, a, b, dy


def time_fwd(fn, args, **kw):
    r, w, k, v, a, b, dy = args
    for _ in range(WARMUP):
        mx.eval(fn(r, w, k, v, a, b, **kw))
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        mx.eval(fn(r, w, k, v, a, b, **kw))
    dt = (time.perf_counter() - t0) / ITERS
    return dt, mx.get_peak_memory() / GB


def time_fwdbwd(fn, args, **kw):
    r, w, k, v, a, b, dy = args
    def loss(r, w, k, v, a, b):
        return mx.sum(fn(r, w, k, v, a, b, **kw) * dy)
    gfn = mx.grad(loss, argnums=[0, 1, 2, 3, 4, 5])
    for _ in range(WARMUP):
        mx.eval(gfn(r, w, k, v, a, b))
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        mx.eval(gfn(r, w, k, v, a, b))
    dt = (time.perf_counter() - t0) / ITERS
    return dt, mx.get_peak_memory() / GB


def row(label, dt, peak):
    toks = B * T / dt
    print(f"  {label:28s} {dt*1e3:9.2f} ms  {toks:10.0f} tok/s  {peak:7.2f} GB", flush=True)


def main():
    print(f"=== S2 bench  B={B} T={T} H={H} D={D} fp32  (warmup{WARMUP} iters{ITERS}) ===", flush=True)
    battle = make_wkv7_checkpoint(B=B, T=T, H=H, D=D)
    args = inputs()

    print("--- tok/s + peak memory ---", flush=True)
    try:
        dt, pk = time_fwd(battle, args);                 row("battle  fwd", dt, pk)
    except Exception as e: print("  battle fwd ERR", repr(e)[:140], flush=True)
    try:
        dt, pk = time_fwdbwd(battle, args);              row("battle  fwd+bwd", dt, pk)
    except Exception as e: print("  battle fwd+bwd ERR", repr(e)[:140], flush=True)
    for C in (16, 32):
        try:
            dt, pk = time_fwd(dplr_chunkwise_mlx, args, chunk_size=C);    row(f"chunk{C} fwd", dt, pk)
        except Exception as e: print(f"  chunk{C} fwd ERR", repr(e)[:140], flush=True)
        try:
            dt, pk = time_fwdbwd(dplr_chunkwise_mlx, args, chunk_size=C); row(f"chunk{C} fwd+bwd", dt, pk)
        except Exception as e: print(f"  chunk{C} fwd+bwd ERR", repr(e)[:140], flush=True)

    print("--- grad stability @ production shape (norms; finite?) ---", flush=True)
    for wl in (0.9, 0.5, 0.27):
        r, w, k, v, a, b, dy = inputs(wl)
        def gnorm(fn, **kw):
            def loss(r, w, k, v, a, b): return mx.sum(fn(r, w, k, v, a, b, **kw) * dy)
            g = mx.grad(loss, argnums=[0, 1, 4])(r, w, k, v, a, b)  # dr,dw,da
            mx.eval(g)
            fin = all(bool(mx.all(mx.isfinite(x)).item()) for x in g)
            nrm = [float(mx.linalg.norm(x).item()) for x in g]
            return fin, nrm
        try:
            bf, bn = gnorm(battle)
        except Exception as e:
            bf, bn = None, repr(e)[:60]
        try:
            cf, cn = gnorm(dplr_chunkwise_mlx, chunk_size=16)
        except Exception as e:
            cf, cn = None, repr(e)[:60]
        print(f"  w={wl:4.2f}  battle finite={bf} norms(dr,dw,da)={bn}", flush=True)
        print(f"           chunk16 finite={cf} norms(dr,dw,da)={cn}", flush=True)
    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
