"""bench_trace.py — рабочая нагрузка под Instruments (Metal System Trace).
Гоняет ОДИН пайплайн фиксированное число раз с eval-барьером на итерацию (каждая
итерация = чёткий GPU-burst в таймлайне). Выбор пайплайна и конфига — аргументами.
  --which fast|saved|battle   --BH 32 --T 64 --iters 30
В трейсе: зумишь одну установившуюся итерацию, смотришь длительности ядер
(dplr_step_save / prescan / dscan / kb / bwd2bh) и зазоры (idle между диспатчами)."""
import os, sys, time, argparse
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal/experiments"))
from s3_dplr_kernel import dplr_bwd_metal_bh_saved, dplr_bwd_metal_bh_fast
from verify_s34b import mk
ap = argparse.ArgumentParser()
ap.add_argument("--which", default="fast", choices=["fast", "saved", "battle"])
ap.add_argument("--BH", type=int, default=32); ap.add_argument("--T", type=int, default=64)
ap.add_argument("--iters", type=int, default=30); ap.add_argument("--warmup", type=int, default=3)
a = ap.parse_args()
BH, T, C, D = a.BH, a.T, 16, 64; N = T // C
heads = [mk(h, T, D, None) for h in range(BH)]
R, W, K, V, A, Bt = (mx.stack([h[i] for h in heads]) for i in range(6))
do = mx.stack([h[6] for h in heads]); mx.eval(R, W, K, V, A, Bt, do)
if a.which == "battle":
    import importlib; battle = importlib.import_module("rwkv_metal.kernel.wkv7")
    b4 = lambda x: x.reshape(BH, T, 1, D)
    r, w, k, v, aa, bb, d_ = map(b4, (R, W, K, V, A, Bt, do))
    vg = mx.value_and_grad(lambda *x: mx.sum(battle.wkv7_train(*x) * d_), argnums=list(range(6)))
    run = lambda: vg(r, w, k, v, aa, bb)
elif a.which == "saved":
    run = lambda: dplr_bwd_metal_bh_saved(R, W, K, V, A, Bt, do, C)
else:
    run = lambda: dplr_bwd_metal_bh_fast(R, W, K, V, A, Bt, do, C)
print(f"trace: which={a.which} BH={BH} T={T} N={N} iters={a.iters}", flush=True)
for _ in range(a.warmup):
    r_ = run(); mx.eval(*r_) if isinstance(r_, tuple) else mx.eval(r_)
t0 = time.perf_counter()
for _ in range(a.iters):
    r_ = run(); mx.eval(*r_) if isinstance(r_, tuple) else mx.eval(r_)
dt = (time.perf_counter() - t0) / a.iters * 1e3
print(f"avg {dt:.3f} ms/iter (wall, для сверки с трейсом)", flush=True)
