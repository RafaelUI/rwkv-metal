"""bench_fast.py — ХОЛОДНЫЙ замер: saved vs fast vs battle. Запускать на холодном чипе /
с вентилятором / под Instruments (xctrace --template 'Metal System Trace').
Один конфиг на процесс (избегаем термал-дрейфа). Тяжёлый прогрев, min-of-many."""
import os, sys, time, argparse
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal/experiments"))
import importlib; battle = importlib.import_module("rwkv_metal.kernel.wkv7")
from s3_dplr_kernel import dplr_bwd_metal_bh_saved, dplr_bwd_metal_bh_fast
from verify_s34b import mk
ap = argparse.ArgumentParser(); ap.add_argument("--BH", type=int, default=32)
ap.add_argument("--T", type=int, default=64); ap.add_argument("--it", type=int, default=80)
ap.add_argument("--wu", type=int, default=30); a = ap.parse_args()
BH, T, C, D = a.BH, a.T, 16, 64; N = T // C
heads = [mk(h, T, D, None) for h in range(BH)]
R, W, K, V, A, Bt = (mx.stack([h[i] for h in heads]) for i in range(6))
do = mx.stack([h[6] for h in heads]); mx.eval(R, W, K, V, A, Bt, do)
b4 = lambda x: x.reshape(BH, T, 1, D)
r, w, k, v, aa, bb, d_ = map(b4, (R, W, K, V, A, Bt, do))
vg = mx.value_and_grad(lambda r, w, k, v, aa, bb: mx.sum(battle.wkv7_train(r, w, k, v, aa, bb) * d_),
                       argnums=[0, 1, 2, 3, 4, 5])
def tt(fn):
    for _ in range(a.wu): mx.eval(*fn() if isinstance(fn(), tuple) else [fn()])
    ts = []
    for _ in range(a.it):
        t0 = time.perf_counter(); r_ = fn(); mx.eval(*r_) if isinstance(r_, tuple) else mx.eval(r_)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort(); return ts[2]
ts = tt(lambda: dplr_bwd_metal_bh_saved(R, W, K, V, A, Bt, do, C))
tf = tt(lambda: dplr_bwd_metal_bh_fast(R, W, K, V, A, Bt, do, C))
tb = tt(lambda: vg(r, w, k, v, aa, bb))
print(f"BH={BH} T={T} N={N}: saved={ts:.2f}  fast={tf:.2f} ({ts/tf:.2f}x vs saved)  "
      f"battle={tb:.2f}  fast/battle={tb/tf:.2f}x")
