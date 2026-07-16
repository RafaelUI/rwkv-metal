"""Гипотеза occupancy: на occupancy-bound en25m память, освобождённая fused CE,
пускает БОЛЬШИЙ реальный forward-батч → выше загрузка GPU → выше tok/s.
(grad-accum даёт эффективный батч, но НЕ поднимает occupancy — микро-батч мал.)

Шаг ИДЕНТИЧЕН реальному тренеру (_make_step_simple): mx.compile(step,
inputs=state, outputs=state). Одинаковый seed ⇒ модели сравнимы, и
loss naive@B == fused@B служит санити-проверкой эквивалентности CE.

Запуск (из корня репо):
  GRAD_CKPT=0 BATCHES="8,16,24,32,48,64" ITERS=3 MEM_STOP=14 \
      ./.venv/bin/python experiments/bench_occupancy.py
"""
import os, sys, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx
import mlx.optimizers as optim
import mlx.nn as nn
from fused_ce import make_fused_ce
import rwkv_metal as rk
from rwkv_metal.model import RWKV7X070

T         = 512
VOCAB     = int(os.environ.get("VOCAB", 32000))
CHUNK     = int(os.environ.get("CHUNK", 2048))
ITERS     = int(os.environ.get("ITERS", 3))
MEM_STOP  = float(os.environ.get("MEM_STOP", 14.0))
GRAD_CKPT = bool(int(os.environ.get("GRAD_CKPT", "0")))
BATCHES   = [int(b) for b in os.environ.get("BATCHES", "8,16,24,32,48,64").split(",")]
GB = 1 / (1024 ** 3)
_fused = make_fused_ce(chunk_size=CHUNK)


def build(seed=0):
    mx.random.seed(seed)
    cfg = rk.preset("25m", vocab_size=VOCAB)
    model = RWKV7X070(cfg).set_dtype("bfloat16")
    model._grad_ckpt = GRAD_CKPT
    opt = optim.AdamW(learning_rate=1.5e-3)
    return model, opt


def make_step(model, opt, use_fused):
    def loss_fn(x, y):
        if use_fused:
            H = model.body(x); B, t, D = H.shape
            return _fused(H.reshape(B * t, D), model.head.weight,
                          y.reshape(B * t)).astype(mx.float32)
        return model.loss(x, y).astype(mx.float32)
    lvg = nn.value_and_grad(model, loss_fn)        # Module-aware (mx.value_and_grad по Module падает)
    state = [model.state, opt.state]

    def _step(x, y):
        l, g = lvg(x, y)
        g, _ = optim.clip_grad_norm(g, max_norm=1.0)
        opt.update(model, g)
        return l

    return mx.compile(_step, inputs=state, outputs=state)


def sweep(use_fused):
    label = f"{'fused' if use_fused else 'naive'} CE | ckpt {'ON' if GRAD_CKPT else 'OFF'}"
    print(f"\n══ {label} ═══════════════════════════════════")
    best = (0, 0.0)
    for B in BATCHES:
        try:
            model, opt = build(seed=0)
            step = make_step(model, opt, use_fused)
            mx.random.seed(B)
            x = mx.random.randint(0, VOCAB, (B, T))
            y = mx.random.randint(0, VOCAB, (B, T))
            mx.eval(x, y)
            for _ in range(2):                       # прогрев + компиляция
                l = step(x, y); mx.eval(l, model.state, opt.state)
            mx.clear_cache(); mx.reset_peak_memory()
            t0 = time.perf_counter()
            for _ in range(ITERS):
                l = step(x, y); mx.eval(l, model.state, opt.state)
            dt = (time.perf_counter() - t0) / ITERS
            peak = mx.get_peak_memory() * GB
            toks = B * T / dt
            swap = peak > MEM_STOP
            if not swap and toks > best[1]:
                best = (B, toks)
            print(f"  batch {B:4d} | {dt*1e3:8.1f} ms/it | peak {peak:5.2f} GB "
                  f"| {toks:8.0f} tok/s | loss {float(l):.3f}"
                  f"{'  ⚠ своп — не в зачёт' if swap else ''}")
            del model, opt, step; gc.collect(); mx.clear_cache()
            if swap:
                print("  -> порог памяти, стоп свипа"); break
        except Exception as ex:
            print(f"  batch {B:4d} | OOM/err: {type(ex).__name__}: {str(ex)[:50]}")
            break
    print(f"  ЛУЧШЕЕ (без свопа): batch {best[0]} -> {best[1]:.0f} tok/s")
    return best


if __name__ == "__main__":
    print(f"RWKV-7 25m | vocab={VOCAB} ctx={T} | compiled | chunk={CHUNK} "
          f"| ckpt={'ON' if GRAD_CKPT else 'OFF'} | iters={ITERS} | стоп>{MEM_STOP}GB")
    a = sweep(use_fused=False)
    b = sweep(use_fused=True)
    print("\n══ ИТОГ ═══════════════════════════════════")
    print(f"  naive:  batch {a[0]:4d} -> {a[1]:8.0f} tok/s")
    print(f"  fused:  batch {b[0]:4d} -> {b[1]:8.0f} tok/s")
    if a[1] > 0 and b[1] > 0:
        r = b[1] / a[1]
        print(f"  -> fused: {('%.2f× быстрее' % r) if r>=1 else ('%.2f× МЕДЛЕННЕЕ' % (1/r))} "
              f"по пиковому throughput (за счёт большего forward-батча)")
