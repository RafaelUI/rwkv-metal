"""Решающий end-to-end тест: реальный шаг ВСЕЙ модели (32M), а не изолированный CE.

  A) baseline  — наивный CE (model.loss) + grad_checkpoint ON  (твой текущий режим)
  B) fused     — body + fused_ce, grad_checkpoint OFF

Вопрос: какой пиковый tok/s достижим в бюджете 16 ГБ. Память от fused CE
освобождается -> больше batch -> крупнее матмулы тела -> выше occupancy GPU.

Шаг = nn.value_and_grad + clip + AdamW.update (как в трейнере), bf16.
Свип останавливается при peak > MEM_STOP ГБ (своп) или OOM.

Запуск (из корня репо):
    ./.venv/bin/python experiments/bench_step.py
Опц.:
    CHUNK=2048 MEM_STOP=14.0 ITERS=5 BATCHES="36,48,64,96,128" \
        ./.venv/bin/python experiments/bench_step.py
"""
import os, sys, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from fused_ce import make_fused_ce
import rwkv_metal as rk
from rwkv_metal.model import RWKV7X070

T        = 512
VOCAB    = int(os.environ.get("VOCAB", 32000))
CHUNK    = int(os.environ.get("CHUNK", 2048))
ITERS    = int(os.environ.get("ITERS", 5))
MEM_STOP = float(os.environ.get("MEM_STOP", 11.0))
BATCHES  = [int(b) for b in os.environ.get(
              "BATCHES", "16,24,36").split(",")]
GRAD_CKPT = bool(int(os.environ.get("GRAD_CKPT", "0")))  # общий для A и B
GB = 1 / (1024 ** 3)


def build():
    cfg = rk.preset("25m", vocab_size=VOCAB)
    model = RWKV7X070(cfg).set_dtype("bfloat16")
    opt = optim.AdamW(learning_rate=1.5e-3)
    return model, opt


def run_variant(name, use_fused):
    print(f"\n══ {name} ════════════════════════════════════════")
    model, opt = build()
    model._grad_ckpt = GRAD_CKPT                   # одинаково для A и B
    fused = make_fused_ce(chunk_size=CHUNK)

    def loss_fn(x, y):
        if use_fused:
            H = model.body(x)
            B, t, D = H.shape
            return fused(H.reshape(B * t, D), model.head.weight,
                         y.reshape(B * t)).astype(mx.float32)
        return model.loss(x, y).astype(mx.float32)

    lvg = nn.value_and_grad(model, loss_fn)

    def step(x, y):
        l, g = lvg(x, y)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(model, g)
        return l

    best = (0, 0.0)
    for B in BATCHES:
        try:
            mx.random.seed(0)
            x = mx.random.randint(0, VOCAB, (B, T))
            y = mx.random.randint(0, VOCAB, (B, T))
            mx.eval(x, y)
            for _ in range(2):                       # warmup
                l = step(x, y); mx.eval(l, model.parameters(), opt.state)
            mx.clear_cache(); mx.reset_peak_memory()
            t0 = time.perf_counter()
            for _ in range(ITERS):
                l = step(x, y); mx.eval(l, model.parameters(), opt.state)
            dt = (time.perf_counter() - t0) / ITERS
            peak = mx.get_peak_memory() * GB
            toks = B * T / dt
            if toks > best[1]:
                best = (B, toks)
            flag = "  ⚠ своп-зона" if peak > MEM_STOP else ""
            print(f"  batch {B:4d} | {dt*1e3:8.1f} ms/it | peak {peak:5.2f} GB "
                  f"| {toks:8.0f} tok/s | loss {float(l):.3f}{flag}")
            if peak > MEM_STOP:
                print("  -> порог памяти, стоп свипа")
                break
        except Exception as ex:
            print(f"  batch {B:4d} | OOM/err: {type(ex).__name__}: {str(ex)[:60]}")
            break
        gc.collect(); mx.clear_cache()
    print(f"  ЛУЧШЕЕ: batch {best[0]} -> {best[1]:.0f} tok/s")
    return best


if __name__ == "__main__":
    print(f"RWKV-7 25m | vocab={VOCAB} ctx={T} | chunk={CHUNK} "
          f"| iters={ITERS} | стоп при peak>{MEM_STOP}GB")
    a = run_variant("A: naive CE + grad_ckpt ON  (текущий)", use_fused=False)
    b = run_variant("B: fused CE + grad_ckpt OFF", use_fused=True)
    print("\n══ ИТОГ ══════════════════════════════════════════")
    print(f"  A (текущий):  batch {a[0]:4d} -> {a[1]:8.0f} tok/s")
    print(f"  B (fused):    batch {b[0]:4d} -> {b[1]:8.0f} tok/s")
    if a[1] > 0 and b[1] > 0:
        r = b[1] / a[1]
        verdict = f"{r:.2f}× быстрее" if r >= 1 else f"{1/r:.2f}× МЕДЛЕННЕЕ"
        print(f"  -> fused даёт {verdict} по пиковому throughput")
        days = lambda tps: 3e9 / tps / 3600 / 24
        print(f"  -> 3B токенов:  A ≈ {days(a[1]):.1f} сут   B ≈ {days(b[1]):.1f} сут")
