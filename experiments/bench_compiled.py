"""Компилируемый end-to-end бенч (повторяет паттерн трейнера: mx.compile со
state donation). Меряет БОЕВУЮ скорость и память.

Матрица: голова × checkpointing, в каждой ячейке свип по batch до порога памяти.
  A) naive CE + grad_ckpt ON   — текущий режим
  C) naive CE + grad_ckpt OFF  — самый быстрый на токен; вопрос — влезет ли
  B) fused CE + grad_ckpt OFF  — страховка по памяти, если C не влезает

Вывод: какой конфиг даёт максимальный tok/s в бюджете 16 ГБ, без потери качества
(все три математически эквивалентны по лоссу — это только память/скорость).

Запуск (из корня репо, ЛУЧШЕ С ЗАКРЫТЫМ ПРИЛОЖЕНИЕМ):
    ./.venv/bin/python experiments/bench_compiled.py
Опц.: CHUNK=2048 MEM_STOP=14.0 ITERS=8 BATCHES="16,24,36,48,64"
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
ITERS    = int(os.environ.get("ITERS", 8))
MEM_STOP = float(os.environ.get("MEM_STOP", 14.0))
BATCHES  = [int(b) for b in os.environ.get(
              "BATCHES", "16,24,36,48,64,96").split(",")]
GB = 1 / (1024 ** 3)


def run(name, use_fused, use_ckpt):
    print(f"\n══ {name} ════════════════════════════════")
    cfg = rk.preset("25m", vocab_size=VOCAB)
    model = RWKV7X070(cfg).set_dtype("bfloat16")
    model._grad_ckpt = use_ckpt
    opt = optim.AdamW(learning_rate=1.5e-3)
    fused = make_fused_ce(chunk_size=CHUNK)

    def loss_fn(x, y):
        if use_fused:
            H = model.body(x)
            B, t, D = H.shape
            return fused(H.reshape(B * t, D), model.head.weight,
                         y.reshape(B * t)).astype(mx.float32)
        return model.loss(x, y).astype(mx.float32)

    lvg = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    def _step(x, y):
        l, g = lvg(x, y)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(model, g)
        return l

    step = mx.compile(_step, inputs=state, outputs=state)   # как в трейнере

    best = (0, 0.0)
    for B in BATCHES:
        try:
            mx.random.seed(0)
            x = mx.random.randint(0, VOCAB, (B, T))
            y = mx.random.randint(0, VOCAB, (B, T))
            mx.eval(x, y)
            for _ in range(3):                               # warmup + compile
                l = step(x, y); mx.eval(l, state)
            mx.clear_cache(); mx.reset_peak_memory()
            t0 = time.perf_counter()
            for _ in range(ITERS):
                l = step(x, y); mx.eval(l, state)
            dt = (time.perf_counter() - t0) / ITERS
            peak = mx.get_peak_memory() * GB
            toks = B * T / dt
            if toks > best[1]:
                best = (B, toks)
            flag = "  ⚠ своп" if peak > MEM_STOP else ""
            print(f"  batch {B:4d} | {dt*1e3:7.1f} ms/it | peak {peak:5.2f} GB "
                  f"| {toks:8.0f} tok/s{flag}")
            if peak > MEM_STOP:
                print("  -> порог памяти, стоп"); break
        except Exception as ex:
            print(f"  batch {B:4d} | OOM/err: {type(ex).__name__}: {str(ex)[:55]}")
            break
        gc.collect(); mx.clear_cache()
    print(f"  ЛУЧШЕЕ: batch {best[0]} -> {best[1]:.0f} tok/s")
    return best


if __name__ == "__main__":
    print(f"COMPILED | RWKV-7 25m | vocab={VOCAB} ctx={T} chunk={CHUNK} "
          f"iters={ITERS} | стоп peak>{MEM_STOP}GB")
    res = {}
    res['A'] = run("A: naive CE + grad_ckpt ON  (текущий)", False, True)
    res['C'] = run("C: naive CE + grad_ckpt OFF",           False, False)
    res['B'] = run("B: fused CE + grad_ckpt OFF",           True,  False)
    print("\n══ ИТОГ (пиковый tok/s в бюджете) ══════════════")
    for k in ['A', 'C', 'B']:
        b, t = res[k]
        d = 3e9 / t / 86400 if t else 0
        print(f"  {k}: batch {b:4d} -> {t:8.0f} tok/s   (3B ≈ {d:4.1f} сут)")
    base = res['A'][1]
    if base:
        for k in ['C', 'B']:
            if res[k][1]:
                r = res[k][1] / base
                print(f"  {k} vs A: {r:.2f}× {'быстрее' if r>=1 else 'МЕДЛЕННЕЕ'}")
