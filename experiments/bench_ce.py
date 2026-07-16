"""Бенчмарк fused CE против наивного пути (как в model.loss сейчас).

Изолирует только кросс-энтропию: H = body(idx) эмулируется случайным
[N, D] тензором, W = head.weight случайным [V, D]. Меряем:
  1) корректность: лосс и градиенты (dH, dW) fused == naive (fp32, малый N);
  2) пик памяти naive vs fused (реальные формы, bf16);
  3) скорость forward+backward, ms/итер и эквивалент tok/s.

Запуск:  ./.venv/bin/python experiments/bench_ce.py
Опц.:    N=18432 V=32000 D=256 CHUNK=2048 ITERS=20 ./.venv/bin/python experiments/bench_ce.py
"""
import os, time, math
import mlx.core as mx
from fused_ce import naive_ce, make_fused_ce

D     = int(os.environ.get("D", 256))
V     = int(os.environ.get("V", 32000))
N     = int(os.environ.get("N", 18432))      # 36 * 512
CHUNK = int(os.environ.get("CHUNK", 2048))
ITERS = int(os.environ.get("ITERS", 20))
MB = 1 / (1024 ** 2)


def mem_reset():
    mx.clear_cache(); mx.reset_peak_memory()


def correctness():
    print("── 1. КОРРЕКТНОСТЬ (fp32, N=256) ─────────────────────────────")
    mx.random.seed(0)
    n = 256
    H = mx.random.normal((n, D)); W = mx.random.normal((V, D))
    t = mx.random.randint(0, V, (n,))
    fused = make_fused_ce(chunk_size=64)

    ln = naive_ce(H, W, t); lf = fused(H, W, t)
    mx.eval(ln, lf)
    dloss = abs(float(ln) - float(lf))

    gn = mx.value_and_grad(naive_ce, argnums=(0, 1))(H, W, t)[1]
    gf = mx.value_and_grad(fused,    argnums=(0, 1))(H, W, t)[1]
    mx.eval(gn, gf)
    dH_err = float(mx.abs(gn[0] - gf[0]).max())
    dW_err = float(mx.abs(gn[1] - gf[1]).max())

    print(f"  loss  naive={float(ln):.6f}  fused={float(lf):.6f}  |Δ|={dloss:.2e}")
    print(f"  max|Δ dH| = {dH_err:.2e}")
    print(f"  max|Δ dW| = {dW_err:.2e}")
    ok = dloss < 1e-4 and dH_err < 1e-4 and dW_err < 1e-4
    print(f"  -> {'PASS ✓' if ok else 'FAIL ✗'}")
    return ok


def bench_one(name, loss_callable, H, W, t):
    grad_fn = mx.value_and_grad(loss_callable, argnums=(0, 1))
    # warmup
    l, g = grad_fn(H, W, t); mx.eval(l, g)
    mem_reset()
    l, g = grad_fn(H, W, t); mx.eval(l, g, g[0], g[1])
    peak = mx.get_peak_memory() * MB
    # timing
    t0 = time.perf_counter()
    for _ in range(ITERS):
        l, g = grad_fn(H, W, t); mx.eval(l, g[0], g[1])
    dt = (time.perf_counter() - t0) / ITERS
    toks = N / dt
    print(f"  {name:<22} {dt*1e3:8.2f} ms/it   peak {peak:8.1f} MB   "
          f"{toks:8.0f} tok/s   loss {float(l):.4f}")
    return dt, peak


def perf():
    print(f"\n── 2. ПАМЯТЬ + СКОРОСТЬ (bf16, N={N} V={V} D={D} chunk={CHUNK}) ──")
    mx.random.seed(1)
    H = mx.random.normal((N, D)).astype(mx.bfloat16)
    W = mx.random.normal((V, D)).astype(mx.bfloat16)
    t = mx.random.randint(0, V, (N,))
    mx.eval(H, W, t)
    fused = make_fused_ce(chunk_size=CHUNK)

    print(f"  (полные логиты [N,V] в bf16 = {N*V*2*MB:.0f} MB — вот что ест naive)")
    dn = pn = None
    try:
        dn, pn = bench_one("naive (текущий)", naive_ce, H, W, t)
    except Exception as ex:
        print(f"  naive: OOM/err -> {type(ex).__name__}: {str(ex)[:60]}")
    df, pf = bench_one(f"fused (chunk={CHUNK})", fused, H, W, t)

    if dn:
        print(f"\n  Память:   {pn/pf:.1f}× меньше   ({pn:.0f} -> {pf:.0f} MB)")
        spd = dn / df
        tag = f"{spd:.2f}× быстрее" if spd >= 1 else f"{1/spd:.2f}× медленнее"
        print(f"  Скорость: {tag}   ({dn*1e3:.1f} -> {df*1e3:.1f} ms/it)")


if __name__ == "__main__":
    ok = correctness()
    perf()
    print("\nЗамечание: это изолированный CE. В реальном шаге сверху ещё forward/"
          "backward тела (~32M) — там память от fused CE освобождается под отказ "
          "от grad_checkpoint, что и есть основной выигрыш по скорости.")
