"""A/B GRAD_CHECKPOINT НА ШАГЕ ОБУЧЕНИЯ QLoRA (22.08).

Арифметика по манифесту: шаг T=512 -- это 1.51 ТФЛОП на проход, при
чекпоинтинге проходов ЧЕТЫРЕ (fwd + пересчёт + dW + dX) = 6.05 ТФЛОП,
без него ТРИ = 4.54 ТФЛОП. Пол при 2.80 ТФЛОП/с -- 2160 против 1620 мс,
то есть чекпоинтинг стоит ~25% времени. Цена -- память: активации, которые
чекпоинтинг пересчитывает (записано ~1.3 ГБ транзиента). Этот бенч меряет
обе стороны размена в одном процессе, чередованием (закон 1), своп на
границе замера (закон 11).

    python tests/bench_ckpt_ab.py [T] [раундов]
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))

import mlx.core as mx                      # noqa: E402
import mlx.nn as nn                        # noqa: E402
import mlx.optimizers as optim             # noqa: E402
import numpy as np                         # noqa: E402

T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
PATH = os.environ.get("RWKVQ_BASE", "/tmp/reduction_new.rwkvq")


def swap_mb():
    out = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True,
                         env={**os.environ, "LC_ALL": "C"}).stdout
    p = out.replace("=", " ").split()
    return float(p[p.index("used") + 1].rstrip("M"))


def main():
    from rwkv_metal.lora import load_rwkvq_model

    model, cfg, info = load_rwkvq_model(PATH, rank=16, verbose=False)
    rs = np.random.RandomState(5)
    x = mx.array(rs.randint(1, 60000, size=(1, T)).astype(np.int32))
    y = mx.array(rs.randint(1, 60000, size=(1, T)).astype(np.int32))

    def loss_fn(m, a, b):
        return m.loss(a, b).astype(mx.float32)

    gf = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=1e-4)

    def step():
        loss, grads = gf(model, x, y)
        opt.update(model, grads)
        mx.eval(loss, model.state, opt.state)

    VAR = [("ckpt=True", True), ("ckpt=False", False)]
    acc = {n: [] for n, _ in VAR}
    pk = {n: 0.0 for n, _ in VAR}
    sw0 = swap_mb()
    # ПОРЯДОК ПЛЕЧ РАНДОМИЗИРОВАН (закон 24): фиксированный порядок на
    # безвентиляторной машине -- систематический сдвиг в пользу первого
    # плеча, а не шум, и он не сокращается числом раундов.
    rng = np.random.RandomState(23)
    for rnd in range(ROUNDS):
        order = VAR if rnd % 2 == 0 else VAR[::-1]
        if rng.rand() < 0.5:
            order = order[::-1]
        for name, ck in order:
            model._grad_ckpt = ck
            step(); mx.clear_cache()               # прогрев варианта
            mx.reset_peak_memory()
            for _ in range(3):
                t1 = time.perf_counter()
                step()
                acc[name].append(time.perf_counter() - t1)
            pk[name] = max(pk[name], mx.get_peak_memory() / 1e6)
        print("  раунд %d: своп %.0f МБ" % (rnd, swap_mb()), flush=True)
    sw1 = swap_mb()
    model._grad_ckpt = True

    base = np.median(acc["ckpt=True"])
    print("T=%d, раундов %d" % (T, ROUNDS))
    for name, _ in VAR:
        m = np.median(acc[name])
        sp = (max(acc[name]) - min(acc[name])) / m * 100
        print("%-12s %7.0f мс/шаг (разброс %.1f%%), пик %.0f МБ, "
              "выигрыш %+.0f мс" % (name, m * 1e3, sp, pk[name],
                                    (base - m) * 1e3))
    print("своп %.0f -> %.0f МБ (%s)"
          % (sw0, sw1, "замер валиден" if sw1 - sw0 < 64 else "НЕВАЛИДЕН"))


if __name__ == "__main__":
    main()
