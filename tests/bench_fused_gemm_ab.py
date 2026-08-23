"""A/B СКВОЗНОГО ШАГА: fused dequant+GEMM против цепочки «деквант+матмул».

Изолированная проба (dev_fused_gemm_probe) дала ПАРИТЕТ на отдельных
тензорах -- выигрыш заявлен не в изолированном матмуле, а в шаге: минус
292 запуска декванта, минус плотный транзиент W, минус конкуренцию за
кеш. Арбитр -- полный шаг обучения (закон 29).

Обе ветки в одном процессе, переключение флагом rl.NOFUSED (закон 27:
«база» -- это сам прежний код-путь, а не захват копии), порядок внутри
раунда РАНДОМИЗИРОВАН (закон 24), своп на границе (закон 11).

    python tests/bench_fused_gemm_ab.py [T] [раундов]
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
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
PATH = os.environ.get("RWKVQ_BASE", "/tmp/reduction_new.rwkvq")


def swap_mb():
    out = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True,
                         env={**os.environ, "LC_ALL": "C"}).stdout
    p = out.replace("=", " ").split()
    return float(p[p.index("used") + 1].rstrip("M"))


def main():
    from rwkv_metal.lora import load_rwkvq_model
    from rwkv_metal.lora import rwkvq_linear as rl

    model, cfg, info = load_rwkvq_model(PATH, rank=16, verbose=False)
    model._grad_ckpt = True

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

    VAR = [("fused (новый)", False), ("цепочка (старый)", True)]
    acc = {n: [] for n, _ in VAR}
    pk = {n: 0.0 for n, _ in VAR}
    fused_hits = 0
    sw0 = swap_mb()
    rng = np.random.RandomState(23)
    for rnd in range(ROUNDS):
        order = VAR if rnd % 2 == 0 else VAR[::-1]
        if rng.rand() < 0.5:
            order = order[::-1]
        for name, nofused in order:
            rl.NOFUSED = nofused
            c0 = rl.FUSED_CALLS
            step(); mx.clear_cache()
            mx.reset_peak_memory()
            for _ in range(3):
                t1 = time.perf_counter()
                step()
                acc[name].append(time.perf_counter() - t1)
            pk[name] = max(pk[name], mx.get_peak_memory() / 1e6)
            if not nofused:
                fused_hits += rl.FUSED_CALLS - c0
    sw1 = swap_mb()
    rl.NOFUSED = False

    assert fused_hits > 0, "fused-путь ни разу не включился -- замер пуст"

    med = {n: float(np.median(acc[n])) for n, _ in VAR}
    for name, _ in VAR:
        m = med[name]
        sp = (max(acc[name]) - min(acc[name])) / m * 100
        print("%-18s %7.0f мс/шаг (разброс %.1f%%), пик %.0f МБ"
              % (name, m * 1e3, sp, pk[name]))
    new, old = med[VAR[0][0]], med[VAR[1][0]]
    print("выигрыш fused: %+.0f мс (%.1f%%), пик %+.0f МБ, вызовов fused %d"
          % ((old - new) * 1e3, 100 * (old - new) / old,
             pk[VAR[0][0]] - pk[VAR[1][0]], fused_hits))
    print("своп %.0f -> %.0f МБ (%s)"
          % (sw0, sw1, "замер валиден" if sw1 - sw0 < 64 else "НЕВАЛИДЕН"))


if __name__ == "__main__":
    main()
