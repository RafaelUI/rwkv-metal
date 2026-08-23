"""A/B СКВОЗНОГО ШАГА: прямая bf16-ветка декванта против прежней цепочки.

`bench_dequant_bw` намерил -63% на изолированном проходе, но изолированный
замер НЕ включает наложение с матмулами настоящего шага (закон 29: выигрыш
в шаге может быть МЕНЬШЕ напечатанного). Здесь арбитр: полный шаг
обучения, обе ветки в одном процессе, старая -- ЗАМОРОЖЕННОЙ копией
(fp32-кернель + astype), чередование, своп на границе (закон 11).

    python tests/bench_dq_direct_ab.py [T] [раундов]
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
    from rwkv_metal.lora import rwkvq_linear as rl

    model, cfg, info = load_rwkvq_model(PATH, rank=16, verbose=False)
    model._grad_ckpt = True

    dq_new = rl.RwkvqSymLinear._dequant_w

    def dq_old(self):
        return self._sym._dequant_w(mx.float32).astype(mx.bfloat16)

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

    VAR = [("прямая bf16 (новая)", dq_new), ("цепочка fp32+astype (старая)", dq_old)]
    acc = {n: [] for n, _ in VAR}
    pk = {n: 0.0 for n, _ in VAR}
    sw0 = swap_mb()
    for rnd in range(ROUNDS):
        for name, fn in VAR:
            rl.RwkvqSymLinear._dequant_w = fn
            step(); mx.clear_cache()
            mx.reset_peak_memory()
            for _ in range(3):
                t1 = time.perf_counter()
                step()
                acc[name].append(time.perf_counter() - t1)
            pk[name] = max(pk[name], mx.get_peak_memory() / 1e6)
    sw1 = swap_mb()
    rl.RwkvqSymLinear._dequant_w = dq_new

    new = np.median(acc[VAR[0][0]])
    old = np.median(acc[VAR[1][0]])
    print("T=%d, раундов %d" % (T, ROUNDS))
    for name, _ in VAR:
        m = np.median(acc[name])
        sp = (max(acc[name]) - min(acc[name])) / m * 100
        print("%-28s %7.0f мс/шаг (разброс %.1f%%), пик %.0f МБ"
              % (name, m * 1e3, sp, pk[name]))
    print("выигрыш прямой bf16: %+.0f мс (%.1f%%), пик %+.0f МБ"
          % ((old - new) * 1e3, 100 * (old - new) / old, pk[VAR[0][0]] - pk[VAR[1][0]]))
    print("своп %.0f -> %.0f МБ (%s)"
          % (sw0, sw1, "замер валиден" if sw1 - sw0 < 64 else "НЕВАЛИДЕН"))


if __name__ == "__main__":
    main()
