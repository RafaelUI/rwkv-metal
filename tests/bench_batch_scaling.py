"""ПРОПУСКНАЯ СПОСОБНОСТЬ ШАГА ПО БАТЧУ: недогружен ли GPU при B=1.

Изолированный замер ядер (bench_wkv_fwd_vs_bwd) показал, что forward WKV
при B=1 недогружен примерно вдвое: рост B с 1 до 2 стоил +31% времени при
удвоении работы. Вопрос шире: сколько стоит ШАГ обучения на батч, и падает
ли цена одной последовательности с ростом B. Если падает -- на 16 ГБ есть
запас памяти, который можно обменять на пропускную способность обучения
(та же выборка проходится за меньшее время), и это рычаг, не трогающий
ни качество, ни формат.

Меряется цена ОДНОГО ШАГА и цена НА ПОСЛЕДОВАТЕЛЬНОСТЬ. Второе -- то, что
определяет время эпохи; первое само по себе обязано расти.

Своп печатается по раундам (закон 11): на больших B пик уходит за 8 ГБ, и
замер скорости с ростом свопа недействителен. Порядок B рандомизирован
внутри раунда (законы 1, 24) -- иначе прогрев/нагрев ложится на последние.

    python tests/bench_batch_scaling.py [T] [раундов] [B,B,B]
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
BATCHES = ([int(x) for x in sys.argv[3].split(",")]
           if len(sys.argv) > 3 else [1, 2, 4])
PATH = os.environ.get("RWKVQ_BASE", "/tmp/reduction_new.rwkvq")


def swap_mb():
    out = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True,
                         env={**os.environ, "LC_ALL": "C"}).stdout
    p = out.replace("=", " ").split()
    return float(p[p.index("used") + 1].rstrip("M"))


def main():
    from rwkv_metal.lora import load_rwkvq_model
    from rwkv_metal.kernel import wkv7_checkpoint as ck
    print("backward WKV: %s" % ("v2" if ck.BWD_V2 else "v1"), flush=True)

    model, cfg, info = load_rwkvq_model(PATH, rank=16, verbose=False)
    model._grad_ckpt = True

    def loss_fn(m, a, b):
        return m.loss(a, b).astype(mx.float32)

    gf = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=1e-4)

    data = {}
    for B in BATCHES:
        rs = np.random.RandomState(5)
        data[B] = (mx.array(rs.randint(1, 60000, size=(B, T)).astype(np.int32)),
                   mx.array(rs.randint(1, 60000, size=(B, T)).astype(np.int32)))

    def step(B):
        x, y = data[B]
        loss, grads = gf(model, x, y)
        opt.update(model, grads)
        mx.eval(loss, model.state, opt.state)

    acc = {B: [] for B in BATCHES}
    pk = {B: 0.0 for B in BATCHES}
    sw0 = swap_mb()
    rng = np.random.RandomState(23)
    failed = set()
    for rnd in range(ROUNDS):
        order = list(BATCHES)
        rng.shuffle(order)
        for B in order:
            if B in failed:
                continue
            try:
                step(B); mx.clear_cache()
                mx.reset_peak_memory()
                for _ in range(2):
                    t1 = time.perf_counter()
                    step(B)
                    acc[B].append(time.perf_counter() - t1)
                pk[B] = max(pk[B], mx.get_peak_memory() / 1e6)
            except Exception as e:                       # noqa: BLE001
                print("  B=%d упал: %s" % (B, str(e)[:120]), flush=True)
                failed.add(B)
                mx.clear_cache()
        print("  раунд %d: своп %.0f МБ" % (rnd, swap_mb()), flush=True)
    sw1 = swap_mb()

    print("\nT=%d, раундов %d" % (T, ROUNDS))
    print("%-4s %10s %14s %12s %10s" % (
        "B", "мс/шаг", "мс/послед.", "ток/с", "пик МБ"))
    for B in BATCHES:
        if not acc[B]:
            print("%-4d %10s" % (B, "не влез"))
            continue
        m = float(np.median(acc[B])) * 1e3
        print("%-4d %10.0f %14.0f %12.0f %10.0f"
              % (B, m, m / B, B * T / (m / 1e3), pk[B]))
    print("своп %.0f -> %.0f МБ (смотреть трассу по раундам)" % (sw0, sw1))


if __name__ == "__main__":
    main()
