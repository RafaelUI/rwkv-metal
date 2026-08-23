"""A/B СКОРОСТИ ШАГА: int8-база (штатный affine) против sym-базы (.rwkvq).

ПОВОД. Записанное «native 0.7-0.8 с против наивного 3.4-3.6 с» снято
19-20.07 на T=128, и переносить его на нынешние 2.7 с при T=512 нельзя
(закон 29: флопсы шага линейны по T). Этот бенч отвечает на вопрос
замером: сколько стоит шаг на ОДНОЙ И ТОЙ ЖЕ длине у двух баз.

ЧТО СРАВНИВАЕТСЯ -- ПЛЕЧИ КАК ОНИ ЕСТЬ, и различий между ними ДВА, а не
одно (иначе вывод припишут не тому):
  1. сетка: int8 = affine gs=64, идёт РОДНЫМ mx.quantized_matmul;
     sym = блок 16, родного пути нет, обязан ходить деквант -> плотный GEMM;
  2. покрытие: у int8 ветки TMix остаются ПЛОТНЫМИ (их нет в
     BIG_QUANT_TARGETS), в .rwkvq квантованы все 192 матрицы веток.
Плечо int8full закрывает второе различие (ветки тоже квантованы) -- если
разрыв int8 -> int8full мал, дело в сетке, если велик -- в ветках.

МЕТОДИКА: обе модели в ОДНОМ процессе, чередование с рандомизированным
порядком (законы 1, 24), своп печатается по раундам (закон 11: важно не
«вырос ли к концу», а «двигался ли во время замера»), пик -- на плечо.

    python tests/bench_base_int8_vs_sym_ab.py [T] [раундов] [плечи]
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
ARMS = (sys.argv[3].split(",") if len(sys.argv) > 3
        else ["int8", "reduction"])

_argv = sys.argv
sys.argv = ["bench_qlora_arms"]            # его argv-разбор нам не нужен
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import bench_qlora_arms as qa              # noqa: E402
sys.argv = _argv


def swap_mb():
    out = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True,
                         env={**os.environ, "LC_ALL": "C"}).stdout
    p = out.replace("=", " ").split()
    return float(p[p.index("used") + 1].rstrip("M"))


def main():
    rs = np.random.RandomState(5)
    x = mx.array(rs.randint(1, 60000, size=(1, T)).astype(np.int32))
    y = mx.array(rs.randint(1, 60000, size=(1, T)).astype(np.int32))

    def loss_fn(m, a, b):
        return m.loss(a, b).astype(mx.float32)

    built = {}
    for arm in ARMS:
        t0 = time.perf_counter()
        model, tag = qa.build(arm)
        model._grad_ckpt = True
        mx.eval(model.parameters())
        mx.clear_cache()
        gf = nn.value_and_grad(model, loss_fn)
        opt = optim.AdamW(learning_rate=1e-4)

        def step(_m=model, _gf=gf, _opt=opt):
            loss, grads = _gf(_m, x, y)
            _opt.update(_m, grads)
            mx.eval(loss, _m.state, _opt.state)

        built[arm] = {"step": step, "tag": tag, "t": [], "pk": 0.0}
        print("собрано %-12s %s (%.1f с, своп %.0f МБ)"
              % (arm, tag, time.perf_counter() - t0, swap_mb()), flush=True)

    sw0 = swap_mb()
    rng = np.random.RandomState(23)
    for rnd in range(ROUNDS):
        order = list(ARMS) if rnd % 2 == 0 else list(ARMS)[::-1]
        if rng.rand() < 0.5:
            order = order[::-1]
        for arm in order:
            b = built[arm]
            b["step"]()                    # прогрев плеча
            mx.clear_cache()
            mx.reset_peak_memory()
            for _ in range(3):
                t1 = time.perf_counter()
                b["step"]()
                b["t"].append(time.perf_counter() - t1)
            b["pk"] = max(b["pk"], mx.get_peak_memory() / 1e6)
        print("  раунд %d: своп %.0f МБ" % (rnd, swap_mb()), flush=True)
    sw1 = swap_mb()

    print("\nT=%d, раундов %d, замеров на плечо %d"
          % (T, ROUNDS, ROUNDS * 3))
    base = float(np.median(built[ARMS[-1]]["t"]))
    for arm in ARMS:
        b = built[arm]
        m = float(np.median(b["t"]))
        sp = (max(b["t"]) - min(b["t"])) / m * 100
        print("%-12s %7.0f мс/шаг (разброс %.1f%%), пик %.0f МБ, x%.2f к %s"
              % (arm, m * 1e3, sp, b["pk"], base / m, ARMS[-1]))
    print("своп %.0f -> %.0f МБ (%s; смотреть трассу по раундам)"
          % (sw0, sw1, "ровно" if sw1 - sw0 < 64 else "ДВИГАЛСЯ"))


if __name__ == "__main__":
    main()
