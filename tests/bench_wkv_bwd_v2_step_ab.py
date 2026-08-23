"""СКВОЗНОЙ A/B ШАГА: backward-ядро WKV v1 против v2 (порт Бо Пэна).

Микрозамер одного слоя дал x1.89 (13.57 -> 7.17 мс) и обещает по
арифметике 24*(2*fwd+bwd) = 418 -> 265 мс на шаг. Обещание проверяется
здесь: закон 29 -- микрозамер завышает абсолюты, потому что в шаге работа
24 слоёв перекрывается, а тут каждый вызов обрамлён mx.eval.

Переключение -- рантайм-флагом wkv7_checkpoint.BWD_V2 в ОДНОМ процессе
(закон 27), порядок рандомизирован (закон 24), своп по раундам (закон 11).

КОНТРОЛЬ ВКЛЮЧЕНИЯ обязателен: v2 отличается от v1 только переассоциацией
сумм, поэтому "разницы во времени нет" и "флаг не сработал" выглядят
одинаково. Контроль требует, чтобы градиенты РАЗЛИЧАЛИСЬ (relmax > 0) и
при этом были близки (< 1e-3).

    python tests/bench_wkv_bwd_v2_step_ab.py [T] [раундов]
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
from mlx.utils import tree_flatten         # noqa: E402

from rwkv_metal.kernel import wkv7_checkpoint as ck   # noqa: E402

T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
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
    model._grad_ckpt = True
    rs = np.random.RandomState(5)
    x = mx.array(rs.randint(1, 60000, size=(1, T)).astype(np.int32))
    y = mx.array(rs.randint(1, 60000, size=(1, T)).astype(np.int32))

    def loss_fn(m, a, b):
        return m.loss(a, b).astype(mx.float32)

    gf = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=1e-4)

    def grads_vec(v2):
        ck.BWD_V2 = v2
        loss, grads = gf(model, x, y)
        mx.eval(loss, grads)
        return float(loss), np.concatenate(
            [np.array(g.astype(mx.float32)).ravel() for _, g in tree_flatten(grads)])

    l1, g1 = grads_vec(False)
    l2, g2 = grads_vec(True)
    rel = np.abs(g1 - g2).max() / max(np.abs(g1).max(), 1e-30)
    l2n = np.linalg.norm(g1 - g2) / max(np.linalg.norm(g1), 1e-30)
    assert rel > 0, "флаг не сработал: градиенты бит-в-бит, мерится одно ядро"

    # ШУМОВОЙ ПОЛ ИЗМЕРЯЕТСЯ, А НЕ НАЗНАЧАЕТСЯ. Порог "меньше 1e-3" тут
    # подгонялся бы на глаз: градиенты приводятся к bf16 (решётка 2^-8),
    # проходят 24 слоя, и мелкие компоненты рождаются вычитанием близких
    # чисел -- у ЛЮБЫХ двух порядков суммирования расхождение будет
    # заметным. Поэтому берём эталон того же рода: у v1 порядок задаёт
    # константа ACC_BWD, и её смена даёт МАТЕМАТИЧЕСКИ ТУ ЖЕ величину,
    # посчитанную иначе. Если v2 отстоит от v1 не дальше, чем v1 с ACC=2
    # от v1 с ACC=4, расхождение -- свойство типа и перестановки сумм,
    # а не ошибка ядра.
    acc0 = ck.ACC_BWD
    try:
        ck.ACC_BWD = 2 if acc0 != 2 else 1
        ck._bwd_cache.clear()
        _, g1b = grads_vec(False)
    finally:
        ck.ACC_BWD = acc0
        ck._bwd_cache.clear()
    floor_l2 = np.linalg.norm(g1 - g1b) / max(np.linalg.norm(g1), 1e-30)
    floor_max = np.abs(g1 - g1b).max() / max(np.abs(g1).max(), 1e-30)
    assert floor_l2 > 0, ("эталон шума нулевой -- смена ACC_BWD не пересобрала "
                          "ядро, контроль пуст")
    assert l2n < 3 * floor_l2, (
        "v2 отстоит от v1 ДАЛЬШЕ шумового пола: %.2e против %.2e"
        % (l2n, floor_l2))
    print("контроль включения: лосс %.6f / %.6f" % (l1, l2), flush=True)
    print("  v2 против v1:          ||dg||/||g|| %.2e, relmax %.2e"
          % (l2n, rel), flush=True)
    print("  ПОЛ (v1 ACC=%d/ACC=%d): ||dg||/||g|| %.2e, relmax %.2e"
          % (acc0, 2 if acc0 != 2 else 1, floor_l2, floor_max), flush=True)

    def step():
        loss, grads = gf(model, x, y)
        opt.update(model, grads)
        mx.eval(loss, model.state, opt.state)

    VAR = [("bwd v1", False), ("bwd v2", True)]
    acc = {n: [] for n, _ in VAR}
    pk = {n: 0.0 for n, _ in VAR}
    sw0 = swap_mb()
    rng = np.random.RandomState(23)
    for rnd in range(ROUNDS):
        order = VAR if rnd % 2 == 0 else VAR[::-1]
        if rng.rand() < 0.5:
            order = order[::-1]
        for name, v2 in order:
            ck.BWD_V2 = v2
            step(); mx.clear_cache()
            mx.reset_peak_memory()
            for _ in range(3):
                t1 = time.perf_counter()
                step()
                acc[name].append(time.perf_counter() - t1)
            pk[name] = max(pk[name], mx.get_peak_memory() / 1e6)
        print("  раунд %d: своп %.0f МБ" % (rnd, swap_mb()), flush=True)
    sw1 = swap_mb()
    ck.BWD_V2 = False

    print("\nT=%d, раундов %d, замеров на плечо %d" % (T, ROUNDS, ROUNDS * 3))
    med = {}
    for name, _ in VAR:
        m = float(np.median(acc[name])); med[name] = m
        sp = (max(acc[name]) - min(acc[name])) / m * 100
        print("%-8s %7.0f мс/шаг (разброс %.1f%%), пик %.0f МБ"
              % (name, m * 1e3, sp, pk[name]))
    d = (med["bwd v1"] - med["bwd v2"]) * 1e3
    print("выигрыш v2: %+.0f мс (%.1f%%)" % (d, 100 * d / (med["bwd v1"] * 1e3)))
    print("своп %.0f -> %.0f МБ" % (sw0, sw1))


if __name__ == "__main__":
    main()
