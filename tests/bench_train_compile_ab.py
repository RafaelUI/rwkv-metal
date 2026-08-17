"""ДВЕ ГИПОТЕЗЫ О 3.3 с НА ШАГ, ПРОВЕРЯЕМЫЕ ОДНИМ ПРОГОНОМ.

  compile   -- на тренировочном пути НЕТ mx.compile вовсе. На префилле он
               давал +35%, на декоде +17%, и брал это СОКРАЩЕНИЕМ ЧИСЛА
               ПРИМИТИВОВ (589 диспатчей на шаг -- есть что сокращать).
               РИСК, обязательный к замеру: компиляция может протрассировать
               сквозь nn.utils.checkpoint и вернуть пик 12 ГБ вместо 4.3.
               Поэтому меряется И время, И пик.

  runahead  -- прямая проверка версии «CPU подаёт работу по требованию, а
               не уходит вперёд»: mx.eval зовётся раз в K шагов, и если
               постановка лежит на критическом пути, K>1 обязан помочь. На
               декоде это уже проверялось и не дало ничего (эффект
               немонотонен по K и внутри разброса), но обучение -- другой
               режим, и переносить вывод автоматически нельзя.

Против АДДИТИВНОГО прочтения латентности есть арифметика, и её стоит
держать рядом: 1790 диспатчей x 62 мс = 111 с в окне на 10 с. Значит
задержка перекрывается, а не складывается, и высокая CPU-to-GPU latency
означает «GPU занят предыдущими буферами», а не «CPU опоздал».

Контроль численности: лосс всех вариантов обязан совпасть -- иначе
сравниваются разные вычисления, а не разные способы их запустить.

    python bench_train_compile_ab.py [T] [раундов]
"""
import os
import sys
import time
from functools import partial

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
import numpy as np  # noqa: E402

T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
STEPS = 3
PATH = os.environ.get("RWKVQ_BASE", "/tmp/reduction_new.rwkvq")


def main():
    from rwkv_metal.lora import load_rwkvq_model

    model, cfg, _ = load_rwkvq_model(PATH, rank=16, verbose=False)
    model._grad_ckpt = True
    opt = optim.AdamW(learning_rate=1e-4)

    rs = np.random.RandomState(5)
    x = mx.array(rs.randint(1, 60000, size=(1, T)).astype(np.int32))
    y = mx.array(rs.randint(1, 60000, size=(1, T)).astype(np.int32))

    def loss_fn(m, a, b):
        return m.loss(a, b).astype(mx.float32)

    gf = nn.value_and_grad(model, loss_fn)

    def raw_step(a, b):
        loss, grads = gf(model, a, b)
        opt.update(model, grads)
        return loss

    state = [model.state, opt.state]
    try:
        comp_step = partial(mx.compile, inputs=state, outputs=state)(raw_step)
        comp_step(x, y)
        mx.eval(state)
        compile_ok = True
    except Exception as e:                                  # noqa: BLE001
        compile_ok = False
        print("mx.compile НЕ СОБРАЛСЯ: %s: %s"
              % (type(e).__name__, str(e)[:300]), flush=True)

    def run(fn, k):
        """k шагов между mx.eval: k=1 -- как сейчас, k>1 -- забег вперёд."""
        losses = []
        for _ in range(k):
            losses.append(fn(x, y))
        mx.eval(losses, state)
        return float(losses[-1])

    VAR = [("сырой, eval каждый", raw_step, 1),
           ("сырой, eval раз в 2", raw_step, 2),
           ("сырой, eval раз в 4", raw_step, 4)]
    if compile_ok:
        VAR += [("compiled, eval каждый", comp_step, 1),
                ("compiled, eval раз в 2", comp_step, 2)]

    acc = {v[0]: [] for v in VAR}
    peak = {v[0]: 0.0 for v in VAR}
    loss0 = {}
    for rnd in range(ROUNDS):
        for name, fn, k in VAR:
            run(fn, k)                                       # прогрев
            mx.clear_cache()
            if hasattr(mx, "reset_peak_memory"):
                mx.reset_peak_memory()
            for _ in range(STEPS):
                t1 = time.time()
                lv = run(fn, k)
                acc[name].append((time.time() - t1) / k)     # НА ОДИН шаг
            peak[name] = max(peak[name], mx.get_peak_memory() / 1e6)
            loss0[name] = lv
        print("  раунд %d" % rnd, flush=True)

    base = float(np.median(acc[VAR[0][0]]))
    print("\n%-24s%10s%9s%10s%10s%12s" % ("вариант", "мс/шаг", "разброс",
                                          "x к базе", "пик МБ", "loss"))
    for name, _, _ in VAR:
        m = float(np.median(acc[name]))
        sp = (max(acc[name]) - min(acc[name])) / m * 100
        print("%-24s%10.0f%8.1f%%%10.3f%10.0f%12.4f"
              % (name, m * 1e3, sp, base / m, peak[name], loss0[name]))
    ls = list(loss0.values())
    print("\nлоссы %s (разброс %.2e) -- сравниваются %s вычисления"
          % ("совпали" if max(ls) - min(ls) < 1e-2 else "РАЗОШЛИСЬ",
             max(ls) - min(ls),
             "одни и те же" if max(ls) - min(ls) < 1e-2 else "РАЗНЫЕ"))


if __name__ == "__main__":
    main()
