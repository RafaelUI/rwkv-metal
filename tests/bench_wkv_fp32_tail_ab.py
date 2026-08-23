"""ЦЕНА fp32-ХВОСТА ПОСЛЕ WKV И ДВА СПОСОБА ЕЁ НЕ ПЛАТИТЬ.

Выход WKV-ядра -- fp32, и `CAST_WKV_OUTPUT = False`, поэтому всё, что стоит
после него в блоке (ln_x, bonus, gate и, главное, `o_proj`), считается в
fp32 вместо bf16. У квантованных линейных слоёв каста входа НЕТ вовсе
(`return x @ self._dequant_w().T`), значит fp32-вход тянет за собой и
fp32-копию деквантованной матрицы, и fp32-матмул -- на 24 слоя это самая
крупная статья тренировочного шага после самих проекций.

`probe_wkv_train_surround` измерил её двумя независимыми путями: 679 мс на
настоящем ядре и 750 на заглушке (шаг 3198 мс). Здесь меряются СПОСОБЫ её
убрать, и они не равноценны по числам:

  база          -- как сейчас
  каст в слое   -- `x.astype(w.dtype) @ w.T` в квантованных линейных.
                   Ровно то, что УЖЕ делает инференсный путь rwkv-quant
                   (`quant_model._matmul`), то есть правка сближает две
                   реализации, а не разводит их. Хвост до o_proj остаётся
                   в fp32 -- меняется только точность входа матмула
  CAST_WKV_OUT  -- приведение выхода ядра сразу, весь хвост в bf16.
                   Меняет числа во всём стеке, а rwkv-metal -- источник
                   эталонов паритета для SwiftRWKV
  оба           -- контроль на аддитивность (аблации складываться НЕ обязаны)

Контроль качества здесь ТОЛЬКО индикативный: relmax логитов и значение
лосса на одном батче. Это не ppl-гейт; решение о внедрении принимается
по мультиязычному корпусу, а не по этому числу (закон 9).

Чередование в одном процессе, порядок вариантов внутри раунда
РАНДОМИЗИРОВАН (закон 24), окно короткое (закон 25), своп на границе
замера (закон 11).

    python bench_wkv_fp32_tail_ab.py [T] [раундов] [grad|fwd]
"""
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))

import mlx.core as mx
import mlx.nn as nn
import numpy as np

T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MODE = sys.argv[3] if len(sys.argv) > 3 else "grad"
STEPS = 3
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
    import rwkv_metal.model.rwkv7_x070 as mod

    model, cfg, info = load_rwkvq_model(PATH, rank=16, verbose=False)
    model._grad_ckpt = True
    n_layer = len(model.blocks)

    # Классы квантованных линейных: правка одна и та же, а классов два
    # (sb6 и sym) -- закон 23: список мест перечисляется, а не угадывается.
    # С 22.08 каст входа -- УМОЛЧАНИЕ в rwkvq_linear._matmul_cast, поэтому
    # «база» здесь -- ЗАМОРОЖЕННАЯ КОПИЯ прежнего вызова (закон 27), а не
    # «что стоит в модуле»: иначе «база» молча совпала бы с «кастом в
    # слое», и таблица выродилась бы в сравнение самих с собой.
    CLASSES = [c for c in (getattr(rl, "RwkvqLinear", None),
                           getattr(rl, "RwkvqSymLinear", None)) if c is not None]

    def nocast_call(self, x):
        return x @ self._dequant_w().T

    def cast_call(self, x):
        w = self._dequant_w()
        return (x.astype(w.dtype) @ w.T).astype(x.dtype)

    def set_lincast(on):
        for c in CLASSES:
            c.__call__ = cast_call if on else nocast_call

    def set_default():
        """Вернуть боевое умолчание модуля (каст включен), чтобы прогон
        бенча не оставлял классы в «базовом» режиме."""
        for c in CLASSES:
            c.__call__ = rl._matmul_cast

    def set_wkvcast(on):
        mod.CAST_WKV_OUTPUT = on

    VAR = [("база", False, False),
           ("каст в слое", True, False),
           ("CAST_WKV_OUT", False, True),
           ("оба", True, True)]

    rs = np.random.RandomState(5)
    x = mx.array(rs.randint(1, 60000, size=(1, T)).astype(np.int32))
    y = mx.array(rs.randint(1, 60000, size=(1, T)).astype(np.int32))

    def loss_fn(m, a, b):
        return m.loss(a, b).astype(mx.float32)

    gf = nn.value_and_grad(model, loss_fn)

    if MODE == "grad":
        def run():
            loss, grads = gf(model, x, y)
            mx.eval(loss, grads)
    else:
        def run():
            mx.eval(model(x))

    # индикативный контроль качества
    print("\nконтроль (один батч, НЕ ppl-гейт):")
    ref_lg = None
    for name, lc, wc in VAR:
        set_lincast(lc)
        set_wkvcast(wc)
        lg = model(x)
        ls = float(model.loss(x, y))
        mx.eval(lg)
        if ref_lg is None:
            ref_lg = lg
            print("  %-13s лосс %.6f  (эталон)" % (name, ls))
        else:
            d = float(mx.max(mx.abs(lg - ref_lg)) /
                      (mx.max(mx.abs(ref_lg)) + 1e-30))
            print("  %-13s лосс %.6f  relmax логитов %.3e" % (name, ls, d))
    set_default()
    set_wkvcast(False)
    del ref_lg
    mx.clear_cache()

    acc = {v[0]: [] for v in VAR}
    pk = {v[0]: 0.0 for v in VAR}
    sw0 = swap_mb()
    order = list(range(len(VAR)))
    rng = random.Random(20260817)

    for rnd_i in range(ROUNDS):
        rng.shuffle(order)
        for idx in order:
            name, lc, wc = VAR[idx]
            set_lincast(lc)
            set_wkvcast(wc)
            run()                                  # прогрев варианта
            mx.clear_cache()
            if hasattr(mx, "reset_peak_memory"):
                mx.reset_peak_memory()
            for _ in range(STEPS):
                t1 = time.perf_counter()
                run()
                acc[name].append(time.perf_counter() - t1)
            pk[name] = max(pk[name], mx.get_peak_memory() / 1e6)
        print("  раунд %d (порядок %s)" % (rnd_i, order), flush=True)
    sw1 = swap_mb()
    set_default()
    set_wkvcast(False)

    base = float(np.median(acc["база"]))
    print("\nрежим %s, T=%d, слоёв %d" % (MODE, T, n_layer))
    print("%-14s%10s%10s%12s%12s%11s"
          % ("вариант", "мс", "разброс", "выигрыш", "x к базе", "пик МБ"))
    for name, _, _ in VAR:
        m = float(np.median(acc[name]))
        sp = (max(acc[name]) - min(acc[name])) / m * 100
        print("%-14s%10.1f%9.1f%%%12.1f%12.3f%11.0f"
              % (name, m * 1e3, sp, (base - m) * 1e3, base / m, pk[name]))

    def med(n):
        return float(np.median(acc[n])) * 1e3

    s = (med("база") - med("каст в слое")) + (med("база") - med("CAST_WKV_OUT"))
    both = med("база") - med("оба")
    print("\nсумма по отдельности %.1f мс против %.1f вместе -- %sаддитивны"
          % (s, both, "" if abs(s - both) < 0.1 * base * 1e3 else "НЕ "))
    print("своп %.0f -> %.0f МБ (%s)"
          % (sw0, sw1, "замер валиден" if sw1 - sw0 < 1 else "НЕВАЛИДЕН"))


if __name__ == "__main__":
    main()
