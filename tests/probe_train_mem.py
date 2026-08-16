"""ИЗ ЧЕГО СОСТОИТ ПАМЯТЬ ШАГА ОБУЧЕНИЯ QLoRA (1.5B, sym-база).

ЗАЧЕМ. Записанные 9.15 ГБ -- это пик СБОРКИ (модель + один forward на
четыре токена, backward не было вовсе), и к обучению он отношения не
имеет. Арифметика на бумаге даёт три кандидата: плотная база, которую
держит автоград (~2.68 ГБ в bf16), классические активации (~1.2 ГБ на
24 слоя при T=512) и логиты с копиями в лоссе (~0.3-0.4 ГБ). Но
`grad_checkpoint=True` стоит умолчанием, и если он делает то, что должен,
плотная база живёт по одному блоку за раз -- тогда первый кандидат
обнуляется, а вместе с ним и смысл писать свой VJP.

Разделяется это тремя прогонами ПО ПРОЦЕССУ на конфигурацию (пик памяти
процессный, в одном процессе его не разделить):

    ckpt   -- как сейчас, nn.utils.checkpoint на блок;
    nockpt -- то же без чекпоинтинга: разница и есть цена хранения;
    nodq   -- деквант базы подменён нулями нужной формы. Уходит и
              распаковка, и её транзиент, матмул остаётся. Это ВЕРХНЯЯ
              оценка того, что может дать свой VJP: он убирает хранение
              W, но не сам деквант.

Пик снимается снаружи `/usr/bin/time -l` (закон 22), внутри печатается
ещё и аллокаторный `mx.get_peak_memory` -- он врёт в плюс на накладные,
но годится для СРАВНЕНИЯ статей внутри одного прогона.

    /usr/bin/time -l python probe_train_mem.py <ckpt|nockpt|nodq> [T] [шагов]
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
import numpy as np  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "ckpt"
T = int(sys.argv[2]) if len(sys.argv) > 2 else 512
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
PATH = os.environ.get("RWKVQ_BASE", "/tmp/reduction_new.rwkvq")


def swap_mb():
    env = dict(os.environ, LC_ALL="C", LANG="C")
    o = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                       capture_output=True, text=True).stdout
    u = o.split("used =")[1].split()[0]
    unit, num = u[-1], u[:-1]
    if "," in num:
        num = num.replace(".", "").replace(",", ".")
    return float(num) * (1024 if unit == "G" else 1)


def main():
    from rwkv_metal.lora import load_rwkvq_model
    from rwkv_metal.lora import rwkvq_linear as rl

    if MODE == "nodq":
        # Подмена ДО сборки модели: деквант отдаёт нули нужной формы.
        # Уходит распаковка и её транзиент, матмул остаётся -- то есть
        # это верхняя оценка выигрыша от любой правки, которая трогает
        # ТОЛЬКО деквант (свой VJP в том числе).
        def _zeros(self):
            return mx.zeros((self.out_features, self.in_features),
                            dtype=mx.bfloat16)
        rl.RwkvqSymLinear._dequant_w = _zeros

    t0 = time.time()
    model, cfg, info = load_rwkvq_model(PATH, rank=16, verbose=False)
    if hasattr(model, "_grad_ckpt"):
        model._grad_ckpt = (MODE != "nockpt")
    load_s = time.time() - t0
    peak_load = mx.get_peak_memory() / 1e9

    rs = np.random.RandomState(5)
    x = mx.array(rs.randint(1, 60000, size=(1, T)).astype(np.int32))
    y = mx.array(rs.randint(1, 60000, size=(1, T)).astype(np.int32))

    def loss_fn(m, a, b):
        return m.loss(a, b).astype(mx.float32)

    grad_fn = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=1e-4)

    mx.reset_peak_memory() if hasattr(mx, "reset_peak_memory") else None
    sw0 = swap_mb()
    ts = []
    for i in range(STEPS):
        t1 = time.time()
        loss, grads = grad_fn(model, x, y)
        opt.update(model, grads)
        mx.eval(loss, model.state, opt.state)
        ts.append(time.time() - t1)
        if i == 0:
            first = float(loss)
    sw1 = swap_mb()

    print(f"режим {MODE}, T={T}, шагов {STEPS}")
    print(f"  сборка {load_s:.1f} с, аллокаторный пик после сборки "
          f"{peak_load:.2f} ГБ")
    print(f"  loss[0] = {first:.4f}, шаг: первый {ts[0]*1e3:.0f} мс, "
          f"медиана дальше {np.median(ts[1:])*1e3:.0f} мс"
          if len(ts) > 1 else "")
    print(f"  аллокаторный пик за шаг: {mx.get_peak_memory()/1e9:.2f} ГБ")
    print(f"  своп {sw0:.0f} -> {sw1:.0f} МБ"
          + ("   *** ВЫРОС ВО ВРЕМЯ ЗАМЕРА, скорость недействительна ***"
             if sw1 > sw0 + 0.5 else ""))
    print(f"  обучаемых: {info.get('trainable_params', '?')}")


if __name__ == "__main__":
    main()
