"""ИЗ ЧЕГО СОСТОИТ ШАГ ОБУЧЕНИЯ: аблации, чередование в одном процессе.

Трасса на этот вопрос ответить не может -- в Metal System Trace нет
атрибуции по ядрам (трек Compute показывает команды буфера). Значит
вычитание, как в bench_step_decompose для декода.

Компоненты подменяются ГЛОБАЛЯМИ МОДУЛЯ, а не пересборкой модели:
`rwkv7_x070` берёт `wkv7` из своих глобалей при каждом вызове, а
`mx.compile` на тренировочном пути нет вовсе -- значит варианты можно
чередовать в одном процессе (закон 1), а не гонять по процессу на
вариант с втрое большим разбросом.

  wkv     -- рекуррентность заменена поэлементной комбинацией входов.
             Уходят и forward-, и backward-ядра скана; все тензоры
             остаются в графе, формы те же, градиенты текут.
  dq      -- деквант базы отдаёт нули нужной формы: уходит распаковка,
             матмул и плотный транзиент остаются.
  оба     -- контроль на аддитивность: аблации НЕ обязаны складываться.

    python probe_train_components.py [T] [раундов]
"""
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
STEPS = 3
PATH = os.environ.get("RWKVQ_BASE", "/tmp/reduction_new.rwkvq")


def main():
    from rwkv_metal.lora import load_rwkvq_model
    from rwkv_metal.lora import rwkvq_linear as rl
    import rwkv_metal.model.rwkv7_x070 as mod

    model, cfg, info = load_rwkvq_model(PATH, rank=16, verbose=False)
    model._grad_ckpt = True
    print("grad_checkpoint=%s, T=%d" % (model._grad_ckpt, T), flush=True)

    wkv_real = mod.wkv7
    dq_real = rl.RwkvqSymLinear._dequant_w

    def wkv_stub(r, w, k, v, a, b, training=True, state=None,
                 return_state=False):
        out = r * 0.5 + v * 0.5 + (k + w + a + b) * 0.0
        return out, None

    def dq_stub(self):
        return mx.zeros((self.out_features, self.in_features),
                        dtype=mx.bfloat16)

    rs = np.random.RandomState(5)
    x = mx.array(rs.randint(1, 60000, size=(1, T)).astype(np.int32))
    y = mx.array(rs.randint(1, 60000, size=(1, T)).astype(np.int32))

    def loss_fn(m, a, b):
        return m.loss(a, b).astype(mx.float32)

    gf = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=1e-4)
    loss, grads = gf(model, x, y)
    opt.update(model, grads)
    mx.eval(loss, model.state, opt.state)

    VAR = [("полный", False, False), ("без WKV", True, False),
           ("без декванта", False, True), ("без обоих", True, True)]
    acc = {v[0]: [] for v in VAR}
    peak = {v[0]: 0.0 for v in VAR}

    for rnd in range(ROUNDS):
        for name, ab_wkv, ab_dq in VAR:
            mod.wkv7 = wkv_stub if ab_wkv else wkv_real
            rl.RwkvqSymLinear._dequant_w = dq_stub if ab_dq else dq_real
            loss, grads = gf(model, x, y)          # прогрев варианта
            mx.eval(loss, grads)
            mx.clear_cache()
            if hasattr(mx, "reset_peak_memory"):
                mx.reset_peak_memory()
            for _ in range(STEPS):
                t1 = time.time()
                loss, grads = gf(model, x, y)
                opt.update(model, grads)
                mx.eval(loss, model.state, opt.state)
                acc[name].append(time.time() - t1)
            peak[name] = max(peak[name], mx.get_peak_memory() / 1e6)
        print("  раунд %d готов" % rnd, flush=True)

    mod.wkv7 = wkv_real
    rl.RwkvqSymLinear._dequant_w = dq_real

    full = float(np.median(acc["полный"]))
    print("\n%-16s%10s%9s%12s%11s" % ("вариант", "мс/шаг", "разброс",
                                      "статья мс", "пик МБ"))
    for name, _, _ in VAR:
        m = float(np.median(acc[name]))
        sp = (max(acc[name]) - min(acc[name])) / m * 100
        print("%-16s%10.0f%8.1f%%%12.0f%11.0f"
              % (name, m * 1e3, sp, (full - m) * 1e3, peak[name]))
    d_w = full - float(np.median(acc["без WKV"]))
    d_q = full - float(np.median(acc["без декванта"]))
    d_b = full - float(np.median(acc["без обоих"]))
    print("\nWKV %.0f мс (%.0f%% шага), деквант %.0f мс (%.0f%%)"
          % (d_w * 1e3, 100 * d_w / full, d_q * 1e3, 100 * d_q / full))
    print("сумма по отдельности %.0f мс против %.0f мс вместе -- "
          "аблации %sаддитивны"
          % ((d_w + d_q) * 1e3, d_b * 1e3,
             "" if abs(d_w + d_q - d_b) < 0.1 * full else "НЕ "))


if __name__ == "__main__":
    main()
