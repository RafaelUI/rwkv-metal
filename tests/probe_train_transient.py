"""РАЗБОР ТРАНЗИЕНТА ШАГА ОБУЧЕНИЯ QLoRA (22.08, после внедрения каста).

Записанные 17.08: шаг 3288 мс, пик 4261 МБ, атрибутировано только ~470 МБ
декванта и ~0 у кросс-энтропии; ~2.6 ГБ транзиента не разложено. С тех пор
две правки поменяли саму базу разложения: найден и измерен fp32-хвост
(679-750 МБ), каст входа внедрён 22.08 (снимает ~530 МБ пика). Этот пробник
начинает разбор ЗАНОВО на новой базе.

ЧТО МЕРЯЕТСЯ. `mx.reset_peak_memory()` вокруг сегмента, `mx.get_peak_memory()`
после -- пик АЛЛОКАТОРА MLX. Он врёт в плюс на накладные аллокатора и не
видит страниц (закон 11), но для ОТНОШЕНИЯ статей внутри одного процесса
годится; абсолют сверяется с /usr/sbin/sysctl vm.swapusage на границе.

СЕГМЕНТЫ:
  resident    -- активная память до шага (модель + адаптеры + state
                 оптимизатора) после clear_cache;
  fwd+loss    -- forward и лосс БЕЗ градиента (чекпоинтинг не работает --
                 он для backward);
  grad        -- value_and_grad без шага оптимизатора;
  полный      -- grad + opt.update + eval;
АБЛАЦИИ полного шага (подмена глобалей, как в probe_train_components):
  без WKV     -- заглушка с ВЫХОДОМ fp32 (закон 34: заглушка обязана
                 повторять dtype заменяемого компонента);
  без деквант -- `_dequant_w` отдаёт нули (уходит распаковка);
  без CE      -- сумма логитов вместо лосса (уходят log_softmax и копии
                 на [T, 65536]);
A/B:
  nocast      -- rl.NOCAST=True на полном шаге -- сколько пика держал
                 fp32-хвост ДО внедрения (ожидание ~530 МБ).

    python tests/probe_train_transient.py [T] [раундов]
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
    import rwkv_metal.model.rwkv7_x070 as mod

    model, cfg, info = load_rwkvq_model(PATH, rank=16, verbose=False)
    model._grad_ckpt = True

    wkv_real = mod.wkv7
    dq_real = rl.RwkvqSymLinear._dequant_w
    loss_real = type(model).loss

    def wkv_stub(r, w, k, v, a, b, training=True, state=None,
                 return_state=False):
        # dtype-верная заглушка: ядро отдаёт fp32 (закон 34)
        out = (r * 0.5 + v * 0.5).astype(mx.float32)
        return out, None

    def dq_stub(self):
        return mx.zeros((self.out_features, self.in_features),
                        dtype=mx.bfloat16)

    def loss_sum(self, a, b):
        lg = self(a)
        return lg.astype(mx.float32).sum() / lg.size

    rs = np.random.RandomState(5)
    x = mx.array(rs.randint(1, 60000, size=(1, T)).astype(np.int32))
    y = mx.array(rs.randint(1, 60000, size=(1, T)).astype(np.int32))

    def loss_fn(m, a, b):
        return m.loss(a, b).astype(mx.float32)

    gf = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=1e-4)

    # прогрев и построение state оптимизатора
    loss, grads = gf(model, x, y)
    opt.update(model, grads)
    mx.eval(loss, model.state, opt.state)

    def seg_full():
        loss, grads = gf(model, x, y)
        opt.update(model, grads)
        mx.eval(loss, model.state, opt.state)

    def seg_grad():
        loss, grads = gf(model, x, y)
        mx.eval(loss, grads)

    def seg_fwd():
        loss = model.loss(x, y)
        mx.eval(loss)

    SEG = [("fwd+loss", seg_fwd), ("grad", seg_grad), ("полный", seg_full)]

    def measure(fn, steps=2):
        fn()                                   # прогрев сегмента
        mx.clear_cache()
        mx.reset_peak_memory()
        for _ in range(steps):
            fn()
        return mx.get_peak_memory() / 1e6

    sw0 = swap_mb()
    res = {}
    for rnd in range(ROUNDS):
        mx.clear_cache()
        res.setdefault("resident", []).append(mx.get_active_memory() / 1e6)
        for name, fn in SEG:
            res.setdefault(name, []).append(measure(fn))
        # аблации полного шага
        mod.wkv7 = wkv_stub
        res.setdefault("полный/без WKV", []).append(measure(seg_full))
        mod.wkv7 = wkv_real
        rl.RwkvqSymLinear._dequant_w = dq_stub
        res.setdefault("полный/без декв", []).append(measure(seg_full))
        rl.RwkvqSymLinear._dequant_w = dq_real
        type(model).loss = loss_sum
        res.setdefault("полный/без CE", []).append(measure(seg_full))
        type(model).loss = loss_real
        rl.NOCAST = True
        res.setdefault("полный/nocast", []).append(measure(seg_full))
        rl.NOCAST = False
        print("  раунд %d готов" % rnd, flush=True)
    sw1 = swap_mb()

    order = ["resident", "fwd+loss", "grad", "полный",
             "полный/без WKV", "полный/без декв", "полный/без CE",
             "полный/nocast"]
    print("\n%-18s%12s%12s%14s" % ("сегмент", "пик МБ", "разброс",
                                    "Δ к полному"))
    full = np.median(res["полный"])
    for k in order:
        v = np.array(res[k])
        print("%-18s%12.0f%12.0f%14s"
              % (k, np.median(v), v.max() - v.min(),
                 "%+.0f" % (np.median(v) - full) if k != "полный" else "--"))
    print("\nтранзиент полного шага над resident: %.0f МБ"
          % (full - np.median(res["resident"])))
    print("статьи (полный минус абляция): WKV %.0f, деквант %.0f, CE %.0f, "
          "fp32-хвост (nocast) %.0f МБ"
          % (full - np.median(res["полный/без WKV"]),
             full - np.median(res["полный/без декв"]),
             full - np.median(res["полный/без CE"]),
             np.median(res["полный/nocast"]) - full))
    print("своп %.0f -> %.0f МБ (%s)"
          % (sw0, sw1, "ок" if sw1 - sw0 < 64 else "РОС -- пики занижены"))


if __name__ == "__main__":
    main()
