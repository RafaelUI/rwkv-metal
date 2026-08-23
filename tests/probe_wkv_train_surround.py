"""ГДЕ ЛЕЖИТ РАЗРЫВ 5.9x: в ядре WKV или вокруг него (forward обучения).

`bench_wkv_train_vs_infer_ab` уже ответил на первую половину вопроса: на
ОДНИХ И ТЕХ ЖЕ входах тренировочное ядро медленнее инференсного всего в
1.09 раза (1.95 против 1.78 мс на слой), то есть 5.9x в ядре нет. Значит
разрыв лежит в ОБВЯЗКЕ тренировочного вызова, и здесь она разбирается
лестницей на НАСТОЯЩЕЙ модели, forward без градиента:

  полный      -- wkv7(training=True) как есть: custom_function, паддинг до
                 CHUNK, срез out[:, :T], astype(fp32) шести входов, четыре
                 выхода (out, h_out, sa_out, h_checkpoints)
  полный+cast -- он же с CAST_WKV_OUTPUT=True: выход приводится к dtype
                 остатка сразу. Разность с полным = цена ХВОСТА TMix в fp32
  ckpt-ядро   -- то же ядро, но ВЫЗВАННОЕ НАПРЯМУЮ: ни custom_function, ни
                 паддинга, ни среза. Разность с полным = цена обвязки
  infer-ядро  -- инференсное ядро на тех же входах: два выхода вместо
                 четырёх. Разность с ckpt-ядром = цена sa_out и h_checkpoints
  заглушка32  -- рекуррентности нет, но выход ТОГО ЖЕ dtype, что у ядра
                 (fp32). Разность с infer-ядром = цена самого скана
  заглушка16  -- заглушка из probe_train_components: выход bf16. Разность с
                 заглушкой32 = снова цена хвоста в fp32, и она обязана
                 совпасть с (полный - полный+cast), иначе разбор неверен

ПОЧЕМУ ЭТО ВАЖНО ДЛЯ ПРЕЖНЕГО ЧИСЛА. `CAST_WKV_OUTPUT = False` в модели,
значит выход ядра остаётся fp32 и ВЕСЬ хвост блока (ln_x, bonus, gate,
o_proj) считается в fp32, тогда как заглушка отдавала bf16 и хвост шёл в
bf16. Аблация «убрать WKV» тем самым меняла ДВЕ вещи разом, и записанные
321 мс -- это скан ПЛЮС удвоенный по трафику хвост.

Аблации подменяют ГЛОБАЛЬ `mod.wkv7` (её модель читает при каждом вызове),
`mx.compile` на тренировочном пути нет -- значит варианты чередуются в одном
процессе (закон 1), порядок внутри раунда РАНДОМИЗИРОВАН (закон 24), окно
короткое (закон 25).

ЧТО ЭТОТ ИНСТРУМЕНТ ЗАВЫШАЕТ, А ЧТО ЗАНИЖАЕТ (закон 29). Аблации НЕ
аддитивны: убирая компонент, меняешь и то, с чем он перекрывался. Он ищет,
на какой ступени лестницы появляются сотни миллисекунд, а не строит бюджет
до миллисекунды.

    python probe_wkv_train_surround.py [T] [раундов] [grad|fwd]
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
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
MODE = sys.argv[3] if len(sys.argv) > 3 else "fwd"
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
    import rwkv_metal.model.rwkv7_x070 as mod
    from rwkv_metal.kernel import wkv7_checkpoint as ck
    from rwkv_metal.kernel.wkv7 import wkv7_infer, CHUNK

    model, cfg, info = load_rwkvq_model(PATH, rank=16, verbose=False)
    model._grad_ckpt = True
    n_layer = len(model.blocks)
    print("T=%d, слоёв %d, режим %s" % (T, n_layer, MODE), flush=True)

    wkv_real = mod.wkv7
    D = 64
    assert T % CHUNK == 0
    N = T // CHUNK

    def wkv_ckpt_raw(r, w, k, v, a, b, training=True, state=None,
                     return_state=False):
        """Ровно то же ядро, но напрямую: без custom_function/паддинга/среза."""
        B, TT, H, _ = r.shape
        h0 = mx.zeros((B, H, D, D), dtype=mx.float32)
        res = ck._get_ckpt_fwd(H, TT)(
            inputs=[x.astype(mx.float32) for x in [r, w, k, v, a, b, h0]],
            grid=(B * H, D, 1), threadgroup=(1, D, 1),
            output_shapes=[(B, TT, H, D), (B, H, D, D), (B, TT, H, D),
                           (B, H, TT // CHUNK, D, D)],
            output_dtypes=[mx.float32] * 4,
        )
        return res[0], None

    def wkv_infer_k(r, w, k, v, a, b, training=True, state=None,
                    return_state=False):
        B, TT, H, _ = r.shape
        h0 = mx.zeros((B, H, D, D), dtype=mx.float32)
        out, _h = wkv7_infer(r, w, k, v, a, b, h0)
        return out, None

    def wkv_stub16(r, w, k, v, a, b, training=True, state=None,
                   return_state=False):
        """Заглушка probe_train_components: выход в dtype ВХОДА (bf16)."""
        return r * 0.5 + v * 0.5 + (k + w + a + b) * 0.0, None

    def wkv_stub32(r, w, k, v, a, b, training=True, state=None,
                   return_state=False):
        """То же, но выход fp32 -- как у настоящего ядра."""
        o = r * 0.5 + v * 0.5 + (k + w + a + b) * 0.0
        return o.astype(mx.float32), None

    # Флаг читается на каждом проходе, поэтому ветка «с приведением» --
    # это подмена ОДНОГО флага в одном процессе (закон 27).
    class _Cast:
        def __init__(self, fn, val):
            self.fn, self.val = fn, val

        def __call__(self, *a, **kw):
            return self.fn(*a, **kw)

    def set_cast(v):
        mod.CAST_WKV_OUTPUT = v

    VAR = [("полный", (wkv_real, False)),
           ("полный+cast", (wkv_real, True)),
           ("ckpt-ядро", (wkv_ckpt_raw, False)),
           ("infer-ядро", (wkv_infer_k, False)),
           ("заглушка32", (wkv_stub32, False)),
           ("заглушка16", (wkv_stub16, False))]
    if MODE == "grad":
        # без VJP у сырых ядер градиента нет -- остаются четыре ступени
        VAR = [("полный", (wkv_real, False)),
               ("полный+cast", (wkv_real, True)),
               ("заглушка32", (wkv_stub32, False)),
               ("заглушка16", (wkv_stub16, False))]

    rs = np.random.RandomState(5)
    x = mx.array(rs.randint(1, 60000, size=(1, T)).astype(np.int32))
    y = mx.array(rs.randint(1, 60000, size=(1, T)).astype(np.int32))

    if MODE == "grad":
        def loss_fn(m, a, b):
            return m.loss(a, b).astype(mx.float32)
        gf = nn.value_and_grad(model, loss_fn)

        def run():
            loss, grads = gf(model, x, y)
            mx.eval(loss, grads)
    else:
        def run():
            mx.eval(model(x))

    # контроль: ступени с настоящим ядром обязаны считать ОДНО И ТО ЖЕ.
    # «полный+cast» -- НЕ обязан: приведение к bf16 меняет числа, ради чего
    # флаг и заведён. Заглушки считают другое по построению.
    outs = {}
    for name, (fn, cast) in VAR:
        mod.wkv7 = fn
        set_cast(cast)
        o = model(x)
        mx.eval(o)
        outs[name] = o
    set_cast(False)
    ref = outs["полный"]
    print("\nконтроль логитов против полного:")
    for name, _ in VAR[1:]:
        d = float(mx.max(mx.abs(outs[name] - ref)) /
                  (mx.max(mx.abs(ref)) + 1e-30))
        strict = name in ("ckpt-ядро", "infer-ядро")
        ok = "OK" if (not strict or d < 1e-3) else "!!! РАЗНЫЕ ВЫЧИСЛЕНИЯ"
        print("  %-12s relmax %.3e %s" % (name, d, ok))
    del outs, ref
    mx.clear_cache()

    acc = {n: [] for n, _ in VAR}
    sw0 = swap_mb()
    order = list(range(len(VAR)))
    rng = random.Random(20260817)

    for rnd_i in range(ROUNDS):
        rng.shuffle(order)
        for idx in order:
            name, (fn, cast) = VAR[idx]
            mod.wkv7 = fn
            set_cast(cast)
            run()                                   # прогрев варианта
            mx.clear_cache()
            for _ in range(STEPS):
                t1 = time.perf_counter()
                run()
                acc[name].append(time.perf_counter() - t1)
        print("  раунд %d (порядок %s)" % (rnd_i, order), flush=True)
    sw1 = swap_mb()
    mod.wkv7 = wkv_real
    set_cast(False)

    full = float(np.median(acc["полный"]))
    print("\n%-14s%10s%10s%13s%13s" % ("вариант", "мс", "разброс",
                                       "статья мс", "на слой мс"))
    for name, _ in VAR:
        m = float(np.median(acc[name]))
        sp = (max(acc[name]) - min(acc[name])) / m * 100
        print("%-14s%10.1f%9.1f%%%13.1f%13.3f"
              % (name, m * 1e3, sp, (full - m) * 1e3, m * 1e3 / n_layer))

    def med(n):
        return float(np.median(acc[n])) * 1e3

    print("\nступени лестницы (разность соседних):")
    for i in range(len(VAR) - 1):
        a, b = med(VAR[i][0]), med(VAR[i + 1][0])
        print("  %-12s -> %-12s  %7.1f мс  (%.2f мс/слой)"
              % (VAR[i][0], VAR[i + 1][0], a - b, (a - b) / n_layer))

    print("\nдва независимых замера ОДНОЙ величины -- цены хвоста в fp32:")
    t1_ = med("полный") - med("полный+cast")
    t2_ = med("заглушка32") - med("заглушка16")
    print("  на настоящем ядре   %7.1f мс" % t1_)
    print("  на заглушке         %7.1f мс" % t2_)
    print("  расхождение         %7.1f мс (%s)"
          % (t1_ - t2_, "сходятся" if abs(t1_ - t2_) < 0.25 * max(abs(t1_), 1)
             else "НЕ сходятся -- разбор неверен"))
    if "infer-ядро" in acc:
        print("\nчистая цена скана (infer-ядро - заглушка32): %.1f мс "
              "(%.2f мс/слой)" % (med("infer-ядро") - med("заглушка32"),
                                  (med("infer-ядро") - med("заглушка32"))
                                  / n_layer))
    print("прежняя аблация (полный - заглушка16): %.1f мс -- вот эти 321"
          % (med("полный") - med("заглушка16")))
    print("\nсвоп %.0f -> %.0f МБ (%s)"
          % (sw0, sw1, "замер валиден" if sw1 - sw0 < 1 else "НЕВАЛИДЕН"))


if __name__ == "__main__":
    main()
