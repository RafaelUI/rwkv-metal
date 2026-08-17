"""РОВНАЯ ТРЕНИРОЧНАЯ НАГРУЗКА ПОД Metal System Trace.

Аналог trace_decode_steady, но для шага обучения: один конфиг, одинаковые
шаги, ничего кроме них. Сборка модели и прогрев происходят ДО того, как
печатается маркер готовности, чтобы трасса цеплялась на устоявшийся режим
и не описывала загрузку.

ОКНО ЗАХВАТА -- ПЕРВЫЕ 20-25 СЕКУНД ПОСЛЕ МАРКЕРА. Дальше безвентиляторный
корпус греется: на декоде записано падение 18.1 -> 25.9 мс/ток к
шестьдесят пятой секунде, то есть трасса начинает описывать перегретую
машину, а не ядро. Скрипт печатает мс/шаг раз в 5 с именно для того,
чтобы дрейф был виден и окно можно было выбрать по числам, а не на глаз.

    python trace_train_steady.py [секунд] [T] [ранг]
    SILENT=1 -- без периодической печати (её ввод-вывод тоже виден в трассе)

Печатает свой PID: цеплять xctrace надо --attach, а не --launch, иначе в
трассу попадут сборка и прогрев.
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

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
T = int(sys.argv[2]) if len(sys.argv) > 2 else 512
RANK = int(sys.argv[3]) if len(sys.argv) > 3 else 16
SILENT = os.environ.get("SILENT") == "1"
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
    import rwkv_metal.model.rwkv7_x070 as _mod

    # ABLATE=wkv -- та же нагрузка с ЗАГЛУШЁННОЙ рекуррентностью.
    # Нужна для проверки гипотезы о дырах в счётчиках: если дыры -- это
    # WKV (2048 потоков независимо от T, счётчикам нечего сэмплировать),
    # они обязаны исчезнуть, а F32-лимитер просесть. Если останутся --
    # виновата диспетчеризация, и лечится она оп-каунтом.
    if os.environ.get("ABLATE") == "wkv":
        def _stub(r, w, k, v, a, b, training=True, state=None,
                  return_state=False):
            return r * 0.5 + v * 0.5 + (k + w + a + b) * 0.0, None
        _mod.wkv7 = _stub

    model, cfg, info = load_rwkvq_model(PATH, rank=RANK, verbose=False)
    model._grad_ckpt = True          # ставится ЯВНО: умолчание модели -- False
    print(f"pid {os.getpid()} | {os.path.basename(PATH)} | T={T} rank={RANK} "
          f"| адаптеров {info['num_lora_adapters']} "
          f"| grad_checkpoint={model._grad_ckpt}"
          f"{' | WKV ЗАГЛУШЁН' if os.environ.get('ABLATE') == 'wkv' else ''}",
          flush=True)

    rs = np.random.RandomState(5)
    x = mx.array(rs.randint(1, 60000, size=(1, T)).astype(np.int32))
    y = mx.array(rs.randint(1, 60000, size=(1, T)).astype(np.int32))

    def loss_fn(m, a, b):
        return m.loss(a, b).astype(mx.float32)

    gf = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=1e-4)

    for _ in range(2):               # прогрев: компиляция и формы состояния
        loss, grads = gf(model, x, y)
        opt.update(model, grads)
        mx.eval(loss, model.state, opt.state)
    mx.synchronize()
    mx.clear_cache()

    sw0 = swap_mb()
    print(f"=== УСТАНОВИВШИЙСЯ РЕЖИМ, цепляйте xctrace ({SECONDS:.0f} с) ===",
          flush=True)
    t0 = last = time.time()
    n = n_last = 0
    times = []
    while time.time() - t0 < SECONDS:
        t1 = time.time()
        loss, grads = gf(model, x, y)
        opt.update(model, grads)
        mx.eval(loss, model.state, opt.state)
        times.append(time.time() - t1)
        n += 1
        if not SILENT and time.time() - last >= 5.0:
            win = times[n_last:]
            print(f"  t={time.time()-t0:5.1f} с  {np.median(win)*1e3:7.0f} мс/шаг"
                  f"  (шагов {len(win)})", flush=True)
            last, n_last = time.time(), n
    sw1 = swap_mb()

    first = times[:max(1, len(times) // 4)]
    print(f"\nвсего шагов {n}")
    print(f"  первая четверть (окно захвата): {np.median(first)*1e3:.0f} мс/шаг")
    print(f"  всё окно целиком:               {np.median(times)*1e3:.0f} мс/шаг")
    print(f"  дрейф: {(np.median(times)/np.median(first)-1)*100:+.1f}%")
    print(f"  своп {sw0:.0f} -> {sw1:.0f} МБ"
          + ("   *** РОС: трасса описывает свопящую машину ***"
             if sw1 > sw0 + 0.5 else ""))


if __name__ == "__main__":
    main()
