"""СВИП РАНГОВ QLoRA: скорость шага и ПАМЯТЬ ПО СТАТЬЯМ (1.5B, sym-база).

ЧТО МЕРЯЕТСЯ И ПОЧЕМУ ИМЕННО ТАК

Ранги чередуются В ОДНОМ ПРОЦЕССЕ: база строится РАЗ, а адаптеры
переобёртываются поверх тех же самых frozen-модулей. Три процесса дали бы
межпроцессный разброс (втрое-вчетверо больше внутрипроцессного) плюс
прогретую машину на последнем прогоне -- то есть нарисовали бы зависимость
от ранга там, где её может не быть. Цена решения: база в памяти одна на
все ранги, и статью «база» надо читать как общую, а не как долю ранга.

ПАМЯТЬ РАЗЛОЖЕНА НА ЖИВОЕ И ТРАНЗИЕНТ, а не оценена одним числом:

  база          -- буферы RwkvqSymLinear (qblk/qs/d), обходом модели;
  плотное       -- всё остальное в parameters(): emb, нормы, LoRA-ветки;
  адаптеры      -- обучаемые A/B;
  оптимизатор   -- opt.state после первого шага (у AdamW это m и v);
  градиенты     -- дерево grads;
  транзиент     -- пик минус сумма живого. Это активации, пересчёт блоков,
                   логиты и копии в лоссе -- то, что живёт только внутри шага.

Плюс ОДНА АБЛАЦИЯ, которая отделяет машинерию лосса: тот же шаг, но
вместо кросс-энтропии берётся сумма логитов. Логиты при этом считаются
оба раза, значит разность -- это ровно log_softmax и его копии на
[T, 65536], то есть цена рычага «чанкованный лосс».

ЧЕГО ЭТОТ ЗАМЕР НЕ ГОВОРИТ. Пик -- величина процессная, и статьи в нём не
складываются: аллокатор переиспользует освобождённое. Поэтому «транзиент»
здесь -- ВЕРХНЯЯ оценка суммы внутришаговых статей, а не их сумма.

    python bench_qlora_rank.py [ранги через запятую] [T] [раундов]
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
from mlx.utils import tree_flatten  # noqa: E402

RANKS = [int(r) for r in (sys.argv[1] if len(sys.argv) > 1 else "16,32,64").split(",")]
T = int(sys.argv[2]) if len(sys.argv) > 2 else 512
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
STEPS = 3
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


def tree_mb(tree):
    return sum(v.nbytes for _, v in tree_flatten(tree)
               if isinstance(v, mx.array)) / 1e6


def main():
    from rwkv_metal.lora.add_rwkvq import TMIX_TARGETS, _unfreeze_adapters
    from rwkv_metal.lora import load_rwkvq_model
    from rwkv_metal.lora.lora import LoRALinear
    from rwkv_metal.lora.rwkvq_linear import RwkvqSymLinear

    t0 = time.time()
    model, cfg, _ = load_rwkvq_model(PATH, rank=RANKS[0], verbose=False)
    # ЧЕКПОИНТИНГ ВЫСТАВЛЯЕТСЯ ЯВНО И ПЕЧАТАЕТСЯ. Умолчание модели -- НЕ
    # то же, что умолчание тренера (train_lora ставит его сам из
    # LoRAConfig), и первая редакция этого свипа его не тронула: шаг
    # честно мерился БЕЗ чекпоинтинга и пикал 12.4 ГБ вместо 4.3.
    # Разница втрое, а по выводу скрипта не видно ничего.
    model._grad_ckpt = True
    print(f"база собрана за {time.time()-t0:.1f} с, "
          f"пик сборки {mx.get_peak_memory()/1e9:.2f} ГБ, "
          f"grad_checkpoint={getattr(model, '_grad_ckpt', 'НЕТ АТРИБУТА')}")

    # --- живые статьи, не зависящие от ранга ------------------------------
    base_mb = 0.0
    for blk in model.blocks:
        for name in list(TMIX_TARGETS) + ["key", "value"]:
            m = getattr(blk.tmix, name, None) or getattr(blk.cmix, name, None)
            m = getattr(m, "linear", m)
            if isinstance(m, RwkvqSymLinear):
                s = m._sym
                base_mb += (s.qblk.nbytes + s.qs.nbytes + s.d.nbytes) / 1e6
    hm = getattr(model.head, "_sym", None)
    if hm is not None:
        base_mb += (hm.qblk.nbytes + hm.qs.nbytes + hm.d.nbytes) / 1e6

    def rewrap(rank):
        """Переобёртка адаптеров ПОВЕРХ ТЕХ ЖЕ frozen-баз."""
        for blk in model.blocks:
            for name in TMIX_TARGETS:
                m = getattr(blk.tmix, name, None)
                if m is None:
                    continue
                base = getattr(m, "linear", m)
                setattr(blk.tmix, name,
                        LoRALinear(rank=rank, alpha=2.0 * rank,
                                   base_module=base))
        model.freeze()
        _unfreeze_adapters(model)
        mx.eval(model.parameters())
        mx.clear_cache()

    rs = np.random.RandomState(5)
    x = mx.array(rs.randint(1, 60000, size=(1, T)).astype(np.int32))
    y = mx.array(rs.randint(1, 60000, size=(1, T)).astype(np.int32))

    def loss_ce(m, a, b):
        return m.loss(a, b).astype(mx.float32)

    def loss_sum(m, a, b):
        o = m(a)
        o = o[0] if isinstance(o, tuple) else o
        return o.astype(mx.float32).sum()

    res = {}
    for rnd in range(ROUNDS):
        for rank in RANKS:                    # чередование внутри раунда
            for tag, lf in (("ce", loss_ce), ("sum", loss_sum)):
                rewrap(rank)
                gf = nn.value_and_grad(model, lf)
                opt = optim.AdamW(learning_rate=1e-4)
                loss, grads = gf(model, x, y)   # прогрев + формы состояния
                opt.update(model, grads)
                mx.eval(loss, model.state, opt.state)
                mx.clear_cache()
                if hasattr(mx, "reset_peak_memory"):
                    mx.reset_peak_memory()
                sw0 = swap_mb()
                ts = []
                for _ in range(STEPS):
                    t1 = time.time()
                    loss, grads = gf(model, x, y)
                    opt.update(model, grads)
                    mx.eval(loss, model.state, opt.state)
                    ts.append(time.time() - t1)
                k = (rank, tag)
                r = res.setdefault(k, {"t": [], "peak": 0.0, "sw": 0.0})
                r["t"] += ts
                r["peak"] = max(r["peak"], mx.get_peak_memory() / 1e6)
                r["sw"] = max(r["sw"], swap_mb() - sw0)
                print(f"  r={rank} {tag}: {np.median(ts)*1e3:.0f} мс/шаг, "
                      f"пик {mx.get_peak_memory()/1e6:.0f} МБ", flush=True)
                if rnd == 0 and tag == "ce":
                    r["ad"] = tree_mb(model.trainable_parameters())
                    r["opt"] = tree_mb(opt.state)
                    r["gr"] = tree_mb(grads)
                    r["dense"] = tree_mb(model.parameters()) - r["ad"]

    print(f"\nживое, не зависящее от ранга: база {base_mb:.0f} МБ, "
          f"плотное {res[(RANKS[0], 'ce')]['dense']:.0f} МБ")
    print(f"\n{'ранг':>5}{'мс/шаг':>10}{'разброс':>9}{'адапт':>8}{'opt':>8}"
          f"{'град':>8}{'пик МБ':>10}{'транзиент':>11}{'своп':>7}")
    for rank in RANKS:
        r = res[(rank, "ce")]
        live = base_mb + r["dense"] + r["ad"] + r["opt"] + r["gr"]
        med = float(np.median(r["t"])) * 1e3
        sp = (max(r["t"]) - min(r["t"])) / np.median(r["t"]) * 100
        print(f"{rank:>5}{med:>10.0f}{sp:>8.1f}%{r['ad']:>8.0f}{r['opt']:>8.0f}"
              f"{r['gr']:>8.0f}{r['peak']:>10.0f}{r['peak']-live:>11.0f}"
              f"{r['sw']:>7.0f}")
    print("\nаблация лосса (логиты считаются в обоих; разность -- "
          "log_softmax и копии на [T, vocab]):")
    print(f"{'ранг':>5}{'CE мс':>9}{'sum мс':>9}{'дельта':>9}"
          f"{'CE пик':>9}{'sum пик':>9}{'дельта МБ':>11}")
    for rank in RANKS:
        a, b = res[(rank, "ce")], res[(rank, "sum")]
        ta, tb = float(np.median(a["t"])) * 1e3, float(np.median(b["t"])) * 1e3
        print(f"{rank:>5}{ta:>9.0f}{tb:>9.0f}{ta-tb:>9.0f}"
              f"{a['peak']:>9.0f}{b['peak']:>9.0f}{a['peak']-b['peak']:>11.0f}")
    bad = [k for k, v in res.items() if v["sw"] > 0.5]
    print("\nсвоп не рос -- замер валиден" if not bad else
          f"\n*** СВОП РОС в {bad}: скорость недействительна (закон 11) ***")


if __name__ == "__main__":
    main()
