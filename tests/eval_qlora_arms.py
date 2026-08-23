"""ЗНАЧИМОСТЬ РАЗНИЦЫ МЕЖДУ ПЛЕЧАМИ: по-оконный NLL и парный бутстрэп.

`bench_qlora_arms` сохранял только СРЕДНИЙ held-out лосс, и на нём разницу
плеч в 0.003 наты нельзя объявить ни значимой, ни шумом -- сказать про неё
нечего вообще. Здесь та же оценка считается ПО ОКНАМ, и разница двух плеч
получает 95% интервал.

Бутстрэп ПАРНЫЙ, по окнам корпуса: оба плеча прогнаны по ОДНОМУ И ТОМУ ЖЕ
тексту, поэтому разброс самого корпуса (одни окна просто труднее других)
сокращается. Непарный мерил бы дисперсию корпуса вместо дисперсии эффекта
и объявил бы шумом почти всё -- ровно то, ради чего парный бутстрэп заведён
в `ablate_emb_head`.

База собирается ТЕМ ЖЕ `build()`, что и в бенчмарке (импортом, а не копией:
два списка разъехались бы, и «разница плеч» оказалась бы разницей сборок).
Адаптеры берутся из файлов, сохранённых бенчмарком.

    python eval_qlora_arms.py <bf16|reduction|int8>   # считает по-оконный NLL
    python eval_qlora_arms.py --report                # сводка и интервалы
"""
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

MODE = sys.argv[1] if len(sys.argv) > 1 else "--report"
N_EVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 64
T = 512
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/sr_qlora_%d.npz" % T)
# Сид-суффикс для замера разброса по сидам (QLORA_SEED_TAG, см. bench).
SEED_TAG = os.environ.get("QLORA_SEED_TAG", "")
OUT = os.path.expanduser(
    "~/Develop/WKV-kvant/qlora_nll_%s%s.json" % ("%s", SEED_TAG))


def per_window(arm):
    import mlx.core as mx
    sys.argv = ["x", arm]                 # bench-модуль читает argv на импорте
    import bench_qlora_arms as B
    from rwkv_metal.lora import load_lora

    d = np.load(CORPUS)
    ev = d["eval"][:N_EVAL]
    xe = mx.array(ev[:, :-1].astype(np.int32))
    ye = mx.array(ev[:, 1:].astype(np.int32))

    mx.random.seed(20260818)
    model, tag = B.build(arm)
    model._grad_ckpt = True

    def nlls():
        out = []
        for i in range(xe.shape[0]):
            l = model.loss(xe[i:i + 1], ye[i:i + 1]).astype(mx.float32)
            mx.eval(l)
            out.append(float(l))
        return out

    before = nlls()
    ad = "/tmp/qlora_arm_%s%s.safetensors" % (arm, SEED_TAG)
    assert os.path.exists(ad), "нет адаптеров %s" % ad
    load_lora(model, ad)
    after = nlls()

    # КОНТРОЛЬ: адаптеры обязаны что-то изменить. Без него молчаливый
    # промах загрузки дал бы before == after и «плечи неразличимы»
    # (ровно ловушка, на которой погорела аблация фьюза).
    same = sum(1 for a, b in zip(before, after) if a == b)
    res = {"arm": arm, "tag": tag, "n": len(before),
           "nll_before": before, "nll_after": after,
           "mean_before": float(np.mean(before)),
           "mean_after": float(np.mean(after)),
           "identical_windows": same}
    with open(OUT % arm, "w") as f:
        json.dump(res, f, ensure_ascii=False)
    print("%s: %.5f -> %.5f (ppl %.4f -> %.4f), совпавших окон %d/%d %s"
          % (arm, res["mean_before"], res["mean_after"],
             np.exp(res["mean_before"]), np.exp(res["mean_after"]),
             same, len(before),
             "OK" if same == 0 else "!!! АДАПТЕРЫ НЕ ПРИМЕНИЛИСЬ"))


def boot(a, b, iters=20000):
    """Парный бутстрэп разности средних по окнам."""
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    rs = np.random.RandomState(4242)
    idx = rs.randint(0, len(d), size=(iters, len(d)))
    s = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(s, 2.5)), \
        float(np.percentile(s, 97.5))


def report():
    arms, data = [], {}
    for a in ("bf16", "reduction", "reduction_lora8", "reduction_dense",
              "int8", "int8full"):
        p = OUT % a
        if os.path.exists(p):
            data[a] = json.load(open(p))
            arms.append(a)
    if not arms:
        raise SystemExit("нет ни одного qlora_nll_*.json")

    print("%-17s%12s%12s%12s%12s" % ("плечо", "до", "после", "Δ обучения",
                                      "ppl после"))
    for a in arms:
        r = data[a]
        print("%-17s%12.5f%12.5f%12.5f%12.4f"
              % (a, r["mean_before"], r["mean_after"],
                 r["mean_after"] - r["mean_before"], np.exp(r["mean_after"])))

    # Все пары при шести плечах -- тридцать строк, в которых тонет то, ради
    # чего замер ставился. Печатаются: каждое плечо против опорного
    # (`reduction`, нынешний пресет) и отдельно пары, разделяющие ГИПОТЕЗЫ.
    REF = "reduction" if "reduction" in data else arms[0]
    pairs = [(a, REF) for a in arms if a != REF]
    for extra in (("int8full", "reduction_dense"),
                  ("int8full", "int8"),
                  ("reduction_lora8", "reduction_dense"),
                  ("bf16", "int8full")):
        if extra[0] in data and extra[1] in data and extra not in pairs:
            pairs.append(extra)

    print("\nпарные разности (95% CI, бутстрэп по окнам, 20000 итераций):")
    for x, y in pairs:
        for key, name in (("nll_before", "до обучения"),
                          ("nll_after", "после обучения")):
            m, lo, hi = boot(data[x][key], data[y][key])
            sig = "значимо" if (lo > 0) == (hi > 0) else "НЕТ"
            print("  %-16s - %-16s %-14s %+.5f [%+.5f; %+.5f]  %s"
                  % (x, y, name, m, lo, hi, sig))
    print("\nположительная разность = ПЕРВОЕ плечо хуже (лосс больше).")
    print("окон в оценке: %d" % data[arms[0]]["n"])


if __name__ == "__main__":
    if MODE == "--report":
        report()
    else:
        per_window(MODE)
