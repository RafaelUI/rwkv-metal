"""QLoRA-БЕНЧМАРК, ТРИ ПЛЕЧА: bf16-LoRA / QLoRA на REDUCTION / QLoRA на int8.

Записанное «REDUCTION ведёт себя лучше int8» до сих пор было ссылкой на
владельца проекта, а не числом. Здесь оно становится числом: три плеча
учатся НА ОДНОМ И ТОМ ЖЕ потоке батчей, в одном и том же порядке, с одним
сидом инициализации адаптеров, и меряются по held-out лоссу на сербской
википедии.

    bf16      -- база в bf16, адаптеры поверх. Верхняя граница качества.
    reduction -- QLoRA поверх `.rwkvq` (нынешний пресет REDUCTION).
    int8      -- QLoRA поверх штатного `mx.quantize` affine, 8 бит, gs=64.

ПОЧЕМУ ПО ПРОЦЕССУ НА ПЛЕЧО (закон 2, и ослабленный закон 1). Три модели
1.5B в одном процессе -- это 3.05 + ~1.6 + ~1.5 ГБ базы плюс по своему
воркспейсу на каждую; на 16 ГБ это своп, а своп делает недействительным
любой замер (закон 11). Качество от этого не страдает -- лосс не зависит
от того, кто ещё живёт в процессе. СКОРОСТЬ страдает: межпроцессный
разброс втрое-вчетверо больше внутрипроцессного (закон 24), поэтому
мс/шаг здесь -- порядок величины, а не точное отношение.

ПОЧЕМУ СЕРБСКИЙ (закон 9). Домен обязан быть тем, где база слаба, иначе
дельты плеч утонут в шуме. Сербский в этом проекте -- канарейка: он
ломался и от `small=8`, и от `proj=asym_sb6_search` на 2.9B.

ЧТО ЗДЕСЬ НЕ МЕРЯЕТСЯ. Пик памяти процесса -- снаружи, `/usr/bin/time -l`
(закон 22); внутренний `mx.get_peak_memory` печатается рядом, но он
аллокаторный и врёт в плюс на накладные (закон 11).

    /usr/bin/time -l python bench_qlora_arms.py \\
        <bf16|reduction|reduction_lora8|reduction_dense|int8|int8full> \\
        [шагов] [T] [eval_окон]
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ARM = sys.argv[1] if len(sys.argv) > 1 else "reduction"
# РАЗБРОС ПО СИДАМ (приоритет 1 от 22.08): интервалы бутстрэпа в плечах
# меряют дисперсию КОРПУСА, а не обучения. Сид здесь меняет И инициализацию
# адаптеров, И порядок батчей -- это и есть «другой прогон обучения».
# Умолчание -- сид записи 18.08, чтобы старые JSON продолжали воспроизводиться.
SEED = int(os.environ.get("QLORA_SEED", "20260818"))
ORDER_SEED = int(os.environ.get("QLORA_ORDER", str(SEED)))
SEED_TAG = os.environ.get("QLORA_SEED_TAG", "")
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
T = int(sys.argv[3]) if len(sys.argv) > 3 else 512
N_EVAL = int(sys.argv[4]) if len(sys.argv) > 4 else 64
EVAL_EVERY = 50
RANK = 16
LR = 1e-4

CORPUS = os.path.expanduser("~/Develop/WKV-kvant/sr_qlora_%d.npz" % T)
RWKVQ = os.environ.get("RWKVQ_BASE", "/tmp/reduction_new.rwkvq")
RWKVQ_L8 = os.environ.get("RWKVQ_L8", "/tmp/reduction_lora8.rwkvq")
PTH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
OUT = os.path.expanduser("~/Develop/WKV-kvant/qlora_arms_%s%s.json" % (ARM, SEED_TAG))


def swap_mb():
    out = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True,
                         env={**os.environ, "LC_ALL": "C"}).stdout
    p = out.replace("=", " ").split()
    return float(p[p.index("used") + 1].rstrip("M"))


# ВЕТКИ TMix -- это обычные `nn.Linear`, и `nn.quantize` их достаёт. В
# BIG_QUANT_TARGETS их нет, поэтому плечо int8 оставляло их плотными, тогда
# как в `.rwkvq` они квантованы ВСЕ (144 тензора asym@6 у w/a/v и 48 rtn@8
# у g). Это и был главный конфаунд первого захода.
BRANCHES = ("w_lora_A", "w_lora_B", "a_lora_A", "a_lora_B",
            "v_lora_A", "v_lora_B", "g_lora_A", "g_lora_B")


def _quantize_branches(model, bits=8, group_size=32):
    """Квантует LoRA-ветки TMix и ПРОВЕРЯЕТ, что они действительно квантованы.

    ПОЧЕМУ gs=32, А НЕ 64. `nn.quantize` группирует вдоль ВХОДНОЙ оси
    матмула. У `*_lora_A` вход 2048 -- делится на что угодно. У `*_lora_B`
    вход равен рангу: 96 у w и a, 64 у v, 256 у g. Шестьдесят четыре НЕ
    делит 96, тридцать два делит все четыре -- ровно тот же вывод, что
    сделан на инференсной стороне rwkv-quant 16.08 («up-проекции gs=32,
    делит 96, 64 и 256»). Именно кратность и была причиной, по которой
    `finetune.py` держал ветки плотными, а не какой-то запрет.

    Контроль обязателен: без него молчаливый промах предиката дал бы
    плечо, неотличимое от прежнего int8, и «разницы нет» читалось бы как
    результат, а не как несработавшая правка.
    """
    import mlx.nn as _nn

    def targets():
        out = []
        for blk in model.blocks:
            for nm in BRANCHES:
                m = getattr(blk.tmix, nm, None)
                if m is not None:
                    out.append((blk, nm, m))
        return out

    before = targets()
    n_want = sum(1 for _, _, m in before if hasattr(m, "to_quantized"))

    def pred(path, module):
        return hasattr(module, "to_quantized") and path.endswith(BRANCHES)

    _nn.quantize(model, group_size=group_size, bits=bits, class_predicate=pred)
    mx.eval(model.parameters())
    mx.clear_cache()

    n_got = sum(1 for _, _, m in targets()
                if type(m).__name__.startswith("Quantized"))
    assert n_got == n_want and n_want > 0, (
        "ветки НЕ квантовались: заказано %d, получилось %d" % (n_want, n_got))
    return n_got


def _quantize_incremental(model, bits, group_size):
    """`nn.quantize` ПО БЛОКУ, а не по всей модели сразу.

    Штатный `quantize_base_model` зовёт `nn.quantize` на всю модель: пока
    он идёт, живы И плотные копии всех матриц, И квантованные. На 1.5B это
    3.05 ГБ поверх уже отображённого файла, и машина уходит в своп прямо на
    сборке (замерено: своп 0.7 -> 2.6 ГБ, плечо не успело дойти до первого
    шага). Поблочно лишняя копия ограничена ОДНИМ блоком.

    Это тот же приём и та же причина, что у закона 20 (полосы строк в
    грид-поиске) и у нарезки декванта при загрузке: считать надо не
    скорость поэлементной математики, а число полноразмерных копий,
    которые она держит одновременно.
    """
    import mlx.nn as _nn
    from rwkv_metal.lora import BIG_QUANT_TARGETS

    def pred(path, module):
        return hasattr(module, "to_quantized") and any(
            path.endswith(t) for t in BIG_QUANT_TARGETS)

    n = 0
    for i, blk in enumerate(model.blocks):
        _nn.quantize(blk, group_size=group_size, bits=bits,
                     class_predicate=pred)
        mx.eval(blk.parameters())
        mx.clear_cache()
        n += 1
    # head/emb живут вне blocks -- их перечисляет тот же предикат, но
    # обойти их надо отдельно, иначе они молча останутся плотными
    # (закон 23: список мест перечисляется, а не подразумевается).
    for name in ("head", "emb"):
        mod = getattr(model, name, None)
        if mod is not None and hasattr(mod, "to_quantized"):
            _nn.quantize(model, group_size=group_size, bits=bits,
                         class_predicate=lambda p, m, _n=name: p == _n
                         and hasattr(m, "to_quantized"))
            mx.eval(model.parameters())
            mx.clear_cache()
    return n


def build(arm):
    """Три плеча. ОДИНАКОВЫ: ранг, alpha, цели, сид адаптеров, покрытие
    квантованием (проекции tmix + cmix + head). Различается РОВНО база."""
    from rwkv_metal.lora import add_lora, load_rwkvq_model
    from rwkv_metal.model.convert import load_pretrained

    if arm in ("reduction", "reduction_lora8"):
        path = RWKVQ if arm == "reduction" else RWKVQ_L8
        model, cfg, info = load_rwkvq_model(path, rank=RANK, verbose=False)
        return model, "rwkvq:%s" % os.path.basename(path)

    if arm == "reduction_dense":
        # ПОТОЛОК ТОГО, ЧТО МОЖНО ВЫИГРАТЬ НА ВЕТКАХ: база из `.rwkvq`, а
        # ветки -- исходные bf16 из .pth. Если это плечо сравняется с
        # честным int8, разрыв целиком в ветках и сетка базы ни при чём.
        #
        # Донор строится ТЕМ ЖЕ конвертером, что и плечо bf16, поэтому
        # значения веток совпадают с ним по построению -- переносить надо
        # веса, а не пересчитывать. Донор удаляется ДО сборки rwkvq-модели,
        # иначе в памяти одновременно живут 3.05 ГБ плотной базы и 1.5 ГБ
        # квантованной.
        out = load_pretrained(PTH, verbose=False)
        donor_model = out[0] if isinstance(out, tuple) else out
        assert donor_model is not None
        donor = []
        for blk in donor_model.blocks:
            d = {}
            for nm in BRANCHES:
                m = getattr(blk.tmix, nm, None)
                if m is None:
                    continue
                e = {"weight": mx.array(m.weight)}
                b = getattr(m, "bias", None)
                if b is not None:
                    e["bias"] = mx.array(b)
                d[nm] = e
            donor.append(d)
        mx.eval([v for d in donor for e in d.values() for v in e.values()])
        del donor_model, out
        mx.clear_cache()

        model, cfg, info = load_rwkvq_model(RWKVQ, rank=RANK, verbose=False)
        moved = same = 0
        for blk, d in zip(model.blocks, donor):
            for nm, e in d.items():
                m = getattr(blk.tmix, nm)
                delta = float(mx.max(mx.abs(m.weight.astype(mx.float32)
                                            - e["weight"].astype(mx.float32))))
                m.weight = e["weight"]
                if "bias" in e and getattr(m, "bias", None) is not None:
                    m.bias = e["bias"]
                moved += 1
                same += 1 if delta == 0.0 else 0
        mx.eval(model.parameters())
        mx.clear_cache()
        # Контроль: деквантованная ветка НЕ может совпасть с исходной bf16
        # побитово. Если совпала -- либо ветки в файле не квантованы, либо
        # перенос не туда, и плечо мерит само себя.
        assert moved > 0 and same == 0, (
            "перенос веток подозрителен: перенесено %d, совпало побитово %d"
            % (moved, same))
        return model, "rwkvq:%s + ветки bf16 из .pth (%d матриц)" % (
            os.path.basename(RWKVQ), moved)

    out = load_pretrained(PTH, verbose=False)
    model = out[0] if isinstance(out, tuple) else out
    assert model is not None, "load_pretrained вернул None -- конверсия грязная"
    mx.eval(model.parameters())
    mx.clear_cache()
    if arm == "bf16":
        add_lora(model, rank=RANK, quantize_base=0)
        mx.eval(model.parameters())
        return model, "bf16:%s" % os.path.basename(PTH)
    if arm in ("int8", "int8full"):
        # ПОРЯДОК ВАЖЕН и он документирован в quantize_base_model:
        # сначала большие замороженные матрицы, потом add_lora квантует
        # целевые проекции. Наоборот -- часть матриц осталась бы плотной,
        # и плечо мерило бы другую базу.
        nb = _quantize_incremental(model, bits=8, group_size=64)
        add_lora(model, rank=RANK, quantize_base=8, q_group_size=64)
        tag = "int8:affine gs=64 (поблочно, %d блоков)" % nb
        if arm == "int8full":
            nbr = _quantize_branches(model, bits=8, group_size=32)
            tag += " + ВЕТКИ gs=32 (%d матриц)" % nbr
        mx.eval(model.parameters())
        mx.clear_cache()
        return model, tag
    raise SystemExit("плечо: bf16 | reduction | reduction_lora8 | "
                     "reduction_dense | int8 | int8full")


def main():
    assert os.path.exists(CORPUS), "нет корпуса, сперва build_sr_corpus.py"
    d = np.load(CORPUS)
    tr, ev = d["train"], d["eval"][:N_EVAL]
    print("плечо %s | корпус %s train %s eval %s"
          % (ARM, os.path.basename(CORPUS), tr.shape, ev.shape), flush=True)

    mx.random.seed(SEED)            # сид адаптеров; см. QLORA_SEED выше
    swb = swap_mb()
    t0 = time.time()
    model, tag = build(ARM)
    model._grad_ckpt = True
    print("база: %s, сборка %.1f с, пик аллокатора %.2f ГБ, своп %.0f -> %.0f МБ"
          % (tag, time.time() - t0, mx.get_peak_memory() / 1e9,
             swb, swap_mb()), flush=True)
    if os.environ.get("QLORA_BUILD_ONLY"):
        # Сборка -- отдельная статья памяти, и мерить её надо ДО того, как
        # ставить долгий прогон (закон 22). Пик процесса снимает
        # /usr/bin/time -l снаружи.
        print("QLORA_BUILD_ONLY -- выходим до обучения")
        return

    xe = mx.array(ev[:, :-1].astype(np.int32))
    ye = mx.array(ev[:, 1:].astype(np.int32))

    def evaluate():
        """held-out лосс по одному окну за раз: батч из 64 окон на T=512
        даёт логиты [64, 511, 65536] -- это 8 ГБ только на них."""
        tot = 0.0
        for i in range(xe.shape[0]):
            l = model.loss(xe[i:i + 1], ye[i:i + 1]).astype(mx.float32)
            mx.eval(l)
            tot += float(l)
        return tot / xe.shape[0]

    ev0 = evaluate()
    print("held-out лосс ДО обучения: %.5f (ppl %.4f)"
          % (ev0, float(np.exp(ev0))), flush=True)

    from rwkv_metal.lora import LoRAConfig, finetune
    cfg = LoRAConfig(lr=LR, max_steps=STEPS, grad_accum=1, log_every=10,
                     grad_checkpoint=True,
                     adapter_path="/tmp/qlora_arm_%s%s.safetensors" % (ARM, SEED_TAG))

    # Поток батчей: порядок ФИКСИРОВАН и одинаков у трёх плеч.
    order = np.random.RandomState(ORDER_SEED).permutation(len(tr))

    def batches():
        for i in range(STEPS):
            row = tr[order[i % len(order)]]
            yield (mx.array(row[None, :-1].astype(np.int32)),
                   mx.array(row[None, 1:].astype(np.int32)))

    hist, times, evals = [], [], [(0, ev0)]
    last = [time.perf_counter()]
    sw0 = swap_mb()

    def on_step(step, loss, peak):
        now = time.perf_counter()
        times.append(now - last[0])
        last[0] = now
        hist.append(loss)
        if (step + 1) % EVAL_EVERY == 0 and step + 1 < STEPS:
            e = evaluate()
            evals.append((step + 1, e))
            print("    held-out на шаге %d: %.5f" % (step + 1, e), flush=True)
            last[0] = time.perf_counter()      # eval не входит в мс/шаг

    finetune(model, batches(), cfg, on_step=on_step)
    ev1 = evaluate()
    evals.append((STEPS, ev1))
    sw1 = swap_mb()

    # первый шаг выбрасывается: в нём трассировка и прогрев (закон 26 по духу)
    tms = np.array(times[1:]) * 1e3
    res = {
        "arm": ARM, "tag": tag, "seed": SEED, "order_seed": ORDER_SEED,
        "lr": LR, "n_eval": int(xe.shape[0]),
        "eval_loss_before": ev0, "eval_loss_after": ev1,
        "eval_ppl_before": float(np.exp(ev0)),
        "eval_ppl_after": float(np.exp(ev1)),
        "eval_curve": evals,
        "train_loss_first50": float(np.median(hist[:50])),
        "train_loss_last50": float(np.median(hist[-50:])),
        "ms_per_step_median": float(np.median(tms)),
        "ms_per_step_spread_pct": float((tms.max() - tms.min())
                                        / np.median(tms) * 100),
        "mx_peak_gb": float(mx.get_peak_memory() / 1e9),
        "swap_mb": [sw0, sw1],
        "train_loss": hist,
    }
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    print("\n=== плечо %s ===" % ARM)
    print("held-out лосс %.5f -> %.5f (ppl %.4f -> %.4f), Δ %.5f"
          % (ev0, ev1, np.exp(ev0), np.exp(ev1), ev1 - ev0))
    print("train лосс: первые 50 %.4f -> последние 50 %.4f"
          % (res["train_loss_first50"], res["train_loss_last50"]))
    print("мс/шаг %.0f (разброс %.1f%%), пик аллокатора %.2f ГБ"
          % (res["ms_per_step_median"], res["ms_per_step_spread_pct"],
             res["mx_peak_gb"]))
    print("своп %.0f -> %.0f МБ (%s)"
          % (sw0, sw1, "ок для замера качества" if sw1 - sw0 < 2048
             else "СЛИШКОМ МНОГО"))
    print("JSON: %s" % OUT)


if __name__ == "__main__":
    main()
