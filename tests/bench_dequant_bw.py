"""ПОЛОСА ДЕКВАНТА sym-БАЗЫ: сколько стоит fp32-перекладка (23.08).

Подозрение: `RwkvqSymLinear._dequant_w` = кернель(fp32) + `.astype(bf16)`
-- то есть на каждый проход пишется плотный fp32 (5.9 ГБ на модель),
перечитывается и перезаписывается bf16. Алгоритмически достаточно
прочитать ~1.2 ГБ кодов и записать 2.96 ГБ bf16. Этот бенч меряет
обе ветки на ВСЕХ sym-тензорах файла и печатает достигнутые ГБ/с против
обоих вариантов учёта трафика (закон 29: сначала говорим, куда врёт --
изолированный замер кернеля НЕ включает наложение с матмулами реального
шага, так что выигрыш в шаге может быть МЕНЬШЕ напечатанного).

    python tests/bench_dequant_bw.py [rwkvq] [раундов]
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx                      # noqa: E402
import numpy as np                         # noqa: E402

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/reduction_new.rwkvq"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def swap_mb():
    out = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True,
                         env={**os.environ, "LC_ALL": "C"}).stdout
    p = out.replace("=", " ").split()
    return float(p[p.index("used") + 1].rstrip("M"))


from rwkv_quant.formats import codec       # noqa: E402
from rwkv_metal.lora.rwkvq_linear import RwkvqSymLinear  # noqa: E402

manifest, buf = codec.open_rwkvq(os.path.expanduser(PATH))
keys = [k for k, m in manifest["tensors"].items()
        if m.get("kind") == "sym" and m["shape"][1] % 256 == 0]
n_par = sum(int(np.prod(manifest["tensors"][k]["shape"])) for k in keys)
bits = {manifest["tensors"][k]["bits"] for k in keys}
q_bytes = sum(int(np.prod(manifest["tensors"][k]["shape"]))
              * manifest["tensors"][k]["bits"] / 8 for k in keys)
print("sym-тензоров %d, параметров %.2f млрд, битности %s, кодов %.2f ГБ"
      % (len(keys), n_par / 1e9, sorted(bits), q_bytes / 1e9), flush=True)

lins = [RwkvqSymLinear.from_sidecar(PATH, k) for k in keys]
mx.eval([l._sym.parameters() for l in lins] if hasattr(lins[0]._sym, "parameters")
        else [])
mx.clear_cache()

# трафик на ОДИН проход по всем тензорам:
TR_AF = q_bytes + 4 * n_par            # кернель fp32: чтение кодов + запись fp32
TR_RT = TR_AF + 4 * n_par + 2 * n_par  # + перечитать fp32 + записать bf16
TR_BF = q_bytes + 2 * n_par            # целевое: чтение кодов + запись bf16


def run(variant, rounds=ROUNDS):
    ts = []
    for _ in range(rounds):
        mx.clear_cache()
        t0 = mx.eval_synchronize() if hasattr(mx, "eval_synchronize") else None
        import time
        t0 = time.perf_counter()
        for l in lins:
            if variant == "chain":
                w = l._sym._dequant_w(mx.float32).astype(mx.bfloat16)
            elif variant == "fp32":
                w = l._sym._dequant_w(mx.float32)
            elif variant == "direct":
                w = l._sym._dequant_w(mx.bfloat16)
            mx.eval(w)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


sw0 = swap_mb()
t_fp32 = run("fp32")
t_chain = run("chain")
t_direct = run("direct")
sw1 = swap_mb()
for name, t, tr in (("кернель fp32", t_fp32, TR_AF),
                    ("цепочка fp32+astype(bf16)", t_chain, TR_RT),
                    ("[цель] прямая bf16 (пол)", TR_BF / 104e9, TR_BF)):
    if isinstance(t, float):
        print("%-26s %7.1f мс/проход, %.1f ГБ/с (трафик %.1f ГБ)"
              % (name, t * 1e3, tr / t / 1e9, tr / 1e9))
    else:
        print("%-26s пол %.0f мс при 104 ГБ/с" % (name, t * 1e3))
print("прямая bf16             %7.1f мс/проход, %.1f ГБ/с (трафик %.1f ГБ)"
      % (t_direct * 1e3, TR_BF / t_direct / 1e9, TR_BF / 1e9))
print("перекладка astype стоит %.1f мс (%.0f%% цепочки)"
      % ((t_chain - t_fp32) * 1e3, 100 * (t_chain - t_fp32) / t_chain))
print("своп %.0f -> %.0f МБ" % (sw0, sw1))


# --- СВИП РАЗМЕРА THREADGROUP (закон 24: чередование внутри раунда) ------
def sweep_tg(tgs=(128, 256, 512, 1024), rounds=4):
    import time
    res = {tg: [] for tg in tgs}
    for _ in range(rounds):
        for tg in tgs:
            os.environ["RWKVQ_DQ_TG"] = str(tg)
            mx.clear_cache()
            t0 = time.perf_counter()
            for l in lins:
                w = l._sym._dequant_w(mx.bfloat16)
                mx.eval(w)
            res[tg].append(time.perf_counter() - t0)
    del os.environ["RWKVQ_DQ_TG"]
    print("\nсвип RWKVQ_DQ_TG (прямая bf16, проход по всем тензорам):")
    med = {tg: float(np.median(v)) for tg, v in res.items()}
    best = min(med, key=med.get)
    for tg in tgs:
        print("  tg=%-5d %7.1f мс  (%.1f ГБ/с, %+.1f%% к лучшему)"
              % (tg, med[tg] * 1e3, TR_BF / med[tg] / 1e9,
                 100 * (med[tg] / med[best] - 1)))


# разложение по битностям: у 8-битного кернеля НЕТ распаковки -- если он
# тоже ~50 ГБ/с, лимит в записи/чтении; если быстрый, лимит в целочисленной
# распаковке 6-битного.
import time as _t
for b in (6, 8):
    sub = [l for l in lins if manifest["tensors"][l._key]["bits"] == b] if hasattr(lins[0], "_key") else None
    if sub is None:
        sub = [l for l, k in zip(lins, keys) if manifest["tensors"][k]["bits"] == b]
    n = sum(int(np.prod(manifest["tensors"][k]["shape"]))
            for k in keys if manifest["tensors"][k]["bits"] == b)
    qb = sum(int(np.prod(manifest["tensors"][k]["shape"])) * b / 8
             for k in keys if manifest["tensors"][k]["bits"] == b)
    ts = []
    for _ in range(3):
        mx.clear_cache(); t0 = _t.perf_counter()
        for l in sub:
            w = l._sym._dequant_w(mx.bfloat16); mx.eval(w)
        ts.append(_t.perf_counter() - t0)
    m = float(np.median(ts))
    print("bits=%d: %d тензоров, %.2f млрд пар-ов, %6.1f мс, %.1f ГБ/с"
          % (b, len(sub), n/1e9, m*1e3, (qb + 2*n)/m/1e9))

sweep_tg()


# --- A/B СКАЛЯРНАЯ/ВЕКТОРНАЯ bits==8, ЧЕРЕДОВАНИЕМ (закон 24) -----------
def ab_vec(rounds=5):
    import time
    sub8 = [l for l, k in zip(lins, keys) if manifest["tensors"][k]["bits"] == 8]
    res = {"вектор": [], "скаляр": []}
    for _ in range(rounds):
        for name, v in (("вектор", "1"), ("скаляр", "0")):
            os.environ["RWKVQ_DQ_VEC"] = v
            mx.clear_cache()
            t0 = time.perf_counter()
            for l in sub8:
                w = l._sym._dequant_w(mx.bfloat16); mx.eval(w)
            res[name].append(time.perf_counter() - t0)
    del os.environ["RWKVQ_DQ_VEC"]
    med = {k: float(np.median(v)) for k, v in res.items()}
    print("\nA/B bits==8 (чередование, %d раундов):" % rounds)
    for k in ("вектор", "скаляр"):
        print("  %-7s %6.1f мс" % (k, med[k] * 1e3))
    print("  отношение скаляр/вектор: %.3f" % (med["скаляр"] / med["вектор"]))


ab_vec()
