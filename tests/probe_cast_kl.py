"""KL(bf16 ‖ квантованная+каст) ДЛЯ ДВУХ РЫЧАГОВ fp32-ХВОСТА.

Спор, который этот пробник решает: ppl говорит, что CAST_WKV_OUTPUT
УЛУЧШАЕТ квантованную базу (−0.23% на 1.5B, −0.17% на 2.9B, значимо),
а relmax логитов у него впятеро больше, чем у каста в слое (5.7e-2 против
1.06e-2), и на плотной модели знак обратный (+0.043%). Прецедент уже был:
на emb ppl и KL расходились, и правая оказалась KL (для векторных моделей
и генерации меряет распределение, а не вероятность одного токена).

Здесь та же проверка для кастов: KL расхождения с bf16 на ТРЁХ вариантах
квантованной базы (как есть / каст в слое / CAST_WKV_OUTPUT). Если
WKV-каст улучшает ppl за счёт УХУДШЕНИЯ KL -- его выигрыш локален для
корпуса, и «каст в слое» выигрывает как безопасный. Если KL тоже
улучшается -- переоткрывать.

Эталон bf16 считается один раз и лежит в /tmp fp32 (логиты порядка 10,
шаг fp16 сравним с эффектом -- см. ablate_subgroups). Адаптеров в модели
нет смысла избегать: они нулевые при инициализации и ОДИНАКОВЫ во всех
трёх вариантах, а сравниваются варианты между собой.

    python tests/probe_cast_kl.py [nseq]
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))

import mlx.core as mx                      # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt")
PTH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
RWKVQ = os.environ.get("RWKVQ_BASE", "/tmp/reduction_new.rwkvq")
REF = "/tmp/kl_cast_ref_%d.npz" % N


def swap_mb():
    out = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True,
                         env={**os.environ, "LC_ALL": "C"}).stdout
    p = out.replace("=", " ").split()
    return float(p[p.index("used") + 1].rstrip("M"))


import torch                               # noqa: E402
toks = torch.load(CORPUS, weights_only=False)["tokens"][:N]


def seq_logits(model, row):
    x = mx.array(row[:-1].numpy().astype(np.int32)[None, :])
    lg = model(x)
    lg = lg[0] if lg.ndim == 3 else lg
    return lg.astype(mx.float32)


sw0 = swap_mb()
if not os.path.exists(REF):
    import rwkv_metal.model.rwkv7_x070 as m070        # noqa: F401
    from rwkv_metal.model.convert import load_pretrained
    model, cfg = load_pretrained(PTH, verbose=False)
    out = []
    for row in toks:
        lg = seq_logits(model, row)
        mx.eval(lg)
        out.append(np.asarray(lg))
    np.savez(REF, *out)
    del model
    mx.clear_cache()
    print("эталон bf16 записан: %s (%d последовательностей)" % (REF, N),
          flush=True)
else:
    print("эталон bf16 найден: %s" % REF, flush=True)

ref = np.load(REF)
from rwkv_metal.lora import load_rwkvq_model           # noqa: E402
from rwkv_metal.lora import rwkvq_linear as rl         # noqa: E402

model, cfg, info = load_rwkvq_model(RWKVQ, verbose=False)

CLASSES = [c for c in (getattr(rl, "RwkvqLinear", None),
                       getattr(rl, "RwkvqSymLinear", None)) if c is not None]
orig = {c: c.__call__ for c in CLASSES}


def cast_call(self, x):
    w = self._dequant_w()
    return (x.astype(w.dtype) @ w.T).astype(x.dtype)


def set_lincast(on):
    for c in CLASSES:
        c.__call__ = cast_call if on else orig[c]


def set_wkvcast(on):
    import rwkv_metal.model.rwkv7_x070 as m070
    m070.CAST_WKV_OUTPUT = on


def logprobs(lg):
    return lg - mx.logsumexp(lg, axis=-1, keepdims=True)


def run_variant():
    """per-seq KL(bf16 ‖ вариант), наты на токен."""
    kls = []
    for i, row in enumerate(toks):
        p_lg = mx.array(ref["arr_%d" % i])
        q_lg = seq_logits(model, row)
        mx.eval(q_lg)
        lp, lq = logprobs(p_lg), logprobs(q_lg)
        kl = (mx.exp(lp) * (lp - lq)).sum(axis=-1)
        mx.eval(kl)
        kls.append(float(kl.mean()))
    return kls


# КОНТРОЛЬ ВКЛЮЧЕНИЯ: логиты варианта обязаны отличаться от базы
# (иначе «KL не изменился» читался бы как «правка нейтральна»).
x0 = mx.array(toks[0][:-1].numpy().astype(np.int32)[None, :])
set_lincast(False)
set_wkvcast(False)
lg_base0 = seq_logits(model, toks[0])
mx.eval(lg_base0)

res = {}
for name, lc, wc in (("база", False, False),
                     ("каст_в_слое", True, False),
                     ("CAST_WKV_OUTPUT", False, True)):
    set_lincast(lc)
    set_wkvcast(wc)
    if lc or wc:
        lg0 = seq_logits(model, toks[0])
        rel = float(mx.max(mx.abs(lg0 - lg_base0))
                   / (mx.max(mx.abs(lg_base0)) + 1e-30))
        mx.eval(lg0)
        assert rel > 0, "вариант %s НЕ ВКЛЮЧИЛСЯ" % name
        print("включение %-15s relmax %.3e" % (name, rel), flush=True)
    res[name] = run_variant()
    print("KL %-15s %.6f наты/токен" % (name, np.mean(res[name])), flush=True)
set_lincast(False)
set_wkvcast(False)

b = np.array(res["база"])
print("\nпарные разности KL (вариант − база), 95%% CI бутстрэпом по %d seq:"
      % N)
for name in ("каст_в_слое", "CAST_WKV_OUTPUT"):
    d = np.array(res[name]) - b
    rs = np.random.RandomState(4242)
    idx = rs.randint(0, len(d), size=(20000, len(d)))
    s = d[idx].mean(axis=1)
    lo, hi = np.percentile(s, [2.5, 97.5])
    sig = "значимо" if (lo > 0) == (hi > 0) else "НЕТ"
    print("  %-15s %+.6f [%+.6f; %+.6f]  %s"
          % (name, d.mean(), lo, hi, sig))
print("положительная разность = распределение УШЛО от bf16 дальше.")
print("своп за прогон: %.0f -> %.0f МБ (на KL не влияет)" % (sw0, swap_mb()))
