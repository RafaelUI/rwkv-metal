"""
Гейт: загрузка из одного .rwkvq против прежней пары .pth + .rwkvq.

Сравниваются ВЕСА, а не логиты. Первая версия сверяла логиты и порог
брала от измеренного сдвига ppl (+0.118% на 1.5B) -- и провалилась на
0.1B с расхождением 0.40. Это не поймало бы ошибку, а поймало масштаб:
у мелких моделей квантование бьёт кратно сильнее (в ablation вклад
`small` на 1.5B в десять раз выше, чем на 2.9B), да и argmax на
произвольных токенах решает шум. Порог, откалиброванный на одном
масштабе, на другом ничего не значит -- это ровно закон 10 из
rwkv-quant.

По весам вопрос ставится однозначно и от масштаба не зависит:

  1. Модули поверх sb6 (proj/cmix/head) обязаны быть ИДЕНТИЧНЫ: оба
     пути берут их из одного и того же .rwkvq. Любое расхождение --
     ошибка проводки.
  2. Всё остальное (нормировки, token-shift миксы, LoRA-ветки) в одном
     пути приезжает из .pth в исходном bf16, в другом -- деквантованным
     из .rwkvq. Расхождение обязано БЫТЬ (иначе .pth не читался) и
     обязано быть порядка ошибки квантования группы, а не порядка
     самого веса.

    python tests/dev_rwkvq_only_vs_pth.py <model.rwkvq> <model.pth>
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

import rwkv_metal as rk  # noqa: E402

# ошибка деквантования asym/rtn-веток -- около 2e-2 относительной нормы
# (замерено в rwkv-quant/tests/ablate_qlora_lora_source.py по группам).
# 0.15 -- с запасом на мелкие модели и на dense-тензоры, где ошибки нет
# вовсе, но всё равно на порядок ниже перепутанной проводки.
TOL_OTHER = 0.15
FAILS = []


def check(name, cond, extra=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + extra) if extra else ''}")
    if not cond:
        FAILS.append(name)


def params_of(model):
    return {k: v for k, v in tree_flatten(model.parameters())
            if isinstance(v, mx.array)}


def main():
    rwkvq, pth = sys.argv[1], sys.argv[2]
    kw = dict(rank=8, layers=range(0, 2), verbose=False)

    m_only, _, _ = rk.lora.load_rwkvq_model(rwkvq, **kw)
    p_only = params_of(m_only)
    mx.eval(list(p_only.values()))
    p_only = {k: np.array(v.astype(mx.float32)) for k, v in p_only.items()}
    del m_only
    mx.clear_cache()

    m_pth, _, _ = rk.lora.load_lora_rwkvq_model(pth, rwkvq, **kw)
    p_pth = params_of(m_pth)
    mx.eval(list(p_pth.values()))
    p_pth = {k: np.array(v.astype(mx.float32)) for k, v in p_pth.items()}
    del m_pth
    mx.clear_cache()

    common = sorted(set(p_only) & set(p_pth))
    check("состав параметров совпадает",
          set(p_only) == set(p_pth),
          f"{len(p_only)} против {len(p_pth)}, общих {len(common)}")

    # sb6-подложка узнаётся по имени модуля: wq/qblk и т.п. живут внутри
    # заменённых модулей, а не среди обычных весов
    def is_quant_backed(k):
        return any(s in k for s in ("wq", "qblk", "qsqm", "ddm",
                                    "scale", "bias_q"))

    # LoRA-адаптеры инициализируются СЛУЧАЙНО при каждой загрузке --
    # сравнивать их бессмысленно, они и должны различаться
    common = [k for k in common
              if ".lora_a" not in k and ".lora_b" not in k]

    ident, differ, big = [], [], []
    for k in common:
        a, b = p_only[k], p_pth[k]
        if a.shape != b.shape:
            big.append(f"{k}: формы {a.shape}/{b.shape}")
            continue
        same = bool((a == b).all())
        rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
        if is_quant_backed(k):
            (ident if same else big).append(k if same else f"{k}: rel {rel:.3e}")
        else:
            if same:
                ident.append(k)
            elif rel <= TOL_OTHER:
                differ.append(k)
            else:
                big.append(f"{k}: rel {rel:.3e}")

    check(f"квантованная подложка идентична ({len(ident)} совпало)",
          not big, "; ".join(big[:8]))
    check("неквантованное расходится, но в пределах ошибки кванта",
          bool(differ),
          f"{len(differ)} тензоров" if differ
          else "НИ ОДИН не разошёлся — значит .pth не читался, "
               "и сравниваются два одинаковых прогона")

    print(f"\nвсего {len(common)} параметров: идентичных {len(ident)}, "
          f"расходящихся в допуске {len(differ)}, вне допуска {len(big)}")
    print("\nГЕЙТ " + ("ПРОЙДЕН" if not FAILS else "ПРОВАЛЕН"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
