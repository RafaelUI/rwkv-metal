"""
Гейт: чтение .rwkvq напрямую даёт ровно то же, что прежний сайдкар.

Смысл проверки. `load_sidecar` теперь умеет два источника -- готовый
сайдкар и .rwkvq, из которого K3-интерлив строится на месте
(rwkv_quant.formats.codec, numpy, без torch). Если веса из двух
источников совпадают бит-в-бит, промежуточный файл можно перестать
возить: он был нужен только потому, что интерлив умел строить лишь
export_mlx через torch.

Сверяются не буферы, а ДЕКВАНТОВАННЫЕ веса -- то, что реально уходит в
матмул. Совпадение буферов уже проверено на стороне rwkv-quant
(tests/test_k3_from_canonical.py), здесь важно, что путь целиком, вместе
с построением RwkvqLinear, ведёт к тем же числам.

    python tests/dev_rwkvq_direct.py <model.rwkvq> <сайдкар_без_расширения>
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from rwkv_metal.lora.rwkvq_linear import RwkvqLinear, load_sidecar  # noqa: E402

PER_SHAPE = 2
FAILS = []


def check(name, cond, extra=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + extra) if extra else ''}")
    if not cond:
        FAILS.append(name)


def main():
    rwkvq, sidecar = sys.argv[1], sys.argv[2]

    a_side, m_side = load_sidecar(sidecar)
    a_direct, m_direct = load_sidecar(rwkvq)
    print(f"сайдкар: {len(m_side['tensors'])} тензоров, "
          f".rwkvq напрямую: {len(m_direct['tensors'])} sb6")

    common = sorted(set(m_side["tensors"]) & set(m_direct["tensors"]))
    check("состав пересекается", bool(common), f"{len(common)} общих")
    if not common:
        return 1

    seen, picked = {}, []
    for k in common:
        sh = tuple(m_side["tensors"][k]["shape"])
        if len(seen.setdefault(sh, [])) < PER_SHAPE:
            seen[sh].append(k)
            picked.append(k)

    n_el = 0
    for key in picked:
        ws = RwkvqLinear.from_sidecar(sidecar, key)._dequant_w()
        wd = RwkvqLinear.from_sidecar(rwkvq, key)._dequant_w()
        mx.eval(ws, wd)
        s = np.array(ws.astype(mx.float32))
        d = np.array(wd.astype(mx.float32))
        same = s.shape == d.shape and bool((s == d).all())
        n_el += s.size
        check(f"{key} {s.shape}", same,
              "" if same else f"расхождений {int((s != d).sum())}, "
                              f"max|Δ| {np.abs(s - d).max():.3e}")

    print(f"\nсверено {len(picked)} тензоров ({len(seen)} форм), "
          f"{n_el / 1e6:.1f}M элементов")
    print("\nГЕЙТ " + ("ПРОЙДЕН" if not FAILS else f"ПРОВАЛЕН: {len(FAILS)}"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
