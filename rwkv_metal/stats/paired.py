"""Парный бутстрэп — один для всех сравнений качества (25.10).

До него у каждого замера был свой скрипт (`paired2.py`, `paired15.py`,
`qtsum.py`, …) со своей копией одних и тех же десяти строк: среднее
поштучных разностей и перцентили бутстрэпа по объектам. Вторая копия
арифметики — отложенное расхождение, поэтому она здесь, с гейтом и мутациями
(tests/test_paired_stats.py).

Протокол, который здесь закреплён (правила проекта):
  * плечи сравниваются ПОПАРНО на одних и тех же объектах: разность считается
    на объекте, а не между средними;
  * несколько сидов обучения: по умолчанию счёт объекта — среднее по сидам,
    бутстрэп — по объектам (`units="items"`); либо единица — пара
    (сид, объект) (`units="seed_item"`), когда сиды плеч сопоставлены;
  * интервал — перцентильный, 95 % по умолчанию, генератор с сидом: тот же
    вход — тот же интервал до бита.

    python -m rwkv_metal.stats.paired --a 'w/s*_on.json' --b 'w/s*_off.json' --ranks
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass

import numpy as np

# Сколько повторов бутстрэпа считается за раз: [chunk, n] индексов в памяти.
_CHUNK = 1000


@dataclass(frozen=True)
class Interval:
    """Средняя разность a − b и её интервал."""
    mean: float
    lo: float
    hi: float
    n: int          # число единиц бутстрэпа (объектов или пар сид × объект)
    level: float

    @property
    def significant(self) -> bool:
        """Интервал не содержит нуля."""
        return self.lo > 0 or self.hi < 0

    def __str__(self) -> str:
        return f"{self.mean:+.3f} [{self.lo:+.3f}; {self.hi:+.3f}]"


def item_means(runs) -> np.ndarray:
    """Счёт объекта, усреднённый по прогонам: [прогоны, объекты] → [объекты]."""
    a = np.asarray(runs, dtype=np.float64)
    if a.ndim == 1:
        return a
    if a.ndim != 2:
        raise ValueError(f"ожидается [объекты] или [прогоны, объекты], форма {a.shape}")
    return a.mean(axis=0)


def _units(a: np.ndarray, units: str) -> np.ndarray:
    if units == "items":
        return item_means(a)
    if units == "seed_item":
        return a.reshape(-1)
    raise ValueError(f"units: items | seed_item, а не {units!r}")


def paired_bootstrap(a, b, *, n_boot: int = 10000, level: float = 0.95, seed: int = 0,
                     units: str = "items", chunk: int = _CHUNK) -> Interval:
    """Средняя по единицам разность a − b и перцентильный интервал бутстрэпа.

    a, b — одной формы: [объекты] или [сиды, объекты]. Разность берётся на
    единице (объекте или паре сид × объект), и единицы пересэмплируются
    ВМЕСТЕ для обоих плеч — это и делает сравнение парным.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"плечи разной формы: {a.shape} и {b.shape} — сравнение не парное")
    if not 0 < level < 1:
        raise ValueError(f"level вне (0, 1): {level}")
    d = _units(a, units) - _units(b, units)
    n = d.size
    if n == 0:
        raise ValueError("нет объектов")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, chunk):
        k = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(k, n))
        means[start:start + k] = d[idx].mean(axis=1)
    alpha = 1.0 - level
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Interval(mean=float(d.mean()), lo=float(lo), hi=float(hi), n=n, level=level)


def reciprocal_ranks(ranks) -> np.ndarray:
    """Ранги (1 — лучший) → 1/ранг, поштучный вклад в MRR."""
    r = np.asarray(ranks, dtype=np.float64)
    if (r < 1).any():
        raise ValueError("ранг меньше 1")
    return 1.0 / r


def load_per_row(path: str, key: str = "per_row", ranks: bool = False) -> np.ndarray:
    """Поштучные результаты из отчёта прогона (embed-run, gen-run: поле `key`).
    ranks=True — в отчёте ранги, вернуть 1/ранг."""
    with open(path, encoding="utf-8") as f:
        v = json.load(f)[key]
    return reciprocal_ranks(v) if ranks else np.asarray(v, dtype=np.float64)


def compare_table(arms: dict, pairs, subsets: dict | None = None, **kw) -> str:
    """Таблица markdown: строка на пару плеч, столбец на подмножество объектов.
    arms — имя → [объекты] или [сиды, объекты]; subsets — имя → индексы объектов."""
    first = next(iter(arms.values()))
    n_items = np.asarray(first).shape[-1]
    subsets = subsets or {"все": list(range(n_items))}
    head = "| разница | " + " | ".join(f"{k} (n={len(v)})" for k, v in subsets.items()) + " |"
    rows = [head, "|---|" + "---|" * len(subsets)]
    for x, y in pairs:
        cells = []
        for idx in subsets.values():
            ax = np.asarray(arms[x])[..., idx]
            ay = np.asarray(arms[y])[..., idx]
            cells.append(str(paired_bootstrap(ax, ay, **kw)))
        rows.append(f"| {x} − {y} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _load_arm(pattern: str, key: str, ranks: bool) -> np.ndarray:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"нет файлов: {pattern}")
    return np.stack([load_per_row(p, key, ranks) for p in paths]), paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="парный бутстрэп двух плеч по поштучным результатам")
    ap.add_argument("--a", required=True, help="шаблон файлов плеча A (по файлу на сид)")
    ap.add_argument("--b", required=True, help="шаблон файлов плеча B, сиды в том же порядке")
    ap.add_argument("--key", default="per_row")
    ap.add_argument("--ranks", action="store_true", help="в отчёте ранги — сравнивать 1/ранг (MRR)")
    ap.add_argument("--units", default="items", choices=["items", "seed_item"])
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    A, pa = _load_arm(a.a, a.key, a.ranks)
    B, pb = _load_arm(a.b, a.key, a.ranks)
    if A.shape != B.shape:
        raise SystemExit(f"плечи разной формы: {A.shape} ({len(pa)} файлов) и {B.shape} ({len(pb)})")
    iv = paired_bootstrap(A, B, n_boot=a.boot, level=a.level, seed=a.seed, units=a.units)
    print(f"A: {len(pa)} файлов, среднее {A.mean():.4f}; по сидам: " + " / ".join(f"{m:.4f}" for m in A.mean(axis=1)))
    print(f"B: {len(pb)} файлов, среднее {B.mean():.4f}; по сидам: " + " / ".join(f"{m:.4f}" for m in B.mean(axis=1)))
    print(f"A − B: {iv}  (единиц {iv.n}, {a.units}, {int(a.level * 100)} %, бутстрэп {a.boot})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
