"""ГЕЙТ ПАРНОГО БУТСТРЭПА (25.10): rwkv_metal/stats/paired.py.

Свойства:
 1. одинаковые плечи — разность 0 и интервал [0; 0]; сдвиг на константу c —
    ровно c и [c; c];
 2. среднее — среднее поштучных разностей (ручной счёт);
 3. ПАРНОСТЬ: на данных с большой разницей между объектами и малой
    устойчивой разницей плеч интервал узкий и не содержит нуля, а у той же
    разницы без пар (независимые пересэмплы) — в разы шире;
 4. сиды: units="items" — то же, что бутстрэп по средним объекта; units=
    "seed_item" — единиц сиды × объекты;
 5. ПОКРЫТИЕ: на 300 наборах с известной разностью 95 %-интервал накрывает
    её в 92…98 % случаев; 90 % вложен в 95 %;
 6. детерминизм по сиду, независимость от размера пачки повторов;
 7. отказы: разные формы, пустые плечи, ранг < 1, неизвестные единицы;
 8. чтение отчёта (ранги → 1/ранг) и командная строка.

МУТАЦИИ (STATS_MUT=...): правка в исходник модуля до импорта; гейт обязан
упасть. Мутация, подстроки которой в исходнике нет или она не единственна, —
СЛОМАНА, а не поймана: это проверяется до запуска.

    python tests/test_paired_stats.py
    python tests/test_paired_stats.py --mutations
"""
import importlib.abc
import importlib.machinery
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MOD = "rwkv_metal/stats/paired.py"
MUT = os.environ.get("STATS_MUT", "")
MUTATIONS = {
    "unpaired": ("        means[start:start + k] = d[idx].mean(axis=1)",
                 "        idx2 = rng.integers(0, n, size=(k, n))\n"
                 "        means[start:start + k] = (_units(a, units)[idx] - _units(b, units)[idx2]).mean(axis=1)"),
    "seedaxis": ("    return a.mean(axis=0)", "    return a.mean(axis=1)"),
    "seeditem": ('    if units == "seed_item":\n        return a.reshape(-1)',
                 '    if units == "seed_item":\n        return item_means(a)'),
    "level":    ("    alpha = 1.0 - level", "    alpha = level"),
    "pointboot": ("    return Interval(mean=float(d.mean()),", "    return Interval(mean=float(means.mean()),"),
    "noseed":   ("    rng = np.random.default_rng(seed)", "    rng = np.random.default_rng()"),
    "chunk":    ("        k = min(chunk, n_boot - start)", "        k = chunk"),
    "lastchunk": ("    for start in range(0, n_boot, chunk):", "    for start in range(0, n_boot - chunk + 1, chunk):"),
    "rr":       ("    return 1.0 / r", "    return r"),
    "shape":    ("    if a.shape != b.shape:\n        raise ValueError", "    if False:\n        raise ValueError"),
    "rank0":    ("    if (r < 1).any():", "    if (r < 0).any():"),
    "sig":      ("        return self.lo > 0 or self.hi < 0", "        return self.lo > 0"),
}

if MUT:
    _a, _b = MUTATIONS[MUT]
    _path = os.path.join(ROOT, MOD)
    sys.dont_write_bytecode = True

    class _Loader(importlib.machinery.SourceFileLoader):
        def path_stats(self, path):
            raise OSError
        def get_data(self, path):
            data = super().get_data(path)
            if os.path.abspath(path) == _path:
                data = data.decode("utf-8").replace(_a, _b).encode("utf-8")
            return data

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is not None and spec.origin and os.path.abspath(spec.origin) == _path:
                spec.loader = _Loader(name, spec.origin)
            return spec

    sys.meta_path.insert(0, _Finder())

import numpy as np                                                       # noqa: E402
from rwkv_metal.stats import paired as P                                 # noqa: E402


def raises(f, *a, **kw):
    try:
        f(*a, **kw)
    except (ValueError, SystemExit):
        return True
    return False


def main():
    rs = np.random.RandomState(0)

    # 1. тождество и сдвиг
    x = rs.rand(50)
    iv = P.paired_bootstrap(x, x, n_boot=500)
    assert (iv.mean, iv.lo, iv.hi) == (0.0, 0.0, 0.0), iv
    assert not iv.significant
    iv = P.paired_bootstrap(x + 0.25, x, n_boot=500)
    assert abs(iv.mean - 0.25) < 1e-12 and abs(iv.lo - 0.25) < 1e-12 and abs(iv.hi - 0.25) < 1e-12, iv
    assert iv.significant
    assert P.paired_bootstrap(x - 0.25, x, n_boot=500).significant, "отрицательная разность не значима"
    print("  тождество и сдвиг")

    # 2. среднее — по поштучным разностям
    a = np.array([1.0, 0.0, 0.5, 0.25]); b = np.array([0.5, 0.5, 0.5, 0.0])
    assert abs(P.paired_bootstrap(a, b, n_boot=200).mean - 0.0625) < 1e-15   # (0.5 − 0.5 + 0 + 0.25) / 4
    assert str(P.Interval(0.0631, 0.02, 0.105, 199, 0.95)) == "+0.063 [+0.020; +0.105]"
    print("  среднее и печать")

    # 3. парность: объекты разной трудности, плечо A стабильно лучше на 0.02
    hard = rs.rand(300)
    b3 = hard + rs.randn(300) * 0.01
    a3 = b3 + 0.02 + rs.randn(300) * 0.01
    paired = P.paired_bootstrap(a3, b3, n_boot=2000)
    rng = np.random.default_rng(1)
    unp = np.array([a3[rng.integers(0, 300, 300)].mean() - b3[rng.integers(0, 300, 300)].mean()
                    for _ in range(2000)])
    w_unp = np.quantile(unp, 0.975) - np.quantile(unp, 0.025)
    w_p = paired.hi - paired.lo
    assert paired.significant and paired.lo > 0.015, paired
    assert w_unp > 5 * w_p, f"интервал без пар не шире парного: {w_unp:.4f} против {w_p:.4f}"
    print(f"  парность: {paired}, ширина {w_p:.4f} против {w_unp:.4f} без пар")

    # 4. сиды
    A = rs.rand(3, 40); B = rs.rand(3, 40)
    it = P.paired_bootstrap(A, B, n_boot=1000, units="items")
    ref = P.paired_bootstrap(A.mean(axis=0), B.mean(axis=0), n_boot=1000)
    assert it == ref, (it, ref)
    assert it.n == 40
    si = P.paired_bootstrap(A, B, n_boot=1000, units="seed_item")
    flat = P.paired_bootstrap(A.reshape(-1), B.reshape(-1), n_boot=1000)
    assert si == flat and si.n == 120
    assert abs(si.mean - it.mean) < 1e-12, "средняя разность зависит от единиц"
    assert (si.lo, si.hi) != (it.lo, it.hi)
    assert np.allclose(P.item_means(A), A.mean(axis=0))
    print("  сиды: items и seed_item")

    # 5. покрытие и вложенность уровней
    hits = 0
    for k in range(300):
        r = np.random.RandomState(100 + k)
        base = r.rand(60)
        aa = base + 0.05 + r.randn(60) * 0.1
        hits += P.paired_bootstrap(aa, base, n_boot=800, seed=k).lo <= 0.05 <= \
            P.paired_bootstrap(aa, base, n_boot=800, seed=k).hi
    assert 0.92 <= hits / 300 <= 0.98, f"покрытие 95 %-интервала {hits / 300:.3f}"
    w95 = P.paired_bootstrap(a3, b3, n_boot=2000, level=0.95)
    w90 = P.paired_bootstrap(a3, b3, n_boot=2000, level=0.90)
    assert w95.lo < w90.lo < w90.hi < w95.hi, (w90, w95)
    print(f"  покрытие {hits / 300:.3f}; 90 % внутри 95 %")

    # 6. детерминизм и пачки повторов
    one = P.paired_bootstrap(a3, b3, n_boot=1500, seed=5)
    assert one == P.paired_bootstrap(a3, b3, n_boot=1500, seed=5)
    assert one != P.paired_bootstrap(a3, b3, n_boot=1500, seed=6)
    for c in (1500, 700, 1):
        assert P.paired_bootstrap(a3, b3, n_boot=1500, seed=5, chunk=c) == one, f"пачка {c} меняет интервал"
    print("  детерминизм; пачки повторов")

    # 7. отказы
    assert raises(P.paired_bootstrap, np.zeros(3), np.zeros(4))
    assert raises(P.paired_bootstrap, np.zeros((2, 3)), np.zeros((3, 3)))
    assert raises(P.paired_bootstrap, np.zeros(0), np.zeros(0))
    assert raises(P.paired_bootstrap, x, x, units="seeds")
    assert raises(P.paired_bootstrap, x, x, level=1.0)
    assert raises(P.reciprocal_ranks, [1, 0, 2])
    print("  отказы")

    # 8. отчёты и командная строка
    with tempfile.TemporaryDirectory() as d:
        for s in range(2):
            json.dump({"per_row": [1, 2, 4, 1]}, open(f"{d}/s{s}_on.json", "w"))
            json.dump({"per_row": [2, 2, 4, 1]}, open(f"{d}/s{s}_off.json", "w"))
        v = P.load_per_row(f"{d}/s0_on.json", ranks=True)
        assert np.array_equal(v, [1.0, 0.5, 0.25, 1.0])
        assert np.array_equal(P.load_per_row(f"{d}/s0_on.json"), [1, 2, 4, 1])
        buf = io.StringIO()
        with redirect_stdout(buf):
            P.main(["--a", f"{d}/s*_on.json", "--b", f"{d}/s*_off.json", "--ranks", "--boot", "300"])
        want = P.paired_bootstrap([[1, .5, .25, 1]] * 2, [[.5, .5, .25, 1]] * 2, n_boot=300)
        assert f"A − B: {want}" in buf.getvalue(), buf.getvalue()
        assert "+0.125" in str(want)
        t = P.compare_table({"x": a3, "y": b3}, [("x", "y")], {"все": list(range(300)), "первые": list(range(10))},
                            n_boot=300)
        assert "| x − y |" in t and str(P.paired_bootstrap(a3[:10], b3[:10], n_boot=300)) in t, t
    print("  отчёты, таблица, командная строка")
    print("ГЕЙТ ПРОЙДЕН")


def all_mutations():
    import subprocess
    src = open(os.path.join(ROOT, MOD), encoding="utf-8").read()
    broken = [k for k, (a, _) in MUTATIONS.items() if src.count(a) != 1]
    for k in broken:
        print(f"  {k:10s} СЛОМАНА: подстрока встречается {src.count(MUTATIONS[k][0])} раз")
    missed = []
    for name in MUTATIONS:
        if name in broken:
            continue
        r = subprocess.run([sys.executable, os.path.abspath(__file__)],
                           env=dict(os.environ, STATS_MUT=name), capture_output=True, text=True)
        why = [l for l in (r.stdout + r.stderr).splitlines() if "Error" in l]
        print(f"  {name:10s} {'ПОЙМАНА' if r.returncode else 'НЕ ПОЙМАНА'}  {(why[-1] if why else '')[:110]}")
        if r.returncode == 0:
            missed.append(name)
    assert not broken and not missed, f"сломаны: {broken}, не пойманы: {missed}"
    print(f"все {len(MUTATIONS)} мутаций пойманы")


if __name__ == "__main__":
    all_mutations() if "--mutations" in sys.argv else main()
