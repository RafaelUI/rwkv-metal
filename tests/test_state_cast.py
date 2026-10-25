"""ГЕЙТ ТИПОВ В ШАГЕ STATE-TUNING (23.10): приведение входа плотной проекции.

`rwkv7_x070.dense` при CAST_DENSE_INPUT считает матмул в типе веса и возвращает
выход в типе входа (как `_matmul_cast` квантованного слоя, 22.08); state_tune
включает его на время обучения (`StateTuneConfig.cast_projection_input`, по выбору),
а `cast_wkv_output` -- приведение выхода WKV (`CAST_WKV_OUTPUT`, умолчание с 25.10). После
обучения флаги модуля -- прежние. База -- плотная 0.1B из .pth в bf16: в fp32 оба
приведения -- тождество, и гейт ничего бы не видел.

1. СЕМАНТИКА dense: при fp32-входе с флагом -- независимая запись формулы
   (x.astype(w.dtype) @ w.T).astype(x.dtype), тип выхода -- тип входа, и
   отличается от пути без флага; без флага -- ровно layer(x); при входе в типе
   веса флаг -- тождество бит в бит.
2. ПРОВОДКА: при включённом флаге один проход модели зовёт dense 6·L + 1 раз
   (r/k/v/o, ключ и значение cmix в каждом блоке, голова).
3. ГРАДИЕНТ по блокам == автоград по всему графу при обоих приведениях.
4. ОБУЧЕНИЕ: исход зависит только от StateTuneConfig, а не от флагов модуля на
   входе; флаги после обучения -- прежние; оба поля действуют; умолчания.

МУТАЦИОННЫЕ КРЫШКИ (CAST_MUT=...): правка вносится в ИСХОДНИК модуля до импорта,
гейт обязан упасть. Список -- MUTATIONS ниже.

    python tests/test_state_cast.py                # гейт
    python tests/test_state_cast.py --mutations    # все крышки по очереди
"""
import importlib.abc
import importlib.machinery
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MUT = os.environ.get("CAST_MUT", "")
X070 = "rwkv_metal/model/rwkv7_x070.py"
TUNE = "rwkv_metal/lora/state_tune.py"
SEQ = "rwkv_metal/lora/sequential.py"
MUTATIONS = {
    "default": (TUNE, "    cast_projection_input: bool = False", "    cast_projection_input: bool = True"),
    "wkvdefault": (TUNE, "    cast_wkv_output:       bool = True", "    cast_wkv_output:       bool = False"),
    "noset": (TUNE, "    _x070.CAST_DENSE_INPUT, _x070.CAST_WKV_OUTPUT = cfg.cast_projection_input, cfg.cast_wkv_output",
                    "    pass"),
    "swap": (TUNE, "    _x070.CAST_DENSE_INPUT, _x070.CAST_WKV_OUTPUT = cfg.cast_projection_input, cfg.cast_wkv_output",
                   "    _x070.CAST_DENSE_INPUT, _x070.CAST_WKV_OUTPUT = cfg.cast_wkv_output, cfg.cast_projection_input"),
    "norestore": (TUNE, "        _x070.CAST_DENSE_INPUT, _x070.CAST_WKV_OUTPUT = saved", "        pass"),
    "nodtype": (X070, "        return y.astype(x.dtype)\n    return layer(x)", "        return y\n    return layer(x)"),
    "fp32": (X070, "        y = x.astype(layer.weight.dtype) @ layer.weight.T", "        y = x.astype(mx.float32) @ layer.weight.T"),
    "cond": (X070, "x.dtype != layer.weight.dtype:", "x.dtype == layer.weight.dtype:"),
    "flag": (X070, "    if CAST_DENSE_INPUT and type(layer) is nn.Linear", "    if False and type(layer) is nn.Linear"),
    "site_r": (X070, "        r = dense(self.r_proj, xr).reshape(B, T, H, S)", "        r = self.r_proj(xr).reshape(B, T, H, S)"),
    "site_value": (X070, "        return dense(self.value, nn.relu(dense(self.key, xk)) ** 2)",
                         "        return self.value(nn.relu(dense(self.key, xk)) ** 2)"),
    "site_head": (X070, "        return dense(self.head, self.body(idx))", "        return self.head(self.body(idx))"),
    "seq_head": (SEQ, "            logits = _x070.dense(model.head, hc)", "            logits = model.head(hc)"),
}

if MUT:
    assert MUT in MUTATIONS, f"нет мутации {MUT}"
    _target, _a, _b = MUTATIONS[MUT]
    _target_path = os.path.join(ROOT, _target)
    sys.dont_write_bytecode = True

    class _Loader(importlib.machinery.SourceFileLoader):
        def path_stats(self, path):            # мимо кэша байткода
            raise OSError
        def get_data(self, path):
            data = super().get_data(path)
            if os.path.abspath(path) == _target_path:
                src = data.decode("utf-8")
                assert src.count(_a) == 1, f"мутация {MUT}: подстрока встречается {src.count(_a)} раз"
                data = src.replace(_a, _b).encode("utf-8")
            return data

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is not None and spec.origin and os.path.abspath(spec.origin) == _target_path:
                spec.loader = _Loader(name, spec.origin)
            return spec

    sys.meta_path.insert(0, _Finder())

import mlx.core as mx                      # noqa: E402
import mlx.nn as nn                        # noqa: E402
import numpy as np                         # noqa: E402
from mlx.utils import tree_flatten         # noqa: E402

import rwkv_metal.lora.sequential as seq                       # noqa: E402
import rwkv_metal.model.rwkv7_x070 as x070                     # noqa: E402
from rwkv_metal.lora import StateTuneConfig, state_tune        # noqa: E402
from rwkv_metal.model.convert import load_pretrained           # noqa: E402
from rwkv_metal.model.state import RWKVState                   # noqa: E402
from rwkv_metal.model.initial_state import prepare_state_tuning, zero_state   # noqa: E402

PTH = os.path.expanduser(os.environ.get(
    "RWKV_PTH", "~/Develop/SwiftRWKV/.testdata/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth"))


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and mx.array_equal(a, b).item()


def maxrel(ref, got):
    r = ref.astype(mx.float32)
    return (mx.abs(r - got.astype(mx.float32)).max() / (mx.abs(r).max() + 1e-12)).item()


def batch(B, T, seed=7):
    rs = np.random.RandomState(seed)
    ids = rs.randint(1, 60000, size=(B, T + 1)).astype(np.int32)
    return mx.array(ids[:, :-1]), mx.array(ids[:, 1:])


def random_state(model, seed=3, scale=0.05):
    rs = np.random.RandomState(seed)
    z = zero_state(model)
    return RWKVState(wkv=mx.array((rs.normal(size=z.wkv.shape) * scale).astype(np.float32)),
                     tmix_shift=z.tmix_shift, cmix_shift=z.cmix_shift).eval()


def main():
    assert x070.CAST_DENSE_INPUT is False and x070.CAST_WKV_OUTPUT is False, "умолчания модуля сдвинулись"
    c = StateTuneConfig()
    assert c.cast_projection_input is False and c.cast_wkv_output is True, "умолчания StateTuneConfig (25.10)"
    m = load_pretrained(PTH, verbose=False)[0]
    L, D = len(m.blocks), m.blocks[0].tmix.r_proj.weight.shape[1]

    # ── 1. семантика dense ──
    rs = np.random.RandomState(11)
    for name, layer in [("r_proj", m.blocks[0].tmix.r_proj), ("cmix.key", m.blocks[1].cmix.key),
                        ("head", m.head)]:
        w = layer.weight
        assert w.dtype == mx.bfloat16, (name, w.dtype)
        x = mx.array(rs.normal(size=(1, 16, w.shape[1])).astype(np.float32))
        x070.CAST_DENSE_INPUT = True
        on, on_half = x070.dense(layer, x), x070.dense(layer, x.astype(mx.bfloat16))
        x070.CAST_DENSE_INPUT = False
        off, off_half = x070.dense(layer, x), x070.dense(layer, x.astype(mx.bfloat16))
        mx.eval(on, on_half, off, off_half)
        assert on.dtype == mx.float32, f"{name}: выход не в типе входа ({on.dtype})"
        assert same(on, (x.astype(w.dtype) @ w.T).astype(x.dtype)), f"{name}: с флагом -- не матмул в типе веса"
        assert same(off, layer(x)), f"{name}: без флага путь изменился"
        assert not same(on, off), f"{name}: флаг ничего не меняет на fp32-входе"
        assert same(on_half, off_half), f"{name}: на входе в типе веса флаг обязан быть тождеством"
    print("  семантика dense: формула, тип выхода, тождество на bf16-входе")

    # ── 2. проводка ──
    calls = [0]
    real = x070.dense
    def counting(layer, x):
        calls[0] += 1
        return real(layer, x)
    x070.dense = counting
    try:
        x070.CAST_DENSE_INPUT = True
        mx.eval(m(batch(1, 16)[0]))
    finally:
        x070.dense = real
        x070.CAST_DENSE_INPUT = False
    assert calls[0] == 6 * L + 1, f"проход зовёт dense {calls[0]} раз, ждали {6 * L + 1}"
    print(f"  проводка: {calls[0]} вызовов dense на проход (6·{L} + 1)")

    # ── 3. градиент по блокам == автоград ──
    def lm_loss(model, a, b):
        return model.loss(a, b).astype(mx.float32)
    x, y = batch(2, 32)
    for what, proj, wkv in [("вход проекций", True, False), ("выход WKV", False, True), ("оба", True, True)]:
        mm = load_pretrained(PTH, verbose=False)[0]
        prepare_state_tuning(mm, init=random_state(mm), train=("wkv",))
        x070.CAST_DENSE_INPUT, x070.CAST_WKV_OUTPUT = proj, wkv
        try:
            loss_ref, g_ref = nn.value_and_grad(mm, lm_loss)(mm, x, y)
            loss, g = seq.value_and_grad(mm, x, y, 0)
            mx.eval(loss_ref, g_ref, loss, g)
            x070.CAST_DENSE_INPUT, x070.CAST_WKV_OUTPUT = False, False
            _, g_plain = seq.value_and_grad(mm, x, y, 0)
            mx.eval(g_plain)
        finally:
            x070.CAST_DENSE_INPUT, x070.CAST_WKV_OUTPUT = False, False
        gr, gs, gp = dict(tree_flatten(g_ref)), dict(tree_flatten(g)), dict(tree_flatten(g_plain))
        worst = max(maxrel(gr[k], gs[k]) for k in gr)
        k0 = "blocks.0.state0.wkv"
        print(f"  {what}: по блокам к автограду {worst:.1e}, лосс {loss.item():.6f} против {loss_ref.item():.6f}; "
              f"к счёту без приведения {maxrel(gp[k0], gs[k0]):.1e}")
        assert worst <= 1e-5, f"{what}: по блокам разошёлся с автоградом на {worst:.2e}"
        assert abs(loss.item() - loss_ref.item()) <= 1e-5 * abs(loss_ref.item()), what
        assert not same(gp[k0], gs[k0]), f"{what}: приведение не дошло до градиента"

    # ── 4. обучение ──
    def tune(cfg, pre):
        model = load_pretrained(PTH, verbose=False)[0]
        x070.CAST_DENSE_INPUT, x070.CAST_WKV_OUTPUT = pre
        xb, yb = batch(1, 32, seed=13)
        during = []
        res = state_tune(model, [(xb, yb)], cfg,
                         on_step=lambda s, l, p: during.append((x070.CAST_DENSE_INPUT, x070.CAST_WKV_OUTPUT)))
        assert during and all(d == (cfg.cast_projection_input, cfg.cast_wkv_output) for d in during), \
            f"во время обучения флаги {during[:1]}, а в конфигурации {(cfg.cast_projection_input, cfg.cast_wkv_output)}"
        after = (x070.CAST_DENSE_INPUT, x070.CAST_WKV_OUTPUT)
        x070.CAST_DENSE_INPUT, x070.CAST_WKV_OUTPUT = False, False
        return res["state"].wkv, after
    base = dict(lr=0.05, lr_final=0.01, warmup_steps=0, max_steps=3, log_every=0, cache_limit_gb=0,
                loss_chunk_tokens=0, state_path=None)
    a, after1 = tune(StateTuneConfig(**base), (False, False))
    a2, after2 = tune(StateTuneConfig(**base), (True, True))
    off = dict(cast_projection_input=False, cast_wkv_output=False)
    lin = dict(cast_projection_input=True, cast_wkv_output=False)
    n, after3 = tune(StateTuneConfig(**base, **off), (True, False))
    n2, _ = tune(StateTuneConfig(**base, **off), (False, True))
    w, after4 = tune(StateTuneConfig(**base, **lin), (False, False))
    w2, _ = tune(StateTuneConfig(**base, **lin), (False, True))
    assert same(a, a2), "умолчание: исход зависит от флагов модуля"
    assert same(n, n2), "без приведения: исход зависит от флагов модуля"
    assert same(w, w2), "вход проекций: исход зависит от флагов модуля"
    assert not same(a, n), "cast_wkv_output (умолчание) не действует"
    assert not same(w, n), "cast_projection_input не действует"
    assert (after1, after2, after3, after4) == ((False, False), (True, True), (True, False), (False, False)), \
        f"флаги модуля после обучения не возвращены: {(after1, after2, after3, after4)}"
    print("  обучение: исход -- от конфигурации, флаги возвращены, оба поля действуют")
    print("ГЕЙТ ПРОЙДЕН")


def all_mutations():
    """Каждая крышка -- отдельным процессом; гейт обязан упасть на каждой."""
    import subprocess
    missed = []
    # Мутация, подстроки которой нет в исходнике или она не единственна, —
    # СЛОМАНА, а не поймана: в дочернем процессе она падает на сверке
    # подстроки, и без этой проверки такой падёж засчитывался как «поймана».
    broken = [n for n, (t, a, _) in MUTATIONS.items()
              if open(os.path.join(ROOT, t), encoding="utf-8").read().count(a) != 1]
    for n in broken:
        print(f"  {n} СЛОМАНА: подстрока не единственна или её нет")
    for name in MUTATIONS:
        if name in broken:
            continue
        r = subprocess.run([sys.executable, os.path.abspath(__file__)],
                           env=dict(os.environ, CAST_MUT=name), capture_output=True, text=True)
        why = [l for l in (r.stdout + r.stderr).splitlines() if "Error" in l]
        print(f"  {name:11s} {'ПОЙМАНА' if r.returncode else 'НЕ ПОЙМАНА'}  {(why[-1] if why else '')[:110]}")
        if r.returncode == 0:
            missed.append(name)
    assert not broken and not missed, f"сломаны: {broken}, не пойманы: {missed}"
    print(f"все {len(MUTATIONS)} крышек держат")


if __name__ == "__main__":
    all_mutations() if "--mutations" in sys.argv else main()
