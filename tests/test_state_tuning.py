"""ГЕЙТ STATE-TUNING (model/initial_state.py, lora/state_tune.py, блок x070).

Что проверяется:

1. НУЛЕВОЕ подключённое состояние счёта не меняет: логиты побитно те же.
2. ПОДКЛЮЧЁННОЕ == ЯВНОЕ: body(idx) с подключённым S побитно равен
   body(idx, state=S); явное состояние подключённое вытесняет.
3. СКЛЕЙКА: свёрнутый промпт, подключённый как старт, продолжает так же, как
   сплошной проход (все три части состояния); «обученное -> промпт -> кэш».
4. ГРАДИЕНТ состояния обратным проходом по блокам против автограда по всему
   графу: одна строка, батч 2, со сдвигами, лосс кусками; над .rwkvq.
5. БАТЧ: градиент на батче из двух строк == среднее градиентов по строкам
   (трансляция [1, ...] и сумма по батчу), и конечная разность по направлению
   (fp32) -- проверка, не зависящая от автограда.
6. ОБУЧЕНИЕ: state_tune идёт по блокам, выключатель даёт то же состояние,
   база не меняется, WKV остаётся fp32 и после set_dtype, лосс падает.
7. ФАЙЛ: туда-обратно побитно; отпечаток базы один у bf16 и fp32, разный у
   плотной, REDUCTION и COMPRESSION; чужая база -- отказ, чужая архитектура --
   отказ всегда.

МУТАЦИОННЫЕ КРЫШКИ (STATE_MUT=...): правка вносится в ИСХОДНИК модуля до его
импорта, гейт обязан упасть. Список -- MUTATIONS ниже.

    python tests/test_state_tuning.py                # гейт
    python tests/test_state_tuning.py --mutations    # все крышки по очереди
"""
import importlib.abc
import importlib.machinery
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MUT = os.environ.get("STATE_MUT", "")
X070 = "rwkv_metal/model/rwkv7_x070.py"
INIT = "rwkv_metal/model/initial_state.py"
TUNE = "rwkv_metal/lora/state_tune.py"
MUTATIONS = {
    # блок
    "explicit": (X070, "            if h_in is None:\n                h_in = mx.broadcast_to(s0.wkv",
                       "            if True:\n                h_in = mx.broadcast_to(s0.wkv"),
    "tshift": (X070, "tmix_prev = mx.broadcast_to(s0.tmix_shift", "tmix_prev = mx.broadcast_to(s0.cmix_shift"),
    "cshift": (X070, "cmix_prev = mx.broadcast_to(s0.cmix_shift", "cmix_prev = mx.broadcast_to(0 * s0.cmix_shift"),
    "nowkv": (X070, "h_in = mx.broadcast_to(s0.wkv.astype(mx.float32)[None]",
                    "h_in = mx.broadcast_to(0 * s0.wkv.astype(mx.float32)[None]"),
    "dtype": (X070, '(k, v if k.endswith("state0.wkv") else v.astype(dtype))', "(k, v.astype(dtype))"),
    # подключение
    "layer": (INIT, "s0 = InitialState(wkv=state.wkv[l, 0]", "s0 = InitialState(wkv=state.wkv[0, 0]"),
    "detach": (INIT, '            del blk["state0"]', "            pass"),
    "train": (INIT, "            s0.unfreeze(keys=list(train))", "            s0.unfreeze()"),
    "frozen": (INIT, "    model.freeze()\n    attach_state(model, init, train=train)",
                     "    attach_state(model, init, train=train)"),
    "fold": (INIT, "return model.states(ids, state=state).stop_gradient().eval()",
                   "return model.states(ids[:, :-1], state=state).stop_gradient().eval()"),
    # отпечаток и файл
    "fpvalues": (INIT, 'a = flat[name].astype(mx.bfloat16).astype(mx.float32).reshape(-1)\n'
                       '        h.update(name.encode() + b"\\0")\n'
                       '        h.update(np.asarray(a, dtype="<f4").tobytes())',
                       'a = flat[name].astype(mx.bfloat16).astype(mx.float32).reshape(-1)\n'
                       '        h.update(name.encode() + b"\\0")'),
    "fpcanon": (INIT, "a = flat[name].astype(mx.bfloat16).astype(mx.float32).reshape(-1)",
                      "a = flat[name].astype(mx.float32).reshape(-1)"),
    "fprows": (INIT, "    for name in fingerprint_rows(c[\"n_layer\"]):", "    for name in []:"),
    "fplattice": (INIT, "lattice = mx.bfloat16 if isinstance(mod, nn.Linear) else mx.float16",
                        "lattice = mx.bfloat16"),
    "fpblocks": (INIT, "    for l in range(n_layer):\n        names +=", "    for l in range(1):\n        names +="),
    "accept": (INIT, 'if meta.get("base_fingerprint") != fp and not allow_other_base:',
                     'if meta.get("base_fingerprint") != fp and allow_other_base:'),
    "contract": (INIT, "            if meta.get(k) != str(v):", "            if False:"),
    "format": (INIT, '    if meta.get("format") != FORMAT:', "    if False:"),
    "savefp": (INIT, '"trained": ",".join(trained), "base_fingerprint": base_fingerprint(model),',
                     '"trained": ",".join(trained), "base_fingerprint": "",'),
    # обучение
    "seqoff": (TUNE, "    if cfg.sequential_backward and _seq.supported(model):", "    if False:"),
    "lr": (TUNE, "        opt.learning_rate = lr_at(step, cfg)", "        opt.learning_rate = 0.0"),
    "kind": (TUNE, 'kind="tuned" if init is None else "tuned+prompt"', 'kind="prompt"'),
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
from rwkv_metal.lora import load_rwkvq_model, StateTuneConfig, state_tune   # noqa: E402
from rwkv_metal.model.convert import load_pretrained           # noqa: E402
from rwkv_metal.model.state import RWKVState                   # noqa: E402
from rwkv_metal.model.initial_state import (                   # noqa: E402
    attach_state, detach_state, attached_state, prepare_state_tuning, fold_prompt,
    base_fingerprint, save_state, load_state, StateBaseMismatch, zero_state, trained_parts)

PTH = os.path.expanduser(os.environ.get(
    "RWKV_PTH", "~/Develop/SwiftRWKV/.testdata/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth"))
RWKVQ = os.path.expanduser(os.environ.get(
    "RWKVQ_BASE", "~/Develop/WKV-kvant/files_0110/0p1b_reduction.rwkvq"))
RWKVQ_C = os.path.expanduser(os.environ.get(
    "RWKVQ_BASE_C", "~/Develop/WKV-kvant/files_0110/0p1b_compression.rwkvq"))
ALL = ("wkv", "tmix_shift", "cmix_shift")


def batch(B, T, seed=7):
    rs = np.random.RandomState(seed)
    ids = rs.randint(1, 60000, size=(B, T + 1)).astype(np.int32)
    return mx.array(ids[:, :-1]), mx.array(ids[:, 1:])


def random_state(model, seed=3, scale=0.05, shifts=True) -> RWKVState:
    """Ненулевое состояние: на нулях часть градиентов вырождена."""
    rs = np.random.RandomState(seed)
    z = zero_state(model)
    # сдвиги -- представимые в bf16: явное состояние приводится к типу потока
    # целиком (`RWKVState.astype`), подключённое -- по месту, и на числах вне
    # решётки bf16 два пути разошлись бы округлением, а не счётом
    f = lambda a, s: mx.array((rs.normal(size=a.shape) * s).astype(np.float32)) \
        .astype(mx.bfloat16).astype(mx.float32)
    return RWKVState(wkv=f(z.wkv, scale),
                     tmix_shift=f(z.tmix_shift, 0.5 if shifts else 0.0),
                     cmix_shift=f(z.cmix_shift, 0.5 if shifts else 0.0)).eval()


def same(a, b):
    return mx.array_equal(a.astype(mx.float32), b.astype(mx.float32)).item()


def maxdiff(a, b):
    return mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()


def state_equal(a: RWKVState, b: RWKVState):
    return same(a.wkv, b.wkv) and same(a.tmix_shift, b.tmix_shift) and same(a.cmix_shift, b.cmix_shift)


def lm_loss(m, a, b):
    return m.loss(a, b).astype(mx.float32)


def reference(model, x, y):
    loss, grads = nn.value_and_grad(model, lm_loss)(model, x, y)
    mx.eval(loss, grads)
    return loss.item(), dict(tree_flatten(grads))


def compare(what, model, x, y, chunk, bound):
    loss_ref, g_ref = reference(model, x, y)
    loss, grads = seq.value_and_grad(model, x, y, chunk)
    mx.eval(loss, grads)
    g = dict(tree_flatten(grads))
    assert set(g) == set(g_ref), (what, set(g) ^ set(g_ref))
    worst = 0.0
    for k in sorted(g_ref):
        r = g_ref[k].astype(mx.float32)
        scale = mx.abs(r).max().item()
        assert scale > 0, f"{what}: эталонный градиент {k} нулевой -- сверять нечего"
        assert g[k].dtype == g_ref[k].dtype, (what, k, g[k].dtype, g_ref[k].dtype)
        e = maxdiff(r, g[k]) / scale
        worst = max(worst, e)
        assert e <= bound, f"{what}: {k} разошёлся на {e:.2e} (допуск {bound:.0e})"
    assert abs(loss.item() - loss_ref) <= abs(loss_ref) * 1e-5, (what, loss.item(), loss_ref)
    print(f"  {what}: лосс {loss.item():.6f} против {loss_ref:.6f}; "
          f"худший градиент {worst:.1e} ({len(g_ref)} параметров)")
    return g


def base_snapshot(model):
    return {k: v for k, v in tree_flatten(model.parameters()) if ".state0." not in k}


def main():
    print(f"state-tuning, мутация: {MUT or 'нет'}")
    m = load_pretrained(PTH, verbose=False)[0]
    L = len(m.blocks)
    x, y = batch(2, 48)

    # ── 1. нулевое состояние счёта не меняет ──
    plain = m(x); mx.eval(plain)
    assert attached_state(m) is None
    attach_state(m)
    zero = m(x); mx.eval(zero)
    assert same(plain, zero), f"нулевое состояние изменило логиты на {maxdiff(plain, zero):.2e}"
    assert trained_parts(m) == () and not tree_flatten(m.blocks[0].state0.trainable_parameters())
    assert detach_state(m) is not None and attached_state(m) is None
    assert not any(".state0." in k for k, _ in tree_flatten(m.parameters())), "состояние не отключилось"
    print("  нулевое подключённое состояние: логиты побитно те же")

    # ── 2. подключённое == явное; явное вытесняет подключённое ──
    S = random_state(m)
    attach_state(m, S)
    back = attached_state(m)
    assert state_equal(back, S), "подключённое состояние не то, что подключали"
    att = m.body(x); mx.eval(att)
    detach_state(m)
    expl = m.body(x, state=S.repeat(2).astype(m.ln0.weight.dtype)); mx.eval(expl)
    none = m.body(x); mx.eval(none)
    assert same(att, expl), f"подключённое != явное: {maxdiff(att, expl):.2e}"
    assert maxdiff(att, none) > 1e-2, "состояние ни на что не влияет"
    attach_state(m, random_state(m, seed=5))
    over = m.body(x, state=S.repeat(2).astype(m.ln0.weight.dtype)); mx.eval(over)
    assert same(over, expl), "явное состояние не вытеснило подключённое"
    detach_state(m)
    # каждая часть состояния действует сама
    for part in ALL:
        z = zero_state(m)
        setattr(z, part, getattr(S, part))
        attach_state(m, z)
        d = maxdiff(m.body(x), none)
        detach_state(m)
        assert d > 1e-3, f"часть {part} состояния не действует ({d:.1e})"
    print(f"  подключённое == явное побитно; сдвиг к пустому старту {maxdiff(att, none):.2f}")

    # ── 3. склейка: промпт -> кэш -> продолжение == сплошной проход ──
    P, Q = 32, 32
    full_ids, _ = batch(1, P + Q, seed=21)
    whole = m.body(full_ids)[:, P:]; mx.eval(whole)
    cached = fold_prompt(m, full_ids[0, :P])
    assert cached.batch == 1 and cached.wkv.shape[0] == L
    attach_state(m, cached)
    cont = m.body(full_ids[:, P:]); mx.eval(cont)
    detach_state(m)
    e_plain = maxdiff(cont, whole) / mx.abs(whole).max().item()
    assert e_plain < 2e-2, f"свёрнутый промпт как старт расходится со сплошным проходом: {e_plain:.2e}"
    # обученное -> промпт -> кэш
    T0 = random_state(m, seed=9, shifts=False)
    attach_state(m, T0)
    whole_t = m.body(full_ids)[:, P:]; mx.eval(whole_t)
    cached_t = fold_prompt(m, full_ids[0, :P])
    cont_t = m.body(full_ids[:, P:], state=cached_t); mx.eval(cont_t)
    detach_state(m)
    e_tuned = maxdiff(cont_t, whole_t) / mx.abs(whole_t).max().item()
    assert e_tuned < 2e-2, f"обученное -> промпт -> кэш расходится со сплошным проходом: {e_tuned:.2e}"
    assert maxdiff(cached_t.wkv, cached.wkv) > 1e-4, "кэш промпта не зависит от подключённого состояния"
    print(f"  склейка: промпт как старт {e_plain:.1e}, обученное -> промпт -> кэш {e_tuned:.1e}")

    # ── 4. градиент: по блокам против автограда по всему графу ──
    base_before = base_snapshot(m)
    n = prepare_state_tuning(m, init=random_state(m, shifts=False))
    H, S_ = m.blocks[0].tmix.H, m.blocks[0].tmix.S
    assert n == L * H * S_ * S_, f"обучаемых чисел {n}, а у WKV {L * H * S_ * S_}"
    assert trained_parts(m) == ("wkv",)
    assert seq.supported(m)
    x1, y1 = batch(1, 48)
    g_all = compare("плотная, WKV, одна строка", m, x1, y1, chunk=0, bound=0.0)
    assert set(g_all) == {f"blocks.{l}.state0.wkv" for l in range(L)}
    compare("плотная, WKV, батч 2", m, x, y, chunk=0, bound=1e-6)
    compare("плотная, WKV, лосс кусками по 16", m, x, y, chunk=16, bound=1e-2)
    compare("плотная, WKV, длина не кратна 16", m, *batch(2, 40), chunk=0, bound=1e-6)

    # ── 5а. батч: градиент двух строк == среднее градиентов по строкам ──
    _, g2 = seq.value_and_grad(m, x, y, 0)
    _, ga = seq.value_and_grad(m, x[:1], y[:1], 0)
    _, gb = seq.value_and_grad(m, x[1:], y[1:], 0)
    mx.eval(g2, ga, gb)
    g2, ga, gb = (dict(tree_flatten(t)) for t in (g2, ga, gb))
    worst = max(maxdiff(g2[k], (ga[k] + gb[k]) * 0.5) / mx.abs(g2[k]).max().item() for k in g2)
    # котангенты идут через базу в bf16: сложение по строкам -- до решётки
    assert worst < 5e-3, f"градиент по батчу -- не среднее по строкам: {worst:.2e}"
    assert max(maxdiff(ga[k], gb[k]) / mx.abs(ga[k]).max().item() for k in ga) > 1e-2
    print(f"  батч 2 == среднее по строкам: {worst:.1e}")

    detach_state(m)
    prepare_state_tuning(m, init=random_state(m), train=ALL)
    assert trained_parts(m) == ALL
    g_s = compare("плотная, WKV и сдвиги, батч 2", m, x, y, chunk=0, bound=1e-6)
    assert len(g_s) == 3 * L
    detach_state(m)
    now = base_snapshot(m)
    assert all(same(now[k], base_before[k]) for k in base_before), "счёт градиента изменил базу"

    # ── 6. обучение ──
    def run(sequential, train=("wkv",)):
        model = load_pretrained(PTH, verbose=False)[0]
        before = base_snapshot(model)
        xb, yb = batch(2, 32, seed=13)
        cfg = StateTuneConfig(lr=0.05, lr_final=0.01, warmup_steps=0, max_steps=4, log_every=0,
                              cache_limit_gb=0, loss_chunk_tokens=0, train=train,
                              state_path="/tmp/_state_test.safetensors", note="гейт")
        assert StateTuneConfig().sequential_backward is True and StateTuneConfig().train == ("wkv",), "умолчания"
        cfg.sequential_backward = sequential
        calls, losses = seq.CALLS, []
        res = state_tune(model, [(xb, yb)], cfg, on_step=lambda s, l, p: losses.append(l))
        after = base_snapshot(model)
        assert all(same(after[k], before[k]) for k in before), "state-tuning изменил базу"
        assert [k for k, _ in tree_flatten(model.trainable_parameters())] == \
            [f"blocks.{l}.state0.wkv" for l in range(L)] or train != ("wkv",)
        return model, res, losses, seq.CALLS - calls
    m_seq, r_seq, l_seq, n_seq = run(True)
    assert n_seq == 4, f"state_tune не пошёл обратным проходом по блокам ({n_seq} вызовов)"
    assert l_seq[-1] < l_seq[0] - 1e-3, f"лосс не падает: {l_seq}"
    st = r_seq["state"]
    assert mx.abs(st.wkv).max().item() > 1e-3, "состояние не сдвинулось"
    assert mx.abs(st.tmix_shift).max().item() == 0 and mx.abs(st.cmix_shift).max().item() == 0, \
        "сдвиги обучались, хотя просили только WKV"
    assert state_equal(attached_state(m_seq), st), "после обучения подключено не то состояние"
    m_seq.set_dtype("bfloat16")
    assert m_seq.blocks[0].state0.wkv.dtype == mx.float32, "set_dtype привёл WKV состояния к типу параметров"
    assert state_equal(attached_state(m_seq), st), "set_dtype изменил WKV состояния"
    _, r_ref, l_ref, n_ref = run(False)
    assert n_ref == 0, "выключатель обратного прохода по блокам не действует"
    e = maxdiff(r_ref["state"].wkv, st.wkv) / mx.abs(st.wkv).max().item()
    assert e < 1e-4, f"по блокам и одним графом -- разные состояния: {e:.2e}"
    print(f"  state_tune: лосс {l_seq[0]:.4f} -> {l_seq[-1]:.4f}; по блокам против одного графа {e:.1e}")

    # ── 7. файл ──
    path = "/tmp/_state_test.safetensors"
    loaded, meta = load_state(path, m_seq)
    assert state_equal(loaded, st), "файл туда-обратно не побитно"
    assert loaded.wkv.dtype == mx.float32
    assert meta["kind"] == "tuned" and meta["trained"] == "wkv" and meta["note"] == "гейт", meta
    assert meta["n_layer"] == str(L) and meta["base_fingerprint"] == base_fingerprint(m_seq), meta
    again = "/tmp/_state_test_2.safetensors"
    save_state(again, loaded, m_seq, kind=meta["kind"], trained=("wkv",), note=meta["note"])
    assert open(path, "rb").read() == open(again, "rb").read(), "повторная запись дала другой файл"
    # отпечаток: один у bf16 и fp32, не меняется от подключённого состояния
    fp = base_fingerprint(m)
    assert fp == meta["base_fingerprint"], "отпечаток зависит от подключённого состояния"
    m32 = load_pretrained(PTH, verbose=False)[0]
    m32.set_dtype("float32")
    assert base_fingerprint(m32) == fp, "отпечаток зависит от типа, в котором загружена база"
    # fp32-источник: числа вне решётки bf16 (сдвиг много меньше её шага) -- та же база
    last = m32.blocks[L - 1]
    keep = (m32.ln0.weight, last.ln2.bias, last.tmix.r_k)
    m32.ln0.weight = keep[0] * (1 + 1e-6)
    last.tmix.r_k = keep[2] * (1 - 1e-6)
    assert not same(m32.ln0.weight, keep[0])
    assert base_fingerprint(m32) == fp, "fp32-источник и его bf16-загрузка дали разные отпечатки"
    m32.ln0.weight, last.tmix.r_k = keep[0], keep[2]
    # а настоящая разница в любом слое -- другая база
    last.ln2.bias = keep[1] + 1.0
    assert base_fingerprint(m32) != fp, "отпечаток не видит верхний слой"
    last.ln2.bias = keep[1]
    m32.ln_out.weight = m32.ln_out.weight * 1.5
    assert base_fingerprint(m32) != fp, "отпечаток не видит ln_out"
    del m32
    # чужая архитектура -- отказ всегда
    arrays, md = mx.load(path, return_metadata=True)
    bad = dict(md); bad["n_layer"] = str(L + 1)
    mx.save_safetensors("/tmp/_state_test_bad.safetensors", arrays, metadata=bad)
    for kwargs in ({}, {"allow_other_base": True}):
        try:
            load_state("/tmp/_state_test_bad.safetensors", m, **kwargs)
            raise AssertionError("состояние чужой архитектуры принято")
        except ValueError as err:
            assert not isinstance(err, StateBaseMismatch), "чужая архитектура выдана за чужую базу"
    notstate = dict(md); notstate["format"] = "lora"
    mx.save_safetensors("/tmp/_state_test_bad.safetensors", arrays, metadata=notstate)
    try:
        load_state("/tmp/_state_test_bad.safetensors")
        raise AssertionError("не-файл-состояния принят")
    except ValueError:
        pass
    try:
        save_state("/tmp/_state_test_bad.safetensors", st[0:1].repeat(2), m)
        raise AssertionError("состояние с батчем 2 записано")
    except ValueError:
        pass
    del m_seq
    mx.clear_cache()

    # ── квантованная база: градиент, отпечаток, отказ ──
    q = load_rwkvq_model(RWKVQ, rank=0, verbose=False)[0]
    fq = base_fingerprint(q)
    assert fq != fp, "плотная база и REDUCTION дали один отпечаток"
    try:
        load_state(path, q)
        raise AssertionError("состояние плотной базы принято на REDUCTION без разрешения")
    except StateBaseMismatch:
        pass
    foreign, _ = load_state(path, q, allow_other_base=True)
    assert state_equal(foreign, st)
    assert prepare_state_tuning(q, init=random_state(q, shifts=False)) == n
    assert seq.supported(q)
    compare(".rwkvq REDUCTION, WKV, батч 2", q, *batch(2, 32), chunk=16, bound=1e-2)
    detach_state(q)
    del q
    mx.clear_cache()
    c = load_rwkvq_model(RWKVQ_C, rank=0, verbose=False)[0]
    fc = base_fingerprint(c)
    assert fc not in (fp, fq), "COMPRESSION дал отпечаток другой базы"
    # те же числа обязан давать SwiftRWKV (StateTuningParityTests): решётка
    # строк сжатых проекций -- fp16, иначе отпечатки двух библиотек расходятся
    known = {"0p1b_reduction.rwkvq": "4e56802ad0a0ffaebe7f7ac7955e497a00c5f875dff67cf18ec0e277eebc495e", "0p1b_compression.rwkvq": "8df2516f7b79fb3884d18bf672e74b810de4681d531996e0a9b00ba3c3078117"}
    for path, got in ((RWKVQ, fq), (RWKVQ_C, fc)):
        want = known.get(os.path.basename(path))
        assert want is None or got == want, f"отпечаток {os.path.basename(path)} изменился: {got}"
    del c
    mx.clear_cache()

    # ── 5б. конечная разность по направлению (fp32, не зависит от автограда) ──
    m.set_dtype("float32")
    prepare_state_tuning(m, init=random_state(m, shifts=False))
    xf, yf = batch(1, 32, seed=17)
    _, g = seq.value_and_grad(m, xf, yf, 0)
    mx.eval(g)
    g = dict(tree_flatten(g))
    rs = np.random.RandomState(1)
    d = {k: mx.array(rs.normal(size=v.shape).astype(np.float32)) for k, v in g.items()}
    base = {f"blocks.{l}.state0.wkv": m.blocks[l].state0.wkv for l in range(L)}

    def at(eps, layer):
        k = f"blocks.{layer}.state0.wkv"
        m.blocks[layer].state0.wkv = base[k] + eps * d[k]
        return m.loss(xf, yf).astype(mx.float32).item()
    # По слоям, а не суммой: вклады слоёв разного знака гасят друг друга, и
    # сумма тонет в шуме fp32 лосса (шаг 1e-4: ~5e-3 в производной).
    eps, shown = 1e-4, []
    for layer in (0, L // 2, L - 1):
        k = f"blocks.{layer}.state0.wkv"
        analytic = (g[k] * d[k]).sum().item()
        numeric = (at(eps, layer) - at(-eps, layer)) / (2 * eps)
        at(0.0, layer)
        assert abs(numeric - analytic) <= 0.02 * abs(analytic) + 5e-3, \
            f"слой {layer}: конечная разность {numeric:.5f} против градиента {analytic:.5f}"
        shown.append(f"слой {layer}: {numeric:.4f} / {analytic:.4f}")
    assert abs((g["blocks.0.state0.wkv"] * d["blocks.0.state0.wkv"]).sum().item()) > 0.5
    print("  конечная разность против градиента -- " + "; ".join(shown))
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
                           env=dict(os.environ, STATE_MUT=name), capture_output=True, text=True)
        why = [l for l in (r.stdout + r.stderr).splitlines() if "Error" in l or "Mismatch" in l]
        print(f"  {name:10s} {'ПОЙМАНА' if r.returncode else 'НЕ ПОЙМАНА'}  {(why[-1] if why else '')[:110]}")
        if r.returncode == 0:
            missed.append(name)
    assert not broken and not missed, f"сломаны: {broken}, не пойманы: {missed}"
    print(f"все {len(MUTATIONS)} крышек держат")


if __name__ == "__main__":
    all_mutations() if "--mutations" in sys.argv else main()
