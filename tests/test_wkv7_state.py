"""
Golden-тесты на рекуррентное состояние RWKV-7 (x070).

Проверяется ровно то, на чём стоит реранкер:

  1. Ядро: wkv7_train_with_state против питон-референса (fwd + VJP по h_in).
  2. Ядро: один шаг wkv7_step против того же референса.
  3. Модель: продолжение с состояния == сплошной проход
     (и по скрытым состояниям, и по итоговому состоянию).
  4. Модель: right-padding с маской не меняет НИ состояние, НИ скрытые
     состояния реальных позиций — то есть батчить строки разной длины можно
     без потери точности.
  5. Модель: батч из строк разной длины == набор одиночных проходов.

Запуск:  .venv/bin/python -m pytest tests/test_wkv7_state.py -q
     или .venv/bin/python tests/test_wkv7_state.py
"""
import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwkv_metal.kernel import (
    wkv7_train_with_state, wkv7_step, wkv7_train_py_with_state, HEAD_SIZE,
)
from rwkv_metal.model.state import RWKVState, build_mask
from rwkv_metal.model.rwkv7_x070 import RWKV7X070
from rwkv_metal.pretrain.config import PretrainConfig


def _rand_inputs(B, T, H, D, seed=0):
    rng = np.random.default_rng(seed)
    f = lambda *s: mx.array(rng.standard_normal(s).astype(np.float32))
    r = f(B, T, H, D) * 0.5
    k = f(B, T, H, D) * 0.5
    v = f(B, T, H, D) * 0.5
    # w в (0,1): затухание
    w = mx.sigmoid(f(B, T, H, D)) * 0.5 + 0.5
    kk = f(B, T, H, D)
    kk = kk / mx.sqrt((kk * kk).sum(-1, keepdims=True) + 1e-12)
    iclr = mx.sigmoid(f(B, T, H, D))
    return r, w, k, v, -kk, kk * iclr


def _maxdiff(a, b):
    return float(mx.abs(a - b).max())


# ── 1. Ядро: forward и VJP по h_in ───────────────────────────────────────────

def test_kernel_state_matches_reference():
    B, T, H, D = 2, 48, 3, HEAD_SIZE
    r, w, k, v, a, b = _rand_inputs(B, T, H, D, seed=1)
    rng = np.random.default_rng(7)
    h0 = mx.array(rng.standard_normal((B, H, D, D)).astype(np.float32)) * 0.1

    out_k, h_k = wkv7_train_with_state(r, w, k, v, a, b, h0)
    out_p, h_p = wkv7_train_py_with_state(r, w, k, v, a, b, h0)
    mx.eval(out_k, h_k, out_p, h_p)

    assert _maxdiff(out_k, out_p) < 1e-4, _maxdiff(out_k, out_p)
    assert _maxdiff(h_k, h_p) < 1e-4, _maxdiff(h_k, h_p)


def test_kernel_vjp_wrt_h_in():
    """dh_in из backward-ядра против того же VJP на питон-референсе.

    Это тот градиент, по которому реранкер учится «поверх» состояния.
    """
    B, T, H, D = 1, 32, 2, HEAD_SIZE
    r, w, k, v, a, b = _rand_inputs(B, T, H, D, seed=2)
    rng = np.random.default_rng(3)
    h0 = mx.array(rng.standard_normal((B, H, D, D)).astype(np.float32)) * 0.1
    cot_out = mx.array(rng.standard_normal((B, T, H, D)).astype(np.float32))
    cot_h = mx.array(rng.standard_normal((B, H, D, D)).astype(np.float32))

    def f_kernel(h):
        o, hh = wkv7_train_with_state(r, w, k, v, a, b, h)
        return (o * cot_out).sum() + (hh * cot_h).sum()

    def f_py(h):
        o, hh = wkv7_train_py_with_state(r, w, k, v, a, b, h)
        return (o * cot_out).sum() + (hh * cot_h).sum()

    g_kernel = mx.grad(f_kernel)(h0)
    g_py = mx.grad(f_py)(h0)
    mx.eval(g_kernel, g_py)

    rel = _maxdiff(g_kernel, g_py) / max(1e-9, float(mx.abs(g_py).max()))
    assert rel < 1e-4, rel


# ── 2. Одношаговый MLX-путь (T=1, используется реранкером) ───────────────────

def test_single_step_matches_reference():
    B, H, D = 3, 4, HEAD_SIZE
    r, w, k, v, a, b = _rand_inputs(B, 1, H, D, seed=4)
    rng = np.random.default_rng(5)
    h0 = mx.array(rng.standard_normal((B, H, D, D)).astype(np.float32)) * 0.2

    out_s, h_s = wkv7_step(r, w, k, v, a, b, h0)
    out_p, h_p = wkv7_train_py_with_state(r, w, k, v, a, b, h0)
    mx.eval(out_s, h_s, out_p, h_p)

    assert _maxdiff(out_s, out_p) < 1e-5, _maxdiff(out_s, out_p)
    assert _maxdiff(h_s, h_p) < 1e-5, _maxdiff(h_s, h_p)


# ── 3-5. Уровень модели ──────────────────────────────────────────────────────

def _tiny_model(seed=0, n_layer=3, n_embd=128, vocab=512):
    mx.random.seed(seed)
    cfg = PretrainConfig(n_layer=n_layer, n_embd=n_embd, vocab_size=vocab)
    model = RWKV7X070(cfg)
    # дефолтная инициализация MLX даёт слишком «плоскую» модель; чуть шумим,
    # чтобы состояние было содержательным
    from mlx.utils import tree_map
    model.update(tree_map(lambda p: p + mx.random.normal(p.shape) * 0.02,
                          model.parameters()))
    mx.eval(model.parameters())
    return model, cfg


def test_continuation_equals_full_pass():
    """Прогнать [док | запрос] целиком == прогнать док, затем запрос с его
    состояния. Это и есть кэш состояния документа."""
    model, cfg = _tiny_model(seed=11)
    rng = np.random.default_rng(12)
    T1, T2 = 37, 13
    idx = mx.array(rng.integers(1, cfg.vocab_size, size=(1, T1 + T2)).astype(np.int32))

    h_full, st_full = model.body(idx, return_state=True)
    h1, st1 = model.body(idx[:, :T1], return_state=True)
    h2, st2 = model.body(idx[:, T1:], state=st1, return_state=True)
    mx.eval(h_full, st_full.wkv, h2, st2.wkv)

    d_hidden = _maxdiff(h_full[:, T1:], h2)
    d_state = _maxdiff(st_full.wkv, st2.wkv)
    scale = float(mx.abs(st_full.wkv).max())
    assert d_hidden < 1e-3, f"скрытые состояния разошлись: {d_hidden}"
    assert d_state / max(1e-9, scale) < 1e-4, f"состояние разошлось: {d_state} (scale {scale})"


def test_padding_does_not_touch_state():
    """Right-padding с маской: и состояние, и скрытые состояния реальных
    позиций совпадают с прогоном без паддинга."""
    model, cfg = _tiny_model(seed=13)
    rng = np.random.default_rng(14)
    T = 29
    idx = mx.array(rng.integers(1, cfg.vocab_size, size=(1, T)).astype(np.int32))

    h_ref, st_ref = model.body(idx, return_state=True)

    pad = 23
    idx_p = mx.concatenate([idx, mx.zeros((1, pad), dtype=mx.int32)], axis=1)
    mask = build_mask([T], T + pad)
    end_idx = mx.array([T - 1])
    h_p, st_p = model.body(idx_p, mask=mask, end_idx=end_idx, return_state=True)
    mx.eval(st_ref.wkv, st_p.wkv, h_ref, h_p)

    scale = float(mx.abs(st_ref.wkv).max())
    assert _maxdiff(st_ref.wkv, st_p.wkv) / max(1e-9, scale) < 1e-5
    assert _maxdiff(st_ref.tmix_shift, st_p.tmix_shift) < 1e-4
    assert _maxdiff(st_ref.cmix_shift, st_p.cmix_shift) < 1e-4
    assert _maxdiff(h_ref, h_p[:, :T]) < 1e-4


def test_ragged_batch_equals_individual_passes():
    """Батч строк разной длины == набор одиночных проходов."""
    model, cfg = _tiny_model(seed=15)
    rng = np.random.default_rng(16)
    lens = [41, 17, 8]
    T = max(lens)
    seqs = [rng.integers(1, cfg.vocab_size, size=L).tolist() for L in lens]
    padded = [s + [0] * (T - len(s)) for s in seqs]
    idx = mx.array(np.array(padded, dtype=np.int32))
    mask = build_mask(lens, T)
    end_idx = mx.array([L - 1 for L in lens])

    st_batch = model.states(idx, mask=mask, end_idx=end_idx)
    mx.eval(st_batch.wkv)

    for i, s in enumerate(seqs):
        st_i = model.states(mx.array(np.array([s], dtype=np.int32)))
        mx.eval(st_i.wkv)
        scale = float(mx.abs(st_i.wkv).max())
        d = _maxdiff(st_batch.wkv[:, i:i + 1], st_i.wkv)
        assert d / max(1e-9, scale) < 1e-5, f"строка {i}: {d} (scale {scale})"


def test_state_roundtrip_batched_continuation():
    """Кэш документа, размноженный на несколько запросов: состояние одного
    документа + разные хвосты == прогоны пар целиком."""
    model, cfg = _tiny_model(seed=17)
    rng = np.random.default_rng(18)
    doc = rng.integers(1, cfg.vocab_size, size=45).tolist()
    queries = [rng.integers(1, cfg.vocab_size, size=n).tolist() for n in (11, 7, 5)]
    T = max(len(q) for q in queries)

    st_doc = model.states(mx.array(np.array([doc], dtype=np.int32)))
    st_rep = st_doc.repeat(len(queries))
    q_pad = np.array([q + [0] * (T - len(q)) for q in queries], dtype=np.int32)
    mask = build_mask([len(q) for q in queries], T)
    end_idx = mx.array([len(q) - 1 for q in queries])
    st_batch = model.states(mx.array(q_pad), mask=mask, end_idx=end_idx, state=st_rep)
    mx.eval(st_batch.wkv)

    for i, q in enumerate(queries):
        full = mx.array(np.array([doc + q], dtype=np.int32))
        st_full = model.states(full)
        mx.eval(st_full.wkv)
        scale = float(mx.abs(st_full.wkv).max())
        d = _maxdiff(st_batch.wkv[:, i:i + 1], st_full.wkv)
        assert d / max(1e-9, scale) < 1e-4, f"запрос {i}: {d} (scale {scale})"


def test_streaming_decode_matches_full_pass():
    """Потокенный декод с переносом состояния == пересчёт всего контекста.

    Это то, на чём держится пример потокового инференса в docs/inference.md.
    """
    model, cfg = _tiny_model(seed=19)
    rng = np.random.default_rng(20)
    prompt = rng.integers(1, cfg.vocab_size, size=23).tolist()
    tail = rng.integers(1, cfg.vocab_size, size=9).tolist()

    h, st = model.body(mx.array(np.array([prompt], np.int32)), return_state=True)
    outs = [h[:, -1]]
    for t in tail:
        h, st = model.body(mx.array(np.array([[t]], np.int32)), state=st,
                           return_state=True)
        outs.append(h[:, -1])
    stream = mx.stack(outs, axis=1)[:, :, :]                    # [1, 1+len(tail), D]

    full = model.body(mx.array(np.array([prompt + tail], np.int32)))
    ref = mx.concatenate([full[:, len(prompt) - 1:len(prompt)],
                          full[:, len(prompt):]], axis=1)
    mx.eval(stream, ref)

    scale = float(mx.abs(ref).max())
    d = _maxdiff(stream, ref)
    assert d / max(1e-9, scale) < 1e-4, f"{d} (scale {scale})"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
