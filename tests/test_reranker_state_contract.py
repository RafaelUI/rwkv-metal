"""ГЕЙТ: подключённое начальное состояние в контрактах кэшей реранкера (17.10).

Индекс префиксов (`DocIndex`) и кэш состояний пар (`StateCache`) хранят
результат счёта, а счёт стартует с подключённого к базе состояния. Кэш, снятый
с одним состоянием, при другом даёт правдоподобные и неверные числа. Поэтому:

1. ОТПЕЧАТОК `state_fingerprint` -- тот же бит в бит, что в SwiftRWKV
   (константа на фиксированном состоянии); None и нули -- `none`; каждая из
   трёх частей и форма входят в отпечаток, тип хранения -- нет.
2. ИНДЕКС: с другим состоянием и без состояния -- отказ; с тем же -- принят,
   скоры побитно те же; индекс без поля (`DocIndex(...)` без него) -- `none`.
3. КЭШ ПАР: пишет отпечаток, переживает save/load; старый .idx.npz без
   ключа -- `none`; `check_initial_state` отвергает чужое; обучение не берёт
   отложенный кэш с другим состоянием.

МУТАЦИОННЫЕ КРЫШКИ (RSC_MUT=...): правка вносится в исходник до импорта.

    python tests/test_reranker_state_contract.py
    python tests/test_reranker_state_contract.py --mutations
"""
import importlib.abc
import importlib.machinery
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MUT = os.environ.get("RSC_MUT", "")
INIT = "rwkv_metal/model/initial_state.py"
RR = "rwkv_metal/reranker/rerank.py"
ENC = "rwkv_metal/reranker/encode.py"
TRN = "rwkv_metal/reranker/train.py"
MUTATIONS = {
    "fp_nocmix": (INIT, '("cmix_shift", state.cmix_shift)]', ']'),
    "fp_noshape": (INIT, '(",".join(map(str, a.shape)) + "\\n")', '"\\n"'),
    "fp_zero": (INIT, "    if all(not a.any() for _, a in arrs):\n        return NO_INITIAL_STATE\n", ""),
    # хеш по байтам типа хранения, а не fp32 (прежняя крышка «без astype» была
    # равносильной: ascontiguousarray(dtype="<f4") приводит тип сам)
    "fp_dtype": (INIT, 'np.ascontiguousarray(np.asarray(a.astype(mx.float32)), dtype="<f4")',
                       "np.ascontiguousarray(np.asarray(a))"),
    "idx_build": (RR, "initial_state=initial_state_fingerprint(self.model.base))", ")"),
    "idx_check": (RR, "        if index.initial_state != now:", "        if False:"),
    "cache_enc": (ENC, "        initial_state=initial_state_fingerprint(reranker.base),\n", ""),
    "cache_save": (ENC, ",\n                 initial_state=np.array(self.initial_state))", ")"),
    "cache_load": (ENC, "        return StateCache(states, idx[\"pair_index\"], idx[\"labels\"],\n                          initial_state=state)",
                        "        return StateCache(states, idx[\"pair_index\"], idx[\"labels\"])"),
    "cache_check": (ENC, "        if self.initial_state != now:", "        if False:"),
    "train_eval": (TRN, "    if eval_cache is not None and eval_cache.initial_state != train_cache.initial_state:",
                        "    if False:"),
}

if MUT:
    assert MUT in MUTATIONS, f"нет мутации {MUT}"
    _target, _a, _b = MUTATIONS[MUT]
    _target_path = os.path.join(ROOT, _target)
    sys.dont_write_bytecode = True

    class _Loader(importlib.machinery.SourceFileLoader):
        def path_stats(self, path):
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
import numpy as np                         # noqa: E402

from rwkv_metal.model.convert import load_pretrained            # noqa: E402
from rwkv_metal.model.state import RWKVState                    # noqa: E402
from rwkv_metal.model.initial_state import (                    # noqa: E402
    attach_state, detach_state, zero_state, state_fingerprint, initial_state_fingerprint)
from rwkv_metal.reranker import (                               # noqa: E402
    Reranker, RerankerConfig, RerankerInference, RerankTrainConfig, StateCache,
    build_candidates, encode_pairs)
from rwkv_metal.reranker.rerank import DocIndex                  # noqa: E402
from rwkv_metal.reranker.train import train_reranker            # noqa: E402
from rwkv_metal.tokenizer.world_tokenizer import WorldTokenizer  # noqa: E402

PTH = os.path.expanduser(os.environ.get(
    "RWKV_PTH", "~/Develop/SwiftRWKV/.testdata/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth"))
VOCAB = os.path.expanduser("~/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt")
SWIFT_FIXED = "sha256:b51da87efab20d186e0e30949566886fd0e293594a3ffe31e672b336e08b2f58"

DOCS = ["Пчёлы опыляют растения и производят мёд из нектара цветов.",
        "Паровая машина Уатта запустила промышленную революцию в Англии.",
        "Медоносная пчела танцем сообщает улью направление к взятку."]
QUERY = "как пчёлы делают мёд"


def fixed_state() -> RWKVState:
    return RWKVState(
        wkv=mx.array(((np.arange(64) - 20) * 0.125).reshape(2, 1, 2, 4, 4).astype(np.float32)),
        tmix_shift=mx.array((np.arange(16) * 0.25 - 1).reshape(2, 1, 1, 8).astype(np.float32)),
        cmix_shift=mx.array((np.arange(16) * -0.5 + 3).reshape(2, 1, 1, 8).astype(np.float32)))


def random_state(model, seed, scale=0.05) -> RWKVState:
    """Значения на решётке bf16: подключённые сдвиги приводятся к типу потока
    по месту, и вне решётки индексный и прямой путь разошлись бы округлением."""
    rs = np.random.RandomState(seed)
    z = zero_state(model)
    f = lambda a: mx.array((rs.normal(size=a.shape) * scale).astype(np.float32)) \
        .astype(mx.bfloat16).astype(mx.float32)
    return RWKVState(wkv=f(z.wkv), tmix_shift=f(z.tmix_shift), cmix_shift=f(z.cmix_shift)).eval()


def raises(fn, what):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"не отвергнуто: {what}")


def check(cond, what):
    if not cond:
        raise AssertionError(what)
    print(f"  ок: {what}")


def main():
    print(f"контракт состояния реранкера, мутация: {MUT or 'нет'}")
    # 1. отпечаток
    s = fixed_state()
    f = state_fingerprint(s)
    check(f == SWIFT_FIXED, "отпечаток совпал со SwiftRWKV бит в бит")
    check(state_fingerprint(None) == "none", "None -> none")
    z = RWKVState(wkv=mx.zeros(s.wkv.shape), tmix_shift=mx.zeros(s.tmix_shift.shape),
                  cmix_shift=mx.zeros(s.cmix_shift.shape))
    check(state_fingerprint(z) == "none", "нули -> none")
    bump = lambda a: a + (mx.arange(a.size).reshape(a.shape) == a.size - 1).astype(mx.float32)
    variants = [f,
                state_fingerprint(RWKVState(bump(s.wkv), s.tmix_shift, s.cmix_shift)),
                state_fingerprint(RWKVState(s.wkv, bump(s.tmix_shift), s.cmix_shift)),
                state_fingerprint(RWKVState(s.wkv, s.tmix_shift, bump(s.cmix_shift))),
                state_fingerprint(RWKVState(s.wkv.reshape(2, 1, 4, 2, 4), s.tmix_shift, s.cmix_shift))]
    check(len(set(variants)) == len(variants), "каждая часть и форма входят в отпечаток")
    half = RWKVState(s.wkv.astype(mx.float16), s.tmix_shift.astype(mx.float16),
                     s.cmix_shift.astype(mx.float16))
    check(state_fingerprint(half) == f, "тип хранения в отпечаток не входит")

    # 2. индекс
    base, _ = load_pretrained(PTH, verbose=False)
    tok = WorldTokenizer(VOCAB)
    mx.random.seed(0)
    rr = Reranker(base, RerankerConfig(), head_dtype=mx.float32)
    rs = np.random.RandomState(17)
    w = rr.head.parameters()
    # голова не вырождена: штатный fc2 = 0 даёт нулевые скоры при любом счёте
    from mlx.utils import tree_flatten, tree_unflatten
    flat = dict(tree_flatten(w))
    flat = {k: (mx.array(rs.normal(size=v.shape).astype(np.float32) * 0.3) if "fc2" in k else v)
            for k, v in flat.items()}
    rr.head.update(tree_unflatten(list(flat.items())))
    inf = RerankerInference(rr, tok, max_doc_tokens=48, max_query_tokens=16, compile=False)
    sA, sB = random_state(base, 3), random_state(base, 4)

    check(initial_state_fingerprint(base) == "none", "без состояния -- none")
    plain_index = inf.build_index(DOCS)
    check(plain_index.initial_state == "none", "индекс без состояния -- none")
    attach_state(base, sA)
    index = inf.build_index(DOCS)
    check(index.initial_state == state_fingerprint(sA), "индекс записал отпечаток A")
    ref = inf.score_indexed(QUERY, index)
    direct_a = np.asarray(inf.score(QUERY, DOCS))
    dd = np.abs(np.asarray(ref) - direct_a).max()
    # база bf16: индексный и прямой путь расходятся округлением (без состояния
    # замерено 4e-3 при скорах ~7, то есть 5e-4 относительно)
    check(dd < 2e-3 * np.abs(direct_a).max(), f"индекс A сходится с прямым путём при A ({dd:.2g})")
    detach_state(base)
    d = np.abs(np.asarray(inf.score(QUERY, DOCS)) - direct_a).max()
    check(d > 10 * dd, f"состояние меняет скоры (на {d:.3g}) -- отказ не придирка")
    attach_state(base, sA)
    raises(lambda: inf.score_indexed(QUERY, plain_index), "индекс без состояния при A")
    attach_state(base, sB)
    raises(lambda: inf.score_indexed(QUERY, index), "индекс A при B")
    detach_state(base)
    raises(lambda: inf.score_indexed(QUERY, index), "индекс A без состояния")
    old = DocIndex(state=plain_index.state, docs=plain_index.docs, instruct=plain_index.instruct,
                   doc_first=plain_index.doc_first, max_doc_tokens=plain_index.max_doc_tokens)
    check(np.array_equal(inf.score_indexed(QUERY, old), inf.score_indexed(QUERY, plain_index)),
          "старый индекс без поля принят без состояния, побитно")
    attach_state(base, zero_state(base))
    inf.score_indexed(QUERY, old)
    print("  ок: нулевое состояние -- как без него")
    attach_state(base, sA)
    raises(lambda: inf.score_indexed(QUERY, old), "старый индекс при A")
    check(np.array_equal(inf.score_indexed(QUERY, index), ref), "тот же индекс при A -- скоры побитно те же")
    detach_state(base)
    attach_state(base, sB)
    attach_state(base, sA)
    check(np.array_equal(inf.score_indexed(QUERY, index), ref), "после переключений -- побитно те же")

    # 3. кэш пар
    rows = [{"anchor": f"Instruct: инструкция\nQuery: вопрос номер {i}",
             "positive": f"правильный пассаж про тему {i}",
             "negative": f"похожий но неверный пассаж {i}"} for i in range(4)]
    pool, samples = build_candidates(rows, n_candidates=3, seed=0)
    kw = dict(max_doc_tokens=32, max_query_tokens=16, dtype=mx.float32, verbose=False)
    with tempfile.TemporaryDirectory() as d:
        cache = encode_pairs(rr, tok, pool, samples, out_path=os.path.join(d, "a"), **kw)
        check(cache.initial_state == state_fingerprint(sA), "кэш пар записал отпечаток A")
        loaded = StateCache.load(os.path.join(d, "a"))
        check(loaded.initial_state == cache.initial_state, "отпечаток пережил save/load")
        loaded.check_initial_state(rr)
        print("  ок: кэш A принят при A")
        attach_state(base, sB)
        raises(lambda: loaded.check_initial_state(rr), "кэш A при B")
        detach_state(base)
        raises(lambda: loaded.check_initial_state(base), "кэш A без состояния")
        plain = encode_pairs(rr, tok, pool, samples, **kw)
        check(plain.initial_state == "none", "кэш пар без состояния -- none")
        # .idx.npz, записанный до появления ключа
        idx = np.load(os.path.join(d, "a.idx.npz"))
        np.savez(os.path.join(d, "a.idx.npz"), pair_index=idx["pair_index"], labels=idx["labels"])
        old_cache = StateCache.load(os.path.join(d, "a"))
        check(old_cache.initial_state == "none", "старый кэш без ключа -- none")
        old_cache.check_initial_state(rr)
        attach_state(base, sA)
        raises(lambda: old_cache.check_initial_state(rr), "старый кэш при A")
        detach_state(base)
        raises(lambda: train_reranker(rr, plain, eval_cache=cache,
                                      cfg=RerankTrainConfig(epochs=1)),
               "отложенный кэш с другим состоянием")
    print("ГЕЙТ ПРОЙДЕН")


def all_mutations():
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
                           env=dict(os.environ, RSC_MUT=name), capture_output=True, text=True)
        why = [l for l in (r.stdout + r.stderr).splitlines() if "Error" in l]
        print(f"  {name:12s} {'ПОЙМАНА' if r.returncode else 'НЕ ПОЙМАНА'}  {(why[-1] if why else '')[:110]}")
        if r.returncode == 0:
            missed.append(name)
    assert not broken and not missed, f"сломаны: {broken}, не пойманы: {missed}"
    print(f"все {len(MUTATIONS)} крышек держат")


if __name__ == "__main__":
    all_mutations() if "--mutations" in sys.argv else main()
