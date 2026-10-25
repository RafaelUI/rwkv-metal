"""ГЕЙТ ПОДКАЧКИ ВЕСОВ (24.10): `model/weight_stream.py`, порт SwiftRWKV loadStreaming.

Крошечная модель (6 слоёв, D 512, словарь 4096 -- голова сравнима с блоком) в fp32 в
файле, модель -- в bf16: приведение при перечитывании обязано совпасть с приведением при
сборке.

1. ЧИСЛА: логиты, градиент состояния по блокам (дважды подряд), запись с шагом и обучение
   состояния -- те же бит в бит, что у модели из того же файла в памяти.
2. ПАМЯТЬ (живые массивы MLX): после шага прочитанного не остаётся (отпустить всё --
   освобождать нечего); пик шага -- как у модели в памяти плюс читаемые блоки; пересчёт
   входа при записи с шагом не копит веса.
3. ОТКАЗ: файл без модели.

МУТАЦИОННЫЕ КРЫШКИ (WS_MUT=...):  python tests/test_weight_stream.py [--mutations]
"""
import importlib.abc
import importlib.machinery
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MUT = os.environ.get("WS_MUT", "")
WS = "rwkv_metal/model/weight_stream.py"
SEQ = "rwkv_metal/lora/sequential.py"
MUTATIONS = {
    "fwd": (SEQ, "        mx.eval(h, v_first)\n        _ws.release_layer(model, l)", "        mx.eval(h, v_first)"),
    "bwd": (SEQ, "        mx.eval(ready)\n        _ws.release_layer(model, l)", "        mx.eval(ready)"),
    "restore": (SEQ, "        mx.eval(h)\n        _ws.release_layer(model, j)", "        mx.eval(h)"),
    "head": (SEQ, "    mx.eval(dx, loss)\n    _ws.release_head(model)", "    mx.eval(dx, loss)"),
    "stem": (SEQ, "        mx.eval(h)\n        _ws.release_stem(model)", "        mx.eval(h)"),
    "dtype": (WS, "        return {k: w[k].astype(self.dtype) for k in keys}", "        return {k: w[k] for k in keys}"),
    "neighbor": (WS, "        _put(model, s.fresh(s.by_layer[layer]))",
                     "        _put(model, s.fresh(s.by_layer[(layer + 1) % len(s.by_layer)]))"),
    "refuse": (WS, "        raise ValueError(", "        print("),
    "nostream": (WS, "    model._weight_stream = WeightStream(path, dtype, by_layer)", "    pass"),
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
from mlx.utils import tree_flatten         # noqa: E402

import rwkv_metal.lora.sequential as seq                      # noqa: E402
import rwkv_metal.model.weight_stream as ws                   # noqa: E402
from rwkv_metal.lora import StateTuneConfig, state_tune       # noqa: E402
from rwkv_metal.model.initial_state import prepare_state_tuning, zero_state   # noqa: E402
from rwkv_metal.model.rwkv7_x070 import RWKV7X070             # noqa: E402
from rwkv_metal.model.state import RWKVState                  # noqa: E402
from rwkv_metal.pretrain.config import PretrainConfig         # noqa: E402

L, D, V = 6, 512, 4096


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and mx.array_equal(a, b).item()


def make_file(path):
    mx.random.seed(42)
    cfg = PretrainConfig(n_layer=L, n_embd=D, vocab_size=V, head_size=64, ctx_len=64, batch_size=1,
                         train_data="", val_data="", max_steps=1)
    m = RWKV7X070(cfg)
    params = {k: (v * 0.5).astype(mx.float32) for k, v in tree_flatten(m.parameters())}
    mx.save_safetensors(path, params)


def resident(path):
    m = ws.load_streaming(path, dtype=mx.bfloat16)
    m._weight_stream = None                      # та же сборка, но веса остаются в памяти
    mx.eval(m.parameters())
    return m


def batch(B=1, T=32, seed=7):
    rs = np.random.RandomState(seed)
    ids = rs.randint(1, V, size=(B, T + 1)).astype(np.int32)
    return mx.array(ids[:, :-1]), mx.array(ids[:, 1:])


def state(model, seed=3):
    rs = np.random.RandomState(seed)
    z = zero_state(model)
    return RWKVState(wkv=mx.array((rs.normal(size=z.wkv.shape) * 0.05).astype(np.float32)),
                     tmix_shift=z.tmix_shift, cmix_shift=z.cmix_shift).eval()


def block_bytes(m, l=1):
    return sum(v.nbytes for k, v in tree_flatten(m.blocks[l].parameters()) if "state0" not in k)


def main():
    tmp = tempfile.mkdtemp(prefix="swr_ws_")
    path = os.path.join(tmp, "tiny.safetensors")
    make_file(path)
    s = ws.load_streaming(path, dtype=mx.bfloat16)
    r = resident(path)
    assert ws.streaming(s) and not ws.streaming(r), "подкачка не включилась"

    # ── 1. числа ──
    x, y = batch(B=2)
    assert same(s(x), r(x)), "логиты разошлись"
    ws.release_all(s)
    init = state(r)
    prepare_state_tuning(s, init=init)
    prepare_state_tuning(r, init=init)
    xb, yb = batch()
    for stride in (1, 1, 3):
        ls, gs = seq.value_and_grad(s, xb, yb, loss_chunk=0, trace_stride=stride)
        lr, gr = seq.value_and_grad(r, xb, yb, loss_chunk=0, trace_stride=stride)
        fs, fr = tree_flatten(gs), tree_flatten(gr)
        assert same(ls, lr), f"лосс разошёлся (запись с шагом {stride})"
        assert len(fs) == len(fr) and all(same(a, b) for (_, a), (_, b) in zip(fs, fr)), \
            f"градиент состояния разошёлся (запись с шагом {stride})"
    print("  числа: логиты, градиент по блокам дважды и с записью через 3 -- бит в бит")

    # ── 2. память ──
    mx.set_cache_limit(0)
    block = block_bytes(r)
    head = r.head.weight.nbytes

    def measure(m):
        seq.value_and_grad(m, xb, yb, loss_chunk=0)          # разогрев
        mx.clear_cache()
        before = mx.get_active_memory()
        mx.reset_peak_memory()
        loss, g = seq.value_and_grad(m, xb, yb, loss_chunk=0)
        mx.eval(loss, g)
        peak = mx.get_peak_memory()
        del g
        mx.clear_cache()
        return peak - before, mx.get_active_memory() - before

    pr, kr = measure(r)
    ps, ks = measure(s)
    print(f"  блок {block // 1000} КБ, голова {head // 1000} КБ; пик сверх начала: в памяти {pr // 1000} КБ, "
          f"подкачка {ps // 1000} КБ; осталось после шага {kr // 1000} / {ks // 1000} КБ")
    assert ks < min(block, head) // 2, f"после шага остались прочитанные веса ({ks} байт)"
    mx.clear_cache()
    held = mx.get_active_memory()
    ws.release_all(s)
    mx.clear_cache()
    freed = held - mx.get_active_memory()
    assert freed < min(block, head) // 2, f"после шага держались прочитанные веса ({freed} байт)"
    allw = sum(v.nbytes for k, v in tree_flatten(r.parameters()) if "state0" not in k)
    assert 0 < ps < pr + 3 * block + allw // 8, f"пик подкачки {ps} выше весов в памяти {pr} и читаемого блока"
    # пересчёт входа при записи с шагом не копит веса
    h, block_input, v_first = seq.forward_trace(s, xb, 0, stride=3)
    mx.clear_cache()
    before = mx.get_active_memory()
    seq._restore_input(s, block_input, 5, 0, v_first)
    mx.clear_cache()
    grew = mx.get_active_memory() - before
    assert grew < block // 2, f"пересчёт входа оставил веса ({grew} байт)"
    del h, block_input, v_first
    ws.release_all(s)
    print("  память: один блок в работе, после шага прочитанного не остаётся")

    # ── 3. обучение и отказ ──
    base = dict(lr=0.05, lr_final=0.01, warmup_steps=0, max_steps=3, log_every=0, cache_limit_gb=0,
                loss_chunk_tokens=0, state_path=None)
    s2 = ws.load_streaming(path, dtype=mx.bfloat16)
    r2 = resident(path)
    a = state_tune(s2, [batch()], StateTuneConfig(**base), init=init)["state"].wkv
    b = state_tune(r2, [batch()], StateTuneConfig(**base), init=init)["state"].wkv
    assert same(a, b), "обученное состояние разошлось"
    junk = os.path.join(tmp, "junk.safetensors")
    mx.save_safetensors(junk, {"x": mx.zeros((2,))})
    try:
        ws.load_streaming(junk)
        raise AssertionError("не-модель принята")
    except ValueError as e:                  # понятный отказ, а не KeyError из глубины сборки
        assert "не плотный чекпоинт" in str(e), e
    print("  обучение состояния -- то же; не-модель -- отказ")
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
                           env=dict(os.environ, WS_MUT=name), capture_output=True, text=True)
        why = [l for l in (r.stdout + r.stderr).splitlines() if "Error" in l]
        print(f"  {name:9s} {'ПОЙМАНА' if r.returncode else 'НЕ ПОЙМАНА'}  {(why[-1] if why else '')[:110]}")
        if r.returncode == 0:
            missed.append(name)
    assert not broken and not missed, f"сломаны: {broken}, не пойманы: {missed}"
    print(f"все {len(MUTATIONS)} крышек держат")


if __name__ == "__main__":
    all_mutations() if "--mutations" in sys.argv else main()
