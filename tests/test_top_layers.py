"""ГЕЙТ: обучение верхних слоёв над сжатой базой (lora/top_layers.py).

1. Развёртка: плотный вес каждой проекции верхних блоков -- побитно
   `dense_weight` сжатой (в fp32); ниже -- ничего не тронуто (те же объекты).
2. Обучаемое -- ровно верхние N блоков и ln_out, всё в fp32; база ниже,
   основание и голова языковой модели заморожены; нижний обучаемый блок L - N.
3. Эмбеддер: градиент обратным проходом по блокам (с кусками и шагом записи)
   против автограда по всему графу; REDUCTION и COMPRESSION.
4. finetune_embedding: три шага двигают верхние слои и не трогают низ
   (побитно), лосс падает.
5. Отказы: адаптеры LoRA на верхних слоях, N вне [1, L].

МУТАЦИОННЫЕ КРЫШКИ (TOP_MUT=...): правка в исходник top_layers.py до импорта.

    python tests/test_top_layers.py              # гейт
    python tests/test_top_layers.py --mutations  # все крышки
"""
import importlib.abc
import importlib.machinery
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

MUT = os.environ.get("TOP_MUT", "")
SRC = os.path.join(ROOT, "rwkv_metal/lora/top_layers.py")
MUTATIONS = {
    "dtype": ("blk.update(tree_map(lambda v: v.astype(dtype), blk.parameters()))", "pass"),
    "weight": ("    lin.weight = w\n", "    lin.weight = w * 0.5\n"),
    "range": ("    for blk in model.blocks[L - n:]:\n        blk.unfreeze()",
              "    for blk in model.blocks[L - n + 1:]:\n        blk.unfreeze()"),
    "freeze": ("    model.freeze()\n    L = len(model.blocks)", "    L = len(model.blocks)"),
    "lnout": ("    model.ln_out.unfreeze()", "    pass"),
    "lnout32": ("    model.ln_out.update(tree_map(lambda v: v.astype(dtype), model.ln_out.parameters()))", "    pass"),
    "unpack": ("                if type(child) is not nn.Linear:", "                if False:"),
    "lora": ('                raise ValueError("верхние слои с адаптерами LoRA: загрузите базу с rank=0")', "                pass"),
    "bounds": ("    if not 0 < n <= L:", "    if False:"),
}
if MUT:
    assert MUT in MUTATIONS, MUT
    _a, _b = MUTATIONS[MUT]
    sys.dont_write_bytecode = True

    class _Loader(importlib.machinery.SourceFileLoader):
        def path_stats(self, path):
            raise OSError

        def get_data(self, path):
            data = super().get_data(path)
            if os.path.abspath(path) == SRC:
                src = data.decode("utf-8")
                assert src.count(_a) == 1, f"мутация {MUT}: {src.count(_a)} вхождений"
                data = src.replace(_a, _b).encode("utf-8")
            return data

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is not None and spec.origin and os.path.abspath(spec.origin) == SRC:
                spec.loader = _Loader(name, spec.origin)
            return spec

    sys.meta_path.insert(0, _Finder())

import mlx.core as mx                                   # noqa: E402
import mlx.nn as nn                                     # noqa: E402
import numpy as np                                      # noqa: E402
from mlx.utils import tree_flatten                      # noqa: E402

from rwkv_metal.lora import load_rwkvq_model, prepare_top_layers, unpack_top_layers   # noqa: E402
from rwkv_metal.lora.lora import dense_weight          # noqa: E402
import rwkv_metal.lora.sequential as body               # noqa: E402
import rwkv_metal.embedding.sequential as seq           # noqa: E402
from rwkv_metal.embedding import tasks                  # noqa: E402
from rwkv_metal.embedding.train import EmbeddingModel, EmbedTrainConfig, finetune_embedding   # noqa: E402
from test_embedding_sequential import Pieces, triplet, liven   # noqa: E402

K = os.path.expanduser("~/Develop/WKV-kvant/files_0110/")
RED, COM = K + "0p1b_reduction.rwkvq", K + "0p1b_compression.rwkvq"
TIGHT = 2e-5


def load(path):
    return load_rwkvq_model(path, rank=0, verbose=False)[0]


def reference(model, batch, chunk):
    wrap = (lambda m: Pieces(m, chunk)) if chunk else (lambda m: m)
    v, g = nn.value_and_grad(model, lambda m, b: tasks.retrieval_loss(wrap(m), b, 0.05).astype(mx.float32))(model, batch)
    mx.eval(v, g)
    return v.item(), dict(tree_flatten(g))


def compare(what, model, chunk=0, stride=1):
    b = triplet()
    lref, gref = reference(model, b, chunk)
    v, g = seq.value_and_grad(model, b, tasks.retrieval_loss, 0.05, chunk, stride)
    mx.eval(v, g)
    g = dict(tree_flatten(g))
    assert set(g) == set(gref), (what, sorted(set(g) ^ set(gref))[:4])
    worst = 0.0
    for k in gref:
        r = gref[k].astype(mx.float32)
        s = mx.abs(r).max().item()
        assert s > 0, f"{what}: эталонный градиент {k} нулевой"
        worst = max(worst, mx.abs(r - g[k].astype(mx.float32)).max().item() / s)
    assert worst <= TIGHT, f"{what}: градиент разошёлся на {worst:.2e}"
    assert abs(v.item() - lref) <= abs(lref) * 1e-5, (what, v.item(), lref)
    print(f"  {what}: лосс {v.item():.6f} против {lref:.6f}; худший градиент {worst:.1e} ({len(gref)} параметров)")


def main():
    print(f"верхние слои над сжатой базой, мутация: {MUT or 'нет'}")
    q = load(RED)
    L = len(q.blocks)
    low_before = {k: v for k, v in tree_flatten(q.blocks[0].parameters())}
    dense_ref = {name: dense_weight(getattr(q.blocks[L - 1].tmix, name), mx.float32)
                 for name in ("r_proj", "k_proj", "v_proj", "o_proj")}
    dense_ref["cmix.value"] = dense_weight(q.blocks[L - 1].cmix.value, mx.float32)
    low_mod = q.blocks[L - 3].tmix.r_proj
    mx.eval(dense_ref)

    q.unfreeze()                  # база пришла обучаемой (как плотная): заморозка -- дело prepare
    n = prepare_top_layers(q, 2)
    # 1. развёртка
    for name, w in dense_ref.items():
        mod = q.blocks[L - 1].cmix.value if name == "cmix.value" else getattr(q.blocks[L - 1].tmix, name)
        assert type(mod) is nn.Linear, f"{name} не развёрнут: {type(mod).__name__}"
        assert mx.array_equal(mod.weight, w).item(), f"{name}: развёрнутый вес не равен dense_weight"
    assert q.blocks[L - 3].tmix.r_proj is low_mod, "развёрнут слой ниже верхних"
    assert type(low_mod) is not nn.Linear
    # 2. обучаемое
    names = [k for k, _ in tree_flatten(q.trainable_parameters())]
    assert all(k.startswith((f"blocks.{L - 2}.", f"blocks.{L - 1}.", "ln_out.")) for k in names), names[:5]
    assert any(k.startswith(f"blocks.{L - 2}.") for k in names) and "ln_out.weight" in names
    assert all(v.dtype == mx.float32 for _, v in tree_flatten(q.trainable_parameters())), "обучаемое не в fp32"
    assert body.first_layer(q) == L - 2
    assert n == sum(v.size for _, v in tree_flatten(q.trainable_parameters()))
    print(f"  REDUCTION, верхние 2: обучаемых чисел {n / 1e6:.2f} М, нижний обучаемый блок {L - 2}")

    # 3. градиент
    m = EmbeddingModel(q)
    liven(m)
    compare("REDUCTION, верхние 2", m)
    compare("REDUCTION, верхние 2, куски и шаг 2", m, chunk=40, stride=2)
    del m, q
    mx.clear_cache()
    c = load(COM)
    prepare_top_layers(c, 3)
    m = EmbeddingModel(c)
    liven(m)
    compare("COMPRESSION, верхние 3", m)
    del m, c
    mx.clear_cache()

    # 4. обучение
    q = load(RED)
    prepare_top_layers(q, 2)
    m = EmbeddingModel(q)
    low = {k: v for k, v in tree_flatten(q.blocks[0].parameters())}
    top = {k: v for k, v in tree_flatten(q.blocks[L - 1].parameters())}
    losses = []
    cfg = EmbedTrainConfig(lr=1e-3, max_steps=4, log_every=0, cache_limit_gb=0,
                           checkpoint_path="/tmp/_top_layers_test.safetensors")
    calls = seq.CALLS
    finetune_embedding(m, [triplet()], cfg, tasks.retrieval_loss, on_step=lambda s, l, p: losses.append(l))
    assert seq.CALLS - calls == 4, "обучение не пошло обратным проходом по блокам"
    assert all(mx.array_equal(v, dict(tree_flatten(q.blocks[0].parameters()))[k]).item() for k, v in low.items()), \
        "обучение тронуло нижние слои"
    moved = [k for k, v in tree_flatten(q.blocks[L - 1].parameters()) if not mx.array_equal(v, top[k]).item()]
    assert len(moved) > len(top) // 2, f"верхние слои почти не сдвинулись: {len(moved)} из {len(top)}"
    assert losses[-1] < losses[0], f"лосс не падает: {losses}"
    print(f"  finetune_embedding: лосс {losses[0]:.4f} -> {losses[-1]:.4f}; низ не тронут, сдвинулось {len(moved)} из {len(top)}")

    # 5. отказы
    for bad in (0, L + 1):
        try:
            unpack_top_layers(load(RED), bad)
            raise AssertionError(f"N={bad} принят")
        except ValueError:
            pass
    ql = load_rwkvq_model(RED, rank=4, alpha=8.0, verbose=False, layers=[L - 1])[0]
    try:
        unpack_top_layers(ql, 1)
        raise AssertionError("верхние слои с адаптерами приняты")
    except ValueError:
        pass
    print("ГЕЙТ ПРОЙДЕН")


def all_mutations():
    import subprocess
    missed = []
    for name in MUTATIONS:
        r = subprocess.run([sys.executable, os.path.abspath(__file__)], env=dict(os.environ, TOP_MUT=name),
                           capture_output=True, text=True)
        why = [l for l in (r.stdout + r.stderr).splitlines() if "Error" in l]
        print(f"  {name:8s} {'ПОЙМАНА' if r.returncode else 'НЕ ПОЙМАНА'}  {(why[-1] if why else '')[:100]}")
        if r.returncode == 0:
            missed.append(name)
    assert not missed, f"не пойманы: {missed}"
    print(f"все {len(MUTATIONS)} крышек держат")


if __name__ == "__main__":
    all_mutations() if "--mutations" in sys.argv else main()
