"""Подкачка весов с диска при обучении по блокам -- по выбору, не по умолчанию (24.10).

Порт SwiftRWKV `X070Backbone.loadStreaming(safetensors:)` (23.10). Обратный проход по
блокам (`lora/sequential.py`) держит в работе один блок, но веса всей модели при этом
лежат в памяти: у 1.5B в bf16 это 3 ГБ. С подкачкой модель собирается из плотного
.safetensors ЛЕНИВО (`mx.load` не читает данные до вычисления), и после каждого блока
прямого и обратного прохода его веса заменяются свежими ленивыми ссылками на тот же файл:
прочитанное освобождается, при следующем обращении читается снова. То же -- таблица
эмбеддингов после основания и голова после лосса.

Числа те же до бита: тот же файл, то же приведение типа. Цена -- чтение файла на каждом
проходе. Освобождение делают пути по блокам (`sequential.value_and_grad` и всё, что на нём:
state_tune, обучение адаптеров по блокам); проход одним графом (`model(x)`) веса прочитает
и оставит в памяти до `release_all`. Обучаемые параметры (состояние, адаптеры) не трогаются:
заменяются только веса из файла.

    model = load_streaming("g1j_1.5b.safetensors", dtype=mx.bfloat16)
"""
from typing import Dict, List

import mlx.core as mx
from mlx.utils import tree_unflatten

STEM = ["emb.weight"]
HEAD = ["head.weight"]


class WeightStream:
    """Откуда брать веса заново: файл, тип, имена по блокам."""

    def __init__(self, path: str, dtype, by_layer: List[List[str]]):
        self.path, self.dtype, self.by_layer = path, dtype, by_layer

    def fresh(self, keys: List[str]) -> Dict[str, mx.array]:
        """Свежие ленивые ссылки на веса `keys` (данные не читаются до вычисления)."""
        w = mx.load(self.path)
        missing = [k for k in keys if k not in w]
        assert not missing, f"подкачка весов: в {self.path} нет {missing[:3]}"
        return {k: w[k].astype(self.dtype) for k in keys}


def load_streaming(path: str, dtype=mx.bfloat16):
    """Модель RWKV-7 x070 из плотного .safetensors (формат `convert.save_converted`) с
    подкачкой весов. Числа всех путей -- те же, что у модели из того же файла в памяти."""
    from ..pretrain.config import PretrainConfig
    from .rwkv7_x070 import RWKV7X070
    w = mx.load(path)                                    # ленивые: данные не читаются
    layers = [int(k.split(".")[1]) for k in w if k.startswith("blocks.")]
    if not layers or "head.weight" not in w or "emb.weight" not in w or "blocks.0.tmix.k_k" not in w:
        raise ValueError(f"подкачка весов: {path} -- не плотный чекпоинт RWKV-7 (нет blocks.*, emb, head)")
    n_layer = 1 + max(layers)
    V, D = w["emb.weight"].shape
    S = w["blocks.0.tmix.k_k"].shape[1]
    lb = 1 if n_layer > 1 else 0                         # у слоя 0 нет ветки v (остаточный value)
    ranks = {k: int(w[f"blocks.{lb}.tmix.{k}_lora_A.weight"].shape[0]) for k in ("w", "a", "v", "g")
             if f"blocks.{lb}.tmix.{k}_lora_A.weight" in w}
    cfg = PretrainConfig(n_layer=n_layer, n_embd=D, vocab_size=V, head_size=S, ctx_len=512, batch_size=1,
                         train_data="", val_data="", max_steps=1)
    model = RWKV7X070(cfg, ranks=ranks)
    model.load_weights([(k, v.astype(dtype)) for k, v in w.items()], strict=True)
    by_layer = [sorted(k for k in w if k.startswith(f"blocks.{l}.")) for l in range(n_layer)]
    model._weight_stream = WeightStream(path, dtype, by_layer)
    return model


def streaming(model) -> bool:
    """Включена ли подкачка весов."""
    return getattr(model, "_weight_stream", None) is not None


def _put(model, fresh: Dict[str, mx.array]):
    by_block: Dict[int, list] = {}
    top = []
    for k, v in fresh.items():
        if k.startswith("blocks."):
            l, rest = k.split(".", 2)[1:]
            by_block.setdefault(int(l), []).append((rest, v))
        else:
            top.append((k, v))
    for l, items in by_block.items():
        model.blocks[l].update(tree_unflatten(items))
    if top:
        model.update(tree_unflatten(top))


def release_layer(model, layer: int):
    """Веса блока `layer` -- свежими ленивыми ссылками (прочитанные освобождаются)."""
    s = getattr(model, "_weight_stream", None)
    if s is not None:
        _put(model, s.fresh(s.by_layer[layer]))


def release_stem(model):
    """Таблица эмбеддингов -- после основания."""
    s = getattr(model, "_weight_stream", None)
    if s is not None:
        _put(model, s.fresh(STEM))


def release_head(model):
    """Голова -- после лосса."""
    s = getattr(model, "_weight_stream", None)
    if s is not None:
        _put(model, s.fresh(HEAD))


def release_all(model):
    """Отпустить прочитанные веса всех частей модели (после прохода одним графом)."""
    s = getattr(model, "_weight_stream", None)
    if s is not None:
        _put(model, s.fresh([k for ks in s.by_layer for k in ks] + STEM + HEAD))
