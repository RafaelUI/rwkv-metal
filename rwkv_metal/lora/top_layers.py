"""
rwkv_metal.lora.top_layers
==========================
Обучение верхних N слоёв над сжатой базой (.rwkvq). Порт SwiftRWKV
(`X070Backbone.unpackTopLayers` + `BaseTrainingMode.topLayers(N)`, 12.10).

На малых моделях (0.1B, 0.4B) адаптеры несут слишком мало параметров, и
полезный режим -- обучать сами верхние слои. Сжатые коды -- не веса, обучать их
нельзя, поэтому:

  * проекции верхних N блоков разворачиваются из сжатого вида в плотные
    nn.Linear (`dense_weight`), остальные параметры этих блоков приводятся к тому
    же типу; всё, что ниже, остаётся сжатым, замороженным и считается своими
    ядрами;
  * тип развёрнутых слоёв по умолчанию -- fp32, как у обучаемых весов SwiftRWKV:
    там подмена весов (`wOverride`) держит fp32-мастера, и fp16-вход, умноженный
    на fp32-вес, поднимает эти слои в fp32. Шаг Adam порядка 2e-5 на весах
    порядка 0.05 меньше шага решётки fp16 (~3e-5) -- в fp16 он бы терялся;
  * обучаются эти N блоков и ln_out (у эмбеддера ещё голова пулинга); база ниже,
    основание (emb, ln0) и голова языковой модели заморожены.

Обратный проход по блокам (`lora/sequential.py`, `embedding/sequential.py`)
берёт такую модель как есть: нижний обучаемый блок -- L - N.

    from rwkv_metal.lora import load_rwkvq_model, prepare_top_layers
    base = load_rwkvq_model("0p4b_reduction.rwkvq", rank=0)[0]
    n = prepare_top_layers(base, 4)          # обучаемых чисел
    model = EmbeddingModel(base)             # голова пулинга -- обучаемая
"""
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from .lora import LoRALinear, _linear_bias, dense_weight, is_linear_like


def _dense_linear(mod, dtype) -> nn.Linear:
    w = dense_weight(mod, dtype)
    bias = _linear_bias(mod)
    lin = nn.Linear(w.shape[1], w.shape[0], bias=bias is not None)
    lin.weight = w
    if bias is not None:
        lin.bias = bias.astype(dtype)
    return lin


def unpack_top_layers(model, n: int, dtype=mx.float32) -> int:
    """Развернуть проекции верхних `n` блоков в плотные nn.Linear типа `dtype` и
    привести остальные параметры этих блоков к нему же. Возвращает число
    развёрнутых проекций. Адаптеры LoRA на этих блоках не поддерживаются: здесь
    обучаются сами веса."""
    blocks = model.blocks
    L = len(blocks)
    if not 0 < n <= L:
        raise ValueError(f"верхних слоёв {n}, а у модели {L}")
    count = 0

    def replace_in(parent):
        nonlocal count
        for name, child in list(parent.children().items()):
            if isinstance(child, LoRALinear):
                raise ValueError("верхние слои с адаптерами LoRA: загрузите базу с rank=0")
            if isinstance(child, nn.Module) and is_linear_like(child):
                if type(child) is not nn.Linear:
                    setattr(parent, name, _dense_linear(child, dtype))
                    count += 1
            elif isinstance(child, nn.Module):
                replace_in(child)

    for blk in blocks[L - n:]:
        replace_in(blk)
        blk.update(tree_map(lambda v: v.astype(dtype), blk.parameters()))
    mx.eval([b.parameters() for b in blocks[L - n:]])
    return count


def prepare_top_layers(model, n: int, dtype=mx.float32) -> int:
    """Заморозить базу и сделать обучаемыми верхние `n` блоков (развёрнутыми,
    см. `unpack_top_layers`) и ln_out. Возвращает число обучаемых чисел базы."""
    unpack_top_layers(model, n, dtype)
    model.freeze()
    L = len(model.blocks)
    for blk in model.blocks[L - n:]:
        blk.unfreeze()
    model.ln_out.update(tree_map(lambda v: v.astype(dtype), model.ln_out.parameters()))
    model.ln_out.unfreeze()
    # у сжатых слоёв, если они остались в обучаемой части, градиента нет
    for _, m in model.named_modules():
        if is_linear_like(m) and not isinstance(m, nn.Linear):
            m.freeze()
    mx.eval(model.parameters())
    return sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
