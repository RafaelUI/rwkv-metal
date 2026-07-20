"""
LoRA для RWKV-7 (MLX).

Адаптеры ставятся на проекции внутри tmix-блока. Градиенты для
r_proj / k_proj / v_proj текут через наш Metal WKV backward kernel
(wkv7_checkpoint), для o_proj — напрямую после WKV.

Структура целевой модели (rwkv_metal/model/rwkv7.py):
    model.blocks[i].tmix.{r_proj,k_proj,v_proj,o_proj}   (nn.Linear, bias=False)
    model.blocks[i].cmix.{key,value}                     (FFN, опционально)

Заморозка: nn.Module.freeze() + точечный unfreeze адаптеров.
ВАЖНО: обучать через nn.value_and_grad(model, fn) — он уважает
trainable_parameters() (freeze()). Обычный mx.value_and_grad
дифференцирует всё дерево и заморозку игнорирует.
"""

import math
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


class LoRALinear(nn.Module):
    """Обёртка над nn.Linear: y = W·x (frozen) + (alpha/r)·B(A(x))."""

    def __init__(self, linear: nn.Linear = None, rank: int = 16, alpha: float = 32.0,
                 dropout: float = 0.0, quantize_base: int = 0,
                 q_group_size: int = 64, base_module=None, dtype=None):
        super().__init__()
        # base_module: готовый frozen-модуль с (out_features, in_features) и
        # __call__(x) -- напр. RwkvqLinear (rwkvq_linear.py) поверх нативного
        # .rwkvq-квантования rwkv-quant, вместо стокового nn.QuantizedLinear.
        # Тогда `linear` не нужен (dims/dtype берутся из base_module).
        if base_module is not None:
            out_features, in_features = base_module.out_features, base_module.in_features
            dtype = dtype or mx.bfloat16
            self.linear = base_module
        else:
            # dims берём из исходного nn.Linear ДО возможной квантизации
            out_features, in_features = linear.weight.shape
            dtype = linear.weight.dtype
            if quantize_base:
                # QLoRA: замороженная база в 4/8-бит, адаптеры в исходном dtype
                self.linear = nn.QuantizedLinear.from_linear(
                    linear, group_size=q_group_size, bits=quantize_base)
            else:
                self.linear = linear

        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        scale_a = 1.0 / math.sqrt(in_features)
        self.lora_a = mx.random.normal((rank, in_features)).astype(dtype) * scale_a
        self.lora_b = mx.zeros((out_features, rank)).astype(dtype)

    def __call__(self, x):
        base = self.linear(x)
        z = x @ self.lora_a.T
        if self.dropout is not None:
            z = self.dropout(z)
        return base + self.scale * (z @ self.lora_b.T)

    def merged_weight(self):
        delta = self.scale * (self.lora_b @ self.lora_a)
        return self.linear.weight + delta.astype(self.linear.weight.dtype)


TMIX_TARGETS = ("r_proj", "k_proj", "v_proj", "o_proj")
CMIX_TARGETS = ("key", "value")


def add_lora(model, rank: int = 16, alpha: float = 32.0, dropout: float = 0.0,
             tmix_targets=TMIX_TARGETS, cmix_targets=(), quantize_base: int = 0,
             q_group_size: int = 64, layers=None):
    """Оборачивает целевые nn.Linear в LoRALinear, замораживает всё кроме адаптеров.

    quantize_base: 0 = bf16 база; 4 или 8 = QLoRA (база в N-бит, адаптеры в bf16).
    """
    wrapped = []
    def mk(mod):
        return LoRALinear(mod, rank, alpha, dropout, quantize_base, q_group_size)
    n_layer = len(model.blocks)
    sel = set(range(n_layer)) if layers is None else set(
        i % n_layer for i in layers)  # поддержка отрицательных индексов
    for li, blk in enumerate(model.blocks):
        if li not in sel:
            continue
        for name in tmix_targets:
            mod = getattr(blk.tmix, name, None)
            if isinstance(mod, nn.Linear):
                setattr(blk.tmix, name, mk(mod))
                wrapped.append(f"tmix.{name}")
        for name in cmix_targets:
            mod = getattr(blk.cmix, name, None)
            if isinstance(mod, nn.Linear):
                setattr(blk.cmix, name, mk(mod))
                wrapped.append(f"cmix.{name}")

    model.freeze()
    _unfreeze_adapters(model)
    mx.eval(model.parameters())

    info = _param_stats(model)
    info["wrapped_per_block"] = sorted(set(wrapped))
    info["num_adapters"] = len(wrapped)
    return model, info


def _unfreeze_adapters(model):
    def visit(m):
        if isinstance(m, LoRALinear):
            m.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
        if isinstance(m, nn.Module):
            for _, child in m.children().items():
                if isinstance(child, nn.Module):
                    visit(child)
                elif isinstance(child, list):
                    for c in child:
                        if isinstance(c, nn.Module):
                            visit(c)
    visit(model)


def _param_stats(model):
    total = sum(v.size for _, v in tree_flatten(model.parameters()))
    train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    return {
        "total_params": total,
        "trainable_params": train,
        "trainable_pct": 100.0 * train / max(1, total),
    }


def lora_state(model):
    return dict(tree_flatten(model.trainable_parameters()))


def save_lora(model, path):
    mx.save_safetensors(path, lora_state(model))


def load_lora(model, path):
    weights = mx.load(path)
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model


def merge_lora(model):
    """In-place слияние LoRA обратно в базовые nn.Linear (для inference/экспорта)."""
    def replace_in(parent):
        for cname, child in list(parent.children().items()):
            if isinstance(child, LoRALinear):
                base = child.linear
                base.weight = child.merged_weight()
                setattr(parent, cname, base)
            elif isinstance(child, nn.Module):
                replace_in(child)
            elif isinstance(child, list):
                for i, c in enumerate(child):
                    if isinstance(c, LoRALinear):
                        base = c.linear
                        base.weight = c.merged_weight()
                        child[i] = base
                    elif isinstance(c, nn.Module):
                        replace_in(c)
    replace_in(model)
    model.unfreeze()
    mx.eval(model.parameters())
    return model
