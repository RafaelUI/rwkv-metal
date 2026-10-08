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

    def merged_weight(self, dtype=None):
        """Плотный вес базы + дельта адаптера. База -- любая (см. dense_weight):
        прежде здесь стояло `self.linear.weight + delta`, что у nn.QuantizedLinear
        складывало дельту с УПАКОВАННЫМИ uint32-кодами, а у баз .rwkvq падало
        (атрибута weight у них нет). dtype=None -- тип плотной базы, у
        квантованной -- тип адаптеров."""
        if dtype is None:
            dtype = (self.linear.weight.dtype if type(self.linear) is nn.Linear
                     else self.lora_a.dtype)
        return dense_weight(self, dtype)


def is_linear_like(mod) -> bool:
    """Линейный слой любой базы: nn.Linear, nn.QuantizedLinear, LoRALinear и
    замороженные базы .rwkvq (RwkvqLinear / RwkvqSymLinear / RwkvqDenseLinear --
    по _dequant_w_as, RwkvqNativeLinear / RwkvqHybridLinear -- по wq)."""
    return (isinstance(mod, (nn.Linear, nn.QuantizedLinear, LoRALinear))
            or hasattr(mod, "_dequant_w_as")
            or (isinstance(mod, nn.Module) and "wq" in mod and hasattr(mod, "bits")))


def dense_weight(mod, dtype=None) -> mx.array:
    """Плотный вес [OUT, IN] линейного слоя любой базы (08.10).

    Зачем. Всё, что читает веса модели по именам (`init_from_base`
    реранкера, merge_lora, экспорт), видело у квантованной базы не
    `weight`, а коды и масштабы, и молча пропускало такие слои: голова
    реранкера над .rwkvq-базой стартовала со СЛУЧАЙНЫМИ проекциями.

    Деквант -- в fp32 (значения .rwkvq точны в fp16, fp32 их не портит),
    LoRALinear -- база плюс scale * B @ A в fp32. dtype=None -- fp32, кроме
    плотного nn.Linear, который отдаётся в своём типе как есть."""
    from .rwkvq_native import GROUP_SIZE
    if isinstance(mod, LoRALinear):
        w = dense_weight(mod.linear, mx.float32)
        w = w + mod.scale * (mod.lora_b.astype(mx.float32) @ mod.lora_a.astype(mx.float32))
    elif isinstance(mod, nn.QuantizedLinear):
        w = mx.dequantize(mod.weight, mod.scales, mod.biases,
                          group_size=mod.group_size, bits=mod.bits).astype(mx.float32)
    elif isinstance(mod, nn.Linear):
        w = mod.weight
        return w if dtype is None else w.astype(dtype)
    elif hasattr(mod, "_dequant_w_as"):
        w = mod._dequant_w_as(mx.float32)
    elif isinstance(mod, nn.Module) and "wq" in mod:
        if hasattr(mod, "_expand_scale_bias"):          # RwkvqHybridLinear
            scale, bias = mod._expand_scale_bias()
        else:                                           # RwkvqNativeLinear
            scale, bias = mod.scale, mod.bias
        w = mx.dequantize(mod.wq, scale, bias, group_size=GROUP_SIZE,
                          bits=mod.bits).astype(mx.float32)
    else:
        raise TypeError(f"dense_weight: {type(mod).__name__} -- не линейный слой")
    return w.astype(dtype or mx.float32)


def _linear_bias(mod):
    """Смещение линейного слоя. Только у nn.Linear / nn.QuantizedLinear: у
    RwkvqNativeLinear `bias` -- это сдвиг КВАНТОВАНИЯ [OUT, NB], не слоя."""
    base = mod.linear if isinstance(mod, LoRALinear) else mod
    if isinstance(base, (nn.Linear, nn.QuantizedLinear)) and "bias" in base:
        return base["bias"]
    return None


def dense_parameters(module, dtype=None) -> dict:
    """Плоский словарь параметров `module` в ПЛОТНОМ виде: каждый линейный
    слой любой базы -- как `<путь>.weight` (+ `.bias`, если есть), остальное --
    как в tree_flatten(module.parameters()). Ключи совпадают с ключами той же
    архитектуры на плотных nn.Linear, поэтому результат годится для update()
    плотной копии (голова реранкера, экспорт)."""
    out = {}

    def walk(m, prefix):
        if is_linear_like(m):
            out[prefix + "weight"] = dense_weight(m, dtype)
            b = _linear_bias(m)
            if b is not None:
                out[prefix + "bias"] = b
            return
        for k, v in m.items():
            if k.startswith("_"):
                continue
            if isinstance(v, mx.array):
                out[prefix + k] = v
            elif isinstance(v, nn.Module):
                walk(v, f"{prefix}{k}.")
            elif isinstance(v, (list, tuple)):
                for i, c in enumerate(v):
                    if isinstance(c, nn.Module):
                        walk(c, f"{prefix}{k}.{i}.")
                    elif isinstance(c, mx.array):
                        out[f"{prefix}{k}.{i}"] = c
            elif isinstance(v, dict):
                for kk, vv in tree_flatten(v):
                    if isinstance(vv, mx.array):
                        out[f"{prefix}{k}.{kk}"] = vv

    walk(module, "")
    return out


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


def _merged_linear(lora: "LoRALinear", dtype=None) -> nn.Linear:
    """LoRALinear -> плотный nn.Linear с влитым адаптером. Плотная база
    переиспользуется (как прежде); квантованная (QLoRA) становится плотной:
    слить дельту в коды без переквантования нельзя, а переквантование --
    дело rwkv-quant, не этого слияния."""
    w = lora.merged_weight(dtype)
    if type(lora.linear) is nn.Linear:
        lora.linear.weight = w
        return lora.linear
    OUT, IN = w.shape
    lin = nn.Linear(IN, OUT, bias=_linear_bias(lora) is not None)
    lin.weight = w
    if "bias" in lin:
        lin.bias = _linear_bias(lora)
    return lin


def merge_lora(model, dtype=None):
    """In-place слияние LoRA обратно в базу (для inference/экспорта).

    Плотная база -- как прежде: вес nn.Linear получает дельту в своём типе.
    Квантованная база (QLoRA: nn.QuantizedLinear или .rwkvq) -- слой
    становится плотным nn.Linear в типе адаптеров (bf16) либо `dtype`;
    память этого слоя растёт до плотной. Незавёрнутые квантованные слои
    (cmix, head у .rwkvq) остаются как есть."""
    def replace_in(parent):
        for cname, child in list(parent.children().items()):
            if isinstance(child, LoRALinear):
                setattr(parent, cname, _merged_linear(child, dtype))
            elif isinstance(child, nn.Module):
                replace_in(child)
            elif isinstance(child, list):
                for i, c in enumerate(child):
                    if isinstance(c, LoRALinear):
                        child[i] = _merged_linear(c, dtype)
                    elif isinstance(c, nn.Module):
                        replace_in(c)
    replace_in(model)
    model.unfreeze()
    # квантованные слои, оставшиеся квантованными, обучаемыми не становятся:
    # у их целочисленных кодов градиента нет
    for _, m in model.named_modules():
        if is_linear_like(m) and not isinstance(m, nn.Linear):
            m.freeze()
    mx.eval(model.parameters())
    return model
