"""
rwkv_metal.lora
===============
LoRA / QLoRA fine-tuning for RWKV-7 on Apple Silicon.

Adapters are placed on the tmix projections (r/k/v/o_proj); gradients flow
through the Metal WKV backward kernel. QLoRA (4/8-bit frozen base) is supported.

Low-level engine:
    add_lora, save_lora, load_lora, merge_lora, lora_state, LoRALinear

High-level fine-tuning (validated recipe baked in):
    LoRAConfig, finetune, quantize_base_model

The validated recipe (see docs/lora.md):
    - forward via mlx.nn.utils.checkpoint(block)
    - nn.value_and_grad(model, loss_fn)  (NOT mx.value_and_grad)
    - no mx.compile
    - mx.set_cache_limit(int(1.5e9))
    - QLoRA: quantize only big frozen matrices (cmix.key/value, head, emb)
"""
from .lora import (
    LoRALinear,
    add_lora,
    save_lora,
    load_lora,
    merge_lora,
    lora_state,
    TMIX_TARGETS,
    CMIX_TARGETS,
)
from .finetune import (
    LoRAConfig,
    finetune,
    quantize_base_model,
    BIG_QUANT_TARGETS,
)

__all__ = [
    # engine
    "LoRALinear",
    "add_lora",
    "save_lora",
    "load_lora",
    "merge_lora",
    "lora_state",
    "TMIX_TARGETS",
    "CMIX_TARGETS",
    # high-level fine-tuning
    "LoRAConfig",
    "finetune",
    "quantize_base_model",
    "BIG_QUANT_TARGETS",
]
