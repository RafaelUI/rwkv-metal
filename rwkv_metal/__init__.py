"""
rwkv_metal — RWKV-7 on Apple Silicon (Metal / MLX)
==================================================
ImpulseLeap's framework for pretraining and LoRA/QLoRA fine-tuning RWKV-7 with a
custom Metal WKV-7 kernel (forward + backward + inference).

Public API
----------
Models:
    RWKV7, init_weights            from-scratch training
    RWKV7X070, load_pretrained     load official x070 World weights

WKV-7 kernel (Metal):
    wkv7, wkv7_train, wkv7_infer

Pretraining:
    PretrainConfig, preset, pretrain, load_dataset, tokenize_to_bin

LoRA / QLoRA:
    add_lora, save_lora, load_lora, merge_lora
    LoRAConfig, finetune, quantize_base_model

Tokenizer:
    WorldTokenizer                 official World 65536-token vocab

Quick start (pretraining):
    import rwkv_metal as rk
    rk.pretrain(rk.preset("25m",
        train_data="data/train.bin",
        val_data="data/val.bin",
        vocab_size=21248,
    ))

Quick start (LoRA fine-tune of official weights):
    import rwkv_metal as rk
    from rwkv_metal.lora import LoRAConfig, finetune, quantize_base_model

    model, cfg = rk.load_pretrained("weights/RWKV-x070-World-1.5B.pth")
    quantize_base_model(model, bits=4)
    model, info = rk.add_lora(model, rank=16, alpha=16.0,
                              quantize_base=4, layers=range(12, 24))
    finetune(model, batches, LoRAConfig(lr=1e-4, grad_accum=8, max_steps=2000))
"""

__version__ = "0.1.0"

# ── Models ──────────────────────────────────────────────────────────────────
from .model import RWKV7, init_weights, RWKV7X070, load_pretrained, lora_ranks

# ── WKV-7 kernel ──────────────────────────────────────────────────────────────
from .kernel import (
    wkv7,
    wkv7_train,
    wkv7_infer,
    HEAD_SIZE,
    CHUNK,
)

# ── Pretraining ───────────────────────────────────────────────────────────────
from .pretrain import (
    PretrainConfig,
    preset,
    PRESETS,
    pretrain,
    load_dataset,
    tokenize_to_bin,
)

# ── LoRA / QLoRA ──────────────────────────────────────────────────────────────
from .lora import (
    add_lora,
    save_lora,
    load_lora,
    merge_lora,
    lora_state,
    LoRALinear,
    LoRAConfig,
    finetune,
    quantize_base_model,
)

# ── Tokenizer ─────────────────────────────────────────────────────────────────
from .tokenizer import WorldTokenizer

__all__ = [
    "__version__",
    # models
    "RWKV7",
    "init_weights",
    "RWKV7X070",
    "load_pretrained",
    "lora_ranks",
    # kernel
    "wkv7",
    "wkv7_train",
    "wkv7_infer",
    "HEAD_SIZE",
    "CHUNK",
    # pretraining
    "PretrainConfig",
    "preset",
    "PRESETS",
    "pretrain",
    "load_dataset",
    "tokenize_to_bin",
    # lora
    "add_lora",
    "save_lora",
    "load_lora",
    "merge_lora",
    "lora_state",
    "LoRALinear",
    "LoRAConfig",
    "finetune",
    "quantize_base_model",
    # tokenizer
    "WorldTokenizer",
]
