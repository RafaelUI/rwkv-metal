"""
rwkv_metal.pretrain
===================
Предобучение RWKV-7 с нуля на Apple Silicon.

Быстрый старт:
    from rwkv_metal.pretrain import pretrain, preset

    pretrain(preset("25m",
        train_data="data/train.bin",
        val_data="data/val.bin",
        vocab_size=21248,
    ))
"""
from .config import PretrainConfig, preset, PRESETS
from .trainer import pretrain
from .dataset import (
    load_dataset,
    tokenize_to_bin,
    BinDataset,
    TextDataset,
)

__all__ = [
    "PretrainConfig",
    "preset",
    "PRESETS",
    "pretrain",
    "load_dataset",
    "tokenize_to_bin",
    "BinDataset",
    "TextDataset",
]
