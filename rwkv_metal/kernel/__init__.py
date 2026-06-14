"""
rwkv_metal.kernel
=================
WKV-7 ядро на Metal: forward / backward (checkpoint) / inference.

Публичные функции:
    wkv7            — единая точка входа (training | inference)
    wkv7_train      — обучение (autograd через Metal backward kernel)
    wkv7_infer      — пошаговый inference с состоянием
    wkv7_train_py   — Python-референс (отладка/проверка корректности)
    make_wkv7_checkpoint — фабрика checkpoint-kernel под фикс. (B, T, H, D)

Константы:
    HEAD_SIZE = 64   — размерность головы (фиксирована ядром)
    CHUNK     = 32   — размер чанка для checkpoint-стратегии
"""
from .wkv7 import (
    wkv7,
    wkv7_train,
    wkv7_infer,
    wkv7_train_py,
    HEAD_SIZE,
    CHUNK,
)
from .wkv7_checkpoint import make_wkv7_checkpoint

__all__ = [
    "wkv7",
    "wkv7_train",
    "wkv7_infer",
    "wkv7_train_py",
    "make_wkv7_checkpoint",
    "HEAD_SIZE",
    "CHUNK",
]
