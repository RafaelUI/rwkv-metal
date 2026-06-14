"""
Удобная точка входа для запуска предобучения без установки пакета:

    cd rwkv-metal
    python pretrain.py --preset 25m --train_data data/train.bin --val_data data/val.bin

Вся логика CLI живёт в rwkv_metal/pretrain/cli.py (она же доступна как команда
`rwkv-metal-pretrain` после `pip install`).
"""

import os
import sys

# Позволяет запускать без установки пакета (python pretrain.py из rwkv-metal/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rwkv_metal.pretrain.cli import main

if __name__ == "__main__":
    main()
