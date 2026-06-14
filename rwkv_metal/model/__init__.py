"""
rwkv_metal.model
================
RWKV-7 "Goose" architectures on MLX.

Two architectures, deliberately separate:

    RWKV7       - from-scratch training reference (LayerNorm ln_x, hardcoded
                  low-rank size 64, inter-block token-shift carry). Pair with
                  init_weights() for training from scratch.

    RWKV7X070   - exact x070 architecture for loading OFFICIAL weights
                  (GroupNorm ln_x by head, low-rank ranks derived from D,
                  zero-pad token-shift). Use with load_pretrained().

Both expose the same LoRA target names (tmix.{r,k,v,o}_proj), so the LoRA engine
works with either.

Loading official weights:
    from rwkv_metal.model import load_pretrained
    model, cfg = load_pretrained("weights/RWKV-x070-World-1.5B.pth")
"""
from .rwkv7 import (
    RWKV7,
    init_weights,
    RWKVBlock,
    RWKV_Tmix_x070,
    RWKV_CMix_x070,
    l2_norm,
)
from .rwkv7_x070 import RWKV7X070, lora_ranks
from .convert import load_pretrained, save_converted, load_pth, convert

__all__ = [
    # from-scratch
    "RWKV7",
    "init_weights",
    "RWKVBlock",
    "RWKV_Tmix_x070",
    "RWKV_CMix_x070",
    "l2_norm",
    # official weights
    "RWKV7X070",
    "lora_ranks",
    "load_pretrained",
    "save_converted",
    "load_pth",
    "convert",
]
