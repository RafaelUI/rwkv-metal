# rwkv-metal

**RWKV-7 on Apple Silicon — pretraining, LoRA/QLoRA, and a custom Metal WKV-7 kernel.**

`rwkv-metal` is [ImpulseLeap](https://impulseleap.com)'s framework for training
and fine-tuning [RWKV-7 "Goose"](https://github.com/BlinkDL/RWKV-LM) models on
Apple Silicon (M-series) using [MLX](https://github.com/ml-explore/mlx). The
WKV-7 recurrence — the part that doesn't map onto standard ops — runs as a
hand-written Metal kernel with a checkpointed backward pass, so training is
fast and fits in unified memory.

- Train RWKV-7 **from scratch** with a simple config.
- **LoRA / QLoRA** fine-tune your own checkpoints or official RWKV-7 World weights.
- A custom **Metal WKV-7 kernel** (forward + checkpointed backward + inference).
- Designed for **16 GB** Macs: bf16, gradient checkpointing, QLoRA 4-bit base.

> Status: early (v0.1). The kernel and training/LoRA stacks are validated; APIs
> may still change.

---

## Install

Requires macOS on Apple Silicon and Python 3.10+.

```bash
git clone <repo>
cd rwkv-metal
pip install -e .
# optional extras:
pip install -e ".[data]"    # tokenizers, for .txt -> .bin tokenization
pip install -e ".[wandb]"   # Weights & Biases logging
```

You can also run without installing, from the repo root:

```bash
python pretrain.py --preset 25m --train_data data/train.bin --val_data data/val.bin
```

---

## Quick start

### Pretraining from scratch

```python
import rwkv_metal as rk

rk.pretrain(rk.preset("25m",
    train_data = "data/train.bin",      # uint16 token ids, or a .txt + tokenizer
    val_data   = "data/val.bin",
    vocab_size = 21248,
    max_tokens = 3_000_000_000,
))
```

See **[docs/pretraining.md](docs/pretraining.md)** for the full config, presets,
data formats, precision, and memory guidance.

### LoRA / QLoRA fine-tuning

```python
import rwkv_metal as rk
from rwkv_metal.lora import LoRAConfig, finetune, quantize_base_model

# Load official RWKV-7 World weights (torch-free .pth loader) + World tokenizer
model, cfg = rk.load_pretrained("weights/RWKV-x070-World-1.5B.pth")
tok = rk.WorldTokenizer()

# QLoRA: 4-bit frozen base, LoRA on the top 12 layers
quantize_base_model(model, bits=4)
model, info = rk.add_lora(model, rank=16, alpha=16.0,
                          quantize_base=4, layers=range(12, 24))
print(f"trainable: {info['trainable_pct']:.3f}%")

finetune(model, batches, LoRAConfig(lr=1e-4, grad_accum=8, max_steps=2000))
```

See **[docs/lora.md](docs/lora.md)** for the full LoRA/QLoRA guide and the
validated low-memory recipe.

---

## What's inside

```
rwkv_metal/
├── kernel/       Metal WKV-7 kernel: forward, checkpointed backward, inference
│                 + a pure-Python reference for correctness checks
├── model/        RWKV7 (from-scratch) and RWKV7X070 (official x070 weights)
│                 + torch-free .pth loader / converter
├── pretrain/     PretrainConfig, presets, dataset loaders, training loop, CLI
├── lora/         LoRA/QLoRA engine, high-level finetune(), QLoRA helpers
└── tokenizer/    RWKV World tokenizer (65536-token vocab)
```

Two architectures share the same LoRA target names, so the LoRA engine works
with either:

| | `RWKV7` | `RWKV7X070` |
|---|---|---|
| Purpose | train from scratch | load official weights |
| `ln_x` | LayerNorm | GroupNorm (per head) |
| low-rank size | fixed 64 | derived from model width |
| pair with | `init_weights()` | `load_pretrained()` |

---

## The Metal WKV-7 kernel

The WKV-7 recurrence is bandwidth-bound (~1 FLOP/byte) and sequential, so it
doesn't fit standard fused ops. The kernel handles the whole sequence in one
forward dispatch and one backward dispatch, checkpointing the hidden state every
32 tokens so the backward pass reconstructs each chunk stably (avoids the
`(1/decay)^T` blow-up of full reconstruction).

Roughly 7–8× faster than a pure-Python einsum baseline on the same model. The
kernel verifies bit-for-bit against the Python reference (`wkv7_train_py`).

---

## Open problems / contributions welcome

A few things are deliberately **not** implemented yet — good entry points for
the community:

- **Metal-kernel cross-entropy.** The loss over a large vocabulary (World =
  65536) materializes a `[B·T, vocab]` logits tensor, which sets a memory
  "floor" for big models / long contexts. Chunking it in Python does *not*
  help — measured worse than dense, because the autograd graph keeps every
  chunk alive and the Python loop reintroduces the dispatch overhead the WKV
  kernel was built to remove. The right fix is a **single Metal kernel** that
  streams over the vocab on-GPU (online softmax for the forward, softmax-form
  gradients for the backward), in the same spirit as the WKV kernel. This would
  lower the floor for 2.9B+ models. **If you want to benchmark or implement
  this, PRs are welcome.**
- **Larger models (2.9B+)** end-to-end on 16 GB.
- **Instruction-tuning datasets / pipelines** for the World models.

---

## Acknowledgements

- RWKV-7 "Goose" architecture and official weights by [BlinkDL](https://github.com/BlinkDL/RWKV-LM).
- Built on [MLX](https://github.com/ml-explore/mlx) by Apple.

## License

Apache-2.0.
