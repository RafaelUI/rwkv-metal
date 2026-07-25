# rwkv-metal

**RWKV-7 on Apple Silicon — pretraining, LoRA/QLoRA, and a custom Metal WKV-7 kernel.**

`rwkv-metal` is [ImpulseLeap](https://www.impulseleap.com)'s framework for training
and fine-tuning [RWKV-7 "Goose"](https://github.com/BlinkDL/RWKV-LM) models on
Apple Silicon (M-series) using [MLX](https://github.com/ml-explore/mlx). The
WKV-7 recurrence — the part that doesn't map onto standard ops — runs as a
hand-written Metal kernel with a checkpointed backward pass, so training is
fast and fits in unified memory.

- Train RWKV-7 **from scratch** with a simple config.
- **LoRA / QLoRA** fine-tune your own checkpoints or official RWKV-7 World weights.
- **Text embeddings**: pooled sentence vectors, contrastive fine-tuning, GradCache.
- **Reranking**: a cross-encoder that scores query–document pairs off the
  recurrent state, with a cacheable document prefix.
- **Recurrent state as a first-class object**: streaming decode, prefix caching,
  exact right-padding for ragged batches.
- A custom **Metal WKV-7 kernel** (forward + checkpointed backward + inference).
- Designed for **16 GB** Macs: bf16, gradient checkpointing, QLoRA 4-bit base.

---

## Install

Requires macOS on Apple Silicon and Python 3.10+.

```bash
pip install rwkv-metal
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

### Inference

```python
import mlx.core as mx
import rwkv_metal as rk

model, cfg = rk.load_pretrained("weights/RWKV-x070-World-1.5B.pth")
tok = rk.WorldTokenizer()

ids = tok.encode("User: What is the capital of France?\n\nAssistant:")
logits = model(mx.array(ids)[None, :])[0, -1]   # next-token distribution
```

Works the same way for a quantized `.rwkvq` checkpoint (2-3x smaller, via the
companion [rwkv-quant](https://github.com/impulseleap/rwkv-quant) repo). See
**[docs/inference.md](docs/inference.md)** for sampling, streaming decode with
state, and the quantized-inference pipeline.

### Embeddings

RWKV folds a whole sequence into a fixed-size state, so a pooled hidden state is
already a usable sentence embedding — no head, no training:

```python
from rwkv_metal.embedding import Embedder, cosine_similarity_matrix

emb = Embedder(model, tok)
vecs = emb(["The cat sits by the window.",
            "A cat settled on the windowsill.",
            "The share price fell three percent."])   # [3, D], L2-normalized
print(cosine_similarity_matrix(vecs))
```

Contrastive fine-tuning turns that rough signal into a real retrieval model
(held-out MRR 0.035 → 0.725 in 750 steps, measured). See
**[docs/embedding.md](docs/embedding.md)**.

### Reranking

Second stage: a cross-encoder that reads the query and the document *together*
and scores the pair, reading straight off the recurrent state.

```python
from rwkv_metal.reranker import Reranker, RerankerInference

model = Reranker(base)                          # base frozen automatically
model.load_head("reranker_head.safetensors")
rr = RerankerInference(model, tok)

for doc_id, score in rr.rank("how do bees overwinter?", docs, top_k=5):
    print(f"{score:+.2f}  {docs[doc_id][:80]}")
```

The template puts the document before the query, so `Document: …` is a cacheable
prefix — `rr.build_index(docs)` precomputes it and each later query costs only
its own tokens (73 ms → 3.5 ms per pair on a 0.1B base). Training the head runs
on cached states, which makes an epoch over tens of thousands of pairs take
seconds. See **[docs/reranker.md](docs/reranker.md)**.

---

## What's inside

```
rwkv_metal/
├── kernel/       Metal WKV-7 kernel: forward, checkpointed backward, inference,
│                 stateful variants + a pure-Python reference for correctness
├── model/        RWKV7 (from-scratch) and RWKV7X070 (official x070 weights),
│                 RWKVState + torch-free .pth loader / converter
├── pretrain/     PretrainConfig, presets, dataset loaders, training loop, CLI
├── lora/         LoRA/QLoRA engine, high-level finetune(), QLoRA helpers
├── embedding/    pooling head, contrastive losses, curriculum trainer, GradCache
├── reranker/     cross-encoder head over the state, state cache, inference API
└── tokenizer/    RWKV World tokenizer (65536-token vocab)
```

Two architectures share the same LoRA target names, so the LoRA engine works
with either:

| | `RWKV7` | `RWKV7X070` |
|---|---|---|
| Purpose | train from scratch | load official weights |
| `ln_x` | LayerNorm | GroupNorm (per head) |
| low-rank size | fixed 64 | derived from model width |
| token-shift | zero-padded per block | zero-padded per block |
| state API (`return_state`, `mask`) | yes | yes |
| pair with | `init_weights()` | `load_pretrained()` |

> **Fixed: `RWKV7` used to leak future information.** It carried token-shift
> *between* blocks — block `i+1` received block `i`'s **last** output as its
> "previous token", which is in the future relative to position 0. The damage
> was smaller than it sounds: the perturbation walks forward one position per
> layer, so it touched the first ~`n_layer` positions of the window (measured
> `max|Δh|` 0.209 at position 0, under 0.007 after that, nothing past position
> 9 for a 6-layer model at ctx 512), and a 400-step from-scratch A/B came out
> within noise. It mattered because train and serve disagreed and because
> "continue from a saved state" could not be defined. Both architectures now
> zero-pad per block, both are causal (`tests/test_wkv7_state.py`), and both
> support the state API. `RWKV7(cfg, legacy_token_shift=True)` reproduces the
> old behaviour for checkpoints trained before the fix.

---

## The Metal WKV-7 kernel

The WKV-7 recurrence is bandwidth-bound (~1 FLOP/byte) and sequential, so it
doesn't fit standard fused ops. The kernel handles the whole sequence in one
forward dispatch and one backward dispatch, checkpointing the hidden state every
16 tokens so the backward pass reconstructs each chunk stably (avoids the
`(1/decay)^T` blow-up of full reconstruction).

Roughly 7–8× faster than a pure-Python einsum baseline on the same model, and
verified against the Python reference (`wkv7_train_py`) to ~4e-8 — the level of
float32 reassociation noise, against a 1e-5 test tolerance.

Three entry points, all sharing the same kernels:

| | |
|---|---|
| `wkv7_train(r,w,k,v,a,b)` | training, zero initial state |
| `wkv7_train_with_state(..., h_in)` | training/inference with an explicit state, differentiable **through** `h_in` |
| `wkv7_step(..., h_in)` | one token, plain MLX ops — the kernel needs `T` divisible by 16, so a single token would otherwise cost sixteen |

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
- **A `model.generate()`.** State plumbing for streaming decode is in place
  (`body(idx, state=..., return_state=True)`); the sampler, stop conditions and
  batching wrapper are not. A single-token step costs ~10.6 ms at `B=1` on
  0.1B, and `mx.compile` only takes it to 9.9 ms — unlike the reranker head,
  this path is not dispatch-bound, so the win has to come from elsewhere.
- **Reranker on a non-frozen base.** Training the head on cached states is what
  makes it cheap, but it rules out LoRA on the base. An online-encoding trainer
  would trade speed for that flexibility.
- **Larger models (2.9B+)** end-to-end on 16 GB.
- **Instruction-tuning datasets / pipelines** for the World models.

---

## Acknowledgements

- RWKV-7 "Goose" architecture and official weights by [BlinkDL](https://github.com/BlinkDL/RWKV-LM).
- Built on [MLX](https://github.com/ml-explore/mlx) by Apple.

## License

Apache-2.0.

## Author

Alexei Goncharov
