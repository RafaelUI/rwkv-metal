# LoRA / QLoRA fine-tuning with `rwkv_metal`

This guide covers parameter-efficient fine-tuning of RWKV-7 on Apple Silicon:
adding LoRA adapters, QLoRA (4-bit frozen base), and the high-level `finetune()`
loop that bakes in the validated low-memory recipe.

- [Concepts](#concepts)
- [Quick start](#quick-start)
- [Loading official weights](#loading-official-weights)
- [`add_lora`](#add_lora)
- [QLoRA: 4-bit base (stock)](#qlora-4-bit-base-stock)
- [QLoRA on a quantized `.rwkvq` base (rwkv-quant)](#qlora-on-a-quantized-rwkvq-base-rwkv-quant)
- [`LoRAConfig` and `finetune`](#loraconfig-and-finetune)
- [Saving, loading, merging adapters](#saving-loading-merging-adapters)
- [The validated recipe (why these defaults)](#the-validated-recipe-why-these-defaults)
- [Memory levers](#memory-levers)
- [Full example: LoRA on World 1.5B](#full-example-lora-on-world-15b)

---

## Concepts

LoRA adds small trainable rank-`r` adapters to frozen weight matrices:

```
y = W·x  (frozen)  +  (alpha/r)·B(A(x))     A ∈ R[r×in], B ∈ R[out×r]
```

`B` is initialized to zero, so at the start the adapter is a no-op and the model
output is **identical** to the base — training only nudges it from there.

In `rwkv_metal`, adapters go on the tmix projections `r_proj`, `k_proj`,
`v_proj`, `o_proj` (and optionally cmix `key`/`value`). Gradients for
`r/k/v_proj` flow **through the Metal WKV-7 backward kernel**; `o_proj` is after
WKV. The same engine works for both architectures:

- `RWKV7`      — from-scratch reference,
- `RWKV7X070`  — exact x070 for official World weights.

QLoRA additionally quantizes the frozen base to 4-bit, cutting base memory ~4×.

---

## Quick start

```python
import rwkv_metal as rk
from rwkv_metal.lora import LoRAConfig, finetune

# 1. Build / load a model
model = rk.RWKV7(rk.PretrainConfig(n_layer=12, n_embd=768, vocab_size=21248,
                                   train_data="", val_data="", max_steps=1))
model = rk.init_weights(model)
model.set_dtype("bfloat16")

# 2. Add LoRA adapters (freezes everything else)
model, info = rk.add_lora(model, rank=16, alpha=16.0)
print(f"trainable: {info['trainable_pct']:.3f}%  ({info['trainable_params']/1e6:.2f}M)")

# 3. Fine-tune
def batches():
    while True:
        # yield (x, y) int token-id arrays of shape [B, T]
        yield get_next_batch()

finetune(model, batches(), LoRAConfig(lr=1e-4, max_steps=2000, grad_accum=8))
```

---

## Loading official weights

To fine-tune an official RWKV-7 World checkpoint (`.pth` from BlinkDL):

```python
import rwkv_metal as rk

# torch-free loader: reads the torch zip/pickle directly into MLX (bf16 kept)
model, cfg = rk.load_pretrained("weights/RWKV-x070-World-1.5B.pth")

# World tokenizer (65536-token vocab) — bundled with the package
tok = rk.WorldTokenizer()
ids = tok.encode("User: Привет\n\nAssistant:")
```

`load_pretrained` infers `n_layer / D / H / S / vocab` from the checkpoint,
builds an `RWKV7X070`, converts the official tensor names/layout, and verifies
the mapping is clean (it returns `(None, cfg)` and prints diagnostics if any key
is missing/extra/mis-shaped, so problems fail loudly rather than silently).

> The from-scratch tokenizer (your own BPE) and the World tokenizer are **not**
> interchangeable. Use `WorldTokenizer` only with official x070 weights.

---

## `add_lora`

```python
model, info = rk.add_lora(
    model,
    rank          = 16,                 # adapter rank
    alpha         = 16.0,               # scale = alpha / rank
    dropout       = 0.0,
    tmix_targets  = ("r_proj", "k_proj", "v_proj", "o_proj"),
    cmix_targets  = (),                 # e.g. ("key", "value") to also adapt the FFN
    quantize_base = 0,                  # 0 = bf16 base; 4 or 8 = QLoRA
    q_group_size  = 64,
    layers        = None,               # None = all blocks; e.g. range(12, 24) = top half
)
```

`info` contains:

| key | meaning |
|---|---|
| `total_params` | all parameters |
| `trainable_params` | adapter parameters only |
| `trainable_pct` | percentage trainable |
| `num_adapters` | number of wrapped projections |
| `wrapped_per_block` | which projections got adapters |

After `add_lora`, the base is frozen and only adapters are trainable. **You must
train with `nn.value_and_grad` (not `mx.value_and_grad`)** so freezing is
respected — `finetune()` does this for you.

### Choosing `layers`

Restricting adapters to the top layers is a strong speed lever: MLX prunes the
backward pass below the lowest adapter. On a 1.5B base (measured):

| layers | speed | peak |
|---|---|---|
| all 24 | 168 tok/s | 4.07 GB |
| top 12 | 250 tok/s (+49%) | 3.40 GB |
| top 6  | 337 tok/s (+100%) | 3.18 GB |
| top 3  | 404 tok/s (+140%) | 3.22 GB |

Fewer trainable layers = smaller backward = faster. Quality trades off with
capacity, so `range(12, 24)` (top half) is a good balance.

---

## QLoRA: 4-bit base (stock)

Quantizing the frozen base to 4-bit is the **main memory lever** for large
models. The recipe quantizes only the *big* frozen matrices and leaves the small
low-rank matrices in bf16 (their ranks aren't multiples of `group_size`, and
quantizing them hurts the in-context dynamics).

```python
import rwkv_metal as rk
from rwkv_metal.lora import quantize_base_model

model, cfg = rk.load_pretrained("weights/RWKV-x070-World-1.5B.pth")

# 1. Quantize the big frozen matrices: cmix.key/value, head, emb
quantize_base_model(model, bits=4)            # group_size=64 by default

# 2. Add LoRA; quantize_base=4 also quantizes the r/k/v/o_proj targets
model, info = rk.add_lora(model, rank=16, alpha=16.0,
                          quantize_base=4, layers=range(12, 24))
```

Effect on a World 1.5B base (measured): quantizing the whole base brings active
memory from ~2.48 GB down to ~0.89 GB (peak ~4.07 GB), at about −9% speed from
dequantization. This is what lets a 1.5B fine-tune fit comfortably in 16 GB.

> Note for x070: do **not** blanket-quantize every `nn.Linear`. The internal
> low-rank matrices (`w/a/g/v` with ranks like 96/256/64/32) are not multiples of
> `group_size` and will fail/degrade. `quantize_base_model` targets only the big
> ones (`cmix.key/value`, `head`, `emb`) on purpose.

> This is the generic path: `mlx.nn.quantize` with a uniform bit width and
> group size, no RWKV-7-specific calibration. It's the simplest option (no
> second repo, no extra files) and a reasonable default. For a pretrained base
> where you want the best size/quality tradeoff, see the next section.

---

## QLoRA on a quantized `.rwkvq` base (rwkv-quant)

[`rwkv-quant`](https://github.com/impulseleap/rwkv-quant) is a companion
project that calibrates RWKV-7-specific quantization (per-group bit widths,
outlier handling) instead of quantizing every matrix uniformly, and ships two
presets tuned by direct perplexity measurement:

| Preset | Size vs bf16 | ppl vs bf16 | Use for |
|---|---|---|---|
| `reduction` | 2.35x smaller | +0.12% | QLoRA base — calibrated for near-zero degradation |
| `compression` | 3.04x smaller | +2.47% | Smaller footprint, small but real quality cost |

`reduction` is the validated default for QLoRA: it's specifically calibrated
so the base itself contributes negligible error before you even start
training adapters on top of it. `compression` uses the same code path in
`rwkv-metal` (the kernel is generic across 4/5/6-bit groups) but hasn't been
benchmarked end-to-end for QLoRA training the way `reduction` has — treat it
as untested rather than unsupported if you want to try it.

This is a two-repo pipeline: `rwkv-quant` produces and exports the quantized
weights (needs torch), `rwkv-metal` trains against them (does not need torch).

### 1. Quantize + export (in `rwkv-quant`)

```bash
cd rwkv-quant
python -c "
from rwkv_quant.api import quantize
quantize('weights/RWKV-x070-World-1.5B.pth', '/tmp/world15b.rwkvq', preset='reduction')
"
python -m rwkv_quant.formats.export_mlx /tmp/world15b.rwkvq /tmp/world15b.rwkvq_mlx
```

`export_mlx` is a one-time step: it repacks the `.rwkvq` (which requires torch
to read) into a `*.rwkvq_mlx.safetensors` + `.json` sidecar that `rwkv-metal`
loads with plain `mx.load`, no torch involved from here on.

### 2. QLoRA against the sidecar (in `rwkv-metal`)

```python
import rwkv_metal as rk
from rwkv_metal.lora import LoRAConfig, finetune

model, cfg, info = rk.lora.load_lora_rwkvq_model(
    "weights/RWKV-x070-World-1.5B.pth",   # shape/name metadata + non-quantized tensors
    "/tmp/world15b.rwkvq_mlx",            # sidecar path (no extension)
    rank=16, alpha=32.0,
    layers=range(12, 24),                 # same speed lever as stock QLoRA
)
print(f"trainable: {info['trainable_pct']:.3f}%")

finetune(model, batches, LoRAConfig(lr=1e-4, grad_accum=8, max_steps=2000))
```

`load_lora_rwkvq_model` replaces the big frozen matrices (`r/k/v/o_proj`,
`cmix.key/value`, `head`) with quantized modules *before* their bf16
placeholder weights are materialized from the `.pth`, so the loading peak is
lower too, not just the steady-state footprint — see
`model/convert.py::load_pretrained_partial`. `add_lora_rwkvq` is the
equivalent entry point if you already have a fully-loaded bf16 model in hand
(simpler, but pays the full bf16 loading peak first).

### Backend choice (`native=`)

| `native=` | Mechanism | Measured step time (1.5B, rank 16, top-half layers) | Notes |
|---|---|---|---|
| `True` (default) | Repacked into MLX's own `quantized_matmul` | 0.7-0.8s — ties stock QLoRA exactly (same underlying kernel) | Verified only for `bits=6` (REDUCTION); MLX's internal packing differs by bit width |
| `False` | Custom fused Metal dequant kernel | 1.2-1.3s | Best memory footprint of the three; bit-width generic (REDUCTION and COMPRESSION both work); doesn't depend on MLX-internal packing details |
| `"hybrid"` | Native code layout + compact scale/bias unpacked on the fly | 0.8-0.9s | Didn't clearly beat the other two in measurement; kept for reference |

If you want the smallest possible memory footprint, use `native=False`. If you
want QLoRA training speed to exactly match the stock path while still getting
`rwkv-quant`'s calibrated accuracy, use `native=True` (REDUCTION only).

### Caveats

- **`emb.weight` is never quantized** by this path — embedding lookup is a
  gather, not `x @ W^T`, so it stays bf16 regardless of preset.
- **`merge_lora()` doesn't apply here.** It folds the adapter delta into
  `linear.weight`, but the quantized modules have no dense `.weight` — the
  base is dequantized on the fly on every call. Keep base + adapter composed
  at inference time; see [`inference.md`](./inference.md#quantized-inference-rwkvq).
  There's no built-in "bake the adapter into a new quantized file" step yet.
- **`native=True`/`"hybrid"` are only verified for `bits=6`** (the REDUCTION
  preset's group bit width). If you use COMPRESSION (mixed 4/5-bit groups),
  use `native=False`.

---

## `LoRAConfig` and `finetune`

```python
from rwkv_metal.lora import LoRAConfig, finetune

cfg = LoRAConfig(
    # Optimization
    lr           = 1e-4,     # keep small for pretrained bases (2e-3 diverges)
    grad_clip    = 1.0,
    weight_decay = 0.0,
    beta1        = 0.9,
    beta2        = 0.95,
    adam_eps     = 1e-8,

    # Schedule
    max_steps    = 2000,
    grad_accum   = 8,        # effective batch via accumulation; keep micro-batch low
    warmup_steps = 0,

    # Memory recipe
    grad_checkpoint = True,  # per-block nn.utils.checkpoint (big RAM + speed win)
    cache_limit_gb  = 1.5,   # mx.set_cache_limit; <=0 disables

    # Logging / checkpoints
    log_every    = 10,
    save_every   = 0,        # 0 = save only at the end
    adapter_path = "lora_adapters.safetensors",
)

result = finetune(model, batches, cfg, on_step=None)
# result -> {"final_loss": ..., "steps": ..., "adapter_path": ...}
```

`batches` is any iterable yielding `(x, y)` token-id arrays of shape `[B, T]`.
It is automatically cycled, so a small dataset is fine. The effective batch is
`B × grad_accum`.

`on_step(step, loss, peak_gb)` is an optional callback for custom logging.

`finetune()` saves only the adapters (small `.safetensors`), not the base.

---

## Saving, loading, merging adapters

```python
from rwkv_metal.lora import save_lora, load_lora, merge_lora, lora_state

# Save / load just the adapter tensors
save_lora(model, "adapters.safetensors")
model = load_lora(model, "adapters.safetensors")

# Inspect adapter tensors
state = lora_state(model)            # dict[str, mx.array]

# Merge adapters back into plain nn.Linear (for inference / export)
model = merge_lora(model)            # LoRALinear -> nn.Linear with W += scale·B·A
```

Adapter save/load is exact (bit-identical). `merge_lora` folds the adapter into
the base weight and replaces `LoRALinear` with a plain `nn.Linear`, so the merged
model has zero LoRA overhead at inference.

---

## The validated recipe (why these defaults)

`finetune()` encodes a recipe validated empirically on a real World 1.5B base
(see the project notes). The key rules:

1. **`nn.value_and_grad(model, loss_fn)`**, not `mx.value_and_grad`. The latter
   differentiates the whole tree and ignores `freeze()`, so it would train the
   base too. `nn.value_and_grad` respects `trainable_parameters()`.

2. **No `mx.compile`.** For LoRA on big models, compile was measured ~5.5× *slower*
   (43 vs 239 tok/s) with no memory benefit — the custom-function WKV across many
   layers doesn't cache well in the compiled graph.

3. **`nn.utils.checkpoint` per block** (via `grad_checkpoint=True`). On 1.5B this
   gave both **−2.9× peak memory and +2.2× speed** (the earlier slowness was memory
   pressure / swap; checkpointing removed it). Use the framework's flag, not a bare
   `mx.checkpoint` (which drops adapter-parameter gradients and stalls the loss).

4. **`mx.set_cache_limit`** (via `cache_limit_gb`). The Metal buffer cache defaults
   to ~2.25 GB resident; capping it stops "swap while RAM is free".

5. **`lr=1e-4, alpha=16, grad_clip=1.0` for pretrained bases.** Aggressive updates
   (`lr=2e-3, alpha=32`) diverge instantly on a strong model (3.02 → 13.9). Small
   `lr` converges smoothly.

6. **Effective batch via grad accumulation.** Throughput is roughly flat across
   batch size (the GPU is saturated by the `B·T` dimension), so a low micro-batch
   with `grad_accum` keeps memory down while raising the effective batch — batch is
   a quality lever, not a speed lever.

---

## Memory levers

In order of impact for large-model LoRA on 16 GB:

| Lever | How | Effect |
|---|---|---|
| QLoRA 4-bit base | `quantize_base_model(m, 4)` + `add_lora(quantize_base=4)` | base ~4× smaller (active ~0.9 GB on 1.5B) |
| Gradient checkpoint | `LoRAConfig(grad_checkpoint=True)` | ~2.5–2.9× less peak, often faster |
| Fewer trainable layers | `add_lora(layers=range(12, 24))` | smaller backward, big speedup |
| Cache limit | `LoRAConfig(cache_limit_gb=1.5)` | stops swap under free RAM |
| Effective batch | `LoRAConfig(grad_accum=N)` | large batch without large activations |

The remaining "floor" (forward + the cross-entropy over a 65536 vocab) does not
depend on the backward pass. Lowering it further would require a Metal-kernel
cross-entropy — see the note in the repo README; it's an open item, not needed
to fit 1.5B in 16 GB.

---

## Full example: LoRA on World 1.5B

```python
import mlx.core as mx
import rwkv_metal as rk
from rwkv_metal.lora import LoRAConfig, finetune, quantize_base_model

mx.set_cache_limit(int(1.5e9))

# 1. Load official weights + World tokenizer
model, cfg = rk.load_pretrained("weights/RWKV-x070-World-1.5B.pth")
tok = rk.WorldTokenizer()

# 2. QLoRA: 4-bit big frozen matrices, LoRA on the top 12 layers
quantize_base_model(model, bits=4)
model, info = rk.add_lora(model, rank=16, alpha=16.0,
                          quantize_base=4, layers=range(12, 24))
print(f"trainable: {info['trainable_pct']:.3f}%")

# 3. Build batches from your data (World-tokenized)
def batches(ctx=256, B=1):
    ids = tok.encode(open("data.txt", encoding="utf-8").read())
    n = (len(ids) - 1) // ctx
    i = 0
    while True:
        s = (i % n) * ctx
        x = mx.array(ids[s:s+ctx]).reshape(1, ctx)
        y = mx.array(ids[s+1:s+ctx+1]).reshape(1, ctx)
        i += 1
        yield x, y

# 4. Fine-tune (small lr for a strong base!)
finetune(model, batches(), LoRAConfig(
    lr=1e-4, grad_clip=1.0, max_steps=2000, grad_accum=8,
    grad_checkpoint=True, cache_limit_gb=1.5,
    adapter_path="world15b_lora.safetensors",
))
```

See also: [`pretraining.md`](./pretraining.md) for training from scratch.
