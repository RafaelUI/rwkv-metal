# Pretraining RWKV-7 with `rwkv_metal`

This guide covers pretraining an RWKV-7 model from scratch on Apple Silicon
(Metal / MLX) using the `rwkv_metal` framework.

- [Quick start](#quick-start)
- [The `PretrainConfig`](#the-pretrainconfig)
- [Presets](#presets)
- [Data: `.bin` and `.txt`](#data-bin-and-txt)
- [Precision: bf16 vs fp32](#precision-bf16-vs-fp32)
- [Memory: what actually uses RAM](#memory-what-actually-uses-ram)
- [Checkpoints & resuming](#checkpoints--resuming)
- [CLI reference](#cli-reference)
- [Programmatic API](#programmatic-api)

---

## Quick start

```python
import rwkv_metal as rk

# Start from a preset, override what you need
cfg = rk.preset("25m",
    train_data = "data/train.bin",
    val_data   = "data/val.bin",
    vocab_size = 21248,
    max_tokens = 3_000_000_000,
)

rk.pretrain(cfg)
```

Or from the command line:

```bash
cd rwkv-metal
python pretrain.py --preset 25m \
    --train_data data/train.bin \
    --val_data   data/val.bin \
    --vocab_size 21248
```

> **Tip (macOS GPU memory limit).** Long contexts may exceed the default GPU
> wired limit. Raise it for the session before training:
> ```bash
> sudo sysctl iogpu.wired_limit_mb=14336
> ```

---

## The `PretrainConfig`

Every training run is described by a single dataclass. All fields have sane
defaults; you only set what you care about.

```python
from rwkv_metal import PretrainConfig
import rwkv_metal as rk

cfg = PretrainConfig(
    # ── Architecture ──────────────────────────────────────────────
    n_layer     = 18,
    n_embd      = 256,
    vocab_size  = 21248,
    head_size   = 64,            # n_embd must be divisible by head_size

    # ── Data ──────────────────────────────────────────────────────
    train_data      = "data/train.bin",   # .bin (uint16) or .txt
    val_data        = "data/val.bin",
    tokenizer       = "tokenizer.json",    # required only if data is .txt
    ctx_len         = 512,
    batch_size      = 18,
    grad_accum      = 1,                   # effective batch = batch_size * grad_accum

    # ── How long to train (set one of two) ────────────────────────
    max_steps       = None,                # explicit step count (takes priority)
    max_tokens      = 3_000_000_000,       # OR a token budget (steps are derived)

    # ── Optimizer (AdamW) ─────────────────────────────────────────
    lr              = 1.5e-3,
    lr_min          = 1e-4,
    lr_schedule     = "cosine",            # cosine | linear | constant
    warmup_steps    = 200,
    weight_decay    = 0.0,
    beta1           = 0.9,
    beta2           = 0.95,
    adam_eps        = 1e-18,               # RWKV-7 uses a very small eps
    grad_clip       = 1.0,

    # ── Hardware ──────────────────────────────────────────────────
    dtype           = "bfloat16",          # bfloat16 | float32
    grad_checkpoint = False,               # trade ~15% speed for ~2.5x less RAM

    # ── Checkpoints & logging ─────────────────────────────────────
    checkpoint_dir  = "checkpoints",
    resume          = True,                # auto-continue from latest checkpoint
    save_every      = 500,
    save_best_only  = False,
    eval_every      = 500,
    eval_batches    = 20,
    log_every       = 50,
    wandb           = False,
    wandb_project   = "rwkv-metal",
)
rk.pretrain(cfg)
```

### Field reference

| Field | Type | Default | Meaning |
|---|---|---|---|
| `n_layer` | int | 12 | Number of RWKV blocks (depth). |
| `n_embd` | int | 768 | Model width. Must be divisible by `head_size`. |
| `vocab_size` | int | 21248 | Token vocabulary size. Must match your data/tokenizer. |
| `head_size` | int | 64 | Attention head size. **Fixed at 64** by the Metal kernel. |
| `train_data` | str | `data/train.bin` | Path to training data (`.bin` or `.txt`). |
| `val_data` | str | `data/val.bin` | Path to validation data. |
| `tokenizer` | str \| None | None | Path to `tokenizer.json`; required only for `.txt` data. |
| `ctx_len` | int | 512 | Sequence length per sample. |
| `batch_size` | int | 8 | Micro-batch size (sequences processed at once). |
| `grad_accum` | int | 1 | Gradient accumulation steps. Effective batch = `batch_size * grad_accum`. |
| `max_steps` | int \| None | None | Explicit step count. If set, takes priority over `max_tokens`. |
| `max_tokens` | int \| None | 3e9 | Token budget; steps are derived as `tokens / (batch * ctx * accum)`. |
| `lr` | float | 1.5e-3 | Peak learning rate (after warmup). |
| `lr_min` | float | 1e-4 | Final learning rate at end of schedule. |
| `lr_schedule` | str | `cosine` | Decay shape: `cosine` \| `linear` \| `constant`. |
| `warmup_steps` | int | 200 | Linear warmup from 0 to `lr`. |
| `weight_decay` | float | 0.0 | AdamW weight decay. |
| `beta1`, `beta2` | float | 0.9, 0.95 | AdamW momentum coefficients. |
| `adam_eps` | float | 1e-18 | AdamW epsilon (RWKV-7 needs a tiny value). |
| `grad_clip` | float | 1.0 | Global gradient-norm clip. |
| `dtype` | str | `bfloat16` | Weight/optimizer precision: `bfloat16` \| `float32`. See below. |
| `grad_checkpoint` | bool | False | Recompute activations in backward to save RAM. |
| `checkpoint_dir` | str | `checkpoints` | Where checkpoints are written. |
| `resume` | bool | True | Auto-resume from the latest checkpoint if present. |
| `save_every` | int | 500 | Save the `latest` checkpoint every N steps. |
| `save_best_only` | bool | False | Only save when validation loss improves. |
| `eval_every` | int | 500 | Run validation every N steps. |
| `eval_batches` | int | 20 | Number of batches per validation pass. |
| `log_every` | int | 50 | Print/log training metrics every N steps. |
| `wandb` | bool | False | Enable Weights & Biases logging. |
| `wandb_project` | str | `rwkv-metal` | W&B project name. |
| `wandb_run` | str \| None | None | W&B run name (None = auto). |

### Helper methods

```python
cfg.n_head              # property: n_embd // head_size
cfg.resolve_max_steps()  # final step count (explicit or derived from max_tokens)
print(cfg.summary())     # human-readable summary (params, batch, token budget, ...)
```

`summary()` prints something like:

```
──────────────────────────────────────────────────
  Model:    18L × 256d  (~27.4M params)
  Vocab:    21248  |  ctx: 512
  Batch:    18 × grad_accum 1 = 18 eff.
  Train:    325,521 steps  (~3.00B tokens)
  LR:       0.0015 → 0.0001  (cosine, warmup 200)
  dtype:    bfloat16  |  grad_ckpt: False
  Data:     data/train.bin
──────────────────────────────────────────────────
```

---

## Presets

Presets are starting points you can override with any field.

```python
import rwkv_metal as rk

cfg = rk.preset("170m", train_data="data/train.bin", val_data="data/val.bin")
# equivalent:
cfg = rk.PretrainConfig.from_preset("170m", train_data="...", val_data="...")
```

| Preset | Layers × Width | ctx | batch | grad_accum | lr |
|---|---|---|---|---|---|
| `25m`  | 18 × 256  | 512  | 18 | 1 | 1.5e-3 |
| `50m`  | 24 × 384  | 512  | 12 | 1 | 1.2e-3 |
| `170m` | 24 × 768  | 1024 | 4  | 1 | 6e-4 |
| `430m` | 24 × 1024 | 1024 | 2  | 4 | 4e-4 |

List them programmatically with `rwkv_metal.PRESETS`.

---

## Data: `.bin` and `.txt`

Two input formats are supported. The loader picks one automatically by file
extension.

### `.bin` (recommended)

A flat `uint16` array of token ids (pre-tokenized). Read lazily via `np.memmap`,
so it never loads the whole corpus into RAM.

### `.txt`

Raw UTF-8 text, tokenized on the fly. Requires a `tokenizer` path. Convenient
for small corpora; slower than `.bin`.

### Pre-tokenizing text to `.bin`

```python
from rwkv_metal import tokenize_to_bin

result = tokenize_to_bin(
    "data/corpus.txt",         # input: documents separated by blank lines
    "tokenizer.json",          # HuggingFace tokenizers JSON
    "data/train.bin",          # output (a sibling data/train_val.bin is also written)
)
print(result["train_tokens"], result["val_tokens"], result["vocab_size"])
```

Or via CLI:

```bash
python pretrain.py --tokenize \
    --input    data/corpus.txt \
    --tokenizer tokenizer.json \
    --output   data/train.bin
```

> **OOV check.** Before training, the trainer scans the `.bin` for tokens
> `>= vocab_size`. Out-of-vocabulary ids cause `NaN` loss. If you see a warning,
> fix `vocab_size` or re-tokenize.

---

## Precision: bf16 vs fp32

Choose with `dtype="bfloat16"` or `dtype="float32"` in the config (or
`model.set_dtype(...)` if you build the model yourself).

| | `bfloat16` (default) | `float32` |
|---|---|---|
| Weights + optimizer RAM | **~2× smaller** | baseline |
| Speed | **~+10%** | baseline |
| Precision | mixed-precision (see note) | full |

**`bfloat16` is mixed-precision, not "pure" bf16.** The numerically sensitive
parts always run in fp32 regardless of this setting:

- the loss / cross-entropy reduction,
- the WKV-7 recurrence inside the Metal kernel.

So the accuracy loss from bf16 weights is tiny in practice (on small models the
loss difference vs fp32 is well under 0.001%), while you get smaller weight
files and a speed bump.

**When does bf16 matter?**

- On small models (~36M) the savings are modest — peak RAM is dominated by
  activations, not weights.
- On larger models (430M+, and any LoRA on 1.5B bases) bf16 is the difference
  between *fits in 16 GB* and *does not*.

```python
# Model-level control (outside the trainer):
model = rk.RWKV7(cfg)
model = rk.init_weights(model)
model.set_dtype("bfloat16")   # or "float32"
```

---

## Memory: what actually uses RAM

Peak memory during training is roughly:

```
weights + AdamW state (2 buffers)  +  activations
```

For small models, **activations dominate** (~95% of peak). That means:

1. **`grad_checkpoint=True`** is the strongest memory lever — it recomputes
   activations in the backward pass for ~2.5× lower RAM at ~15% slower speed.
   Use it when you hit OOM.
2. **`bfloat16`** mainly shrinks weights + optimizer state (the smaller share),
   but becomes essential at scale.
3. **`grad_accum`** lets you keep a large *effective* batch with a small
   *micro*-batch (`batch_size`), capping activation memory.

> Rule of thumb on M-series 16 GB: keep `batch_size` modest and grow the
> effective batch through `grad_accum`. Throughput is mostly flat across batch
> size (the GPU is already saturated by the `B*T` dimension), so batch is a
> *quality* lever, not a *speed* lever.

---

## Checkpoints & resuming

- Checkpoints are written to `checkpoint_dir` as
  `rwkv7_{n_layer}l{n_embd}d_latest.npz` (and `..._best.npz`).
- A sibling `..._latest.step` file records the step count for resuming.
- With `resume=True` (default), a new run automatically continues from
  `latest` if the file exists. Set `resume=False` (CLI: `--no_resume`) to start
  fresh.
- `save_best_only=True` writes only when validation loss improves.

Weights are saved in the model's `dtype` — so a `bfloat16` run produces
half-size checkpoint files automatically.

---

## CLI reference

The CLI mirrors every config field. Anything not passed falls back to the
preset (if `--preset` is given) or the `PretrainConfig` default.

```bash
# Preset + data override
python pretrain.py --preset 25m \
    --train_data data/train.bin --val_data data/val.bin --vocab_size 21248

# Train on raw text (tokenize on the fly)
python pretrain.py --preset 25m \
    --train_data data/wiki.txt --tokenizer tokenizer.json

# Full manual control
python pretrain.py \
    --n_layer 18 --n_embd 256 --vocab_size 21248 \
    --train_data data/train.bin --val_data data/val.bin \
    --max_tokens 3_000_000_000 \
    --lr 1.5e-3 --batch_size 18 --ctx_len 512 \
    --dtype bfloat16 --grad_checkpoint \
    --checkpoint_dir checkpoints/

# Start fresh, ignoring existing checkpoints
python pretrain.py --preset 25m --train_data ... --val_data ... --no_resume
```

After `pip install`, the same CLI is available as the `rwkv-metal-pretrain`
command, and as `python -m rwkv_metal.pretrain.cli`.

Key flags that differ from field names:

| Flag | Effect |
|---|---|
| `--no_resume` | Sets `resume=False`. |
| `--grad_checkpoint` | Sets `grad_checkpoint=True`. |
| `--save_best_only` | Sets `save_best_only=True`. |
| `--wandb` | Sets `wandb=True`. |
| `--tokenize` | Switches to tokenization mode (needs `--input/--tokenizer/--output`). |

---

## Programmatic API

For full control, drive the pieces yourself instead of `pretrain(cfg)`:

```python
import mlx.core as mx
import mlx.optimizers as optim
import rwkv_metal as rk
from rwkv_metal.pretrain import load_dataset

cfg = rk.PretrainConfig(n_layer=18, n_embd=256, vocab_size=21248,
                        train_data="data/train.bin", val_data="data/val.bin",
                        dtype="bfloat16", max_steps=10_000)

# Model
model = rk.RWKV7(cfg)
model = rk.init_weights(model)        # REQUIRED for from-scratch training
model.set_dtype(cfg.dtype)

# Data
train_ds = load_dataset(cfg.train_data, cfg.ctx_len, cfg.tokenizer)

# Optimizer
opt = optim.AdamW(learning_rate=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                  eps=cfg.adam_eps, weight_decay=cfg.weight_decay)

# Compiled train step (loss MUST be cast to fp32 for bf16 models)
state = [model.state, opt.state]

def loss_fn(m, x, y):
    return m.loss(x, y).astype(mx.float32)

def step(x, y):
    loss, grads = mx.value_and_grad(loss_fn)(model, x, y)
    grads, norm = optim.clip_grad_norm(grads, max_norm=cfg.grad_clip)
    opt.update(model, grads)
    return loss, norm

step = mx.compile(step, inputs=state, outputs=state)

for s in range(cfg.resolve_max_steps()):
    x, y = train_ds.batch(cfg.batch_size, s)
    loss, norm = step(x, y)
    mx.eval(loss, norm)
```

> **`init_weights` is mandatory** when training from scratch. RWKV-7 has a
> specific initialization (LoRA-B matrices zeroed, damped `k_proj`,
> depth-scaled `r/v_proj`); without it the first step produces `NaN`.

See also: [`lora.md`](./lora.md) for LoRA / QLoRA fine-tuning of pretrained
models.
