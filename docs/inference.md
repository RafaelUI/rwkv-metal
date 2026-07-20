# Inference with `rwkv_metal`

This guide covers running RWKV-7 for inference on Apple Silicon: scoring or
generating from a **bf16** checkpoint, and running a **quantized `.rwkvq`**
checkpoint produced by [`rwkv-quant`](https://github.com/impulseleap/rwkv-quant).

- [Concepts](#concepts)
- [bf16 inference](#bf16-inference)
- [Quantized inference (`.rwkvq`)](#quantized-inference-rwkvq)
- [Choosing bf16 vs quantized](#choosing-bf16-vs-quantized)
- [Current limitations](#current-limitations)

---

## Concepts

`rwkv_metal` gives you a model (`RWKV7X070` for official World weights, or
`RWKV7` for your own from-scratch checkpoint) whose `__call__(idx)` runs the
**full sequence** through every block and returns logits `[B, T, vocab]`. That
is the same forward pass used for training loss, and it is also what you use
for inference: feed a prompt, read the logits at the last position, sample a
token, append it, repeat.

There are two ways to get weights into that model:

| | bf16 | quantized (`.rwkvq`) |
|---|---|---|
| Where the weights come from | `.pth` (official World checkpoint) or your own pretrain/finetune output | `rwkv-quant` (separate repo) quantizes a `.pth` into `.rwkvq`, then exports an MLX sidecar |
| Loader | `rk.load_pretrained(...)` | `rk.lora.load_lora_rwkvq_model(...)` |
| Memory | full size (e.g. ~3 GB for World 1.5B) | 2–3× smaller (REDUCTION/COMPRESSION presets) |
| Dependencies | `rwkv_metal` only | `rwkv_metal` + a one-time `rwkv-quant` export step (torch, run separately — see below) |

Both paths produce an ordinary `rwkv_metal` model you call the same way —
`model(idx)` — the difference is only in how the weights got there.

---

## bf16 inference

```python
import mlx.core as mx
import rwkv_metal as rk

model, cfg = rk.load_pretrained("weights/RWKV-x070-World-1.5B.pth")
tok = rk.WorldTokenizer()

prompt = "User: What is the capital of France?\n\nAssistant:"
ids = tok.encode(prompt)
```

### Scoring a prompt

A single forward pass gives you logits for every position — useful for
perplexity, ranking completions, or just getting the next-token distribution:

```python
x = mx.array(ids)[None, :]          # [1, T]
logits = model(x)                   # [1, T, vocab]
next_token_logits = logits[0, -1]   # distribution after the last prompt token
```

### Sampling a token

`rwkv_metal` does not ship a sampler — logits are plain `mx.array`, so use
whatever policy you want. A minimal temperature + top-p sampler:

```python
def sample(logits, temperature=1.0, top_p=0.9):
    if temperature == 0:
        return int(mx.argmax(logits).item())
    probs = mx.softmax(logits.astype(mx.float32) / temperature)
    order = mx.argsort(-probs)
    sorted_probs = probs[order]
    cutoff = int(mx.sum(mx.cumsum(sorted_probs) < top_p).item()) + 1
    keep = order[:cutoff]
    kept_probs = probs[keep]
    kept_probs = kept_probs / mx.sum(kept_probs)
    choice = mx.random.categorical(mx.log(kept_probs))
    return int(keep[choice].item())
```

### A generation loop

```python
generated = list(ids)
for _ in range(200):
    x = mx.array(generated)[None, :]
    logits = model(x)[0, -1]
    next_id = sample(logits, temperature=0.8, top_p=0.9)
    generated.append(next_id)
    if next_id == 0:            # end-of-text, if your tokenizer/model uses it
        break

print(tok.decode(generated[len(ids):]))
```

> **Cost note.** This loop re-runs the *entire* growing context through every
> block on each new token (no state is carried between steps), so per-token
> cost grows with sequence length — fine for short completions, wasteful for
> long ones. See [Current limitations](#current-limitations) below: the WKV-7
> kernel already has a stateful, O(1)-per-token primitive (`wkv7_infer`), it is
> just not yet wired into `RWKV7X070`/`RWKV7` as a `model.generate()`-style API.

---

## Quantized inference (`.rwkvq`)

Running a quantized checkpoint is a two-repo pipeline: **rwkv-quant**
quantizes and exports, **rwkv-metal** loads and runs.

### 1. Quantize + export (in `rwkv-quant`, needs torch)

```bash
cd rwkv-quant
python -c "
from rwkv_quant.api import quantize
quantize('weights/RWKV-x070-World-1.5B.pth', '/tmp/world15b.rwkvq', preset='reduction')
"
python -m rwkv_quant.formats.export_mlx /tmp/world15b.rwkvq /tmp/world15b.rwkvq_mlx
```

`preset` is `"reduction"` (near-zero quality loss, ~2.35× smaller, the
validated default for a quantized *base* you intend to keep accurate) or
`"compression"` (~3× smaller, a small but real quality cost — see
[`lora.md`](./lora.md#qlora-on-a-quantized-rwkvq-base-rwkv-quant) for the
tradeoff). `export_mlx` is the one place torch is required — it converts the
`.rwkvq` into a torch-free `*.rwkvq_mlx.safetensors` + `.json` sidecar that
`rwkv_metal` loads directly. Run it in whatever environment has `rwkv-quant`
installed; the output sidecar is the only thing `rwkv-metal` needs afterwards.

### 2. Load + run (in `rwkv-metal`, torch-free)

```python
import rwkv_metal as rk

model, cfg, info = rk.lora.load_lora_rwkvq_model(
    "weights/RWKV-x070-World-1.5B.pth",   # only used for shape/name metadata + non-quantized tensors
    "/tmp/world15b.rwkvq_mlx",            # sidecar path (no extension)
    rank=1,                               # see note below — no adapter training happening here
)
tok = rk.WorldTokenizer()
```

Generation from here is identical to the bf16 case — `model(idx)` returns
logits, sample as above.

### Why `rank=1` and no training

`load_lora_rwkvq_model` / `add_lora_rwkvq` are QLoRA entry points — they wrap
each quantized projection in a `LoRALinear`. There is currently no separate
"just load quantized weights, no adapter" function. That is not a correctness
problem: `LoRALinear`'s adapter (`lora_b`) is zero-initialized, so an untrained
adapter is a mathematical no-op — `model(idx)` returns exactly the quantized
model's output, plus one small extra matmul per wrapped projection. Use the
smallest `rank` you're comfortable with (`rank=1` minimizes that overhead) if
you only want inference. If you *do* want to fine-tune on top of the quantized
base, see [`lora.md`](./lora.md#qlora-on-a-quantized-rwkvq-base-rwkv-quant).

### Backend choice (`native=`)

```python
model, cfg, info = rk.lora.load_lora_rwkvq_model(pth_path, sidecar_path,
                                                  rank=1, native=True)
```

| `native=` | What it does | Best for |
|---|---|---|
| `True` (default) | Repacks into MLX's own `quantized_matmul` layout at load time | Fastest steady-state; ties stock MLX quantization for speed |
| `False` | Custom fused Metal dequant kernel (`rwkvq_kernel.py`), one launch per weight | Best memory/speed balance; no dependency on MLX-internal packing details; ~1.5× slower than `native=True` |
| `"hybrid"` | Native code layout + compact scale/bias unpacked on the fly | Rarely the right choice — didn't beat the other two in measurement, kept for reference |

`native=True` is only verified against `bits=6` (the REDUCTION preset) — it
reverse-engineers MLX's internal packing, which differs by bit width.
`native=False` (the fused kernel) is bit-width generic and works for both
REDUCTION and COMPRESSION.

---

## Choosing bf16 vs quantized

- **Just want to run the official World weights as-is** → bf16
  (`load_pretrained`). Simplest path, no second repo involved.
- **Model doesn't fit in memory at bf16, or you want a smaller checkpoint to
  ship** → quantized. REDUCTION for accuracy-sensitive use, COMPRESSION for
  maximum size reduction.
- **Fine-tuning** → see [`lora.md`](./lora.md); QLoRA on a quantized base is
  the way to fit larger models in 16 GB.

---

## Current limitations

- **No `model.generate()`.** You assemble the loop yourself (see above). The
  loop re-runs the full context each step; there is no built-in KV/state
  cache at the model level yet.
- **No stateful streaming decode wired up.** The Metal WKV-7 kernel has a
  stateful, step-by-step primitive (`rwkv_metal.wkv7_infer(r, w, k, v, a, b,
  state) -> (out, new_state)`, O(1) per token) and it is unit-tested
  (`tests/test_wkv7_infer_var.py`), but `RWKV7X070`/`RWKV7` don't expose a
  path that threads that state through token-shift and `v_first` across
  blocks. Wiring this up would turn the naive loop above into real streaming
  inference — a good contribution if you want to take it on.
- **`emb.weight` is never quantized**, even in the `.rwkvq` path — embedding
  lookup is a gather, not a matmul, so it stays bf16 regardless of preset.
- **`merge_lora()` doesn't apply to `.rwkvq`-based adapters.** It writes the
  adapter delta into `linear.weight`, but `RwkvqLinear`/`RwkvqNativeLinear`
  have no dense `.weight` — the base is dequantized on the fly. If you trained
  a QLoRA adapter on a quantized base, keep base + adapter composed at
  inference time (the normal `LoRALinear.__call__` path); there's no built-in
  "bake the adapter into a smaller quantized file" step yet.

See also: [`lora.md`](./lora.md) for fine-tuning, [`pretraining.md`](./pretraining.md)
for training from scratch.
