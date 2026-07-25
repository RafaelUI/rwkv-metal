# Inference with `rwkv_metal`

This guide covers running RWKV-7 for inference on Apple Silicon: scoring or
generating from a **bf16** checkpoint, and running a **quantized `.rwkvq`**
checkpoint produced by [`rwkv-quant`](https://github.com/impulseleap/rwkv-quant).

- [Concepts](#concepts)
- [bf16 inference](#bf16-inference)
- [Streaming decode with state](#streaming-decode-with-state)
- [Quantized inference (`.rwkvq`)](#quantized-inference-rwkvq)
- [Choosing bf16 vs quantized](#choosing-bf16-vs-quantized)
- [Embedding RWKV: extracting vectors](#embedding-rwkv-extracting-vectors)
- [Reranking: scoring query–document pairs](#reranking-scoring-querydocument-pairs)
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
> block on each new token, so per-token cost grows with sequence length. Carry
> the state instead — see the next section.

---

## Streaming decode with state

`RWKV7X070` threads its full recurrent state through `body()`, so each new token
costs the same regardless of how much context precedes it:

```python
import mlx.core as mx
import rwkv_metal as rk

model, cfg = rk.load_pretrained("weights/RWKV-x070-World-0.1B.pth")
tok = rk.WorldTokenizer()

ids = tok.encode("User: What is the capital of France?\n\nAssistant:")

# fold the prompt into a state, once
h, state = model.body(mx.array(ids)[None, :], return_state=True)
logits = model.head(h[:, -1])

generated = []
for _ in range(200):
    next_id = sample(logits[0], temperature=0.8, top_p=0.9)
    generated.append(next_id)
    if next_id == 0:
        break
    h, state = model.body(mx.array([[next_id]]), state=state, return_state=True)
    logits = model.head(h[:, -1])

print(tok.decode(generated))
```

`state` is an `RWKVState`: the WKV matrix per layer plus the token-shift inputs
of `tmix` and `cmix`. All three are needed — carrying only the WKV matrix
produces a continuation that differs from a straight pass on the first token of
every layer.

A single-token step takes the `wkv7_step` path (plain MLX ops) rather than the
Metal checkpoint kernel, which requires `T` to be a multiple of 16 and would
otherwise do sixteen steps' worth of work for one token.

### Batching sequences of different lengths

Padding normally corrupts the final state, since padded tokens run through the
recurrence like any other. Pass a mask and padded positions become exact
no-ops (`w ← 1`, `k ← 0`, `b ← 0`, so `h_next = h`):

```python
from rwkv_metal.model import build_mask

lengths = [41, 17, 8]
idx = ...                                    # [3, 41], right-padded with 0
mask = build_mask(lengths, idx.shape[1])     # [3, 41]
end_idx = mx.array([L - 1 for L in lengths])

state = model.states(idx, mask=mask, end_idx=end_idx)
```

Each row's state freezes at its last real token, independent of padding and of
batch neighbours. Verified against individual unpadded passes in
`tests/test_wkv7_state.py`.

> Not available on `RWKV7` (the from-scratch pretraining architecture): it
> carries token-shift between blocks, which makes an exact continuation
> impossible to define. `RWKV7.body` raises `NotImplementedError` for state
> arguments — details in [`reranker.md`](./reranker.md#not-available-on-rwkv7).

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

## Embedding RWKV: extracting vectors

Inference doesn't have to mean sampling tokens. Because RWKV is an RNN, it folds
an entire sequence into a fixed-size state, so the hidden state at the last
position **is** a sentence embedding — no extra head, no training. This is the
"free sentence embedding" property, and it works on any base checkpoint,
official World weights included.

```python
import rwkv_metal as rk
from rwkv_metal.embedding import Embedder, cosine_similarity_matrix

model, cfg = rk.load_pretrained("weights/RWKV-x070-World-0.1B.pth")
tok = rk.WorldTokenizer()

emb = Embedder(model, tok)                    # terminator=0, pooling="last"
vecs = emb(["The cat sits by the window.",
            "A cat settled on the windowsill.",
            "The share price fell three percent."])   # [3, D], L2-normalized

print(cosine_similarity_matrix(vecs))         # [3, 3] cosine similarities
```

`Embedder` runs `model.body(idx)` — everything except the vocab projection — so
it costs less than a scoring pass, and the `head` weights are never touched.
Use `embed_texts(model, tok, texts)` for a one-shot call without keeping the
object.

Two switches:

| Argument | Values | Meaning |
|---|---|---|
| `terminator` | int, or `None` | Token appended to every text before pooling, so the pooled position means the same thing across texts. `0` is unmapped in the World vocab, which is what makes it a safe choice there; for a BPE vocab use `BPETokenizer.terminator_id`. |
| `pooling` | `"last"` / `"mean"` | State at the terminator position, or the mean over all positions. |

Each text gets its own forward pass, so a short text and a long one never share
a padded batch.

**Untrained quality.** Straight off a base checkpoint, with no fine-tuning,
vectors already cluster by language and domain: on 21 mixed chunks the 0.1B
World model gave 0.98–0.99 similarity within Russian agricultural texts,
0.97–0.99 within English wiki articles with English physics forming its own
cluster, and 0.89–0.94 for Serbian, separate from both.

That is enough for coarse clustering or deduplication, but it is not a retrieval
model. Contrastive fine-tuning is what closes that gap — measured on a Russian
61M base, held-out retrieval MRR goes from 0.035 to 0.725 in 750 steps. The
training path, the loss functions, the GradCache memory schedule, and advice on
building the negative pool are all in **[`embedding.md`](./embedding.md)**.

---

## Reranking: scoring query–document pairs

An embedder scores a query and a document separately — it never sees them
together. A **reranker** does: it reads the pair jointly and produces one
relevance score. That is much more accurate and much more expensive, so it runs
as a second stage over the handful of candidates the embedder shortlisted.

```python
import rwkv_metal as rk
from rwkv_metal.reranker import Reranker, RerankerInference

base, cfg = rk.load_pretrained("weights/rwkv7-g1d-0.1b.pth")
model = Reranker(base)                          # base frozen automatically
model.load_head("reranker_head.safetensors")    # trained scoring head

rr = RerankerInference(model, rk.WorldTokenizer())

query = "how do bees overwinter?"
docs = [...]                                     # top-k from your embedder

for doc_id, score in rr.rank(query, docs, top_k=5):
    print(f"{score:+.2f}  {docs[doc_id][:80]}")
```

`rank` returns `(index, score)` pairs sorted by descending score; `score`
returns the raw array in input order. Scores are logits — comparable **within**
one query, not across queries (unless the head was trained with a pointwise
term; see [`reranker.md`](./reranker.md#losses)).

### The two-stage pipeline

```python
from rwkv_metal.embedding import Embedder, cosine_similarity_matrix

emb = Embedder(base, tok)
corpus_vecs = emb(corpus)                        # once, offline
q_vec = emb(query)

sims = (q_vec @ corpus_vecs.T)[0]
shortlist = [int(i) for i in mx.argsort(-sims)[:50].tolist()]

ranked = rr.rank(query, [corpus[i] for i in shortlist], top_k=10)
final = [(shortlist[i], s) for i, s in ranked]
```

Both stages share the same frozen base, so only one set of weights is loaded.

### Serving the same documents repeatedly

The template puts the document **before** the query, which makes
`Instruct: … \nDocument: {doc}\n` a cacheable prefix. Precompute it:

```python
index = rr.build_index(docs)                     # encode prefixes once
scores = rr.score_indexed(query, index)          # each query pays only its own tokens
ranked = rr.rank_indexed(query, index, top_k=5)
```

Measured on a 0.1B base, M4 Air: **73 ms per pair without the index, 3.5 ms with
it** — about 20×. The price is memory: a full state is
`n_layer · n_head · 64 · 64 · 4` bytes, ~2.4 MB per document at 0.1B (halve it
with `build_index(..., dtype=mx.bfloat16)`). An index of a thousand documents is
2.4 GB, so build it for a hot subset, not a corpus.

An index is bound to the base, the template and the instruction string it was
built with — the state is a function of exactly that prefix, and nothing checks
this for you.

### Knobs

```python
rr = RerankerInference(model, tok,
    instruct="Given a search query, retrieve relevant passages that answer the query",
    max_doc_tokens=512,      # documents are truncated at the tail
    max_query_tokens=96,
)
```

`instruct` can also be passed per call (`rr.rank(query, docs, instruct=...)`),
but it must match what the head was trained with, and what an index was built
with.

Training your own head, choosing which base layers it reads, and the measured
numbers are all in **[`reranker.md`](./reranker.md)**.

---

## Current limitations

- **No `model.generate()`.** You assemble the loop yourself (see
  [Streaming decode with state](#streaming-decode-with-state)). The state
  plumbing exists; the convenience wrapper, sampler and stop-condition handling
  do not.
- **Per-step overhead dominates short steps.** A single-token step costs ~14 ms
  at `B=1` on a 0.1B model, almost all of it fixed cost — twelve layers of
  Python dispatch and kernel launches, not arithmetic. Batch your decode or
  wrap the step in `mx.compile` if throughput matters.
- **State plumbing is x070-only.** `RWKV7` (from-scratch pretraining
  architecture) cannot support it — see
  [`reranker.md`](./reranker.md#not-available-on-rwkv7).
- **`emb.weight` is never quantized**, even in the `.rwkvq` path — embedding
  lookup is a gather, not a matmul, so it stays bf16 regardless of preset.
- **`merge_lora()` doesn't apply to `.rwkvq`-based adapters.** It writes the
  adapter delta into `linear.weight`, but `RwkvqLinear`/`RwkvqNativeLinear`
  have no dense `.weight` — the base is dequantized on the fly. If you trained
  a QLoRA adapter on a quantized base, keep base + adapter composed at
  inference time (the normal `LoRALinear.__call__` path); there's no built-in
  "bake the adapter into a smaller quantized file" step yet.

See also: [`lora.md`](./lora.md) for fine-tuning, [`pretraining.md`](./pretraining.md)
for training from scratch, [`embedding.md`](./embedding.md) for text embeddings.
