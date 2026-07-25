# Text embeddings with `rwkv_metal`

This guide covers turning RWKV-7 into a text embedding model on Apple Silicon:
pulling vectors out of a base checkpoint, contrastive fine-tuning them with
GradCache, and evaluating the result.

RWKV is an RNN, so it already folds an entire sequence into a fixed-size state.
Pooling the hidden state gives a usable sentence embedding **with no training at
all** — the "free sentence embedding" property. Contrastive fine-tuning then
turns that rough signal into a real retrieval model: measured below, held-out
MRR goes from 0.035 to 0.725 in 750 steps.

The approach ports [howard-hou/EmbeddingRWKV](https://github.com/howard-hou/EmbeddingRWKV)
to MLX; the training recipe (curriculum staging, task losses) follows its
`sft_curriculum`, the memory schedule (GradCache) and the padding convention are
new here.

- [Concepts](#concepts)
- [Quick start: vectors from a base model](#quick-start-vectors-from-a-base-model)
- [Pooling and the terminator token](#pooling-and-the-terminator-token)
- [The three tasks](#the-three-tasks)
- [Data and batchers](#data-and-batchers)
- [Losses](#losses)
- [`EmbeddingModel` and the pooling head](#embeddingmodel-and-the-pooling-head)
- [`EmbedTrainConfig`: every switch](#embedtrainconfig-every-switch)
- [GradCache: negative pool vs memory](#gradcache-negative-pool-vs-memory)
- [Building the negative pool](#building-the-negative-pool)
- [Evaluation](#evaluation)
- [Full curriculum runs](#full-curriculum-runs)
- [Measured results](#measured-results)
- [Using a non-World base](#using-a-non-world-base)
- [Current limitations](#current-limitations)

---

## Concepts

Three layers, each usable on its own:

| Layer | Module | What it gives you |
|---|---|---|
| Inference | `embedding/embed.py` | `Embedder` — vectors from any base model, no training |
| Training | `embedding/{train,loss,tasks,dataset}.py` | contrastive fine-tuning, one trainer for full-FT / frozen / LoRA / QLoRA |
| Memory | `embedding/gradcache.py` | decouples the contrastive batch size from activation memory |

Everything runs on `model.body(idx) -> [B, T, D]` — the hidden states before the
language-model head. The vocab projection is never used, so an embedding model
carries the LM head as dead weight (see [Current limitations](#current-limitations)).

---

## Quick start: vectors from a base model

```python
import rwkv_metal as rk
from rwkv_metal.embedding import Embedder, cosine_similarity_matrix

model, cfg = rk.load_pretrained("weights/RWKV-x070-World-0.1B.pth")
tok = rk.WorldTokenizer()

emb = Embedder(model, tok)                       # terminator=0, pooling="last"
vecs = emb(["The cat sits by the window.",
            "A cat settled on the windowsill."])  # [2, D], L2-normalized
print(cosine_similarity_matrix(vecs))
```

`embed_texts(model, tok, texts)` is the one-shot form if you don't want to keep
the object around.

Each text is a **separate forward pass**, deliberately: batching would right-pad
them to a common length, and while padding after the pooled position is
harmless (see below), a per-text pass keeps the code obvious and costs little at
inference scale. The training path does batch, with explicit pool indices.

---

## Pooling and the terminator token

`Embedder(model, tokenizer, terminator=0, pooling="last")`

| Argument | Values | Meaning |
|---|---|---|
| `terminator` | int, or `None` | Token id appended to every text before pooling. `None` disables it. |
| `pooling` | `"last"` / `"mean"` | Read the state at the terminator position, or average over all positions. |

**Why a terminator at all.** With `pooling="last"` the vector is the model's
state at one specific position. Appending a fixed token gives every text the
same final "read" instruction, so the pooled position means the same thing
across texts of different lengths and endings.

**Which token.** It should be one the model never has to *predict* — otherwise
you are pooling at a position whose representation is busy doing language
modelling. For the World vocab that is id `0`, which is unmapped in
`rwkv_vocab_v20230424.txt`. For a BPE vocab, use a special token. `BPETokenizer`
exposes `.terminator_id`, which prefers `<eos>`, falls back to `<pad>`, then `0`.

This is worth measuring rather than assuming. On 200 held-out Russian retrieval
rows with an untrained ru60m base:

| terminator | MRR |
|---|---|
| `<eos>` (id 3) | **0.0425** |
| `<bos>` (id 2) | 0.0384 |
| `<pad>` (id 0) | 0.0376 |

The spread is small at this sample size — treat it as "no candidate is broken"
rather than a strong ranking, and re-check on your own vocab.

**Right-padding.** `encode_batch` pads on the **right**, unlike EmbeddingRWKV's
CUDA-kernel convention of left-padding. RWKV-7 is causal, so tokens after the
pooled position physically cannot influence it — no attention mask is needed and
padding simply sits outside the loss's computational path. `pool_idx[b]` carries
each row's terminator position.

---

## The three tasks

The reference dataset is [LitRetrieval](https://huggingface.co/datasets) —
554 096 `{anchor, positive, negative, task}` triplets over Russian and English
literature, with hard negatives mined by FAISS over Qwen-3 8B embeddings.

| Task | Rows | Relationship | Loss | Symmetric |
|---|---|---|---|---|
| `retrieval` | 186 949 | query → passage | `triplet_pool_loss` | no |
| `sts` | 181 480 | document ↔ document | `triplet_pool_loss` | yes |
| `classification` | 185 667 | passage → emotion label | `zero_shot_classification_loss` | n/a |

Curriculum training means running the trainer once per task, sequentially, over
the same model — one code path, three calls. That mirrors EmbeddingRWKV's own
staging.

**Language split matters.** Measured over the whole file with a script-count
heuristic: retrieval and sts are ~57% Russian (106 300 and 106 183 Russian rows
respectively), but classification is **100% English** — both the passages and
the seven emotion labels. A Russian-only base therefore gets a two-stage
curriculum, and `tools/run_embedding_curriculum.py` refuses `--lang ru` together
with `--stages classification` rather than silently producing an empty stage.

**Classification is zero-shot.** There is no learned classifier head: each row's
candidate labels are embedded like any other text and compared by cosine
similarity, exactly like retrieval. The label pool is parsed per row from the
instruction line in the anchor, because every row carries its own set — verified
on 20 000 sampled rows, all had a distinct 7-label set. Hence padding to the
batch's max `K` plus a mask.

---

## Data and batchers

```python
from rwkv_metal.embedding import load_triplets_jsonl, TripletBatcher

rows = load_triplets_jsonl("train.jsonl", task="retrieval", limit=20000, seed=1234)
batcher = TripletBatcher(rows, tok, batch_size=32, terminator=3, max_chars=800)
```

`load_triplets_jsonl` streams the file once with **reservoir sampling**
(Algorithm R). With `limit=N` at most `N` rows are ever in memory regardless of
file size, and the result is a genuine uniform sample — not the first `N` rows,
which for a task-sorted file would be a biased slice. The reference file is
2.6 GB, so this is not optional.

| Batcher | Yields | Pairs with |
|---|---|---|
| `PairBatcher` | `q_idx, q_pool, d_idx, d_pool` | `pair_loss` (in-batch negatives only) |
| `TripletBatcher` | `a_idx, a_pool, p_idx, p_pool, n_idx, n_pool` | `retrieval_loss`, `sts_loss` |
| `ClassificationBatcher` | `a_idx, a_pool, cand_idx, cand_pool, mask, target_idx` | `classification_loss` |

All three take `terminator` and `max_chars`. `max_chars` clips text before
tokenization; step time scales roughly linearly with it, so it is the cheapest
speed lever. Retrieval anchors are short (~210 chars, they are queries), but
sts and classification anchors run to ~1700 chars median and ~9000 max.

---

## Losses

All losses take L2-normalized `[B, D]` embeddings and a `temperature`
(default 0.05).

**`info_nce_loss(q, d, temperature)`** — symmetric InfoNCE, off-diagonal batch
pairs as negatives. Kept for data with no explicit hard negatives.

**`triplet_pool_loss(anchor, positive, negative, temperature, symmetric)`** —
the workhorse. The candidate set is `concat(all positives, all hard negatives)`,
so with batch `B` each anchor is scored against **2B candidates**: its own hard
negative, plus every other row's positive *and* hard negative. That extra pool
is free — those embeddings were computed anyway.

- `symmetric=False` (retrieval): anchor → candidate only. A query and the
  passage that answers it do not play interchangeable roles, so training
  passage → query would be modelling a relation that isn't there.
- `symmetric=True` (sts): both directions, since anchor and positive are both
  documents in a genuinely symmetric relationship. The reverse half uses only
  the `[B, B]` positive block — hard negatives have no anchor to point back to.

**`zero_shot_classification_loss(anchor, candidates, mask, target_idx, temperature)`**
— `[B, K, D]` candidates with a `[B, K]` mask, masked logits set to `-1e9`
before cross-entropy.

---

## `EmbeddingModel` and the pooling head

```python
from rwkv_metal.embedding import EmbeddingModel
model = EmbeddingModel(base)          # base = RWKV7 or RWKV7X070
vecs = model.embed(idx, pool_idx)     # [B, D], L2-normalized
```

`EmbeddingHead` is a zero-init residual block: `fc2.weight` starts at zero, so
at step 0 the head is the identity and does not disturb the base's pretrained
geometry. It learns a correction from there.

> **The head is a sibling of the base, not a submodule.** `add_lora()` and
> `quantize_base_model()` call `model.freeze()` on the base tree. If the head
> lived inside that tree it would be silently frozen, and LoRA runs would train
> adapters only — with no error and a plausible-looking loss curve.

---

## `EmbedTrainConfig`: every switch

```python
from rwkv_metal.embedding import EmbedTrainConfig, finetune_embedding
cfg = EmbedTrainConfig(lr=2e-5, max_steps=1000, warmup_steps=100)
finetune_embedding(model, batcher, cfg, retrieval_loss, gradcache_spec=RETRIEVAL_GC)
```

### Optimization

| Field | Default | What it does |
|---|---|---|
| `lr` | `2e-5` | AdamW peak learning rate. Full-FT of a pretrained base wants a small value; 2e-5 is the validated setting. |
| `grad_clip` | `1.0` | Max gradient norm. Contrastive gradients spike early — measured norms of 400–600 in the first steps — so this is load-bearing, not decoration. |
| `weight_decay` | `0.0` | AdamW decay. Zero by default: the base is pretrained, and decaying it toward zero discards that. |
| `beta1`, `beta2` | `0.9`, `0.99` | AdamW moments. |
| `adam_eps` | `1e-8` | |
| `temperature` | `0.05` | Softmax temperature on cosine logits. Lower = sharper = harsher penalty on near-misses. Interacts with pool size: a bigger pool at a low temperature makes false negatives more damaging. |

### Schedule

| Field | Default | What it does |
|---|---|---|
| `max_steps` | `1000` | Optimizer steps. Batches cycle, so this may exceed one epoch. |
| `grad_accum` | `1` | Micro-batches summed per step. **Do not use this to grow the negative pool** — see below. |
| `warmup_steps` | `0` | Linear ramp from 0 to `lr`. Worth setting (~10% of steps) given the early gradient spikes. |
| `lr_schedule` | `"cosine"` | `"cosine"` / `"linear"` / `"constant"` decay after warmup. |
| `lr_min` | `0.0` | Floor the decay lands on. |

### Memory

| Field | Default | What it does |
|---|---|---|
| `grad_checkpoint` | `True` | Recompute block activations in backward. Trades ~30% time for a large memory cut. |
| `cache_limit_gb` | `1.5` | MLX allocator cache ceiling. |
| `gradcache_chunk` | `0` | Rows per GradCache chunk. `0` = eager. Needs a `gradcache_spec` to take effect. |

### Logging

| Field | Default | What it does |
|---|---|---|
| `log_every` | `10` | Step-log interval. |
| `save_every` | `0` | Checkpoint interval; `0` = only at the end. Worth setting for multi-hour runs — a run killed mid-stage otherwise leaves nothing. |
| `checkpoint_path` | `"embedding_model.safetensors"` | Destination. |

`save_embedding_model` writes **trainable parameters only**: the whole model
under full-FT, just adapters and head under LoRA. Load it back with
`model.update(tree_unflatten(list(mx.load(path).items())))`, or use
`tools/eval_embedding_checkpoint.py`, which also rebuilds the run's held-out
split from its seed.

### Why the trainer uses `nn.value_and_grad`

Unlike the pretrain trainer (`mx.value_and_grad`), this one is `freeze()`-aware.
That single choice is why the *same* loop covers full fine-tune, frozen bottom
layers, LoRA, and QLoRA on a 4-bit base, with no branching. Structurally
verified: wrapping the base in `add_lora(rank=8)` gave 1.77M / 192.8M trainable
(0.92%) and loss 1.52 → 1.30 in 10 steps, with no trainer changes.

---

## GradCache: negative pool vs memory

`triplet_pool_loss`'s negative pool **is** the batch, so retrieval quality
scales with batch size — but so does activation memory. GradCache
([Gao et al. 2021](https://arxiv.org/abs/2101.06983)) breaks that coupling:

1. Forward every chunk with no grad; cache the `[N, D]` embeddings and `mx.eval`
   them so no autograd graph survives.
2. Compute the loss on the **full** cached matrix and take `dL/dE`. This is
   where the whole batch interacts — an `[N, N]` matmul plus softmax.
3. Re-forward each chunk with grad, seeded with its slice of `dL/dE`.

Phase 3 uses the identity `d/dθ Σ(embed(chunk) · stop_grad(dL/dE)) == VJP`,
because MLX's `mx.vjp` takes flat array lists rather than pytrees. Routing
through `nn.value_and_grad` keeps it `freeze()`-aware, so GradCache composes
with LoRA and QLoRA exactly like the eager path.

**It is exact, not an approximation.** Measured on the 0.1B checkpoint, fp32,
batch 8:

| Setup | loss diff | gradient rel. diff |
|---|---|---|
| GradCache, 1 chunk | `0.0` exactly | `1.5e-8` |
| GradCache, 4 chunks | `2.4e-7` | `1.3e-4` |
| grad accumulation, 4 chunks | — | **`3.9`** |

The 1-chunk row landing exactly on the eager-vs-eager noise floor is the
decisive result: cutting the graph at `E` and re-seeding introduces no error at
all. The `1.3e-4` at 4 chunks is floating-point summation *order* — inherent to
any chunked accumulation.

The last row is the point. **Gradient accumulation is not a substitute.** It
deviates by ~400% because each micro-batch only sees its own negatives, which
genuinely changes the contrastive math. Use `grad_accum` to smooth gradient
noise; use GradCache to grow the pool.

### Two independent knobs

- **`batch_size`** — the negative pool. Each anchor is contrasted against
  `2 × batch_size` candidates. This is a **quality** knob.
- **`gradcache_chunk`** — how many rows are forwarded at once. This is a
  **memory** knob and does not change the math.

Measured memory on the 0.1B, bf16, 800-char passages, `chunk=4`:

| batch | eager | GradCache |
|---|---|---|
| 8 | 3.43 GB | 3.15 GB |
| 16 | 4.50 GB | 3.17 GB |
| 32 | 7.00 GB | 3.20 GB |
| 48 | 9.68 GB | 3.20 GB |

Eager grows linearly with the pool; GradCache stays flat. Losses agreed to four
decimals at every size.

### Picking `gradcache_chunk`

Bigger chunks are *not* reliably faster — measured on ru60m, batch 32, 800
chars:

| chunk | s/step | peak |
|---|---|---|
| 4 | 14.5 | 2.21 GB |
| 8 | 17.6 | 2.89 GB |
| 12 | **13.8** | 3.47 GB |
| 16 | 16.7 | 3.72 GB |

Non-monotonic, so tune it empirically on your own data rather than reasoning
about GPU occupancy. (The chunk-12 row was measured on the real 20k-row slice
and the others on a 300-row sample, so the text-length mix differs slightly —
another reason to measure rather than interpolate.) Total compute per step is
roughly chunk-independent; only the schedule changes.

Classification has no GradCache spec on purpose: its candidate pool is per-row,
not shared across the batch, so a bigger batch buys it no extra negatives.

---

## Building the negative pool

This is the lever that decides retrieval quality, so it is worth being
deliberate about.

**1. The pool is `2 × batch_size`, and it is free.** Every anchor sees its own
mined hard negative plus every other row's positive and negative. Going from
batch 8 to 32 takes the pool from 16 to 64 candidates at flat memory under
GradCache. Start at 32 and raise it as long as time allows.

**2. Watch the loss for pool exhaustion.** In the run below, loss fell 2.72 →
0.26 and then plateaued while held-out MRR was still only 0.73. A cross-entropy
of 0.26 over 64 candidates means the model assigns the right answer ~0.77
probability — the random in-batch negatives have stopped being informative and
the gradient has mostly vanished. At that point more *steps* buy little; more or
harder *negatives* buy a lot. Read the training loss as a pool-difficulty
gauge, not as progress.

**3. Mined hard negatives are only as good as the miner.** LitRetrieval's
negatives come from FAISS over Qwen-3 8B embeddings, so they are hard by
construction. But `triplet_pool_loss` weights them identically to the incidental
in-batch ones — there is no separate hard-negative term. If your data has no
mined negatives, `PairBatcher` + `info_nce_loss` degrades gracefully to
in-batch-only, which is meaningfully weaker.

**4. False negatives get worse as the pool grows.** With 64 candidates drawn
from one literary corpus, some other row's positive is genuinely relevant to
your anchor — and the loss punishes the model for ranking it highly. This is the
real ceiling on batch size, and it arrives before the memory ceiling does.
**There is no false-negative filtering implemented yet** (see
[Current limitations](#current-limitations)); a similarity threshold against the
anchor is the usual remedy.

**5. Temperature and pool size interact.** At `temperature=0.05` the softmax is
sharp, so a false negative near the top of a large pool produces a large
gradient. If you push the batch well past 64, consider raising the temperature
rather than leaving it at the small-pool default.

**6. Don't grow the pool with `grad_accum`.** It doesn't — see the 3.9 row
above. It averages gradients across micro-batches that each saw only their own
negatives.

**7. Symmetry is a property of the data, not a tuning knob.** `symmetric=False`
for query→document, `True` for document↔document. Getting it backwards trains a
relation the data doesn't contain.

---

## Evaluation

```python
from rwkv_metal.embedding import evaluate_retrieval, evaluate_sts_pairwise, evaluate_classification
evaluate_retrieval(model, tok, held_out_rows, max_chars=800, terminator=3)
```

Dependency-free (MLX + stdlib), enough signal to tell whether a stage helped —
not an MTEB harness.

| Function | Metrics | Candidate pool |
|---|---|---|
| `evaluate_retrieval` | MRR, Recall@k, nDCG@10 | every eval row's positive + every eval row's negative (`2N`) |
| `evaluate_sts_pairwise` | pairwise accuracy, mean sim(pos)/sim(neg) | the row's own pair |
| `evaluate_classification` | zero-shot top-1 | the row's own label pool |

**The pool size is part of the metric.** Doubling `n_eval` doubles the
candidates and makes MRR strictly harder — numbers from different `n_eval` are
not comparable. Always report both.

`evaluate_sts_pairwise` is **not** a Spearman-correlation STS eval. LitRetrieval
provides binary positive/negative pairs, not graded human similarity scores, so
the honest measurement is "does `cos(a, pos)` beat `cos(a, neg)`". Its
`mean_sim_pos` / `mean_sim_neg` gap is the more informative number, since
accuracy saturates near 1.0 quickly.

---

## Full curriculum runs

`tools/run_embedding_curriculum.py` is the production runner (as opposed to
`tools/test_embedding_curriculum_smoke.py`, which is a 30-step smoke test).

```bash
python tools/run_embedding_curriculum.py \
    --model /path/to/ru60m \
    --data  /path/to/train.jsonl \
    --out_dir runs/ru60m_curriculum \
    --lang ru --stages retrieval,sts \
    --n_per_task 20000 --n_eval 500 \
    --steps 1000 --batch_size 32 --gc_chunk 12 --max_chars 800 \
    --lr 2e-5 --warmup 100 --save_every 250
```

| Flag | What it does |
|---|---|
| `--model` | A `.pth` (official World weights + `WorldTokenizer`) **or** a checkpoint directory (`config.json` + `model.safetensors` + `tokenizer*.json`, with its own BPE tokenizer). Detected by whether the path is a directory. |
| `--data` | The triplet JSONL. Streamed once. |
| `--lang` | `any` / `ru` / `en`. Filters rows *before* the reservoir counter, so the sample stays uniform over the kept subset. |
| `--stages` | Comma-separated subset of `retrieval,sts,classification`, run in the order given. |
| `--n_per_task`, `--n_eval` | Training and held-out slice sizes per task. The eval slice is cut from the same reservoir sample, so overlap is impossible. |
| `--steps` | Optimizer steps per stage. |
| `--batch_size`, `--gc_chunk` | Negative pool and memory schedule for retrieval/sts. |
| `--cls_batch` | Classification batch (eager). It also embeds `B × K` labels per step, so keep it small. |
| `--max_chars` | Text clip length. Roughly linear in step time. |
| `--lr`, `--warmup`, `--lr_min` | Schedule. |
| `--save_every` | Checkpoint interval. |
| `--seed` | Seeds the reservoir *and* the train/eval split, so a run is exactly reproducible — which is what lets `eval_embedding_checkpoint.py` rebuild the same held-out rows later. |

Everything — config, per-step losses, before/after metrics, timings, peak
memory — is streamed to `<out_dir>/run.json` every 50 steps, so a run is
inspectable while in flight and survives being killed.

If a run *is* killed mid-stage, the periodic checkpoint has weights but no
`eval AFTER`. Recover the numbers with:

```bash
python tools/eval_embedding_checkpoint.py \
    --run_dir runs/ru60m_curriculum --stage retrieval
```

It reads `run.json`, rebuilds the identical held-out split from the recorded
seed, and prints before/after side by side.

---

## Measured results

Base: ru60m (RWKV-7, 18 layers, 448 dim, 16k Russian BPE, 61.3M params).
Data: LitRetrieval, Russian rows only. Hardware: M4, 16 GB.

Stage `retrieval`, batch 32, `gc_chunk` 12, 800 chars, 20 000 training rows,
750 steps (a 1000-step run stopped early), 13.8 s/step, peak 3.47 GB.
Held-out: 500 rows, 1000-candidate pool.

| metric | before | after | |
|---|---|---|---|
| MRR | 0.035 | **0.725** | ×21 |
| Recall@1 | 0.018 | **0.646** | ×36 |
| Recall@5 | 0.034 | **0.820** | ×24 |
| Recall@10 | 0.054 | **0.880** | ×16 |
| nDCG@10 | 0.033 | **0.759** | ×23 |

Training loss went 2.72 → 0.26 and plateaued after ~step 700. The held-out
numbers rule out memorization: 750 × 32 = 24 000 samples over 20 000 rows is
~1.2 epochs, and the metrics are on rows the model never saw. As argued in
[Building the negative pool](#building-the-negative-pool), the plateau is pool
exhaustion, not convergence.

For scale, on the same hardware the 0.1B World base runs ~25 s/step at batch 32
with a 5.77 GB peak (`gc_chunk` 4), against ru60m's 13.8 s and 3.47 GB
(`gc_chunk` 12) — roughly 1.8× faster and 1.7× lighter, at the cost of being
Russian-only. The chunk settings differ, so treat this as an order-of-magnitude
comparison rather than a controlled one.

Earlier smoke-test numbers on the 0.1B World base (30 steps, 500 rows/task,
100 eval rows / 200-candidate pool — a much easier pool, not comparable to the
table above):

| stage | metric | before | after |
|---|---|---|---|
| retrieval | MRR | 0.047 | 0.71 |
| sts | pairwise acc | 0.97 | 0.98 |
| | sim(pos) − sim(neg) | 0.22 | 0.51 |
| classification | top-1 (7 classes) | 0.15 | 0.32 |

---

## Using a non-World base

A checkpoint directory holding `config.json` + `model.safetensors` +
`tokenizer*.json` — the layout produced by remapping from
`flash-linear-attention` — loads directly:

```python
import rwkv_metal as rk
base, cfg = rk.load_local_rwkv7("/path/to/ru60m")   # -> RWKV7X070
tok  = rk.load_local_tokenizer("/path/to/ru60m")    # -> BPETokenizer
```

LoRA ranks are read off the weights themselves rather than the config, since
fla picks them by its own formula and rarely records them. `cmix.x_k` is
reshaped from `[D]` to `[1, 1, D]`. Loading is strict by default: any missing or
mismatched tensor raises, because a silently half-loaded checkpoint produces
plausible-looking but meaningless embeddings.

> **Pick the architecture, don't guess it.** `RWKV7` (the from-scratch
> reference) and `RWKV7X070` (official x070) have **identical tensor names and
> shapes** but different math — tanh inside the ICLR path, LayerNorm instead of
> per-head GroupNorm, a different ln_x/bonus order, inter-block token-shift
> carry. A checkpoint therefore loads into the wrong one without a single error
> and quietly degrades. Measured on ru60m over 40 Russian passages: PPL **42.3**
> as `RWKV7X070` versus **254** as `RWKV7`, against 16000 for random weights.
> `load_local_rwkv7` defaults to `arch="x070"`; verify with
> `tools/verify_local_checkpoint.py`, which measures both and reports the
> winner.

---

## Current limitations

- **No false-negative filtering.** Other rows' positives enter the negative pool
  unchecked. This is the first thing to fix before pushing batch size much past
  64 — see [Building the negative pool](#building-the-negative-pool).
- **One head for all tasks.** EmbeddingRWKV splits `[CLS]` / `[STS]` / `[RETR]`;
  here a single `EmbeddingHead` is shared, so curriculum stages overwrite each
  other's specialization to some degree.
- **Fixed temperature.** Not learnable, and no Matryoshka (nested-dimension)
  training.
- **The LM head is dead weight.** `body()` never calls it, but under full-FT it
  still occupies optimizer state — 50M of the 0.1B base's 192M parameters.
  Freezing `base.head` before training is an easy win that isn't done
  automatically.
- **No reranker.** The larger missing piece. It needs per-layer
  `(wkv_state, x_prev, v_first)` threaded out through `body()` / `RWKVBlock` /
  `Tmix` so a document's state can be cached and only the query recomputed —
  `O(L_query)` instead of `O(L_doc + L_query)`. The foundation exists:
  `make_wkv7_checkpoint_with_state` already provides a differentiable `h_in`
  with a full VJP, and the state format matches the original's `state[1]`
  (`[Layers, Batch, Heads, HeadSize, HeadSize]`) exactly. It just isn't exposed
  — `make_wkv7_checkpoint` hard-zeroes `h0`.

See also: [`inference.md`](./inference.md) for running the base model,
[`lora.md`](./lora.md) for parameter-efficient fine-tuning,
[`pretraining.md`](./pretraining.md) for training from scratch.
