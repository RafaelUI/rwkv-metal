# Reranking with `rwkv_metal`

This guide covers the RWKV-7 **cross-encoder reranker**: scoring
`(query, document)` pairs directly from the model's recurrent state, training
the scoring head, and serving it with a cached document index.

A reranker is the second stage of retrieval. An embedding model
([`embedding.md`](./embedding.md)) is fast but scores a query and a document
independently — it never sees them together. A cross-encoder reads the pair
jointly, which is far more accurate and far more expensive, so it only runs on
the handful of candidates the embedder already shortlisted.

- [The idea](#the-idea)
- [Quick start](#quick-start)
- [Why the document comes first](#why-the-document-comes-first)
- [Anatomy of the head](#anatomy-of-the-head)
- [`RerankerConfig`: every switch](#rerankerconfig-every-switch)
- [The state cache: why training is cheap](#the-state-cache-why-training-is-cheap)
- [Data and candidates](#data-and-candidates)
- [Losses](#losses)
- [`RerankTrainConfig`: every switch](#reranktrainconfig-every-switch)
- [Evaluation](#evaluation)
- [Full runs](#full-runs)
- [Measured results](#measured-results)
- [Model-level state API](#model-level-state-api)
- [Practical advice](#practical-advice)
- [Current limitations](#current-limitations)

---

## The idea

A transformer cross-encoder concatenates query and document, runs the stack,
and reads a score off `[CLS]`. RWKV already folds the whole pair into a
fixed-size recurrent state `h [H, S, S]` per layer, so the score can be read
straight from the state instead of from per-token activations.

Concretely: run the pair through a **frozen** base model, take the final WKV
state, then push one small learnable "probe" token through a short stack of
RWKV blocks whose recurrence *starts* from that state, and project the result
to a scalar. Reading `y = h·r` is one attention-like query against everything
the state accumulated.

Three consequences:

| | |
|---|---|
| The head sees the state matrix `[S, S]` per head, not just a `[D]` vector | more to read than a pooled embedding |
| Head cost is independent of pair length | one or two tokens per block, always |
| The prefix state is cacheable | a document is encoded once, each query costs only its own tokens |

The design follows the reranker in
[howard-hou/EmbeddingRWKV](https://github.com/howard-hou/EmbeddingRWKV) in
spirit; the padding scheme, the state cache, the candidate construction and the
listwise loss are different (see [Differences](#differences-from-embeddingrwkv)).

---

## Quick start

```python
import rwkv_metal as rk
from rwkv_metal.reranker import Reranker, RerankerInference

base, cfg = rk.load_pretrained("weights/rwkv7-g1d-0.1b.pth")

# reads which layers the head was trained on straight from the checkpoint
model = Reranker.from_head(base, "reranker_head.safetensors")
rr = RerankerInference.from_checkpoint(model, rk.WorldTokenizer(),
                                       "reranker_head.safetensors")

docs = [...]                                  # top-k from your embedder
for doc_id, score in rr.rank("how do bees overwinter?", docs, top_k=5):
    print(f"{score:+.2f}  {docs[doc_id][:80]}")
```

Scores are raw logits: comparable **within** one query, not across queries
unless you trained with a pointwise term (see [Losses](#losses)).

Use `from_head` / `from_checkpoint` rather than constructing by hand. A head
trained on layer 5 and a head trained on layer 11 have **identical tensor
shapes**, so loading one into the other's model succeeds silently and produces
confident nonsense. The checkpoint carries its configuration and the text
contract (template order, truncation limits, terminator, instruction) in
safetensors metadata, and `load_head` refuses a mismatch:

```python
Reranker(base, RerankerConfig(layer_idx=(11,))).load_head(path)
# ValueError: чекпоинт не соответствует модели: layer_idx='5' в файле против '11' здесь
```

Pass `strict=False` to load a checkpoint written before this metadata existed.

### Serving many queries against the same documents

```python
index = rr.build_index(docs)                  # encodes prefixes once
scores = rr.score_indexed("how do bees overwinter?", index)
```

`build_index` stores the recurrent state after `Instruct: … \nDocument: {doc}\n`.
Every later query only pays for its own ~30 tokens. On a 0.1B base this is
**73 ms → 3.5 ms per pair**, about 20×. The cost is memory: a full state is
`n_layer · n_head · 64 · 64 · 4` bytes — 2.4 MB per document at 0.1B — so an
index is for a hot subset (the embedder's top-100), not a whole corpus.

An index is only valid for the base, template, truncation and instruction it was
built with — the state is a function of exactly that prefix. `DocIndex` records
those and `score_indexed` raises on a mismatch instead of returning plausible
numbers. The one thing it cannot check is that you swapped the base checkpoint
underneath it.

The indexed and the direct path are the same computation, but not bit-identical:
splitting the sequence changes the order of accumulation, and official
checkpoints are bf16. Measured on the 0.1B base, scores agreed to
`max|Δ| = 0.023` on a score range of about 13 — far below any gap that decides
an ordering, but do not expect the two paths to match to the last digit.

---

## Why the document comes first

The default template is

```
Instruct: {instruction}
Document: {document}
Query: {query}
```

Document before query, which reads backwards until you remember that RWKV is an
RNN: the state after a prefix depends only on that prefix. Putting the document
first makes `Instruct + Document` a cacheable prefix. Putting the query first
would throw that away — the state would depend on the query, and nothing could
be precomputed.

The trade-off is real but small: with the document first, the model encodes it
without yet knowing the query. It is not "blind", though — the head reads the
state *after* the query tokens have been folded in, so query-conditioned
retrieval still happens; it just happens in the head's read-out rather than
during document encoding.

The other order is available for comparison:

```python
from rwkv_metal.reranker import PairTemplate
rr = RerankerInference(model, tok, template=PairTemplate(doc_first=False))
```

`build_index` refuses to run in that mode rather than silently returning a
useless index.

### Does the document survive the query tail?

Fair question for an RNN: does the document still matter after 30 more tokens
of decay? Measured on the 0.1B World base, cosine between the final states of
two unrelated documents under the same query:

| query tail length | 4 tok | 19 tok | 34 tok | 64 tok | 124 tok | 244 tok |
|---|---|---|---|---|---|---|
| cos(state_A, state_B) | 0.9960 | 0.9974 | 0.9977 | 0.9974 | 0.9974 | 0.9974 |

Flat. RWKV-7's decay is per-channel and many channels sit at `w ≈ 1`, so
document identity is not washed out by the tail — it stops changing after a few
tokens and stays. (A *randomly initialised* model behaves very differently:
`w ≈ 0.74` uniformly, and the document is gone within twenty tokens. That is a
property of untrained weights, not of the architecture — relevant only if you
write tests against a toy base, see `tools/test_reranker_smoke.py`.)

---

## Anatomy of the head

```
probe token(s)  ──▶ ln0 ──▶ RWKV block 0 ──▶ … ──▶ RWKV block n-1 ──▶ ln_out ──▶ Linear ─ tanh ─ Linear ──▶ scalar
                              ▲                        ▲
                    h_in = base state of          h_in = base state of
                        layer_idx[0]                 layer_idx[n-1]
```

- **Probe tokens** are learnable embeddings, not vocabulary tokens. Their scale
  does not matter — `ln0` normalises each token.
- **Blocks** are ordinary `RWKVBlock`s from the x070 architecture, initialised
  from the base's blocks at `layer_idx`. They are the same code path used in
  the base model, so no separate kernel or numerics.
- Because a block runs on a single token, the WKV recurrence takes the
  `wkv7_step` path — plain MLX ops rather than the Metal checkpoint kernel,
  which needs `T` to be a multiple of 16. Same math, 16× less work, autograd
  for free.
- The final `Linear` is **zero-initialised**: before training all scores are
  exactly `0`, so the listwise loss starts at exactly `ln(C)`. If your first
  logged loss is not `ln(C)`, the data or the head is wired wrong — that is the
  cheapest bug detector in the pipeline.

The head is a **sibling** of the base, never a submodule of it. `base.freeze()`
(and `add_lora` / `quantize_base_model`, which call `freeze()` internally)
would silently freeze the head if it lived inside that tree.

Parameter count: one block at `D=768` is ~7.5 M, plus ~0.6 M for the MLP. So a
one-layer head is ~8.1 M trainable parameters, a two-layer head ~15.6 M.

---

## `RerankerConfig`: every switch

```python
from rwkv_metal.reranker import Reranker, RerankerConfig

model = Reranker(base, RerankerConfig(
    layer_idx    = (5, 11),
    shared_state = False,
    n_probe      = 1,
    head_hidden  = None,
))
```

| Field | Default | What it does |
|---|---|---|
| `layer_idx` | `(-1,)` | Which base layers the head reads. One block per index; negative indices count from the end. More layers = more signal from different depths, more parameters, and a proportionally bigger state cache. |
| `shared_state` | `False` | Every block reads the **last** layer's state while the stack keeps its depth. A way to make the head deeper without making it read more layers (and without growing the cache). |
| `n_probe` | `1` | Number of probe tokens. Each is another read of the state; the score comes from the last one. |
| `head_hidden` | `n_embd` | Hidden width of the scoring MLP. |

`Reranker(base, cfg, freeze_base=True, init_from_base=True,
head_dtype=mx.float32)` — the extra flags exist for experiments. `head_dtype`
deserves a word: `init_from_base` copies the base's weights as they are, and
official checkpoints are bf16, so without this the head would *train* in bf16.
With 8 mantissa bits, weights around 0.05 and `lr ≈ 1e-4`, part of every
optimiser step falls below the representable quantum and is rounded away. The
head is 8–23 M parameters, so fp32 costs nothing. Measured effect on this
benchmark: +0.004 MRR for a one-layer head, nil for a three-layer one — i.e.
inside the noise. Fixed on principle, not because it showed up.

### Choosing layers

This is the one choice that measurably matters, and the last layer is not the
right default. Three seeds each, same cache, same schedule:

| layers | held-out MRR |
|---|---|
| `(0,)` | 0.555 ± 0.011 |
| `(11,)` — last | 0.922 ± 0.014 |
| `(5,)` — middle | **0.978 ± 0.004** |
| `(0, 5, 11)` | 0.976 ± 0.006 |

Layer 0 is barely better than the raw embedder (0.491): its state has not yet
accumulated anything worth reading. The last layer works but loses ~0.06 MRR to
the middle one — consistent with the state geometry, where layer 11 is dominated
by a component shared across all documents (cos ≈ 0.996 between unrelated
documents, versus 0.955 at layer 5), so the discriminative part is a smaller
fraction of what the head reads. Combining layers does **not** help here: adding
layer 0 and layer 11 to layer 5 is within noise of layer 5 alone.

Start from the middle of the stack, and sweep. Comparing is cheap — cache a
superset of layers once and slice it per configuration:
`tools/run_reranker.py --cache_layers 0,5,11 --configs -1 5 0,5,11`, or
`tools/ablate_reranker.py` for multi-seed comparisons on an existing cache.

---

## The state cache: why training is cheap

With the base frozen, "pair text → state" is a fixed function. So it does not
belong inside the training loop:

1. **Encode once.** `encode_pairs` folds every `(document, query)` pair into a
   state, reusing the document prefix across all queries that share it.
2. **Keep only what the head reads.** `RerankerHead.unique_sources` — one layer
   out of twelve by default: 98 KB per pair in fp16 versus 2.4 MB for a full
   fp32 state, 25× less. (fp16 rather than bf16: same size, 10 mantissa bits
   instead of 7, and the range is never in question — states peak around 50
   against an fp16 ceiling of 65504. `encode_pairs` checks and raises rather
   than silently saturating.)
3. **Train on the cache.** A step touches only the head: one or two RWKV blocks
   on a single token. An epoch over tens of thousands of pairs takes seconds.

```python
from rwkv_metal.reranker import encode_pairs, build_candidates, load_rows

rows = load_rows("train.jsonl", task="retrieval", limit=1000)
pool, samples = build_candidates(rows, n_candidates=8)
cache = encode_pairs(model, tok, pool, samples,
                     max_doc_tokens=512, max_query_tokens=96,
                     out_path="cache.npy")     # пишется прямо на диск
```

Reload it later with `StateCache.load("cache.npy")` — memory-mapped by default.

Measured on an M4 Air, 0.1B base, 1000 queries × 8 candidates:

| | |
|---|---|
| unique document prefixes | 1986 out of 8000 pairs (75 % saved by dedup) |
| encoding | ~5 min, once |
| cache size | 2.36 GB (3 layers, fp16) |
| training | seconds per epoch |

Practical consequence: hyperparameter search, layer ablations and long
schedules become free. The expensive part runs once and is reused via
`--cache_path`.

### The cache lives in numpy, not MLX

`StateCache.states` is a numpy array — optionally a `np.memmap` backed by the
`.npy` file on disk. Two reasons, both about not running out of memory on a
16 GB machine:

- **No duplicate.** Scattering batch results into an `mx.array` rebuilds the
  whole array each time, so a 2.4 GB cache costs gigabytes of transient
  allocation per batch. numpy writes in place.
- **It does not have to fit.** With `out_path=` (or `--cache_path`), states are
  written straight to a memmap and paged in on access. A cache larger than RAM
  is fine; a cache that *almost* fits is the dangerous case, because the machine
  swaps and every timing you take afterwards is fiction.

Only the batch crosses into MLX, in its own size: `cache.gather(rows)`.

> **Measure memory with the system, not with MLX.** `mx.get_peak_memory()`
> reports the MLX pool and knows nothing about numpy buffers, model weights,
> memory-mapped pages, or whether the machine swapped — and a process that
> swaps invalidates every timing taken alongside it. On the full run the MLX
> pool reported 2.29 GB against an actual process RSS of 3.98 GB.
> `tools/bench_reranker.py` prints process RSS plus the swapin/swapout delta and
> says outright when a measurement is untrustworthy; for a long run, sample
> `ps -o rss= -p <pid>` from outside the process.
>
> This is not a hypothetical. An early version of `StateCache.save` re-wrote the
> `.npy` from a memmap of that same file — truncating the file underneath the
> array it was reading — which drove the process to **11.7 GB** and corrupted
> the cache. Peak RSS after the fix: **4.0 GB**. Nothing in the MLX counters
> would have shown it. `tools/test_reranker_smoke.py` now covers the
> save → memmap-load → re-save cycle.

### `mx.compile`

The head is a few hundred small operations on a single token, so its cost is
dispatch, not arithmetic. Compiling it (on by default, `RerankTrainConfig.compile`
and `RerankerInference(compile=True)`):

| | eager | compiled |
|---|---|---|
| training step (32 queries × 8 candidates) | 49.1 ms | **34.5 ms** |
| head forward, 256 pairs | 13.4 ms | **7.1 ms** |
| base, one token through 12 layers | 10.6 ms | 9.9 ms |

The last row is the interesting one: the base forward barely moves, because it
is dominated by the WKV kernels and the `D×D` projections rather than by
dispatch. Compile where the operations are small and numerous, not everywhere.

One trap worth naming: `mx.compile` must be given `inputs=[head.state]`.
Without it the weights are frozen into the graph on the first call, and every
subsequent `load_head` or optimiser step silently does nothing.

The catch: the cache is only valid for that base, template, truncation and
candidate set. Change any of them and it must be rebuilt — `run_reranker.py`
checks shape compatibility and refuses to load a mismatched cache, but it
cannot detect a changed base checkpoint. Name your cache files accordingly.

---

## Data and candidates

Input format is the LitRetrieval-style jsonl used by the embedding side:

```json
{"anchor": "Instruct: ...\nQuery: ...", "positive": "...", "negative": "...", "task": "retrieval"}
```

```python
pool, samples = build_candidates(rows, n_candidates=8, seed=0)
```

Each sample gets:

1. its **positive**,
2. its **hard negative** (already mined in the dataset),
3. the rest sampled from the shared document pool.

The pool is deduplicated, so a document appearing under several queries is
encoded once — which is what makes extra candidates nearly free. The position
of the positive is **shuffled**; with a listwise loss, a fixed position is a
shortcut the head will happily learn instead of the task.

`load_rows` streams the file with reservoir sampling (Algorithm R), so a 2.6 GB
/ 554 k-row source never lands in memory — `limit` rows do, and they are a fair
uniform sample rather than "the first N".

`split_train_eval(samples, n_eval)` splits by **query**. Documents are shared
across the split on purpose: what is being measured is ranking, not memorising
which documents exist.

---

## Losses

```python
from rwkv_metal.reranker import listwise_loss, bce_loss, mixed_loss
```

| Loss | What it optimises | When |
|---|---|---|
| `listwise_loss` | softmax cross-entropy over the candidate list | **default.** Optimises the ordering, which is what the metrics measure |
| `bce_loss` | per-candidate binary cross-entropy (positive → 1, rest → 0) | the original EmbeddingRWKV recipe; calibrates the absolute level of the scores |
| `mixed_loss(α)` | `α · listwise + (1-α) · BCE` | use `α < 1` if you need scores comparable **across** queries (thresholding, score fusion) |

`RerankTrainConfig.loss_alpha` selects this; `1.0` (pure listwise) is the
default — chosen because it optimises the quantity being measured, **not**
because it measured better: on this benchmark listwise (0.9781 ± 0.0043), mixed
(0.9763 ± 0.0050) and pure BCE (0.9733 ± 0.0027) are indistinguishable. Expect
the difference to appear on a harder candidate set, not here.

Absolute-score calibration is the one concrete reason to keep a pointwise term.
If you only ever sort candidates within a single query, you do not need it.

---

## `RerankTrainConfig`: every switch

```python
from rwkv_metal.reranker import RerankTrainConfig, train_reranker

res = train_reranker(model, train_cache, eval_cache, RerankTrainConfig(
    lr=2e-4, batch_size=32, epochs=8,
))
```

| Field | Default | Notes |
|---|---|---|
| `lr` | `3e-5` | Conservative. The head is small and trained from scratch on a frozen feature map, so it tolerates much more — `1e-4`–`3e-4` is a reasonable range. Watch the first epochs: if the loss barely moves, raise it. |
| `weight_decay` | `0.01` | AdamW. The head is the only thing that can overfit here. |
| `grad_clip` | `1.0` | |
| `batch_size` | `32` | **queries** per step; pairs per step is `batch_size × n_cand`. Steps are cheap — the limit is state memory, not compute. |
| `epochs` | `4` | Cheap. Combine with `keep_best`. |
| `warmup_frac` | `0.05` | Fraction of total steps spent warming up. |
| `lr_schedule` | `"cosine"` | `cosine` / `linear` / `constant`. |
| `loss_alpha` | `1.0` | See [Losses](#losses). |
| `temperature` | `1.0` | Divides the logits in the listwise loss. |
| `compile` | `True` | `mx.compile` over the whole step (grad + clip + update). ~1.4× — see [`mx.compile`](#mxcompile). Turn it off only when debugging, since it makes tracebacks point into the compiled graph. |
| `eval_every` | `0` | Steps between held-out evaluations; `0` = once per epoch. |
| `keep_best` | `True` | Restore the weights of the best held-out epoch at the end instead of the last. |
| `checkpoint_path` | `reranker_head.safetensors` | Only the head is saved — the base is unchanged by definition. |

---

## Evaluation

```python
from rwkv_metal.reranker import evaluate
m = evaluate(model.head, eval_cache)
# {"mrr": ..., "recall@1": ..., "recall@3": ..., "recall@5": ..., "ndcg@10": ..., "loss": ...}
```

Ranks use **average tie-breaking**: `1 + (strictly greater) + (ties − 1)/2`.
This matters more than it sounds. An untrained zero-initialised head gives every
candidate the same score; with optimistic tie-breaking that scores a perfect
MRR of 1.0, and the "before" column of every table becomes a lie. With average
ties it reports `2/(C+1)` — exactly random guessing, which is the truth.

That same number is the floor to compare against: `0.222` for 8 candidates.
The more informative baseline is the **embedder on the same frozen base and the
same candidate sets** — it isolates what the head adds, with the base held
constant. `tools/run_reranker.py` reports both.

---

## Full runs

```bash
.venv/bin/python tools/run_reranker.py \
    --model ~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth \
    --data  ~/Develop/retrieval_literature/train.jsonl \
    --queries 1000 --candidates 8 --eval_queries 150 \
    --max_doc_tokens 512 \
    --cache_layers 0,5,11 --configs -1 5 0,5,11 \
    --epochs 8 --batch_size 32 --lr 2e-4 \
    --cache_path runs/reranker_0.1b/cache \
    --out runs/reranker_0.1b
```

The runner reads data, builds candidates, computes the random and embedder
baselines, encodes the state cache once, then trains every requested layer
configuration from that one cache. Everything — config, per-step losses,
before/after metrics, timings, peak memory — is written to `report.json` after
each stage, so a run is inspectable while it is still going and survives being
killed.

Re-running with the same `--cache_path` skips encoding entirely, which is what
makes hyperparameter search practical.

---

## Measured results

0.1B World base (`rwkv7-g1d-0.1b`), LitRetrieval `retrieval` task, 1000 queries
× 8 candidates (its positive, its mined hard negative, six sampled from the
pool), 150 held-out queries, documents truncated at 512 tokens, 8 epochs,
`lr=2e-4`, listwise loss. M4 Air, 8 GPU cores, 16 GB.

| | MRR | R@1 | nDCG@10 | vs hard neg | vs sampled neg |
|---|---|---|---|---|---|
| random guessing | 0.222 | 0.125 | — | 0.500 | 0.500 |
| embedder on the same frozen base | 0.491 | 0.253 | 0.615 | **0.493** | 0.749 |
| reranker, last layer `(-1,)` | 0.930 | 0.873 | 0.948 | 0.893 | 0.987 |
| reranker, middle layer `(5,)` | 0.969 | 0.940 | 0.977 | 0.947 | 0.998 |
| reranker, `(0, 5, 11)` | **0.975** | **0.953** | **0.981** | **0.953** | 0.997 |

(Single seed. `(5,)` and `(0,5,11)` swap places between runs — see the
three-seed ablation below, where they are indistinguishable.)

The last two columns are the ones that matter. Overall MRR flatters everyone:
telling a passage about beekeeping from a passage about steam engines is easy,
and six of eight candidates are that kind of negative. Split it apart and:

- the raw embedder is at **chance** (0.493) against the mined hard negative — it
  separates topics, not relevance;
- the reranker gets 0.96 on exactly that comparison, while being essentially
  perfect (0.999) on the easy one.

That gap is what a reranker is for. Both stages use the **same frozen base** —
the difference is entirely what the head does with the state.

### What actually moves the number

Ten configurations, three seeds each, same cache and schedule
(`tools/ablate_reranker.py`, held-out MRR):

| variant | MRR |
|---|---|
| layer `(5,)`, fp32 head | 0.9781 ± 0.0043 |
| layer `(5,)`, `shared_state` ×3 blocks | 0.9774 ± 0.0019 |
| layer `(5,)`, mixed loss α=0.7 | 0.9763 ± 0.0050 |
| layers `(0,5,11)`, bf16 head | 0.9759 ± 0.0029 |
| layers `(0,5,11)`, fp32 head | 0.9756 ± 0.0059 |
| layer `(5,)`, bf16 head | 0.9741 ± 0.0019 |
| layer `(5,)`, pointwise BCE | 0.9733 ± 0.0027 |
| layer `(5,)`, 2 probe tokens | 0.9730 ± 0.0028 |
| layer `(11,)` — last | 0.9219 ± 0.0141 |
| layer `(0,)` | 0.5552 ± 0.0107 |

Read this the right way. The top eight rows span 0.005 MRR while seed spread
within a variant is 0.002–0.006 — **they are indistinguishable**. Head depth,
probe count, listwise vs pointwise loss, fp32 vs bf16 head: none of them are
resolvable on this benchmark. Only the source layer separates, and it separates
hugely.

So the honest conclusion is not "listwise beats BCE" (it doesn't, here) but:
*this task saturates*. 8 candidates, 6 of them randomly sampled, 850 training
queries — the head reaches the ceiling almost immediately and everything after
that is noise. Distinguishing these choices needs a harder benchmark: more
candidates, all of them mined hard negatives, and more data. The defaults chosen
here (listwise, fp32 head, one probe) are picked on principle — they optimise
the right objective and don't throw away precision — not because this table
proves them better.

Cost, end to end:

| stage | time | peak process RSS |
|---|---|---|
| encoding 8000 pairs (1986 unique prefixes) into a 2.36 GB fp16 cache | ~4.5 min, once | 1.5 GB |
| training one head configuration, 8 epochs | 10–30 s | 4.0 GB |
| embedder baseline (917 documents + 150 queries) | 1.7 min | — |
| **full run, two head configurations** | **5.3 min** | **4.0 GB** |

Training peaks higher than encoding because it reads random rows across the
whole cache, so most of the memmap becomes resident — but those are file-backed
pages the OS can evict, not hard allocations.

Re-running from a saved cache: **1 min** without the embedder baseline. That is
what makes the ten-variant, three-seed ablation above a thing you run while
making coffee rather than a project.

Single-seed spread is ~±0.01 MRR for a good configuration and ~±0.014 for the
last-layer one; `--seed` fixes the head initialisation, and
`tools/ablate_reranker.py` averages over seeds.

### Caveats

- **The benchmark saturates.** See the ablation above: eight of ten
  configurations are statistically indistinguishable. Treat these numbers as
  "the head learns the task quickly and cheaply", not as a ranking of design
  choices, and not as a quality ceiling.
- **Six of eight candidates are randomly sampled**, which is why the headline
  MRR is high. The hard-negative column is the honest one.
- **Sampled negatives can be false negatives.** A candidate drawn from the pool
  is another query's positive; nothing checks whether it also answers *this*
  query. On literary passages collisions are rare, but the metric is optimistic
  by an unmeasured amount.
- **Train and eval share the document pool.** The split is by query, so eval
  queries are unseen, but a document may have been a training positive and
  appear as an eval negative. The shortcut this would enable ("this document is
  a positive-ish document") has little predictive value, since every document is
  a positive for exactly one query — but this has not been measured, and a
  document-disjoint split would settle it.
- **One dataset, one language mix, one base size.** LitRetrieval is literary
  Russian/English prose. Nothing here says how this transfers to code, chat logs
  or short-form web text.
- **Hard negatives come from whatever miner built the dataset.** "0.96 against
  hard negatives" is against *those* negatives.
- **850 training queries is small.**

---

## Model-level state API

The reranker is built on a general state API added to `RWKV7X070`. It is useful
on its own — streaming inference, prefix caching, state tuning:

```python
h, state = model.body(idx, return_state=True)     # RWKVState
h2, state2 = model.body(idx2, state=state, return_state=True)   # continue
state = model.states(idx, mask=mask, end_idx=end_idx)           # state only
```

`RWKVState` carries three things per layer: the WKV matrix, and the token-shift
inputs of `tmix` and `cmix`. The last two are easy to forget and produce a
continuation that differs from a straight pass on the first token of every
layer. `v_first` is deliberately **not** in the state — in x070 it is recomputed
per position by layer 0, not carried.

### Right padding is exact

Batching sequences of different lengths normally corrupts the final state,
which is why the original implementation left-pads. Here padded positions are
made **neutral** for the recurrence — `w ← 1`, `k ← 0`, `b ← 0`, so
`h_next = 1·h + v·0ᵀ + sa·0ᵀ = h`. The state freezes at the row's last real
token and does not depend on padding or on batch neighbours.

Pass `mask` (1 for real tokens) and `end_idx` (position of each row's last real
token) — `rwkv_metal.model.build_mask(lengths, T)` builds the mask.

Verified in `tests/test_wkv7_state.py`: a ragged padded batch reproduces
individual unpadded passes, and a split pass (document, then query from its
state) reproduces the full pass.

### The `RWKV7` token-shift leak (fixed)

Building this exposed a bug in the *other* architecture. The from-scratch
pretraining model (`rwkv_metal.model.RWKV7`) used to carry token-shift **between
blocks**: block `i+1` received block `i`'s last output as its "previous token".
That token is in the future relative to position 0, so during teacher-forced
training the model could see the end of the window while predicting the start —
and at inference no such token exists, so train and serve disagreed. It also
made "continue from a saved state" undefinable: the value needed at the boundary
depends on tokens that have not arrived.

Measured: changing only the last input token moved earlier hidden states, where
the same test on x070 gives exactly `0.0`.

**How much did it actually cost?** Less than the description suggests, and the
honest answer is worth spelling out. The perturbation enters at position 0 of
each block and walks forward one position per layer, so it touches roughly the
first `n_layer` positions of the window and nothing else. On a 6-layer model at
`ctx_len=512`, measured per position:

| position | 0 | 1 | 2 | 3 | 4 | 5 | 6+ |
|---|---|---|---|---|---|---|---|
| `max|Δh|` | 0.209 | 0.006 | 0.003 | 0.004 | 0.007 | 0.003 | < 0.002 |

Ten affected positions out of 511. So the effect on average loss is
negligible — a 400-step from-scratch A/B on 3 M tokens (same seed, same data)
gave val loss 5.73 legacy versus 5.77 fixed, and re-running the legacy
checkpoint through the *causal* forward changed its loss by less than 0.001.
The two runs are within single-run noise of each other.

That is a nice result to be able to state rather than guess at, and it also
shows where the leak *would* bite: short contexts and deep models, where
`n_layer / ctx_len` is not 2 % but a fifth.

The reason to fix it was never the loss. It was that train and serve disagreed,
and that "continue from a saved state" could not be defined. Both architectures
now zero-pad token-shift per block, both are causal, and both support the full
state API — `tests/test_wkv7_state.py` checks causality, continuation and ragged
batching for each. `RWKV7(cfg, legacy_token_shift=True)` (or
`PretrainConfig(legacy_token_shift=True)`) restores the old behaviour, which you
only want in order to resume a checkpoint trained before the fix; the state API
refuses to work in that mode.

---

## Practical advice

- **Two-stage or nothing.** A cross-encoder scores one pair at a time. Use the
  embedder to shortlist, then rerank the top 20–100. Reranking a corpus is not
  a thing you do.
- **Read the first loss value.** Zero-init means it must be exactly `ln(C)`.
- **Raise the learning rate.** `3e-5` is a fine-tuning default and is too low
  for a head trained from scratch on frozen features.
- **Truncate documents deliberately.** `max_doc_tokens` truncates the **tail**.
  LitRetrieval passages are ~530 tokens median, so `512` keeps almost all of
  them; smaller values buy encoding speed at a measurable cost.
- **Watch the candidate count.** More candidates make the listwise loss harder
  and the metric stricter. Comparing MRR across different `n_candidates` is
  meaningless — the random floor moves.
- **The index is per-instruction.** Change the instruction string and the cached
  prefixes are wrong. `score_indexed` catches this; a swapped base checkpoint it
  cannot catch.
- **Ablate on a cache, with seeds.** A single run's ±0.01–0.03 will happily
  "show" you an improvement that isn't there. `tools/ablate_reranker.py` runs
  variants over several seeds on an existing cache in about a minute each.

---

## Differences from EmbeddingRWKV

| | EmbeddingRWKV | here |
|---|---|---|
| Padding | left, padding runs through the recurrence | right, padded positions are exact no-ops |
| State read at | end of the padded row | last real token (`end_idx`) |
| Loss | pointwise BCE | listwise softmax (BCE available, mixable) |
| Training cost | full base forward every step | states cached once, head trained on the cache |
| Negatives | easy/medium/hard sampling per row | shared deduplicated document pool, so extra candidates cost only the query tail |
| Head init | last layer of the score MLP randomly initialised | zero-init, so the starting loss is exactly `ln(C)` |
| Probe tokens | 1 | `n_probe`, configurable |
| Head precision | inherits the base's bf16 | fp32 by default (`head_dtype`) |
| Checkpoint | weights only | weights + configuration + text contract, verified on load |

---

## Current limitations

- **Base is frozen, full stop.** There is no LoRA-on-the-base path for the
  reranker yet. It would break the state cache during training (the feature map
  would move every step), so it needs the online-encoding trainer that does not
  exist yet.
- **No multi-task heads.** One scalar score, one task. The instruction string is
  the only task conditioning.
- **The index is memory-hungry.** 2.4 MB per document at 0.1B, fp32. bf16 via
  `build_index(..., dtype=mx.bfloat16)` halves it; the WKV part stays fp32
  because continuation accuracy depends on it.
- **Base-forward dispatch overhead.** Encoding a batch carries ~10 ms of fixed
  cost at `B=1` (twelve layers of Python and kernel launches) before any real
  work, and unlike the head this path does **not** respond to `mx.compile`
  (10.6 → 9.9 ms/token). It is the floor on indexed scoring of short queries.
- **No cross-encoder distillation.** Training a reranker from a stronger
  teacher's scores is the standard way to get a good one; only the
  positive/negative signal from the dataset is used here.

See also: [`embedding.md`](./embedding.md) for the first-stage retriever,
[`inference.md`](./inference.md) for serving, [`lora.md`](./lora.md) for
fine-tuning the base itself.
