# Changelog

## 0.3.1

A dtype leak in the residual stream, found while porting to Swift and measured
on both sides.

### Added

- **`CAST_WKV_OUTPUT`** (`rwkv_metal.model.rwkv7_x070`, module flag, **off by
  default**). The WKV kernel computes its recurrence in float32 — it has to —
  but it also *returns* float32 into the residual stream, from layer 0 onward.
  Weights are bfloat16, so every matmul below that point promotes the whole
  weight matrix to float32. The flag casts the kernel's **output** back to the
  block's input dtype; `h_out` stays float32, so the recurrence is untouched.

  Measured on 0.1B, A/B interleaved in one process, swap unmoved: decode
  **15.81 → 5.69 ms/token (2.78×)**, perplexity **18.2512 → 18.2691
  (+0.098%)** on a mixed-domain Russian/Serbian/English corpus.

  On 1.5B (24 layers) the perplexity cost is **+0.018%** — it *falls* with
  depth rather than rising, which is the opposite of what was expected. The
  1.5B timing is not quoted: 3 GB of weights on a 16 GB machine put the run
  into swap, and the spread (190–216 ms) makes the number meaningless.

  Off by default because this repository is the parity reference for
  [SwiftRWKV](https://github.com/RafaelUI/SwiftRWKV), where the same flag
  exists and is also off. Turning it on is a decision to move the numbers on
  both sides at once.

- **`tests/dev_wkv_cast_ab.py`** — the A/B harness for the flag, with
  `speed | ppl | both` modes. The modes are separate on purpose: the
  perplexity pass holds `[T, 65536]` float32 logits (134 MB per chunk at
  T=512) and pushes a 1.5B run into swap. Swap does not affect perplexity —
  it is a deterministic number — but it invalidates any timing taken in the
  same process.

- **`NEXT_SESSION.md`** — working notes on the above: what is measured, under
  what conditions, and what is left.

## 0.3.0

Recurrent state becomes a first-class object, and a reranker is built on top of
it.

### Added

- **`rwkv_metal.reranker`** — a cross-encoder that scores `(query, document)`
  pairs by reading the base model's recurrent state instead of per-token
  activations. Frozen base, small trainable head (one or more RWKV blocks over a
  learnable probe token), listwise or pointwise loss, held-out evaluation.
  Measured on a 0.1B World base and LitRetrieval: MRR 0.222 (random) → 0.491
  (raw embedder on the same base) → 0.975 (reranker); against *mined* hard
  negatives, pairwise accuracy 0.493 → 0.953. See
  [`docs/reranker.md`](docs/reranker.md).
- **State API on the model.** `body(idx, state=…, mask=…, end_idx=…,
  return_state=True)` and `states(...)` on both architectures, with `RWKVState`
  carrying the WKV matrix plus the token-shift inputs of `tmix` and `cmix`.
  Enables streaming decode at constant cost per token, prefix caching, and
  ragged batches.
- **Exact right-padding.** Padded positions are made neutral for the recurrence
  (`w←1, k←0, b←0`), so a row's state freezes at its last real token and does not
  depend on padding or on batch neighbours. Previously this required left
  padding and accepted the contamination.
- **Kernel: stateful entry points.** `wkv7_train_with_state(..., h_in)`
  (differentiable through `h_in`) and `wkv7_step(...)` for a single token in
  plain MLX ops — the checkpoint kernel needs `T` divisible by 16, so one token
  would otherwise cost sixteen.
- **Document index for reranking.** `build_index` / `score_indexed` precompute
  the `Instruct + Document` prefix state; each later query then costs only its
  own tokens — 73 ms → 3.5 ms per pair on 0.1B.
- **State cache for training.** With a frozen base, pair encoding is a fixed
  function, so it runs once (`encode_pairs`) and training reads from a
  memory-mapped fp16 cache. An epoch over tens of thousands of pairs takes
  seconds, which makes multi-seed ablations practical.
- **Tools**: `tools/run_reranker.py` (full run + baselines + report),
  `tools/ablate_reranker.py` (multi-seed ablations on an existing cache),
  `tools/bench_reranker.py` (speed and memory, measured via system RSS and swap
  counters), `tools/test_reranker_smoke.py`, `tests/test_wkv7_state.py`.

### Fixed

- **Future-information leak in `RWKV7`** (the from-scratch pretraining
  architecture). It carried token-shift *between* blocks, so block `i+1`
  received block `i`'s last output — a future token relative to position 0.
  The perturbation moves forward one position per layer, so it touched roughly
  the first `n_layer` positions of the window (measured on 6 layers at ctx 512:
  `max|Δh|` 0.209 at position 0, under 0.007 after, nothing past position 9); a
  400-step from-scratch A/B came out within single-run noise. It mattered
  because training and inference disagreed and because continuation from a saved
  state could not be defined. Both architectures now zero-pad token-shift per
  block and both are causal. `RWKV7(cfg, legacy_token_shift=True)` /
  `PretrainConfig(legacy_token_shift=True)` reproduces the old behaviour for
  checkpoints trained before the fix.
- **Reranker head no longer trains in bf16.** It inherited the base's dtype when
  initialised from it; now fp32 by default (`head_dtype`).

### Changed

- `mx.compile` on the reranker head: training step 49.1 → 34.5 ms, head forward
  on 256 pairs 13.4 → 7.1 ms. Not applied to the base forward, which is not
  dispatch-bound (10.6 → 9.9 ms per token).
- Ranking metrics use average tie-breaking, so an untrained head reports
  `2/(C+1)` — random guessing — instead of a perfect 1.0.

### Notes

- Reranker checkpoints store their configuration and text contract (layers,
  probe count, template order, truncation, instruction) in safetensors metadata,
  and `load_head` refuses a mismatch. A head trained on layer 5 and one trained
  on layer 11 have identical tensor shapes, so this would otherwise fail
  silently.
- Measure memory with the system, not `mx.get_peak_memory()`: the latter reports
  the MLX pool only. On the reference run it showed 2.29 GB against an actual
  process RSS of 3.98 GB.

## 0.2.0 and earlier

See the git history.
