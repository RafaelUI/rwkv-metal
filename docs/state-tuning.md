# State-tuning and pluggable initial states

RWKV is an RNN: a sequence does not have to start from zeros. `rwkv-metal`
treats the starting state as a pluggable object — trained (state-tuning), folded
from a prompt, or one on top of the other — kept in its own small file.

Code: `rwkv_metal/model/initial_state.py`, `rwkv_metal/lora/state_tune.py`.
Gate: `tests/test_state_tuning.py` (`--mutations` runs its 22 mutation caps).

## One object

A trained state and a state after a prompt are the same thing, an `RWKVState`
with batch 1:

| tensor | shape | |
|---|---|---|
| `wkv` | `[L, 1, H, S, S]` | WKV recurrence matrix, always fp32 |
| `tmix_shift` | `[L, 1, 1, D]` | token-shift input of tmix |
| `cmix_shift` | `[L, 1, 1, D]` | token-shift input of cmix |

State-tuning trains only `wkv` by default (as RWKV-PEFT does); the shifts stay
zero. A folded prompt fills all three. The file always holds all three, so a
consumer never needs to know where the state came from.

| model | `wkv` numbers | both shifts |
|---|---|---|
| 0.1B (12 layers, D 768) | 589,824 | 18,432 |
| 0.4B (24, 1024) | 1,572,864 | 49,152 |
| 1.5B (24, 2048) | 3,145,728 | 98,304 |

## Attaching

```python
from rwkv_metal import attach_state, detach_state, fold_prompt, load_state

state, meta = load_state("task.state.safetensors", model)   # checks the base
attach_state(model, state)

model.body(idx)                    # starts from the attached state
cached = fold_prompt(model, ids)   # attached -> run the prompt -> cache
model.body(idx, state=cached.repeat(B).astype(dtype))   # explicit state wins
detach_state(model)
```

`attach_state` puts a `state0` submodule into every block. A block takes `h_in`
and the shifts from it only where the caller passed none, so "trained -> system
prompt -> cache" is ordinary continuation and has no code of its own. The
`[1, ...]` state is broadcast to the batch inside the block. An attached zero
state changes nothing: the output is bit-for-bit the same.

The embedder and the reranker use the same call on their base
(`attach_state(embedding_model.base, state)`): a trained state replaces the
`Instruct: ...` line of the task and costs no tokens per text.

## Training

```python
from rwkv_metal.lora import StateTuneConfig, state_tune

res = state_tune(model, batches,                       # (x, y) token ids [B, T]
                 StateTuneConfig(max_steps=500, state_path="task.state.safetensors"))
```

The whole base is frozen. `state0.wkv` is a parameter of its block, so the step
runs backward block by block (`lora/sequential.py`) with one block in memory,
and the gradient is bit-for-bit the one `nn.value_and_grad` gives over the
whole graph. The state enters every layer from layer 0 up, so the gradient goes
through the full depth: a step costs about as much as adapters on all layers.

- `train=("wkv", "tmix_shift", "cmix_shift")` also trains the shifts.
- `init=` starts from a given state, e.g. a folded prompt.
- Learning rate: the default (3e-4 -> 3e-5, cosine) is measured, not borrowed.
  The 1 -> 0.01 schedule usually quoted for state-tuning destroys the model
  here: g1d 0.1B on Russian prose, 100 steps of 4 x 256, held-out loss 3.228
  without a state -- 3.374 at lr 1e-2, 3.307 at 3e-3, 3.197 at 1e-3, 3.176 at
  3e-4. Adam moves every one of the state's numbers by about `lr` per step.
- Works over a dense base and over `.rwkvq` (`load_rwkvq_model(path, rank=0)`).

For embedding training call `prepare_state_tuning(model.base)` and run
`finetune_embedding` as usual: the trainable set is then the state and the
pooling head, and `embedding/sequential.py` takes it block by block. Save the
state with `save_state(path, attached_state(model.base), model.base)`.

### What it buys (measured)

Language modelling, Russian prose (held-out windows of 256 tokens, 300 steps of
4 x 256, lr 3e-4, three seeds; paired bootstrap over windows):

| base | loss without state | with state | change |
|---|---|---|---|
| 0.1B dense (g1d) | 3.221 | 3.155 | −0.066 [−0.073; −0.060] |
| 0.1B `reduction` | 3.224 | 3.157 | −0.066 [−0.073; −0.060] |
| 0.1B `compression` | 3.251 | 3.179 | −0.072 [−0.078; −0.065] |
| 0.4B dense | 2.946 | 2.895 | −0.051 [−0.059; −0.043] |
| 0.4B `reduction` | 2.947 | 2.897 | −0.050 [−0.058; −0.042] |
| 0.4B `compression` | 2.965 | 2.925 | −0.040 [−0.056; −0.016] |

- A compressed base gains as much as the dense one.
- **A state moves between bases almost for free.** Trained on one of the three
  0.1B bases and attached to another (`allow_other_base=True`), it is
  0.000…0.004 worse than a state trained on the target itself, i.e. keeps
  94…100% of the gain (0.4B: −0.008…+0.012). The refusal on a different base is therefore stricter
  than quality requires between a model and its own quantized files; it is
  kept so that a state is never silently used on an unrelated model.
- **The learning rate matters more than anything else.** The 1 -> 0.01 schedule
  often quoted for state-tuning destroys the model here (held-out loss 3.228 ->
  3.374 at 1e-2 after 100 steps); 3e-4 is the measured default.

**As an embedder, from a language-model checkpoint** (state and pooling head
trained, 0.1B, 200 steps, MRR over 300 queries with a pool of 600): 0.42…0.74
on eight runs of nine, comparable to training the top 2 layers (0.46) with 24
times fewer trainable numbers, well below the top 4 or 6 layers (0.76 / 0.78).
It is unstable: one run of nine diverged (MRR 0.02, as before training), and
the useful step size is narrow (state lr 1e-3: 0.63; 3e-4: 0.25; 1e-2 and
above: nothing learnt). Prefer top-layer training for a new embedder; a state is
better suited to adapting one that already works.

**In place of the instruction line**, on top of a trained embedder (0.1B, top 4
layers trained with `Instruct: … Query: …` queries; then the instruction — 17.8
tokens per query — is removed and only the state is trained, lr 1e-4; control:
the same steps training the head instead; three seeds, paired bootstrap):

| base | with instruction | without | without + state | without + head (control) | state − with instruction |
|---|---|---|---|---|---|
| dense | 0.759 | 0.752 | 0.758 | 0.754 | −0.001 [−0.012; +0.010] |
| `reduction` | 0.757 | 0.748 | 0.756 | 0.750 | −0.000 [−0.012; +0.011] |

The state brings quality back to the level with the instruction while every
query gets 17.8 tokens shorter. The loss from dropping the instruction was small
to begin with (−0.007…−0.009): this data has one instruction for all queries,
and the embedder barely relies on it. The state beats the control by
+0.004…+0.007, at the edge of significance. The state was attached to the
model, so it applied to documents too, not only to queries. With several or
long instructions the gain may be larger; neither that nor a query-only state
was measured.

## The file

A separate safetensors file: the three tensors in fp32 and metadata —
`format` (`rwkv-state`), `version`, the contract (`n_layer`, `n_embd`, `n_head`,
`head_size`), `kind` (`tuned`, `prompt`, `tuned+prompt`), `trained`, `note`, and
`base_fingerprint`.

A state means something only on the base it was made on; on another base it
gives a plausible but different result. `load_state(path, model)` therefore

- always refuses a different architecture (contract mismatch);
- raises `StateBaseMismatch` on a different base, unless
  `allow_other_base=True`.

`base_fingerprint(model)` is a sha256 over the contract, the small tensors of
the base (norms, token-shift mixes, `k_k` / `k_a` / `r_k`) and row 0 of the dense
weight of two projections (first and last), all cast to bf16 (rows of compressed `.rwkvq`
projections go through fp16, the lattice they live on). The same base
loaded as bf16 and as fp32 gives one fingerprint. A dense model, its
`reduction` file and its `compression` file are three different bases. It is a
fingerprint of the base, not of adapters on top of it. SwiftRWKV computes the
same value, so files move between the two libraries.
