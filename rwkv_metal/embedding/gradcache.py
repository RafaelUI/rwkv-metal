"""
rwkv_metal.embedding.gradcache
==============================
GradCache (Gao et al. 2021, "Scaling Deep Contrastive Learning Batch Size
under Memory Limited Setup") for MLX -- decouples the contrastive negative
pool size from activation memory.

Why it matters here: `triplet_pool_loss`'s negative pool IS the batch, so
retrieval quality scales with batch size -- but activation memory scales with
batch size too, and RWKV-7 activations over long LitRetrieval passages
(sts/classification anchors run to ~9k chars) hit unified memory fast. With
GradCache, activation memory scales with the CHUNK size while the loss still
sees the whole batch, so a 16GB Mac can train with a negative pool that would
otherwise need many times that.

Algorithm (3 phases, mathematically exact -- not an approximation):
  1. Forward every chunk with NO grad; cache the resulting embeddings and
     `mx.eval` them so no autograd graph is retained. Memory: [N, D] vectors
     only, no activations.
  2. Compute the loss on the FULL cached embedding matrix and take its
     gradient w.r.t. those embeddings (dL/dE). This is where the whole batch
     interacts -- an [N, N] similarity matmul + softmax, fully vectorized on
     the GPU, no Python-level looping.
  3. Re-forward each chunk WITH grad, seeding the backward pass with that
     chunk's slice of dL/dE, and accumulate parameter gradients.

Phase 3 uses the surrogate identity
        d/dtheta  sum(embed(chunk) * stop_grad(dL/dE_chunk))  ==  VJP
i.e. the parameter gradient of that scalar equals the vector-Jacobian product
seeded with dL/dE_chunk -- exactly the term GradCache needs. Going through
`nn.value_and_grad` (rather than raw `mx.vjp`, which only accepts flat array
lists) keeps this freeze()-aware, so GradCache composes with full-FT, frozen
layers, and LoRA/QLoRA identically to the normal path.

Verified on the 0.1B checkpoint (tools/test_gradcache.py,
tools/test_gradcache_accum_baseline.py), fp32, batch 8:

    chunk = batch (1 chunk):  loss diff 0.0 exactly,
                              grad rel diff 1.5e-8 == the eager-vs-eager
                              run-to-run noise floor
    chunk = 2 (4 chunks):     loss diff 2.4e-7, grad rel diff 1.3e-4
    grad accumulation, same chunking: grad rel diff 3.9 (i.e. ~400%)

The 1-chunk case matching the noise floor *exactly* is the decisive result:
cutting the graph at E and re-seeding introduces no error at all. The 1.3e-4
at 4 chunks is therefore purely floating-point summation ORDER -- a shared
parameter's gradient is summed as 4 groups of 2 instead of one group of 8.
That is inherent to any chunked accumulation, not to this implementation.
The grad-accumulation row is the contrast that matters: plain accumulation
deviates by ~400% because it genuinely changes the contrastive math (each
micro-batch sees only its own negatives), which is exactly the problem
GradCache exists to solve.

Measured memory, bf16, 800-char passages, chunk=4 (same script):

    batch    eager     gradcache
        8   3.43 GB     3.15 GB
       16   4.50 GB     3.17 GB
       32   7.00 GB     3.20 GB
       48   9.68 GB     3.20 GB

Eager grows linearly with the negative pool; GradCache stays flat, at a cost
of ~30% wall-clock (the extra no-grad forward, partly offset by cheaper
backward chunks). Losses agreed to 4 decimals at every batch size.

On Metal / performance: GradCache is a memory-SCHEDULING algorithm, not a
compute kernel -- there's no elementwise or reduction bottleneck here to fuse
into a custom Metal kernel. The per-chunk loop is the algorithm itself (that
loop is *what* bounds activation memory), and it runs only a handful of
iterations per step, each dispatching a full model forward/backward. All the
actual arithmetic already runs on the GPU: the model's WKV-7 recurrence via
the project's existing hand-written Metal kernel, and phase 2's [N, N]
similarity/softmax via MLX's own GPU ops. Python here only sequences a dozen
GPU dispatches; its overhead is far below the noise floor of a single
forward. A custom kernel would add no speed, only a second correctness
surface to maintain.

REQUIREMENT: `embed_chunk` must be deterministic -- phases 1 and 3 must
produce identical embeddings for the same input. Our model has no dropout by
default, so this holds; if you add dropout, GradCache needs a fixed RNG per
chunk or the cached dL/dE will be seeded against a different forward.
"""
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map


@dataclass
class GradCacheSpec:
    """How to run a given task under GradCache.

    split: (batch, chunk_size) -> list of chunks
    embed: (model, chunk) -> tuple of [b, D] embedding fields
    loss:  (tuple of full [N, D] fields, temperature) -> scalar loss
    """
    split: Callable
    embed: Callable
    loss: Callable


def _tree_add(a, b):
    return tree_map(lambda x, y: x + y, a, b)


def gradcache_value_and_grad(
    model,
    chunks: Sequence,
    embed_chunk: Callable,
    loss_from_embeddings: Callable,
) -> Tuple[mx.array, dict]:
    """Exact loss + parameter gradients with activation memory bounded by the
    largest single chunk instead of the whole batch.

    Args:
        model: EmbeddingModel (or anything nn.value_and_grad accepts).
        chunks: sequence of per-chunk inputs; each is passed to embed_chunk.
        embed_chunk: (model, chunk) -> tuple of [b_i, D] embedding arrays.
            The tuple is the set of "fields" a chunk produces -- e.g.
            (anchor, positive, negative) for triplet tasks. Every chunk must
            return the same number of fields, and field f's rows must be a
            consistent per-chunk slice (they're concatenated along axis 0).
        loss_from_embeddings: tuple of full [N, D] field matrices -> scalar
            loss. This sees the WHOLE batch -- that's the point.

    Returns:
        (loss, grads) with the same semantics as nn.value_and_grad(...)(...),
        so callers can drop this in without changing clipping/optimizer code.
    """
    if len(chunks) == 0:
        raise ValueError("gradcache_value_and_grad: no chunks given")

    # ── Phase 1: no-grad forward, cache embeddings ────────────────────────
    # Calling the model outside any grad transform builds no autograd graph;
    # mx.eval materializes the vectors so the lazy graph (and the activations
    # behind it) is released before the next chunk is forwarded.
    cached: List[Tuple[mx.array, ...]] = []
    for chunk in chunks:
        embs = embed_chunk(model, chunk)
        if isinstance(embs, mx.array):
            embs = (embs,)
        embs = tuple(mx.stop_gradient(e) for e in embs)
        mx.eval(embs)
        cached.append(embs)

    n_fields = len(cached[0])
    for c in cached:
        if len(c) != n_fields:
            raise ValueError("embed_chunk returned inconsistent field counts across chunks")

    full = tuple(
        mx.concatenate([c[f] for c in cached], axis=0) for f in range(n_fields)
    )
    mx.eval(full)

    # ── Phase 2: loss + dL/dE on the FULL batch ───────────────────────────
    # Differentiating w.r.t. the embedding matrices (not parameters) -- cheap:
    # [N, D] in, [N, N] similarity inside, no model activations involved.
    def _loss_wrt_embeddings(*fields):
        return loss_from_embeddings(fields)

    argnums = tuple(range(n_fields))
    loss, d_full = mx.value_and_grad(_loss_wrt_embeddings, argnums=argnums)(*full)
    mx.eval(loss, d_full)
    if isinstance(d_full, mx.array):
        d_full = (d_full,)

    # Slice dL/dE back into per-chunk cotangents (row offsets per field).
    offsets = [0] * n_fields
    cotangents: List[Tuple[mx.array, ...]] = []
    for c in cached:
        cots = []
        for f in range(n_fields):
            rows = c[f].shape[0]
            cots.append(d_full[f][offsets[f]: offsets[f] + rows])
            offsets[f] += rows
        cotangents.append(tuple(cots))

    # ── Phase 3: per-chunk surrogate backward, accumulate parameter grads ──
    def surrogate(m, chunk, cots):
        embs = embed_chunk(m, chunk)
        if isinstance(embs, mx.array):
            embs = (embs,)
        total = None
        for e, c in zip(embs, cots):
            term = (e * mx.stop_gradient(c)).sum()
            total = term if total is None else total + term
        return total

    grad_fn = nn.value_and_grad(model, surrogate)

    total_grads = None
    for chunk, cots in zip(chunks, cotangents):
        _, g = grad_fn(model, chunk, cots)
        mx.eval(g)
        total_grads = g if total_grads is None else _tree_add(total_grads, g)
        mx.eval(total_grads)

    return loss, total_grads


# ── Chunking helpers ─────────────────────────────────────────────────────


def split_triplet_batch(batch, chunk_size: int) -> List[Tuple]:
    """(a_idx,a_pool, p_idx,p_pool, n_idx,n_pool) -> list of same-shaped
    tuples with at most `chunk_size` rows each (retrieval / sts batches)."""
    a_idx, a_pool, p_idx, p_pool, n_idx, n_pool = batch
    B = a_idx.shape[0]
    out = []
    for s in range(0, B, chunk_size):
        e = min(s + chunk_size, B)
        out.append((a_idx[s:e], a_pool[s:e],
                    p_idx[s:e], p_pool[s:e],
                    n_idx[s:e], n_pool[s:e]))
    return out


def embed_triplet_chunk(model, chunk):
    """(anchor, positive, negative) embeddings for one triplet chunk."""
    a_idx, a_pool, p_idx, p_pool, n_idx, n_pool = chunk
    return (model.embed(a_idx, a_pool),
            model.embed(p_idx, p_pool),
            model.embed(n_idx, n_pool))
