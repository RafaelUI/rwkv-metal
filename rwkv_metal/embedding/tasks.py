"""
Per-task compute_loss(model, batch) glue: ties EmbeddingModel.embed(...) to
the pure tensor losses in `rwkv_metal.embedding.loss`, matching the batch
shapes produced by `rwkv_metal.embedding.dataset`'s batchers.

These are what you pass as `compute_loss` to
`rwkv_metal.embedding.train.finetune_embedding`. Each task gets its own
compute_loss/batcher pair; curriculum training just means running
`finetune_embedding` once per task, sequentially, over the same model.
"""
import mlx.core as mx

from .loss import info_nce_loss, triplet_pool_loss, zero_shot_classification_loss
from .gradcache import GradCacheSpec, split_triplet_batch, embed_triplet_chunk


def pair_loss(model, batch, temperature: float = 0.05):
    """batch: (q_idx,q_pool, d_idx,d_pool) from PairBatcher -- in-batch
    negatives only, no explicit hard negative. Use triplet_pool_loss +
    TripletBatcher instead when hard negatives are available (LitRetrieval)."""
    q_idx, q_pool, d_idx, d_pool = batch
    q = model.embed(q_idx, q_pool)
    d = model.embed(d_idx, d_pool)
    return info_nce_loss(q, d, temperature)


def retrieval_loss(model, batch, temperature: float = 0.05):
    """batch: (a_idx,a_pool, p_idx,p_pool, n_idx,n_pool) from TripletBatcher.
    Asymmetric: a query and the passage that answers it aren't a symmetric
    pair, so we only train anchor->candidate, not candidate->anchor."""
    a_idx, a_pool, p_idx, p_pool, n_idx, n_pool = batch
    a = model.embed(a_idx, a_pool)
    p = model.embed(p_idx, p_pool)
    n = model.embed(n_idx, n_pool)
    return triplet_pool_loss(a, p, n, temperature, symmetric=False)


def sts_loss(model, batch, temperature: float = 0.05):
    """batch: same shape as retrieval_loss. Symmetric: anchor and positive
    are both "documents" in a genuinely symmetric similarity relationship."""
    a_idx, a_pool, p_idx, p_pool, n_idx, n_pool = batch
    a = model.embed(a_idx, a_pool)
    p = model.embed(p_idx, p_pool)
    n = model.embed(n_idx, n_pool)
    return triplet_pool_loss(a, p, n, temperature, symmetric=True)


# ── GradCache specs (retrieval / sts) ────────────────────────────────────
# Same math as the eager retrieval_loss / sts_loss above, but expressed as
# (split, embed, loss-on-embeddings) so `finetune_embedding` can run them
# chunk-wise with the full batch still visible to the loss. Classification
# has no spec: its candidate pool is per-row (not a shared in-batch negative
# pool), so a bigger batch doesn't buy it more negatives -- GradCache's whole
# reason for existing doesn't apply there.

def _triplet_loss_from_embeddings(fields, temperature, symmetric):
    a, p, n = fields
    return triplet_pool_loss(a, p, n, temperature, symmetric=symmetric)


RETRIEVAL_GC = GradCacheSpec(
    split=split_triplet_batch,
    embed=embed_triplet_chunk,
    loss=lambda fields, temperature: _triplet_loss_from_embeddings(fields, temperature, False),
)

STS_GC = GradCacheSpec(
    split=split_triplet_batch,
    embed=embed_triplet_chunk,
    loss=lambda fields, temperature: _triplet_loss_from_embeddings(fields, temperature, True),
)


def classification_loss(model, batch, temperature: float = 0.05):
    """batch: (a_idx,a_pool, cand_idx,cand_pool, mask, target_idx) from
    ClassificationBatcher. Zero-shot: the label pool is embedded like any
    other text and compared by cosine similarity -- no learned classifier
    head, no fixed global label set (each row carries its own pool)."""
    a_idx, a_pool, cand_idx, cand_pool, mask, target_idx = batch
    anchor = model.embed(a_idx, a_pool)                       # [B, D]
    B, K, T = cand_idx.shape
    flat = model.embed(cand_idx.reshape(B * K, T), cand_pool.reshape(B * K))
    candidates = flat.reshape(B, K, -1)                        # [B, K, D]
    return zero_shot_classification_loss(anchor, candidates, mask, target_idx, temperature)
