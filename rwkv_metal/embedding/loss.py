"""Contrastive losses for embedding training.

Pure tensor math only (no model/tokenizer knowledge) -- see
`rwkv_metal.embedding.tasks` for the per-task glue that calls these with
`EmbeddingModel.embed(...)` outputs.
"""
import mlx.core as mx
import mlx.nn as nn


def info_nce_loss(q: mx.array, d: mx.array, temperature: float = 0.05) -> mx.array:
    """q, d: [B, D], L2-normalized, positives aligned on the diagonal
    (q[i] <-> d[i]). Off-diagonal batch pairs serve as negatives, both
    directions. Kept for backwards compat / cases with no explicit negatives."""
    logits = (q.astype(mx.float32) @ d.astype(mx.float32).T) / temperature  # [B, B]
    labels = mx.arange(q.shape[0])
    loss_q2d = nn.losses.cross_entropy(logits, labels).mean()
    loss_d2q = nn.losses.cross_entropy(logits.T, labels).mean()
    return (loss_q2d + loss_d2q) / 2.0


def triplet_pool_loss(anchor: mx.array, positive: mx.array, negative: mx.array,
                       temperature: float = 0.05, symmetric: bool = False) -> mx.array:
    """anchor, positive, negative: [B, D], L2-normalized.

    Negative pool = every batch's positive AND every batch's explicit hard
    negative (2B candidates total), not just the one paired hard negative --
    this is the "negative pool" piece: each anchor is contrasted against its
    own hard negative *and* every other sample's positive/negative in the
    same step, for free.

    symmetric=False (retrieval): anchor->candidate only. A query and the
    document that answers it don't play a symmetric role, so we don't also
    train document->query.
    symmetric=True (sts): both directions -- anchor and positive are both
    "documents" in a genuinely symmetric semantic-similarity relationship.
    """
    a = anchor.astype(mx.float32)
    p = positive.astype(mx.float32)
    n = negative.astype(mx.float32)
    candidates = mx.concatenate([p, n], axis=0)              # [2B, D]
    logits = (a @ candidates.T) / temperature                 # [B, 2B]
    labels = mx.arange(a.shape[0])                            # positive i is at column i
    loss = nn.losses.cross_entropy(logits, labels).mean()
    if not symmetric:
        return loss
    # doc->anchor direction: only positives can "retrieve" back (negatives
    # have no defined anchor to point to), so this half uses just the [B,B]
    # positive block, transposed.
    logits_back = (p @ a.T) / temperature                     # [B, B]
    loss_back = nn.losses.cross_entropy(logits_back, labels).mean()
    return (loss + loss_back) / 2.0


def zero_shot_classification_loss(anchor: mx.array, candidates: mx.array,
                                   mask: mx.array, target_idx: mx.array,
                                   temperature: float = 0.05) -> mx.array:
    """anchor: [B, D]. candidates: [B, K, D] (per-sample candidate/label
    pool -- label vocab differs per row in this dataset, so it's padded to
    the batch's max K with `mask` [B, K] marking real (1.0) vs pad (0.0)
    slots). target_idx: [B] index of the correct label within each row's
    candidate list.

    This is zero-shot classification, not a learned classifier head: the
    label set is just embedded like any other text and compared by cosine
    similarity, exactly like retrieval query-vs-document.
    """
    a = anchor.astype(mx.float32)[:, None, :]                 # [B, 1, D]
    c = candidates.astype(mx.float32)                         # [B, K, D]
    logits = (a * c).sum(axis=-1) / temperature                # [B, K]
    logits = mx.where(mask > 0, logits, mx.full(logits.shape, -1e9, dtype=logits.dtype))
    return nn.losses.cross_entropy(logits, target_idx).mean()
