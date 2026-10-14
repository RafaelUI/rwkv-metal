"""
rwkv_metal.embedding.eval
=========================
Small, dependency-free (mlx + stdlib only) held-out evaluation for the three
LitRetrieval task types -- not an MTEB harness, just enough signal to tell
whether a curriculum stage actually helped, instead of judging purely by the
training loss on data the model has already seen.

  - retrieval / sts: rank the true positive against a pool made of every
    eval row's positive + every eval row's explicit hard negative -- MRR,
    Recall@k, nDCG@10. Note: LitRetrieval only has binary positive/negative
    labels, not graded similarity scores, so the "sts" eval here is a
    pairwise ranking check (does cos(anchor,positive) beat
    cos(anchor,negative)?), not a classic Spearman-correlation STS eval --
    that needs graded human-similarity scores, which this dataset doesn't
    provide.
  - classification: zero-shot top-1 accuracy over each row's own candidate
    label pool (parsed from the instruction text, same as training).

Usage:
    from rwkv_metal.embedding.eval import evaluate_retrieval, evaluate_sts_pairwise, evaluate_classification
    evaluate_retrieval(model, tok, held_out_retrieval_rows)
"""
import math
import random
from typing import Dict, List, Optional, Sequence

import mlx.core as mx

from .dataset import encode_batch, parse_classification_candidates


# Токенов на пачку оценки (строки × длина самой длинной) -- порт SwiftRWKV
# `EmbeddingMetrics.evaluationBatchTokens`.
EVALUATION_BATCH_TOKENS = 4096

# Пачек оценки, прошедших тело по блокам (для тестов).
BATCHES = 0


def _embed_texts(model, tokenizer, texts: Sequence[str], batch_size: int = 16,
                  terminator: int = 0, batch_tokens: Optional[int] = None) -> mx.array:
    """Encode a flat list of texts to L2-normalized [N, D] vectors.

    A batch is filled by rows (`batch_size`) AND by tokens (`batch_tokens`,
    rows × longest row; default `EVALUATION_BATCH_TOKENS`, 0 -- rows only), and
    the body runs block by block, evaluated after each block: the same operations
    as `model.embed`, but one lazy graph over the whole body held every layer's
    activations at once. Port of SwiftRWKV `EmbeddingMetrics.embedAll`
    (32 rows of 1024 tokens, peak MLX: 0.1B 1.85 -> 0.84 GB). Models without
    `base` / `embed_from_top` (anything not an `EmbeddingModel`) go through
    `model.embed` as before."""
    global BATCHES
    from ..lora.sequential import forward_trace
    budget = EVALUATION_BATCH_TOKENS if batch_tokens is None else batch_tokens
    by_blocks = hasattr(model, "embed_from_top") and hasattr(model, "base")
    lengths = [len(tokenizer.encode(t)) + 1 for t in texts]
    vecs = []
    i = 0
    while i < len(texts):
        end, longest = i + 1, lengths[i]
        while end < len(texts) and end - i < batch_size:
            grown = max(longest, lengths[end])
            if budget and budget > 0 and (end - i + 1) * grown > budget:
                break
            longest, end = grown, end + 1
        idx, pool = encode_batch(tokenizer, texts[i:end], terminator)
        if by_blocks:
            top, _, _ = forward_trace(model.base, idx, None)
            v = model.embed_from_top(top, pool)
            BATCHES += 1
        else:
            v = model.embed(idx, pool)
        mx.eval(v)                      # иначе ленивый граф удержит активации
        vecs.append(v)
        i = end
    out = mx.concatenate(vecs, axis=0)
    mx.eval(out)
    return out


def _ranking_metrics(sims: List[List[float]], correct_idx: List[int],
                      ks: Sequence[int]) -> Dict[str, float]:
    n = len(sims)
    mrr = 0.0
    recall = {k: 0 for k in ks}
    ndcg = 0.0
    for i in range(n):
        row = sims[i]
        target = row[correct_idx[i]]
        # 1-indexed rank of the true positive among all candidates
        rank = 1 + sum(1 for s in row if s > target)
        mrr += 1.0 / rank
        for k in ks:
            if rank <= k:
                recall[k] += 1
        if rank <= 10:
            ndcg += 1.0 / math.log2(rank + 1)
    return {
        "mrr": mrr / n,
        **{f"recall@{k}": recall[k] / n for k in ks},
        "ndcg@10": ndcg / n,
        "n": n,
    }


def evaluate_retrieval(model, tokenizer, rows: Sequence[Dict], max_chars: int = 800,
                        pool_size: Optional[int] = None, ks=(1, 5, 10),
                        seed: int = 0, terminator: int = 0) -> Dict[str, float]:
    """rows: {anchor, positive, negative} dicts (retrieval task rows).
    Candidate pool = every row's positive + every row's negative (2 * len(rows)
    candidates); the correct answer for row i is its own positive."""
    rng = random.Random(seed)
    rows = list(rows)
    if pool_size is not None and len(rows) > pool_size:
        rows = rng.sample(rows, pool_size)

    anchors = [r["anchor"][:max_chars] for r in rows]
    positives = [r["positive"][:max_chars] for r in rows]
    negatives = [r["negative"][:max_chars] for r in rows]
    candidates = positives + negatives
    correct_idx = list(range(len(rows)))  # candidates[i] is row i's positive

    a_vecs = _embed_texts(model, tokenizer, anchors, terminator=terminator)
    c_vecs = _embed_texts(model, tokenizer, candidates, terminator=terminator)
    sims = (a_vecs @ c_vecs.T).tolist()

    return _ranking_metrics(sims, correct_idx, ks)


def evaluate_sts_pairwise(model, tokenizer, rows: Sequence[Dict], max_chars: int = 800,
                           sample_size: Optional[int] = None, seed: int = 0,
                           terminator: int = 0) -> Dict[str, float]:
    """Pairwise ranking accuracy: does cos(anchor,positive) > cos(anchor,negative)?
    NOT a Spearman-correlation STS eval -- see module docstring."""
    rng = random.Random(seed)
    rows = list(rows)
    if sample_size is not None and len(rows) > sample_size:
        rows = rng.sample(rows, sample_size)

    anchors = [r["anchor"][:max_chars] for r in rows]
    positives = [r["positive"][:max_chars] for r in rows]
    negatives = [r["negative"][:max_chars] for r in rows]

    a = _embed_texts(model, tokenizer, anchors, terminator=terminator)
    p = _embed_texts(model, tokenizer, positives, terminator=terminator)
    n = _embed_texts(model, tokenizer, negatives, terminator=terminator)

    sim_pos = (a * p).sum(axis=-1)
    sim_neg = (a * n).sum(axis=-1)
    correct = (sim_pos > sim_neg).astype(mx.float32)
    mx.eval(correct, sim_pos, sim_neg)

    return {
        "pairwise_accuracy": float(correct.mean().item()),
        "mean_sim_pos": float(sim_pos.mean().item()),
        "mean_sim_neg": float(sim_neg.mean().item()),
        "n": len(rows),
    }


def evaluate_classification(model, tokenizer, rows: Sequence[Dict], max_chars: int = 800,
                             sample_size: Optional[int] = None, seed: int = 0,
                             terminator: int = 0) -> Dict[str, float]:
    """Zero-shot top-1 accuracy over each row's own candidate label pool."""
    rng = random.Random(seed)
    prepared = []
    for r in rows:
        cands = parse_classification_candidates(r["anchor"])
        if cands is None or r["positive"] not in cands:
            continue
        prepared.append((r["anchor"][:max_chars], cands, cands.index(r["positive"])))
    if sample_size is not None and len(prepared) > sample_size:
        prepared = rng.sample(prepared, sample_size)
    if not prepared:
        return {"accuracy": float("nan"), "n": 0}

    anchors = [p[0] for p in prepared]
    a_vecs = _embed_texts(model, tokenizer, anchors, terminator=terminator)

    correct = 0
    for i, (_, cands, target) in enumerate(prepared):
        c_vecs = _embed_texts(model, tokenizer, cands, terminator=terminator)
        sims = (a_vecs[i:i + 1] @ c_vecs.T)[0]
        pred = int(mx.argmax(sims).item())
        if pred == target:
            correct += 1
    return {"accuracy": correct / len(prepared), "n": len(prepared)}
