"""
Contrastive data loading: (query, positive) pairs, {anchor, positive,
negative, task} triplets, and zero-shot classification batches (per-sample
candidate label pools).

Right-padding (not left, unlike EmbeddingRWKV's CUDA-kernel convention) is
deliberate throughout: RWKV-7 is a causal RNN, so padding placed AFTER the
pooled position can never leak into it -- no attention-mask bookkeeping
needed, padding tokens are simply outside the loss's computational path.
"""
import json
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

import mlx.core as mx

# ── Low-level tokenize + right-pad ───────────────────────────────────────────


def load_pairs_jsonl(path: str) -> List[Tuple[str, str]]:
    """Each line: {"query": "...", "positive": "..."}"""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pairs.append((obj["query"], obj["positive"]))
    return pairs


def load_triplets_jsonl(path: str, task: Optional[str] = None,
                         limit: Optional[int] = None, seed: int = 0) -> List[Dict]:
    """Streams a (possibly huge) {anchor, positive, negative, task} jsonl
    once. If `task` is given, keeps only matching rows. If `limit` is given,
    uses reservoir sampling (Algorithm R) so at most `limit` rows are ever
    held in memory regardless of file size -- needed here since the source
    file is 2.6GB / 554k rows and a full task subset can still be ~150MB+.
    """
    rng = random.Random(seed)
    reservoir: List[Dict] = []
    seen = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if task is not None and obj.get("task") != task:
                continue
            if limit is None:
                reservoir.append(obj)
                continue
            seen += 1
            if len(reservoir) < limit:
                reservoir.append(obj)
            else:
                j = rng.randint(0, seen - 1)
                if j < limit:
                    reservoir[j] = obj
    return reservoir


def encode_batch(tokenizer, texts: Sequence[str], terminator: int = 0):
    """Tokenize + append terminator + right-pad to the batch max length.

    Returns (idx [B,T] mx.array int32, pool_idx [B] mx.array int32) --
    pool_idx[b] is the position of text b's terminator token (where its
    pooled embedding should be read from).
    """
    seqs = [tokenizer.encode(t) + [terminator] for t in texts]
    pool_idx = [len(s) - 1 for s in seqs]
    max_len = max(len(s) for s in seqs)
    for s in seqs:
        s.extend([terminator] * (max_len - len(s)))
    return mx.array(seqs), mx.array(pool_idx)


# ── Batchers ──────────────────────────────────────────────────────────────


class _CyclingIndexSampler:
    """Shared index-cycling logic used by all batchers below."""

    def __init__(self, n: int, batch_size: int, shuffle: bool):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._order = list(range(n))
        self._pos = 0
        if shuffle:
            random.shuffle(self._order)

    def next_indices(self) -> List[int]:
        if self._pos + self.batch_size > len(self._order):
            if self.shuffle:
                random.shuffle(self._order)
            self._pos = 0
        idx = self._order[self._pos:self._pos + self.batch_size]
        self._pos += self.batch_size
        return idx


class PairBatcher:
    """Cycles a list of (query, positive) pairs into
    (q_idx, q_pool, d_idx, d_pool) batches, for use with `info_nce_loss`
    (in-batch negatives only, no explicit hard negative)."""

    def __init__(self, pairs: Sequence[Tuple[str, str]], tokenizer,
                 batch_size: int, terminator: int = 0, shuffle: bool = True):
        if len(pairs) < batch_size:
            raise ValueError(
                f"batch_size={batch_size} > number of pairs ({len(pairs)}); "
                "add more pairs or lower batch_size."
            )
        self.pairs = list(pairs)
        self.tok = tokenizer
        self.terminator = terminator
        self._sampler = _CyclingIndexSampler(len(self.pairs), batch_size, shuffle)

    def __iter__(self):
        return self

    def __next__(self):
        idxs = self._sampler.next_indices()
        queries = [self.pairs[i][0] for i in idxs]
        docs = [self.pairs[i][1] for i in idxs]
        q_idx, q_pool = encode_batch(self.tok, queries, self.terminator)
        d_idx, d_pool = encode_batch(self.tok, docs, self.terminator)
        return q_idx, q_pool, d_idx, d_pool


class TripletBatcher:
    """Cycles {anchor, positive, negative} triplets (retrieval or sts rows)
    into (a_idx,a_pool, p_idx,p_pool, n_idx,n_pool) batches, for use with
    `triplet_pool_loss` -- explicit hard negative + every other sample's
    positive/negative in the batch as additional negatives, for free.
    """

    def __init__(self, triplets: Sequence[Dict], tokenizer, batch_size: int,
                 terminator: int = 0, shuffle: bool = True, max_chars: Optional[int] = None):
        if len(triplets) < batch_size:
            raise ValueError(
                f"batch_size={batch_size} > number of triplets ({len(triplets)}); "
                "add more data or lower batch_size."
            )
        self.triplets = list(triplets)
        self.tok = tokenizer
        self.terminator = terminator
        self.max_chars = max_chars
        self._sampler = _CyclingIndexSampler(len(self.triplets), batch_size, shuffle)

    def _clip(self, s: str) -> str:
        return s if self.max_chars is None else s[: self.max_chars]

    def __iter__(self):
        return self

    def __next__(self):
        idxs = self._sampler.next_indices()
        rows = [self.triplets[i] for i in idxs]
        a = [self._clip(r["anchor"]) for r in rows]
        p = [self._clip(r["positive"]) for r in rows]
        n = [self._clip(r["negative"]) for r in rows]
        a_idx, a_pool = encode_batch(self.tok, a, self.terminator)
        p_idx, p_pool = encode_batch(self.tok, p, self.terminator)
        n_idx, n_pool = encode_batch(self.tok, n, self.terminator)
        return a_idx, a_pool, p_idx, p_pool, n_idx, n_pool


# ── Zero-shot classification (per-sample candidate label pool) ─────────────

_CATEGORY_RE = re.compile(r"categories:\s*(.+?)\n")


def parse_classification_candidates(anchor_text: str) -> Optional[List[str]]:
    """Pulls the candidate label list out of the instruction line baked into
    the anchor text, e.g. '...categories: joy, fear, shame\\n...' -> [joy,
    fear, shame]. Each row in this dataset carries its OWN label vocabulary
    (verified: 20000/20000 sampled rows had a distinct 7-label set), so
    this can't be a fixed global list -- it's parsed per row."""
    m = _CATEGORY_RE.search(anchor_text)
    if not m:
        return None
    return [x.strip().rstrip(".") for x in m.group(1).split(",")]


class ClassificationBatcher:
    """Zero-shot classification as retrieval-over-candidate-labels: each
    row's anchor carries its own candidate label pool (parsed from the
    instruction text); positive is the correct label string. Candidate
    counts are padded to the batch's max K with a mask (label vocab differs
    per row -- see parse_classification_candidates), not assumed fixed.

    Yields (a_idx, a_pool, cand_idx [B,K,T], cand_pool [B,K], mask [B,K],
    target_idx [B]) for use with `zero_shot_classification_loss`.
    """

    def __init__(self, triplets: Sequence[Dict], tokenizer, batch_size: int,
                 terminator: int = 0, shuffle: bool = True, max_chars: Optional[int] = None):
        rows = []
        for t in triplets:
            # parse candidates from the FULL anchor (instruction line lives
            # up front); only the text handed to the tokenizer gets clipped.
            cands = parse_classification_candidates(t["anchor"])
            if cands is None or t["positive"] not in cands:
                continue
            anchor_text = t["anchor"] if max_chars is None else t["anchor"][:max_chars]
            rows.append((anchor_text, cands, cands.index(t["positive"])))
        if len(rows) < batch_size:
            raise ValueError(
                f"batch_size={batch_size} > usable classification rows ({len(rows)}); "
                "add more data or lower batch_size."
            )
        self.rows = rows
        self.tok = tokenizer
        self.terminator = terminator
        self._sampler = _CyclingIndexSampler(len(self.rows), batch_size, shuffle)

    def __iter__(self):
        return self

    def __next__(self):
        idxs = self._sampler.next_indices()
        batch = [self.rows[i] for i in idxs]
        anchors = [b[0] for b in batch]
        a_idx, a_pool = encode_batch(self.tok, anchors, self.terminator)

        B = len(batch)
        K = max(len(b[1]) for b in batch)
        target_idx = [b[2] for b in batch]

        # tokenize every candidate per row (python-side padding to a common
        # length -- avoids needing in-place mx.array assignment)
        tok_rows: List[List[List[int]]] = []
        for _, cands, _ in batch:
            row = [self.tok.encode(c) + [self.terminator] for c in cands]
            while len(row) < K:
                row.append([self.terminator])  # dummy 1-token pad slot, masked out below
            tok_rows.append(row)

        max_t = max(len(seq) for row in tok_rows for seq in row)
        cand_idx, cand_pool, mask = [], [], []
        for bi, row in enumerate(tok_rows):
            n_real = len(batch[bi][1])
            row_idx, row_pool, row_mask = [], [], []
            for k, seq in enumerate(row):
                pool_pos = len(seq) - 1
                seq = seq + [self.terminator] * (max_t - len(seq))
                row_idx.append(seq)
                row_pool.append(pool_pos)
                row_mask.append(1.0 if k < n_real else 0.0)
            cand_idx.append(row_idx)
            cand_pool.append(row_pool)
            mask.append(row_mask)

        return (a_idx, a_pool,
                mx.array(cand_idx), mx.array(cand_pool),
                mx.array(mask), mx.array(target_idx))
