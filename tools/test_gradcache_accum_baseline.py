"""
Is GradCache's small numeric deviation an algorithmic error, or just
floating-point summation order?

GradCache sums each shared parameter's gradient across N chunk-backward
passes; the eager path produces it as one sum over the whole batch. Different
grouping of the same additions = different rounding. Ordinary gradient
accumulation has exactly the same property, so it makes the right yardstick:

  A) eager full batch                      (reference)
  B) grad accumulation, same chunk size    (known-good, universally accepted)
  C) GradCache, same chunk size

If |C - A| is on the order of |B - A|, GradCache is as numerically exact as
gradient accumulation, and the deviation is pure summation-order rounding.

NOTE: B is only comparable to A/C in gradient STRUCTURE, not value -- with a
contrastive loss, accumulation over micro-batches genuinely changes the math
(each micro-batch only sees its own negatives), which is precisely why
GradCache exists. So B's loss/grad will differ substantially; what we compare
is A-vs-C (should be tiny) against the fp32 noise floor established by
re-running A twice with different batch groupings.
"""
import argparse

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

import rwkv_metal as rk
from rwkv_metal.embedding import (
    EmbeddingModel, TripletBatcher, load_triplets_jsonl,
    retrieval_loss, RETRIEVAL_GC,
)
from rwkv_metal.embedding.gradcache import gradcache_value_and_grad


def max_rel_diff(g1, g2):
    d1, d2 = dict(tree_flatten(g1)), dict(tree_flatten(g2))
    worst, worst_key = 0.0, None
    scale = max(mx.abs(v.astype(mx.float32)).max().item() for v in d1.values())
    for k in d1:
        diff = mx.abs(d1[k].astype(mx.float32) - d2[k].astype(mx.float32)).max().item()
        if diff > worst:
            worst, worst_key = diff, k
    return worst, worst / max(scale, 1e-12), worst_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Users/s/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth")
    ap.add_argument("--data", default="/Users/s/Develop/retrieval_literature/train.jsonl")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--max_chars", type=int, default=400)
    args = ap.parse_args()

    tok = rk.WorldTokenizer()
    base, _ = rk.load_pretrained(args.model)
    model = EmbeddingModel(base)
    model.update(tree_map(lambda x: x.astype(mx.float32) if isinstance(x, mx.array) else x,
                          model.parameters()))
    mx.eval(model.parameters())
    model.base._grad_ckpt = True

    rows = load_triplets_jsonl(args.data, task="retrieval", limit=64, seed=7)
    batcher = TripletBatcher(rows, tok, args.batch_size, max_chars=args.max_chars, shuffle=False)
    batch = next(iter(batcher))

    eager_fn = nn.value_and_grad(model, lambda m, b: retrieval_loss(m, b, 0.05))

    # A: eager full batch (reference)
    loss_a, g_a = eager_fn(model, batch)
    mx.eval(loss_a, g_a)

    # A': identical call again -- establishes the run-to-run noise floor
    loss_a2, g_a2 = eager_fn(model, batch)
    mx.eval(loss_a2, g_a2)
    abs_n, rel_n, key_n = max_rel_diff(g_a, g_a2)
    print(f"noise floor (eager vs eager):      abs {abs_n:.3e}  rel {rel_n:.3e}  at {key_n}")

    # C: GradCache
    chunks = RETRIEVAL_GC.split(batch, args.chunk)
    loss_c, g_c = gradcache_value_and_grad(
        model, chunks, RETRIEVAL_GC.embed,
        lambda fields: RETRIEVAL_GC.loss(fields, 0.05))
    mx.eval(loss_c, g_c)
    abs_c, rel_c, key_c = max_rel_diff(g_a, g_c)
    print(f"gradcache vs eager:                abs {abs_c:.3e}  rel {rel_c:.3e}  at {key_c}")
    print(f"  loss  eager={loss_a.item():.7f}  gradcache={loss_c.item():.7f}")

    # B: plain gradient accumulation over the same chunks (NOT the same math
    # for a contrastive loss -- shown to quantify how different a genuinely
    # different-math baseline looks)
    total = None
    for ch in chunks:
        _, g = eager_fn(model, ch)
        mx.eval(g)
        total = g if total is None else tree_map(lambda x, y: x + y, total, g)
        mx.eval(total)
    n = len(chunks)
    total = tree_map(lambda x: x * (1.0 / n) * n, total)  # keep sum semantics explicit
    abs_b, rel_b, key_b = max_rel_diff(g_a, total)
    print(f"grad-accum(sum over chunks) vs eager: abs {abs_b:.3e}  rel {rel_b:.3e}  at {key_b}")

    print("\ninterpretation:")
    print(f"  gradcache deviation / noise floor      = {rel_c/max(rel_n,1e-12):.1f}x")
    print(f"  grad-accum deviation / noise floor     = {rel_b/max(rel_n,1e-12):.1f}x")
    print("  (grad-accum changes the contrastive math; gradcache should not)")


if __name__ == "__main__":
    main()
