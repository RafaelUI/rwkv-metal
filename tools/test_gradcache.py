"""
GradCache verification on the real 0.1B checkpoint.

1. Equivalence: GradCache loss + parameter gradients must match the eager
   (non-cached) path -- GradCache is an exact memory schedule, not an
   approximation, so any real discrepancy is a bug.
2. Memory: peak memory should stay roughly flat as the effective batch (and
   therefore the contrastive negative pool) grows, which is the entire point.

Usage:
    python tools/test_gradcache.py --model /Users/s/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth
"""
import argparse
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

import rwkv_metal as rk
from rwkv_metal.embedding import (
    EmbeddingModel, TripletBatcher, load_triplets_jsonl,
    retrieval_loss, RETRIEVAL_GC,
)
from rwkv_metal.embedding.gradcache import gradcache_value_and_grad


def max_grad_diff(g1, g2):
    d1, d2 = dict(tree_flatten(g1)), dict(tree_flatten(g2))
    assert set(d1) == set(d2), "gradient trees differ in structure"
    worst, worst_key = 0.0, None
    for k in d1:
        diff = mx.abs(d1[k].astype(mx.float32) - d2[k].astype(mx.float32)).max().item()
        if diff > worst:
            worst, worst_key = diff, k
    return worst, worst_key


def grad_scale(g):
    d = dict(tree_flatten(g))
    return max(mx.abs(v.astype(mx.float32)).max().item() for v in d.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Users/s/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth")
    ap.add_argument("--data", default="/Users/s/Develop/retrieval_literature/train.jsonl")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--max_chars", type=int, default=400)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--skip_mem", action="store_true")
    args = ap.parse_args()

    tok = rk.WorldTokenizer()
    base, cfg = rk.load_pretrained(args.model)
    model = EmbeddingModel(base)
    # cast the WHOLE model (base + head) so base/head dtypes can't disagree
    dt = mx.bfloat16 if args.dtype == "bfloat16" else mx.float32
    model.update(tree_map(lambda x: x.astype(dt) if isinstance(x, mx.array) else x,
                          model.parameters()))
    mx.eval(model.parameters())
    print(f"model dtype: {args.dtype}")
    model.base._grad_ckpt = True

    rows = load_triplets_jsonl(args.data, task="retrieval", limit=64, seed=7)
    batcher = TripletBatcher(rows, tok, args.batch_size, max_chars=args.max_chars, shuffle=False)
    batch = next(iter(batcher))

    print("\n=== 1. equivalence: eager vs GradCache ===")
    eager_fn = nn.value_and_grad(model, lambda m, b: retrieval_loss(m, b, 0.05))
    t0 = time.time()
    loss_eager, g_eager = eager_fn(model, batch)
    mx.eval(loss_eager, g_eager)
    t_eager = time.time() - t0
    print(f"  eager     loss={loss_eager.item():.6f}  ({t_eager:.1f}s)")

    chunks = RETRIEVAL_GC.split(batch, args.chunk)
    t0 = time.time()
    loss_gc, g_gc = gradcache_value_and_grad(
        model, chunks, RETRIEVAL_GC.embed,
        lambda fields: RETRIEVAL_GC.loss(fields, 0.05),
    )
    mx.eval(loss_gc, g_gc)
    t_gc = time.time() - t0
    print(f"  gradcache loss={loss_gc.item():.6f}  ({t_gc:.1f}s, {len(chunks)} chunks)")

    print(f"  loss abs diff: {abs(loss_eager.item() - loss_gc.item()):.3e}")
    worst, key = max_grad_diff(g_eager, g_gc)
    scale = grad_scale(g_eager)
    print(f"  max grad abs diff: {worst:.3e}  (at {key})")
    print(f"  max grad magnitude: {scale:.3e}  -> relative: {worst/max(scale,1e-12):.3e}")

    if args.skip_mem:
        return

    print("\n=== 2. peak memory vs effective batch ===")
    print(f"  {'batch':>6} {'mode':>10} {'peak GB':>9} {'time s':>8} {'loss':>9}")
    for bs in (8, 16, 32, 48):
        try:
            b = TripletBatcher(rows, tok, bs, max_chars=args.max_chars, shuffle=False)
            bb = next(iter(b))

            mx.clear_cache()
            mx.reset_peak_memory()
            t0 = time.time()
            le, ge = eager_fn(model, bb)
            mx.eval(le, ge)
            print(f"  {bs:>6} {'eager':>10} {mx.get_peak_memory()/1e9:>9.2f} "
                  f"{time.time()-t0:>8.1f} {le.item():>9.4f}")
            del ge

            mx.clear_cache()
            mx.reset_peak_memory()
            t0 = time.time()
            ch = RETRIEVAL_GC.split(bb, args.chunk)
            lg, gg = gradcache_value_and_grad(
                model, ch, RETRIEVAL_GC.embed,
                lambda fields: RETRIEVAL_GC.loss(fields, 0.05),
            )
            mx.eval(lg, gg)
            print(f"  {bs:>6} {'gradcache':>10} {mx.get_peak_memory()/1e9:>9.2f} "
                  f"{time.time()-t0:>8.1f} {lg.item():>9.4f}")
            del gg
        except Exception as e:
            print(f"  {bs:>6}  failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
