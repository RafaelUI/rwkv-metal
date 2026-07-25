"""
Smoke-test for rwkv_metal.embedding.train on a real checkpoint.

Builds synthetic (query, positive) pairs out of test.txt's multilingual
chunks (first sentence -> rest of the chunk) and runs a few dozen
contrastive fine-tuning steps in two configurations:

  1. full fine-tune (nothing frozen) -- the default recipe for small models.
  2. LoRA on top of a frozen base -- structural compatibility check for the
     path that matters once models get too big to full-FT (future 1.4B/4B).
     Only proves the code path works end-to-end here (0.1B); the actual
     memory-saving benefit only shows up on bigger models.

Usage:
    python tools/test_embedding_train_smoke.py \
        --model /Users/s/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth \
        --chunks /Users/s/Develop/test.txt
"""
import argparse
import re

import rwkv_metal as rk
from rwkv_metal.embedding import EmbeddingModel, EmbedTrainConfig, finetune_embedding, PairBatcher, pair_loss
from rwkv_metal.lora import add_lora


def load_chunks(path, max_chars=1200):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    raw_chunks = [c for c in text.split("CHUNK ———————————————") if c.strip()]
    chunks = []
    for c in raw_chunks:
        c = c.strip().strip("—").strip()
        if c:
            chunks.append(c[:max_chars])
    return chunks


_SENT_END = re.compile(r"[.!?]\s")


def make_pairs(chunks):
    """query = first sentence-ish (~<=120 chars), positive = rest of the chunk."""
    pairs = []
    for c in chunks:
        m = _SENT_END.search(c[:200])
        cut = m.end() if m else min(80, len(c) // 2)
        query, positive = c[:cut].strip(), c[cut:].strip()
        if query and positive:
            pairs.append((query, positive))
    return pairs


def run(label, model, tok, pairs, steps, checkpoint_path):
    batcher = PairBatcher(pairs, tok, batch_size=min(8, len(pairs)))
    cfg = EmbedTrainConfig(
        lr=2e-5, max_steps=steps, log_every=5,
        grad_checkpoint=True, save_every=0,
        checkpoint_path=checkpoint_path,
    )
    losses = []
    result = finetune_embedding(model, batcher, cfg, pair_loss,
                                 on_step=lambda step, loss, peak: losses.append(loss))
    print(f"[{label}] loss[0]={losses[0]:.4f} -> loss[-1]={losses[-1]:.4f}")
    return result, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Users/s/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth")
    ap.add_argument("--chunks", default="/Users/s/Develop/test.txt")
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()

    tok = rk.WorldTokenizer()
    chunks = load_chunks(args.chunks)
    pairs = make_pairs(chunks)
    print(f"{len(pairs)} synthetic (query, positive) pairs from {len(chunks)} chunks")

    # ── 1. Full fine-tune ────────────────────────────────────────────────────
    print("\n=== full fine-tune ===")
    base, cfg = rk.load_pretrained(args.model)
    model = EmbeddingModel(base)
    run("full-FT", model, tok, pairs, args.steps, "/tmp/embed_fullft_smoke.safetensors")

    # ── 2. LoRA on a frozen base (structural check) ─────────────────────────
    print("\n=== LoRA on frozen base ===")
    base2, cfg2 = rk.load_pretrained(args.model)
    base2, info = add_lora(base2, rank=8, alpha=16.0)
    print(f"LoRA trainable (base only): {info['trainable_pct']:.3f}%")
    model2 = EmbeddingModel(base2)
    run("LoRA", model2, tok, pairs, min(10, args.steps), "/tmp/embed_lora_smoke.safetensors")


if __name__ == "__main__":
    main()
