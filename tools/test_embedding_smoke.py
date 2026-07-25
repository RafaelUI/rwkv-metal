"""
Smoke-test for rwkv_metal.embedding on a real checkpoint.

Loads a base RWKV-7 G1 0.1B checkpoint (official .pth format), embeds a set
of multilingual text chunks (en/ru/sr, mixed domains), and prints nearest
neighbours per chunk so the result can be eyeballed for sanity (this is a
*base* LM, not a contrastively fine-tuned embedding model — expect "usable
but rough", not SOTA retrieval quality).

Usage:
    python tools/test_embedding_smoke.py \
        --model /Users/s/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth \
        --chunks /Users/s/Develop/test.txt
"""
import argparse
import time

import rwkv_metal as rk
from rwkv_metal.embedding import embed_texts, cosine_similarity_matrix


def load_chunks(path, max_chars=1200):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    parts = text.split("———————————————", 1)  # noop safeguard if marker changes
    raw_chunks = [c for c in text.split("CHUNK ———————————————") if c.strip()]
    chunks = []
    for c in raw_chunks:
        c = c.strip().strip("—").strip()
        if not c:
            continue
        chunks.append(c[:max_chars])
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Users/s/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth")
    ap.add_argument("--chunks", default="/Users/s/Develop/test.txt")
    ap.add_argument("--pooling", default="last", choices=["last", "mean"])
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    print(f"loading model from {args.model} ...")
    t0 = time.time()
    model, cfg = rk.load_pretrained(args.model)
    print(f"loaded in {time.time()-t0:.1f}s  n_layer={cfg.n_layer} n_embd={cfg.n_embd}")

    tok = rk.WorldTokenizer()

    chunks = load_chunks(args.chunks)
    print(f"{len(chunks)} chunks loaded from {args.chunks}")

    t0 = time.time()
    vecs = embed_texts(model, tok, chunks, pooling=args.pooling)
    print(f"embedded {len(chunks)} chunks in {time.time()-t0:.1f}s  -> shape {vecs.shape}")

    sim = cosine_similarity_matrix(vecs)
    sim_list = sim.tolist()

    for i, chunk in enumerate(chunks):
        preview = chunk[:70].replace("\n", " ")
        row = sim_list[i]
        ranked = sorted(
            [(j, s) for j, s in enumerate(row) if j != i],
            key=lambda x: -x[1],
        )[: args.topk]
        print(f"\n[{i:2d}] {preview!r}")
        for j, s in ranked:
            other_preview = chunks[j][:60].replace("\n", " ")
            print(f"      -> [{j:2d}] sim={s:.3f}  {other_preview!r}")


if __name__ == "__main__":
    main()
