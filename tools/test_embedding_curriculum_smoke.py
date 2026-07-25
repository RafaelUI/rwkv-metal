"""
Curriculum smoke-test on the real LitRetrieval dataset
(Develop/retrieval_literature/train.jsonl, 554k {anchor,positive,negative,
task} triplets, ~1/3 each retrieval/sts/classification, ru+en literature).

Subsamples a small slice per task via reservoir sampling (the full 554k
would take a week+ to train on a single Mac -- per the plan, we stage
curriculum-style: retrieval, then sts, then classification, each with its
own loss:

  - retrieval: asymmetric triplet_pool_loss (query vs candidate passages;
    explicit hard negative + every other sample's positive/negative in the
    batch as additional negatives).
  - sts: symmetric triplet_pool_loss (doc<->doc).
  - classification: zero-shot -- anchor passage vs its own per-row candidate
    emotion-label pool (parsed from the instruction text), no learned
    classifier head.

Each task's rows are split into a train slice and a disjoint held-out eval
slice (single reservoir sample of train+eval size, then sliced -- guarantees
no overlap). Eval runs before AND after each stage, so we can tell whether
the stage actually helped, not just that the training loss went down.

Usage:
    python tools/test_embedding_curriculum_smoke.py \
        --model /Users/s/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth \
        --data /Users/s/Develop/retrieval_literature/train.jsonl
"""
import argparse
import random

import rwkv_metal as rk
from rwkv_metal.embedding import (
    EmbeddingModel, EmbedTrainConfig, finetune_embedding,
    TripletBatcher, ClassificationBatcher,
    retrieval_loss, sts_loss, classification_loss,
    load_triplets_jsonl,
    evaluate_retrieval, evaluate_sts_pairwise, evaluate_classification,
)


def split_train_eval(data_path, task, n_train, n_eval, seed=42):
    combined = load_triplets_jsonl(data_path, task=task, limit=n_train + n_eval, seed=seed)
    random.Random(seed).shuffle(combined)
    return combined[:n_train], combined[n_train:]


def run_stage(label, model, tok, train_rows, eval_rows, batcher_cls, compute_loss,
              eval_fn, steps, checkpoint_path, lr=2e-5, batch_size=8, max_chars=800):
    print(f"\n=== stage: {label} ===")
    before = eval_fn(model, tok, eval_rows)
    print(f"  eval BEFORE: {before}")

    batcher = batcher_cls(train_rows, tok, batch_size, max_chars=max_chars)
    cfg = EmbedTrainConfig(
        lr=lr, max_steps=steps, log_every=5,
        grad_checkpoint=True, save_every=0,
        checkpoint_path=checkpoint_path,
    )
    losses = []
    finetune_embedding(model, batcher, cfg, compute_loss,
                        on_step=lambda step, loss, peak: losses.append(loss))
    print(f"  loss[0]={losses[0]:.4f} -> loss[-1]={losses[-1]:.4f}")

    after = eval_fn(model, tok, eval_rows)
    print(f"  eval AFTER:  {after}")
    return before, after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Users/s/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth")
    ap.add_argument("--data", default="/Users/s/Develop/retrieval_literature/train.jsonl")
    ap.add_argument("--n_per_task", type=int, default=500)
    ap.add_argument("--n_eval", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--max_chars", type=int, default=800)
    args = ap.parse_args()

    tok = rk.WorldTokenizer()
    base, cfg = rk.load_pretrained(args.model)
    model = EmbeddingModel(base)

    print(f"\n=== sampling {args.n_per_task}+{args.n_eval} rows per task (reservoir, train/eval split) ===")
    retrieval_train, retrieval_eval = split_train_eval(args.data, "retrieval", args.n_per_task, args.n_eval, seed=1)
    sts_train, sts_eval = split_train_eval(args.data, "sts", args.n_per_task, args.n_eval, seed=2)
    cls_train, cls_eval = split_train_eval(args.data, "classification", args.n_per_task, args.n_eval, seed=3)
    print(f"retrieval train={len(retrieval_train)} eval={len(retrieval_eval)} | "
          f"sts train={len(sts_train)} eval={len(sts_eval)} | "
          f"classification train={len(cls_train)} eval={len(cls_eval)}")

    run_stage("retrieval", model, tok, retrieval_train, retrieval_eval,
              TripletBatcher, retrieval_loss, evaluate_retrieval,
              args.steps, "/tmp/embed_curriculum_retrieval.safetensors",
              batch_size=args.batch_size, max_chars=args.max_chars)

    run_stage("sts", model, tok, sts_train, sts_eval,
              TripletBatcher, sts_loss, evaluate_sts_pairwise,
              args.steps, "/tmp/embed_curriculum_sts.safetensors",
              batch_size=args.batch_size, max_chars=args.max_chars)

    run_stage("classification", model, tok, cls_train, cls_eval,
              ClassificationBatcher, classification_loss, evaluate_classification,
              args.steps, "/tmp/embed_curriculum_classification.safetensors",
              batch_size=args.batch_size, max_chars=args.max_chars)


if __name__ == "__main__":
    main()
