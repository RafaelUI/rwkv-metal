"""
Full curriculum run for EmbeddingRWKV-on-MLX (not a smoke test).

Differences from tools/test_embedding_curriculum_smoke.py:

  - ONE streaming pass over the 2.6GB jsonl fills the reservoirs for every
    requested task instead of one pass per task, and can filter by language
    on the way through (--lang).
  - Real slice sizes and step counts, with warmup + cosine LR decay.
  - retrieval / sts run under GradCache, so the contrastive negative pool is
    the full batch (2*B candidates per anchor) while activation memory stays
    bounded by --gc_chunk. Classification stays eager on purpose: its
    candidate pool is per-row, so a bigger batch buys it no extra negatives.
  - Works with either base: the official World .pth (--model <file.pth>,
    World tokenizer) or a local from-scratch/fla-remap checkpoint directory
    (--model <dir>, its own BPE tokenizer), e.g. ru60m.
  - Everything (config, per-step losses, before/after metrics, timings,
    peak memory) is streamed to a JSON file after every stage, so the run is
    inspectable while it is still going and survives being killed.

Language note: LitRetrieval is ~57% Russian for retrieval/sts, but its
classification task is 100% English (both passages and emotion labels), so
--lang ru implies a two-stage curriculum. The runner refuses to build a
stage it has no rows for rather than silently skipping it.

Usage (Russian, ru60m base, two stages):
    python tools/run_embedding_curriculum.py \
        --model /Users/s/Develop/WKV-kvant/ru60m \
        --data  /Users/s/Develop/retrieval_literature/train.jsonl \
        --lang ru --stages retrieval,sts \
        --out_dir runs/ru60m_curriculum
"""
import argparse
import json
import os
import random
import re
import time
from typing import Dict, List

import mlx.core as mx
import rwkv_metal as rk
from rwkv_metal.embedding import (
    EmbeddingModel, EmbedTrainConfig, finetune_embedding,
    TripletBatcher, ClassificationBatcher,
    retrieval_loss, sts_loss, classification_loss,
    RETRIEVAL_GC, STS_GC,
    evaluate_retrieval, evaluate_sts_pairwise, evaluate_classification,
)

ALL_TASKS = ("retrieval", "sts", "classification")

_CYR = re.compile(r"[\u0430-\u044f\u0451\u0410-\u042f\u0401]")
_LAT = re.compile(r"[a-zA-Z]")


def row_language(obj: Dict, probe_chars: int = 600) -> str:
    """Cheap script-count heuristic on the positive field. LitRetrieval rows
    are monolingual (anchor/positive/negative all share a language), so one
    field is enough."""
    s = obj.get("positive", "")[:probe_chars]
    return "ru" if len(_CYR.findall(s)) > len(_LAT.findall(s)) else "en"


def sample_all_tasks(path: str, limit_per_task: int, tasks=ALL_TASKS,
                     lang: str = "any", seed: int = 0) -> Dict[str, List[Dict]]:
    """One pass, one reservoir per task (Algorithm R). Memory stays at
    limit_per_task rows per task no matter how big the file is.

    lang: "any" | "ru" | "en" -- rows of other languages are dropped BEFORE
    the reservoir counter, so the sample stays uniform over the kept subset.
    """
    rngs = {t: random.Random(seed + i) for i, t in enumerate(tasks)}
    res = {t: [] for t in tasks}
    seen = {t: 0 for t in tasks}
    dropped = 0
    t0 = time.time()
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            task = obj.get("task")
            if task not in res:
                continue
            if lang != "any" and row_language(obj) != lang:
                dropped += 1
                continue
            seen[task] += 1
            r = res[task]
            if len(r) < limit_per_task:
                r.append(obj)
            else:
                j = rngs[task].randint(0, seen[task] - 1)
                if j < limit_per_task:
                    r[j] = obj
            if (lineno + 1) % 100000 == 0:
                print(f"  ...{lineno+1} lines ({time.time()-t0:.0f}s)", flush=True)
    print(f"  scanned {sum(seen.values())} matching rows in {time.time()-t0:.0f}s"
          + (f" ({dropped} dropped by --lang {lang})" if lang != "any" else "")
          + ": " + " | ".join(f"{t}={seen[t]}" for t in tasks), flush=True)
    return res


def split_train_eval(rows, n_eval, seed):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[n_eval:], rows[:n_eval]


class Journal:
    """Incremental JSON log -- rewritten after every stage and every eval."""

    def __init__(self, path, config):
        self.path = path
        self.data = {"config": config, "started": time.time(), "stages": []}
        self.flush()

    def flush(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    def add_stage(self, entry):
        self.data["stages"].append(entry)
        self.flush()
        return entry


def run_stage(label, model, tok, train_rows, eval_rows, batcher, compute_loss,
              eval_fn, cfg, journal, gradcache_spec=None, eval_kwargs=None):
    print(f"\n{'='*64}\n=== stage: {label}  ({len(train_rows)} train / {len(eval_rows)} eval)\n{'='*64}", flush=True)
    eval_kwargs = eval_kwargs or {}

    entry = {"stage": label, "n_train": len(train_rows), "n_eval": len(eval_rows),
             "steps": cfg.max_steps, "gradcache": gradcache_spec is not None,
             "batch_size": batcher._sampler.batch_size}
    journal.add_stage(entry)

    t0 = time.time()
    before = eval_fn(model, tok, eval_rows, **eval_kwargs)
    entry["eval_before"] = before
    entry["eval_before_secs"] = round(time.time() - t0, 1)
    print(f"  eval BEFORE: {before}  ({entry['eval_before_secs']}s)", flush=True)
    journal.flush()

    losses, peaks = [], []
    t0 = time.time()

    def on_step(step, loss, peak):
        losses.append(loss)
        peaks.append(peak)
        if step % 50 == 0:
            entry["loss_curve"] = losses
            entry["peak_gb"] = max(peaks)
            entry["elapsed_secs"] = round(time.time() - t0, 1)
            journal.flush()

    finetune_embedding(model, batcher, cfg, compute_loss,
                       on_step=on_step, gradcache_spec=gradcache_spec)

    entry["loss_curve"] = losses
    entry["loss_first"] = losses[0]
    entry["loss_last"] = losses[-1]
    entry["loss_last50_mean"] = sum(losses[-50:]) / len(losses[-50:])
    entry["peak_gb"] = max(peaks)
    entry["train_secs"] = round(time.time() - t0, 1)
    entry["secs_per_step"] = round(entry["train_secs"] / max(1, len(losses)), 3)
    entry["checkpoint"] = cfg.checkpoint_path
    journal.flush()
    print(f"  loss {losses[0]:.4f} -> {losses[-1]:.4f} "
          f"(last-50 mean {entry['loss_last50_mean']:.4f}) | "
          f"{entry['train_secs']:.0f}s @ {entry['secs_per_step']}s/step | "
          f"peak {entry['peak_gb']:.2f} GB", flush=True)

    t0 = time.time()
    after = eval_fn(model, tok, eval_rows, **eval_kwargs)
    entry["eval_after"] = after
    entry["eval_after_secs"] = round(time.time() - t0, 1)
    journal.flush()
    print(f"  eval AFTER:  {after}", flush=True)
    return entry


def load_base(model_path: str):
    """Accepts either an official World .pth (-> RWKV7X070 + WorldTokenizer)
    or a local checkpoint directory (config.json + model.safetensors +
    tokenizer*.json -> its own BPE tokenizer). Returns (base, tokenizer,
    terminator, label)."""
    if os.path.isdir(model_path):
        base, cfg = rk.load_local_rwkv7(model_path)
        tok = rk.load_local_tokenizer(model_path)
        # For a BPE vocab the terminator is a real special token (<eos>),
        # not the World vocab's unmapped id 0. Measured on 200 held-out
        # Russian retrieval rows with the untrained ru60m base:
        # <eos> MRR 0.0425 > <bos> 0.0384 > <pad> 0.0376.
        term = tok.terminator_id
        label = f"{os.path.basename(model_path.rstrip('/'))} (local, vocab {tok.vocab_size})"
    else:
        base, cfg = rk.load_pretrained(model_path)
        tok = rk.WorldTokenizer()
        term = 0  # unmapped id in the World vocab -- see embedding/embed.py
        label = f"{os.path.basename(model_path)} (World, vocab {cfg.vocab_size})"
    return base, tok, term, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Users/s/Develop/WKV-kvant/ru60m",
                    help=".pth (official World) or a local checkpoint directory")
    ap.add_argument("--data", default="/Users/s/Develop/retrieval_literature/train.jsonl")
    ap.add_argument("--out_dir", default="runs/curriculum_full")
    ap.add_argument("--lang", default="ru", choices=["any", "ru", "en"])
    ap.add_argument("--stages", default="retrieval,sts",
                    help="comma-separated subset of retrieval,sts,classification")
    ap.add_argument("--n_per_task", type=int, default=20000)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch_size", type=int, default=32, help="retrieval/sts (GradCache)")
    ap.add_argument("--gc_chunk", type=int, default=4)
    ap.add_argument("--cls_batch", type=int, default=8,
                    help="classification is eager; it also embeds B*K labels/step")
    ap.add_argument("--max_chars", type=int, default=800)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lr_min", type=float, default=1e-6)
    ap.add_argument("--save_every", type=int, default=250)
    ap.add_argument("--cls_eval_sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    stages = [t.strip() for t in args.stages.split(",") if t.strip()]
    bad = [t for t in stages if t not in ALL_TASKS]
    if bad:
        raise SystemExit(f"unknown stage(s): {bad}; expected a subset of {list(ALL_TASKS)}")
    if args.lang == "ru" and "classification" in stages:
        raise SystemExit(
            "LitRetrieval's classification task is 100% English (passages AND "
            "emotion labels), so --lang ru leaves it with no rows. Drop it from "
            "--stages, or run it separately with --lang any/en on a base whose "
            "tokenizer covers English.")

    os.makedirs(args.out_dir, exist_ok=True)
    journal = Journal(os.path.join(args.out_dir, "run.json"), vars(args))

    print("=== loading model ===", flush=True)
    t0 = time.time()
    base, tok, terminator, label = load_base(args.model)
    model = EmbeddingModel(base)
    print(f"  {label} | terminator id {terminator} | loaded in {time.time()-t0:.1f}s", flush=True)
    journal.data["base"] = label
    journal.data["terminator"] = terminator
    journal.flush()

    print(f"\n=== sampling {args.n_per_task}+{args.n_eval} rows per task "
          f"(single pass, lang={args.lang}, tasks={stages}) ===", flush=True)
    pools = sample_all_tasks(args.data, args.n_per_task + args.n_eval,
                             tasks=stages, lang=args.lang, seed=args.seed)
    splits = {}
    for i, t in enumerate(stages):
        if len(pools[t]) <= args.n_eval:
            raise SystemExit(f"task {t!r}: only {len(pools[t])} rows after filtering "
                             f"-- not enough for n_eval={args.n_eval} plus training")
        tr, ev = split_train_eval(pools[t], args.n_eval, seed=args.seed + i)
        splits[t] = (tr, ev)
        print(f"  {t}: train={len(tr)} eval={len(ev)}", flush=True)
    del pools

    def cfg_for(name, gc_chunk):
        return EmbedTrainConfig(
            lr=args.lr, max_steps=args.steps,
            warmup_steps=args.warmup, lr_schedule="cosine", lr_min=args.lr_min,
            grad_checkpoint=True, log_every=25,
            save_every=args.save_every,
            checkpoint_path=os.path.join(args.out_dir, f"embed_{name}.safetensors"),
            gradcache_chunk=gc_chunk,
        )

    SPECS = {
        # task -> (batcher factory, compute_loss, eval fn, gradcache spec, extra eval kwargs)
        "retrieval": (
            lambda rows: TripletBatcher(rows, tok, args.batch_size,
                                        terminator=terminator, max_chars=args.max_chars),
            retrieval_loss, evaluate_retrieval, RETRIEVAL_GC, {}),
        "sts": (
            lambda rows: TripletBatcher(rows, tok, args.batch_size,
                                        terminator=terminator, max_chars=args.max_chars),
            sts_loss, evaluate_sts_pairwise, STS_GC, {}),
        "classification": (
            lambda rows: ClassificationBatcher(rows, tok, args.cls_batch,
                                               terminator=terminator, max_chars=args.max_chars),
            classification_loss, evaluate_classification, None,
            {"sample_size": args.cls_eval_sample}),
    }

    for name in stages:
        make_batcher, loss_fn, eval_fn, gc_spec, extra = SPECS[name]
        tr, ev = splits[name]
        run_stage(name, model, tok, tr, ev, make_batcher(tr), loss_fn, eval_fn,
                  cfg_for(name, args.gc_chunk if gc_spec is not None else 0),
                  journal, gradcache_spec=gc_spec,
                  eval_kwargs={"max_chars": args.max_chars,
                               "terminator": terminator, **extra})

    # ── final: every stage's eval re-run on the fully-trained model ───────
    print(f"\n{'='*64}\n=== final cross-task eval (after all stages)\n{'='*64}", flush=True)
    final = {}
    for name in stages:
        _, _, eval_fn, _, extra = SPECS[name]
        final[name] = eval_fn(model, tok, splits[name][1], max_chars=args.max_chars,
                              terminator=terminator, **extra)
        print(f"  {name}: {final[name]}", flush=True)
    journal.data["final_eval"] = final
    journal.data["finished"] = time.time()
    journal.data["total_secs"] = round(journal.data["finished"] - journal.data["started"], 1)
    journal.flush()
    print(f"\nDone in {journal.data['total_secs']/60:.1f} min -> {journal.path}", flush=True)


if __name__ == "__main__":
    main()
