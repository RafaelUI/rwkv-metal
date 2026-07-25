"""
Абляции головы реранкера на ГОТОВОМ кэше состояний.

Кэш считается один раз (`tools/run_reranker.py --cache_path ...`), после чего
обучение головы стоит секунды — значит, сравнивать варианты можно честно, с
несколькими семенами, а не «прогнали разок, вроде лучше».

Пример:
    .venv/bin/python tools/ablate_reranker.py \
        --cache runs/reranker_0.1b/cache.safetensors \
        --cache_layers 0,5,11 --eval_queries 150 \
        --seeds 0 1 2 --out runs/reranker_0.1b/ablation.json
"""
import argparse
import json
import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwkv_metal.model import load_pretrained
from rwkv_metal.reranker import (
    Reranker, RerankerConfig, RerankTrainConfig, StateCache,
    evaluate, resolve_layer_indices, train_reranker,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth")
    p.add_argument("--cache", required=True)
    p.add_argument("--cache_layers", default="0,5,11")
    p.add_argument("--eval_queries", type=int, default=150)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--only", default=None,
                   help="подстрока: гонять только совпадающие варианты")
    p.add_argument("--out", default=None)
    return p.parse_args()


def slice_cache(cache, cache_sources, wanted):
    pos = [cache_sources.index(s) for s in wanted]
    return StateCache(states=cache.states[:, mx.array(np.array(pos, np.int32))],
                      pair_index=cache.pair_index, labels=cache.labels)


def subset(cache, rows):
    a = mx.array(np.array(rows, np.int32))
    return StateCache(states=cache.states, pair_index=cache.pair_index[a],
                      labels=cache.labels[a])


def main():
    args = parse_args()
    base, cfg = load_pretrained(os.path.expanduser(args.model), verbose=False)
    cache = StateCache.load(os.path.expanduser(args.cache))
    cache_layers = sorted(set(resolve_layer_indices(
        [int(x) for x in args.cache_layers.split(",")], cfg.n_layer)))

    # тот же разрез, что в run_reranker.py: split_train_eval перемешивает
    # индексы Random(seed=0) и отдаёт первые n_eval в eval
    import random
    idx = list(range(cache.n_samples))
    random.Random(0).shuffle(idx)
    ev_rows, tr_rows = idx[:args.eval_queries], idx[args.eval_queries:]

    variants = [
        ("слои (11,) последний",        dict(layer_idx=(11,)),     mx.float32,  1.0),
        ("слои (0,)",                   dict(layer_idx=(0,)),      mx.float32,  1.0),
        ("слои (5,), голова fp32",      dict(layer_idx=(5,)),      mx.float32,  1.0),
        ("слои (5,), голова bf16",      dict(layer_idx=(5,)),      mx.bfloat16, 1.0),
        ("слои (0,5,11), голова fp32",  dict(layer_idx=(0, 5, 11)), mx.float32, 1.0),
        ("слои (0,5,11), голова bf16",  dict(layer_idx=(0, 5, 11)), mx.bfloat16, 1.0),
        ("слои (5,), 2 зонда",          dict(layer_idx=(5,), n_probe=2), mx.float32, 1.0),
        ("слои (5,), shared_state x3",  dict(layer_idx=(5, 5, 5), shared_state=True), mx.float32, 1.0),
        ("слои (5,), BCE (alpha=0)",    dict(layer_idx=(5,)),      mx.float32,  0.0),
        ("слои (5,), микс (alpha=0.7)", dict(layer_idx=(5,)),      mx.float32,  0.7),
    ]

    results = {}
    for name, kw, dt, alpha in variants:
        if args.only and args.only not in name:
            continue
        want = sorted(set(resolve_layer_indices(list(kw["layer_idx"]), cfg.n_layer)))
        if any(l not in cache_layers for l in want):
            print(f"пропуск «{name}»: слоёв нет в кэше {cache_layers}")
            continue
        sub = slice_cache(cache, cache_layers, want)
        tr, ev = subset(sub, tr_rows), subset(sub, ev_rows)

        runs = []
        for seed in args.seeds:
            mx.random.seed(seed)
            model = Reranker(base, RerankerConfig(**kw), head_dtype=dt)
            train_reranker(model, tr, ev, RerankTrainConfig(
                lr=args.lr, batch_size=args.batch_size, epochs=args.epochs,
                loss_alpha=alpha, log_every=0, seed=seed,
                checkpoint_path="/tmp/_ablate_head.safetensors"))
            runs.append(evaluate(model.head, ev))
        mrr = np.array([r["mrr"] for r in runs])
        r1 = np.array([r["recall@1"] for r in runs])
        results[name] = {"mrr_mean": float(mrr.mean()), "mrr_std": float(mrr.std()),
                         "r1_mean": float(r1.mean()), "r1_std": float(r1.std()),
                         "seeds": args.seeds, "runs": runs}
        print(f"\n>>> {name}: MRR {mrr.mean():.4f} ± {mrr.std():.4f} | "
              f"R@1 {r1.mean():.4f} ± {r1.std():.4f}\n", flush=True)

    print("\n" + "=" * 72)
    print(f"{'вариант':>30} | {'MRR':>15} | {'R@1':>15}")
    print("-" * 72)
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["mrr_mean"]):
        print(f"{name:>30} | {r['mrr_mean']:.4f} ± {r['mrr_std']:.4f} | "
              f"{r['r1_mean']:.4f} ± {r['r1_std']:.4f}")
    print("=" * 72)
    print(f"{len(args.seeds)} семени на вариант; ± это разброс по семенам, "
          "а не доверительный интервал")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"отчёт: {args.out}")


if __name__ == "__main__":
    main()
