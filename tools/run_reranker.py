"""
Полноценный прогон реранкера на LitRetrieval.

Что делает:
  1. читает N строк задачи retrieval (потоково, reservoir sampling);
  2. собирает кандидатов: позитив, hard-негатив из датасета, остальные —
     случайные документы из общего пула;
  3. один раз сворачивает все пары в состояния (кэш префиксов документов —
     см. rwkv_metal/reranker/encode.py) и держит только те слои базы,
     которые читает голова;
  4. считает базовые линии на ТЕХ ЖЕ наборах кандидатов: случайное
     угадывание и эмбеддер на той же (недообученной) базе;
  5. обучает голову — при желании несколько конфигураций слоёв из одного
     кэша — и меряет held-out;
  6. пишет всё в JSON после каждого этапа.

Пример:
    .venv/bin/python tools/run_reranker.py \
        --model ~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth \
        --data  ~/Develop/retrieval_literature/train.jsonl \
        --queries 1000 --candidates 8 --eval_queries 150 \
        --cache_layers 0,5,11 --configs -1 5 0,5,11 \
        --out runs/reranker_0.1b
"""
import argparse
import json
import os
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwkv_metal.model import load_pretrained
from rwkv_metal.tokenizer import WorldTokenizer
from rwkv_metal.reranker import (
    Reranker, RerankerConfig, RerankerInference, RerankTrainConfig,
    PairTemplate, StateCache, build_candidates, encode_pairs, evaluate,
    load_rows, resolve_layer_indices, split_train_eval, train_reranker,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth")
    p.add_argument("--data", default="~/Develop/retrieval_literature/train.jsonl")
    p.add_argument("--task", default="retrieval")
    p.add_argument("--lang", default=None, help="ru | en | None (все)")
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--eval_queries", type=int, default=150)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--max_doc_tokens", type=int, default=512)
    p.add_argument("--max_query_tokens", type=int, default=96)
    p.add_argument("--doc_batch", type=int, default=8)
    p.add_argument("--query_batch", type=int, default=16)
    p.add_argument("--cache_layers", default="0,5,11",
                   help="слои базы, состояния которых кладутся в кэш")
    p.add_argument("--configs", nargs="+", default=["-1"],
                   help="конфигурации головы: списки слоёв через запятую")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--loss_alpha", type=float, default=1.0)
    p.add_argument("--n_probe", type=int, default=1)
    p.add_argument("--skip_embedder_baseline", action="store_true")
    p.add_argument("--cache_path", default=None,
                   help="файл кэша состояний: есть — загрузить, нет — посчитать "
                        "и сохранить. Позволяет подбирать гиперпараметры, не "
                        "пересчитывая базу.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/reranker")
    return p.parse_args()


def slice_cache(cache: StateCache, cache_sources, wanted) -> StateCache:
    """Взять из кэша только нужные конфигурации слои (ось источников)."""
    pos = [cache_sources.index(s) for s in wanted]
    return StateCache(states=cache.states[:, mx.array(np.array(pos, np.int32))],
                      pair_index=cache.pair_index, labels=cache.labels)


def subset(cache: StateCache, rows) -> StateCache:
    a = mx.array(np.array(rows, np.int32))
    return StateCache(states=cache.states, pair_index=cache.pair_index[a],
                      labels=cache.labels[a])


def ranking_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    gold = scores[np.arange(len(labels)), labels]
    greater = (scores > gold[:, None]).sum(axis=1)
    ties = (scores == gold[:, None]).sum(axis=1) - 1
    ranks = 1 + greater + ties / 2.0
    ri = np.ceil(ranks)
    return {
        "mrr": float((1.0 / ranks).mean()),
        "recall@1": float((ri <= 1).mean()),
        "recall@3": float((ri <= 3).mean()),
        "recall@5": float((ri <= 5).mean()),
        "ndcg@10": float((1.0 / np.log2(ranks + 1) * (ranks <= 10)).mean()),
        "n": int(len(labels)),
    }


def hard_negative_breakdown(scores: np.ndarray, samples) -> dict:
    """Разложение общей метрики на «против hard-негатива» и «против случайных».

    MRR по восьми кандидатам, где семь взяты из пула наугад, льстит модели:
    отличить пассаж про пчёл от пассажа про паровые машины несложно. Здесь
    отдельно считается попарная точность против МАЙНЕННОГО hard-негатива —
    это та часть задачи, ради которой реранкер вообще нужен.
    """
    hard_ok, hard_n, easy_ok, easy_n = 0, 0, 0, 0
    for i, s in enumerate(samples):
        gold = scores[i, s.label]
        if s.hard_neg is not None:
            hard_n += 1
            hard_ok += int(gold > scores[i, s.hard_neg])
        for j in range(scores.shape[1]):
            if j == s.label or j == s.hard_neg:
                continue
            easy_n += 1
            easy_ok += int(gold > scores[i, j])
    return {
        "pairwise_vs_hard_negative": hard_ok / max(1, hard_n),
        "n_hard": hard_n,
        "pairwise_vs_sampled_negative": easy_ok / max(1, easy_n),
        "n_sampled": easy_n,
    }


def head_scores(head, cache: StateCache) -> np.ndarray:
    from rwkv_metal.reranker import batch_scores
    out = []
    for start in range(0, cache.n_samples, 64):
        rows = mx.arange(start, min(start + 64, cache.n_samples))
        s = batch_scores(head, cache, rows)
        mx.eval(s)
        out.append(np.array(s.astype(mx.float32)))
    return np.concatenate(out, axis=0)


def embedder_baseline(base, tok, pool, samples) -> dict:
    """Косинусная близость эмбеддингов запроса и документа на ТОЙ ЖЕ базе.

    Это не дообученный эмбеддер, а сырой претрейн — то, с чего стартует и
    сам реранкер. Сравнение честное: одна и та же база, одни и те же
    кандидаты; разница показывает ровно то, что добавляет голова.
    """
    from rwkv_metal.embedding import Embedder
    emb = Embedder(base, tok)
    t0 = time.time()
    doc_vecs = {}
    need = sorted({d for s in samples for d in s.doc_ids})
    for i, did in enumerate(need):
        doc_vecs[did] = emb.embed_one(pool[did])
        if i % 200 == 0:
            print(f"    эмбеддинги документов {i}/{len(need)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    scores, labels = [], []
    for s in samples:
        q = emb.embed_one(s.query)
        row = [float((q * doc_vecs[d]).sum()) for d in s.doc_ids]
        scores.append(row)
        labels.append(s.label)
    scores = np.array(scores)
    m = ranking_metrics(scores, np.array(labels))
    m.update(hard_negative_breakdown(scores, samples))
    return m


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    report_path = os.path.join(args.out, "report.json")
    report = {"args": vars(args), "stages": {}}

    def flush():
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    t_start = time.time()
    # инициализация головы (зонд, MLP) идёт через mx.random — без фиксации
    # семени два прогона с одинаковыми флагами расходятся по held-out MRR
    # примерно на ±0.03
    mx.random.seed(args.seed)

    # ── база ─────────────────────────────────────────────────────────────
    t0 = time.time()
    base, cfg = load_pretrained(os.path.expanduser(args.model), verbose=False)
    tok = WorldTokenizer()
    print(f"база: L={cfg.n_layer} D={cfg.n_embd} H={cfg.n_head} "
          f"({time.time()-t0:.1f}s)")
    report["base"] = {"n_layer": cfg.n_layer, "n_embd": cfg.n_embd,
                      "n_head": cfg.n_head, "path": args.model}

    # ── данные ───────────────────────────────────────────────────────────
    t0 = time.time()
    rows = load_rows(os.path.expanduser(args.data), task=args.task,
                     limit=args.queries, seed=args.seed, lang=args.lang)
    pool, samples = build_candidates(rows, n_candidates=args.candidates,
                                     seed=args.seed)
    train_s, eval_s = split_train_eval(samples, args.eval_queries, seed=args.seed)
    print(f"данные: {len(samples)} запросов, пул {len(pool)} документов, "
          f"train {len(train_s)} / eval {len(eval_s)} ({time.time()-t0:.0f}s)")
    report["data"] = {"n_samples": len(samples), "pool": len(pool),
                      "train": len(train_s), "eval": len(eval_s),
                      "candidates": args.candidates,
                      "seconds": time.time() - t0}
    flush()

    # ── базовые линии ────────────────────────────────────────────────────
    rand_mrr = 2.0 / (args.candidates + 1)
    report["stages"]["random"] = {"mrr": rand_mrr,
                                  "recall@1": 1.0 / args.candidates}
    print(f"случайное угадывание: MRR {rand_mrr:.3f}")

    if not args.skip_embedder_baseline:
        print("эмбеддер на той же базе (базовая линия)...")
        t0 = time.time()
        m = embedder_baseline(base, tok, pool, eval_s)
        m["seconds"] = time.time() - t0
        report["stages"]["embedder"] = m
        print(f"  эмбеддер: MRR {m['mrr']:.3f} R@1 {m['recall@1']:.3f} "
              f"nDCG@10 {m['ndcg@10']:.3f} ({m['seconds']:.0f}s)")
        flush()

    # ── кэш состояний ────────────────────────────────────────────────────
    cache_layers = resolve_layer_indices(
        [int(x) for x in args.cache_layers.split(",")], cfg.n_layer)
    cache_layers = sorted(set(cache_layers))
    probe = Reranker(base, RerankerConfig(layer_idx=tuple(cache_layers),
                                          n_probe=args.n_probe))
    t0 = time.time()
    if args.cache_path and os.path.exists(os.path.expanduser(args.cache_path)):
        print(f"кэш состояний загружается из {args.cache_path}")
        cache = StateCache.load(os.path.expanduser(args.cache_path))
        if cache.n_samples != len(samples) or cache.n_cand != args.candidates:
            raise SystemExit(
                f"кэш не соответствует данным: в кэше {cache.n_samples}×"
                f"{cache.n_cand}, ожидается {len(samples)}×{args.candidates}. "
                "Совпадать должны --queries/--candidates/--seed/--lang/--task."
            )
    else:
        print(f"кэш состояний по слоям {cache_layers}...")
        cache = encode_pairs(probe, tok, pool, samples, template=PairTemplate(),
                             max_doc_tokens=args.max_doc_tokens,
                             max_query_tokens=args.max_query_tokens,
                             doc_batch=args.doc_batch,
                             query_batch=args.query_batch,
                             dtype=mx.bfloat16, verbose=True)
        if args.cache_path:
            cache.save(os.path.expanduser(args.cache_path))
            print(f"кэш сохранён в {args.cache_path}")
    report["stages"]["encode"] = {
        "seconds": time.time() - t0, "pairs": cache.n_pairs,
        "layers": cache_layers, "gb": cache.nbytes() / 1e9,
        "peak_gb": mx.get_peak_memory() / 1e9,
    }
    print(f"кэш готов: {cache.n_pairs} пар, {cache.nbytes()/1e9:.2f} ГБ, "
          f"{time.time()-t0:.0f}s")
    flush()

    # индексы train/eval внутри общего списка samples
    pos = {id(s): i for i, s in enumerate(samples)}
    tr_rows = [pos[id(s)] for s in train_s]
    ev_rows = [pos[id(s)] for s in eval_s]

    # ── конфигурации головы ──────────────────────────────────────────────
    report["stages"]["configs"] = {}
    for spec in args.configs:
        want = resolve_layer_indices([int(x) for x in spec.split(",")], cfg.n_layer)
        if any(l not in cache_layers for l in want):
            print(f"пропуск конфигурации {spec}: слоёв нет в кэше {cache_layers}")
            continue
        print(f"\n=== конфигурация слоёв {spec} → {want} ===")
        model = Reranker(base, RerankerConfig(layer_idx=tuple(want),
                                              n_probe=args.n_probe))
        sub = slice_cache(cache, cache_layers, sorted(set(want)))
        tr, ev = subset(sub, tr_rows), subset(sub, ev_rows)

        before = evaluate(model.head, ev)
        ckpt = os.path.join(args.out, f"head_{spec.replace(',', '_')}.safetensors")
        # контракт подачи текста пишется в чекпоинт вместе с весами: он такая
        # же часть обученной модели, как и веса, и расходится так же молча
        serving = RerankerInference(
            model, tok, max_doc_tokens=args.max_doc_tokens,
            max_query_tokens=args.max_query_tokens).serving_metadata()
        res = train_reranker(model, tr, ev, RerankTrainConfig(
            lr=args.lr, batch_size=args.batch_size, epochs=args.epochs,
            loss_alpha=args.loss_alpha, checkpoint_path=ckpt, seed=args.seed),
            save_extra=serving)
        after = evaluate(model.head, ev)
        after.update(hard_negative_breakdown(head_scores(model.head, ev), eval_s))
        report["stages"]["configs"][spec] = {
            "layers": want, "before": before, "after": after,
            "best": res["best"], "seconds": res["seconds"],
            "checkpoint": ckpt,
            "history": res["history"][::max(1, len(res["history"]) // 200)],
        }
        print(f"итог {spec}: MRR {before['mrr']:.3f} → {after['mrr']:.3f} | "
              f"R@1 {before['recall@1']:.3f} → {after['recall@1']:.3f} | "
              f"nDCG@10 {before['ndcg@10']:.3f} → {after['ndcg@10']:.3f}")
        flush()

    report["total_seconds"] = time.time() - t_start
    report["peak_gb"] = mx.get_peak_memory() / 1e9
    flush()

    print("\n" + "=" * 80)
    print(f"{'конфигурация':>16} | {'MRR':>6} | {'R@1':>6} | {'nDCG@10':>7} | "
          f"{'vs hard':>7} | {'vs случ.':>8}")
    print("-" * 80)
    print(f"{'случайно':>16} | {rand_mrr:6.3f} | {1/args.candidates:6.3f} | "
          f"{'—':>7} | {0.5:7.3f} | {0.5:8.3f}")
    if "embedder" in report["stages"]:
        e = report["stages"]["embedder"]
        print(f"{'эмбеддер (сырой)':>16} | {e['mrr']:6.3f} | {e['recall@1']:6.3f} | "
              f"{e['ndcg@10']:7.3f} | {e['pairwise_vs_hard_negative']:7.3f} | "
              f"{e['pairwise_vs_sampled_negative']:8.3f}")
    for spec, r in report["stages"]["configs"].items():
        a = r["after"]
        print(f"{'реранкер ' + spec:>16} | {a['mrr']:6.3f} | {a['recall@1']:6.3f} | "
              f"{a['ndcg@10']:7.3f} | {a['pairwise_vs_hard_negative']:7.3f} | "
              f"{a['pairwise_vs_sampled_negative']:8.3f}")
    print("=" * 80)
    print("vs hard — попарная точность против майненного hard-негатива; "
          "vs случ. — против случайного документа пула")
    print(f"всего {report['total_seconds']/60:.1f} мин, пик {report['peak_gb']:.2f} ГБ")
    print(f"отчёт: {report_path}")


if __name__ == "__main__":
    main()
