"""
Оценка сохранённого embedding-чекпоинта на том же held-out срезе, который
использовал прогон.

Зачем отдельный инструмент: `finetune_embedding` сохраняет чекпоинт каждые
`save_every` шагов, но `eval AFTER` считается только по завершении стадии.
Если прогон остановлен посреди стадии (кончилось время, перегрев, Ctrl-C),
веса есть, а числа к ним — нет.

Срез восстанавливается детерминированно: те же `--seed`, `--lang`,
`--n_per_task`, `--n_eval` и тот же файл данных дают ровно тот же reservoir
и то же разбиение train/eval, что и в прогоне. Значения берутся из
`run.json`, так что руками их указывать не нужно.

Usage:
    python tools/eval_embedding_checkpoint.py \\
        --run_dir runs/ru60m_curriculum \\
        --checkpoint runs/ru60m_curriculum/embed_retrieval.safetensors \\
        --stage retrieval
"""
import argparse
import json
import os
import sys

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

import rwkv_metal as rk
from rwkv_metal.embedding import (
    EmbeddingModel,
    evaluate_retrieval, evaluate_sts_pairwise, evaluate_classification,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_embedding_curriculum import sample_all_tasks, split_train_eval, load_base, ALL_TASKS

EVALS = {
    "retrieval": evaluate_retrieval,
    "sts": evaluate_sts_pairwise,
    "classification": evaluate_classification,
}


def load_embedding_checkpoint(model: EmbeddingModel, path: str, verbose: bool = True):
    """Чекпоинт содержит ТОЛЬКО обучаемые параметры (при full-FT это вся
    модель, при LoRA — только адаптеры и голова). Поэтому проверяем, что
    каждый ключ из файла существует в модели и совпадает по форме, но НЕ
    требуем полноты: отсутствующие ключи легитимны и остаются от базы."""
    saved = dict(mx.load(path))
    params = dict(tree_flatten(model.parameters()))
    unknown = sorted(set(saved) - set(params))
    mismatch = [(k, tuple(saved[k].shape), tuple(params[k].shape))
                for k in sorted(set(saved) & set(params))
                if tuple(saved[k].shape) != tuple(params[k].shape)]
    if unknown or mismatch:
        raise ValueError(f"чекпоинт не подходит к модели:\n"
                         f"  неизвестные ключи: {unknown[:10]}\n"
                         f"  формы не сходятся: {mismatch[:10]}")
    if verbose:
        n = sum(v.size for v in saved.values())
        frac = 100 * len(saved) / len(params)
        print(f"  загружено {len(saved)}/{len(params)} тензоров ({frac:.0f}%), "
              f"{n/1e6:.2f}M параметров из {path}")
    model.update(tree_unflatten(list(saved.items())))
    mx.eval(model.parameters())
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="каталог с run.json")
    ap.add_argument("--checkpoint", default=None,
                    help="по умолчанию embed_<stage>.safetensors внутри run_dir")
    ap.add_argument("--stage", required=True, choices=list(ALL_TASKS))
    ap.add_argument("--baseline", action="store_true",
                    help="дополнительно померить необученную базу (контроль)")
    args = ap.parse_args()

    run = json.load(open(os.path.join(args.run_dir, "run.json"), encoding="utf-8"))
    cfg = run["config"]
    ckpt = args.checkpoint or os.path.join(args.run_dir, f"embed_{args.stage}.safetensors")

    stages = [t.strip() for t in cfg["stages"].split(",") if t.strip()]
    if args.stage not in stages:
        raise SystemExit(f"стадии {args.stage!r} не было в прогоне (были: {stages})")

    print("=== база ===", flush=True)
    base, tok, terminator, label = load_base(cfg["model"])
    model = EmbeddingModel(base)
    print(f"  {label} | terminator {terminator}", flush=True)

    print(f"\n=== восстановление held-out среза (seed={cfg['seed']}, lang={cfg['lang']}) ===", flush=True)
    pools = sample_all_tasks(cfg["data"], cfg["n_per_task"] + cfg["n_eval"],
                             tasks=stages, lang=cfg["lang"], seed=cfg["seed"])
    _, eval_rows = split_train_eval(pools[args.stage], cfg["n_eval"],
                                    seed=cfg["seed"] + stages.index(args.stage))
    print(f"  {args.stage}: {len(eval_rows)} held-out строк", flush=True)

    eval_fn = EVALS[args.stage]
    kw = {"max_chars": cfg["max_chars"], "terminator": terminator}
    if args.stage == "classification":
        kw["sample_size"] = cfg["cls_eval_sample"]

    recorded = next((s.get("eval_before") for s in run["stages"]
                     if s["stage"] == args.stage), None)

    if args.baseline:
        print("\n=== контроль: необученная база ===", flush=True)
        print("  ", eval_fn(model, tok, eval_rows, **kw), flush=True)

    print(f"\n=== чекпоинт ===", flush=True)
    load_embedding_checkpoint(model, ckpt)
    after = eval_fn(model, tok, eval_rows, **kw)

    print("\n" + "=" * 64)
    print(f"стадия {args.stage}: {len(eval_rows)} held-out строк"
          + (f", пул кандидатов {2*len(eval_rows)}" if args.stage in ("retrieval",) else ""))
    print("=" * 64)
    if recorded:
        keys = [k for k in after if k != "n"]
        w = max(len(k) for k in keys)
        print(f"  {'метрика':<{w}}  {'до':>10}  {'после':>10}   изменение")
        for k in keys:
            b, a = recorded.get(k), after[k]
            if isinstance(b, (int, float)) and isinstance(a, (int, float)):
                delta = f"x{a/b:.1f}" if b else "—"
                print(f"  {k:<{w}}  {b:>10.4f}  {a:>10.4f}   {delta:>8}")
    else:
        print(f"  {after}")

    out = os.path.join(args.run_dir, f"eval_{args.stage}_from_checkpoint.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"stage": args.stage, "checkpoint": ckpt,
                   "eval_before": recorded, "eval_after": after,
                   "n_eval": len(eval_rows)}, f, indent=2, ensure_ascii=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
