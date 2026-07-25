"""
Проверка внешнего чекпоинта (load_local_rwkv7) перплексией.

Имена и формы тензоров у `RWKV7` и `RWKV7X070` совпадают, поэтому чекпоинт
загрузится в обе архитектуры без единой ошибки — и молча даст испорченный
выход в неправильной. Единственный надёжный арбитр — перплексия на тексте
того языка, на котором модель обучалась:

    случайные веса   -> loss = ln(vocab_size)
    неверная арх.    -> заметно лучше случайного (веса-то настоящие), но плохо
    верная арх.      -> резко ниже

Usage:
    python tools/verify_local_checkpoint.py \
        --model_dir /Users/s/Develop/WKV-kvant/ru60m \
        --data /Users/s/Develop/retrieval_literature/train.jsonl
"""
import argparse
import json
import math
import re

import mlx.core as mx
import mlx.nn as nn
import rwkv_metal as rk

CYR = re.compile(r"[а-яёА-ЯЁ]")
LAT = re.compile(r"[a-zA-Z]")


def collect_texts(path, n=40, min_len=600, max_len=1200, lang="ru", scan=20000):
    """Набирает n одноязычных отрывков из triplet-jsonl."""
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= scan or len(out) >= n:
                break
            s = json.loads(line).get("positive", "")
            if len(s) < min_len:
                continue
            ru = len(CYR.findall(s)) > 3 * len(LAT.findall(s))
            if (lang == "ru") == ru:
                out.append(s[:max_len])
    return out


def mean_loss(model, tok, texts, ctx=512):
    """Средняя кросс-энтропия следующего токена, в натах."""
    tot, ntok = 0.0, 0
    for t in texts:
        ids = tok.encode(t)[:ctx]
        if len(ids) < 16:
            continue
        logits = model(mx.array(ids)[None, :])[0, :-1].astype(mx.float32)
        l = nn.losses.cross_entropy(logits, mx.array(ids[1:])).sum()
        mx.eval(l)
        tot += float(l.item())
        ntok += len(ids) - 1
    return tot / ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="/Users/s/Develop/WKV-kvant/ru60m")
    ap.add_argument("--data", default="/Users/s/Develop/retrieval_literature/train.jsonl")
    ap.add_argument("--lang", default="ru", choices=["ru", "en"])
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    tok = rk.load_local_tokenizer(args.model_dir)
    texts = collect_texts(args.data, n=args.n, lang=args.lang)
    print(f"{len(texts)} отрывков, язык={args.lang}\n")

    results = {}
    for arch in ("x070", "scratch"):
        try:
            model, cfg = rk.load_local_rwkv7(args.model_dir, arch=arch, verbose=False)
        except ValueError as e:
            # RWKV7 жёстко зашивает ранг 64, так что чекпоинт с другими
            # рангами в него просто не встанет — это само по себе ответ.
            print(f"  {arch:8s}: не грузится ({str(e).splitlines()[0]})")
            continue
        loss = mean_loss(model, tok, texts, ctx=cfg.ctx_len)
        results[arch] = loss
        print(f"  {arch:8s}: loss {loss:.4f} nats/token | PPL {math.exp(loss):9.1f}")
        del model

    with open(f"{args.model_dir}/config.json", encoding="utf-8") as f:
        rnd = math.log(json.load(f)["vocab_size"])
    print(f"  {'random':8s}: loss {rnd:.4f} nats/token | PPL {math.exp(rnd):9.1f}")

    best = min(results, key=results.get)
    others = [a for a in results if a != best]
    if others:
        print(f"\n  -> верная архитектура: {best} "
              f"(в {math.exp(results[others[0]] - results[best]):.1f}x лучше по PPL, "
              f"чем {others[0]})")
    else:
        print(f"\n  -> архитектура: {best} "
              f"(в {math.exp(rnd - results[best]):.0f}x лучше случайной)")
    if results[best] > rnd - 2.0:
        print("  -> ВНИМАНИЕ: всё близко к случайному — чекпоинт либо не обучен, "
              "либо раскладка весов не та")


if __name__ == "__main__":
    main()
