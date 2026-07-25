"""
Смоук-тест реранкера: маленькая модель, синтетические данные, вся цепочка.

Что проверяется:
  1. Голова инициализируется из базы и на старте выдаёт РОВНО нули
     (zero-init), значит listwise-лосс равен ln(C).
  2. Кэшированный путь (encode_pairs, через состояния префиксов) даёт те же
     состояния, что сплошной проход (encode_pairs_direct). Это главный
     инвариант всей конструкции.
  3. Обучение действительно учит: на выучиваемой синтетике лосс падает,
     MRR растёт.
  4. Инференс: score/rank и индексный путь совпадают между собой.
  5. Сохранение и загрузка головы.

Запуск: .venv/bin/python tools/test_reranker_smoke.py
"""
import os
import sys
import tempfile

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlx.utils import tree_map

from rwkv_metal.model.rwkv7_x070 import RWKV7X070
from rwkv_metal.pretrain.config import PretrainConfig
from rwkv_metal.reranker import (
    Reranker, RerankerConfig, RerankerInference,
    PairTemplate, RerankSample, encode_pairs, encode_pairs_direct,
    RerankTrainConfig, train_reranker, evaluate, listwise_loss, batch_scores,
)

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok    {name} {detail}")
    else:
        FAILED.append(name)
        print(f"  FAIL  {name} {detail}")


class ToyTokenizer:
    """Байтовый токенизатор: детерминированный, без внешних файлов."""
    def __init__(self, vocab=512):
        self.vocab = vocab

    def encode(self, s: str):
        return [1 + (b % (self.vocab - 2)) for b in s.encode("utf-8")]


def tiny_base(seed=0, n_layer=3, n_embd=128, vocab=512):
    """Крошечная база со ЗДОРОВОЙ памятью.

    У случайно инициализированной RWKV-7 затухание w ≈ exp(-0.6·sigmoid(0)) ≈
    0.74 на шаг: за двадцать токенов хвоста «Query: ...» вклад документа
    падает в тысячу раз, состояния всех кандидатов схлопываются, и проверки
    начинают проходить вхолостую на нулевом сигнале. Уводим bias decay-LoRA в
    -6 (sigmoid ≈ 0.002, w ≈ 0.9985) — получается игрушечная, но помнящая
    база. У обученной 0.1B это не нужно: у неё память длинная сама по себе
    (замерено: cos между состояниями двух разных документов не меняется даже
    после 244 токенов хвоста).
    """
    mx.random.seed(seed)
    cfg = PretrainConfig(n_layer=n_layer, n_embd=n_embd, vocab_size=vocab)
    m = RWKV7X070(cfg)
    m.update(tree_map(lambda p: p + mx.random.normal(p.shape) * 0.03, m.parameters()))
    for blk in m.blocks:
        blk.tmix.w_lora_B.bias = mx.full(blk.tmix.w_lora_B.bias.shape, -6.0)
    mx.eval(m.parameters())
    return m, cfg


def make_synthetic(n_queries=48, n_cand=4, seed=0):
    """Синтетика для СТРУКТУРНЫХ проверок (не для проверки качества).

    Случайно инициализированная крошечная база не умеет ничего, поэтому
    требовать от неё осмысленного ранжирования бессмысленно. Здесь важно
    другое: чтобы документы были короткими (влезали в max_doc_tokens
    целиком) и различались — иначе состояния совпадут и проверки пройдут
    вхолостую на нулях.
    """
    rng = np.random.default_rng(seed)
    words = [f"тема{i}" for i in range(n_queries)]
    filler = "текст про"
    pool, samples = [], []
    for i in range(n_queries):
        base = len(pool)
        pool.append(f"{filler} {words[i]}")
        for j in range(n_cand - 1):
            other = words[(i + 1 + j) % n_queries]
            pool.append(f"{filler} {other}")
        cand = list(range(base, base + n_cand))
        order = list(rng.permutation(n_cand))
        shuffled = [cand[k] for k in order]
        samples.append(RerankSample(
            instruct="Найди документ по теме",
            query=f"нужен документ про {words[i]}",
            doc_ids=shuffled,
            label=shuffled.index(base),
        ))
    return pool, samples


def make_random_state_cache(n_samples=64, n_cand=4, n_src=1, H=2, S=64, seed=0):
    """Кэш состояний, собранный НАПРЯМУЮ, минуя базу.

    Проверяет ровно одно: умеет ли голова учиться отделять одно состояние от
    других. Правильный кандидат помечен фиксированным сдвигом, общим для всех
    примеров, — выучиваемо, но не тривиально (сдвиг мал по сравнению с шумом).
    """
    from rwkv_metal.reranker.encode import StateCache
    rng = np.random.default_rng(seed)
    n_pairs = n_samples * n_cand
    states = rng.standard_normal((n_pairs, n_src, H, S, S)).astype(np.float32) * 0.5
    mark = rng.standard_normal((n_src, H, S, S)).astype(np.float32) * 0.5
    labels = rng.integers(0, n_cand, size=n_samples).astype(np.int32)
    pair_index = np.arange(n_pairs, dtype=np.int32).reshape(n_samples, n_cand)
    for si in range(n_samples):
        states[pair_index[si, labels[si]]] += mark
    return StateCache(states=mx.array(states), pair_index=mx.array(pair_index),
                      labels=mx.array(labels))


def main():
    base, cfg = tiny_base()
    model = Reranker(base, RerankerConfig(layer_idx=(-1,)))
    tok = ToyTokenizer(vocab=cfg.vocab_size)
    pool, samples = make_synthetic()

    # ── 1. zero-init ─────────────────────────────────────────────────────
    cache = encode_pairs(model, tok, pool, samples, max_doc_tokens=96,
                         max_query_tokens=48, doc_batch=8, query_batch=16,
                         dtype=mx.float32, verbose=False)
    rows = mx.arange(8)
    s0 = batch_scores(model.head, cache, rows)
    mx.eval(s0)
    check("голова zero-init даёт нулевые скоры",
          float(mx.abs(s0).max()) < 1e-6, f"max|score| = {float(mx.abs(s0).max()):.2e}")
    l0 = float(listwise_loss(s0, cache.labels[rows]))
    check("стартовый лосс == ln(C)", abs(l0 - np.log(cache.n_cand)) < 1e-5,
          f"{l0:.6f} против {np.log(cache.n_cand):.6f}")

    # ── 1b. состояния кандидатов вообще различаются ──────────────────────
    # Защита от «вакуумного» прохождения остальных проверок: если документы
    # после обрезки совпадают, состояния совпадут, градиент занулится, а
    # сравнения кэша со сплошным проходом пройдут на нулях.
    pair0 = cache.pair_index[0]
    spread = float(mx.abs(cache.states[pair0[0]] - cache.states[pair0[1]]).max())
    check("состояния разных кандидатов различаются", spread > 1e-3,
          f"max|Δ| = {spread:.3e}")

    # ── 2. кэш префиксов == сплошной проход (оба порядка шаблона) ────────
    direct_pairs = []
    for s in samples[:6]:
        for did in s.doc_ids:
            direct_pairs.append((s.instruct, pool[did], s.query))

    for name, tmpl, cch in (
        ("документ первым", PairTemplate(doc_first=True), cache),
        ("запрос первым", PairTemplate(doc_first=False),
         encode_pairs(model, tok, pool, samples, template=PairTemplate(doc_first=False),
                      max_doc_tokens=96, max_query_tokens=48, doc_batch=8,
                      query_batch=16, dtype=mx.float32, verbose=False)),
    ):
        direct = encode_pairs_direct(model, tok, direct_pairs, template=tmpl,
                                     max_doc_tokens=96, max_query_tokens=48,
                                     batch_size=8)
        cached = cch.states[cch.pair_index[:6].reshape(-1)]
        mx.eval(direct, cached)
        d = float(mx.abs(direct - cached).max())
        scale = float(mx.abs(direct).max())
        check(f"кэш == сплошной проход ({name})", d / max(1e-9, scale) < 1e-4,
              f"отн. расхождение {d/max(1e-9,scale):.2e}")

    # ── 3. голова обучается ──────────────────────────────────────────────
    # На состояниях, собранных напрямую: проверяется обучаемость головы, а не
    # осведомлённость случайно инициализированной базы.
    from rwkv_metal.reranker.encode import StateCache
    full = make_random_state_cache(n_samples=96, n_cand=4,
                                   n_src=len(model.head.unique_sources),
                                   H=cfg.n_head, S=cfg.head_size)
    def subcache(c, idx):
        a = mx.array(np.array(idx, np.int32))
        return StateCache(states=c.states, pair_index=c.pair_index[a],
                          labels=c.labels[a])
    tr = subcache(full, list(range(0, 80)))
    ev = subcache(full, list(range(80, 96)))

    before = evaluate(model.head, ev)
    # все скоры равны → все кандидаты в связке → средний ранг (C+1)/2
    expected = 2.0 / (ev.n_cand + 1)
    check("необученная голова ~ случайное угадывание",
          abs(before["mrr"] - expected) < 0.02,
          f"MRR {before['mrr']:.3f} против ожидаемого {expected:.3f}")

    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, "head.safetensors")
        res = train_reranker(model, tr, ev, RerankTrainConfig(
            lr=1e-3, batch_size=16, epochs=30, log_every=0,
            checkpoint_path=ckpt, keep_best=False))
        after = evaluate(model.head, ev)
        check("лосс падает", res["history"][-1]["loss"] < res["history"][0]["loss"] - 0.05,
              f"{res['history'][0]['loss']:.4f} → {res['history'][-1]['loss']:.4f}")
        check("MRR растёт", after["mrr"] > before["mrr"] + 0.05,
              f"{before['mrr']:.3f} → {after['mrr']:.3f}")

        # ── 5. сохранение/загрузка ───────────────────────────────────────
        s_before = batch_scores(model.head, ev, mx.arange(4))
        mx.eval(s_before)
        model2 = Reranker(base, RerankerConfig(layer_idx=(-1,)))
        model2.load_head(ckpt)
        s_after = batch_scores(model2.head, ev, mx.arange(4))
        mx.eval(s_after)
        check("голова сохраняется и грузится",
              float(mx.abs(s_before - s_after).max()) < 1e-5,
              f"расхождение {float(mx.abs(s_before - s_after).max()):.2e}")

        # ── 5b. чужая конфигурация ловится, своя восстанавливается ───────
        # Голова над слоем 1 и над слоем 2 совпадают по формам всех тензоров,
        # поэтому без проверки метаданных подмена прошла бы молча.
        wrong = Reranker(base, RerankerConfig(layer_idx=(1,)))
        try:
            wrong.load_head(ckpt)
            check("несовпадающая конфигурация отвергается", False,
                  "загрузилась молча")
        except ValueError:
            check("несовпадающая конфигурация отвергается", True)

        model3 = Reranker.from_head(base, ckpt)
        s3 = batch_scores(model3.head, ev, mx.arange(4))
        mx.eval(s3)
        check("from_head восстанавливает конфигурацию",
              model3.head.layer_idx == model.head.layer_idx
              and float(mx.abs(s_before - s3).max()) < 1e-5,
              f"слои {model3.head.layer_idx}")

    # ── 4. инференс: прямой путь == индексный ────────────────────────────
    rr = RerankerInference(model, tok, instruct="Найди документ по теме",
                           max_doc_tokens=96, max_query_tokens=48)
    q = samples[0].query
    docs = [pool[d] for d in samples[0].doc_ids]
    s_direct = rr.score(q, docs)
    index = rr.build_index(docs, instruct="Найди документ по теме")
    s_indexed = rr.score_indexed(q, index)
    check("score == score_indexed",
          np.abs(s_direct - s_indexed).max() < 1e-3,
          f"расхождение {np.abs(s_direct - s_indexed).max():.2e}")
    ranked = rr.rank(q, docs)
    check("rank сортирует по убыванию",
          all(ranked[i][1] >= ranked[i + 1][1] for i in range(len(ranked) - 1)))
    check("скоры различаются между документами",
          float(np.std(s_direct)) > 1e-6, f"std = {float(np.std(s_direct)):.3e}")

    # ── 4b. индекс с чужой инструкцией отвергается ───────────────────────
    rr_other = RerankerInference(model, tok, instruct="Совсем другая задача",
                                 max_doc_tokens=96, max_query_tokens=48)
    try:
        rr_other.score_indexed(q, index)
        check("индекс с чужой инструкцией отвергается", False, "прошло молча")
    except ValueError:
        check("индекс с чужой инструкцией отвергается", True)

    print()
    print(f"{'ПРОВАЛЕНО: ' + ', '.join(FAILED) if FAILED else 'все проверки пройдены'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
