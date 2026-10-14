"""ГЕЙТ ОЦЕНКИ ЭМБЕДДЕРА ПО БЛОКАМ (embedding/eval.py, `_embed_texts`).

1. Пачка по блокам == `model.embed` той же пачки -- побитно (те же операции).
2. Пачки набираются по строкам и по токенам: ни одна не превышает бюджет
   (кроме пачки из одной строки длиннее бюджета), строки идут подряд, порядок
   выхода -- порядок текстов.
3. С бюджетом и без -- те же векторы в пределах зависимости прямого прохода от
   числа строк (fp32: ~1e-6).
4. Модель без `base` идёт через `embed`, как раньше.

МУТАЦИИ (EVAL_MUT=...): budget -- бюджет токенов не действует; order --
пачки идут не по порядку; graph -- тело одним графом (не по блокам).

    python tests/test_embedding_eval.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx                                   # noqa: E402

import rwkv_metal.embedding.eval as ev_orig             # noqa: E402
from rwkv_metal.embedding.dataset import encode_batch   # noqa: E402
from rwkv_metal.embedding.train import EmbeddingModel   # noqa: E402
from rwkv_metal.model.convert import load_pretrained    # noqa: E402
from rwkv_metal.tokenizer.world_tokenizer import WorldTokenizer   # noqa: E402

MUT = os.environ.get("EVAL_MUT", "")
MUTATIONS = {
    "budget": ("if budget and budget > 0 and (end - i + 1) * grown > budget:", "if False:"),
    "order": ("        vecs.append(v)\n", "        vecs.insert(0, v)\n"),
    "graph": ('    by_blocks = hasattr(model, "embed_from_top") and hasattr(model, "base")', "    by_blocks = False"),
}
ev = ev_orig
if MUT:
    a, b = MUTATIONS[MUT]
    src = open(ev_orig.__file__, encoding="utf-8").read()
    assert src.count(a) == 1, (MUT, src.count(a))
    ev = types.ModuleType("eval_mutated")
    ev.__file__, ev.__package__ = ev_orig.__file__, ev_orig.__package__
    exec(compile(src.replace(a, b), ev_orig.__file__, "exec"), ev.__dict__)

PTH = os.path.expanduser("~/Develop/SwiftRWKV/.testdata/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth")
VOCAB = os.path.expanduser("~/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt")


def main():
    print(f"оценка эмбеддера по блокам, мутация: {MUT or 'нет'}")
    base = load_pretrained(PTH, verbose=False)[0]
    base.set_dtype("float32")
    model = EmbeddingModel(base)
    tok = WorldTokenizer(VOCAB)
    words = "мама мыла раму а кот смотрел на дождь за окном и думал о весне".split()
    texts = [" ".join(words[: 2 + (7 * i) % len(words)] * (1 + i % 5)) for i in range(23)]
    lengths = [len(tok.encode(t)) + 1 for t in texts]

    # 1. пачка по блокам == model.embed
    idx, pool = encode_batch(tok, texts[:6])
    a = model.embed(idx, pool)
    from rwkv_metal.lora.sequential import forward_trace
    top, _, _ = forward_trace(base, idx, None)
    b = model.embed_from_top(top, pool)
    mx.eval(a, b)
    assert mx.array_equal(a, b).item(), "тело по блокам и одним графом дали разные биты"
    print("  пачка по блокам == model.embed побитно")

    # 2. набор пачек по токенам
    n0 = ev.BATCHES
    budget = 60
    out = ev._embed_texts(model, tok, texts, batch_size=16, batch_tokens=budget)
    n_batches = ev.BATCHES - n0
    want, i = 0, 0
    while i < len(texts):                         # тот же набор, посчитанный здесь
        end, longest = i + 1, lengths[i]
        while end < len(texts) and end - i < 16 and (end - i + 1) * max(longest, lengths[end]) <= budget:
            longest, end = max(longest, lengths[end]), end + 1
        want, i = want + 1, end
    assert n_batches == want and n_batches > 3, f"пачек {n_batches}, ожидалось {want}"
    one = mx.concatenate([model.embed(*encode_batch(tok, [t])) for t in texts], axis=0)
    full = ev._embed_texts(model, tok, texts, batch_size=16, batch_tokens=0)
    mx.eval(out, one, full)
    d1 = mx.abs(out - one).max().item()
    d2 = mx.abs(full - one).max().item()
    assert out.shape == (len(texts), one.shape[1])
    assert d1 < 1e-4 and d2 < 1e-4, f"векторы разошлись с поштучными: {d1:.1e} / {d2:.1e}"
    print(f"  {len(texts)} текстов, бюджет {budget} токенов: {n_batches} пачек; к поштучным {d1:.1e}, без бюджета {d2:.1e}")

    # 4. модель без base
    class Plain:
        def embed(self, idx, pool):
            return model.embed(idx, pool)
    n0 = ev.BATCHES
    p = ev._embed_texts(Plain(), tok, texts[:5])
    mx.eval(p)
    assert ev.BATCHES == n0 and p.shape[0] == 5
    print("ГЕЙТ ПРОЙДЕН")


if __name__ == "__main__":
    main()
