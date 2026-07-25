"""
rwkv_metal.embedding — извлечение векторов текста из базовой RWKV-7 модели.
===========================================================================
Это НЕ специально дообученная embedding-модель (как howard-hou/EmbeddingRWKV) —
это сырой претрейн. Но RWKV, будучи RNN, по конструкции сворачивает всю
последовательность в state фиксированного размера ("free sentence embedding"),
так что пулинг hidden state на последней позиции даёт рабочий, пусть и
неотшлифованный, эмбеддинг без какого-либо дообучения.

Качество дальше улучшается contrastive/curriculum дообучением (см. рецепт
EmbeddingRWKV: sft_curriculum) — это уже следующий шаг, не этот модуль.
"""
from typing import List, Sequence, Union

import mlx.core as mx

Texts = Union[str, Sequence[str]]


def _l2_normalize(x: mx.array) -> mx.array:
    return x / mx.sqrt((x * x).sum(axis=-1, keepdims=True) + 1e-12)


class Embedder:
    """Пулинг hidden state RWKV-7 в вектор фиксированной размерности.

    model:       RWKV7 или RWKV7X070 (нужен только .body(idx) -> [B,T,D]).
    tokenizer:   WorldTokenizer.
    terminator:  id токена, добавляемого в конец каждой последовательности
                 перед пулингом. 0 — зарезервированный/неиспользуемый id в
                 World-вокабе (не сопоставлен ни одной byte-строке в
                 rwkv_vocab_v20230424.txt) — естественный выбор терминатора
                 для базовой (не дообученной) модели. Передай None, чтобы
                 не добавлять терминатор вообще.
    pooling:     "last" (state на позиции терминатора) или "mean" (среднее
                 по всем позициям, включая терминатор).
    """

    def __init__(self, model, tokenizer, terminator: int = 0, pooling: str = "last"):
        assert pooling in ("last", "mean"), f"unknown pooling: {pooling}"
        self.model = model
        self.tok = tokenizer
        self.terminator = terminator
        self.pooling = pooling

    def encode(self, text: str) -> List[int]:
        ids = self.tok.encode(text)
        if self.terminator is not None:
            ids = ids + [self.terminator]
        return ids

    def embed_one(self, text: str) -> mx.array:
        ids = self.encode(text)
        idx = mx.array(ids)[None, :]            # [1, T]
        h = self.model.body(idx)                # [1, T, D], уже float
        if self.pooling == "last":
            vec = h[0, -1, :]
        else:
            vec = h[0].mean(axis=0)
        return _l2_normalize(vec)

    def embed(self, texts: Texts) -> mx.array:
        """Эмбеддинг списка текстов. Каждый текст считается отдельным
        forward-проходом (без паддинга) — так короткие/длинные
        последовательности никогда не загрязняют state друг друга через
        общий батч. Возвращает [N, D], L2-нормировано."""
        if isinstance(texts, str):
            texts = [texts]
        vecs = [self.embed_one(t) for t in texts]
        out = mx.stack(vecs, axis=0)
        mx.eval(out)
        return out

    __call__ = embed


def embed_texts(model, tokenizer, texts: Texts, terminator: int = 0, pooling: str = "last") -> mx.array:
    """Разовая обёртка над Embedder, без явного создания объекта."""
    return Embedder(model, tokenizer, terminator=terminator, pooling=pooling).embed(texts)


def cosine_similarity_matrix(a: mx.array, b: mx.array = None) -> mx.array:
    """a: [N,D] (L2-нормировано), b: [M,D] или None (= a). Возвращает [N,M]."""
    if b is None:
        b = a
    return a @ b.T
