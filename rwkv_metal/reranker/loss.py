"""
Лоссы реранкера.

Оригинальный EmbeddingRWKV учит голову поточечно: BCE на логитах позитива
(цель 1) и негативов (цель 0). Это работает, но оптимизирует не то, что
меряется: реранкеру не нужно попасть в абсолютную «релевантность», ему нужно
поставить правильный документ выше остальных. Listwise softmax по кандидатам
оптимизирует ровно порядок и по умолчанию используется здесь; BCE оставлен
для сравнения и для подмешивания (у поточечного члена есть полезный побочный
эффект — он калибрует абсолютный уровень скоров, если их потом сравнивают
между запросами).
"""
import mlx.core as mx
import mlx.nn as nn


def listwise_loss(scores: mx.array, labels: mx.array,
                  temperature: float = 1.0) -> mx.array:
    """scores: [B, C] логиты кандидатов, labels: [B] индекс правильного.

    При нулевой голове (zero-init) все скоры равны нулю, и лосс в точности
    равен ln(C) — удобная проверка, что данные и голова собраны верно.
    """
    return nn.losses.cross_entropy(scores.astype(mx.float32) / temperature,
                                   labels).mean()


def bce_loss(scores: mx.array, labels: mx.array) -> mx.array:
    """Поточечный BCE: позитив → 1, остальные → 0 (рецепт оригинала)."""
    B, C = scores.shape
    targets = mx.zeros((B, C))
    targets[mx.arange(B), labels] = 1.0
    return nn.losses.binary_cross_entropy(scores.astype(mx.float32), targets,
                                          with_logits=True).mean()


def mixed_loss(scores: mx.array, labels: mx.array, alpha: float = 0.9,
               temperature: float = 1.0) -> mx.array:
    """alpha·listwise + (1-alpha)·BCE."""
    if alpha >= 1.0:
        return listwise_loss(scores, labels, temperature)
    if alpha <= 0.0:
        return bce_loss(scores, labels)
    return (alpha * listwise_loss(scores, labels, temperature)
            + (1.0 - alpha) * bce_loss(scores, labels))
