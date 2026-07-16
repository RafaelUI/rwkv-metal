"""Fused / chunked cross-entropy for large-vocab small models.

Проблема: при vocab=32k и маленькой модели полные логиты [N, V] (N = B*T)
материализуются целиком (~6.6 ГБ на batch=36, ctx=512) и доминируют по памяти
и по трафику на base M4 (~120 ГБ/с). Тело модели 256-dim рядом с этим — пыль.

Идея (cut-cross-entropy / chunked CE): не материализовать полный [N, V] тензор.
- forward считает лосс по чанкам строк, держа в памяти только [chunk, V];
- backward — кастомный vjp, который ПЕРЕСЧИТЫВАЕТ логиты по чанкам и копит
  градиенты, тоже не материализуя [N, V]. Кастом-функция сохраняет на backward
  только примитивы (H, W, targets), а не промежуточные логиты.

Совпадает с nn.losses.cross_entropy(...).mean() по значению и по градиентам
(в пределах fp-погрешности). Размер словаря и качество модели не трогает.
"""
import mlx.core as mx


def naive_ce(H, W, targets):
    """Референс: материализует полные логиты [N, V]. Так делает model.loss сейчас."""
    logits = (H @ W.T).astype(mx.float32)            # [N, V]
    lse = mx.logsumexp(logits, axis=-1)              # [N]
    tgt = mx.take_along_axis(logits, targets[:, None], axis=-1).squeeze(-1)
    return (lse - tgt).mean()


def make_fused_ce(chunk_size: int = 2048):
    """Фабрика: возвращает custom_function fused CE с зашитым размером чанка."""

    @mx.custom_function
    def fused_ce(H, W, targets):
        # H: [N, D]  W: [V, D]  targets: [N] int
        N = H.shape[0]
        total = mx.zeros((), dtype=mx.float32)
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            logits = (H[s:e] @ W.T).astype(mx.float32)            # [c, V]
            lse = mx.logsumexp(logits, axis=-1)                   # [c]
            tgt = mx.take_along_axis(
                logits, targets[s:e][:, None], axis=-1).squeeze(-1)
            total = total + (lse - tgt).sum()
        return total / N

    @fused_ce.vjp
    def fused_ce_vjp(primals, cotangent, output):
        H, W, targets = primals
        N = H.shape[0]
        Wf = W.astype(mx.float32)
        scale = cotangent / N
        dH_parts = []
        dW = mx.zeros_like(Wf)
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            Hc = H[s:e].astype(mx.float32)                        # [c, D]
            logits = Hc @ Wf.T                                    # [c, V]
            P = mx.softmax(logits, axis=-1)                       # [c, V]
            oh = mx.zeros_like(P)
            oh[mx.arange(e - s), targets[s:e]] = 1.0
            G = (P - oh) * scale                                  # [c, V]
            dH_parts.append((G @ Wf).astype(H.dtype))             # [c, D]
            dW = dW + G.T @ Hc                                    # [V, D]
        dH = mx.concatenate(dH_parts, axis=0)
        return dH, dW.astype(W.dtype), mx.zeros(targets.shape, dtype=mx.int32)

    return fused_ce
