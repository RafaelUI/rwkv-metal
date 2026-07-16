"""
verify_grad_w.py — свип по w: устойчивость градиента нового DPLR-пути.

Заход 1 (этот файл): true grad (autograd рекуррентности) vs новый чанк (autograd).
  Оракул правды = dplr_recurrence_mlx (лог-decay, без деления на w) — autograd точен.
  Зонд: равномерный w=w_level, включая под-floor 0.27, где боевой checkpoint-bwd
  (деление на w) даёт 3.6e16. Новый путь не делит на w → ждём конечность на всех w.
Заход 2 (позже): добавить колонку боевого Metal vjp из wkv7_checkpoint.

Запуск: .venv/bin/python experiments/verify_grad_w.py
"""
import os, sys
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv_metal.kernel.wkv7 import HEAD_SIZE
from dplr_mlx import dplr_recurrence_mlx, dplr_chunkwise_mlx
from rwkv_metal.kernel.wkv7_checkpoint import make_wkv7_checkpoint

NAMES = ["dr", "dw", "dk", "dv", "da", "db"]
W_LEVELS = [0.995, 0.747, 0.496, 0.270]


def make_inputs(w_level, B=2, T=64, H=4, D=HEAD_SIZE, seed=0):
    mx.random.seed(seed)
    r = mx.random.normal((B, T, H, D)) * 0.5
    k = mx.random.normal((B, T, H, D)) * 0.5
    v = mx.random.normal((B, T, H, D)) * 0.5
    kk = mx.random.normal((B, T, H, D))
    kk = kk / (mx.linalg.norm(kk, axis=-1, keepdims=True) + 1e-6)
    a = -kk
    b = kk * 0.1
    w = mx.full((B, T, H, D), w_level)          # равномерный decay-зонд
    dy = mx.random.normal((B, T, H, D))         # фиксированный котангенс
    return r, w, k, v, a, b, dy


def grads(fn, r, w, k, v, a, b, dy, **kw):
    def loss(r, w, k, v, a, b):
        return mx.sum(fn(r, w, k, v, a, b, **kw) * dy)
    return mx.grad(loss, argnums=[0, 1, 2, 3, 4, 5])(r, w, k, v, a, b)


def rel(ref, got):
    return (mx.max(mx.abs(ref - got)) / (mx.max(mx.abs(ref)) + 1e-12)).item()


def finite(x):
    return bool(mx.all(mx.isfinite(x)).item())


def main():
    B, T, H, D = 2, 64, 4, HEAD_SIZE
    battle_fn = make_wkv7_checkpoint(B=B, T=T, H=H, D=D)  # боевой Metal vjp (деление на w)

    hdr = f"{'w':>7} | " + " | ".join(f"{c:>22}" for c in ("dr (battle|chunk)", "dw (battle|chunk)", "da (battle|chunk)"))
    print(hdr); print("-" * len(hdr))
    for wl in W_LEVELS:
        r, w, k, v, a, b, dy = make_inputs(wl, B, T, H, D)
        gt = grads(dplr_recurrence_mlx, r, w, k, v, a, b, dy, scale=1.0)          # истина
        gc = grads(dplr_chunkwise_mlx, r, w, k, v, a, b, dy, scale=1.0, chunk_size=16)
        gb = grads(battle_fn, r, w, k, v, a, b, dy)                               # боевой
        cells = []
        for n in ("dr", "dw", "da"):
            i = NAMES.index(n)
            cells.append(f"{rel(gt[i], gb[i]):8.1e} | {rel(gt[i], gc[i]):8.1e}")
        print(f"{wl:7.3f} | " + " | ".join(f"{c:>22}" for c in cells))
    print()
    print("(rel к истинному градиенту = autograd рекуррентности; левое=боевой, правое=новый чанк)")

if __name__ == "__main__":
    main()
