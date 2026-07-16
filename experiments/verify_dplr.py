"""
verify_dplr.py — паритет MLX-порта DPLR против боевого wkv7_train_py.

stage 0: dplr_recurrence_mlx vs wkv7_train_py (ловушки scale/gk/layout).
Запуск: .venv/bin/python experiments/verify_dplr.py
"""
import os, sys
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv_metal.kernel.wkv7 import wkv7_train_py, HEAD_SIZE
from dplr_mlx import dplr_recurrence_mlx, dplr_chunkwise_mlx


def make_inputs(B=2, T=64, H=4, D=HEAD_SIZE, seed=0):
    mx.random.seed(seed)
    r = mx.random.normal((B, T, H, D)) * 0.5
    k = mx.random.normal((B, T, H, D)) * 0.5
    v = mx.random.normal((B, T, H, D)) * 0.5
    # структурированные a,b как в RWKV-7: a=-kk, b=kk*scale, kk нормирован →
    # переход diag(w)+b aᵀ контрактивен, состояние O(1) (а не взрыв)
    kk = mx.random.normal((B, T, H, D))
    kk = kk / (mx.linalg.norm(kk, axis=-1, keepdims=True) + 1e-6)
    a = -kk
    b = kk * 0.1
    # реальный decay как в модели: w = exp(-0.606531*sigmoid(.)) in (0.5455, 1.0)
    w = mx.exp(-0.606531 * mx.sigmoid(mx.random.normal((B, T, H, D))))
    return r, w, k, v, a, b


def report(name, ref, got):
    d = mx.abs(ref - got)
    amax = mx.max(d).item()
    amean = mx.mean(d).item()
    rel = (amax / (mx.max(mx.abs(ref)).item() + 1e-12))
    print(f"[{name}] max_abs={amax:.3e}  mean_abs={amean:.3e}  max_rel={rel:.3e}")
    return amax


def main():
    r, w, k, v, a, b = make_inputs()
    ref = wkv7_train_py(r, w, k, v, a, b)
    got = dplr_recurrence_mlx(r, w, k, v, a, b, scale=1.0)
    print("shapes:", "ref", ref.shape, "got", got.shape)
    def relcheck(name, got):
        report(name, ref, got)
        rel = (mx.max(mx.abs(ref - got)) / (mx.max(mx.abs(ref)) + 1e-12)).item()
        ok = rel < 1e-4
        print(f"  refmax={mx.max(mx.abs(ref)).item():.3e}  max_rel={rel:.3e}  {'PASS' if ok else 'FAIL'}")
        return ok

    relcheck("stage0 recurrence", got)
    relcheck("stage1 chunk=16", dplr_chunkwise_mlx(r, w, k, v, a, b, scale=1.0, chunk_size=16))
    relcheck("stage1 chunk=32", dplr_chunkwise_mlx(r, w, k, v, a, b, scale=1.0, chunk_size=32))


if __name__ == "__main__":
    main()
