"""
verify_grad_steps.py — многошаговая grad-стабильность: боевой Metal vjp vs MLX-оракул.
Расширяет verify_grad_w (одиночный backward) на K шагов Adam.

Сетап: один WKV-слой, листовые параметры r,k,v,a,b и θ (через θ задаём
  w = exp(-0.606531*sigmoid(θ)) ∈ (0.5455,1.0) — РОВНО формула модели). θ
  инициализируем бимодально (~половина каналов у пола 0.545, как capture_w),
  чтобы постоянно бить по быстрозабывающим каналам, где боевой bwd (деление на w)
  несёт ~1e-3..взрыв. Лосс = mean((o - target)^2). K шагов Adam.
Логируем по шагам: loss, max|grad|, finite. Истинный grad (autograd рекуррентности)
  как опорная норма не нужен — здесь смотрим ДИНАМИКУ обучения и конечность.

Оракул правды для значений = dplr_recurrence_mlx; здесь сравниваем ТРАЕКТОРИИ
  обучения боевого ядра и оракула при идентичной инициализации.
"""
import os, sys
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv_metal.kernel.wkv7_checkpoint import make_wkv7_checkpoint
from dplr_mlx import dplr_chunkwise_mlx, dplr_recurrence_mlx

B, T, H, D = 2, 64, 4, 64
STEPS = 25
LR = 1e-2


def init_params(seed=0):
    mx.random.seed(seed)
    r = mx.random.normal((B, T, H, D)) * 0.5
    k = mx.random.normal((B, T, H, D)) * 0.5
    v = mx.random.normal((B, T, H, D)) * 0.5
    kk = mx.random.normal((B, T, H, D)); kk = kk / (mx.linalg.norm(kk, axis=-1, keepdims=True) + 1e-6)
    a = -kk; b = kk * 0.1
    # θ бимодально: ~половина каналов сильно отрицательны (sigmoid→0, w→1.0),
    # половина сильно положительны (sigmoid→1, w→floor 0.545) — как capture_w.
    sign = (mx.random.uniform(shape=(B, T, H, D)) < 0.5).astype(mx.float32) * 2 - 1
    theta = sign * 4.0 + mx.random.normal((B, T, H, D)) * 0.3
    return {"r": r, "k": k, "v": v, "a": a, "b": b, "theta": theta}


def w_of(theta):
    return mx.exp(-0.606531 * mx.sigmoid(theta))


def _grads(fn, p, target, keys, **kw):
    def loss_fn(*vals):
        d = dict(zip(keys, vals))
        w = w_of(d["theta"])
        o = fn(d["r"], w, d["k"], d["v"], d["a"], d["b"], **kw)
        return mx.mean((o - target) ** 2)
    L, g = mx.value_and_grad(loss_fn, argnums=list(range(len(keys))))(*[p[k] for k in keys])
    return L, g


def _rel(ref, got):
    return (mx.max(mx.abs(ref - got)) / (mx.max(mx.abs(ref)) + 1e-12)).item()


def run(seed=0):
    """Единая траектория: шагаем ПО ИСТИННОМУ grad (autograd рекуррентности);
    на каждом шаге меряем rel-ошибку боевого Metal vjp и MLX-чанка против истины."""
    keys = ["r", "k", "v", "a", "b", "theta"]
    p = init_params(seed)
    mx.random.seed(seed + 999)
    target = mx.random.normal((B, T, H, D)) * 0.3
    battle = make_wkv7_checkpoint(B=B, T=T, H=H, D=D)
    m = {k: mx.zeros_like(p[k]) for k in keys}; vv = {k: mx.zeros_like(p[k]) for k in keys}
    b1, b2, eps = 0.9, 0.999, 1e-8
    idx = {"dr": 0, "dw_via_theta": 5, "da": 3}  # dw течёт через theta
    print(f"  {'step':>4} {'loss':>9} | {'dr battle|chunk':>20} | {'dθ battle|chunk':>20} | {'da battle|chunk':>20} | floor%", flush=True)
    for step in range(1, STEPS + 1):
        Lt, gt = _grads(dplr_recurrence_mlx, p, target, keys, scale=1.0)   # ИСТИНА
        _,  gb = _grads(battle, p, target, keys)                            # боевой
        _,  gc = _grads(dplr_chunkwise_mlx, p, target, keys, scale=1.0, chunk_size=16)
        mx.eval(Lt, *gt, *gb, *gc)
        dr_b, dr_c = _rel(gt[0], gb[0]), _rel(gt[0], gc[0])
        dt_b, dt_c = _rel(gt[5], gb[5]), _rel(gt[5], gc[5])
        da_b, da_c = _rel(gt[3], gb[3]), _rel(gt[3], gc[3])
        wf = w_of(p["theta"]); fr = float(mx.mean((wf < 0.55).astype(mx.float32)).item())
        if step <= 3 or step % 5 == 0:
            print(f"  {step:4d} {float(Lt.item()):9.5f} | {dr_b:8.1e} {dr_c:8.1e} | {dt_b:8.1e} {dt_c:8.1e} | {da_b:8.1e} {da_c:8.1e} | {fr:4.0%}", flush=True)
        for i, k in enumerate(keys):   # шаг по ИСТИНЕ
            m[k] = b1 * m[k] + (1 - b1) * gt[i]; vv[k] = b2 * vv[k] + (1 - b2) * gt[i] ** 2
            p[k] = p[k] - LR * (m[k] / (1 - b1 ** step)) / (mx.sqrt(vv[k] / (1 - b2 ** step)) + eps)
        mx.eval(*[p[k] for k in keys])


def main():
    print(f"=== multi-step grad ACCURACY vs truth  B={B} T={T} H={H} D={D}  {STEPS} шагов ===", flush=True)
    print("Траектория общая (шаг по истине). Числа = rel-ошибка grad ПРОТИВ ИСТИНЫ", flush=True)
    print("(autograd рекуррентности). Левое=боевой Metal vjp, правое=новый MLX-чанк.", flush=True)
    print("dθ = grad по θ (через него течёт dw).", flush=True)
    run()
    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
