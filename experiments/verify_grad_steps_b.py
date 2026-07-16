"""
verify_grad_steps_b.py — S3.4b: многошаговая стабильность АНАЛИТИЧЕСКОГО межчанк-bwd.
Расширяет verify_grad_steps (тот мерил autograd чанк-формы) на РУЧНОЙ VJP
chunk_bwd_seq — ровно тот код, что пойдёт в Metal. Траектория общая (шаг по
ИСТИНЕ=autograd рекуррентности). Per-step rel-ошибка против истины:
  левое = боевой Metal vjp (деление на w), правое = аналитический межчанк-chunk.
θ бимодально (~половина каналов у пола 0.545, как capture_w).
"""
import os, sys, time
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv_metal.kernel.wkv7_checkpoint import make_wkv7_checkpoint
from dplr_mlx import dplr_recurrence_mlx
from dplr_bwd_chunk_mlx import chunk_fwd_seq, chunk_bwd_seq

B, T, H, D = 2, 64, 4, 64
C = 16
STEPS = 25
LR = 1e-2
CK = 0.606531


def w_of(theta):
    return mx.exp(-CK * mx.sigmoid(theta))


def init_params(seed=0):
    mx.random.seed(seed)
    r = mx.random.normal((B, T, H, D)) * 0.5
    k = mx.random.normal((B, T, H, D)) * 0.5
    v = mx.random.normal((B, T, H, D)) * 0.5
    kk = mx.random.normal((B, T, H, D)); kk = kk / (mx.linalg.norm(kk, axis=-1, keepdims=True) + 1e-6)
    a = -kk; b = kk * 0.1
    sign = (mx.random.uniform(shape=(B, T, H, D)) < 0.5).astype(mx.float32) * 2 - 1
    theta = sign * 4.0 + mx.random.normal((B, T, H, D)) * 0.3
    return {"r": r, "k": k, "v": v, "a": a, "b": b, "theta": theta}


def _rel(ref, got):
    return (mx.max(mx.abs(ref - got)) / (mx.max(mx.abs(ref)) + 1e-12)).item()


keys = ["r", "k", "v", "a", "b", "theta"]


def truth_grads(p, target):
    def loss_fn(*vals):
        d = dict(zip(keys, vals)); w = w_of(d["theta"])
        o = dplr_recurrence_mlx(d["r"], w, d["k"], d["v"], d["a"], d["b"], scale=1.0)
        return mx.mean((o - target) ** 2)
    return mx.value_and_grad(loss_fn, argnums=list(range(6)))(*[p[k] for k in keys])


def battle_grads(p, target, battle):
    def loss_fn(*vals):
        d = dict(zip(keys, vals)); w = w_of(d["theta"])
        o = battle(d["r"], w, d["k"], d["v"], d["a"], d["b"])
        return mx.mean((o - target) ** 2)
    return mx.value_and_grad(loss_fn, argnums=list(range(6)))(*[p[k] for k in keys])


def _assemble(parts):
    """parts[b*H+h] = [T,D]  ->  [B,T,H,D] (через stack+transpose, БЕЗ scatter:
    .at[int,:,int] в MLX раскладывает неверно)."""
    x = mx.stack(parts, axis=0).reshape(B, H, T, D)
    return mx.transpose(x, (0, 2, 1, 3))


def analytic_grads(p, target):
    """Ручной межчанк-VJP, разматывая B*H. dw->dθ цепочкой модели."""
    r, k, v, a, b, theta = (p[x] for x in keys)
    w = w_of(theta)
    o_parts = []
    for bi in range(B):
        for hi in range(H):
            o_bh, _ = chunk_fwd_seq(r[bi, :, hi], w[bi, :, hi], k[bi, :, hi],
                                    v[bi, :, hi], a[bi, :, hi], b[bi, :, hi], C)
            o_parts.append(o_bh)
    o_full = _assemble(o_parts)
    do = 2.0 * (o_full - target) / o_full.size
    gp = {x: [] for x in ["r", "w", "k", "v", "a", "b"]}
    for bi in range(B):
        for hi in range(H):
            dr, dw, dk, dv, da, db = chunk_bwd_seq(
                r[bi, :, hi], w[bi, :, hi], k[bi, :, hi], v[bi, :, hi],
                a[bi, :, hi], b[bi, :, hi], do[bi, :, hi], C)
            for nm, val in zip(["r", "w", "k", "v", "a", "b"], [dr, dw, dk, dv, da, db]):
                gp[nm].append(val)
    g = {nm: _assemble(gp[nm]) for nm in gp}
    s = mx.sigmoid(theta)
    dtheta = g["w"] * (w * (-CK) * s * (1 - s))   # dw/dθ
    L = mx.mean((o_full - target) ** 2)
    return L, [g["r"], g["k"], g["v"], g["a"], g["b"], dtheta]


def run(seed=0):
    p = init_params(seed)
    mx.random.seed(seed + 999)
    target = mx.random.normal((B, T, H, D)) * 0.3
    battle = make_wkv7_checkpoint(B=B, T=T, H=H, D=D)
    m = {x: mx.zeros_like(p[x]) for x in keys}; vv = {x: mx.zeros_like(p[x]) for x in keys}
    b1, b2, eps = 0.9, 0.999, 1e-8
    print(f"  {'step':>4} {'loss':>9} | {'dr  bat|chunk':>19} | {'dθ  bat|chunk':>19} | {'da  bat|chunk':>19} | finite | floor%", flush=True)
    worst_c = 0.0; allfin = True
    for step in range(1, STEPS + 1):
        Lt, gt = truth_grads(p, target)
        _, gb = battle_grads(p, target, battle)
        _, ga = analytic_grads(p, target)
        mx.eval(Lt, *gt, *gb, *ga)
        dr_b, dr_c = _rel(gt[0], gb[0]), _rel(gt[0], ga[0])
        dt_b, dt_c = _rel(gt[5], gb[5]), _rel(gt[5], ga[5])
        da_b, da_c = _rel(gt[3], gb[3]), _rel(gt[3], ga[3])
        fin = all(bool(mx.all(mx.isfinite(x)).item()) for x in ga)
        allfin &= fin
        worst_c = max(worst_c, dr_c, dt_c, da_c)
        wf = w_of(p["theta"]); fr = float(mx.mean((wf < 0.55).astype(mx.float32)).item())
        if step <= 3 or step % 5 == 0:
            print(f"  {step:4d} {float(Lt.item()):9.5f} | {dr_b:8.1e} {dr_c:8.1e} | {dt_b:8.1e} {dt_c:8.1e} | {da_b:8.1e} {da_c:8.1e} |  {str(fin):>5} | {fr:4.0%}", flush=True)
        for i, x in enumerate(keys):
            m[x] = b1 * m[x] + (1 - b1) * gt[i]; vv[x] = b2 * vv[x] + (1 - b2) * gt[i] ** 2
            p[x] = p[x] - LR * (m[x] / (1 - b1 ** step)) / (mx.sqrt(vv[x] / (1 - b2 ** step)) + eps)
        mx.eval(*[p[x] for x in keys])
    print(f"  WORST analytic rel vs truth over {STEPS} steps = {worst_c:.2e}  all-finite={allfin}", flush=True)


def main():
    print(f"=== S3.4b multi-step analytic-bwd ACCURACY vs truth  B={B} T={T} H={H} D={D} C={C}  {STEPS} шагов ===", flush=True)
    run()
    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
