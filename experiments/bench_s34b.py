"""bench_s34b.py — S3.4b-iii глобальный логгер: скорость, память, градиенты,
разбивка времени по стадиям; сверка с battle (rwkv_metal.kernel.wkv7).
Конфиг по умолчанию B2T64H4D64 C16 (handoff). fp32."""
import os, sys, time, argparse
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib; battle = importlib.import_module("rwkv_metal.kernel.wkv7")
from dplr_mlx import dplr_recurrence_mlx
from s3_dplr_kernel import (dplr_forward_metal_save, chunk_bwd_one_metal_bh_saved,
                            _kb_kernel, _bwd2_bh_kernel)

MB = 1024 * 1024


def sync():
    mx.eval(mx.array(0.0))  # барьер


def timed(fn, iters=50, warmup=10):
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); mx.eval(fn()); ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[0]  # median, min (ms)


def peak_mb(fn):
    mx.metal.clear_cache(); mx.reset_peak_memory(); mx.eval(fn())
    return mx.get_peak_memory() / MB


def pack(x):  # [B,T,H,D] -> [BH,T,D]
    B, T, H, D = x.shape
    return mx.transpose(x, (0, 2, 1, 3)).reshape(B * H, T, D)


def gstats(name, g):
    fin = mx.isfinite(g).all().item()
    return (f"  {name:>3}: |g|2={mx.linalg.norm(g).item():.4e}  "
            f"max={mx.max(mx.abs(g)).item():.3e}  mean={mx.mean(mx.abs(g)).item():.3e}  "
            f"finite={fin}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2); ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--H", type=int, default=4); ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--C", type=int, default=16); ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()
    B, T, H, D, C, it = a.B, a.T, a.H, a.D, a.C, a.iters
    BH, N = B * H, T // C
    print(f"=== BENCH S3.4b-iii  B{B}T{T}H{H}D{D} C{C}  (BH={BH}, N={N}, fp32) ===")
    print(f"device: {mx.device_info()['device_name']}  "
          f"max_rec_wss: {mx.device_info().get('max_recommended_working_set_size',0)/MB:.0f}MB\n")

    mx.random.seed(0)
    r = mx.random.normal((B, T, H, D)) * 0.5
    k = mx.random.normal((B, T, H, D)) * 0.5
    v = mx.random.normal((B, T, H, D)) * 0.5
    kk = mx.random.normal((B, T, H, D)); kk = kk / (mx.linalg.norm(kk, axis=-1, keepdims=True) + 1e-6)
    aa = -kk; bb = kk * 0.1
    w = mx.exp(-0.606531 * mx.sigmoid(mx.random.normal((B, T, H, D))))
    do = mx.random.normal((B, T, H, D))
    mx.eval(r, w, k, v, aa, bb, do)
    R, W, K, V, A, Bt, DO = map(pack, (r, w, k, v, aa, bb, do))

    # ---------- ours: fwd(SAVE) ----------
    def ours_fwd():
        o, cache = dplr_forward_metal_save(R, W, K, V, A, Bt, C)
        return o
    o_s, cache = dplr_forward_metal_save(R, W, K, V, A, Bt, C); mx.eval(o_s)

    # ---------- ours: bwd from cache ----------
    Aqk = [cache["Am"][n][:, 0] for n in range(N)]
    def ours_bwd():
        grads = [None] * N; dS = mx.zeros((BH, D, D))
        for n in range(N - 1, -1, -1):
            s = slice(n * C, (n + 1) * C)
            g = chunk_bwd_one_metal_bh_saved(R[:, s], W[:, s], K[:, s], V[:, s], A[:, s], Bt[:, s],
                                             cache["S_in"][n], dS, DO[:, s],
                                             cache["Am"][n], cache["u"][n], cache["wmat"][n], cache["v2"][n])
            grads[n] = g[:6]; dS = g[6]
        return tuple(mx.concatenate([grads[n][i] for n in range(N)], axis=1) for i in range(6))

    # ---------- stage isolation ----------
    def stage_kb():
        outs = []
        for n in range(N):
            s = slice(n * C, (n + 1) * C); Am = cache["Am"][n]
            outs.append(_kb_kernel(C, D)(
                inputs=[R[:, s], K[:, s], V[:, s], Bt[:, s], mx.cumsum(mx.log(W[:, s]), 1), DO[:, s],
                        cache["u"][n], cache["wmat"][n], cache["v2"][n], cache["S_in"][n],
                        mx.zeros((BH, D, D)), Am[:, 0], Am[:, 1], Am[:, 2], Am[:, 3]],
                grid=(32, BH, 1), threadgroup=(32, 1, 1),
                output_shapes=[(BH, C, D)] * 5 + [(BH, C, C)] * 4 + [(BH, D, D)] * 2,
                output_dtypes=[mx.float32] * 11))
        return outs
    def stage_bwd2():
        outs = []
        for n in range(N):
            s = slice(n * C, (n + 1) * C); Am = cache["Am"][n]
            gc = mx.cumsum(mx.log(W[:, s]), 1)
            outs.append(_bwd2_bh_kernel(C, D)(
                inputs=[R[:, s], K[:, s], A[:, s], Bt[:, s], gc, mx.log(W[:, s]),
                        Am[:, 0], Am[:, 1], Am[:, 2], Am[:, 3]],
                grid=(32, BH, 1), threadgroup=(32, 1, 1),
                output_shapes=[(BH, C, D)] * 4, output_dtypes=[mx.float32] * 4))
        return outs

    # ---------- battle ----------
    def battle_fwd():
        return battle.wkv7_train(r, w, k, v, aa, bb)
    lossB = lambda r, w, k, v, aa, bb: mx.sum(battle.wkv7_train(r, w, k, v, aa, bb) * do)
    vg = mx.value_and_grad(lossB, argnums=[0, 1, 2, 3, 4, 5])
    def battle_fb():
        l, g = vg(r, w, k, v, aa, bb); return (l, *g)

    # ============ SPEED ============
    print("--- СКОРОСТЬ (median / min, мс) ---")
    f_med, f_min = timed(ours_fwd, it)
    b_med, b_min = timed(ours_bwd, it)
    kb_med, _ = timed(stage_kb, it)
    b2_med, _ = timed(stage_bwd2, it)
    bf_med, bf_min = timed(battle_fwd, it)
    bfb_med, bfb_min = timed(battle_fb, it)
    print(f"  ours fwd(SAVE)        : {f_med:7.3f} / {f_min:7.3f}   ({f_med/N:.3f}/чанк)")
    print(f"  ours bwd(from cache)  : {b_med:7.3f} / {b_min:7.3f}   ({b_med/N:.3f}/чанк)")
    print(f"  ours fwd+bwd          : {f_med+b_med:7.3f}")
    print(f"  battle fwd            : {bf_med:7.3f} / {bf_min:7.3f}")
    print(f"  battle fwd+bwd (vjp)  : {bfb_med:7.3f} / {bfb_min:7.3f}")
    print(f"  >> speedup ours/battle (fwd+bwd): {bfb_med/(f_med+b_med):.2f}x\n")

    print("--- РАЗБИВКА bwd по стадиям (изолированно, мс; вкл. свой dispatch) ---")
    host = b_med - kb_med - b2_med
    print(f"  KB (град-ядро)        : {kb_med:7.3f}  ({100*kb_med/b_med:4.1f}%)")
    print(f"  bwd2 (hats-grads)     : {b2_med:7.3f}  ({100*b2_med/b_med:4.1f}%)")
    print(f"  host-хвост+carry+overh: {host:7.3f}  ({100*host/b_med:4.1f}%)")
    print(f"  (N={N} чанков × per-chunk dispatch — главный подозреваемый на overhead)\n")

    # ============ MEMORY ============
    print("--- ПАМЯТЬ (peak, MB) ---")
    print(f"  ours fwd(SAVE)        : {peak_mb(ours_fwd):8.2f}  (+кэш Am/u/wmat/v2 ×N)")
    print(f"  ours bwd(from cache)  : {peak_mb(ours_bwd):8.2f}")
    print(f"  battle fwd+bwd (ckpt) : {peak_mb(battle_fb):8.2f}\n")

    # ============ GRADIENTS ============
    print("--- ГРАДИЕНТЫ (ours fwd+bwd) + корректность vs ИСТИНА ---")
    gO = ours_bwd(); mx.eval(*gO)
    # истина по головам
    def relmax(a, b): return (mx.max(mx.abs(a - b)) / (mx.max(mx.abs(a)) + 1e-12)).item()
    names = ["dr", "dw", "dk", "dv", "da", "db"]
    for i, nm in enumerate(names):
        print(gstats(nm, gO[i]))
    # vs truth (одна голова h=0 как репрезентативная — полная сверка в verify_*)
    rh = R[0][None, :, None]; r0 = R[0]; w0 = W[0]; k0 = K[0]; v0 = V[0]; a0 = A[0]; b0 = Bt[0]; d0 = DO[0]
    def lT(r_, w_, k_, v_, a_, b_):
        x = lambda z: z[None, :, None, :]
        return mx.sum(dplr_recurrence_mlx(x(r_), x(w_), x(k_), x(v_), x(a_), x(b_))[0, :, 0] * d0)
    gT = mx.grad(lT, argnums=[0, 1, 2, 3, 4, 5])(r0, w0, k0, v0, a0, b0)
    mx.eval(*gT)
    errs = " ".join(f"{names[i]}={relmax(gT[i], gO[i][0]):.1e}" for i in range(6))
    print(f"  rel vs truth (head0): {errs}")


if __name__ == "__main__":
    main()
