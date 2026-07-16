"""bench_insitu.py — реальный 12L train-step (fwd+bwd): доля wkv7 + fast vs battle.
SAVE-режим: vjp использует кэш из forward-выходов (без рекомпьюта) — наш интендед rwkv-metal.
  без --which: гоняет passthrough/battle/ours + сводка.
  --which battle|ours|passthrough : один режим.
  --n_embd 768 --vocab 21248 --n_layer 12 --B 8 --T 512 --it 20 --wu 8
  --recompute : использовать recompute-vjp (для сравнения с save)."""
import os, sys, time, argparse
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal/experiments"))
import rwkv_metal.model.rwkv7 as M
from rwkv_metal.pretrain.config import PretrainConfig
from s3_dplr_kernel import (dplr_forward_metal_save_v2, dplr_bwd_metal_bh_fast,
                            _fast_bwd_given_cache)

C = 16
_battle_wkv7 = M.wkv7


def _pack(x):
    B, T, H, S = x.shape
    return mx.transpose(x, (0, 2, 1, 3)).reshape(B * H, T, S)


def _unpack(x, B, T, H, S):
    return mx.transpose(x.reshape(B, H, T, S), (0, 2, 1, 3))


# --- SAVE: forward отдаёт o + кэш-массивы; vjp использует их (без рекомпьюта) ---
@mx.custom_function
def _our_save(r, w, k, v, a, b):
    B, T, H, S = r.shape; N = T // C
    o, cache = dplr_forward_metal_save_v2(_pack(r), _pack(w), _pack(k), _pack(v), _pack(a), _pack(b), C)
    S_all = mx.stack(cache["S_in"], axis=1); Am_all = mx.stack(cache["Am"], axis=1)
    u_all = mx.stack(cache["u"], axis=1); wm_all = mx.stack(cache["wmat"], axis=1)
    v2_all = mx.stack(cache["v2"], axis=1)
    return _unpack(o, B, T, H, S), S_all, Am_all, u_all, wm_all, v2_all


@_our_save.vjp
def _our_save_vjp(primals, cotangents, outputs):
    r, w, k, v, a, b = primals
    do = cotangents[0]
    _, S_all, Am_all, u_all, wm_all, v2_all = outputs
    B, T, H, S = r.shape; N = T // C
    cache = dict(S_in=[S_all[:, n] for n in range(N)], Am=[Am_all[:, n] for n in range(N)],
                 u=[u_all[:, n] for n in range(N)], wmat=[wm_all[:, n] for n in range(N)],
                 v2=[v2_all[:, n] for n in range(N)])
    g = _fast_bwd_given_cache(_pack(r), _pack(w), _pack(k), _pack(v), _pack(a), _pack(b), _pack(do), C, cache)
    return tuple(_unpack(gi, B, T, H, S) for gi in g)


# --- RECOMPUTE (для сравнения): vjp пересчитывает форвард ---
@mx.custom_function
def _our_recompute(r, w, k, v, a, b):
    B, T, H, S = r.shape
    o = dplr_forward_metal_save_v2(_pack(r), _pack(w), _pack(k), _pack(v), _pack(a), _pack(b), C)[0]
    return _unpack(o, B, T, H, S)


@_our_recompute.vjp
def _our_recompute_vjp(primals, cotangent, output):
    r, w, k, v, a, b = primals; B, T, H, S = r.shape
    g = dplr_bwd_metal_bh_fast(_pack(r), _pack(w), _pack(k), _pack(v), _pack(a), _pack(b), _pack(cotangent), C)
    return tuple(_unpack(gi, B, T, H, S) for gi in g)


def make_ours(recompute):
    if recompute:
        return lambda r, w, k, v, a, b, training=True, state=None: (_our_recompute(r, w, k, v, a, b), None)
    return lambda r, w, k, v, a, b, training=True, state=None: (_our_save(r, w, k, v, a, b)[0], None)


def passthrough(r, w, k, v, a, b, training=True, state=None):
    return v, None


def measure(which, cfg, x, y, it, wu, recompute):
    fns = {"battle": _battle_wkv7, "ours": make_ours(recompute), "passthrough": passthrough}
    M.wkv7 = fns[which]
    model = M.RWKV7(cfg); mx.eval(model.parameters())
    def loss_fn(m, x, y): return m.loss(x, y).astype(mx.float32)
    vg = mx.value_and_grad(loss_fn)
    def step(): return vg(model, x, y)
    for _ in range(wu):
        l, g = step(); mx.eval(l, g)
    ts = []
    for _ in range(it):
        t0 = time.perf_counter(); l, g = step(); mx.eval(l, g); ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort(); return ts[len(ts) // 2], ts[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all", choices=["all", "battle", "ours", "passthrough"])
    ap.add_argument("--n_embd", type=int, default=768); ap.add_argument("--vocab", type=int, default=21248)
    ap.add_argument("--n_layer", type=int, default=12); ap.add_argument("--head_size", type=int, default=64)
    ap.add_argument("--B", type=int, default=8); ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--it", type=int, default=20); ap.add_argument("--wu", type=int, default=8)
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    cfg = PretrainConfig(); cfg.n_embd = a.n_embd; cfg.vocab_size = a.vocab
    cfg.n_layer = a.n_layer; cfg.head_size = a.head_size; cfg.batch_size = a.B; cfg.ctx_len = a.T
    H = cfg.n_embd // cfg.head_size
    mx.random.seed(0)
    x = mx.random.randint(0, cfg.vocab_size, (a.B, a.T)); y = mx.random.randint(0, cfg.vocab_size, (a.B, a.T))
    mode = "RECOMPUTE" if a.recompute else "SAVE"
    print(f"12L fwd+bwd [B={a.B} T={a.T} L={cfg.n_layer} d={cfg.n_embd} H={H} S={cfg.head_size} "
          f"vocab={cfg.vocab_size}]  wkv7: BH={a.B*H} N={a.T//C}  ours={mode}")
    modes = [a.which] if a.which != "all" else ["passthrough", "battle", "ours"]
    res = {}
    for m_ in modes:
        med, mn = measure(m_, cfg, x, y, a.it, a.wu, a.recompute)
        res[m_] = med
        print(f"  {m_:>11}: {med:8.2f} ms (median, min {mn:.2f})")
    if a.which == "all":
        pt = res["passthrough"]
        print(f"\n  доля wkv7 (12L): battle {res['battle']-pt:7.2f}ms ({100*(res['battle']-pt)/res['battle']:.0f}%) | "
              f"ours {res['ours']-pt:7.2f}ms ({100*(res['ours']-pt)/res['ours']:.0f}%)")
        print(f"  шаг ours/battle = {res['battle']/res['ours']:.3f}x  |  только-wkv7 ours/battle = "
              f"{(res['battle']-pt)/(res['ours']-pt+1e-9):.3f}x")


if __name__ == "__main__":
    main()
