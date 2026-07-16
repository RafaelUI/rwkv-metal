#!/usr/bin/env python3
"""
gen_x070_fixtures.py — (re)generate the Swift parity fixtures for RWKVGen.

Outputs (into SwiftRWKV/Tests/RWKVGenTests/Resources/):
    x070_parity.safetensors    input_ids[1,8] i32 ; ln_out[8,768] bf16 ; logits[8,65536] bf16
    x070_stages.safetensors     after_ln0 / blk0_tmix / after_blk0          (f32, [1,8,768])
    x070_perlayer.safetensors   after_blk0..after_blk11                      (f32, [1,8,768])
    x070_tmix.safetensors       r k v g a w kk k2 wkv out_lnx res            (f32)

The forward here mirrors rwkv_metal/model/rwkv7_x070.py EXACTLY, reading tensors
straight from world_0.1b_x070.safetensors by their MLX names (same layout the
Swift X070Backbone consumes), and uses the project's Metal WKV-7 kernel.

Crucially ln_x is PER-TOKEN (causal) GroupNorm — F.group_norm(out.view(B*T, C), H,
eps=64e-5) — matching the fixed Swift port. The old fixtures baked in the buggy
cross-T ln_x; regenerating with this script is what turns X070ParityTests green.

Run:  .venv/bin/python tools/gen_x070_fixtures.py
"""
import os, mlx.core as mx
from rwkv_metal.kernel.wkv7 import wkv7

HERE   = os.path.dirname(os.path.abspath(__file__))
RES    = os.path.normpath(os.path.join(HERE, "..", "..", "SwiftRWKV",
                                       "Tests", "RWKVGenTests", "Resources"))
WEIGHTS = os.path.join(RES, "world_0.1b_x070.safetensors")

# Geometry of World-0.1B (must match X070Config in the tests).
N_LAYER, N_EMBD, HEAD_SIZE, VOCAB = 12, 768, 64, 65536
N_HEAD = N_EMBD // HEAD_SIZE
LNX_EPS, LN_EPS, DECAY_C = 64e-5, 1e-5, 0.606531

# Token ids the tests feed: input_ids = [1..8].
IDS = mx.arange(1, 9, dtype=mx.int32).reshape(1, 8)

# ── load weights once, cast to compute dtype (fp32 reference truth) ──
DT = mx.float32
_W = mx.load(WEIGHTS)
W  = {k: v.astype(DT) for k, v in _W.items()}
def g(key): return W[key]

# ── primitives (same math/order as the Swift functional backbone) ──
def lin(x, w):            return x @ w.T
def lin_b(x, w, b):       return x @ w.T + b
def l2norm(x):            return x / mx.sqrt((x * x).sum(-1, keepdims=True) + 1e-12)

def token_shift(x):
    prev = mx.concatenate([mx.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
    return prev - x                                   # xx = prev - x

def layer_norm(x, w, b, eps=LN_EPS):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / mx.sqrt(var + eps) * w + b

def ln_x_pertoken(x, w, b, H):                        # causal GroupNorm-by-head
    B, T, D = x.shape; S = D // H
    gg = x.reshape(B, T, H, S)
    mu = gg.mean(-1, keepdims=True)
    var = ((gg - mu) ** 2).mean(-1, keepdims=True)
    normed = ((gg - mu) / mx.sqrt(var + LNX_EPS)).reshape(B, T, D)
    return normed * w + b

def tmix(x, v_first, layer):
    p = f"blocks.{layer}.tmix."
    B, T, _ = x.shape; H, S, D = N_HEAD, HEAD_SIZE, N_EMBD
    xx = token_shift(x)
    xr = x + xx * g(p+"x_r"); xw = x + xx * g(p+"x_w"); xk = x + xx * g(p+"x_k")
    xv = x + xx * g(p+"x_v"); xa = x + xx * g(p+"x_a"); xg = x + xx * g(p+"x_g")

    r = lin(xr, g(p+"r_proj.weight")).reshape(B, T, H, S)
    k = lin(xk, g(p+"k_proj.weight")).reshape(B, T, H, S)          # raw k
    v = lin(xv, g(p+"v_proj.weight")).reshape(B, T, H, S)          # raw v

    gate = lin(mx.sigmoid(lin(xg, g(p+"g_lora_A.weight"))), g(p+"g_lora_B.weight"))

    if layer == 0:
        v_first_out = v
    else:
        vv = mx.sigmoid(lin_b(lin(xv, g(p+"v_lora_A.weight")),
                              g(p+"v_lora_B.weight"), g(p+"v_lora_B.bias"))).reshape(B, T, H, S)
        v = v + (v_first - v) * vv
        v_first_out = v_first

    a = mx.sigmoid(lin_b(lin(xa, g(p+"a_lora_A.weight")),
                         g(p+"a_lora_B.weight"), g(p+"a_lora_B.bias"))).reshape(B, T, H, S)

    w = lin_b(mx.tanh(lin(xw, g(p+"w_lora_A.weight"))),
              g(p+"w_lora_B.weight"), g(p+"w_lora_B.bias"))
    w = mx.exp(-DECAY_C * mx.sigmoid(w.astype(mx.float32))).astype(x.dtype).reshape(B, T, H, S)

    kk = l2norm(k * g(p+"k_k"))                                    # uses raw k
    k2 = k * (1.0 + (a - 1.0) * g(p+"k_a"))

    wkv, _ = wkv7(r, w, k2, v, -kk, kk * a, training=True)         # (B,T,H,S)

    out_lnx = ln_x_pertoken(wkv.reshape(B, T, D),
                            g(p+"ln_x.weight"), g(p+"ln_x.bias"), H).reshape(B, T, H, S)
    bonus = (r * k2 * g(p+"r_k")).sum(-1, keepdims=True) * v
    out = (out_lnx + bonus).reshape(B, T, D)
    res = lin(out * gate, g(p+"o_proj.weight"))

    sub = None
    if layer == 0:
        sub = {"r": r, "k": k, "v": v, "g": gate, "a": a, "w": w,
               "kk": kk, "k2": k2, "wkv": wkv, "out_lnx": out_lnx, "res": res}
    return res, v_first_out, sub

def cmix(x, layer):
    p = f"blocks.{layer}.cmix."
    xx = token_shift(x)
    xk = x + xx * g(p+"x_k")
    h = mx.maximum(lin(xk, g(p+"key.weight")), 0)
    return lin(h * h, g(p+"value.weight"))

def forward(ids):
    emb = g("emb.weight")[ids]                                     # [B,T,D]
    x = layer_norm(emb, g("ln0.weight"), g("ln0.bias"))
    after_ln0 = x
    v_first = None
    per_layer, stages, tmix_sub = {}, {}, None
    for l in range(N_LAYER):
        h, v_first, sub = tmix(layer_norm(x, g(f"blocks.{l}.ln1.weight"),
                                          g(f"blocks.{l}.ln1.bias")), v_first, l)
        if l == 0:
            stages["blk0_tmix"] = h
            tmix_sub = sub
        x = x + h
        x = x + cmix(layer_norm(x, g(f"blocks.{l}.ln2.weight"),
                                g(f"blocks.{l}.ln2.bias")), l)
        per_layer[f"after_blk{l}"] = x
    stages["after_ln0"] = after_ln0
    stages["after_blk0"] = per_layer["after_blk0"]
    ln_out = layer_norm(x, g("ln_out.weight"), g("ln_out.bias"))
    logits = lin(ln_out, g("head.weight"))
    return ln_out, logits, stages, per_layer, tmix_sub

def save(path, d):
    mx.eval(list(d.values()))
    mx.save_safetensors(path, d)
    print(f"  wrote {os.path.basename(path):26s} "
          f"{ {k: tuple(v.shape) for k, v in d.items()} }")

def main():
    print(f"weights : {WEIGHTS}")
    print(f"out dir : {RES}")
    # back up the old fixtures (cross-T) before overwriting
    for n in ["x070_parity", "x070_stages", "x070_perlayer", "x070_tmix"]:
        p = os.path.join(RES, n + ".safetensors")
        if os.path.exists(p) and not os.path.exists(p + ".bak"):
            os.replace(p, p + ".bak"); print(f"  backup  {n}.safetensors -> .bak")

    ln_out, logits, stages, per_layer, tmix_sub = forward(IDS)
    mx.eval(ln_out, logits)

    # parity: bf16 ln_out/logits, batch squeezed to [8, .]  (test accepts bf16)
    save(os.path.join(RES, "x070_parity.safetensors"), {
        "input_ids": IDS,
        "ln_out":  ln_out.reshape(8, N_EMBD).astype(mx.bfloat16),
        "logits":  logits.reshape(8, VOCAB).astype(mx.bfloat16),
    })
    save(os.path.join(RES, "x070_stages.safetensors"),
         {k: stages[k] for k in ["after_ln0", "blk0_tmix", "after_blk0"]})
    save(os.path.join(RES, "x070_perlayer.safetensors"), per_layer)
    save(os.path.join(RES, "x070_tmix.safetensors"), tmix_sub)

    # sanity: argmax of last-token logits + scale
    last = logits.reshape(8, VOCAB)[7]
    print(f"  sanity  last-token argmax={int(mx.argmax(last))}  "
          f"logits[min,max]=[{float(last.min()):.2f},{float(last.max()):.2f}]")
    print("done.")

if __name__ == "__main__":
    main()
