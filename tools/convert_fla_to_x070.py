#!/usr/bin/env python3
"""
convert_fla_to_x070.py — convert an FLA-trained RWKV-7 checkpoint (.pt or
.safetensors) into the x070 tensor naming/layout consumed by the Swift
X070Backbone (and rwkv_metal/model/rwkv7_x070.py).

Why this exists: flash-linear-attention stores RWKV-7 weights under its own
module names (model.layers.N.attn.*, ffn.*, lm_head, model.norm, ...) and keeps
the per-head mix vectors k_k / k_a FLAT as [D]. The x070 loader expects the
official RWKV names (blocks.N.tmix.* / cmix.*, emb, head, ln_out, ln0) and the
per-head weights k_k / k_a shaped [H, S]. This script bridges both.

Config (nLayer, D, H, S) is auto-detected from the checkpoint.

Usage:
    python convert_fla_to_x070.py in.pt -o out.safetensors
    python convert_fla_to_x070.py in.safetensors            # -> in_x070.safetensors
"""
import argparse
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def load_state_dict(path: Path):
    """Load .pt (possibly wrapped {'model':..,'step':..}) or .safetensors."""
    step = None
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        with safe_open(str(path), "pt") as f:
            sd = {k: f.get_tensor(k) for k in f.keys()}
        return sd, step
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt and all(
        not isinstance(v, torch.Tensor) for v in ckpt.values()
    ):
        step = ckpt.get("step")
        ckpt = ckpt["model"]
    def strip(k):
        for p in ("module.", "_orig_mod."):
            if k.startswith(p):
                k = k[len(p):]
        return k
    return {strip(k): v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}, step


def detect_config(sd):
    """Infer (nLayer, D, H, S) from tensor shapes."""
    emb = sd.get("model.embeddings.weight")
    if emb is None:
        sys.exit("error: 'model.embeddings.weight' not found — is this an FLA checkpoint?")
    D = emb.shape[1]
    rk = sd.get("model.layers.0.attn.r_k")
    if rk is None or rk.dim() != 2:
        sys.exit("error: 'model.layers.0.attn.r_k' missing or not [H,S]")
    H, S = int(rk.shape[0]), int(rk.shape[1])
    layers = {int(m.group(1)) for k in sd if (m := re.match(r"model\.layers\.(\d+)\.", k))}
    nLayer = max(layers) + 1
    assert H * S == D, f"H*S ({H}*{S}) != D ({D})"
    return nLayer, D, H, S


def remap_key(k):
    if k == "model.embeddings.weight": return "emb.weight"
    if k == "lm_head.weight":          return "head.weight"
    if k == "model.norm.weight":       return "ln_out.weight"
    if k == "model.norm.bias":         return "ln_out.bias"
    m = re.match(r"model\.layers\.(\d+)\.(.+)$", k)
    if not m:
        return None
    L, rest = m.group(1), m.group(2)
    B = f"blocks.{L}."
    if rest == "pre_norm.weight": return "ln0.weight"
    if rest == "pre_norm.bias":   return "ln0.bias"
    if rest.startswith("attn_norm."): return B + "ln1." + rest.split(".", 1)[1]
    if rest.startswith("ffn_norm."):  return B + "ln2." + rest.split(".", 1)[1]
    if rest == "ffn.x_k":          return B + "cmix.x_k"
    if rest == "ffn.key.weight":   return B + "cmix.key.weight"
    if rest == "ffn.value.weight": return B + "cmix.value.weight"
    if rest.startswith("attn."):
        a = rest[len("attn."):]
        T = B + "tmix."
        if a in {"x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "k_k", "k_a", "r_k"}:
            return T + a
        if a in {"r_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"}:
            return T + a
        if a == "g_norm.weight": return T + "ln_x.weight"
        if a == "g_norm.bias":   return T + "ln_x.bias"
        lm = re.match(r"(\w)_lora\.lora\.(\d)\.(weight|bias)$", a)
        if lm:
            AB = "A" if lm.group(2) == "0" else "B"
            return f"{T}{lm.group(1)}_lora_{AB}.{lm.group(3)}"
    return None


def expected_keys(nLayer):
    exp = {"emb.weight", "ln0.weight", "ln0.bias",
           "ln_out.weight", "ln_out.bias", "head.weight"}
    for L in range(nLayer):
        B = f"blocks.{L}."
        exp |= {B + n for n in ("ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias")}
        T = B + "tmix."
        exp |= {T + n for n in (
            "x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "k_k", "k_a", "r_k",
            "r_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
            "a_lora_A.weight", "a_lora_B.weight", "a_lora_B.bias",
            "g_lora_A.weight", "g_lora_B.weight",
            "w_lora_A.weight", "w_lora_B.weight", "w_lora_B.bias",
            "ln_x.weight", "ln_x.bias")}
        if L > 0:
            exp |= {T + n for n in ("v_lora_A.weight", "v_lora_B.weight", "v_lora_B.bias")}
        exp |= {B + "cmix." + n for n in ("x_k", "key.weight", "value.weight")}
    return exp


def convert(in_path: Path, out_path: Path):
    sd, step = load_state_dict(in_path)
    nLayer, D, H, S = detect_config(sd)
    print(f"config: nLayer={nLayer} D={D} heads={H} head_size={S} "
          f"vocab={sd['model.embeddings.weight'].shape[0]}")

    out = {}
    for k, v in sd.items():
        nk = remap_key(k)
        if nk is None:
            continue
        v = v.contiguous().clone()
        if nk.endswith("tmix.k_k") or nk.endswith("tmix.k_a"):
            v = v.reshape(H, S).contiguous()
        out[nk] = v

    exp = expected_keys(nLayer)
    missing, extra = exp - set(out), set(out) - exp
    if missing:
        sys.exit(f"error: missing {len(missing)} expected keys, e.g. {sorted(missing)[:5]}")
    if extra:
        print(f"warning: {len(extra)} extra keys not read by X070Backbone: {sorted(extra)[:5]}")

    meta = {"format": "pt", "origin": "fla-remap"}
    if step is not None:
        meta["step"] = str(step)
    save_file(out, str(out_path), metadata=meta)
    print(f"ok: {len(out)} tensors -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Convert FLA RWKV-7 checkpoint to x070 naming/layout.")
    ap.add_argument("input", type=Path, help="input .pt or .safetensors (FLA naming)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output .safetensors (default: <input>_x070.safetensors)")
    args = ap.parse_args()
    out = args.output or args.input.with_name(args.input.stem + "_x070.safetensors")
    convert(args.input, out)


if __name__ == "__main__":
    main()
