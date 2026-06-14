"""torch-free загрузчик RWKV .pth (torch zip-serialization) → MLX-массивы."""
import io, zipfile, pickle
import numpy as np
import mlx.core as mx
from collections import OrderedDict

# имя класса storage -> (numpy dtype | 'bf16', itemsize)
_DT = {
    'FloatStorage': (np.float32, 4), 'HalfStorage': (np.float16, 2),
    'BFloat16Storage': ('bf16', 2), 'DoubleStorage': (np.float64, 8),
    'LongStorage': (np.int64, 8), 'IntStorage': (np.int32, 4),
    'ByteStorage': (np.uint8, 1), 'BoolStorage': (np.bool_, 1),
}

class _Stor:
    def __init__(self, dt, isz, key, numel):
        self.dt, self.isz, self.key, self.numel = dt, isz, key, numel

def load_pth(path):
    zf = zipfile.ZipFile(path)
    prefix = zf.namelist()[0].split('/')[0]

    def rebuild(storage, storage_offset, size, stride, *a):
        raw = zf.read(f"{prefix}/data/{storage.key}")
        n = 1
        for s in size: n *= s
        start = storage_offset * storage.isz
        sub = raw[start:start + n * storage.isz]
        if storage.dt == 'bf16':
            arr = mx.array(np.frombuffer(sub, np.uint16).copy()).view(mx.bfloat16)
        else:
            arr = mx.array(np.frombuffer(sub, storage.dt).copy())
        return arr.reshape(tuple(size)) if size else arr

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name in _DT:
                return ('STOR',) + _DT[name]
            if mod == 'torch._utils' and name in ('_rebuild_tensor_v2', '_rebuild_tensor'):
                return rebuild
            if mod == 'collections' and name == 'OrderedDict':
                return OrderedDict
            if mod == 'torch' and name == 'Size':
                return tuple
            try:
                return super().find_class(mod, name)
            except Exception:
                return lambda *a, **k: None
        def persistent_load(self, pid):
            assert pid[0] == 'storage', pid[0]
            _, dt, isz = pid[1]
            return _Stor(dt, isz, str(pid[2]), pid[4])

    return U(io.BytesIO(zf.read(f"{prefix}/data.pkl"))).load()


def convert(z, n_layer, H, S):
    """официальные имена x070 -> наши (rwkv7_x070). bf16 сохраняется."""
    T = lambda a: a.T  # транспонирование low-rank
    out = {
        'emb.weight': z['emb.weight'],
        'ln0.weight': z['blocks.0.ln0.weight'], 'ln0.bias': z['blocks.0.ln0.bias'],
        'ln_out.weight': z['ln_out.weight'], 'ln_out.bias': z['ln_out.bias'],
        'head.weight': z['head.weight'],
    }
    for i in range(n_layer):
        b = f'blocks.{i}.'; att = b + 'att.'; ffn = b + 'ffn.'; P = b + 'tmix.'
        out[b + 'ln1.weight'] = z[b + 'ln1.weight']; out[b + 'ln1.bias'] = z[b + 'ln1.bias']
        out[b + 'ln2.weight'] = z[b + 'ln2.weight']; out[b + 'ln2.bias'] = z[b + 'ln2.bias']
        for x in ('x_r', 'x_w', 'x_k', 'x_v', 'x_a', 'x_g'):
            out[P + x] = z[att + x]
        out[P + 'k_k'] = z[att + 'k_k'].reshape(H, S)
        out[P + 'k_a'] = z[att + 'k_a'].reshape(H, S)
        out[P + 'r_k'] = z[att + 'r_k'].reshape(H, S)
        out[P + 'r_proj.weight'] = z[att + 'receptance.weight']
        out[P + 'k_proj.weight'] = z[att + 'key.weight']
        out[P + 'v_proj.weight'] = z[att + 'value.weight']
        out[P + 'o_proj.weight'] = z[att + 'output.weight']
        out[P + 'w_lora_A.weight'] = T(z[att + 'w1']); out[P + 'w_lora_B.weight'] = T(z[att + 'w2'])
        out[P + 'w_lora_B.bias'] = z[att + 'w0'].reshape(-1)
        out[P + 'a_lora_A.weight'] = T(z[att + 'a1']); out[P + 'a_lora_B.weight'] = T(z[att + 'a2'])
        out[P + 'a_lora_B.bias'] = z[att + 'a0'].reshape(-1)
        out[P + 'g_lora_A.weight'] = T(z[att + 'g1']); out[P + 'g_lora_B.weight'] = T(z[att + 'g2'])
        if i > 0:
            out[P + 'v_lora_A.weight'] = T(z[att + 'v1']); out[P + 'v_lora_B.weight'] = T(z[att + 'v2'])
            out[P + 'v_lora_B.bias'] = z[att + 'v0'].reshape(-1)
        out[P + 'ln_x.weight'] = z[att + 'ln_x.weight']; out[P + 'ln_x.bias'] = z[att + 'ln_x.bias']
        out[b + 'cmix.x_k'] = z[ffn + 'x_k']
        out[b + 'cmix.key.weight'] = z[ffn + 'key.weight']
        out[b + 'cmix.value.weight'] = z[ffn + 'value.weight']
    return out


def load_pretrained(pth_path, config=None, verbose=True):
    """Load official RWKV-7 x070 weights (.pth) into an RWKV7X070 model.

    torch-free: reads the torch zip/pickle directly into MLX arrays (bf16 kept).
    Architecture (n_layer/D/H/S/vocab) is inferred from the checkpoint; pass an
    explicit `config` to override.

    Args:
        pth_path: path to the official BlinkDL/rwkv-7-world .pth file.
        config:   optional PretrainConfig; if None, inferred from the weights.
        verbose:  print conversion diagnostics.

    Returns:
        (model, config). model is an RWKV7X070 with weights loaded; config is the
        PretrainConfig used to build it. Returns (None, config) if conversion is
        not clean (missing/extra/mismatched keys) so problems fail loudly.
    """
    import os
    from mlx.utils import tree_flatten, tree_unflatten
    from ..pretrain.config import PretrainConfig
    from .rwkv7_x070 import RWKV7X070

    pth_path = os.path.expanduser(pth_path)
    z = load_pth(pth_path)
    n_layer = 1 + max(int(k.split('.')[1]) for k in z if k.startswith('blocks.'))
    V, D = z['emb.weight'].shape
    H, S = z['blocks.0.att.r_k'].shape
    if verbose:
        print(f"checkpoint config: n_layer={n_layer} D={D} H={H} S={S} vocab={V}")

    if config is None:
        config = PretrainConfig(
            n_layer=n_layer, n_embd=D, vocab_size=V, head_size=S,
            ctx_len=512, batch_size=1,
            train_data="", val_data="", max_steps=1,
        )

    m = RWKV7X070(config)
    conv = convert(z, n_layer, H, S)

    model_keys = set(k for k, _ in tree_flatten(m.parameters()))
    conv_keys = set(conv.keys())
    miss = model_keys - conv_keys
    extra = conv_keys - model_keys
    md = dict(tree_flatten(m.parameters()))
    bad = [(k, tuple(conv[k].shape), tuple(md[k].shape))
           for k in (model_keys & conv_keys)
           if tuple(conv[k].shape) != tuple(md[k].shape)]

    if verbose:
        print(f"model keys: {len(model_keys)}, converter keys: {len(conv_keys)}")
        print(f"  missing in converter: {len(miss)} {sorted(miss)[:5]}")
        print(f"  extra in converter:   {len(extra)} {sorted(extra)[:5]}")
        print(f"  shape mismatches:     {len(bad)} {bad[:5]}")

    if miss or extra or bad:
        print("!!! conversion is NOT clean - aborting")
        return None, config

    m.update(tree_unflatten(list(conv.items())))
    mx.eval(m.parameters())
    return m, config


def save_converted(pth_path, out_path):
    """Convert an official .pth and save as an MLX safetensors file."""
    from mlx.utils import tree_flatten
    m, _ = load_pretrained(pth_path)
    if m is None:
        return False
    mx.save_safetensors(out_path, dict(tree_flatten(m.parameters())))
    print("saved ->", out_path)
    return True
