"""07.10: побитный отпечаток QLoRA-модели rwkv-metal -- до и после правок памяти загрузки (должен совпасть).
Логиты прямого прохода T=64 и градиенты адаптеров одного шага T=128 (lora_b возмущены сидом), sha256 от fp32-байт.
    python bits_ref_0710.py <out.json> <файл.rwkvq>[:native] ..."""
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, mlx.core as mx, mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from rwkv_metal.lora import load_rwkvq_model, rwkvq_linear as rl
def sha(arrs):
    h = hashlib.sha256()
    for a in arrs: h.update(np.array(a.astype(mx.float32)).tobytes())
    return h.hexdigest()[:16]
res = {}
for spec in sys.argv[2:]:
    path, _, nat = spec.partition(":"); native = nat != "0"
    rl._SIDECAR_CACHE.clear(); mx.random.seed(0)
    model, cfg, info = load_rwkvq_model(path, rank=16, verbose=False, native=native)
    rs = np.random.RandomState(11)
    x = mx.array(rs.randint(1, 60000, size=(1, 64)).astype(np.int32)); lg = model(x); mx.eval(lg)
    r = {"logits": sha([lg])}
    upd = [(k, (v.astype(mx.float32) + mx.array(rs.normal(0, 1e-3, v.shape).astype(np.float32))).astype(v.dtype))
           for k, v in tree_flatten(model.trainable_parameters()) if k.endswith("lora_b")]
    upd += [(k, mx.array(rs.normal(0, 0.02, v.shape).astype(np.float32)).astype(v.dtype))
            for k, v in tree_flatten(model.trainable_parameters()) if k.endswith("lora_a")]
    model.update(tree_unflatten(upd)); mx.eval(model.parameters())
    xa = mx.array(rs.randint(1, 60000, size=(1, 128)).astype(np.int32)); ya = mx.array(rs.randint(1, 60000, size=(1, 128)).astype(np.int32))
    loss, g = nn.value_and_grad(model, lambda m, a, b: m.loss(a, b).astype(mx.float32))(model, xa, ya)
    flat = dict(tree_flatten(g)); mx.eval(loss, flat)
    r["loss"] = float(loss); r["grads"] = sha([flat[k] for k in sorted(flat)])
    res["%s native=%s" % (os.path.basename(path), native)] = r; print(os.path.basename(path), native, r, flush=True)
    del model; mx.clear_cache()
json.dump(res, open(sys.argv[1], "w"), indent=1)
