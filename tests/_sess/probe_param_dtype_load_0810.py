"""08.10: param_dtype загрузчика .rwkvq (fp16 с декванта напрямую, а не каст из bf16) -- по процессу на плечо.
  truth <file> <ckpt> <tag>        истина (RWKV7Ref, .pth, счёт fp32) -> /tmp/pdl_truth_<tag>.npy
  arm <file> <tag> <bf16|fp16|fp32|REF>   KL к истине по окнам + градиенты адаптеров -> /tmp/pdl_<tag>_<arm>.npz
  speed <file> <tag> <arm>         шаг обучения T=512 (чекпоинтинг), 2+4 шага, пик MLX -> /tmp/pdl_speed.jsonl
  cmp <tag>                        сводка с бутстрэпом
REF = fp32 + база без каста входа + обратный fp32."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
NW, T = 8, 256
mode = sys.argv[1]
def wins():
    import torch
    ev = torch.load(os.path.expanduser("~/Develop/WKV-kvant/eval_text_heldout.pt"), weights_only=False)["tokens"]
    rows = [int(i) for i in np.linspace(0, ev.shape[0] - 1, NW).round()]
    return [ev[i, :T].numpy().astype(np.int32) for i in rows]
if mode == "truth":
    import torch
    from rwkv_quant.models.rwkv7_ref import RWKV7Ref
    MO = RWKV7Ref(sys.argv[3], device="cpu", dtype=torch.bfloat16, compute_dtype=torch.float32)
    with torch.no_grad():
        tr = np.stack([torch.log_softmax(MO.forward(torch.as_tensor(w)[None], cfg=None).float(), -1)[0].numpy() for w in wins()])
    np.save("/tmp/pdl_truth_%s.npy" % sys.argv[4], tr.astype(np.float16)); print("truth", tr.shape)
elif mode in ("arm", "speed"):
    import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten
    from rwkv_metal.lora import load_rwkvq_model, rwkvq_linear as rl
    f, tag, arm = sys.argv[2], sys.argv[3], sys.argv[4]
    if arm == "REF":
        rl.BASE_DTYPE, rl.NOCAST, rl.BWD_DTYPE = mx.float16, True, mx.float32
    mx.random.seed(0)
    model, cfg, info = load_rwkvq_model(f, rank=16, verbose=False, param_dtype="fp32" if arm == "REF" else arm)
    def perturb():
        rs = np.random.RandomState(3)
        upd = [(k, (v.astype(mx.float32) + mx.array(rs.normal(0, 1e-3, v.shape).astype(np.float32))).astype(v.dtype))
               for k, v in sorted(tree_flatten(model.trainable_parameters())) if k.endswith("lora_b")]
        model.update(tree_unflatten(upd)); mx.eval(model.parameters())
    if mode == "arm":
        tr = np.load("/tmp/pdl_truth_%s.npy" % tag).astype(np.float32)
        kls = []
        for i, w in enumerate(wins()):          # KL -- ДО возмущения адаптеров: база как есть
            lg = model(mx.array(w[None])).astype(mx.float32)[0]
            lq = lg - mx.logsumexp(lg, -1, keepdims=True); mx.eval(lq); lq = np.array(lq)
            kls.append(float((np.exp(tr[i]) * (tr[i] - lq)).sum(-1).mean()))
        perturb()
        w = wins()[0]; x = mx.array(w[None, :-1]); y = mx.array(w[None, 1:])
        loss, g = nn.value_and_grad(model, lambda m, a, b: m.loss(a, b).astype(mx.float32))(model, x, y)
        flat = dict(tree_flatten(g)); mx.eval(loss, flat)
        gv = np.concatenate([np.array(flat[k].astype(mx.float32)).ravel() for k in sorted(flat)])
        np.savez("/tmp/pdl_%s_%s.npz" % (tag, arm), kl=np.array(kls), g=gv, loss=float(loss))
        print(arm, "KL %.6f" % np.mean(kls), "loss %.5f" % float(loss), "finite", bool(np.isfinite(gv).all()))
    else:
        perturb()
        model._grad_ckpt = True
        r2 = np.random.RandomState(5)
        x = mx.array(r2.randint(1, 60000, size=(1, 512)).astype(np.int32)); y = mx.array(r2.randint(1, 60000, size=(1, 512)).astype(np.int32))
        gf = nn.value_and_grad(model, lambda m, a, b: m.loss(a, b).astype(mx.float32)); opt = optim.AdamW(learning_rate=1e-4)
        def step():
            l, gr = gf(model, x, y); opt.update(model, gr); mx.eval(l, model.state, opt.state); return float(l)
        for _ in range(2): step()
        mx.clear_cache(); mx.reset_peak_memory(); ts = []
        for _ in range(4):
            t0 = time.perf_counter(); l = step(); ts.append(time.perf_counter() - t0)
        r = dict(tag=tag, arm=arm, ms=1e3 * float(np.median(ts)), peak_mb=mx.get_peak_memory() / 2**20,
                 active_mb=mx.get_active_memory() / 2**20, loss=l, finite=bool(np.isfinite(l)))
        open("/tmp/pdl_speed.jsonl", "a").write(json.dumps(r) + "\n"); print(json.dumps(r))
elif mode == "cmp":
    tag = sys.argv[2]
    A = {a: np.load("/tmp/pdl_%s_%s.npz" % (tag, a)) for a in ("bf16", "fp16", "fp32", "REF") if os.path.exists("/tmp/pdl_%s_%s.npz" % (tag, a))}
    rng = np.random.default_rng(0); idx = rng.integers(0, NW, size=(20000, NW))
    ref = A["REF"]["g"]
    for a, d in A.items():
        k = d["kl"]; b = A["bf16"]["kl"]; r = k[idx].mean(1) / b[idx].mean(1) - 1
        print("%-5s KL %.6f  к bf16 %+.2f%% [%+.2f; %+.2f]  град отн.L2 к REF %.3e" % (a, k.mean(), 100 * (k.mean() / b.mean() - 1),
              100 * np.quantile(r, .025), 100 * np.quantile(r, .975), np.linalg.norm(d["g"] - ref) / np.linalg.norm(ref)))
