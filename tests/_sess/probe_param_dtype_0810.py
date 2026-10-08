"""08.10: стоит ли держать плотные параметры модели (emb, нормы, лерпы, LoRA-ветки) не в bf16.
Плечи на .rwkvq (инференс, rank=0): bf16 (как есть) / fp16 / fp32 (тогда и счёт остальной модели в fp32).
KL к fp32-эталону из .pth на окнах calib_corpus, время прямого прохода (медиана), пик памяти MLX.
  python probe_param_dtype_0810.py ref <scale>   -- эталонные логиты в /tmp/pdt_ref_<scale>.npy
  python probe_param_dtype_0810.py arm <scale> <preset> <bf16|fp16|fp32>"""
import os, sys, time, json, numpy as np
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
import mlx.core as mx
from mlx.utils import tree_map
import rwkv_metal as rk
K = os.path.expanduser("~/Develop/WKV-kvant/")
PTH = {"0p1b": "rwkv7-g1d-0.1b.pth", "1p5b": "rwkv7-g1j-1.5b-20260831-ctx16384.pth"}
NW, T = 4, 512
mode, scale = sys.argv[1], sys.argv[2]
txt = open(os.path.expanduser("~/Develop/rwkv-quant/rwkv_quant/data/calib_corpus.txt"), encoding="utf-8").read()
ids = rk.WorldTokenizer().encode(txt[-400000:])
X = mx.array(np.array([ids[i * 3000: i * 3000 + T] for i in range(NW)], dtype=np.int32))
D = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}

def lsm(x):
    return x - mx.logsumexp(x, -1, keepdims=True)

def logits(m):
    out = []
    for w in range(NW):
        out.append(m(X[w:w + 1]).astype(mx.float32)); mx.eval(out[-1])
    return mx.concatenate(out, 0)

if mode == "ref":
    m, _ = rk.load_pretrained(K + PTH[scale], verbose=False)
    m.update(tree_map(lambda a: a.astype(mx.float32) if isinstance(a, mx.array) and a.dtype == mx.bfloat16 else a, m.parameters()))
    lp = lsm(logits(m)); mx.eval(lp)
    np.save("/tmp/pdt_ref_%s.npy" % scale, np.array(lp.astype(mx.float16)))
    print("ref saved", lp.shape)
else:
    preset, dt = sys.argv[3], D[sys.argv[4]]
    m, _ = rk.load_pretrained(K + "files_0110/%s_%s.rwkvq" % (scale, preset), verbose=False)
    m.update(tree_map(lambda a: a.astype(dt) if isinstance(a, mx.array) and a.dtype == mx.bfloat16 else a, m.parameters()))
    mx.eval(m.parameters()); mx.clear_cache(); mx.reset_peak_memory()
    lq = lsm(logits(m)); mx.eval(lq)
    ref = mx.array(np.load("/tmp/pdt_ref_%s.npy" % scale)).astype(mx.float32)
    kl = mx.sum(mx.exp(ref) * (ref - lq), -1)            # [NW, T]
    klw = np.array(mx.mean(kl, -1))
    ts = []
    for _ in range(5):
        t0 = time.perf_counter(); y = m(X[:1]); mx.eval(y); ts.append(time.perf_counter() - t0)
    r = {"scale": scale, "preset": preset, "dtype": sys.argv[4], "kl": float(klw.mean()), "kl_win": klw.tolist(),
         "fwd_ms": 1000 * float(np.median(ts[1:])), "peak_mb": mx.get_peak_memory() / 2**20, "active_mb": mx.get_active_memory() / 2**20}
    print(json.dumps(r))
    open("/tmp/pdt_res.jsonl", "a").write(json.dumps(r) + "\n")
