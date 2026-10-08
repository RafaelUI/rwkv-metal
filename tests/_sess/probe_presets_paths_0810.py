"""08.10: работают ли нынешние пресеты (.rwkvq REDUCTION / COMPRESSION, файлы 01.10) во всех путях
rwkv-metal: прямой проход (логиты), эмбеддинг (body + пул), реранкер (states + голова с init_from_base),
QLoRA (шаг градиента по адаптерам). Сравнение с плотной базой из .pth той же модели."""
import importlib, os, sys, numpy as np
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
import mlx.core as mx
from mlx.utils import tree_flatten
CV = importlib.import_module("rwkv_metal.model.convert")
from rwkv_metal.lora.add_rwkvq import load_rwkvq_model
from rwkv_metal.embedding.train import EmbeddingModel
from rwkv_metal.reranker.model import Reranker

SCALE = sys.argv[1] if len(sys.argv) > 1 else "0p1b"
PTH = {"0p1b": "rwkv7-g1d-0.1b.pth", "0p4b": "rwkv7-g1d-0.4b-20260210-ctx8192.pth",
       "1p5b": "rwkv7-g1j-1.5b-20260831-ctx16384.pth"}[SCALE]
K = os.path.expanduser("~/Develop/WKV-kvant/")
rng = np.random.default_rng(0)
idx = mx.array(rng.integers(100, 60000, size=(2, 96)).astype(np.int32))

dense, _ = CV.load_pretrained(K + PTH, verbose=False)
ld = dense(idx).astype(mx.float32)
hd = dense.body(idx).astype(mx.float32)
rr_d = Reranker(dense)
sd = rr_d(idx); mx.eval(ld, hd, sd)
ref_head = dict(tree_flatten(rr_d.head.parameters()))

def kl(a, b):
    pa = mx.softmax(a, -1); return float(mx.mean(mx.sum(pa * (mx.log(pa + 1e-12) - mx.log(mx.softmax(b, -1) + 1e-12)), -1)))

for preset in ("reduction", "compression"):
    path = K + "files_0110/%s_%s.rwkvq" % (SCALE, preset)
    m, cfg, info = load_rwkvq_model(path, rank=8, verbose=False)
    lq = m(idx).astype(mx.float32)
    hq = m.body(idx).astype(mx.float32)
    cos = mx.sum(hq[:, -1] * hd[:, -1], -1) / (mx.linalg.norm(hq[:, -1], axis=-1) * mx.linalg.norm(hd[:, -1], axis=-1))
    print(f"[{preset}] forward KL={kl(ld, lq):.5f} top1={float(mx.mean(mx.argmax(ld,-1)==mx.argmax(lq,-1))):.3f} "
          f"body cos(last)={np.round(np.array(cos),5)}")
    em = EmbeddingModel(m); e = em(idx, mx.array([95, 95])); mx.eval(e)
    print(f"[{preset}] embedding ok shape={e.shape} finite={bool(mx.all(mx.isfinite(e)))}")
    import mlx.nn as nn
    def loss(mm): return nn.losses.cross_entropy(mm(idx[:, :-1]).astype(mx.float32), idx[:, 1:]).mean()
    l, g = nn.value_and_grad(m, loss)(m)
    gn = sum(float(mx.sum(v.astype(mx.float32) ** 2)) for _, v in tree_flatten(g))
    print(f"[{preset}] qlora loss={float(l):.4f} grad-norm^2={gn:.3e} trainable={info.get('trainable_params', info)}"[:300])
    try:
        rr = Reranker(m); s = rr(idx); mx.eval(s)
        hp = dict(tree_flatten(rr.head.parameters()))
        miss = sorted(k for k in ref_head if k not in hp)
        far = []
        for k, v in ref_head.items():
            if k in hp and v.shape == hp[k].shape:
                d = float(mx.max(mx.abs(hp[k].astype(mx.float32) - v.astype(mx.float32))))
                sc = float(mx.max(mx.abs(v.astype(mx.float32)))) + 1e-12
                if d / sc > 0.05 and not k.startswith(('probe','score')): far.append((k, round(d / sc, 3)))
        print(f"[{preset}] reranker score={np.round(np.array(s),4)} dense={np.round(np.array(sd),4)} "
              f"head keys missing={len(miss)} far-from-dense={len(far)} {far[:6]}")
    except Exception as ex:
        print(f"[{preset}] reranker FAILED: {type(ex).__name__}: {ex}")
    m0, _, _ = load_rwkvq_model(path, rank=0, verbose=False)
    l0 = m0(idx).astype(mx.float32); mx.eval(l0)
    print(f"[{preset}] rank=0 max|dlogit| vs rank=8 = {float(mx.max(mx.abs(l0 - lq))):.3e}")
    del m, m0; mx.clear_cache()
