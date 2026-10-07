"""07.10: стоит ли порту rwkv-metal перейти с bf16-округления декванта на fp16 (норма rwkv-quant с 02.10, codec.py).
Порт округляет в bf16: линейные слои (RwkvqLinear / RwkvqSymLinear._dequant_w) и плотные тензоры
(convert._dequant_to_bf16: emb, нормировки, LoRA-ветки, миксы). В rwkv-quant то же округление у reader стоило REDUCTION
+2-3% KL. Здесь -- замер на самом порте.
Прибор: KL(истина || плечо) по позициям, истина -- исходный чекпоинт, веса bf16, счёт fp32 (RWKV7Ref, cpu; закон 38).
Плечи (одна модель на плечо, один процесс; адаптеры нулевые и одинаковые):
  asis        порт как есть, native=True (умолчание load_rwkvq_model);
  asis_nn     порт как есть, native=False (sb6 через RwkvqLinear) -- чтобы плечи ниже отличались ОДНИМ;
  r_bf16      веса округлены в bf16, но лежат и считаются в fp32 (отделяет округление от типа счёта);
  r_fp16      веса округлены в fp16, счёт fp32;
  r_fp32      без округления, счёт fp32 (потолок);
  lin_fp16    только линейные слои fp16 (плотные -- bf16); dense_fp16 -- наоборот.
Плюс точка отсчёта: Metal-бэкенд rwkv-quant (QuantRWKV7) на том же файле.
    python probe_dequant_dtype_0710.py <файл.rwkvq> <ckpt.pth> <out.json> [окон=12] [T=256]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch, mlx.core as mx
from rwkv_metal.lora import add_rwkvq as AR, rwkvq_linear as RL
import importlib; CV = importlib.import_module("rwkv_metal.model.convert")
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
FILE, CK, OUT = sys.argv[1:4]
NW = int(sys.argv[4]) if len(sys.argv) > 4 else 12; T = int(sys.argv[5]) if len(sys.argv) > 5 else 256
ev = torch.load(os.path.expanduser("~/Develop/WKV-kvant/eval_text_heldout.pt"), weights_only=False)["tokens"]
rows = [int(i) for i in np.linspace(0, ev.shape[0] - 1, NW).round()]
WINS = [ev[i, :T] for i in rows]
def lsm(a): return torch.log_softmax(torch.as_tensor(np.asarray(a, dtype=np.float32)), -1)
def kl(p, q): return float((p.exp() * (p - q)).sum(-1).mean())
MO = RWKV7Ref(CK, device="cpu", dtype=torch.bfloat16, compute_dtype=torch.float32)
with torch.no_grad(): TRUTH = [torch.log_softmax(MO.forward(w[None], cfg=None).float(), -1)[0] for w in WINS]
del MO
# ОБХОД ТОЛЬКО В ПРОБЕ: с 20.09 пресеты держат o_proj слоя 0 плотным (bits_overrides -> 16), а load_rwkvq_model требует,
# чтобы все цели лежали квантованными, и на любом нынешнем файле пресета падает (находка 07.10, порт не правлен).
import mlx.nn as nn
from rwkv_quant.formats import codec as _codec
class DenseBase(nn.Module):
    def __init__(self, w):
        super().__init__(); self._w = w; self.out_features, self.in_features = w.shape; self.freeze()
    @classmethod
    def from_sidecar(cls, path, key):
        man, buf = _codec.open_rwkvq(os.path.expanduser(path)); return cls(CV._dequant_to_bf16(man, buf, key))
    def __call__(self, x):
        return (x.astype(self._w.dtype) @ self._w.T).astype(x.dtype)
_bf = AR._backend_for
def _backend(path, key, native):
    man, _ = _codec.open_rwkvq(os.path.expanduser(path))
    return DenseBase if man["tensors"][key]["kind"] == "dense" else _bf(path, key, native)
AR._backend_for = _backend; AR._check_quantized_coverage = lambda *a, **k: None
ORIG = (CV._dequant_to_bf16, RL.RwkvqLinear._dequant_w, RL.RwkvqSymLinear._dequant_w)
def rnd(x32, how):
    if how == "fp32": return x32
    return x32.astype(mx.bfloat16 if how == "bf16" else mx.float16).astype(mx.float32)
def patch(lin, dense):
    """lin / dense: None -- как в порте; 'bf16' | 'fp16' | 'fp32' -- округление, хранение и счёт fp32."""
    CV._dequant_to_bf16, RL.RwkvqLinear._dequant_w, RL.RwkvqSymLinear._dequant_w = ORIG
    if dense is not None:
        from rwkv_quant.formats import codec
        def deq(manifest, buf, key):
            parts = []
            for _a, _b, band in codec.dequant_key_bands(manifest, buf, key):
                p = rnd(mx.array(band).astype(mx.float32), dense); mx.eval(p); parts.append(p)
            return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
        CV._dequant_to_bf16 = deq
    if lin is not None:
        def dq_sb6(self):
            return rnd(RL.dequant_dense(self.qblk, self.qsqm, self.ddm, self.out_features, self.in_features,
                                        gw_sb=self._gw_sb, xbits=self.xbits), lin)
        def dq_sym(self):
            return rnd(self._sym._dequant_w(mx.float32), lin)
        RL.RwkvqLinear._dequant_w, RL.RwkvqSymLinear._dequant_w = dq_sb6, dq_sym
def logits(model, w):
    lg = model(mx.array(w.numpy().astype(np.int32)[None])); lg = lg[0] if lg.ndim == 3 else lg
    lg = lg.astype(mx.float32); mx.eval(lg); return np.array(lg)
def arm(name, lin, dense, native):
    patch(lin, dense)
    model, cfg, info = AR.load_rwkvq_model(FILE, verbose=False, native=native)
    kinds = sorted({type(m).__name__ for _, m in model.named_modules() if "Rwkvq" in type(m).__name__ or "Native" in type(m).__name__ or "Hybrid" in type(m).__name__})
    dts = sorted({str(v.dtype) for _, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(model.parameters())})
    ks = [kl(TRUTH[i], lsm(logits(model, w))) for i, w in enumerate(WINS)]
    del model; mx.clear_cache(); patch(None, None)
    print("%-11s KL %.6f  (модули %s; типы параметров %s)" % (name, float(np.mean(ks)), kinds, dts), flush=True)
    return ks
RES = {"file": FILE, "rows": rows, "T": T, "arms": {}}
for name, lin, dense, native in (("asis", None, None, True), ("asis_nn", None, None, False), ("r_bf16", "bf16", "bf16", False),
                                 ("r_fp16", "fp16", "fp16", False), ("r_fp32", "fp32", "fp32", False),
                                 ("lin_fp16", "fp16", "bf16", False), ("dense_fp16", "bf16", "fp16", False)):
    RES["arms"][name] = arm(name, lin, dense, native); json.dump(RES, open(OUT, "w"), indent=1)
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
m = QuantRWKV7(load_raw(FILE)); ks = []
for i, w in enumerate(WINS):
    lg, _ = m.step(mx.array(w.numpy().astype(np.int32)[None]), m.init_state(1)); lg = lg.astype(mx.float32); mx.eval(lg)
    ks.append(kl(TRUTH[i], lsm(np.array(lg)[0])))
RES["arms"]["rwkv_quant_metal"] = ks; json.dump(RES, open(OUT, "w"), indent=1)
print("%-11s KL %.6f" % ("rq_metal", float(np.mean(ks))))
A = {k: np.array(v) for k, v in RES["arms"].items()}
rng = np.random.default_rng(0); idx = rng.integers(0, NW, size=(20000, NW))
def ratio(a, b):
    r = A[a][idx].mean(1) / A[b][idx].mean(1) - 1
    return "%+.2f%% [%+.2f; %+.2f]" % (100 * (A[a].mean() / A[b].mean() - 1), 100 * np.quantile(r, .025), 100 * np.quantile(r, .975))
print("ОТНОШЕНИЯ KL (парный бутстрэп по окнам, 95%%):")
for a, b in (("r_bf16", "r_fp32"), ("r_fp16", "r_fp32"), ("r_bf16", "r_fp16"), ("lin_fp16", "r_bf16"), ("dense_fp16", "r_bf16"),
             ("asis_nn", "r_bf16"), ("asis", "asis_nn"), ("asis", "rwkv_quant_metal"), ("asis", "r_fp16")):
    print("   %-11s / %-16s %s" % (a, b, ratio(a, b)))
