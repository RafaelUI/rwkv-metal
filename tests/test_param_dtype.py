"""Гейт: тип плотных параметров модели из .rwkvq -- param_dtype загрузчика (08.10).

Было: всё, что не подменяется квантованным модулем (emb, нормировки, миксы, low-rank ветки, small), приезжало в
bf16, и от него зависел счёт всей неквантованной части модели. Стало: fp16 по умолчанию (тот же размер, деквант
файла в нём точен), "bf16" / "fp32" -- по param_dtype= или RWKVQ_PARAM_DTYPE. Замеры -- model/convert.py.

Свойства (0.1B обоих пресетов из files_0110; CK -- исходный .pth для истины):
  D1 умолчание -- fp16, param_dtype="bf16"/"fp32" дают свой тип у emb, ln_x, w_lora_A, k_k;
  D2 fp16-параметры равны декванту codec, округлённому в fp16 ОДИН раз (emb, w1, ln_x): без двойного округления
     через bf16 (свойство чувствительно: у emb доля элементов, где fp16(bf16(v)) != fp16(v), печатается и > 0);
  D3 RWKVQ_PARAM_DTYPE=bf16 соблюдается, когда param_dtype не задан; явный param_dtype побеждает переменную;
  D4 неизвестный тип -- ValueError до загрузки;
  D5 KL к истине (RWKV7Ref, .pth, счёт fp32; 4 окна отложенного текста по 128) у fp16 меньше, чем у bf16, на REDUCTION.
Мутации (--mutate): деквант игнорирует dtype (всегда bf16) (D1); деквант через bf16 (двойное округление) (D2);
переменная окружения не читается (D3).
    python tests/test_param_dtype.py [--mutate]"""
import importlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
from rwkv_quant.formats import codec
from rwkv_metal.lora import add_rwkvq as AR, rwkvq_linear as RL
CV = importlib.import_module("rwkv_metal.model.convert")

W = os.path.expanduser("~/Develop/WKV-kvant/")
FILES = {"reduction": W + "files_0110/0p1b_reduction.rwkvq", "compression": W + "files_0110/0p1b_compression.rwkvq"}
CK = os.path.expanduser(os.environ.get("CK", W + "rwkv7-g1d-0.1b.pth"))
_TRUTH = {}


def load(f, **kw):
    RL._SIDECAR_CACHE.clear()
    return AR.load_rwkvq_model(f, rank=0, verbose=False, **kw)[0]


def dts(m):
    t = m.blocks[3].tmix
    return {"emb": m.emb.weight.dtype, "ln_x": t.ln_x.weight.dtype, "w_lora_A": t.w_lora_A.weight.dtype, "k_k": t.k_k.dtype}


def truth_and_wins():
    if not _TRUTH:
        import torch
        from rwkv_quant.models.rwkv7_ref import RWKV7Ref
        ev = torch.load(W + "eval_text_heldout.pt", weights_only=False)["tokens"]
        wins = [ev[i, :128] for i in (0, 9, 18, 27)]
        MO = RWKV7Ref(CK, device="cpu", dtype=torch.bfloat16, compute_dtype=torch.float32)
        with torch.no_grad():
            _TRUTH["t"] = [torch.log_softmax(MO.forward(w[None], cfg=None).float(), -1)[0].numpy() for w in wins]
        _TRUTH["w"] = [w.numpy().astype(np.int32) for w in wins]
    return _TRUTH["t"], _TRUTH["w"]


def kl_of(m):
    tr, ws = truth_and_wins()
    out = []
    for t, w in zip(tr, ws):
        lg = m(mx.array(w[None])).astype(mx.float32)[0]
        lq = np.array(lg - mx.logsumexp(lg, -1, keepdims=True))
        out.append(float((np.exp(t) * (t - lq)).sum(-1).mean()))
    return float(np.mean(out))


def props():
    R = []
    def check(n, ok, info=""): R.append((n, bool(ok), str(info)[:300]))
    for preset, f in FILES.items():
        man, arr = codec.open_rwkvq(f)
        saved = os.environ.pop("RWKVQ_PARAM_DTYPE", None)
        try:
            ms = {d: load(f, param_dtype=d) for d in (None, "bf16", "fp32")}
            want = {None: mx.float16, "bf16": mx.bfloat16, "fp32": mx.float32}
            bad = [(d, k, str(v)) for d, m in ms.items() for k, v in dts(m).items() if v != want[d]]
            check("D1 %s: умолчание fp16, bf16 / fp32 по param_dtype" % preset, not bad, bad[:4])
            m = ms[None]
            pairs = {"emb.weight": m.emb.weight, "blocks.3.att.ln_x.weight": m.blocks[3].tmix.ln_x.weight}
            w1 = np.asarray(codec.dequant_key(man, arr, "blocks.3.att.w1"), dtype=np.float32)
            got_w1 = np.array(m.blocks[3].tmix.w_lora_A.weight.astype(mx.float32))
            if got_w1.shape != w1.shape:
                w1 = w1.T
            badv, dbl = [], None
            for k, v in pairs.items():
                ref = np.asarray(codec.dequant_key(man, arr, k), dtype=np.float32).reshape(v.shape)
                if not np.array_equal(np.array(v.astype(mx.float32)), ref.astype(np.float16).astype(np.float32)): badv.append(k)
                if k == "emb.weight":
                    dbl = float((np.array(mx.array(ref).astype(mx.bfloat16).astype(mx.float16).astype(mx.float32)) != ref.astype(np.float16).astype(np.float32)).mean())
            if not np.array_equal(got_w1, w1.astype(np.float16).astype(np.float32)): badv.append("w1")
            check("D2 %s: fp16 == codec, округлённый один раз (у emb двойное округление меняло бы %.1f%% элементов)" % (preset, 100 * (dbl or 0)),
                  not badv and (dbl or 0) > 0, badv)
            os.environ["RWKVQ_PARAM_DTYPE"] = "bf16"
            e = load(f).emb.weight.dtype; e2 = load(f, param_dtype="fp32").emb.weight.dtype
            check("D3 %s: RWKVQ_PARAM_DTYPE=bf16 соблюдается, явный param_dtype побеждает" % preset,
                  e == mx.bfloat16 and e2 == mx.float32, (str(e), str(e2)))
            os.environ.pop("RWKVQ_PARAM_DTYPE")
            try:
                load(f, param_dtype="int8"); check("D4 %s: неизвестный тип -- ValueError" % preset, False, "принят")
            except ValueError:
                check("D4 %s: неизвестный тип -- ValueError" % preset, True)
            if preset == "reduction":
                k16, kb = kl_of(ms[None]), kl_of(ms["bf16"])
                check("D5 %s: KL к истине fp16 < bf16" % preset, k16 < kb, "fp16 %.6f bf16 %.6f (%+.2f%%)" % (k16, kb, 100 * (k16 / kb - 1)))
                print("     D5 fp16 %.6f bf16 %.6f (%+.2f%%)" % (k16, kb, 100 * (k16 / kb - 1)))
            del ms, m
            mx.clear_cache()
        finally:
            os.environ.pop("RWKVQ_PARAM_DTYPE", None)
            if saved is not None:
                os.environ["RWKVQ_PARAM_DTYPE"] = saved
    return R


def run(title):
    print("== %s" % title)
    failed = []
    for n, ok, info in props():
        print("  [%s] %s  %s" % ("OK " if ok else "XX ", n, info if not ok else ""))
        if not ok: failed.append(n)
    return failed


def main():
    ok = not run("КОНТРОЛЬ")
    if "--mutate" in sys.argv:
        dq0, rs0 = CV._dequant_to, CV.resolve_param_dtype
        def m_ignore():
            CV._dequant_to = lambda man, buf, key, dtype: dq0(man, buf, key, mx.bfloat16)
        def m_double():
            CV._dequant_to = lambda man, buf, key, dtype: dq0(man, buf, key, mx.bfloat16).astype(dtype)
        def m_noenv():
            def f(pd=None):
                return rs0("fp16" if pd is None else pd)
            CV.resolve_param_dtype = f
        muts = [("деквант игнорирует dtype", m_ignore, "D1"), ("деквант через bf16", m_double, "D2"), ("переменная не читается", m_noenv, "D3")]
        for name, apply, expect in muts:
            apply()
            try:
                failed = run("МУТАЦИЯ «%s»" % name)
            finally:
                CV._dequant_to, CV.resolve_param_dtype = dq0, rs0
            caught = any(x.startswith(expect) for x in failed)
            print("  -> %s (ожидалось падение %s)" % ("ПОЙМАНА" if caught else "НЕ ПОЙМАНА", expect))
            ok &= caught
        ok &= not run("КОНТРОЛЬ ПОСЛЕ МУТАЦИЙ")
    print("ИТОГ: %s" % ("ЗЕЛЁНЫЙ" if ok else "КРАСНЫЙ"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
