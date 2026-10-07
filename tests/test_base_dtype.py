"""Гейт типа квантованной базы (07.10): деквант и матмул слоя в fp16, обратный проход в bf16.

Было: база bf16 (деквант округлялся в bf16, матмул слоя шёл в bf16). На REDUCTION это стоило KL к исходной модели
+5...+10% против fp16 (замер tests/_sess/probe_base_dtype_0710.py); норма rwkv-quant -- fp16 с 02.10.
Стало: rl.BASE_DTYPE = fp16 (прямой проход), rl.BWD_DTYPE = bf16 (обратный своим VJP: fp16 насквозь портит градиенты
-- узкая экспонента). Прежнее поведение -- rl.BASE_DTYPE = mx.bfloat16.

Свойства (SYM -- файл пресета reduction, SB6 -- compression, CK -- их исходный .pth; по умолчанию 0.1B, files_0110):
  B1 умолчания модуля: база fp16, обратный проход bf16 (если переменные окружения не заданы);
  B2 по слою каждого вида (sym 6 бит, sym 8 бит, плотная цель, sb6): деквант -- fp16 и ПОБИТНО равен нормативной читалке
     codec.dequant_key, округлённой numpy в float16 (независимый источник); прямой проход побитно равен
     (x.astype(fp16) @ W.T).astype(x.dtype) при входе bf16 и fp32;
  B3 обратный проход: градиент по x побитно равен (c.astype(bf16) @ W_bf16).astype(x.dtype), где W_bf16 -- читалка,
     округлённая в bf16; и НЕ равен градиенту fp16-автограда (вовлечение своего VJP);
  B4 прежний режим: rl.BASE_DTYPE = bf16 -- прямой проход побитно равен (x.astype(bf16) @ W_bf16.T).astype(x.dtype);
  B5 (модель) градиенты адаптеров, отн. L2 к плечу без каста входа: умолчание < прежнего bf16 < fp16 насквозь;
  B6 (модель) KL(истина || модель) на окнах: умолчание ниже прежнего bf16 не меньше чем на 2% (замерено -5.4% [-7.3; -3.5]);
  B7 (модель) шаг обучения в умолчании: потери и градиенты конечны, градиент ненулевой.
Мутации (--mutate; без модельных свойств, кроме B5 у второй): умолчание базы возвращено в bf16 (B1); обратный проход
в типе базы (B3); двойное округление декванта bf16 -> fp16 (B2); обратный проход на весах, округлённых дважды (B3).
    python tests/test_base_dtype.py [--mutate]      SKIP_MODEL=1 -- без B5-B7"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from rwkv_quant.formats import codec
from rwkv_metal.lora import rwkvq_linear as rl

W = os.path.expanduser("~/Develop/WKV-kvant/")
SYM = os.path.expanduser(os.environ.get("SYM", W + "files_0110/0p1b_reduction.rwkvq"))
SB6 = os.path.expanduser(os.environ.get("SB6", W + "files_0110/0p1b_compression.rwkvq"))
CK = os.path.expanduser(os.environ.get("CK", W + "rwkv7-g1d-0.1b.pth"))
ENV_FREE = "RWKVQ_BASE_DTYPE" not in os.environ and "RWKVQ_BWD_DTYPE" not in os.environ
DEF = (rl.BASE_DTYPE, rl.BWD_DTYPE)
f32 = lambda a: np.array(a.astype(mx.float32))


def layers():
    """(имя, модуль, нормативный fp32-вес из читалки) -- по представителю каждого вида."""
    out = []
    man, buf = codec.open_rwkvq(SYM)
    t = man["tensors"]
    def first(pred): return next(k for k in sorted(t) if pred(k, t[k]))
    for name, key, cls in (("sym 6 бит", first(lambda k, m: m["kind"] == "sym" and m["bits"] == 6 and m["shape"][1] % 256 == 0), rl.RwkvqSymLinear),
                           ("sym 8 бит", first(lambda k, m: m["kind"] == "sym" and m["bits"] == 8 and m["shape"][1] % 256 == 0), rl.RwkvqSymLinear),
                           ("плотная цель", "blocks.0.att.output.weight", rl.RwkvqDenseLinear)):
        out.append((name + " " + key, cls.from_sidecar(SYM, key), codec.dequant_key(man, buf, key).astype(np.float32)))
    man, buf = codec.open_rwkvq(SB6)
    key = next(k for k in sorted(man["tensors"]) if man["tensors"][k]["kind"] == "sb6" and "att.key" in k)
    out.append(("sb6 " + key, rl.RwkvqLinear.from_sidecar(SB6, key), codec.dequant_key(man, buf, key).astype(np.float32)))
    return out


def model_props(check):
    import torch
    from rwkv_metal.lora import load_rwkvq_model
    from rwkv_quant.models.rwkv7_ref import RWKV7Ref
    ev = torch.load(W + "eval_text_heldout.pt", weights_only=False)["tokens"]
    wins = [ev[int(i), :256] for i in np.linspace(0, ev.shape[0] - 1, 12).round()]
    MO = RWKV7Ref(CK, device="cpu", dtype=torch.bfloat16, compute_dtype=torch.float32)
    with torch.no_grad(): truth = [torch.log_softmax(MO.forward(w[None], cfg=None).float(), -1)[0] for w in wins]
    del MO
    rl._SIDECAR_CACHE.clear()
    model, _, _ = load_rwkvq_model(SYM, rank=16, verbose=False)
    ARMS = {"умолчание": DEF + (False,), "bf16": (mx.bfloat16, None, False), "fp16 насквозь": (mx.float16, None, False), "эталон": (mx.float16, None, True)}
    def use(a): rl.BASE_DTYPE, rl.BWD_DTYPE, rl.NOCAST = ARMS[a]
    def kl_of(a):
        use(a); ks = []
        for w, tr in zip(wins, truth):
            lg = model(mx.array(w.numpy().astype(np.int32)[None])); lg = (lg[0] if lg.ndim == 3 else lg).astype(mx.float32); mx.eval(lg)
            q = torch.log_softmax(torch.as_tensor(np.array(lg)), -1); ks.append(float((tr.exp() * (tr - q)).sum(-1).mean()))
        return float(np.mean(ks))
    kd, kb = kl_of("умолчание"), kl_of("bf16")
    check("B6 KL умолчания ниже прежнего bf16 не меньше чем на 2%", kd < kb * 0.98, "умолчание %.6f, bf16 %.6f (%+.2f%%)" % (kd, kb, 100 * (kd / kb - 1)))
    rs = np.random.RandomState(3)
    upd = [(k, (v.astype(mx.float32) + mx.array(rs.normal(0, 1e-3, v.shape).astype(np.float32))).astype(v.dtype))
           for k, v in tree_flatten(model.trainable_parameters()) if k.endswith("lora_b")]
    model.update(tree_unflatten(upd)); mx.eval(model.parameters())
    w = wins[0]; x = mx.array(w[:-1].numpy().astype(np.int32)[None]); y = mx.array(w[1:].numpy().astype(np.int32)[None])
    gf = nn.value_and_grad(model, lambda m, a, b: m.loss(a, b).astype(mx.float32))
    G, L = {}, {}
    for a in ARMS:
        use(a); loss, g = gf(model, x, y); flat = dict(tree_flatten(g)); mx.eval(loss, flat)
        L[a] = float(loss); G[a] = np.concatenate([f32(flat[k]).ravel() for k in sorted(flat)])
    err = {a: float(np.linalg.norm(G[a] - G["эталон"]) / np.linalg.norm(G["эталон"])) for a in ARMS}
    check("B5 градиенты: умолчание < bf16 < fp16 насквозь (отн. L2 к эталону)", err["умолчание"] < err["bf16"] < err["fp16 насквозь"],
          {a: "%.3e" % e for a, e in err.items()})
    check("B7 шаг в умолчании: потери и градиенты конечны, градиент ненулевой",
          np.isfinite(L["умолчание"]) and np.isfinite(G["умолчание"]).all() and np.linalg.norm(G["умолчание"]) > 0, (L["умолчание"],))
    rl.BASE_DTYPE, rl.BWD_DTYPE = DEF; rl.NOCAST = False
    del model; mx.clear_cache()


def props(with_model=True):
    R = []
    def check(n, ok, info=""): R.append((n, bool(ok), str(info)[:300]))
    if ENV_FREE:
        check("B1 умолчания: база fp16, обратный проход bf16", rl.BASE_DTYPE == mx.float16 and rl.BWD_DTYPE is not None and rl.BWD_DTYPE == mx.bfloat16, (rl.BASE_DTYPE, rl.BWD_DTYPE))
    rs = np.random.RandomState(7)
    for name, lin, w32 in layers():
        w16 = w32.astype(np.float16)
        got = lin._dequant_w(); mx.eval(got)
        check("B2 %s: деквант fp16, побитно == читалке в float16" % name,
              got.dtype == mx.float16 and np.array_equal(np.array(got).view(np.uint16), w16.view(np.uint16)),
              (got.dtype, int((np.array(got.astype(mx.float16)).view(np.uint16) != w16.view(np.uint16)).sum())))
        wm16 = mx.array(w16); wb = mx.array(w32).astype(mx.bfloat16)
        xs = mx.array(rs.randn(4, lin.in_features).astype(np.float32)); c = mx.array(rs.randn(4, lin.out_features).astype(np.float32) * 1e-4)
        bad = []
        for xd in (mx.bfloat16, mx.float32):
            x = xs.astype(xd); y = lin(x); ref = (x.astype(mx.float16) @ wm16.T).astype(xd); mx.eval(y, ref)
            if y.dtype != xd or not np.array_equal(f32(y), f32(ref)): bad.append(str(xd))
        check("B2 %s: прямой проход побитно == (x.fp16 @ W.T)" % name, not bad, bad)
        x = xs
        g = mx.grad(lambda z: (lin(z).astype(mx.float32) * c).sum())(x)
        ref = (c.astype(mx.bfloat16) @ wb).astype(mx.float32)
        auto = mx.grad(lambda z: ((z.astype(mx.float16) @ wm16.T).astype(mx.float32) * c).sum())(x)
        mx.eval(g, ref, auto)
        check("B3 %s: обратный проход побитно == (c.bf16 @ W_bf16)" % name, np.array_equal(f32(g), f32(ref)), float(np.abs(f32(g) - f32(ref)).max()))
        check("B3 %s: свой VJP вовлечён (не равен fp16-автограду)" % name, not np.array_equal(f32(g), f32(auto)))
        rl.BASE_DTYPE, rl.BWD_DTYPE = mx.bfloat16, None
        try:
            xb = xs.astype(mx.bfloat16); y = lin(xb); ref = (xb @ wb.T); mx.eval(y, ref)
            check("B4 %s: прежний режим bf16 побитно == (x.bf16 @ W_bf16.T)" % name, y.dtype == mx.bfloat16 and np.array_equal(f32(y), f32(ref)))
        finally:
            rl.BASE_DTYPE, rl.BWD_DTYPE = DEF
    if with_model and os.environ.get("SKIP_MODEL") != "1":
        model_props(check)
    return R


def run(label, with_model=True):
    R = props(with_model)
    failed = [n for n, ok, _ in R if not ok]
    for n, ok, info in R:
        print("  [%s] %s%s" % ("OK" if ok else "FAIL", n, "" if ok else " -- " + info))
    print("%s: %d свойств, провалов %d" % (label, len(R), len(failed)), flush=True)
    return failed


def main():
    global DEF
    ok = not run("КОНТРОЛЬ")
    if "--mutate" in sys.argv:
        sym_as = rl.RwkvqSymLinear._dequant_w_as; build = rl._build_wide_bwd_fn
        def m_default():
            global DEF
            rl.BASE_DTYPE = mx.bfloat16; DEF = (rl.BASE_DTYPE, rl.BWD_DTYPE)
        def m_bwd_same():
            global DEF
            rl.BWD_DTYPE = None; DEF = (rl.BASE_DTYPE, rl.BWD_DTYPE)
        def m_double(): rl.RwkvqSymLinear._dequant_w_as = lambda self, dt: sym_as(self, mx.bfloat16).astype(dt)
        def m_bwd_double():
            def b(mod):
                @mx.custom_function
                def _f(x):
                    w = mod._dequant_w(); return (x.astype(w.dtype) @ w.T).astype(x.dtype)
                @_f.vjp
                def _v(primals, cotangent, output):
                    x = primals[0] if isinstance(primals, (list, tuple)) else primals
                    ct = cotangent[0] if isinstance(cotangent, (list, tuple)) else cotangent
                    return ((ct.astype(mx.bfloat16) @ mod._dequant_w().astype(mx.bfloat16)).astype(x.dtype),)
                return _f
            rl._build_wide_bwd_fn = b
        muts = [("умолчание базы возвращено в bf16", m_default, "B1"), ("обратный проход в типе базы", m_bwd_same, "B3"),
                ("двойное округление декванта bf16 -> fp16", m_double, "B2"), ("обратный проход на дважды округлённых весах", m_bwd_double, "B3")]
        saved_def = DEF
        for name, apply, expect in muts:
            apply()
            try:
                failed = run("МУТАЦИЯ «%s»" % name, with_model=False)
            finally:
                DEF = saved_def; rl.BASE_DTYPE, rl.BWD_DTYPE = DEF
                rl.RwkvqSymLinear._dequant_w_as = sym_as; rl._build_wide_bwd_fn = build
            caught = any(f.startswith(expect) for f in failed)
            print("  -> %s (ожидалось падение %s)" % ("ПОЙМАНА" if caught else "НЕ ПОЙМАНА", expect))
            ok &= caught
        ok &= not run("КОНТРОЛЬ ПОСЛЕ МУТАЦИЙ", with_model=False)
    print("ИТОГ: %s" % ("ЗЕЛЁНЫЙ" if ok else "КРАСНЫЙ"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
