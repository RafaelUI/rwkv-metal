"""Гейт: ранги low-rank веток берутся по ФОРМАМ чекпоинта, а не по формуле (07.10).

Было: RWKV7X070 строился с рангами lora_ranks(D); у rwkv7-g1d-0.4b (D=1024) ветка g имеет ранг 128, формула даёт 160 --
ни .pth, ни .rwkvq этой модели не открывались («конверсия не чистая»).
Свойства (CKS -- .pth через «:», по умолчанию 0.1B и 0.4B; Q04 -- .rwkvq 0.4B обоих пресетов):
  R1 ранги из ленивого разбора .pth (convert._ranks_from_shapes) == рангам, снятым torch с того же файла (независимо);
     среди чекпоинтов есть хотя бы один, где формула расходится с файлом (иначе гейт не исполняет то, ради чего написан);
  R2 каждый .pth загружается load_pretrained; плотная модель сходится с эталоном RWKV7Ref (веса bf16, счёт fp32) на
     256 токенах: top-1 совпал не менее чем на 97% позиций, KL < 2e-3 (замерено: см. вывод; порог -- на порядок
     ниже перепутанной проводки, которая даёт KL порядка единиц);
  R3 .rwkvq 0.4B обоих пресетов загружаются load_rwkvq_model, ранги те же, логиты конечны.
Мутации (--mutate): ранги снова по формуле (R2 на 0.4B); ранги w и v перепутаны (R1).
    python tests/test_ranks_from_shapes.py [--mutate]"""
import importlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import mlx.core as mx
CV = importlib.import_module("rwkv_metal.model.convert")
from rwkv_metal.model.rwkv7_x070 import lora_ranks
from rwkv_metal.lora import load_rwkvq_model, rwkvq_linear as rl

W = os.path.expanduser("~/Develop/WKV-kvant/")
CKS = [os.path.expanduser(p) for p in os.environ.get("CKS", W + "rwkv7-g1d-0.1b.pth:" + W + "rwkv7-g1d-0.4b-20260210-ctx8192.pth").split(":")]
Q04 = [os.path.expanduser(p) for p in os.environ.get("Q04", W + "files_0110/0p4b_reduction.rwkvq:" + W + "files_0110/0p4b_compression.rwkvq").split(":")]


def props():
    R = []
    def check(n, ok, info=""): R.append((n, bool(ok), str(info)[:300]))
    from rwkv_quant.models.rwkv7_ref import RWKV7Ref
    ev = torch.load(W + "eval_text_heldout.pt", weights_only=False)["tokens"][0, :256]
    x = mx.array(ev.numpy().astype(np.int32)[None])
    differs, real04 = 0, None
    for ck in CKS:
        tag = os.path.basename(ck)
        z = torch.load(ck, map_location="cpu", mmap=True, weights_only=True)
        D = z["emb.weight"].shape[1]; n_layer = 1 + max(int(k.split(".")[1]) for k in z if k.startswith("blocks."))
        real = {k: int(min(z["blocks.1.att.%s1" % k].shape)) for k in "wavg"}
        zl = CV._load_pth_lazy(ck)
        got = CV._ranks_from_shapes(lambda k: zl[k].shape if k in zl else None, n_layer, D)
        check("R1 %s: ранги по формам == снятым torch %s" % (tag, real), got == real, got)
        differs += real != lora_ranks(D)
        if D == 1024: real04 = real
        try:
            m, _ = CV.load_pretrained(ck, verbose=False)
        except Exception as e:
            check("R2 %s загружается и сходится с эталоном" % tag, False, "%s: %s" % (type(e).__name__, str(e)[:160])); continue
        if m is None:
            check("R2 %s загружается и сходится с эталоном" % tag, False, "конверсия не чистая"); continue
        lg = m(x).astype(mx.float32); mx.eval(lg); q = torch.log_softmax(torch.as_tensor(np.array(lg)[0]), -1)
        MO = RWKV7Ref(ck, device="cpu", dtype=torch.bfloat16, compute_dtype=torch.float32)
        with torch.no_grad(): p = torch.log_softmax(MO.forward(ev[None], cfg=None).float(), -1)[0]
        kl = float((p.exp() * (p - q)).sum(-1).mean()); top = float((p.argmax(-1) == q.argmax(-1)).float().mean())
        check("R2 %s загружается и сходится с эталоном (KL %.2e, top-1 %.1f%%)" % (tag, kl, 100 * top), kl < 2e-3 and top >= 0.97 and m.ranks == real, (kl, top, m.ranks))
        del m, MO; mx.clear_cache()
    check("R1 есть чекпоинт, где формула расходится с файлом", differs >= 1, differs)
    for f in Q04:
        rl.drop_sidecar_cache()
        try:
            m, _, _ = load_rwkvq_model(f, verbose=False)
            lg = m(x).astype(mx.float32); mx.eval(lg)
            check("R3 %s загружается, ранги по формам, логиты конечны" % os.path.basename(f), m.ranks == real04 and bool(mx.all(mx.isfinite(lg))), m.ranks)
            del m; mx.clear_cache()
        except Exception as e:
            check("R3 %s загружается, ранги по формам, логиты конечны" % os.path.basename(f), False, "%s: %s" % (type(e).__name__, str(e)[:160]))
    return R


def run(label):
    R = props()
    failed = [n for n, ok, _ in R if not ok]
    for n, ok, info in R:
        print("  [%s] %s%s" % ("OK" if ok else "FAIL", n, "" if ok else " -- " + info))
    print("%s: %d свойств, провалов %d" % (label, len(R), len(failed)), flush=True)
    return failed


def main():
    ok = not run("КОНТРОЛЬ")
    if "--mutate" in sys.argv:
        orig = CV._ranks_from_shapes
        def m_swap(shape_of, n_layer, D):
            r = orig(shape_of, n_layer, D); r["w"], r["v"] = r["v"], r["w"]; return r
        muts = [("ранги снова по формуле", lambda: setattr(CV, "_ranks_from_shapes", lambda s, n, D: dict(lora_ranks(D))), "R2 rwkv7-g1d-0.4b"),
                ("ранги w и v перепутаны", lambda: setattr(CV, "_ranks_from_shapes", m_swap), "R1")]
        for name, apply, expect in muts:
            apply()
            try:
                failed = run("МУТАЦИЯ «%s»" % name)
            finally:
                CV._ranks_from_shapes = orig
            caught = any(f.startswith(expect) for f in failed)
            print("  -> %s (ожидалось падение %s)" % ("ПОЙМАНА" if caught else "НЕ ПОЙМАНА", expect))
            ok &= caught
        ok &= not run("КОНТРОЛЬ ПОСЛЕ МУТАЦИЙ")
    print("ИТОГ: %s" % ("ЗЕЛЁНЫЙ" if ok else "КРАСНЫЙ"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
