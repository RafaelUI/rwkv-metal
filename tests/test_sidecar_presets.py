"""Гейт: прежний вход через сайдкар export_mlx открывает нынешние пресеты (08.10).

Было: load_lora_rwkvq_model(pth, сайдкар) падал на обоих пресетах -- плотная цель (o_proj слоя 0) не распознавалась
в сайдкаре («раскладка 'dense' не поддержана»), а sym-тензоры REDUCTION не собирались в объект («sym-буферов нет»).
Стало: kind берётся из манифеста сайдкара, плотная цель читается из ::dense, sym собирается
SymQuantLinear.from_interleaved (rwkv-quant) из готового интерлива.
Свойства (0.1B обоих пресетов из files_0110; сайдкары пишутся в /tmp/rm_sidecar_gate и перезаписываются, не удаляются):
  S1 load_lora_rwkvq_model(pth, сайдкар) грузится, логиты ПОБИТНО равны load_lora_rwkvq_model(pth, .rwkvq);
  S2 деквант sym-базы из сайдкара == из .rwkvq побитно на всех sym-тензорах.
Мутации (--mutate): kind плотной цели из сайдкара не читается (S1 compression); sym из сайдкара не собирается (S1/S2 reduction).
    python tests/test_sidecar_presets.py [--mutate]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
from rwkv_metal.lora import add_rwkvq as AR, rwkvq_linear as RL
from rwkv_metal.lora import load_lora_rwkvq_model

W = os.path.expanduser("~/Develop/WKV-kvant/")
CK = W + "rwkv7-g1d-0.1b.pth"
OUT = "/tmp/rm_sidecar_gate"
X = mx.array((np.arange(1, 97, dtype=np.int32) * 37 % 60000)[None])


def sidecar(preset):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "0p1b_" + preset)
    if not os.path.exists(p + ".json"):
        from rwkv_quant.formats.export_mlx import export
        export(W + "files_0110/0p1b_%s.rwkvq" % preset, p)
    return p


def logits(src):
    RL._SIDECAR_CACHE.clear()
    mx.random.seed(0)
    m, _, _ = load_lora_rwkvq_model(CK, src, rank=4, verbose=False)
    l = m(X).astype(mx.float32); mx.eval(l)
    return np.array(l)


def props():
    R = []
    def check(n, ok, info=""): R.append((n, bool(ok), str(info)[:300]))
    for preset in ("reduction", "compression"):
        sc, rq = sidecar(preset), W + "files_0110/0p1b_%s.rwkvq" % preset
        try:
            a, b = logits(rq), logits(sc)
            check("S1 %s: pth + сайдкар == pth + .rwkvq побитно" % preset, np.array_equal(a, b), "max|d| %.3e" % np.abs(a - b).max())
        except Exception as e:
            check("S1 %s: pth + сайдкар грузится" % preset, False, "%s: %s" % (type(e).__name__, e))
        if preset == "reduction":
            RL._SIDECAR_CACHE.clear()
            try:
                _, man = RL.load_sidecar(sc)
                keys = [k for k, m in man["tensors"].items() if m.get("kind") == "sym"]
                bad = [k for k in keys if not bool(mx.array_equal(RL.RwkvqSymLinear.from_sidecar(sc, k)._dequant_w_as(mx.float32),
                                                                   RL.RwkvqSymLinear.from_sidecar(rq, k)._dequant_w_as(mx.float32)))]
                check("S2 sym из сайдкара == из .rwkvq, %d тензоров" % len(keys), keys and not bad, bad[:3])
            except Exception as e:
                check("S2 sym из сайдкара", False, "%s: %s" % (type(e).__name__, e))
            RL._SIDECAR_CACHE.clear()
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
        fk0, ls0 = AR._file_kind, RL.load_sidecar
        def m_kind():
            AR._file_kind = lambda p, k: fk0(p, k) if os.path.isfile(os.path.expanduser(p)) else None
        def m_sym():
            def f(path):
                a, m = ls0(path)
                if not path.endswith(".rwkvq"):
                    a = {k: v for k, v in a.items() if not k.endswith("::sym")}
                    RL._SIDECAR_CACHE[path] = (a, m)
                return a, m
            RL.load_sidecar = f
        for name, apply, expect in (("kind плотной цели из сайдкара не читается", m_kind, "S1 compression"),
                                    ("sym из сайдкара не собирается", m_sym, "S1 reduction")):
            apply()
            try:
                failed = run("МУТАЦИЯ «%s»" % name)
            finally:
                AR._file_kind, RL.load_sidecar = fk0, ls0
            caught = any(x.startswith(expect) for x in failed)
            print("  -> %s (ожидалось падение %s)" % ("ПОЙМАНА" if caught else "НЕ ПОЙМАНА", expect))
            ok &= caught
    print("ИТОГ: %s" % ("ЗЕЛЁНЫЙ" if ok else "КРАСНЫЙ"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
