"""Гейт: RwkvqNativeLinear хранит scale / bias в fp16, не меняя ни одного веса (08.10).

Было: fp32, 319 МиБ на 1.5B COMPRESSION при том, что значения sb6 и так округлены в fp16. Стало: fp16 везде, где
деквант от этого побитно не меняется; тензоры, где scale на зажиме 1e-8 при ненулевых кодах, остаются в fp32.
Свойства (FILE, по умолчанию 1.5B COMPRESSION из files_0110 -- на нём есть зажатые блоки cmix.key):
  N1 деквант КАЖДОГО native-тензора (dense_weight) == mx.dequantize тех же кодов с fp32 scale / bias, как до правки
     (_codes_scale_bias), побитно. Не codec: у зажатых блоков (scale = 1e-8, произведение неточно) Metal считает
     q*s+b через fma, numpy -- с двумя округлениями, разница в 1 ulp была и до правки; печатается числом;
  N2 в fp16 не меньше 90% native-тензоров и выход quantized_matmul на fp32-входе у fp16-тензора == у его fp32-копии.
Мутации (--mutate): перевод в fp16 без проверки (N1); перевода нет (N2).
    python tests/test_native_scale_fp16.py [--mutate]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
from rwkv_quant.formats import codec
from rwkv_metal.lora import add_rwkvq as AR, rwkvq_linear as RL, rwkvq_native as RN
from rwkv_metal.lora.lora import dense_weight

FILE = os.path.expanduser(os.environ.get("FILE", "~/Develop/WKV-kvant/files_0110/1p5b_compression.rwkvq"))
OK = {"r_proj": "receptance", "k_proj": "key", "v_proj": "value", "o_proj": "output"}


def natives(m):
    out = {}
    for li, blk in enumerate(m.blocks):
        for n, o in OK.items():
            out["blocks.%d.att.%s.weight" % (li, o)] = getattr(blk.tmix, n)
        for n in ("key", "value"):
            out["blocks.%d.ffn.%s.weight" % (li, n)] = getattr(blk.cmix, n)
    out["head.weight"] = m.head
    return {k: v for k, v in out.items() if isinstance(v, RN.RwkvqNativeLinear)}


def props():
    R = []
    def check(n, ok, info=""): R.append((n, bool(ok), str(info)[:300]))
    RL._SIDECAR_CACHE.clear()
    m = AR.load_rwkvq_model(FILE, rank=0, verbose=False)[0]
    man, arr = codec.open_rwkvq(FILE)
    N = natives(m)
    bad, ulp = [], 0
    for k, mod in N.items():
        _, sc, bi = RN._codes_scale_bias(RL.RwkvqLinear.from_sidecar(FILE, k))
        ref = mx.dequantize(mod.wq, mx.array(sc), mx.array(bi), group_size=RN.GROUP_SIZE, bits=mod.bits)
        got = dense_weight(mod)
        if not bool(mx.array_equal(got, ref)): bad.append(k)
        ulp += int(np.sum(np.array(ref) != np.asarray(codec.dequant_key(man, arr, k), dtype=np.float32)))
    RL._SIDECAR_CACHE.clear()
    check("N1 деквант всех %d native-тензоров == fp32-scale побитно (с codec расходится %d эл. зажатых блоков, как и до правки)"
          % (len(N), ulp), not bad and len(N) > 0, bad[:4])
    f16 = [k for k, mod in N.items() if mod.scale.dtype == mx.float16]
    x = mx.random.normal((2, 5, N[f16[0]].in_features)) if f16 else None
    same = True
    for k in f16[:6] + f16[-6:]:
        mod = N[k]; xx = mx.random.normal((2, 5, mod.in_features), key=mx.random.key(1))
        kw = dict(transpose=True, group_size=RN.GROUP_SIZE, bits=mod.bits)
        a = mx.quantized_matmul(xx, mod.wq, mod.scale, mod.bias, **kw)
        b = mx.quantized_matmul(xx, mod.wq, mod.scale.astype(mx.float32), mod.bias.astype(mx.float32), **kw)
        same &= bool(mx.array_equal(a, b))
    check("N2 в fp16 %d из %d (>= 90%%), выход == fp32-копии" % (len(f16), len(N)), len(f16) >= 0.9 * len(N) and same,
          "fp16 %d/%d same=%s" % (len(f16), len(N), same))
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
        init0 = RN.RwkvqNativeLinear.__init__
        eq0 = mx.array_equal
        def m_nocheck():
            def f(self, lin):
                init0(self, lin)
                self.scale, self.bias = self.scale.astype(mx.float16), self.bias.astype(mx.float16)
            RN.RwkvqNativeLinear.__init__ = f
        def m_never():
            def f(self, lin):
                init0(self, lin)
                self.scale, self.bias = self.scale.astype(mx.float32), self.bias.astype(mx.float32)
            RN.RwkvqNativeLinear.__init__ = f
        for name, apply, expect in (("fp16 без проверки", m_nocheck, "N1"), ("перевода нет", m_never, "N2")):
            apply()
            try:
                failed = run("МУТАЦИЯ «%s»" % name)
            finally:
                RN.RwkvqNativeLinear.__init__ = init0
            caught = any(x.startswith(expect) for x in failed)
            print("  -> %s (ожидалось падение %s)" % ("ПОЙМАНА" if caught else "НЕ ПОЙМАНА", expect))
            ok &= caught
    print("ИТОГ: %s" % ("ЗЕЛЁНЫЙ" if ok else "КРАСНЫЙ"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
