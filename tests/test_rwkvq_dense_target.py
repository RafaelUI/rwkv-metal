"""Гейт: цель QLoRA, лежащая в .rwkvq ПЛОТНОЙ, не мешает загрузке (07.10).

Было: с 20.09 оба пресета rwkv-quant держат o_proj слоя 0 в bf16 (bits_overrides -> 16), а load_rwkvq_model требовал,
чтобы все цели были квантованными, -- ValueError на любом нынешнем файле пресета.
Стало: плотная цель оборачивается RwkvqDenseLinear, LoRA встаёт поверх; rtn / asym / отсутствие ключа -- отказ, как раньше.

Свойства (файлы FILES через «:», по умолчанию 0.1B обоих пресетов из files_0110; CK -- их исходный .pth):
  P1 файл загружается, прямой проход даёт конечные логиты формы [1, T, vocab];
  P2 под o_proj слоя 0 стоит RwkvqDenseLinear, и его вес ПОБИТНО равен blocks.0.att.output.weight исходного .pth
     (независимый источник: torch читает .pth; kind "dense" -- это bf16 как есть);
  P3 плотных баз ровно столько, сколько плотных целей в манифесте; остальные цели -- квантованные модули;
  P4 адаптер над плотной базой обучаем: градиент по lora_b у o_proj слоя 0 ненулевой;
  P5 отказ сохранён: файл, где proj лежит построчно (rtn), -- ValueError с «rtn» в тексте (файл собирается здесь).
Мутации (--mutate): проверка покрытия снова не пускает dense (P1); вес плотной базы транспонирован (P2);
проверка покрытия выключена (P5).
    python tests/test_rwkvq_dense_target.py [--mutate]"""
import copy, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import mlx.core as mx
import mlx.nn as nn
from rwkv_quant.formats import codec
from rwkv_metal.lora import add_rwkvq as AR, rwkvq_linear as RL
from rwkv_metal.lora.lora import LoRALinear

W = os.path.expanduser("~/Develop/WKV-kvant/")
FILES = [os.path.expanduser(p) for p in os.environ.get("FILES", W + "files_0110/0p1b_reduction.rwkvq:" + W + "files_0110/0p1b_compression.rwkvq").split(":")]
CK = os.path.expanduser(os.environ.get("CK", W + "rwkv7-g1d-0.1b.pth"))
TOK = os.path.expanduser(os.environ.get("TOK", W + "rwkv_vocab.txt"))
KEY0 = "blocks.0.att.output.weight"
TARGET_RE = ("att.receptance.weight", "att.key.weight", "att.value.weight", "att.output.weight", "ffn.key.weight", "ffn.value.weight")


def bases(model):
    out = {}
    for li, blk in enumerate(model.blocks):
        for name, okey in (("r_proj", "receptance"), ("k_proj", "key"), ("v_proj", "value"), ("o_proj", "output")):
            m = getattr(blk.tmix, name)
            out["blocks.%d.att.%s.weight" % (li, okey)] = m.linear if isinstance(m, LoRALinear) else m
        for name in ("key", "value"):
            out["blocks.%d.ffn.%s.weight" % (li, name)] = getattr(blk.cmix, name)
    out["head.weight"] = model.head
    return out


def props(d):
    R = []
    def check(n, ok, info=""): R.append((n, bool(ok), str(info)[:300]))
    ref = torch.load(CK, map_location="cpu", mmap=True, weights_only=True)[KEY0].float().numpy()
    for f in FILES:
        tag = os.path.basename(f)
        RL._SIDECAR_CACHE.clear()
        try:
            model, cfg, info = AR.load_rwkvq_model(f, verbose=False)
        except Exception as e:
            check("P1 %s загружается" % tag, False, "%s: %s" % (type(e).__name__, str(e)[:200])); continue
        x = mx.array(np.arange(1, 33, dtype=np.int32)[None])
        lg = model(x); lg = lg.astype(mx.float32); mx.eval(lg)
        check("P1 %s загружается, логиты конечны" % tag, lg.ndim == 3 and lg.shape[1] == 32 and bool(mx.all(mx.isfinite(lg))), lg.shape)
        B = bases(model)
        b0 = B[KEY0]
        w = np.array(b0._w.astype(mx.float32)) if isinstance(b0, RL.RwkvqDenseLinear) else None
        check("P2 %s: o_proj слоя 0 -- плотная база, вес побитно == .pth" % tag,
              w is not None and w.shape == ref.shape and np.array_equal(w, ref) and b0._w.dtype == mx.bfloat16,
              type(b0).__name__ if w is None else float(np.abs(w - ref).max()))
        man, _ = codec.open_rwkvq(f)
        dense_targets = sorted(k for k in B if man["tensors"][k]["kind"] == "dense")
        got_dense = sorted(k for k, m in B.items() if isinstance(m, RL.RwkvqDenseLinear))
        quant_ok = all(type(m).__name__ in ("RwkvqLinear", "RwkvqSymLinear", "RwkvqNativeLinear", "RwkvqHybridLinear")
                       for k, m in B.items() if k not in got_dense)
        check("P3 %s: плотных баз == плотных целей манифеста (%d), прочие квантованные" % (tag, len(dense_targets)),
              got_dense == dense_targets and len(dense_targets) >= 1 and quant_ok, (got_dense, dense_targets))
        lora = model.blocks[0].tmix.o_proj
        def loss(m):
            return m(x).astype(mx.float32)[0, -1].max()
        g = nn.value_and_grad(model, loss)(model)[1]
        gb = g["blocks"][0]["tmix"]["o_proj"]["lora_b"]
        check("P4 %s: градиент по lora_b над плотной базой ненулевой" % tag, float(mx.abs(gb.astype(mx.float32)).max()) > 0)
        del model; mx.clear_cache()

    # P5: файл с целью в rtn
    rtn_file = os.path.join(d, "rtn_proj.rwkvq")
    try:
        from rwkv_quant import presets
        from rwkv_quant.api import quantize
        c = copy.deepcopy(presets.REDUCTION)
        c.group_scale.pop("proj", None); c.group_scale_mode.pop("proj", None); c.bits["proj"] = 8
        quantize(CK, rtn_file, config=c, tokenizer=TOK, gptq=False, autopick=False, verbose=False)
        man, _ = codec.open_rwkvq(rtn_file)
        assert man["tensors"]["blocks.1.att.output.weight"]["kind"] == "rtn", man["tensors"]["blocks.1.att.output.weight"]["kind"]
        RL._SIDECAR_CACHE.clear()
        try:
            AR.load_rwkvq_model(rtn_file, verbose=False)
            check("P5 цель в rtn -> отказ", False, "загрузился")
        except ValueError as e:
            check("P5 цель в rtn -> отказ", "rtn" in str(e) and "dense" not in str(e), str(e)[:200])
        except Exception as e:
            check("P5 цель в rtn -> отказ", False, "%s: %s" % (type(e).__name__, str(e)[:160]))
    except Exception as e:
        check("P5 цель в rtn -> отказ", False, "сборка файла: %s: %s" % (type(e).__name__, str(e)[:200]))
    return R


def run(label):
    with tempfile.TemporaryDirectory() as d:
        R = props(d)
    failed = [n for n, ok, _ in R if not ok]
    for n, ok, info in R:
        print("  [%s] %s%s" % ("OK" if ok else "FAIL", n, "" if ok else " -- " + info))
    print("%s: %d свойств, провалов %d" % (label, len(R), len(failed)), flush=True)
    return failed


def main():
    ok = not run("КОНТРОЛЬ")
    if "--mutate" in sys.argv:
        fs = RL.RwkvqDenseLinear.from_sidecar.__func__
        def m_nodense(): AR._file_kind = lambda p, k: None
        def m_T():
            def f(cls, p, k):
                o = fs(cls, p, k); o._w = o._w.T; return o
            RL.RwkvqDenseLinear.from_sidecar = classmethod(f)
        muts = [("покрытие не пускает dense", m_nodense, "P1"), ("вес плотной базы транспонирован", m_T, "P2"),
                ("проверка покрытия выключена", lambda: setattr(AR, "_check_quantized_coverage", lambda *a, **k: None), "P5")]
        for name, apply, expect in muts:
            saved = (AR._file_kind, AR._check_quantized_coverage)
            apply()
            try:
                failed = run("МУТАЦИЯ «%s»" % name)
            finally:
                AR._file_kind, AR._check_quantized_coverage = saved
                RL.RwkvqDenseLinear.from_sidecar = classmethod(fs)
            caught = any(f.startswith(expect) for f in failed)
            print("  -> %s (ожидалось падение %s)" % ("ПОЙМАНА" if caught else "НЕ ПОЙМАНА", expect))
            ok &= caught
        ok &= not run("КОНТРОЛЬ ПОСЛЕ МУТАЦИЙ")
    print("ИТОГ: %s" % ("ЗЕЛЁНЫЙ" if ok else "КРАСНЫЙ"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
