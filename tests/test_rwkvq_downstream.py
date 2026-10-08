"""Гейт: нынешние пресеты (.rwkvq REDUCTION / COMPRESSION) во всех путях поверх квантованной базы (08.10).

Было: всё, что читает веса модели по именам, видело у квантованной базы коды вместо `weight`.
  - Reranker.init_from_base молча пропускал такие слои: голова реранкера над .rwkvq-базой (и над
    LoRA-обёрнутой плотной) стартовала со СЛУЧАЙНЫМИ r/k/v/o и cmix.
  - merge_lora складывал дельту с упакованными кодами nn.QuantizedLinear и падал на базах .rwkvq.
  - Инференс требовал rank >= 1 (лишний матмул на проекцию).
Стало: lora.dense_weight / dense_parameters (плотный вес любой базы), init_from_base через них
и с отказом при пропуске, merge_lora в плотный nn.Linear, load_rwkvq_model(rank=0).

Свойства (0.1B обоих пресетов из files_0110, COMPRESSION на трёх бэкендах; CK -- исходный .pth):
  P1 dense_weight базы == codec.dequant_key побитно (fp32) для r/k/v/o, cmix.key/value, head
     каждого бэкенда, включая плотный o_proj слоя 0;
  P2 голова реранкера над квантованной базой: каждый параметр её блоков == dense_parameters
     базы побитно, а проекции == codec.dequant_key (независимый источник);
  P3 голова реранкера над LoRA-обёрнутой ПЛОТНОЙ базой (.pth, add_lora, lora_b != 0): o_proj
     головы == W + scale * B @ A;
  P4 merge_lora на QLoRA-модели с lora_b != 0 (merge в fp32): логиты после слияния совпадают с
     логитами до (|d| < 5% эффекта адаптера, эффект > 0.3), слитые слои -- nn.Linear, среди обучаемых нет
     целочисленных массивов;
  P5 rank=0: логиты побитно равны rank=8 с нулевым адаптером, LoRA-параметров нет;
  P6 эмбеддинг над квантованной базой: EmbeddingModel.embed == голова(пул body) вручную, конечен;
  P7 load_pretrained принимает .rwkvq: логиты == load_rwkvq_model(rank=0) побитно.
Мутации (--mutate): init_from_base по parameters(), как было (P2); native-деквант с перепутанными
scale / bias (P1); merged_weight без дельты (P4); rank=0 снова оборачивает (P5); load_pretrained
снова только .pth (P7).
    python tests/test_rwkvq_downstream.py [--mutate]"""
import importlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from rwkv_quant.formats import codec
from rwkv_metal.lora import add_rwkvq as AR, rwkvq_linear as RL, lora as LO
from rwkv_metal.lora.lora import LoRALinear, dense_weight, dense_parameters, merge_lora, add_lora
from rwkv_metal.reranker import model as RM
from rwkv_metal.reranker.model import Reranker, RerankerConfig
from rwkv_metal.embedding.train import EmbeddingModel, _l2_normalize
CV = importlib.import_module("rwkv_metal.model.convert")

W = os.path.expanduser("~/Develop/WKV-kvant/")
FILES = {"reduction": W + "files_0110/0p1b_reduction.rwkvq", "compression": W + "files_0110/0p1b_compression.rwkvq"}
CK = os.path.expanduser(os.environ.get("CK", W + "rwkv7-g1d-0.1b.pth"))
X = mx.array(np.random.default_rng(0).integers(100, 60000, size=(1, 48)).astype(np.int32))
OKEY = {"r_proj": "receptance", "k_proj": "key", "v_proj": "value", "o_proj": "output"}


def bases(model, li):
    out = {}
    blk = model.blocks[li]
    for n, o in OKEY.items():
        m = getattr(blk.tmix, n)
        out["blocks.%d.att.%s.weight" % (li, o)] = (m.linear if isinstance(m, LoRALinear) else m, "tmix.%s.weight" % n)
    for n in ("key", "value"):
        out["blocks.%d.ffn.%s.weight" % (li, n)] = (getattr(blk.cmix, n), "cmix.%s.weight" % n)
    return out


def load(f, **kw):
    mx.random.seed(0)
    RL._SIDECAR_CACHE.clear()
    return AR.load_rwkvq_model(f, verbose=False, **kw)[0]


def randomize_lora(model, seed=1):
    upd, rng = {}, np.random.default_rng(seed)
    for k, v in tree_flatten(model.parameters()):          # не trainable: Reranker замораживает базу
        if k.endswith(".lora_b"):
            upd[k] = mx.array((rng.standard_normal(v.shape) * 0.002).astype(np.float32)).astype(v.dtype)
    model.update(tree_unflatten(list(upd.items())))
    mx.eval(model.parameters())


def props():
    R = []
    def check(n, ok, info=""): R.append((n, bool(ok), str(info)[:300]))
    for preset, f in FILES.items():
        man, arr = codec.open_rwkvq(f)
        ref = lambda k: np.asarray(codec.dequant_key(man, arr, k), dtype=np.float32)
        # P1
        for nat in ((True,) if preset == "reduction" else (True, False, "hybrid")):
            m = load(f, rank=0, native=nat)
            mods = dict(bases(m, 0)); mods.update(bases(m, 7)); mods["head.weight"] = (m.head, None)
            bad = [(k, type(mod).__name__) for k, (mod, _) in mods.items() if not np.array_equal(np.array(dense_weight(mod)), ref(k))]
            check("P1 %s native=%s: dense_weight == codec побитно (%d слоёв)" % (preset, nat, len(mods)), not bad, bad[:3])
            del m
        # P2
        m = load(f, rank=4)
        try:
            rr = Reranker(m, RerankerConfig())
            hp = dict(tree_flatten(rr.head.parameters()))
            bad, n = [], 0
            for i, src in enumerate(rr.head.layer_idx):
                dp = dense_parameters(m.blocks[src])
                for k, v in dp.items():
                    hk = "blocks.%d.%s" % (i, k)
                    if hk in hp:
                        n += 1
                        if not np.array_equal(np.array(hp[hk].astype(mx.float32)), np.array(v.astype(mx.float32))): bad.append(hk)
                for okey, (_, hkey) in bases(m, src).items():
                    if not np.array_equal(np.array(hp["blocks.%d.%s" % (i, hkey)].astype(mx.float32)), ref(okey)): bad.append("codec:" + okey)
            check("P2 %s: голова реранкера == плотная база (%d параметров) и == codec по проекциям" % (preset, n), not bad and n > 0, bad[:4])
        except Exception as e:
            check("P2 %s: голова реранкера над квантованной базой" % preset, False, "%s: %s" % (type(e).__name__, e))
        # P4
        zero = m(X).astype(mx.float32)
        randomize_lora(m)
        before = m(X).astype(mx.float32); mx.eval(zero, before)
        eff = float(mx.max(mx.abs(before - zero)))
        merge_lora(m, dtype=mx.float32)
        after = m(X).astype(mx.float32); mx.eval(after)
        d = float(mx.max(mx.abs(after - before)))
        lin = all(type(getattr(m.blocks[li].tmix, n)) is nn.Linear for li in range(len(m.blocks)) for n in OKEY)
        ints = [k for k, v in tree_flatten(m.trainable_parameters()) if not mx.issubdtype(v.dtype, mx.floating)]
        check("P4 %s: merge_lora сохраняет логиты, слитые -- nn.Linear, целых среди обучаемых нет" % preset,
              eff > 0.3 and d < 0.05 * eff and lin and not ints,
              "max|d|=%.3e при эффекте адаптера %.3e, nn.Linear=%s ints=%s" % (d, eff, lin, ints[:3]))
        del m
        # P5
        m0, m8 = load(f, rank=0), load(f, rank=8)
        l0, l8 = m0(X), m8(X); mx.eval(l0, l8)
        nl = [k for k, _ in tree_flatten(m0.parameters()) if k.endswith((".lora_a", ".lora_b"))]
        check("P5 %s: rank=0 == rank=8 с нулевым адаптером, адаптеров нет" % preset,
              np.array_equal(np.array(l0.astype(mx.float32)), np.array(l8.astype(mx.float32))) and not nl, "lora keys %d" % len(nl))
        # P6
        em = EmbeddingModel(m0)
        e = em(X, mx.array([47]))
        h = m0.body(X)[:, 47]
        e2 = _l2_normalize(em.head(h)); mx.eval(e, e2)
        check("P6 %s: эмбеддинг над квантованной базой == пул body вручную, конечен" % preset,
              np.array_equal(np.array(e), np.array(e2)) and bool(mx.all(mx.isfinite(e))), e.shape)
        # P7
        try:
            RL._SIDECAR_CACHE.clear()
            mp, _ = CV.load_pretrained(f, verbose=False)
            lp = mp(X); mx.eval(lp)
            check("P7 %s: load_pretrained(.rwkvq) == load_rwkvq_model(rank=0) побитно" % preset,
                  np.array_equal(np.array(lp.astype(mx.float32)), np.array(l0.astype(mx.float32))))
            del mp
        except Exception as e:
            check("P7 %s: load_pretrained(.rwkvq)" % preset, False, "%s: %s" % (type(e).__name__, e))
        del m0, m8, em
        mx.clear_cache()
    # P3
    dense, _ = CV.load_pretrained(CK, verbose=False)
    add_lora(dense, rank=4)
    randomize_lora(dense, seed=2)
    rr = Reranker(dense, RerankerConfig(), head_dtype=mx.float32)
    hp = dict(tree_flatten(rr.head.parameters()))
    src = rr.head.layer_idx[0]
    lo = dense.blocks[src].tmix.o_proj
    want = lo.linear.weight.astype(mx.float32) + lo.scale * (lo.lora_b.astype(mx.float32) @ lo.lora_a.astype(mx.float32))
    d = float(mx.max(mx.abs(hp["blocks.0.tmix.o_proj.weight"] - want)))
    check("P3 голова над LoRA-обёрнутой плотной базой: o_proj == W + scale*B@A", d == 0.0, "max|d|=%.3e" % d)
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
        init0, dw0, mw0, rep0 = RM.RerankerHead.init_from_base, LO.dense_weight, LoRALinear.merged_weight, AR._replace_targets_with_rwkvq
        def m_old_init():
            def f(self, base):
                for i, s in enumerate(self.layer_idx):
                    src = dict(tree_flatten(base.blocks[s].parameters()))
                    dst = set(k for k, _ in tree_flatten(self.blocks[i].parameters()))
                    self.blocks[i].update(tree_unflatten([(k, v) for k, v in src.items() if k in dst]))
                self.ln0.update(base.ln0.parameters()); self.ln_out.update(base.ln_out.parameters())
                return self
            RM.RerankerHead.init_from_base = f
        def m_swap():
            from rwkv_metal.lora import rwkvq_native as RN
            def f(mod, dtype=None):
                if isinstance(mod, RN.RwkvqNativeLinear):
                    return mx.dequantize(mod.wq, mod.bias, mod.scale, group_size=RN.GROUP_SIZE, bits=mod.bits).astype(dtype or mx.float32)
                return dw0(mod, dtype)
            LO.dense_weight = f
            globals()["dense_weight"] = f
        def m_nodelta():
            LoRALinear.merged_weight = lambda self, dtype=None: dense_weight(self.linear, dtype or mx.float32)
        def m_wrap0():
            def f(model, path, rank, *a, **k):
                return rep0(model, path, max(rank, 1), *a, **k)
            AR._replace_targets_with_rwkvq = f
        lp0 = CV.load_pretrained
        def m_pth_only():
            def f(path, config=None, verbose=True):
                if str(path).endswith(".rwkvq"):
                    raise ValueError("не .pth")
                return lp0(path, config, verbose)
            CV.load_pretrained = f
        muts = [("load_pretrained только .pth", m_pth_only, "P7"), ("init_from_base по parameters()", m_old_init, "P2"), ("native: scale и bias перепутаны", m_swap, "P1"),
                ("merged_weight без дельты", m_nodelta, "P4"), ("rank=0 оборачивает", m_wrap0, "P5")]
        for name, apply, expect in muts:
            apply()
            try:
                failed = run("МУТАЦИЯ «%s»" % name)
            finally:
                RM.RerankerHead.init_from_base, LO.dense_weight, LoRALinear.merged_weight, AR._replace_targets_with_rwkvq = init0, dw0, mw0, rep0
                globals()["dense_weight"] = dw0
                CV.load_pretrained = lp0
            caught = any(f.startswith(expect) for f in failed)
            print("  -> %s (ожидалось падение %s)" % ("ПОЙМАНА" if caught else "НЕ ПОЙМАНА", expect))
            ok &= caught
        ok &= not run("КОНТРОЛЬ ПОСЛЕ МУТАЦИЙ")
    print("ИТОГ: %s" % ("ЗЕЛЁНЫЙ" if ok else "КРАСНЫЙ"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
