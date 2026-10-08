"""Гейт правок памяти загрузки QLoRA-базы (07.10).

Было (1.5B COMPRESSION, native=True -- умолчание): после загрузки footprint 3.7 ГБ при файле 0.95 ГБ, пик загрузки 5.6 ГБ:
1.4 ГБ «Malloc Large (empty)» (освобождённые крупные транзиенты хоста -- int32-коды родной упаковки и K3-интерлив целыми
тензорами; macOS их системе не отдаёт) и ~0.9 ГБ K3-буферов в кеше загрузчика, которые родному бэкенду после перекладки
не нужны. Стало: перекладка полосами строк (rl.HOST_BAND_MB), кеш загрузчика и кеш MLX отпускаются в конце сборки:
footprint 1.7 ГБ, пик 3.3 ГБ, активная память MLX 2308 -> 1397 МиБ. Выход модели побитно прежний.

Свойства (SB6 -- файл пресета compression, SYM -- reduction; по умолчанию 0.1B, files_0110):
  L1 K3-буферы всех sb6-тензоров файла, собранные полосами (полоса сужена до 1/8 МиБ), ПОБИТНО равны перекладке целого
     тензора нормативной codec.sb6_to_k3; полос больше одной не менее чем у половины тензоров (вовлечение);
  L2 родная упаковка (wq, scale, bias) полосами побитно равна упаковке целого тензора -- по представителю каждой формы,
     включая голову; полос больше одной;
  L3 после load_rwkvq_model файла нет в кеше загрузчика; модель считает (логиты конечны);
  L4 отпущенный кеш -- это память: активная память MLX после загрузки native=True меньше, чем с удержанным кешем, не
     менее чем на 90% байт K3-буферов, которые кеш держал;
  L5 отпечаток модели (логиты T=64, градиенты адаптеров шага T=64) один и тот же при полосе 1/8 МиБ, 16 МиБ и «целиком»
     -- compression native=True и native=False, reduction.
Мутации (--mutate): полосы склеены в обратном порядке (L1); полоса сдвинута на строку (L2); кеш загрузчика не
отпускается, только кеш MLX (L3 и L4).
    python tests/test_load_memory.py [--mutate]"""
import gc, hashlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from rwkv_quant.formats import codec
from rwkv_metal.lora import add_rwkvq as AR, rwkvq_linear as rl, rwkvq_native as RN
from rwkv_metal.lora import load_rwkvq_model

W = os.path.expanduser("~/Develop/WKV-kvant/")
SB6 = os.path.expanduser(os.environ.get("SB6", W + "files_0110/0p1b_compression.rwkvq"))
SYM = os.path.expanduser(os.environ.get("SYM", W + "files_0110/0p1b_reduction.rwkvq"))
BAND0 = rl.HOST_BAND_MB
def eq(a, b):
    a, b = np.array(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
TINY = 0.125                                  # МиБ: полоса, при которой и матрицы 768x768 режутся


def fingerprint(path, native):
    rl.drop_sidecar_cache()
    model, _, _ = load_rwkvq_model(path, rank=8, verbose=False, native=native)
    rs = np.random.RandomState(11)
    x = mx.array(rs.randint(1, 60000, size=(1, 64)).astype(np.int32)); y = mx.array(rs.randint(1, 60000, size=(1, 64)).astype(np.int32))
    lg = model(x); mx.eval(lg)
    upd = [(k, mx.array(rs.normal(0, 0.02, v.shape).astype(np.float32)).astype(v.dtype)) for k, v in tree_flatten(model.trainable_parameters())]
    model.update(tree_unflatten(upd)); mx.eval(model.parameters())
    loss, g = nn.value_and_grad(model, lambda m, a, b: m.loss(a, b).astype(mx.float32))(model, x, y)
    flat = dict(tree_flatten(g)); mx.eval(loss, flat)
    h = hashlib.sha256(np.array(lg.astype(mx.float32)).tobytes())
    for k in sorted(flat): h.update(np.array(flat[k].astype(mx.float32)).tobytes())
    ok = bool(mx.all(mx.isfinite(lg.astype(mx.float32))))
    del model; mx.clear_cache()
    return h.hexdigest()[:16], ok


def props():
    R = []
    def check(n, ok, info=""): R.append((n, bool(ok), str(info)[:300]))
    man, buf = codec.open_rwkvq(SB6)
    sb6 = sorted(k for k, m in man["tensors"].items() if m["kind"] == "sb6")

    # L1
    rl.HOST_BAND_MB = TINY; rl.drop_sidecar_cache()
    try:
        arrays, _ = rl._load_rwkvq_direct(SB6)
    finally:
        rl.HOST_BAND_MB = BAND0
    bad, multi = [], 0
    for k in sb6:
        m = man["tensors"][k]; b = lambda f: buf.get(f"{k}::{f}")
        src = [b(f) for f in ("codes_packed", "gw_qsqm", "gw_d", "gw_dm", "gw_qh", "gw_qh2")]
        multi += m["shape"][0] > max(1, int(TINY * 2 ** 20) // (2 * sum(v[:1].nbytes for v in src if v is not None)))
        qblk, qsqm, ddm, _ = codec.sb6_to_k3(src[0], src[1], src[2], src[3], shape=tuple(m["shape"]), gs=m["gw_gs"], sb=m["gw_sb"],
                                             nb=m.get("n_blocks"), qh=src[4], qh2=src[5])
        for name, ref in (("qblk", qblk), ("qsqm", qsqm), ("ddm", ddm)):
            if not eq(arrays[f"{k}::{name}"], ref): bad.append((k, name))
    check("L1 K3 полосами == целиком, все %d sb6-тензоров" % len(sb6), not bad and len(sb6) > 10, bad[:3])
    check("L1 полос больше одной у %d из %d тензоров" % (multi, len(sb6)), multi >= len(sb6) // 2)

    # L2
    reps = {}
    for k in sb6:
        if k != "emb.weight": reps.setdefault(tuple(man["tensors"][k]["shape"]), k)
    bad, multi = [], 0
    for sh, k in reps.items():
        lin = rl.RwkvqLinear(arrays[f"{k}::qblk"], arrays[f"{k}::qsqm"], arrays[f"{k}::ddm"], sh, man["tensors"][k]["gw_gs"], man["tensors"][k]["gw_sb"],
                             (0 if buf.get(f"{k}::gw_qh") is None else 1) + (0 if buf.get(f"{k}::gw_qh2") is None else 1))
        codes, scale, bias = RN._codes_scale_bias(lin)                       # целый тензор
        wq_ref = RN._pack_codes_mlx(codes, 4 + lin.xbits).reshape(sh[0], -1)
        rl.HOST_BAND_MB = TINY
        try:
            nat = RN.RwkvqNativeLinear(lin)
        finally:
            rl.HOST_BAND_MB = BAND0
        multi += sh[0] > int(TINY * 2 ** 20) // (6 * sh[1])
        # scale / bias с 08.10 хранятся в fp16, где это точно (test_native_scale_fp16) -- сверяются значения
        if not (eq(nat.wq, wq_ref) and eq(nat.scale.astype(mx.float32), scale) and eq(nat.bias.astype(mx.float32), bias)): bad.append(k)
    check("L2 родная упаковка полосами == целиком (%d форм, включая голову)" % len(reps), not bad and "head.weight" in reps.values(), bad)
    check("L2 полос больше одной у %d из %d форм" % (multi, len(reps)), multi == len(reps))
    k3_bytes = sum(arrays[f"{k}::{n}"].nbytes for k in sb6 for n in ("qblk", "qsqm", "ddm"))
    del arrays; gc.collect(); mx.clear_cache()

    # L3 + L4
    def active_after_load():
        gc.collect(); mx.clear_cache(); a0 = mx.get_active_memory()
        model, _, _ = load_rwkvq_model(SB6, rank=8, verbose=False, native=True)
        gc.collect(); mx.clear_cache()
        x = mx.array(np.arange(1, 33, dtype=np.int32)[None]); lg = model(x).astype(mx.float32); mx.eval(lg)
        fin = bool(mx.all(mx.isfinite(lg))); del lg; mx.clear_cache()
        return model, mx.get_active_memory() - a0, fin
    rl.drop_sidecar_cache()
    model, a_drop, fin = active_after_load()
    check("L3 после загрузки файла нет в кеше загрузчика, логиты конечны", SB6 not in rl._SIDECAR_CACHE and len(rl._SIDECAR_CACHE) == 0 and fin, list(rl._SIDECAR_CACHE))
    del model; gc.collect(); mx.clear_cache()
    fin_load = AR._finish_load; AR._finish_load = lambda p: None
    try:
        model, a_keep, _ = active_after_load()
    finally:
        AR._finish_load = fin_load
    del model; rl.drop_sidecar_cache(); gc.collect(); mx.clear_cache()
    check("L4 отпущенный кеш: активная память MLX меньше не менее чем на 90% K3-буферов",
          a_keep - a_drop >= 0.9 * k3_bytes, "с кешем %.0f, без %.0f, K3 %.0f МиБ" % (a_keep / 2**20, a_drop / 2**20, k3_bytes / 2**20))

    # L5
    for tag, path, native in (("compression native", SB6, True), ("compression свой кернель", SB6, False), ("reduction", SYM, True)):
        fps = []
        for band in (TINY, 16, 1 << 20):
            rl.HOST_BAND_MB = band
            try:
                fps.append(fingerprint(path, native))
            finally:
                rl.HOST_BAND_MB = BAND0
        check("L5 %s: отпечаток не зависит от полосы (1/8 МиБ / 16 МиБ / целиком)" % tag, len({f[0] for f in fps}) == 1 and all(f[1] for f in fps), fps)
    rl.drop_sidecar_cache()
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
        bands = rl._mx_bands; csb = RN._codes_scale_bias; fin = AR._finish_load
        def m_rev():
            def f(make, OUT, rows):
                parts = [tuple(mx.array(np.ascontiguousarray(z)) for z in make(a, min(a + rows, OUT))) for a in range(0, OUT, rows)][::-1]
                return tuple(mx.concatenate([p[i] for p in parts], axis=0) for i in range(len(parts[0])))
            rl._mx_bands = f
        def m_shift():
            def f(lin, a=0, e=None):
                if e is None or a == 0: return csb(lin, a, e)
                return csb(lin, a - 1, e - 1)
            RN._codes_scale_bias = f
        muts = [("полосы в обратном порядке", m_rev, "L1"), ("полоса сдвинута на строку", m_shift, "L2"),   # имя, правка, ожидаемые падения
                ("кеш загрузчика не отпускается (только кеш MLX)", lambda: setattr(AR, "_finish_load", lambda p: mx.clear_cache()), "L3", "L4")]
        for name, apply, *expect in muts:
            apply()
            try:
                failed = run("МУТАЦИЯ «%s»" % name)
            finally:
                rl._mx_bands, RN._codes_scale_bias, AR._finish_load = bands, csb, fin
                rl.HOST_BAND_MB = BAND0; rl.drop_sidecar_cache()
            caught = all(any(f.startswith(e) for f in failed) for e in expect)
            print("  -> %s (ожидалось падение %s)" % ("ПОЙМАНА" if caught else "НЕ ПОЙМАНА", " и ".join(expect)))
            ok &= caught
        ok &= not run("КОНТРОЛЬ ПОСЛЕ МУТАЦИЙ")
    print("ИТОГ: %s" % ("ЗЕЛЁНЫЙ" if ok else "КРАСНЫЙ"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
