#!/usr/bin/env python3
"""
gen_rwkvq_fixtures.py -- эталоны для порта SwiftRWKV на НЫНЕШНИЕ пресеты rwkv-quant (08.10).

Прежние фикстуры .testdata/ (июль) сняты с сайдкара старого пресета: REDUCTION тогда был sb6, деквант bf16, плотные
параметры bf16. Нынешние файлы: REDUCTION -- sym (Q6_K, 6/8 бит), COMPRESSION -- sb6 (4/5/6), o_proj слоя 0 плотный,
файл читается НАПРЯМУЮ (без сайдкара), норма декванта -- fp16, плотные параметры модели -- fp16.

Пишет в ~/Develop/SwiftRWKV/.testdata/ (каталог в .gitignore порта; репозиторий SwiftRWKV не трогается):

  rwkvq2_<preset>_dequant.safetensors   деквант codec (fp32, точные значения формата) -- по 2 тензора на каждое
                                        сочетание (kind, bits, форма); emb и head -- строки [0:1024] под ключом
                                        "<ключ>[0:1024]". Ключи -- имена .rwkvq (world). Порт обязан совпасть ПОБИТНО
                                        (fp32) либо, если деквантует в fp16, -- с fp16(эталона).
  rwkvq2_<preset>_model.safetensors     input_ids [1,64] i32; блоки: blk_out [L,64,D] f32 (выход каждого блока, путь
                                        rwkv-metal по умолчанию); logits_metal [64,V] f32 (rwkv-metal: параметры fp16,
                                        база fp16, обратный bf16 -- умолчание); logits_ref [64,V] f32 (fp32-параметры,
                                        база без каста входа -- точный счёт тех же весов); state_wkv [L,H,S,S],
                                        state_tmix [L,D], state_cmix [L,D] после префилла; decode_ids [1,8] и
                                        decode_logits [8,V] -- потоковый декод по одному токену от этого состояния.
  rwkvq2_<preset>.json                  источник (.rwkvq, md5), коммиты rwkv-metal / rwkv-quant, режимы, KL(ref||metal)
                                        и KL(ref||metal-decode) -- ориентир допуска для порта.

    cd ~/Develop/rwkv-metal && .venv/bin/python tools/gen_rwkvq_fixtures.py [0p1b]
"""
import hashlib, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
from rwkv_quant.formats import codec
from rwkv_metal.lora import add_rwkvq as AR, rwkvq_linear as RL
from rwkv_metal.model import convert as _cv  # noqa: F401  (модуль, не функция)

SCALE = sys.argv[1] if len(sys.argv) > 1 else "0p1b"
SRC = os.path.expanduser("~/Develop/WKV-kvant/files_0110/%s_{}.rwkvq" % SCALE)
OUT = os.path.expanduser("~/Develop/SwiftRWKV/.testdata/")
T, ND, ROWS = 64, 8, 1024
IDS = mx.array((np.arange(1, T + 1, dtype=np.int64) * 7919 % 60000 + 100).astype(np.int32)[None])
DEC = mx.array((np.arange(1, ND + 1, dtype=np.int64) * 104729 % 60000 + 100).astype(np.int32)[None])


def git(path):
    return subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def kl(p_logits, q_logits):
    lp = p_logits - mx.logsumexp(p_logits, -1, keepdims=True)
    lq = q_logits - mx.logsumexp(q_logits, -1, keepdims=True)
    return float(mx.mean(mx.sum(mx.exp(lp) * (lp - lq), -1)))


def run_model(path, ref):
    saved = (RL.BASE_DTYPE, RL.NOCAST, RL.BWD_DTYPE)
    if ref:
        RL.BASE_DTYPE, RL.NOCAST = mx.float16, True
    try:
        RL._SIDECAR_CACHE.clear()
        m, _, _ = AR.load_rwkvq_model(path, rank=0, verbose=False, param_dtype="fp32" if ref else None)
        x = m.ln0(m.emb(IDS)); v_first = None; blk = []
        for b in m.blocks:
            x, v_first = b(x, v_first); blk.append(x[0].astype(mx.float32))
        logits = m.head(m.ln_out(x))[0].astype(mx.float32)
        h, st = m.body(IDS, return_state=True)
        logits2 = m.head(h)[0].astype(mx.float32)
        assert bool(mx.array_equal(logits, logits2)), "путь с состоянием расходится с путём без него"
        dec = []
        for i in range(ND):
            h, st2 = m.body(DEC[:, i:i + 1], state=st if i == 0 else st2, return_state=True)
            dec.append(m.head(h)[0, -1].astype(mx.float32))
        out = dict(blk_out=mx.stack(blk), logits=logits, decode_logits=mx.stack(dec),
                   state_wkv=st.wkv[:, 0].astype(mx.float32), state_tmix=st.tmix_shift[:, 0, 0].astype(mx.float32),
                   state_cmix=st.cmix_shift[:, 0, 0].astype(mx.float32))
        mx.eval(out)
        return out
    finally:
        RL.BASE_DTYPE, RL.NOCAST, RL.BWD_DTYPE = saved


def main():
    os.makedirs(OUT, exist_ok=True)
    for preset in ("reduction", "compression"):
        path = SRC.format(preset)
        man, arr = codec.open_rwkvq(path)
        # ---- деквант ----
        combos, deq = {}, {}
        for k, meta in sorted(man["tensors"].items()):
            key = (meta.get("kind"), meta.get("bits"), tuple(meta["shape"]))
            if len(meta["shape"]) != 2 or len(combos.setdefault(key, [])) >= 2:
                continue        # одномерные (нормы, миксы) порт проверяет сквозным эталоном модели
            combos[key].append(k)
            w = np.asarray(codec.dequant_key(man, arr, k), dtype=np.float32)
            if k in ("emb.weight", "head.weight"):
                deq["%s[0:%d]" % (k, ROWS)] = mx.array(np.ascontiguousarray(w[:ROWS]))
            else:
                deq[k] = mx.array(w)
        mx.save_safetensors(OUT + "rwkvq2_%s_dequant.safetensors" % preset, deq)
        # ---- модель ----
        a, b = run_model(path, ref=False), run_model(path, ref=True)
        a2 = run_model(path, ref=False)
        assert bool(mx.array_equal(a["logits"], a2["logits"])), "путь rwkv-metal недетерминирован"
        model = dict(input_ids=IDS, decode_ids=DEC, blk_out=a["blk_out"], logits_metal=a["logits"], logits_ref=b["logits"],
                     decode_logits=a["decode_logits"], decode_logits_ref=b["decode_logits"],
                     state_wkv=a["state_wkv"], state_tmix=a["state_tmix"], state_cmix=a["state_cmix"])
        mx.save_safetensors(OUT + "rwkvq2_%s_model.safetensors" % preset, model)
        info = dict(source=path, md5=md5(path), preset=preset, scale=SCALE,
                    rwkv_metal=git(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    rwkv_quant=git(os.path.expanduser("~/Develop/rwkv-quant")),
                    metal_mode=dict(param_dtype="fp16", base_dtype=str(RL.BASE_DTYPE), nocast=bool(RL.NOCAST)),
                    ref_mode=dict(param_dtype="fp32", base_dtype="float16 (точные веса)", nocast=True),
                    kl_ref_metal=kl(b["logits"], a["logits"]), kl_ref_metal_decode=kl(b["decode_logits"], a["decode_logits"]),
                    dequant=dict(n=len(deq), combos=["%s/%s/%s: %s" % (k[0], k[1], "x".join(map(str, k[2])), ", ".join(v)) for k, v in sorted(combos.items(), key=str)]),
                    note="ключи декванта -- имена .rwkvq; транспонирование LoRA-веток -- codec.is_transposed; "
                         "emb/head -- первые %d строк" % ROWS)
        with open(OUT + "rwkvq2_%s.json" % preset, "w") as f:
            json.dump(info, f, ensure_ascii=False, indent=1)
        print(preset, "деквант %d тензоров, KL(ref||metal) %.3e, декод %.3e" % (len(deq), info["kl_ref_metal"], info["kl_ref_metal_decode"]))


if __name__ == "__main__":
    main()
