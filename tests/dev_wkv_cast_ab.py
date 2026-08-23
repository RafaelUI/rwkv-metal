"""A/B флага CAST_WKV_OUTPUT: скорость декода и перплексия, в одном процессе.

Проверяет на rwkv-metal то, что уже замерено в порту SwiftRWKV на 0.1B
(декод ×2.82, ppl +0.090%), но на модели ПОГЛУБЖЕ. Глубина здесь и есть
предмет: округлений bf16 в остаточном потоке накапливается по одному на
слой, и на 24 слоях их вдвое больше, чем на 12.

Дисциплина:
  * обе ветки в ОДНОМ процессе на ОДНОЙ модели — флаг читается на каждом
    проходе; разными прогонами такое сравнивать нельзя;
  * скорость — A/B ЧЕРЕДОВАНИЕМ, медиана. Машина безвентиляторная, и
    «сначала все off, потом все on» меряет тепловой дрейф, а не ветки;
  * перплексия — независимые чанки со своего нулевого состояния, лосс в
    fp32 при обеих ветках, порядок off → on → off (второй off — контроль
    детерминированности);
  * своп считается до и после: замер при свопе недействителен, а не
    «чуть хуже».

    python dev_wkv_cast_ab.py <путь к .pth> [n_chunks] [speed|ppl|both]

РЕЖИМЫ РАЗДЕЛЕНЫ НЕ ДЛЯ УДОБСТВА. ppl-фаза держит логиты [T, 65536] в fp32
— на T=512 это 134 МБ на чанк, — и на 1.5B она уводит машину в своп. Своп
не портит саму ppl (число детерминированное, своп лишь замедляет счёт), но
портит ЛЮБОЙ замер времени в том же процессе. Поэтому скорость меряется в
своём процессе, где ничего крупного не аллоцируется.
"""
import sys, os, time, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.expanduser("~/Develop/WKV-kvant"))
import numpy as np
import mlx.core as mx
import rwkv_metal.model.rwkv7_x070 as m070
from rwkv_metal.model.convert import load_pretrained
from rwkv_metal.model.state import RWKVState

PTH = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
NCH = int(sys.argv[2]) if len(sys.argv) > 2 else 16
MODE = sys.argv[3] if len(sys.argv) > 3 else "both"
VOCAB = os.path.expanduser(
    "~/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt")
TXT = os.path.expanduser("~/Develop/test.txt")
T = 512
N_DECODE = 32


def swap_mb():
    out = subprocess.check_output(["sysctl", "-n", "vm.swapusage"]).decode()
    return float(out.split("used =")[1].split("M")[0])


def perplexity(model, chunks):
    total, count = 0.0, 0
    for ids in chunks:
        idx = mx.array(np.array(ids, dtype=np.int64)[None, :])
        logits = model(idx)[0].astype(mx.float32)
        pred = logits[: len(ids) - 1]
        tgt = mx.array(np.array(ids[1:], dtype=np.int64))
        lse = mx.logsumexp(pred, axis=-1)
        picked = mx.take_along_axis(pred, tgt[:, None], axis=1).reshape(-1)
        nll = (lse - picked).sum()
        mx.eval(nll)
        total += float(nll)
        count += len(ids) - 1
    return float(np.exp(total / count)), count


def decode_ms(model, cfg, prompt_ids, n=N_DECODE):
    """Пошаговый декод через body(..., return_state=True): промпт свёрнут
    один раз, дальше по одному токену. Это и есть боевой путь генерации."""
    idx = mx.array(np.array(prompt_ids, dtype=np.int64)[None, :])
    h, st = model.body(idx, return_state=True)
    logits = model.head(h[:, -1:])
    mx.eval(logits)
    t0 = time.perf_counter()
    for _ in range(n):
        tok = int(mx.argmax(logits.reshape(-1)).item())
        h, st = model.body(mx.array(np.array([[tok]], dtype=np.int64)),
                           state=st, return_state=True)
        logits = model.head(h)
        mx.eval(logits)
    return (time.perf_counter() - t0) * 1000 / n


def main():
    from world_tokenizer import RWKV_WORLD_TOKENIZER
    tok = RWKV_WORLD_TOKENIZER(VOCAB)
    ids = tok.encode(open(TXT, encoding="utf-8").read())
    chunks = [ids[s:s + T] for s in range(0, len(ids) - T + 1, T)][:NCH]
    print(f"корпус: {len(ids)} токенов, {len(chunks)} чанков по {T}")

    sw0 = swap_mb()
    if MODE != "ppl_q":
        # ppl_q строит СВОЮ модель из `.rwkvq` ниже; .pth (3 ГБ) ей не нужен.
        model, cfg = load_pretrained(PTH, verbose=False)
        assert model is not None
        print(f"модель: L={cfg.n_layer} D={cfg.n_embd} vocab={cfg.vocab_size}")

        prompt = chunks[0][:64]
        print(f"── CAST_WKV_OUTPUT на {os.path.basename(PTH)} (L={cfg.n_layer}) ──")

    if MODE in ("speed", "both"):
        def timed(flag, n=8):
            m070.CAST_WKV_OUTPUT = flag
            return decode_ms(model, cfg, prompt, n)

        timed(False); timed(True)                  # прогрев обеих
        sw_a = swap_mb()
        offs, ons = [], []
        for _ in range(5):
            offs.append(timed(False)); ons.append(timed(True))
        sw_b = swap_mb()
        m_off, m_on = sorted(offs)[2], sorted(ons)[2]
        ok = abs(sw_b - sw_a) < 1
        print(f"""декод:  выкл {m_off:6.2f} мс/ток, вкл {m_on:6.2f} мс/ток  → x{m_off/m_on:.2f}
        разбросы выкл {'/'.join(f'{v:.1f}' for v in offs)}
                 вкл  {'/'.join(f'{v:.1f}' for v in ons)}
        своп вокруг ЗАМЕРА: {sw_a:.0f} → {sw_b:.0f} МБ """
              + ("(не двигался)" if ok else "— ЗАМЕР НЕДЕЙСТВИТЕЛЕН"))

    if MODE == "ppl_q":
        # PPL-ГЕЙТ ДВУХ РЫЧАГЕЙ fp32-ХВОСТА НА КВАНТОВАННОЙ БАЗЕ (22.08).
        # `bench_wkv_fp32_tail_ab` дал скорость (+12.1% / +26.7% шага) и
        # relmax логитов, но это индикатор, а не гейт (закон 9). Здесь те же
        # два рычага меряются ppl на мультиязычном корпусе, по
        # последовательностям -- per-seq NLL пишется в JSON, интервалы
        # считаются бутстрэпом без повторного прогона (как в
        # eval_qlora_arms). Порядок: база -> каст в слое -> CAST_WKV_OUTPUT
        # -> база (второй -- контроль детерминированности). Адаптеров НЕТ:
        # меряется база, какая есть в файле.
        import torch
        from rwkv_metal.lora import load_rwkvq_model
        from rwkv_metal.lora import rwkvq_linear as rl

        cd = torch.load(os.path.expanduser(
            "~/Develop/WKV-kvant/eval_corpus_multiling.pt"), weights_only=False)
        toks = cd["tokens"]
        RWKVQ = os.environ.get("RWKVQ_BASE", "/tmp/reduction_new.rwkvq")
        model, cfg, info = load_rwkvq_model(RWKVQ, verbose=False)
        model._grad_ckpt = True

        CLASSES = [c for c in (getattr(rl, "RwkvqLinear", None),
                               getattr(rl, "RwkvqSymLinear", None))
                   if c is not None]
        # ЗАМОРОЖЕННАЯ «база» (23.08): после внедрения каста 22.08 захват
        # «что стоит в модуле» молча совпал бы с кастом, и «база» vs
        # «каст в слое» вырождались в сравнение самих с собой (закон 27 --
        # та же правка, что в bench_wkv_fp32_tail_ab).
        def nocast_call(self, x):
            return x @ self._dequant_w().T

        def cast_call(self, x):
            w = self._dequant_w()
            return (x.astype(w.dtype) @ w.T).astype(x.dtype)

        # МУТАЦИОННЫЕ КРЫШКИ: включаются средой, НЕ для обычных прогонов.
        # RWKVQ_MUTATION_NOCAST -- каст в слое тихо заменён на прежний
        # вызов; RWKVQ_MUTATION_NOWKV -- CAST_WKV_OUTPUT тихо не
        # включается. Обе обязаны ронять контроль включения ниже; если не
        # роняют, контроль холостый (ловушка «рабочий код как заглушка»).
        MUT_NOCAST = bool(os.environ.get("RWKVQ_MUTATION_NOCAST"))
        MUT_NOWKV = bool(os.environ.get("RWKVQ_MUTATION_NOWKV"))

        if MUT_NOCAST:
            def cast_call(self, x):            # noqa: F811 (мутация)
                return nocast_call(self, x)

        def set_lincast(on):
            for c in CLASSES:
                c.__call__ = cast_call if on else nocast_call

        def set_wkvcast(on):
            m070.CAST_WKV_OUTPUT = on and not MUT_NOWKV

        # КОНТРОЛЬ ВКЛЮЧЕНИЯ: если правка тихо не применяется, ppl
        # варианта совпадает с базой и «Δ = 0» читается как «чисто», а не
        # как «не сработало». Логиты варианта обязаны ОТЛИЧАТЬСЯ от базы.
        x0 = mx.array(toks[0][:-1].numpy().astype(np.int32)[None, :])
        set_lincast(False)
        set_wkvcast(False)
        lg_base = model(x0)
        mx.eval(lg_base)
        for name, lc, wc in (("каст_в_слое", True, False),
                             ("CAST_WKV_OUTPUT", False, True)):
            set_lincast(lc)
            set_wkvcast(wc)
            lg = model(x0)
            rel = float(mx.max(mx.abs(lg - lg_base))
                       / (mx.max(mx.abs(lg_base)) + 1e-30))
            mx.eval(lg)
            print("включение %-15s relmax логитов %.3e" % (name, rel),
                  flush=True)
            assert rel > 0, ("вариант %s НЕ ВКЛЮЧИЛСЯ (логиты как у базы)"
                             % name)
            del lg
        set_lincast(False)
        set_wkvcast(False)
        del lg_base
        mx.clear_cache()

        def run_seqs():
            out = []
            for row in toks:
                x = mx.array(row[:-1].numpy().astype(np.int32)[None, :])
                y = mx.array(row[1:].numpy().astype(np.int32)[None, :])
                l = model.loss(x, y).astype(mx.float32)
                mx.eval(l)
                out.append(float(l))
            return out

        res = {}
        swa = swap_mb()
        for name, lc, wc in (("база1", False, False),
                             ("каст_в_слое", True, False),
                             ("CAST_WKV_OUTPUT", False, True),
                             ("база2", False, False)):
            set_lincast(lc)
            set_wkvcast(wc)
            res[name] = run_seqs()
            ppl = float(np.exp(np.mean(res[name])))
            print("ppl %-15s %8.4f" % (name, ppl), flush=True)
        set_lincast(False)
        set_wkvcast(False)
        b1, b2 = np.mean(res["база1"]), np.mean(res["база2"])
        if abs(b1 - b2) > 1e-6:
            print("ВНИМАНИЕ: контроль разошёлся (%.6f vs %.6f) -- недетерминизм"
                  % (b1, b2))
        for name in ("каст_в_слое", "CAST_WKV_OUTPUT"):
            m = np.mean(res[name])
            print("Δ %-15s %+.4f%% (%d позиций, %d последовательностей)"
                  % (name, 100 * (m - b1) / b1,
                     (len(toks[0]) - 1) * len(toks), len(toks)))
        out_json = os.path.expanduser(
            "~/Develop/WKV-kvant/wkv_cast_ppl_rwkvq%s.json"
            % os.environ.get("WKV_CAST_TAG", ""))
        with open(out_json, "w") as f:
            import json
            json.dump({k: v for k, v in res.items()},
                      f, ensure_ascii=False, indent=1)
        print("per-seq NLL: %s\nсвоп за ppl: %.0f -> %.0f МБ (не влияет)"
              % (out_json, swa, swap_mb()))

    if MODE in ("ppl", "both"):
        m070.CAST_WKV_OUTPUT = False
        ppl_off1, npos = perplexity(model, chunks)
        m070.CAST_WKV_OUTPUT = True
        ppl_on, _ = perplexity(model, chunks)
        m070.CAST_WKV_OUTPUT = False
        ppl_off2, _ = perplexity(model, chunks)
        print(f"""ppl:    выкл {ppl_off1:.4f}  [контроль {ppl_off2:.4f}]
        вкл  {ppl_on:.4f}   → {100*(ppl_on-ppl_off1)/ppl_off1:+.3f}%  ({npos} позиций)""")
        if abs(ppl_off1 - ppl_off2) > 1e-6:
            print("ВНИМАНИЕ: контроль разошёлся — замер недетерминирован")
        # Своп ppl не портит: число детерминированное, своп лишь замедляет.
        print(f"        (своп за весь прогон {sw0:.0f} → {swap_mb():.0f} МБ; "
              "на ppl это не влияет, на время — влияет)")


if __name__ == "__main__":
    main()
