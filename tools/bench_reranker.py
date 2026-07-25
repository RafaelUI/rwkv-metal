"""
Замеры реранкера: скорость шага обучения, скоринга и потокового декода,
с mx.compile и без, плюс память ПО ДАННЫМ СИСТЕМЫ.

Почему память меряется не через MLX: `mx.get_peak_memory()` показывает пик
пула MLX и ничего не знает ни про numpy-буферы, ни про веса модели, ни про
то, ушла ли машина в своп. А если процесс свопится, врут и замеры времени —
поэтому здесь снимается RSS процесса и дельта swapins/swapouts за прогон.

Запуск: .venv/bin/python tools/bench_reranker.py
"""
import argparse
import os
import resource
import subprocess
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwkv_metal.model import load_pretrained
from rwkv_metal.reranker import Reranker, RerankerConfig, RerankTrainConfig, train_reranker
from rwkv_metal.reranker.encode import StateCache


def rss_gb() -> float:
    """Текущий RSS процесса, ГБ (ps — то же, что видит система)."""
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                         capture_output=True, text=True).stdout.strip()
    return int(out) * 1024 / 1e9 if out else float("nan")


def peak_rss_gb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (raw if sys.platform == "darwin" else raw * 1024) / 1e9


def swap_counters() -> tuple:
    try:
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        ins = outs = 0
        for line in vm.splitlines():
            if "Swapins" in line:
                ins = int(line.split(":")[1].strip().rstrip("."))
            elif "Swapouts" in line:
                outs = int(line.split(":")[1].strip().rstrip("."))
        return ins, outs
    except Exception:
        return 0, 0


def timeit(fn, n=20, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000


def synth_cache(n_samples=512, n_cand=8, n_src=1, H=12, S=64, seed=0):
    rng = np.random.default_rng(seed)
    n_pairs = n_samples * n_cand
    states = (rng.standard_normal((n_pairs, n_src, H, S, S)) * 0.5).astype(np.float16)
    labels = rng.integers(0, n_cand, size=n_samples).astype(np.int32)
    pair_index = np.arange(n_pairs, dtype=np.int32).reshape(n_samples, n_cand)
    return StateCache(states=states, pair_index=pair_index, labels=labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth")
    args = ap.parse_args()

    sw0 = swap_counters()
    print(f"RSS на старте: {rss_gb():.2f} ГБ")

    base, cfg = load_pretrained(os.path.expanduser(args.model), verbose=False)
    print(f"RSS после загрузки базы: {rss_gb():.2f} ГБ "
          f"(0.1B в bf16 ≈ {sum(v.size for _, v in __import__('mlx.utils', fromlist=['x']).tree_flatten(base.parameters()))*2/1e9:.2f} ГБ весов)")

    cache = synth_cache(n_samples=512, n_cand=8, H=cfg.n_head, S=cfg.head_size)
    print(f"RSS с кэшем ({cache.nbytes()/1e9:.2f} ГБ): {rss_gb():.2f} ГБ")

    # ── 1. шаг обучения головы ──────────────────────────────────────────
    print("\n── шаг обучения головы (batch 32 запросов × 8 кандидатов) ──")
    for compile_on in (False, True):
        mx.random.seed(0)
        model = Reranker(base, RerankerConfig(layer_idx=(5,)))
        t0 = time.time()
        res = train_reranker(model, cache, None, RerankTrainConfig(
            lr=2e-4, batch_size=32, epochs=2, log_every=0, compile=compile_on,
            checkpoint_path="/tmp/_bench_head.safetensors"))
        dt = time.time() - t0
        per_step = dt / max(1, res["steps"]) * 1000
        print(f"  compile={str(compile_on):5s}: {per_step:6.1f} мс/шаг "
              f"({res['steps']} шагов за {dt:.1f}s)")

    # ── 2. форвард головы (скоринг) ─────────────────────────────────────
    print("\n── форвард головы, 256 пар ──")
    mx.random.seed(0)
    model = Reranker(base, RerankerConfig(layer_idx=(5,)))
    states = cache.gather(np.arange(256))
    mx.eval(states)

    def eager_fwd():
        mx.eval(model.head(states))

    compiled_head = mx.compile(lambda s: model.head(s))

    def comp_fwd():
        mx.eval(compiled_head(states))

    print(f"  без compile: {timeit(eager_fwd):6.2f} мс")
    print(f"  с compile:   {timeit(comp_fwd):6.2f} мс")

    # ── 3. одношаговый проход базы (потоковый декод) ────────────────────
    print("\n── один токен через базу (12 слоёв), потоковый декод ──")
    rng = np.random.default_rng(0)
    prompt = mx.array(rng.integers(1, 60000, size=(1, 128)).astype(np.int32))
    _, st = base.body(prompt, return_state=True)
    st.eval()
    tok1 = mx.array(np.array([[123]], dtype=np.int32))

    def eager_step():
        h, s = base.body(tok1, state=st, return_state=True)
        mx.eval(h, s.wkv)

    print(f"  без compile: {timeit(eager_step, n=15):6.2f} мс/токен")

    def body_step(idx, wkv, tsh, csh):
        from rwkv_metal.model.state import RWKVState
        h, s = base.body(idx, state=RWKVState(wkv, tsh, csh), return_state=True)
        return h, s.wkv, s.tmix_shift, s.cmix_shift

    compiled_body = mx.compile(body_step)

    def comp_step():
        out = compiled_body(tok1, st.wkv, st.tmix_shift, st.cmix_shift)
        mx.eval(out)

    try:
        print(f"  с compile:   {timeit(comp_step, n=15):6.2f} мс/токен")
    except Exception as e:
        print(f"  с compile:   не собралось — {type(e).__name__}: {str(e)[:160]}")

    sw1 = swap_counters()
    print(f"\nпик RSS процесса: {peak_rss_gb():.2f} ГБ | пул MLX: "
          f"{mx.get_peak_memory()/1e9:.2f} ГБ")
    print(f"своп за прогон: +{sw1[0]-sw0[0]} swapins, +{sw1[1]-sw0[1]} swapouts "
          f"({'чисто' if sw1[1] == sw0[1] else 'БЫЛ СВОП — замеры времени недостоверны'})")


if __name__ == "__main__":
    main()
