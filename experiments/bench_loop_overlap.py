"""
bench_loop_overlap.py - memory-safe. Два замера ПООЧЕРЁДНО:
  BEFORE (baseline) = текущий trainer.py: ds.batch()+mx.eval+.item() каждый шаг.
  AFTER  (overlap)  = фоновый прелоад numpy + async_eval с КЭПОМ 2 шага в полёте
                      (block prev перед dispatch next) + .item() раз в K.
Кэп в полёте = защита от OOM/фриза. B=20, мягкий лимит памяти 12GB.
Запуск:  .venv/bin/python -u experiments/bench_loop_overlap.py
"""
import os, sys, time, threading, queue
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
import mlx.optimizers as optim

# ── мягкие лимиты (best-effort, разные версии MLX) ──
def _try(*calls):
    for fn in calls:
        try: fn(); return
        except Exception: pass
_try(lambda: mx.set_memory_limit(12 * 1024**3),
     lambda: mx.metal.set_memory_limit(12 * 1024**3))
_try(lambda: mx.set_cache_limit(2 * 1024**3),
     lambda: mx.metal.set_cache_limit(2 * 1024**3))

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
from rwkv_metal.pretrain.config import PretrainConfig
from rwkv_metal.model.rwkv7 import RWKV7, init_weights
from rwkv_metal.pretrain.trainer import _make_step_simple
from rwkv_metal.pretrain.dataset import BinDataset

cfg = PretrainConfig(n_layer=12, n_embd=256, head_size=64, vocab_size=21248,
                     ctx_len=512, batch_size=20, grad_accum=1, dtype="bfloat16")
BS, T = cfg.batch_size, cfg.ctx_len
TOK = BS * T
WARM, N, K = 3, 20, 10

DATA = "/tmp/rwkv_loopbench.bin"
NEED = 60_000_000
if (not os.path.exists(DATA)) or os.path.getsize(DATA) < NEED * 2:
    print("  Генерирую синтетику (~120MB)...", flush=True)
    np.random.randint(0, cfg.vocab_size, size=NEED, dtype=np.uint16).tofile(DATA)
ds = BinDataset(DATA, cfg.ctx_len)

def np_batch(step):
    stride = T + 1
    starts = [(step * BS + i) * stride % (ds.n - stride) for i in range(BS)]
    x = np.stack([ds.data[s:s + T].astype(np.int32) for s in starts])
    y = np.stack([ds.data[s + 1:s + 1 + T].astype(np.int32) for s in starts])
    return x, y

print("Сборка модели...", flush=True)
model = RWKV7(cfg); model = init_weights(model); model.set_dtype(cfg.dtype)
opt = optim.AdamW(learning_rate=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                  eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
step_fn = _make_step_simple(model, opt, cfg)
def st():
    return [v for _, v in tree_flatten([model.state, opt.state])]

def baseline():
    for s in range(WARM):
        x, y = ds.batch(BS, s); l, nrm = step_fn(x, y); mx.eval(l, nrm); l.item()
    t0 = time.time()
    for s in range(WARM, WARM + N):
        x, y = ds.batch(BS, s); l, nrm = step_fn(x, y); mx.eval(l, nrm); l.item()
    return TOK * N / (time.time() - t0)

def overlap():
    q = queue.Queue(maxsize=2)            # кэп прелоада
    def producer():
        for s in range(WARM + N):
            q.put(np_batch(s))
    th = threading.Thread(target=producer, daemon=True); th.start()
    prev = None
    for _ in range(WARM):
        xnp, ynp = q.get(); x, y = mx.array(xnp), mx.array(ynp)
        l, nrm = step_fn(x, y); mx.async_eval(l, nrm, *st())
        if prev is not None: mx.eval(prev)     # кэп: <=2 шага в полёте
        prev = l
    mx.eval(prev); prev = None
    t0 = time.time(); last = None
    for i in range(N):
        xnp, ynp = q.get(); x, y = mx.array(xnp), mx.array(ynp)
        l, nrm = step_fn(x, y); mx.async_eval(l, nrm, *st())
        if prev is not None: mx.eval(prev)     # block предыдущего -> bounded память
        prev = l; last = l
        if (i + 1) % K == 0: last.item()
    mx.eval(last)
    th.join()
    return TOK * N / (time.time() - t0)

print(f"\nM4 | {cfg.n_layer}L x {cfg.n_embd}d | B{BS} T{T} {cfg.dtype} | N={N}\n" + "-" * 52, flush=True)
print("  [1/2] BEFORE (baseline)...", flush=True)
b = baseline(); print(f"      BEFORE : {b:8.0f} tok/s", flush=True)
_try(lambda: mx.clear_cache(), lambda: mx.metal.clear_cache())
print("  [2/2] AFTER (overlap)...", flush=True)
o = overlap(); print(f"      AFTER  : {o:8.0f} tok/s", flush=True)
print("-" * 52, flush=True)
print(f"  Прирост: {o - b:+.0f} tok/s  (x{o / b:.2f})  | {TOK/b*1000:.1f} -> {TOK/o*1000:.1f} ms/step", flush=True)
