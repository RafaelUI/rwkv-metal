import os, sys, time
import mlx.core as mx
import mlx.optimizers as optim
def _t(*c):
    for f in c:
        try: f(); return
        except Exception: pass
_t(lambda: mx.set_memory_limit(12*1024**3), lambda: mx.metal.set_memory_limit(12*1024**3))
_t(lambda: mx.set_cache_limit(2*1024**3), lambda: mx.metal.set_cache_limit(2*1024**3))
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
from rwkv_metal.pretrain.config import PretrainConfig
from rwkv_metal.model.rwkv7 import RWKV7, init_weights
from rwkv_metal.pretrain.trainer import _make_step_simple
cfg = PretrainConfig(n_layer=12, n_embd=256, head_size=64, vocab_size=21248,
                     ctx_len=512, batch_size=27, grad_accum=1, dtype="bfloat16")
BS, T = cfg.batch_size, cfg.ctx_len; TOK = BS*T; WARM, N = 3, 20
mx.random.seed(0)
x = mx.random.randint(1, cfg.vocab_size, (BS, T))
y = mx.random.randint(1, cfg.vocab_size, (BS, T)); mx.eval(x, y)
model = RWKV7(cfg); model = init_weights(model); model.set_dtype(cfg.dtype)
opt = optim.AdamW(learning_rate=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                  eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
step = _make_step_simple(model, opt, cfg)
for _ in range(WARM):
    l, nrm = step(x, y); mx.eval(l, nrm)
t0 = time.time()
for _ in range(N):
    l, nrm = step(x, y); mx.eval(l, nrm)
dt = time.time() - t0
print(f"  KERNEL: {TOK*N/dt:8.0f} tok/s | {dt/N*1000:7.1f} ms/step  (B{BS} T{T}, host removed)", flush=True)
