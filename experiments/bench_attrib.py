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
                     ctx_len=512, batch_size=20, grad_accum=1, dtype="bfloat16")
BS, T = cfg.batch_size, cfg.ctx_len; TOK = BS*T
mx.random.seed(0)
x = mx.random.randint(1, cfg.vocab_size, (BS, T))
y = mx.random.randint(1, cfg.vocab_size, (BS, T)); mx.eval(x, y)
model = RWKV7(cfg); model = init_weights(model); model.set_dtype(cfg.dtype)
opt = optim.AdamW(learning_rate=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                  eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
step = _make_step_simple(model, opt, cfg)

def timed(fn, n=15, warm=3):
    for _ in range(warm):
        r = fn(); mx.eval(r)
    t0 = time.time()
    for _ in range(n):
        r = fn(); mx.eval(r)
    return (time.time() - t0) / n * 1000

t_body = timed(lambda: model.body(x))                 # WKV-fwd + проекции, без головы/CE
t_loss = timed(lambda: model.loss(x, y))              # + голова + CE, без backward
def full():
    l, nrm = step(x, y); mx.eval(l, nrm); return l
t_full = timed(full)

print(f"  body (fwd WKV+matmul) : {t_body:7.1f} ms", flush=True)
print(f"  +head+CE (loss)       : {t_loss:7.1f} ms   (CE+head ~ {t_loss - t_body:.1f})", flush=True)
print(f"  full step (fwd+bwd+opt): {t_full:7.1f} ms", flush=True)
print(f"  -> backward+opt ~      : {t_full - t_loss:7.1f} ms  ({100*(t_full-t_loss)/t_full:.0f}% шага)", flush=True)
print(f"  -> forward(loss) ~     : {t_loss:7.1f} ms  ({100*t_loss/t_full:.0f}% шага)", flush=True)
