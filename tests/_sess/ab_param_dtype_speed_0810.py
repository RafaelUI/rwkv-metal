"""08.10: шаг обучения QLoRA, param_dtype bf16 против fp16, ОДИН процесс, две модели, чередование ABBA (закон 1).
    python ab_param_dtype_speed_0810.py <file.rwkvq> [раундов=6]"""
import os, sys, time, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from rwkv_metal.lora import load_rwkvq_model
f = sys.argv[1]; R = int(sys.argv[2]) if len(sys.argv) > 2 else 6
r2 = np.random.RandomState(5)
x = mx.array(r2.randint(1, 60000, size=(1, 512)).astype(np.int32)); y = mx.array(r2.randint(1, 60000, size=(1, 512)).astype(np.int32))
S = {}
for pd in ("bf16", "fp16"):
    mx.random.seed(0)
    m, _, _ = load_rwkvq_model(f, rank=16, verbose=False, param_dtype=pd); m._grad_ckpt = True
    gf = nn.value_and_grad(m, lambda mm, a, b: mm.loss(a, b).astype(mx.float32)); opt = optim.AdamW(learning_rate=1e-4)
    S[pd] = (m, gf, opt)
def step(pd):
    m, gf, opt = S[pd]; l, g = gf(m, x, y); opt.update(m, g); mx.eval(l, m.state, opt.state)
for pd in S: step(pd); step(pd)
T = {pd: [] for pd in S}
for r in range(R):
    for pd in (("bf16", "fp16", "fp16", "bf16") if r % 2 == 0 else ("fp16", "bf16", "bf16", "fp16")):
        t0 = time.perf_counter(); step(pd); T[pd].append(time.perf_counter() - t0)
b, h = np.array(T["bf16"]), np.array(T["fp16"])
d = h / b - 1
print(os.path.basename(f), "bf16 %.0f ms, fp16 %.0f ms (медианы), fp16/bf16 %+.1f%% [по парам: медиана %+.1f%%, мин %+.1f, макс %+.1f]" % (
    1e3 * np.median(b), 1e3 * np.median(h), 100 * (np.median(h) / np.median(b) - 1), 100 * np.median(d), 100 * d.min(), 100 * d.max()))
