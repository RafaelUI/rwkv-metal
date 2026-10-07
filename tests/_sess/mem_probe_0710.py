"""07.10: память загрузки QLoRA-базы в rwkv-metal -- есть ли те же болезни, что лечились в Metal-бэкенде rwkv-quant 05.10:
(1) память ВНЕ MLX после загрузки (освобождённые крупные транзиенты хоста, которые аллокатор macOS не отдаёт);
(2) что прирастает лениво на первом проходе / первом шаге и остаётся; (3) пик шага против установившегося уровня.
Стадии: загрузка -> gc + clear_cache -> первый прямой проход T=64 -> первый шаг обучения T=512 -> ещё три шага -> clear_cache.
«вне MLX» = footprint процесса - (активная + кеш MLX).
    python mem_probe_0710.py <файл.rwkvq> [native=1|0]"""
import gc, os, re, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
def foot():
    o = subprocess.run(["/usr/bin/footprint", str(os.getpid())], capture_output=True, text=True).stdout
    m = re.search(r"Footprint:\s*([\d.]+)\s*(KB|MB|GB)", o); return float(m.group(1)) * {"KB": 1e-3, "MB": 1, "GB": 1024}[m.group(2)]
sw = lambda: float(subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split()[5].rstrip("M"))
sw0 = sw(); M = 2 ** 20
def stage(n):
    f, a, c = foot(), mx.get_active_memory() / M, mx.get_cache_memory() / M
    print("%-46s footprint %6.0f | MLX активная %6.0f кеш %5.0f пик %6.0f | вне MLX %6.0f | своп %+5.0f" % (n, f, a, c, mx.get_peak_memory() / M, f - a - c, sw() - sw0), flush=True)
FILE = sys.argv[1]; NATIVE = (sys.argv[2] != "0") if len(sys.argv) > 2 else True
print("файл %.0f МиБ, native=%s" % (os.path.getsize(FILE) / M, NATIVE)); stage("старт")
from rwkv_metal.lora import load_rwkvq_model, rwkvq_linear as rl
t0 = time.time(); model, cfg, info = load_rwkvq_model(FILE, rank=16, verbose=False, native=NATIVE)
stage("загрузка (%.0f с)" % (time.time() - t0))
gc.collect(); mx.clear_cache(); stage("gc + clear_cache")
print("   кеш загрузчика _SIDECAR_CACHE: %d записей; базы: %s" % (len(rl._SIDECAR_CACHE), sorted({type(m).__name__ for _, m in model.named_modules() if type(m).__name__.startswith("Rwkvq")})))
rs = np.random.RandomState(5)
x64 = mx.array(rs.randint(1, 60000, size=(1, 64)).astype(np.int32))
lg = model(x64); mx.eval(lg); del lg; stage("первый прямой проход T=64")
mx.clear_cache(); stage("  clear_cache")
model._grad_ckpt = True
x = mx.array(rs.randint(1, 60000, size=(1, 512)).astype(np.int32)); y = mx.array(rs.randint(1, 60000, size=(1, 512)).astype(np.int32))
gf = nn.value_and_grad(model, lambda m, a, b: m.loss(a, b).astype(mx.float32)); opt = optim.AdamW(learning_rate=1e-4)
def step():
    loss, grads = gf(model, x, y); opt.update(model, grads); mx.eval(loss, model.state, opt.state)
mx.reset_peak_memory(); step(); stage("первый шаг обучения T=512")
mx.reset_peak_memory(); t0 = time.time()
for _ in range(3): step()
stage("ещё три шага (%.0f мс/шаг)" % ((time.time() - t0) / 3 * 1e3))
mx.clear_cache(); gc.collect(); stage("  clear_cache")
