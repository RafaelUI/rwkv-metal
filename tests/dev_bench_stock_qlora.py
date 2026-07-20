import sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-metal")
import mlx.core as mx
from rwkv_metal.model import load_pretrained
from rwkv_metal.lora import add_lora, LoRAConfig, finetune
from rwkv_metal.lora.finetune import quantize_base_model

PTH = "/Users/s/Develop/rwkv7-g1h-1.5b-ctx10240.pth"

t0 = time.time()
model, cfg = load_pretrained(PTH)
print(f"loaded bf16 model in {time.time()-t0:.1f}s, n_layer={cfg.n_layer} n_embd={cfg.n_embd} vocab={cfg.vocab_size}")
print(f"peak after load: {mx.get_peak_memory()/1e9:.2f} GB")

t0 = time.time()
quantize_base_model(model, bits=6)
mx.eval(model.parameters())
print(f"quantize_base_model(6bit) in {time.time()-t0:.1f}s")
print(f"peak after quantize: {mx.get_peak_memory()/1e9:.2f} GB")

t0 = time.time()
model, info = add_lora(model, rank=16, alpha=16.0, quantize_base=6, layers=None)
mx.eval(model.parameters())
print(f"add_lora in {time.time()-t0:.1f}s, info={info}")
print(f"peak after add_lora: {mx.get_peak_memory()/1e9:.2f} GB")

B, T = 1, 128
n_steps = 4

def batches():
    while True:
        x = mx.random.randint(0, cfg.vocab_size, (B, T))
        y = mx.random.randint(0, cfg.vocab_size, (B, T))
        yield x, y

lcfg = LoRAConfig(lr=1e-4, max_steps=n_steps, grad_accum=1, grad_checkpoint=True,
                   cache_limit_gb=1.5, log_every=1, save_every=0,
                   adapter_path="/tmp/bench_stock_lora.safetensors")

step_times = []
last_t = [time.time()]
def on_step(step, loss, peak_gb):
    now = time.time()
    dt = now - last_t[0]
    last_t[0] = now
    step_times.append(dt)
    print(f"  -> step {step} wall={dt:.2f}s loss={loss:.4f} peak={peak_gb:.2f}GB")

finetune(model, batches(), lcfg, on_step=on_step)
print("step times:", step_times)
