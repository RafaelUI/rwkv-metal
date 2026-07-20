import sys, time, gc, resource
sys.path.insert(0, "/Users/s/Develop/rwkv-metal")
import mlx.core as mx
from rwkv_metal.lora import load_lora_rwkvq_model, LoRAConfig, finetune

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9

def mark(label):
    print(f"[{label}] ru_maxrss={rss_gb():.3f}GB mx_active={mx.get_active_memory()/1e9:.3f}GB "
          f"mx_peak={mx.get_peak_memory()/1e9:.3f}GB", flush=True)

PTH = "/Users/s/Develop/rwkv7-g1h-1.5b-ctx10240.pth"
SIDECAR = "/tmp/reduction_v2.rwkvq_mlx"

mark("start")
model, cfg, info = load_lora_rwkvq_model(PTH, SIDECAR, rank=16, alpha=16.0, layers=None, native="hybrid")
mx.eval(model.parameters())
mark("after_lazy_load_and_wire")

gc.collect()
mark("after_gc")

B, T = 1, 128
def batches():
    while True:
        yield mx.random.randint(0, cfg.vocab_size, (B, T)), mx.random.randint(0, cfg.vocab_size, (B, T))

lcfg = LoRAConfig(lr=1e-4, max_steps=4, grad_accum=1, grad_checkpoint=True,
                   cache_limit_gb=1.5, log_every=1, save_every=0,
                   adapter_path="/tmp/hybrid_rwkvq.safetensors")

def on_step(step, loss, peak_gb):
    mark(f"after_step_{step}")

finetune(model, batches(), lcfg, on_step=on_step)
mark("end")
print("DONE_OK", flush=True)
