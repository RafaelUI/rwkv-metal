import sys, time, os, gc, resource
sys.path.insert(0, "/Users/s/Develop/rwkv-metal")
import mlx.core as mx
from mlx.utils import tree_flatten
from rwkv_metal.model import load_pretrained
from rwkv_metal.lora import add_lora_rwkvq, LoRAConfig, finetune

def rss_gb():
    # macOS: ru_maxrss в БАЙТАХ (не КБ, как на Linux)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9

def mark(label):
    print(f"[{label}] ru_maxrss={rss_gb():.3f}GB mx_active={mx.get_active_memory()/1e9:.3f}GB "
          f"mx_peak={mx.get_peak_memory()/1e9:.3f}GB mx_cache={mx.get_cache_memory()/1e9:.3f}GB", flush=True)

PTH = "/Users/s/Develop/rwkv7-g1h-1.5b-ctx10240.pth"
SIDECAR = "/tmp/reduction_v2.rwkvq_mlx"

mark("start")
model, cfg = load_pretrained(PTH)
mark("after_load_pth")

model, info = add_lora_rwkvq(model, SIDECAR, rank=16, alpha=16.0, layers=None)
mx.eval(model.parameters())
mark("after_add_lora_rwkvq")

gc.collect()
mark("after_gc_collect")

B, T = 1, 128
def batches():
    while True:
        yield mx.random.randint(0, cfg.vocab_size, (B, T)), mx.random.randint(0, cfg.vocab_size, (B, T))

lcfg = LoRAConfig(lr=1e-4, max_steps=4, grad_accum=1, grad_checkpoint=True,
                   cache_limit_gb=1.5, log_every=1, save_every=0,
                   adapter_path="/tmp/rss_rwkvq.safetensors")

def on_step(step, loss, peak_gb):
    mark(f"after_step_{step}")

finetune(model, batches(), lcfg, on_step=on_step)
mark("end")
print("DONE_OK", flush=True)
