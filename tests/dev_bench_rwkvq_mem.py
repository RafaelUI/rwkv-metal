import sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-metal")
import mlx.core as mx
from mlx.utils import tree_flatten
from rwkv_metal.model import load_pretrained
from rwkv_metal.lora import add_lora_rwkvq, add_lora, LoRAConfig, finetune
from rwkv_metal.lora.finetune import quantize_base_model

PTH = "/Users/s/Develop/rwkv7-g1h-1.5b-ctx10240.pth"
SIDECAR = "/tmp/reduction_v2.rwkvq_mlx"

def nbytes(model):
    total = 0
    for _, v in tree_flatten(model.parameters()):
        total += v.nbytes
    return total

print("========== RWKVQ QLoRA ==========")
model, cfg = load_pretrained(PTH)
model, info = add_lora_rwkvq(model, SIDECAR, rank=16, alpha=16.0, layers=None)
mx.eval(model.parameters())
print(f"static footprint (sum nbytes model.parameters()) = {nbytes(model)/1e9:.3f} GB")

mx.reset_peak_memory()
print(f"peak reset, current after reset check: {mx.get_active_memory()/1e9:.3f} GB active")

B, T = 1, 128
def batches():
    while True:
        yield mx.random.randint(0, cfg.vocab_size, (B, T)), mx.random.randint(0, cfg.vocab_size, (B, T))

lcfg = LoRAConfig(lr=1e-4, max_steps=3, grad_accum=1, grad_checkpoint=True,
                   cache_limit_gb=1.5, log_every=1, save_every=0,
                   adapter_path="/tmp/mem_rwkvq.safetensors")
finetune(model, batches(), lcfg)
print(f"TRAINING-ONLY peak (after reset, before steps) = {mx.get_peak_memory()/1e9:.3f} GB")
print(f"active memory at end = {mx.get_active_memory()/1e9:.3f} GB")

del model
import gc; gc.collect()
mx.reset_peak_memory()

print("\n========== STOCK nn.quantize QLoRA (bits=6, для честного сравнения) ==========")
model2, cfg2 = load_pretrained(PTH)
quantize_base_model(model2, bits=6)
model2, info2 = add_lora(model2, rank=16, alpha=16.0, quantize_base=6, layers=None)
mx.eval(model2.parameters())
print(f"static footprint (sum nbytes model.parameters()) = {nbytes(model2)/1e9:.3f} GB")

mx.reset_peak_memory()
lcfg2 = LoRAConfig(lr=1e-4, max_steps=3, grad_accum=1, grad_checkpoint=True,
                    cache_limit_gb=1.5, log_every=1, save_every=0,
                    adapter_path="/tmp/mem_stock.safetensors")
finetune(model2, batches(), lcfg2)
print(f"TRAINING-ONLY peak (after reset, before steps) = {mx.get_peak_memory()/1e9:.3f} GB")
print(f"active memory at end = {mx.get_active_memory()/1e9:.3f} GB")
print("DONE_OK")
