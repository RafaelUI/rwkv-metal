"""07.10: из чего состоит память вне MLX после загрузки QLoRA-базы (vmmap -summary: крупные области malloc, живые и пустые).
    python mem_vmmap_0710.py <файл.rwkvq> [native=1|0]"""
import gc, os, re, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import mlx.core as mx
def vm(tag):
    o = subprocess.run(["/usr/bin/vmmap", "-summary", str(os.getpid())], capture_output=True, text=True).stdout
    rows = [l for l in o.splitlines() if re.search(r"Physical footprint|REGION TYPE|TOTAL|\d(\.\d)?G |\d{2,}(\.\d)?M ", l)]
    print("== %s | MLX активная %.0f кеш %.0f МиБ" % (tag, mx.get_active_memory() / 2**20, mx.get_cache_memory() / 2**20))
    for l in rows: print("   " + l[:150])
    sys.stdout.flush()
FILE = sys.argv[1]; NATIVE = (sys.argv[2] != "0") if len(sys.argv) > 2 else True
import numpy  # noqa
vm("после импортов")
from rwkv_metal.lora import load_rwkvq_model
model, cfg, info = load_rwkvq_model(FILE, rank=16, verbose=False, native=NATIVE)
gc.collect(); mx.clear_cache(); import time; time.sleep(3)
vm("после загрузки, gc, clear_cache (native=%s)" % NATIVE)
