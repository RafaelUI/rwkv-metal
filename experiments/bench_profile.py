"""Мишень для профилировки в Instruments: стабильный цикл шагов.
Запусти, дождись маркера, прицепи Instruments к выведенному PID, сними ~15с.

env: D=256 BATCH=16 CKPT=0 FUSED=0 CHUNK=2048 RUN_SEC=45
"""
import os, sys, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from fused_ce import make_fused_ce
import rwkv_metal as rk
from rwkv_metal.model import RWKV7X070

T=512; V=32000
D=int(os.environ.get("D",256)); B=int(os.environ.get("BATCH",16))
CKPT=os.environ.get("CKPT","0")=="1"; FUSED=os.environ.get("FUSED","0")=="1"
CHUNK=int(os.environ.get("CHUNK",2048)); RUN_SEC=int(os.environ.get("RUN_SEC",45))
GB=1/1024**3

cfg=rk.preset("25m",vocab_size=V,n_embd=D)
m=RWKV7X070(cfg).set_dtype("bfloat16"); m._grad_ckpt=CKPT
opt=optim.AdamW(1.5e-3); fused=make_fused_ce(chunk_size=CHUNK)

def loss_fn(x,y):
    if FUSED:
        H=m.body(x); b,t,d=H.shape
        return fused(H.reshape(b*t,d), m.head.weight, y.reshape(b*t)).astype(mx.float32)
    return m.loss(x,y).astype(mx.float32)

lvg=nn.value_and_grad(m, loss_fn); state=[m.state,opt.state]
def _s(x,y):
    l,g=lvg(x,y); g,_=optim.clip_grad_norm(g,1.0); opt.update(m,g); return l
step=mx.compile(_s,inputs=state,outputs=state)

x=mx.random.randint(0,V,(B,T)); y=mx.random.randint(0,V,(B,T)); mx.eval(x,y)

print(f"PID = {os.getpid()}")
print(f"config: d={D} batch={B} ckpt={'ON' if CKPT else 'OFF'} "
      f"CE={'fused' if FUSED else 'naive'}")
print("прогрев (компиляция)...", flush=True)
for _ in range(6): l=step(x,y); mx.eval(l,state)
mx.clear_cache(); mx.reset_peak_memory()

print("\n" + "="*52)
print(f"  ▶▶▶  ПРИЦЕПИ INSTRUMENTS К PID {os.getpid()} СЕЙЧАС  ◀◀◀")
print(f"       цикл идёт {RUN_SEC}с, хватит окна ~15с")
print("="*52 + "\n", flush=True)

t_end=time.time()+RUN_SEC; hb=time.time(); n=0; tA=time.perf_counter()
while time.time()<t_end:
    l=step(x,y); mx.eval(l,state); n+=1
    if time.time()-hb>3:
        tps=n*B*T/(time.perf_counter()-tA); peak=mx.get_peak_memory()*GB
        print(f"  ...{int(t_end-time.time()):2d}с осталось | {tps:.0f} tok/s | peak {peak:.2f} GB", flush=True)
        hb=time.time()
tps=n*B*T/(time.perf_counter()-tA)
print(f"\nготово: {n} шагов, {tps:.0f} tok/s, peak {mx.get_peak_memory()*GB:.2f} GB")
