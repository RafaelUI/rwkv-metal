"""Свип по размерности d: утилизация GPU vs размер модели.
Гипотеза: крупнее d -> крупнее матмулы -> выше % от пика FLOPS.
Метрика: достигнутые эффективные TFLOPS = FLOP/токен(train) * tok/s.
"""
import os, sys, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten
import rwkv_metal as rk
from rwkv_metal.model import RWKV7X070

T=512; V=32000; B=int(os.environ.get("B",8)); ITERS=int(os.environ.get("ITERS",8))
DIMS=[int(d) for d in os.environ.get("DIMS","256,384,512").split(",")]
PEAK=8.6e12  # bf16 номинал base M4
GB=1/1024**3

def flop_per_tok(model, head_size):
    # 6N по матмулам БЕЗ embedding (lookup, 0 FLOP); + WKV рекуррентность
    mm=0; emb=0; L=0; d=0
    for n,w in tree_flatten(model.parameters()):
        nm=n.lower()
        if 'emb' in nm: emb+=w.size
        elif w.ndim==2 and w.shape[0]>=64 and w.shape[1]>=64: mm+=w.size
        if 'blocks.' in nm: L=max(L,int(nm.split('blocks.')[1].split('.')[0])+1)
    d=model.head.weight.shape[1]
    wkv = L*(d//head_size)*5*head_size*head_size
    return 6*mm + 3*wkv

for d in DIMS:
    try:
        cfg=rk.preset('25m',vocab_size=V,n_embd=d)
        m=RWKV7X070(cfg).set_dtype('bfloat16'); m._grad_ckpt=False
        opt=optim.AdamW(1.5e-3)
        fpt=flop_per_tok(m, cfg.head_size)
        x=mx.random.randint(0,V,(B,T)); y=mx.random.randint(0,V,(B,T)); mx.eval(x,y)
        lvg=nn.value_and_grad(m, lambda x,y: m.loss(x,y).astype(mx.float32))
        state=[m.state,opt.state]
        def _s(x,y):
            l,g=lvg(x,y); g,_=optim.clip_grad_norm(g,1.0); opt.update(m,g); return l
        step=mx.compile(_s,inputs=state,outputs=state)
        for _ in range(3): l=step(x,y); mx.eval(l,state)
        mx.clear_cache(); mx.reset_peak_memory()
        t0=time.perf_counter()
        for _ in range(ITERS): l=step(x,y); mx.eval(l,state)
        dt=(time.perf_counter()-t0)/ITERS
        toks=B*T/dt; tflops=fpt*toks/1e12; peak=mx.get_peak_memory()*GB
        tot=sum(w.size for _,w in tree_flatten(m.parameters()))/1e6
        print(f"d={d:4d} ({tot:4.0f}M params, {d//cfg.head_size} heads) | "
              f"{fpt/1e6:5.0f} MFLOP/tok | {toks:6.0f} tok/s | "
              f"{tflops:.2f} TFLOPS эфф = {tflops/PEAK*100:4.1f}% пика | peak {peak:.1f}GB")
        del m,opt; gc.collect(); mx.clear_cache()
    except Exception as ex:
        print(f"d={d}: {type(ex).__name__}: {str(ex)[:55]}")
