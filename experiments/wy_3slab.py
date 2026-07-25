"""
wy_3slab.py — WY-ядро чанкового DPLR с ужатым threadgroup-бюджетом.

Мотивация (замерено): жёсткий лимит threadgroup-памяти на M4 = 32 КБ.
Текущий _wy_kernel держит `arena[6*C*D] + Am[4*C*C]`, что при C=32,D=64 даёт
ровно 65536 байт -> не компилируется. Поэтому чанк застрял на C=16, где
матрицы 16x16 почти не грузят матричные блоки и остаётся 32 последовательных
шага по чанкам.

Ужатие:
  1. Am (4 x C x C) пишется СРАЗУ в device (в оригинале он всё равно тут же
     копировался в AMo) -> -16 КБ при C=32.
  2. arena: 6 слэбов -> 3. Ключ в том, что четыре A-матрицы строятся парами:
        A_qk = qh @ kh^T,  A_qb = qh @ bh^T
        A_ab = ah @ bh^T,  A_ak = ah @ kh^T
     kh и bh нужны в обеих парах, а qh и ah — попеременно. Значит одновременно
     живы ровно три слэба: kh, bh и (qh либо ah).
     Дальше слэбы переиспользуются: s1<-v, s2<-рабочий, s0<-ah/k*dec.

Бюджет: 3*C*D*4 байт. C=16 -> 12 КБ, C=32 -> 24 КБ (влезает), C=64 -> 48 КБ
(не влезает: для C=64 нужен ещё тайлинг по D).
"""
import mlx.core as mx

_HDR = """
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal; using namespace mpp;
"""

_wy3_cache = {}


def wy3_kernel(C, D):
    if (C, D) in _wy3_cache:
        return _wy3_cache[(C, D)]
    CD, CC, DD = C * D, C * C, D * D
    src = f"""
    uint bh=thread_position_in_grid.y; uint lid=thread_index_in_threadgroup;
    const device float* Q=q+bh*{CD}; const device float* K=k+bh*{CD}; const device float* V=v+bh*{CD};
    const device float* AL=alpha+bh*{CD}; const device float* BE=beta+bh*{CD};
    const device float* GC=gc+bh*{CD}; const device float* GK=gk+bh*{CD};
    device float* AMo=Am_out+bh*{4*CC}; device float* Uo=u_out+bh*{CD}; device float* WMo=wmat_out+bh*{CD};
    device float* OBo=o_base+bh*{CD}; device float* SBo=s_base+bh*{DD};

    // ТРИ слэба вместо шести; Am живёт в device (AMo)
    threadgroup float arena[{3*CD}];
    threadgroup float* s0=arena+0u*{CD};
    threadgroup float* s1=arena+1u*{CD};
    threadgroup float* s2=arena+2u*{CD};

    auto TG=[](threadgroup float*p,int c,int r){{return tensor<threadgroup float,dextents<int,2>,tensor_inline>(p,dextents<int,2>(c,r));}};
    auto DV=[](device float*p,int c,int r){{return tensor<device float,dextents<int,2>,tensor_inline>(p,dextents<int,2>(c,r));}};
    #define RUN(L,R,O) {{ auto lf=(L); auto rg=(R); auto ds=(O); auto cc=op.get_destination_cooperative_tensor<decltype(lf),decltype(rg),float>(); op.run(lf,rg,cc); cc.store(ds); }}
    #define BAR threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- s1=kh, s2=bh (живут через обе пары), s0=qh ---
    for(uint e=lid;e<{CD};e+=32u){{ float g=GC[e],em=exp(-g),ep=exp(g);
        s1[e]=K[e]*em; s2[e]=BE[e]*em; s0[e]=Q[e]*ep; }}
    BAR
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{C},{D},false,true); tensor_ops::matmul2d<d,execution_simdgroup> op;
       RUN(TG(s0,{D},{C}),TG(s1,{D},{C}),DV(AMo+0u*{CC},{C},{C}))     // A_qk
       RUN(TG(s0,{D},{C}),TG(s2,{D},{C}),DV(AMo+1u*{CC},{C},{C})) }}  // A_qb
    BAR
    // s0 <- ah (qh больше не нужен)
    for(uint e=lid;e<{CD};e+=32u) s0[e]=AL[e]*exp(GC[e]-GK[e]);
    BAR
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{C},{D},false,true); tensor_ops::matmul2d<d,execution_simdgroup> op;
       RUN(TG(s0,{D},{C}),TG(s2,{D},{C}),DV(AMo+2u*{CC},{C},{C}))     // A_ab
       RUN(TG(s0,{D},{C}),TG(s1,{D},{C}),DV(AMo+3u*{CC},{C},{C})) }}  // A_ak
    threadgroup_barrier(mem_flags::mem_device);

    // маски: le для qk/qb, lt для ab/ak  (на device)
    for(uint e=lid;e<{CC};e+=32u){{ uint i=e/{C},j=e%{C};
        if(j> i){{ AMo[0u*{CC}+e]=0; AMo[1u*{CC}+e]=0; }}
        if(j>=i){{ AMo[2u*{CC}+e]=0; AMo[3u*{CC}+e]=0; }} }}
    threadgroup_barrier(mem_flags::mem_device);

    // --- u = (I-A_ab)^-1 @ (A_ak @ v);  s1<-v, s2<-рабочий ---
    for(uint e=lid;e<{CD};e+=32u) s1[e]=V[e];
    BAR
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{C},false,false); tensor_ops::matmul2d<d,execution_simdgroup> op;
       RUN(DV(AMo+3u*{CC},{C},{C}),TG(s1,{D},{C}),TG(s2,{D},{C})) }}
    BAR
    for(uint i=1;i<{C};i++){{ for(uint dd=lid;dd<{D};dd+=32u){{ float acc=s2[i*{D}+dd];
        for(uint n=0;n<i;n++) acc+=AMo[2u*{CC}+i*{C}+n]*s2[n*{D}+dd]; s2[i*{D}+dd]=acc; }} BAR }}
    for(uint e=lid;e<{CD};e+=32u) Uo[e]=s2[e];
    BAR

    // --- wmat = (I-A_ab)^-1 @ ah;  s0<-ah ---
    for(uint e=lid;e<{CD};e+=32u) s0[e]=exp(GC[e]-GK[e])*AL[e];
    BAR
    for(uint i=1;i<{C};i++){{ for(uint dd=lid;dd<{D};dd+=32u){{ float acc=s0[i*{D}+dd];
        for(uint n=0;n<i;n++) acc+=AMo[2u*{CC}+i*{C}+n]*s0[n*{D}+dd]; s0[i*{D}+dd]=acc; }} BAR }}
    for(uint e=lid;e<{CD};e+=32u) WMo[e]=s0[e];
    BAR

    // --- o_base = A_qk @ v ---
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{C},false,false); tensor_ops::matmul2d<d,execution_simdgroup> op;
       RUN(DV(AMo+0u*{CC},{C},{C}),TG(s1,{D},{C}),DV(OBo,{D},{C})) }}
    BAR

    // --- Sbase = (k*dec)^T @ v ;  s0<-k*dec ---
    for(uint e=lid;e<{CD};e+=32u){{ uint dd=e%{D}; s0[e]=K[e]*exp(GC[({C}-1)*{D}+dd]-GC[e]); }}
    BAR
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({D},{D},{C},true,false); tensor_ops::matmul2d<d,execution_simdgroup> op;
       RUN(TG(s0,{D},{C}),TG(s1,{D},{C}),DV(SBo,{D},{D})) }}
    """
    kern = mx.fast.metal_kernel(
        name=f"dplr_wy3_{C}_{D}",
        input_names=["q", "k", "v", "alpha", "beta", "gc", "gk"],
        output_names=["Am_out", "u_out", "wmat_out", "o_base", "s_base"],
        header=_HDR, source=src)
    _wy3_cache[(C, D)] = kern
    return kern


def wy3(q, k, v, alpha, beta, gc, gk, C, D):
    NB = q.shape[0]
    return wy3_kernel(C, D)(
        inputs=[q, k, v, alpha, beta, gc, gk],
        grid=(32, NB, 1), threadgroup=(32, 1, 1),
        output_shapes=[(NB, 4, C, C), (NB, C, D), (NB, C, D), (NB, C, D), (NB, D, D)],
        output_dtypes=[mx.float32] * 5)
