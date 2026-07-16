"""
s3_dplr_kernel.py — S3: фьюзед чанковый DPLR на Metal (MPP matmul2d) через MLX.

Бэкенд: tensor_ops::matmul2d, операнды — threadgroup tensor_inline (shader-allocated),
strides[0]=1 (extents в порядке cols,rows). transpose_right=true даёт @^T даром.
fp32 (bf16->float требует OS26.1+; валидация в fp32 против MLX-оракула).

S3.1 (готово): конструкция 4 A-матриц чанка со свёрткой decay на чипе.
  qhat=q*exp(gc), khat=k*exp(-gc), bhat=beta*exp(-gc), ahat=alpha*exp(gc-gk)
  A_qk=qhat@khat^T, A_qb=qhat@bhat^T, A_ab=ahat@bhat^T, A_ak=ahat@khat^T (все tr=true).
  Маски (j<=i / j<i) пока накладываются вызывающей стороной (в кернел — на S3.2/3.3).
"""
import mlx.core as mx

_HDR = """
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal; using namespace mpp;
"""

_amats_cache = {}


def _amats_kernel(C, D):
    key = (C, D)
    if key in _amats_cache:
        return _amats_cache[key]
    src = f"""
    threadgroup float qhat[{C*D}], khat[{C*D}], bhat[{C*D}], ahat[{C*D}];
    uint lid = thread_index_in_threadgroup;
    for (uint e = lid; e < {C*D}; e += 32u) {{
        float g = gc[e], em = exp(-g), ep = exp(g);
        qhat[e] = q[e]     * ep;
        khat[e] = k[e]     * em;
        bhat[e] = beta[e]  * em;
        ahat[e] = alpha[e] * exp(g - gk[e]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr auto desc = tensor_ops::matmul2d_descriptor({C}, {C}, {D}, false, true);
    tensor_ops::matmul2d<desc, execution_simdgroup> op;
    auto E = [](threadgroup float* p){{ return tensor<threadgroup float, dextents<int,2>, tensor_inline>(p, dextents<int,2>({D},{C})); }};
    auto O = [](device float* p){{ return tensor<device float, dextents<int,2>, tensor_inline>(p, dextents<int,2>({C},{C})); }};
    auto tQ=E(qhat); auto tK=E(khat); auto tB=E(bhat); auto tA=E(ahat);
    {{ auto c=op.get_destination_cooperative_tensor<decltype(tQ),decltype(tK),float>(); op.run(tQ,tK,c); auto o=O(out_qk); c.store(o); }}
    {{ auto c=op.get_destination_cooperative_tensor<decltype(tQ),decltype(tB),float>(); op.run(tQ,tB,c); auto o=O(out_qb); c.store(o); }}
    {{ auto c=op.get_destination_cooperative_tensor<decltype(tA),decltype(tB),float>(); op.run(tA,tB,c); auto o=O(out_ab); c.store(o); }}
    {{ auto c=op.get_destination_cooperative_tensor<decltype(tA),decltype(tK),float>(); op.run(tA,tK,c); auto o=O(out_ak); c.store(o); }}
    """
    kern = mx.fast.metal_kernel(
        name=f"dplr_amats_{C}_{D}",
        input_names=["q", "k", "beta", "alpha", "gc", "gk"],
        output_names=["out_qk", "out_qb", "out_ab", "out_ak"],
        header=_HDR, source=src)
    _amats_cache[key] = kern
    return kern


def compute_amats(q, k, alpha, beta, gc, gk):
    """Один чанк [C,D]. Возврат RAW (без маски) A_qk,A_qb,A_ab,A_ak [C,C]."""
    C, D = q.shape
    kern = _amats_kernel(C, D)
    return kern(inputs=[q, k, beta, alpha, gc, gk],
                grid=(32, 1, 1), threadgroup=(32, 1, 1),
                output_shapes=[(C, C)] * 4, output_dtypes=[mx.float32] * 4)


# --- S3.2: построчный треугольный solve (I - A_ab) X = RHS ----------------
_trisolve_cache = {}


def _trisolve_kernel(C, D):
    key = (C, D)
    if key in _trisolve_cache:
        return _trisolve_cache[key]
    src = f"""
    threadgroup float Xs[{C*D}];
    threadgroup float Aab[{C*C}];
    uint lid = thread_index_in_threadgroup;
    for (uint e=lid; e<{C*D}; e+=32u) Xs[e] = rhs[e];
    for (uint e=lid; e<{C*C}; e+=32u) Aab[e] = aab[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=1; i<{C}; i++) {{
        for (uint d=lid; d<{D}; d+=32u) {{
            float acc = Xs[i*{D}+d];
            for (uint n=0; n<i; n++) acc += Aab[i*{C}+n] * Xs[n*{D}+d];
            Xs[i*{D}+d] = acc;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    for (uint e=lid; e<{C*D}; e+=32u) out[e] = Xs[e];
    """
    kern = mx.fast.metal_kernel(
        name=f"dplr_trisolve_{C}_{D}",
        input_names=["aab", "rhs"], output_names=["out"],
        header="using namespace metal;\n", source=src)
    _trisolve_cache[key] = kern
    return kern


def trisolve(A_ab, rhs):
    """Решить (I - A_ab) X = rhs. A_ab [C,C] СТРОГО нижнетреуг. (маска j<i). rhs [C,D]."""
    C = A_ab.shape[0]; D = rhs.shape[-1]
    kern = _trisolve_kernel(C, D)
    r = kern(inputs=[A_ab, rhs], grid=(32, 1, 1), threadgroup=(32, 1, 1),
             output_shapes=[(C, D)], output_dtypes=[mx.float32])
    return r[0]


# --- S3.3a: A-матрицы с МАСКОЙ В КЕРНЕЛЕ (le для qk/qb, lt для ab/ak) ------
_amats_masked_cache = {}


def _amats_masked_kernel(C, D):
    key = (C, D)
    if key in _amats_masked_cache:
        return _amats_masked_cache[key]
    src = f"""
    threadgroup float qhat[{C*D}],khat[{C*D}],bhat[{C*D}],ahat[{C*D}];
    threadgroup float Am[4*{C*C}];
    uint lid=thread_index_in_threadgroup;
    for(uint e=lid;e<{C*D};e+=32u){{ float g=gc[e],em=exp(-g),ep=exp(g);
        qhat[e]=q[e]*ep; khat[e]=k[e]*em; bhat[e]=beta[e]*em; ahat[e]=alpha[e]*exp(g-gk[e]); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr auto desc=tensor_ops::matmul2d_descriptor({C},{C},{D},false,true);
    tensor_ops::matmul2d<desc,execution_simdgroup> op;
    auto E=[](threadgroup float*p){{return tensor<threadgroup float,dextents<int,2>,tensor_inline>(p,dextents<int,2>({D},{C}));}};
    auto T=[](threadgroup float*p){{return tensor<threadgroup float,dextents<int,2>,tensor_inline>(p,dextents<int,2>({C},{C}));}};
    auto tQ=E(qhat);auto tK=E(khat);auto tB=E(bhat);auto tA=E(ahat);
    {{auto c=op.get_destination_cooperative_tensor<decltype(tQ),decltype(tK),float>();op.run(tQ,tK,c);auto o=T(Am+0*{C*C});c.store(o);}}
    {{auto c=op.get_destination_cooperative_tensor<decltype(tQ),decltype(tB),float>();op.run(tQ,tB,c);auto o=T(Am+1*{C*C});c.store(o);}}
    {{auto c=op.get_destination_cooperative_tensor<decltype(tA),decltype(tB),float>();op.run(tA,tB,c);auto o=T(Am+2*{C*C});c.store(o);}}
    {{auto c=op.get_destination_cooperative_tensor<decltype(tA),decltype(tK),float>();op.run(tA,tK,c);auto o=T(Am+3*{C*C});c.store(o);}}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint e=lid;e<{C*C};e+=32u){{ uint i=e/{C}, j=e%{C};
        if(j> i){{ Am[0*{C*C}+e]=0; Am[1*{C*C}+e]=0; }}
        if(j>=i){{ Am[2*{C*C}+e]=0; Am[3*{C*C}+e]=0; }} }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint e=lid;e<{C*C};e+=32u){{ out_qk[e]=Am[0*{C*C}+e]; out_qb[e]=Am[1*{C*C}+e];
        out_ab[e]=Am[2*{C*C}+e]; out_ak[e]=Am[3*{C*C}+e]; }}
    """
    kern = mx.fast.metal_kernel(
        name=f"dplr_amats_masked_{C}_{D}",
        input_names=["q", "k", "beta", "alpha", "gc", "gk"],
        output_names=["out_qk", "out_qb", "out_ab", "out_ak"],
        header=_HDR, source=src)
    _amats_masked_cache[key] = kern
    return kern


def compute_amats_masked(q, k, alpha, beta, gc, gk):
    """Один чанк [C,D]. Возврат МАСКИРОВАННЫХ A_qk,A_qb (j<=i), A_ab,A_ak (j<i)."""
    C, D = q.shape
    kern = _amats_masked_kernel(C, D)
    return kern(inputs=[q, k, beta, alpha, gc, gk],
                grid=(32, 1, 1), threadgroup=(32, 1, 1),
                output_shapes=[(C, C)] * 4, output_dtypes=[mx.float32] * 4)


# --- S3.3b: фьюзед forward одного чанка при S=0 --------------------------
# masked A (фаза A) → RHS_u=A_ak@v → in-kernel trisolve → o=A_qk@v + A_qb@u.
# Арена: hats[4*C*D] держат qhat/khat/bhat/ahat в фазе A; после барьера
# мертвы → переиспользуются под v(slot0)/u(slot1)/o(slot2)/tmp(slot3).
# Am[4*C*C] резидентны весь kernel: qk(0) qb(1) ab(2) ak(3).
# apply-матмулы второй формы [C,C]@[C,D] tr=false: desc(C,D,C,false,false),
#   left=A[C,C] extents(C,C); right=v[C,D] extents(D,C); out extents(D,C).
_chunk_s0_cache = {}


def _chunk_s0_kernel(C, D):
    key = (C, D)
    if key in _chunk_s0_cache:
        return _chunk_s0_cache[key]
    CD = C * D
    CC = C * C
    src = f"""
    threadgroup float hats[{4*CD}];   // qhat|khat|bhat|ahat  -> potom v|u|o|tmp
    threadgroup float Am[{4*CC}];     // qk(0) qb(1) ab(2) ak(3)
    uint lid = thread_index_in_threadgroup;
    threadgroup float* qhat = hats + 0u*{CD};
    threadgroup float* khat = hats + 1u*{CD};
    threadgroup float* bhat = hats + 2u*{CD};
    threadgroup float* ahat = hats + 3u*{CD};

    for (uint e=lid; e<{CD}; e+=32u) {{ float g=gc[e], em=exp(-g), ep=exp(g);
        qhat[e]=q[e]*ep; khat[e]=k[e]*em; bhat[e]=beta[e]*em; ahat[e]=alpha[e]*exp(g-gk[e]); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto CDt=[](threadgroup float*p){{return tensor<threadgroup float,dextents<int,2>,tensor_inline>(p,dextents<int,2>({D},{C}));}};
    auto CCt=[](threadgroup float*p){{return tensor<threadgroup float,dextents<int,2>,tensor_inline>(p,dextents<int,2>({C},{C}));}};

    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{C},{D},false,true);
       tensor_ops::matmul2d<d,execution_simdgroup> op;
       auto L=CDt(qhat); auto R=CDt(khat);
       auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c);
       auto o=CCt(Am+0u*{CC}); c.store(o); }}
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{C},{D},false,true);
       tensor_ops::matmul2d<d,execution_simdgroup> op;
       auto L=CDt(qhat); auto R=CDt(bhat);
       auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c);
       auto o=CCt(Am+1u*{CC}); c.store(o); }}
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{C},{D},false,true);
       tensor_ops::matmul2d<d,execution_simdgroup> op;
       auto L=CDt(ahat); auto R=CDt(bhat);
       auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c);
       auto o=CCt(Am+2u*{CC}); c.store(o); }}
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{C},{D},false,true);
       tensor_ops::matmul2d<d,execution_simdgroup> op;
       auto L=CDt(ahat); auto R=CDt(khat);
       auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c);
       auto o=CCt(Am+3u*{CC}); c.store(o); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e=lid; e<{CC}; e+=32u) {{ uint i=e/{C}, j=e%{C};
        if (j> i) {{ Am[0u*{CC}+e]=0; Am[1u*{CC}+e]=0; }}
        if (j>=i) {{ Am[2u*{CC}+e]=0; Am[3u*{CC}+e]=0; }} }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* vbuf = hats + 0u*{CD};
    threadgroup float* ubuf = hats + 1u*{CD};
    threadgroup float* obuf = hats + 2u*{CD};
    threadgroup float* tbuf = hats + 3u*{CD};
    for (uint e=lid; e<{CD}; e+=32u) vbuf[e] = v[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{C},false,false);
       tensor_ops::matmul2d<d,execution_simdgroup> op;
       auto L=CCt(Am+3u*{CC}); auto R=CDt(vbuf);
       auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c);
       auto o=CDt(ubuf); c.store(o); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i=1; i<{C}; i++) {{
        for (uint dd=lid; dd<{D}; dd+=32u) {{
            float acc = ubuf[i*{D}+dd];
            for (uint n=0; n<i; n++) acc += Am[2u*{CC}+i*{C}+n] * ubuf[n*{D}+dd];
            ubuf[i*{D}+dd] = acc;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{C},false,false);
       tensor_ops::matmul2d<d,execution_simdgroup> op;
       auto L=CCt(Am+0u*{CC}); auto R=CDt(vbuf);
       auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c);
       auto o=CDt(obuf); c.store(o); }}
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{C},false,false);
       tensor_ops::matmul2d<d,execution_simdgroup> op;
       auto L=CCt(Am+1u*{CC}); auto R=CDt(ubuf);
       auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c);
       auto o=CDt(tbuf); c.store(o); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e=lid; e<{CD}; e+=32u) out_o[e] = obuf[e] + tbuf[e];
    """
    kern = mx.fast.metal_kernel(
        name=f"dplr_chunk_s0_{C}_{D}",
        input_names=["q", "k", "v", "alpha", "beta", "gc", "gk"],
        output_names=["out_o"],
        header=_HDR, source=src)
    _chunk_s0_cache[key] = kern
    return kern


def chunk_fwd_s0(q, k, v, alpha, beta, gc, gk):
    """Fused forward odnogo chunka pri S=0. Vse vhody [C,D]. Vozvrat o [C,D]."""
    C, D = q.shape
    kern = _chunk_s0_kernel(C, D)
    r = kern(inputs=[q, k, v, alpha, beta, gc, gk],
             grid=(32, 1, 1), threadgroup=(32, 1, 1),
             output_shapes=[(C, D)], output_dtypes=[mx.float32])
    return r[0]


# --- S3.3c: межчанковый step-кернел (размотка B*H) ----------------------
# Один threadgroup = один simdgroup = один (b,h) на ОДНОМ чанке. Грид по BH.
# Драйвер гоняет N чанков последовательно, обновляя S на хосте (тривиальный
# elementwise-рекур; матмулы S-апдейта внутри кернела). Контракт dplr_chunkwise_mlx:
#   v2 = u + wmat@S;  o = A_qk@v + A_qb@v2 + (q·e^gc)@S
#   t1 = (k·decay)^T@v;  t2 = (beta·decay)^T@v2;  decay[i,d]=exp(gc_last[d]-gc[i,d])
#   S' = S·e^gc_last[d] + t1 + t2   (хост)
# Формы матмула (probe-проверены):
#   A_xy=xhat@yhat^T desc(C,C,D,F,T); apply A@x desc(C,D,C,F,F);
#   x@S desc(C,Dv,Dk,F,F) [tg×dev]; kd^T@v desc(Dk,Dv,C,T,F).
# Арена: arena[6*C*D] (s0=qhat..s5) + Am[4*C*C]. C=16: 24KB+4KB=28KB (<32KB).
_step_cache = {}


def _step_kernel(C, D):
    key = (C, D)
    if key in _step_cache:
        return _step_cache[key]
    CD = C * D
    CC = C * C
    DD = D * D
    src = f"""
    uint bh = thread_position_in_grid.y;
    uint lid = thread_index_in_threadgroup;
    const device float* Q = q     + bh*{CD};
    const device float* K = k     + bh*{CD};
    const device float* V = v     + bh*{CD};
    const device float* AL= alpha + bh*{CD};
    const device float* BE= beta  + bh*{CD};
    const device float* GC= gc    + bh*{CD};
    const device float* GK= gk    + bh*{CD};
    device float* Sp = (device float*)(S_in + bh*{DD});
    device float* OO = out_o + bh*{CD};
    device float* T1 = t1    + bh*{DD};
    device float* T2 = t2    + bh*{DD};

    threadgroup float arena[{6*CD}];
    threadgroup float Am[{4*CC}];
    threadgroup float* s0=arena+0u*{CD};  // qhat (живёт до o3)
    threadgroup float* s1=arena+1u*{CD};  // khat -> v
    threadgroup float* s2=arena+2u*{CD};  // bhat -> u -> v2
    threadgroup float* s3=arena+3u*{CD};  // ahat -> wmat
    threadgroup float* s4=arena+4u*{CD};  // scratch матмул-выходов
    threadgroup float* s5=arena+5u*{CD};  // oacc

    auto TG=[](threadgroup float*p,int cols,int rows){{return tensor<threadgroup float,dextents<int,2>,tensor_inline>(p,dextents<int,2>(cols,rows));}};
    auto DV=[](device float*p,int cols,int rows){{return tensor<device float,dextents<int,2>,tensor_inline>(p,dextents<int,2>(cols,rows));}};

    // --- фаза A: hat-векторы (s0..s3) ---
    for (uint e=lid; e<{CD}; e+=32u) {{ float g=GC[e], em=exp(-g), ep=exp(g);
        s0[e]=Q[e]*ep; s1[e]=K[e]*em; s2[e]=BE[e]*em; s3[e]=AL[e]*exp(g-GK[e]); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{C},{D},false,true);
       tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(s0,{D},{C}); auto R=TG(s1,{D},{C});
       auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(Am+0u*{CC},{C},{C}); c.store(o); }}
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{C},{D},false,true);
       tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(s0,{D},{C}); auto R=TG(s2,{D},{C});
       auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(Am+1u*{CC},{C},{C}); c.store(o); }}
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{C},{D},false,true);
       tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(s3,{D},{C}); auto R=TG(s2,{D},{C});
       auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(Am+2u*{CC},{C},{C}); c.store(o); }}
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{C},{D},false,true);
       tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(s3,{D},{C}); auto R=TG(s1,{D},{C});
       auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(Am+3u*{CC},{C},{C}); c.store(o); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e=lid; e<{CC}; e+=32u) {{ uint i=e/{C}, j=e%{C};
        if (j> i) {{ Am[0u*{CC}+e]=0; Am[1u*{CC}+e]=0; }}
        if (j>=i) {{ Am[2u*{CC}+e]=0; Am[3u*{CC}+e]=0; }} }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- фаза B ---  s1=v
    for (uint e=lid; e<{CD}; e+=32u) s1[e]=V[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // RHS_u=A_ak@v -> s2 ; trisolve -> u
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{C},false,false);
       tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(Am+3u*{CC},{C},{C}); auto R=TG(s1,{D},{C});
       auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(s2,{D},{C}); c.store(o); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=1;i<{C};i++) {{ for (uint dd=lid; dd<{D}; dd+=32u) {{
        float acc=s2[i*{D}+dd]; for(uint n=0;n<i;n++) acc+=Am[2u*{CC}+i*{C}+n]*s2[n*{D}+dd]; s2[i*{D}+dd]=acc; }}
        threadgroup_barrier(mem_flags::mem_threadgroup); }}
    // RHS_w=exp(gc-gk)*alpha -> s3 ; trisolve -> wmat
    for (uint e=lid; e<{CD}; e+=32u) s3[e]=exp(GC[e]-GK[e])*AL[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=1;i<{C};i++) {{ for (uint dd=lid; dd<{D}; dd+=32u) {{
        float acc=s3[i*{D}+dd]; for(uint n=0;n<i;n++) acc+=Am[2u*{CC}+i*{C}+n]*s3[n*{D}+dd]; s3[i*{D}+dd]=acc; }}
        threadgroup_barrier(mem_flags::mem_threadgroup); }}
    // v2 = u + wmat@S : wmat(s3)@S -> s4 ; s2 += s4
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{D},false,false);
       tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(s3,{D},{C}); auto R=DV(Sp,{D},{D});
       auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(s4,{D},{C}); c.store(o); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e=lid; e<{CD}; e+=32u) s2[e]+=s4[e];   // s2 = v2
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // o = A_qk@v(s1) + A_qb@v2(s2) + qhat(s0)@S  -> s5
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{C},false,false);
       tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(Am+0u*{CC},{C},{C}); auto R=TG(s1,{D},{C});
       auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(s4,{D},{C}); c.store(o); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e=lid; e<{CD}; e+=32u) s5[e]=s4[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{C},false,false);
       tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(Am+1u*{CC},{C},{C}); auto R=TG(s2,{D},{C});
       auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(s4,{D},{C}); c.store(o); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e=lid; e<{CD}; e+=32u) s5[e]+=s4[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{D},false,false);
       tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(s0,{D},{C}); auto R=DV(Sp,{D},{D});
       auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(s4,{D},{C}); c.store(o); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e=lid; e<{CD}; e+=32u) OO[e]=s5[e]+s4[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // t1 = (k·decay)^T @ v ;  decay[i,d]=exp(gc_last[d]-gc[i,d])
    for (uint e=lid; e<{CD}; e+=32u) {{ uint d=e%{D}; float gl=GC[({C}-1)*{D}+d]; s4[e]=K[e]*exp(gl-GC[e]); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({D},{D},{C},true,false);
       tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(s4,{D},{C}); auto R=TG(s1,{D},{C});
       auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=DV(T1,{D},{D}); c.store(o); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // t2 = (beta·decay)^T @ v2
    for (uint e=lid; e<{CD}; e+=32u) {{ uint d=e%{D}; float gl=GC[({C}-1)*{D}+d]; s4[e]=BE[e]*exp(gl-GC[e]); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({D},{D},{C},true,false);
       tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(s4,{D},{C}); auto R=TG(s2,{D},{C});
       auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=DV(T2,{D},{D}); c.store(o); }}
    """
    kern = mx.fast.metal_kernel(
        name=f"dplr_step_{C}_{D}",
        input_names=["q", "k", "v", "alpha", "beta", "gc", "gk", "S_in"],
        output_names=["out_o", "t1", "t2"],
        header=_HDR, source=src)
    _step_cache[key] = kern
    return kern


def chunk_step(q, k, v, alpha, beta, gc, gk, S):
    """Один чанк для всех BH. Входы [BH,C,D]; S [BH,D,D]. Возврат (o[BH,C,D], t1, t2)."""
    BH, C, D = q.shape
    kern = _step_kernel(C, D)
    o, t1, t2 = kern(inputs=[q, k, v, alpha, beta, gc, gk, S],
                     grid=(32, BH, 1), threadgroup=(32, 1, 1),
                     output_shapes=[(BH, C, D), (BH, D, D), (BH, D, D)],
                     output_dtypes=[mx.float32] * 3)
    return o, t1, t2


def dplr_forward_metal(r, w, k, v, a, b, chunk_size=16):
    """Полный forward через step-кернел. Входы [B,T,H,D], w — реальный decay. Возврат o [B,T,H,D]."""
    B, T, H, D = r.shape
    C = chunk_size
    assert T % C == 0
    N = T // C
    # → [BH, N, C, D]
    def pack(x):
        return mx.transpose(x, (0, 2, 1, 3)).reshape(B * H, N, C, D)
    rq, kk, vv, aa, bb = map(pack, (r, k, v, a, b))
    gk_full = mx.log(w)
    gkp = pack(gk_full)
    gc_full = mx.cumsum(gkp.reshape(B * H, N, C, D), axis=2)  # cumsum внутри чанка
    BH = B * H
    S = mx.zeros((BH, D, D))
    outs = []
    for n in range(N):
        gc_n = gc_full[:, n]
        o, t1, t2 = chunk_step(rq[:, n], kk[:, n], vv[:, n], aa[:, n], bb[:, n], gc_n, gkp[:, n], S)
        last = mx.exp(gc_n[:, -1, :])[:, :, None]          # [BH,D,1] decay по key-оси
        S = S * last + t1 + t2
        outs.append(o)
        mx.eval(S, o)
    o = mx.stack(outs, axis=1).reshape(BH, T, D).reshape(B, H, T, D)
    return mx.transpose(o, (0, 2, 1, 3))


# === S3.4a-ii: backward одиночного чанка (S=0), 2 кернела + хост-хвост ========
# STAGE-1: recompute fwd → dA_qk,dA_qb,dA_ab,dA_ak (masked) + dv. Арена 6*CD+Am=28KB.
_bwd1_cache = {}


def _bwd1_kernel(C, D):
    key = (C, D)
    if key in _bwd1_cache:
        return _bwd1_cache[key]
    CD, CC = C * D, C * C
    src = f"""
    uint lid = thread_index_in_threadgroup;
    threadgroup float arena[{6*CD}];
    threadgroup float Am[{4*CC}];   // qk(0) qb(1) ab(2) ak(3)
    threadgroup float* s0=arena+0u*{CD};
    threadgroup float* s1=arena+1u*{CD};
    threadgroup float* s2=arena+2u*{CD};
    threadgroup float* s3=arena+3u*{CD};
    threadgroup float* s4=arena+4u*{CD};
    threadgroup float* s5=arena+5u*{CD};
    auto TG=[](threadgroup float*p,int cols,int rows){{return tensor<threadgroup float,dextents<int,2>,tensor_inline>(p,dextents<int,2>(cols,rows));}};
    auto DV=[](device float*p,int cols,int rows){{return tensor<device float,dextents<int,2>,tensor_inline>(p,dextents<int,2>(cols,rows));}};

    // --- recompute hats (s0..s3) -> Am ---
    for (uint e=lid;e<{CD};e+=32u){{ float g=gc[e],em=exp(-g),ep=exp(g);
        s0[e]=q[e]*ep; s1[e]=k[e]*em; s2[e]=beta[e]*em; s3[e]=alpha[e]*exp(g-gk[e]); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #define MM_FT(LP,RP,OP) {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{C},{D},false,true); \
        tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(LP,{D},{C}); auto R=TG(RP,{D},{C}); \
        auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(OP,{C},{C}); c.store(o); }}
    MM_FT(s0,s1,Am+0u*{CC}) MM_FT(s0,s2,Am+1u*{CC}) MM_FT(s3,s2,Am+2u*{CC}) MM_FT(s3,s1,Am+3u*{CC})
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e=lid;e<{CC};e+=32u){{ uint i=e/{C},j=e%{C};
        if(j>i){{Am[0u*{CC}+e]=0;Am[1u*{CC}+e]=0;}} if(j>=i){{Am[2u*{CC}+e]=0;Am[3u*{CC}+e]=0;}} }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // v->s0 ; RHS_u=A_ak@v->s1 ; trisolve(A_ab) -> u in s1
    for (uint e=lid;e<{CD};e+=32u) s0[e]=v[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{C},false,false);
       tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(Am+3u*{CC},{C},{C}); auto R=TG(s0,{D},{C});
       auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(s1,{D},{C}); c.store(o); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=1;i<{C};i++){{ for(uint dd=lid;dd<{D};dd+=32u){{ float acc=s1[i*{D}+dd];
        for(uint n=0;n<i;n++) acc+=Am[2u*{CC}+i*{C}+n]*s1[n*{D}+dd]; s1[i*{D}+dd]=acc; }}
        threadgroup_barrier(mem_flags::mem_threadgroup); }}
    // do->s2
    for (uint e=lid;e<{CD};e+=32u) s2[e]=dout[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dv1 = A_qk^T@do -> s3 ; du = A_qb^T@do -> s4
    #define MM_TF(LP,RP,OP) {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{C},true,false); \
        tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(LP,{C},{C}); auto R=TG(RP,{D},{C}); \
        auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(OP,{D},{C}); c.store(o); }}
    MM_TF(Am+0u*{CC}, s2, s3)
    MM_TF(Am+1u*{CC}, s2, s4)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // dc = (I-A_ab)^-T @ du : back-subst na s4
    for (int i={C}-2;i>=0;i--){{ for(uint dd=lid;dd<{D};dd+=32u){{ float acc=s4[i*{D}+dd];
        for(uint n=i+1;n<{C};n++) acc+=Am[2u*{CC}+n*{C}+i]*s4[n*{D}+dd]; s4[i*{D}+dd]=acc; }}
        threadgroup_barrier(mem_flags::mem_threadgroup); }}
    // dv2 = A_ak^T@dc -> s5 ; dv = s3+s5 -> device
    MM_TF(Am+3u*{CC}, s4, s5)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e=lid;e<{CD};e+=32u) dv_out[e]=s3[e]+s5[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dA = (do|dc)@(v|u)^T, masked -> device. s3 как [C,C]-temp.
    #define MM_DA(LP,RP,DEVP,MASKLT) {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{C},{D},false,true); \
        tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(LP,{D},{C}); auto R=TG(RP,{D},{C}); \
        auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(s3,{C},{C}); c.store(o); }} \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for(uint e=lid;e<{CC};e+=32u){{ uint i=e/{C},j=e%{C}; bool keep = MASKLT ? (j<i) : (j<=i); DEVP[e]= keep? s3[e]:0.0f; }} \
        threadgroup_barrier(mem_flags::mem_threadgroup);
    MM_DA(s2,s0, dAqk, false)   // do@v^T  le
    MM_DA(s2,s1, dAqb, false)   // do@u^T  le
    MM_DA(s4,s1, dAab, true)    // dc@u^T  lt
    MM_DA(s4,s0, dAak, true)    // dc@v^T  lt
    """
    kern = mx.fast.metal_kernel(
        name=f"dplr_bwd1_{C}_{D}",
        input_names=["q", "k", "v", "alpha", "beta", "gc", "gk", "dout"],
        output_names=["dAqk", "dAqb", "dAab", "dAak", "dv_out"],
        header=_HDR, source=src)
    _bwd1_cache[key] = kern
    return kern


# STAGE-2: recompute hats; dA_* (device) → dqh,dkh,dbh,dah. Арена 6*CD=24KB.
_bwd2_cache = {}


def _bwd2_kernel(C, D):
    key = (C, D)
    if key in _bwd2_cache:
        return _bwd2_cache[key]
    CD, CC = C * D, C * C
    src = f"""
    uint lid = thread_index_in_threadgroup;
    threadgroup float arena[{6*CD}];
    threadgroup float* qh=arena+0u*{CD};
    threadgroup float* kh=arena+1u*{CD};
    threadgroup float* bh=arena+2u*{CD};
    threadgroup float* ah=arena+3u*{CD};
    threadgroup float* s4=arena+4u*{CD};
    threadgroup float* s5=arena+5u*{CD};
    auto TG=[](threadgroup float*p,int cols,int rows){{return tensor<threadgroup float,dextents<int,2>,tensor_inline>(p,dextents<int,2>(cols,rows));}};
    auto DC=[](device float*p,int cols,int rows){{return tensor<device float,dextents<int,2>,tensor_inline>(p,dextents<int,2>(cols,rows));}};

    for (uint e=lid;e<{CD};e+=32u){{ float g=gc[e],em=exp(-g),ep=exp(g);
        qh[e]=q[e]*ep; kh[e]=k[e]*em; bh[e]=beta[e]*em; ah[e]=alpha[e]*exp(g-gk[e]); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dA_* в device; матмулы dev(left)×tg(right). F,F: A@x ; T,F: A^T@x.
    #define DAFF(DAP,RP,OUTP) {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{C},false,false); \
        tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=DC((device float*)DAP,{C},{C}); auto R=TG(RP,{D},{C}); \
        auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(OUTP,{D},{C}); c.store(o); }}
    #define DATF(DAP,RP,OUTP) {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{C},true,false); \
        tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=DC((device float*)DAP,{C},{C}); auto R=TG(RP,{D},{C}); \
        auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(OUTP,{D},{C}); c.store(o); }}

    // dqh = dA_qk@kh + dA_qb@bh
    DAFF(dAqk,kh,s4) DAFF(dAqb,bh,s5)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint e=lid;e<{CD};e+=32u) dqh[e]=s4[e]+s5[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // dkh = dA_qk^T@qh + dA_ak^T@ah
    DATF(dAqk,qh,s4) DATF(dAak,ah,s5)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint e=lid;e<{CD};e+=32u) dkh[e]=s4[e]+s5[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // dbh = dA_qb^T@qh + dA_ab^T@ah
    DATF(dAqb,qh,s4) DATF(dAab,ah,s5)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint e=lid;e<{CD};e+=32u) dbh[e]=s4[e]+s5[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // dah = dA_ab@bh + dA_ak@kh
    DAFF(dAab,bh,s4) DAFF(dAak,kh,s5)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint e=lid;e<{CD};e+=32u) dah[e]=s4[e]+s5[e];
    """
    kern = mx.fast.metal_kernel(
        name=f"dplr_bwd2_{C}_{D}",
        input_names=["q", "k", "alpha", "beta", "gc", "gk", "dAqk", "dAqb", "dAab", "dAak"],
        output_names=["dqh", "dkh", "dbh", "dah"],
        header=_HDR, source=src)
    _bwd2_cache[key] = kern
    return kern


def chunk_bwd_s0_metal(r, w, k, v, a, b, do):
    """Backward одного чанка (S=0) на Metal: stage1+stage2+хост-хвост. Возврат (dr,dw,dk,dv,da,db)."""
    C, D = r.shape
    gk = mx.log(w); gc = mx.cumsum(gk, axis=0); q = r
    k1 = _bwd1_kernel(C, D)
    dAqk, dAqb, dAab, dAak, dv = k1(
        inputs=[q, k, v, a, b, gc, gk, do],
        grid=(32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(C, C)] * 4 + [(C, D)], output_dtypes=[mx.float32] * 5)
    k2 = _bwd2_kernel(C, D)
    dqh, dkh, dbh, dah = k2(
        inputs=[q, k, a, b, gc, gk, dAqk, dAqb, dAab, dAak],
        grid=(32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(C, D)] * 4, output_dtypes=[mx.float32] * 4)
    # хост-хвост (дёшево)
    Eg = mx.exp(gc); Eng = mx.exp(-gc); Eag = mx.exp(gc - gk)
    qh = q * Eg; kh = k * Eng; bh = b * Eng; ah = a * Eag
    dq = dqh * Eg; dk = dkh * Eng; db = dbh * Eng; da = dah * Eag
    dgc = dqh * qh - dkh * kh - dbh * bh + dah * ah
    dgk = -dah * ah + mx.cumsum(dgc, axis=0, reverse=True)
    dw = dgk / w; dr = dq
    return dr, dw, dk, dv, da, db


# === S3.4b-ii: межчанковый backward (carry dS) ================================
# KB — grad-ядро одного чанка. Вход: fwd-интермедиаты (Am,u,wmat,v2) + S_in,dS,do.
# Выход (device): dv,dkdec,dbdec,dqh_s,dah_extra[C,D]; dAqk..dAak[C,C]; dSin_a,dSin_b[D,D].
# Почти всё — device-операнды; threadgroup только dc,dcw,dv2,scratch + Tcc (~17KB).
# Формы матмулов — все probe-проверены (step/bwd1 + probe_bwd_forms FORM1/2).
_kb_cache = {}


def _kb_kernel(C, D):
    key = (C, D)
    if key in _kb_cache:
        return _kb_cache[key]
    CD, CC, DD = C * D, C * C, D * D
    src = f"""
    uint bh = thread_position_in_grid.y;
    uint lid = thread_index_in_threadgroup;
    const device float* Q  = q    + bh*{CD};
    const device float* K  = k    + bh*{CD};
    const device float* V  = v    + bh*{CD};
    const device float* BE = beta + bh*{CD};
    const device float* GC = gc   + bh*{CD};
    const device float* DO = dout + bh*{CD};
    const device float* U  = u    + bh*{CD};
    const device float* WM = wmat + bh*{CD};
    const device float* V2 = v2   + bh*{CD};
    const device float* SI = S_in + bh*{DD};
    const device float* DSc= dS   + bh*{DD};
    const device float* AQK= Aqk  + bh*{CC};
    const device float* AQB= Aqb  + bh*{CC};
    const device float* AAB= Aab  + bh*{CC};
    const device float* AAK= Aak  + bh*{CC};
    device float* o_dv   = dv      + bh*{CD};
    device float* o_dkd  = dkdec   + bh*{CD};
    device float* o_dbd  = dbdec   + bh*{CD};
    device float* o_dqhs = dqh_s   + bh*{CD};
    device float* o_dahe = dah_extra+ bh*{CD};
    device float* o_AQK  = dAqk    + bh*{CC};
    device float* o_AQB  = dAqb    + bh*{CC};
    device float* o_AAB  = dAab    + bh*{CC};
    device float* o_AAK  = dAak    + bh*{CC};
    device float* o_Sa   = dSin_a  + bh*{DD};
    device float* o_Sb   = dSin_b  + bh*{DD};

    threadgroup float s0[{CD}], s1[{CD}], s2[{CD}], s3[{CD}];  // dv2 ; dc ; dcw ; tmp
    threadgroup float Tcc[{CC}];

    auto TG=[](threadgroup float*p,int c,int r){{return tensor<threadgroup float,dextents<int,2>,tensor_inline>(p,dextents<int,2>(c,r));}};
    auto DV=[](const device float*p,int c,int r){{return tensor<device float,dextents<int,2>,tensor_inline>((device float*)p,dextents<int,2>(c,r));}};

    #define RUN(L,R,O) {{ auto lft=(L); auto rgt=(R); auto dst=(O); auto cc=op.get_destination_cooperative_tensor<decltype(lft),decltype(rgt),float>(); op.run(lft,rgt,cc); cc.store(dst); }}
    #define FF_CDD(L,R,O) {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{D},false,false); tensor_ops::matmul2d<d,execution_simdgroup> op; RUN(L,R,O) }}
    #define FT_CDD(L,R,O) {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{D},false,true ); tensor_ops::matmul2d<d,execution_simdgroup> op; RUN(L,R,O) }}
    #define TF_CDC(L,R,O) {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{C},true ,false); tensor_ops::matmul2d<d,execution_simdgroup> op; RUN(L,R,O) }}
    #define TF_DDC(L,R,O) {{ constexpr auto d=tensor_ops::matmul2d_descriptor({D},{D},{C},true ,false); tensor_ops::matmul2d<d,execution_simdgroup> op; RUN(L,R,O) }}
    #define FT_CCD(L,R,O) {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{C},{D},false,true ); tensor_ops::matmul2d<d,execution_simdgroup> op; RUN(L,R,O) }}
    #define BAR threadgroup_barrier(mem_flags::mem_threadgroup);
    #define SOLVE_T(X) for(int i={C}-2;i>=0;i--){{ for(uint dd=lid;dd<{D};dd+=32u){{ float acc=X[i*{D}+dd]; \
        for(uint n=i+1;n<{C};n++) acc+=AAB[n*{C}+i]*X[n*{D}+dd]; X[i*{D}+dd]=acc; }} BAR }}

    // dv2 = bdec@dS + A_qb^T@do
    for(uint e=lid;e<{CD};e+=32u){{ uint dd=e%{D}; s3[e]=BE[e]*exp(GC[({C}-1)*{D}+dd]-GC[e]); }} BAR  // bdec
    FF_CDD(TG(s3,{D},{C}), DV(DSc,{D},{D}), TG(s0,{D},{C}))  // -> s0=dv2
    BAR
    TF_CDC(DV(AQB,{C},{C}), DV(DO,{D},{C}), TG(s3,{D},{C}))  // A_qb^T@do -> s3
    BAR
    for(uint e=lid;e<{CD};e+=32u) s0[e]+=s3[e]; BAR           // s0 = dv2

    // dc = (I-A_ab)^-T @ dv2
    for(uint e=lid;e<{CD};e+=32u) s1[e]=s0[e]; BAR
    SOLVE_T(s1)                                               // s1 = dc

    // dwmat = dv2@S_in^T -> s3 ; dcw = solve_T(dwmat)
    FT_CDD(TG(s0,{D},{C}), DV(SI,{D},{D}), TG(s3,{D},{C}))    // dv2@S_in^T -> s3
    BAR
    for(uint e=lid;e<{CD};e+=32u) s2[e]=s3[e]; BAR
    SOLVE_T(s2)                                               // s2 = dcw
    for(uint e=lid;e<{CD};e+=32u) o_dahe[e]=s2[e]; BAR        // dah_extra = dcw

    // dSin_a = qh^T@do ; dSin_b = wmat^T@dv2  (dv2=s0 ещё жив)
    for(uint e=lid;e<{CD};e+=32u) s3[e]=Q[e]*exp(GC[e]); BAR  // qh
    TF_DDC(TG(s3,{D},{C}), DV(DO,{D},{C}), DV(o_Sa,{D},{D}))  // qh^T@do -> [D,D]
    BAR
    TF_DDC(DV(WM,{D},{C}), TG(s0,{D},{C}), DV(o_Sb,{D},{D}))  // wmat^T@dv2 -> [D,D]
    BAR

    // dv = kdec@dS + A_qk^T@do + A_ak^T@dc   (s0 свободен -> аккумулятор)
    for(uint e=lid;e<{CD};e+=32u){{ uint dd=e%{D}; s3[e]=K[e]*exp(GC[({C}-1)*{D}+dd]-GC[e]); }} BAR  // kdec
    FF_CDD(TG(s3,{D},{C}), DV(DSc,{D},{D}), TG(s0,{D},{C}))   // kdec@dS -> s0
    BAR
    TF_CDC(DV(AQK,{C},{C}), DV(DO,{D},{C}), TG(s3,{D},{C}))   // A_qk^T@do -> s3
    BAR
    for(uint e=lid;e<{CD};e+=32u) s0[e]+=s3[e]; BAR
    TF_CDC(DV(AAK,{C},{C}), TG(s1,{D},{C}), TG(s3,{D},{C}))   // A_ak^T@dc -> s3
    BAR
    for(uint e=lid;e<{CD};e+=32u) o_dv[e]=s0[e]+s3[e]; BAR

    // dkdec=v@dS^T ; dbdec=v2@dS^T ; dqh_s=do@S_in^T  (полностью device)
    FT_CDD(DV(V ,{D},{C}), DV(DSc,{D},{D}), DV(o_dkd ,{D},{C}))
    FT_CDD(DV(V2,{D},{C}), DV(DSc,{D},{D}), DV(o_dbd ,{D},{C}))
    FT_CDD(DV(DO,{D},{C}), DV(SI ,{D},{D}), DV(o_dqhs,{D},{C}))
    BAR

    // dA's (F,T -> [C,C], mask) -> device
    #define MASK(DST,LT) for(uint e=lid;e<{CC};e+=32u){{ uint i=e/{C},j=e%{C}; bool keep=(LT)?(j<i):(j<=i); DST[e]=keep?Tcc[e]:0.0f; }} BAR
    FT_CCD(DV(DO,{D},{C}), DV(V ,{D},{C}), TG(Tcc,{C},{C})) BAR MASK(o_AQK,false)  // do@v^T  le
    FT_CCD(DV(DO,{D},{C}), DV(V2,{D},{C}), TG(Tcc,{C},{C})) BAR MASK(o_AQB,false)  // do@v2^T le
    FT_CCD(TG(s1,{D},{C}), DV(V ,{D},{C}), TG(Tcc,{C},{C})) BAR MASK(o_AAK,true)   // dc@v^T  lt
    FT_CCD(TG(s1,{D},{C}), DV(U ,{D},{C}), TG(Tcc,{C},{C})) BAR                    // dc@u^T
    for(uint e=lid;e<{CC};e+=32u){{ uint i=e/{C},j=e%{C}; o_AAB[e]=(j<i)?Tcc[e]:0.0f; }} BAR
    FT_CCD(TG(s2,{D},{C}), DV(WM,{D},{C}), TG(Tcc,{C},{C})) BAR                    // dcw@wmat^T
    for(uint e=lid;e<{CC};e+=32u){{ uint i=e/{C},j=e%{C}; if(j<i) o_AAB[e]+=Tcc[e]; }}
    """
    kern = mx.fast.metal_kernel(
        name=f"dplr_kb_{C}_{D}",
        input_names=["q", "k", "v", "beta", "gc", "dout", "u", "wmat", "v2",
                     "S_in", "dS", "Aqk", "Aqb", "Aab", "Aak"],
        output_names=["dv", "dkdec", "dbdec", "dqh_s", "dah_extra",
                      "dAqk", "dAqb", "dAab", "dAak", "dSin_a", "dSin_b"],
        header=_HDR, source=src)
    _kb_cache[key] = kern
    return kern


def _fwd_intermediates_mlx(r, w, k, v, a, b, S_in, C):
    """MLX-форвард-интермедиаты для KB-микрогарда: masked Am, u, wmat, v2."""
    ii = mx.arange(C)[:, None]; jj = mx.arange(C)[None, :]
    le = (jj <= ii); lt = (jj < ii)
    gk = mx.log(w); gc = mx.cumsum(gk, axis=0)
    Eg = mx.exp(gc); Eng = mx.exp(-gc); Eag = mx.exp(gc - gk)
    qh = r * Eg; kh = k * Eng; bh = b * Eng; ah = a * Eag
    A_qk = mx.where(le, qh @ kh.T, 0.0); A_qb = mx.where(le, qh @ bh.T, 0.0)
    A_ab = mx.where(lt, ah @ bh.T, 0.0); A_ak = mx.where(lt, ah @ kh.T, 0.0)
    A_inv = mx.eye(C); P = mx.eye(C)
    for _ in range(C - 1):
        P = P @ A_ab; A_inv = A_inv + P
    u = A_inv @ (A_ak @ v); wmat = A_inv @ ah; v2 = u + wmat @ S_in
    return A_qk, A_qb, A_ab, A_ak, u, wmat, v2


def chunk_bwd_one_metal(r, w, k, v, a, b, S_in, dS, do, fwd="mlx"):
    """Backward одного чанка с состоянием на Metal (KB + _bwd2 + хост). Single head.
    fwd='mlx': fwd-интермедиаты из MLX (микрогард KB-формы)."""
    C, D = r.shape
    gk = mx.log(w); gc = mx.cumsum(gk, axis=0); q = r
    Aqk, Aqb, Aab, Aak, u, wmat, v2 = _fwd_intermediates_mlx(r, w, k, v, a, b, S_in, C)

    kb = _kb_kernel(C, D)
    outs = kb(inputs=[q, k, v, b, gc, do, u, wmat, v2, S_in, dS, Aqk, Aqb, Aab, Aak],
              grid=(32, 1, 1), threadgroup=(32, 1, 1),
              output_shapes=[(C, D)] * 5 + [(C, C)] * 4 + [(D, D)] * 2,
              output_dtypes=[mx.float32] * 11)
    dv, dkdec, dbdec, dqh_s, dah_extra, dAqk, dAqb, dAab, dAak, dSin_a, dSin_b = outs

    k2 = _bwd2_kernel(C, D)
    dqh_h, dkh, dbh, dah_h = k2(
        inputs=[q, k, a, b, gc, gk, dAqk, dAqb, dAab, dAak],
        grid=(32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(C, D)] * 4, output_dtypes=[mx.float32] * 4)

    # host tail (как chunk_bwd_one)
    Eg = mx.exp(gc); Eng = mx.exp(-gc); Eag = mx.exp(gc - gk)
    qh = q * Eg; kh = k * Eng; bh = b * Eng; ah = a * Eag
    gc_last = gc[-1]; last = mx.exp(gc_last); dec = mx.exp(gc_last[None, :] - gc)
    dqh = dqh_h + dqh_s
    dah = dah_h + dah_extra
    dq = dqh * Eg; dk = dkh * Eng; db = dbh * Eng; da = dah * Eag
    dgc = dqh * qh - dkh * kh - dbh * bh + dah * ah
    dgk = -dah * ah
    dk = dk + dkdec * dec; db = db + dbdec * dec
    ddec = dkdec * k + dbdec * b
    dgc = dgc - ddec * dec
    dgc_last = last * (dS * S_in).sum(axis=1) + (ddec * dec).sum(axis=0)
    dgc = dgc.at[-1].add(dgc_last)
    dgk = dgk + mx.cumsum(dgc, axis=0, reverse=True)
    dw = dgk / w; dr = dq
    dS_in = last[:, None] * dS + dSin_a + dSin_b
    return dr, dw, dk, dv, da, db, dS_in


def dplr_bwd_metal(r, w, k, v, a, b, do, C):
    """S3.4b-ii: полный межчанковый backward на Metal (single head).
    Forward-кэш граничных S_in из MLX (KA — поздний перф-своп); обратный цикл
    n=N-1..0 с carry dS через chunk_bwd_one_metal. Возврат dr,dw,dk,dv,da,db."""
    import sys, os
    sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal/experiments"))
    from dplr_bwd_chunk_mlx import chunk_fwd_seq
    T, D = r.shape
    N = T // C
    _, cache = chunk_fwd_seq(r, w, k, v, a, b, C)   # S_in[n] граничные состояния
    sl = [slice(n * C, (n + 1) * C) for n in range(N)]
    grads = [None] * N
    dS = mx.zeros((D, D))
    for n in range(N - 1, -1, -1):
        s = sl[n]
        g = chunk_bwd_one_metal(r[s], w[s], k[s], v[s], a[s], b[s],
                                cache[n]["S_in"], dS, do[s])
        grads[n] = g[:6]
        dS = g[6]
    dr = mx.concatenate([grads[n][0] for n in range(N)], axis=0)
    dw = mx.concatenate([grads[n][1] for n in range(N)], axis=0)
    dk = mx.concatenate([grads[n][2] for n in range(N)], axis=0)
    dv = mx.concatenate([grads[n][3] for n in range(N)], axis=0)
    da = mx.concatenate([grads[n][4] for n in range(N)], axis=0)
    db = mx.concatenate([grads[n][5] for n in range(N)], axis=0)
    return dr, dw, dk, dv, da, db


# === S3.4b-ii BH-батч ========================================================
_bwd2bh_cache = {}


def _bwd2_bh_kernel(C, D):
    key = (C, D)
    if key in _bwd2bh_cache:
        return _bwd2bh_cache[key]
    CD, CC = C * D, C * C
    src = f"""
    uint bh2 = thread_position_in_grid.y;
    uint lid = thread_index_in_threadgroup;
    const device float* Q=q+bh2*{CD}; const device float* K=k+bh2*{CD};
    const device float* AL=alpha+bh2*{CD}; const device float* BE=beta+bh2*{CD};
    const device float* GC=gc+bh2*{CD}; const device float* GK=gk+bh2*{CD};
    const device float* DQK=dAqk+bh2*{CC}; const device float* DQB=dAqb+bh2*{CC};
    const device float* DAB=dAab+bh2*{CC}; const device float* DAK=dAak+bh2*{CC};
    device float* ODqh=dqh+bh2*{CD}; device float* ODkh=dkh+bh2*{CD};
    device float* ODbh=dbh+bh2*{CD}; device float* ODah=dah+bh2*{CD};
    threadgroup float arena[{6*CD}];
    threadgroup float* qh=arena+0u*{CD}; threadgroup float* kh=arena+1u*{CD};
    threadgroup float* bh=arena+2u*{CD}; threadgroup float* ah=arena+3u*{CD};
    threadgroup float* s4=arena+4u*{CD}; threadgroup float* s5=arena+5u*{CD};
    auto TG=[](threadgroup float*p,int c,int r){{return tensor<threadgroup float,dextents<int,2>,tensor_inline>(p,dextents<int,2>(c,r));}};
    auto DC=[](device float*p,int c,int r){{return tensor<device float,dextents<int,2>,tensor_inline>(p,dextents<int,2>(c,r));}};
    for (uint e=lid;e<{CD};e+=32u){{ float g=GC[e],em=exp(-g),ep=exp(g);
        qh[e]=Q[e]*ep; kh[e]=K[e]*em; bh[e]=BE[e]*em; ah[e]=AL[e]*exp(g-GK[e]); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #define DAFF(DAP,RP,OUTP) {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{C},false,false); \
        tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=DC((device float*)DAP,{C},{C}); auto R=TG(RP,{D},{C}); \
        auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(OUTP,{D},{C}); c.store(o); }}
    #define DATF(DAP,RP,OUTP) {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{C},true,false); \
        tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=DC((device float*)DAP,{C},{C}); auto R=TG(RP,{D},{C}); \
        auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(OUTP,{D},{C}); c.store(o); }}
    DAFF(DQK,kh,s4) DAFF(DQB,bh,s5) threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint e=lid;e<{CD};e+=32u) ODqh[e]=s4[e]+s5[e]; threadgroup_barrier(mem_flags::mem_threadgroup);
    DATF(DQK,qh,s4) DATF(DAK,ah,s5) threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint e=lid;e<{CD};e+=32u) ODkh[e]=s4[e]+s5[e]; threadgroup_barrier(mem_flags::mem_threadgroup);
    DATF(DQB,qh,s4) DATF(DAB,ah,s5) threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint e=lid;e<{CD};e+=32u) ODbh[e]=s4[e]+s5[e]; threadgroup_barrier(mem_flags::mem_threadgroup);
    DAFF(DAB,bh,s4) DAFF(DAK,kh,s5) threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint e=lid;e<{CD};e+=32u) ODah[e]=s4[e]+s5[e];
    """
    kern = mx.fast.metal_kernel(
        name=f"dplr_bwd2bh_{C}_{D}",
        input_names=["q", "k", "alpha", "beta", "gc", "gk", "dAqk", "dAqb", "dAab", "dAak"],
        output_names=["dqh", "dkh", "dbh", "dah"], header=_HDR, source=src)
    _bwd2bh_cache[key] = kern
    return kern


def _fwd_intermediates_mlx_bh(R, W, K, V, A, B, S_in, C):
    """Батч-версия: входы [BH,C,D], S_in [BH,D,D]."""
    ii = mx.arange(C)[:, None]; jj = mx.arange(C)[None, :]
    le = (jj <= ii)[None]; lt = (jj < ii)[None]
    gk = mx.log(W); gc = mx.cumsum(gk, axis=1)
    Eg = mx.exp(gc); Eng = mx.exp(-gc); Eag = mx.exp(gc - gk)
    qh = R * Eg; kh = K * Eng; bh = B * Eng; ah = A * Eag
    T = lambda x: mx.swapaxes(x, 1, 2)
    A_qk = mx.where(le, qh @ T(kh), 0.0); A_qb = mx.where(le, qh @ T(bh), 0.0)
    A_ab = mx.where(lt, ah @ T(bh), 0.0); A_ak = mx.where(lt, ah @ T(kh), 0.0)
    BH = R.shape[0]
    eye = mx.broadcast_to(mx.eye(C)[None], (BH, C, C))
    A_inv = eye; P = eye
    for _ in range(C - 1):
        P = P @ A_ab; A_inv = A_inv + P
    u = A_inv @ (A_ak @ V); wmat = A_inv @ ah; v2 = u + wmat @ S_in
    return A_qk, A_qb, A_ab, A_ak, u, wmat, v2


def chunk_bwd_one_metal_bh(R, W, K, V, A, B, S_in, dS, do):
    """BH-батч backward одного чанка. Входы [BH,C,D], S_in/dS [BH,D,D]."""
    BH, C, D = R.shape
    gk = mx.log(W); gc = mx.cumsum(gk, axis=1)
    Aqk, Aqb, Aab, Aak, u, wmat, v2 = _fwd_intermediates_mlx_bh(R, W, K, V, A, B, S_in, C)
    kb = _kb_kernel(C, D)
    outs = kb(inputs=[R, K, V, B, gc, do, u, wmat, v2, S_in, dS, Aqk, Aqb, Aab, Aak],
              grid=(32, BH, 1), threadgroup=(32, 1, 1),
              output_shapes=[(BH, C, D)] * 5 + [(BH, C, C)] * 4 + [(BH, D, D)] * 2,
              output_dtypes=[mx.float32] * 11)
    dv, dkdec, dbdec, dqh_s, dah_extra, dAqk, dAqb, dAab, dAak, dSin_a, dSin_b = outs
    k2 = _bwd2_bh_kernel(C, D)
    dqh_h, dkh, dbh, dah_h = k2(
        inputs=[R, K, A, B, gc, gk, dAqk, dAqb, dAab, dAak],
        grid=(32, BH, 1), threadgroup=(32, 1, 1),
        output_shapes=[(BH, C, D)] * 4, output_dtypes=[mx.float32] * 4)
    Eg = mx.exp(gc); Eng = mx.exp(-gc); Eag = mx.exp(gc - gk)
    qh = R * Eg; kh = K * Eng; bh = B * Eng; ah = A * Eag
    gc_last = gc[:, -1]; last = mx.exp(gc_last)
    dec = mx.exp(gc_last[:, None, :] - gc)
    dqh = dqh_h + dqh_s; dah = dah_h + dah_extra
    dq = dqh * Eg; dk = dkh * Eng; db = dbh * Eng; da = dah * Eag
    dgc = dqh * qh - dkh * kh - dbh * bh + dah * ah
    dgk = -dah * ah
    dk = dk + dkdec * dec; db = db + dbdec * dec
    ddec = dkdec * K + dbdec * B
    dgc = dgc - ddec * dec
    dgc_last = last * (dS * S_in).sum(axis=2) + (ddec * dec).sum(axis=1)
    idx = mx.arange(C)[None, :, None]
    dgc = dgc + mx.where(idx == C - 1, dgc_last[:, None, :], 0.0)
    dgk = dgk + mx.cumsum(dgc, axis=1, reverse=True)
    dw = dgk / W; dr = dq
    dS_in = last[:, :, None] * dS + dSin_a + dSin_b
    return dr, dw, dk, dv, da, db, dS_in


def dplr_bwd_metal_bh(R, W, K, V, A, B, do, C):
    """S3.4b-ii BH-батч полный backward. Входы [BH,T,D]."""
    import sys, os
    sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal/experiments"))
    from dplr_bwd_chunk_mlx import chunk_fwd_seq
    BH, Tt, D = R.shape
    N = Tt // C
    # граничные состояния поголовно (MLX; KA — поздний своп)
    S_ins = []
    for h in range(BH):
        _, cache = chunk_fwd_seq(R[h], W[h], K[h], V[h], A[h], B[h], C)
        S_ins.append([cache[n]["S_in"] for n in range(N)])
    grads = [None] * N
    dS = mx.zeros((BH, D, D))
    for n in range(N - 1, -1, -1):
        s = slice(n * C, (n + 1) * C)
        Sn = mx.stack([S_ins[h][n] for h in range(BH)], axis=0)
        g = chunk_bwd_one_metal_bh(R[:, s], W[:, s], K[:, s], V[:, s], A[:, s], B[:, s],
                                   Sn, dS, do[:, s])
        grads[n] = g[:6]; dS = g[6]
    out = [mx.concatenate([grads[n][i] for n in range(N)], axis=1) for i in range(6)]
    return tuple(out)


# === S3.4b-iii: SAVE-форвард (rwkv-metal) — step + экспорт Am,u,wmat,v2 =======
# Копия проверенного _step_kernel (S3.3c) + сторы интермедиатов. Бюджет threadgroup
# не меняется (выходы device). Backward читает сохранённое — без MLX-рекомпьюта.
_step_save_cache = {}


def _step_save_kernel(C, D):
    key = (C, D)
    if key in _step_save_cache:
        return _step_save_cache[key]
    CD, CC, DD = C * D, C * C, D * D
    src = f"""
    uint bh = thread_position_in_grid.y;
    uint lid = thread_index_in_threadgroup;
    const device float* Q = q     + bh*{CD};
    const device float* K = k     + bh*{CD};
    const device float* V = v     + bh*{CD};
    const device float* AL= alpha + bh*{CD};
    const device float* BE= beta  + bh*{CD};
    const device float* GC= gc    + bh*{CD};
    const device float* GK= gk    + bh*{CD};
    device float* Sp = (device float*)(S_in + bh*{DD});
    device float* OO = out_o + bh*{CD};
    device float* T1 = t1    + bh*{DD};
    device float* T2 = t2    + bh*{DD};
    device float* AMo= Am_out  + bh*{4*CC};
    device float* Uo = u_out   + bh*{CD};
    device float* WMo= wmat_out+ bh*{CD};
    device float* V2o= v2_out  + bh*{CD};

    threadgroup float arena[{6*CD}];
    threadgroup float Am[{4*CC}];
    threadgroup float* s0=arena+0u*{CD};
    threadgroup float* s1=arena+1u*{CD};
    threadgroup float* s2=arena+2u*{CD};
    threadgroup float* s3=arena+3u*{CD};
    threadgroup float* s4=arena+4u*{CD};
    threadgroup float* s5=arena+5u*{CD};
    auto TG=[](threadgroup float*p,int cols,int rows){{return tensor<threadgroup float,dextents<int,2>,tensor_inline>(p,dextents<int,2>(cols,rows));}};
    auto DV=[](device float*p,int cols,int rows){{return tensor<device float,dextents<int,2>,tensor_inline>(p,dextents<int,2>(cols,rows));}};

    for (uint e=lid; e<{CD}; e+=32u) {{ float g=GC[e], em=exp(-g), ep=exp(g);
        s0[e]=Q[e]*ep; s1[e]=K[e]*em; s2[e]=BE[e]*em; s3[e]=AL[e]*exp(g-GK[e]); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{C},{D},false,true); tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(s0,{D},{C}); auto R=TG(s1,{D},{C}); auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(Am+0u*{CC},{C},{C}); c.store(o); }}
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{C},{D},false,true); tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(s0,{D},{C}); auto R=TG(s2,{D},{C}); auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(Am+1u*{CC},{C},{C}); c.store(o); }}
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{C},{D},false,true); tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(s3,{D},{C}); auto R=TG(s2,{D},{C}); auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(Am+2u*{CC},{C},{C}); c.store(o); }}
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{C},{D},false,true); tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(s3,{D},{C}); auto R=TG(s1,{D},{C}); auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(Am+3u*{CC},{C},{C}); c.store(o); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e=lid; e<{CC}; e+=32u) {{ uint i=e/{C}, j=e%{C};
        if (j> i) {{ Am[0u*{CC}+e]=0; Am[1u*{CC}+e]=0; }}
        if (j>=i) {{ Am[2u*{CC}+e]=0; Am[3u*{CC}+e]=0; }} }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e=lid; e<{4*CC}; e+=32u) AMo[e]=Am[e];        // СОХР Am

    for (uint e=lid; e<{CD}; e+=32u) s1[e]=V[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{C},false,false); tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(Am+3u*{CC},{C},{C}); auto R=TG(s1,{D},{C}); auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(s2,{D},{C}); c.store(o); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=1;i<{C};i++) {{ for (uint dd=lid; dd<{D}; dd+=32u) {{
        float acc=s2[i*{D}+dd]; for(uint n=0;n<i;n++) acc+=Am[2u*{CC}+i*{C}+n]*s2[n*{D}+dd]; s2[i*{D}+dd]=acc; }}
        threadgroup_barrier(mem_flags::mem_threadgroup); }}
    for (uint e=lid; e<{CD}; e+=32u) Uo[e]=s2[e];           // СОХР u (до v2)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e=lid; e<{CD}; e+=32u) s3[e]=exp(GC[e]-GK[e])*AL[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=1;i<{C};i++) {{ for (uint dd=lid; dd<{D}; dd+=32u) {{
        float acc=s3[i*{D}+dd]; for(uint n=0;n<i;n++) acc+=Am[2u*{CC}+i*{C}+n]*s3[n*{D}+dd]; s3[i*{D}+dd]=acc; }}
        threadgroup_barrier(mem_flags::mem_threadgroup); }}
    for (uint e=lid; e<{CD}; e+=32u) WMo[e]=s3[e];          // СОХР wmat
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{D},false,false); tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(s3,{D},{C}); auto R=DV(Sp,{D},{D}); auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(s4,{D},{C}); c.store(o); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e=lid; e<{CD}; e+=32u) s2[e]+=s4[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e=lid; e<{CD}; e+=32u) V2o[e]=s2[e];          // СОХР v2
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{C},false,false); tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(Am+0u*{CC},{C},{C}); auto R=TG(s1,{D},{C}); auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(s4,{D},{C}); c.store(o); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e=lid; e<{CD}; e+=32u) s5[e]=s4[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{C},false,false); tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(Am+1u*{CC},{C},{C}); auto R=TG(s2,{D},{C}); auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(s4,{D},{C}); c.store(o); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e=lid; e<{CD}; e+=32u) s5[e]+=s4[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{D},false,false); tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(s0,{D},{C}); auto R=DV(Sp,{D},{D}); auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=TG(s4,{D},{C}); c.store(o); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e=lid; e<{CD}; e+=32u) OO[e]=s5[e]+s4[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e=lid; e<{CD}; e+=32u) {{ uint d=e%{D}; float gl=GC[({C}-1)*{D}+d]; s4[e]=K[e]*exp(gl-GC[e]); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({D},{D},{C},true,false); tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(s4,{D},{C}); auto R=TG(s1,{D},{C}); auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=DV(T1,{D},{D}); c.store(o); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e=lid; e<{CD}; e+=32u) {{ uint d=e%{D}; float gl=GC[({C}-1)*{D}+d]; s4[e]=BE[e]*exp(gl-GC[e]); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({D},{D},{C},true,false); tensor_ops::matmul2d<d,execution_simdgroup> op; auto L=TG(s4,{D},{C}); auto R=TG(s2,{D},{C}); auto c=op.get_destination_cooperative_tensor<decltype(L),decltype(R),float>(); op.run(L,R,c); auto o=DV(T2,{D},{D}); c.store(o); }}
    """
    kern = mx.fast.metal_kernel(
        name=f"dplr_step_save_{C}_{D}",
        input_names=["q", "k", "v", "alpha", "beta", "gc", "gk", "S_in"],
        output_names=["out_o", "t1", "t2", "Am_out", "u_out", "wmat_out", "v2_out"],
        header=_HDR, source=src)
    _step_save_cache[key] = kern
    return kern


def dplr_forward_metal_save(R, W, K, V, A, B, C):
    """SAVE-форвард (BH-вход [BH,T,D]). Возврат:
      o [BH,T,D], и per-chunk кэши (списки длины N, элементы [BH,...]):
      S_ins [BH,D,D], Am [BH,4,C,C], u/wmat/v2 [BH,C,D]."""
    BH, T, D = R.shape
    N = T // C
    gk = mx.log(W)
    gc = mx.cumsum(gk.reshape(BH, N, C, D), axis=2)
    def ch(x, n): return x[:, n * C:(n + 1) * C]
    kern = _step_save_kernel(C, D)
    S = mx.zeros((BH, D, D))
    outs = []; S_ins = []; Ams = []; us = []; wms = []; v2s = []
    for n in range(N):
        gc_n = gc[:, n]; gk_n = gk.reshape(BH, N, C, D)[:, n]
        S_ins.append(S)
        o, t1, t2, Am, u, wm, v2 = kern(
            inputs=[ch(R, n), ch(K, n), ch(V, n), ch(A, n), ch(B, n), gc_n, gk_n, S],
            grid=(32, BH, 1), threadgroup=(32, 1, 1),
            output_shapes=[(BH, C, D), (BH, D, D), (BH, D, D),
                           (BH, 4, C, C), (BH, C, D), (BH, C, D), (BH, C, D)],
            output_dtypes=[mx.float32] * 7)
        last = mx.exp(gc_n[:, -1, :])[:, :, None]
        S = S * last + t1 + t2
        outs.append(o); Ams.append(Am); us.append(u); wms.append(wm); v2s.append(v2)
        mx.eval(S, o, Am, u, wm, v2)
    o = mx.concatenate(outs, axis=1)
    return o, dict(S_in=S_ins, Am=Ams, u=us, wmat=wms, v2=v2s)


def chunk_bwd_one_metal_bh_saved(R, W, K, V, A, B, S_in, dS, do, Am, u, wmat, v2):
    """BH per-chunk backward на СОХРАНЁННЫХ интермедиатах (без MLX-рекомпьюта).
    Am [BH,4,C,C]; u/wmat/v2 [BH,C,D]."""
    BH, C, D = R.shape
    gk = mx.log(W); gc = mx.cumsum(gk, axis=1)
    Aqk = Am[:, 0]; Aqb = Am[:, 1]; Aab = Am[:, 2]; Aak = Am[:, 3]
    kb = _kb_kernel(C, D)
    outs = kb(inputs=[R, K, V, B, gc, do, u, wmat, v2, S_in, dS, Aqk, Aqb, Aab, Aak],
              grid=(32, BH, 1), threadgroup=(32, 1, 1),
              output_shapes=[(BH, C, D)] * 5 + [(BH, C, C)] * 4 + [(BH, D, D)] * 2,
              output_dtypes=[mx.float32] * 11)
    dv, dkdec, dbdec, dqh_s, dah_extra, dAqk, dAqb, dAab, dAak, dSin_a, dSin_b = outs
    k2 = _bwd2_bh_kernel(C, D)
    dqh_h, dkh, dbh, dah_h = k2(
        inputs=[R, K, A, B, gc, gk, dAqk, dAqb, dAab, dAak],
        grid=(32, BH, 1), threadgroup=(32, 1, 1),
        output_shapes=[(BH, C, D)] * 4, output_dtypes=[mx.float32] * 4)
    Eg = mx.exp(gc); Eng = mx.exp(-gc); Eag = mx.exp(gc - gk)
    qh = R * Eg; kh = K * Eng; bh = B * Eng; ah = A * Eag
    gc_last = gc[:, -1]; last = mx.exp(gc_last); dec = mx.exp(gc_last[:, None, :] - gc)
    dqh = dqh_h + dqh_s; dah = dah_h + dah_extra
    dq = dqh * Eg; dk = dkh * Eng; db = dbh * Eng; da = dah * Eag
    dgc = dqh * qh - dkh * kh - dbh * bh + dah * ah
    dgk = -dah * ah
    dk = dk + dkdec * dec; db = db + dbdec * dec
    ddec = dkdec * K + dbdec * B
    dgc = dgc - ddec * dec
    dgc_last = last * (dS * S_in).sum(axis=2) + (ddec * dec).sum(axis=1)
    idx = mx.arange(C)[None, :, None]
    dgc = dgc + mx.where(idx == C - 1, dgc_last[:, None, :], 0.0)
    dgk = dgk + mx.cumsum(dgc, axis=1, reverse=True)
    dw = dgk / W; dr = dq
    dS_in = last[:, :, None] * dS + dSin_a + dSin_b
    return dr, dw, dk, dv, da, db, dS_in


def dplr_bwd_metal_bh_saved(R, W, K, V, A, B, do, C):
    """S3.4b-iii: полный fwd(SAVE)+bwd на Metal БЕЗ MLX в тяжёлом пути. Вход [BH,T,D]."""
    BH, T, D = R.shape
    N = T // C
    _, cache = dplr_forward_metal_save(R, W, K, V, A, B, C)
    grads = [None] * N
    dS = mx.zeros((BH, D, D))
    for n in range(N - 1, -1, -1):
        s = slice(n * C, (n + 1) * C)
        g = chunk_bwd_one_metal_bh_saved(
            R[:, s], W[:, s], K[:, s], V[:, s], A[:, s], B[:, s],
            cache["S_in"][n], dS, do[:, s],
            cache["Am"][n], cache["u"][n], cache["wmat"][n], cache["v2"][n])
        grads[n] = g[:6]; dS = g[6]
    return tuple(mx.concatenate([grads[n][i] for n in range(N)], axis=1) for i in range(6))


def dplr_bwd_metal_bh_batched(R, W, K, V, A, B, do, C):
    """S3.4b-iv: leaf-сборка вынесена из carry-цикла в один батч-проход; bwd2 — один
    батч-диспатч по всем чанкам (BH*N). Натуральная [BH,...]-раскладка: все reshape
    contiguous (без transpose), carry остаётся ленивым. Эквив. dplr_bwd_metal_bh_saved."""
    BH, T, D = R.shape
    N = T // C
    _, cache = dplr_forward_metal_save(R, W, K, V, A, B, C)
    gk_f = mx.log(W); gc_f = mx.cumsum(gk_f.reshape(BH, N, C, D), axis=2)

    # carry-цикл: только KB + ленивое последовательное обновление dS
    kb = _kb_kernel(C, D)
    raw = [None] * N; dS_used = [None] * N
    dS = mx.zeros((BH, D, D))
    for n in range(N - 1, -1, -1):
        s = slice(n * C, (n + 1) * C); Am = cache["Am"][n]; gc_n = gc_f[:, n]
        outs = kb(inputs=[R[:, s], K[:, s], V[:, s], B[:, s], gc_n, do[:, s],
                          cache["u"][n], cache["wmat"][n], cache["v2"][n],
                          cache["S_in"][n], dS, Am[:, 0], Am[:, 1], Am[:, 2], Am[:, 3]],
                  grid=(32, BH, 1), threadgroup=(32, 1, 1),
                  output_shapes=[(BH, C, D)] * 5 + [(BH, C, C)] * 4 + [(BH, D, D)] * 2,
                  output_dtypes=[mx.float32] * 11)
        raw[n] = outs; dS_used[n] = dS
        dS = mx.exp(gc_n[:, -1, :])[:, :, None] * dS + outs[9] + outs[10]   # carry, ленивый

    # стек в натуральную [BH,N,...]
    def S1(i): return mx.stack([raw[n][i] for n in range(N)], axis=1)   # [BH,N,C,D|C,C|D,D]
    dv = S1(0); dkdec = S1(1); dbdec = S1(2); dqh_s = S1(3); dah_extra = S1(4)
    dAqk = S1(5); dAqb = S1(6); dAab = S1(7); dAak = S1(8)
    Su = mx.stack(dS_used, axis=1)                               # [BH,N,D,D]
    Sin = mx.stack([cache["S_in"][n] for n in range(N)], axis=1)  # [BH,N,D,D]

    # bwd2 ОДНИМ батчем (BH*N); reshape contiguous, индексация bh*N+n
    NB = BH * N
    r4 = lambda x: x.reshape(BH, N, C, D).reshape(NB, C, D)        # [BH,T,D]->[BH*N,C,D]
    f4 = lambda x: x.reshape(NB, C, D)                            # [BH,N,C,D]->[BH*N,C,D]
    fcc = lambda x: x.reshape(NB, C, C)
    gc_nb = gc_f.reshape(NB, C, D); gk_nb = gk_f.reshape(NB, C, D)
    k2 = _bwd2_bh_kernel(C, D)
    dqh_h, dkh, dbh, dah_h = k2(
        inputs=[r4(R), r4(K), r4(A), r4(B), gc_nb, gk_nb,
                fcc(dAqk), fcc(dAqb), fcc(dAab), fcc(dAak)],
        grid=(32, NB, 1), threadgroup=(32, 1, 1),
        output_shapes=[(NB, C, D)] * 4, output_dtypes=[mx.float32] * 4)
    u4 = lambda x: x.reshape(BH, N, C, D)
    dqh_h, dkh, dbh, dah_h = map(u4, (dqh_h, dkh, dbh, dah_h))

    # batched leaf-сборка по [BH,N,C,D], cumsum по C=axis2 (всё reshape contiguous)
    g = lambda x: x.reshape(BH, N, C, D)
    Rn, Kn, An, Bn, Wn = g(R), g(K), g(A), g(B), g(W)
    gc = gc_f; gk = gk_f.reshape(BH, N, C, D)
    Eg = mx.exp(gc); Eng = mx.exp(-gc); Eag = mx.exp(gc - gk)
    qh = Rn * Eg; kh = Kn * Eng; bh = Bn * Eng; ah = An * Eag
    gc_last = gc[:, :, -1]; last = mx.exp(gc_last)
    dec = mx.exp(gc_last[:, :, None, :] - gc)
    dqh = dqh_h + dqh_s; dah = dah_h + dah_extra
    dq = dqh * Eg; dk = dkh * Eng; db = dbh * Eng; da = dah * Eag
    dgc = dqh * qh - dkh * kh - dbh * bh + dah * ah
    dgk = -dah * ah
    dk = dk + dkdec * dec; db = db + dbdec * dec
    ddec = dkdec * Kn + dbdec * Bn
    dgc = dgc - ddec * dec
    dgc_last = last * (Su * Sin).sum(axis=3) + (ddec * dec).sum(axis=2)   # [BH,N,D]
    idx = mx.arange(C)[None, None, :, None]
    dgc = dgc + mx.where(idx == C - 1, dgc_last[:, :, None, :], 0.0)
    dgk = dgk + mx.cumsum(dgc, axis=2, reverse=True)
    dw = dgk / Wn; dr = dq
    o = lambda x: x.reshape(BH, T, D)
    return o(dr), o(dw), o(dk), o(dv), o(da), o(db)


def _bwd2_and_leaf(R, W, K, A, B, C, N, raw, Su, Sin):
    """Общий хвост: bwd2 одним батчем (BH*N) + векторизованная leaf-сборка по [BH,N,C,D].
    raw = dict с [BH,N,...] стеками выходов KB. Возврат dr,dw,dk,dv,da,db [BH,T,D]."""
    BH = R.shape[0]; T = R.shape[1]; D = R.shape[2]; NB = BH * N
    gk_f = mx.log(W); gc_f = mx.cumsum(gk_f.reshape(BH, N, C, D), axis=2)
    dAqk, dAqb, dAab, dAak = raw["dAqk"], raw["dAqb"], raw["dAab"], raw["dAak"]
    fcc = lambda x: x.reshape(NB, C, C)
    r4 = lambda x: x.reshape(BH, N, C, D).reshape(NB, C, D)
    k2 = _bwd2_bh_kernel(C, D)
    dqh_h, dkh, dbh, dah_h = k2(
        inputs=[r4(R), r4(K), r4(A), r4(B), gc_f.reshape(NB, C, D), gk_f.reshape(NB, C, D),
                fcc(dAqk), fcc(dAqb), fcc(dAab), fcc(dAak)],
        grid=(32, NB, 1), threadgroup=(32, 1, 1),
        output_shapes=[(NB, C, D)] * 4, output_dtypes=[mx.float32] * 4)
    u4 = lambda x: x.reshape(BH, N, C, D)
    dqh_h, dkh, dbh, dah_h = map(u4, (dqh_h, dkh, dbh, dah_h))
    g = lambda x: x.reshape(BH, N, C, D)
    Rn, Kn, An, Bn, Wn = g(R), g(K), g(A), g(B), g(W)
    gc = gc_f; gk = gk_f.reshape(BH, N, C, D)
    Eg = mx.exp(gc); Eng = mx.exp(-gc); Eag = mx.exp(gc - gk)
    qh = Rn * Eg; kh = Kn * Eng; bh = Bn * Eng; ah = An * Eag
    gc_last = gc[:, :, -1]; last = mx.exp(gc_last); dec = mx.exp(gc_last[:, :, None, :] - gc)
    dqh = dqh_h + raw["dqh_s"]; dah = dah_h + raw["dah_extra"]
    dq = dqh * Eg; dk = dkh * Eng; db = dbh * Eng; da = dah * Eag
    dgc = dqh * qh - dkh * kh - dbh * bh + dah * ah
    dgk = -dah * ah
    dk = dk + raw["dkdec"] * dec; db = db + raw["dbdec"] * dec
    ddec = raw["dkdec"] * Kn + raw["dbdec"] * Bn
    dgc = dgc - ddec * dec
    dgc_last = last * (Su * Sin).sum(axis=3) + (ddec * dec).sum(axis=2)
    idx = mx.arange(C)[None, None, :, None]
    dgc = dgc + mx.where(idx == C - 1, dgc_last[:, :, None, :], 0.0)
    dgk = dgk + mx.cumsum(dgc, axis=2, reverse=True)
    dw = dgk / Wn; dr = dq
    o = lambda x: x.reshape(BH, T, D)
    return o(dr), o(dw), o(dk), o(raw["dv"]), o(da), o(db)


def _dS_carry_seq(R, W, K, V, A, B, do, C, cache):
    """ВРЕМЕННЫЙ producer dS[n] (Фаза 1): последовательный carry через KB. В Фазе 2
    заменяется лёгким scan-ядром. Возврат dS_used [BH,N,D,D]."""
    BH, T, D = R.shape; N = T // C
    gc_f = mx.cumsum(mx.log(W).reshape(BH, N, C, D), axis=2)
    kb = _kb_kernel(C, D); dS = mx.zeros((BH, D, D)); used = [None] * N
    for n in range(N - 1, -1, -1):
        s = slice(n * C, (n + 1) * C); Am = cache["Am"][n]; gc_n = gc_f[:, n]
        outs = kb(inputs=[R[:, s], K[:, s], V[:, s], B[:, s], gc_n, do[:, s],
                          cache["u"][n], cache["wmat"][n], cache["v2"][n],
                          cache["S_in"][n], dS, Am[:, 0], Am[:, 1], Am[:, 2], Am[:, 3]],
                  grid=(32, BH, 1), threadgroup=(32, 1, 1),
                  output_shapes=[(BH, C, D)] * 5 + [(BH, C, C)] * 4 + [(BH, D, D)] * 2,
                  output_dtypes=[mx.float32] * 11)
        used[n] = dS
        dS = mx.exp(gc_n[:, -1, :])[:, :, None] * dS + outs[9] + outs[10]
    return mx.stack(used, axis=1)


def dplr_bwd_metal_bh_parallel(R, W, K, V, A, B, do, C):
    """S3.4b-v Фаза 1: KB ОДНИМ диспатчем по всем чанкам (grid BH*N, параллельно).
    dS[n] пока из _dS_carry_seq (throwaway). Тело KB не менялось."""
    BH, T, D = R.shape; N = T // C; NB = BH * N
    _, cache = dplr_forward_metal_save(R, W, K, V, A, B, C)
    dS_used = _dS_carry_seq(R, W, K, V, A, B, do, C, cache)        # [BH,N,D,D]
    gc_f = mx.cumsum(mx.log(W).reshape(BH, N, C, D), axis=2)
    # flatten всех чанков -> [BH*N, ...]
    f = lambda x: x.reshape(BH, N, C, D).reshape(NB, C, D)
    Am = mx.stack([cache["Am"][n] for n in range(N)], axis=1)      # [BH,N,4,C,C]
    s5 = lambda i: Am[:, :, i].reshape(NB, C, C)
    st = lambda key: mx.stack([cache[key][n] for n in range(N)], axis=1).reshape(NB, C, D)
    Sin_nb = mx.stack([cache["S_in"][n] for n in range(N)], axis=1).reshape(NB, D, D)
    dS_nb = dS_used.reshape(NB, D, D)
    kb = _kb_kernel(C, D)
    outs = kb(inputs=[f(R), f(K), f(V), f(B), gc_f.reshape(NB, C, D), f(do),
                      st("u"), st("wmat"), st("v2"), Sin_nb, dS_nb,
                      s5(0), s5(1), s5(2), s5(3)],
              grid=(32, NB, 1), threadgroup=(32, 1, 1),
              output_shapes=[(NB, C, D)] * 5 + [(NB, C, C)] * 4 + [(NB, D, D)] * 2,
              output_dtypes=[mx.float32] * 11)
    rs = lambda x, last2: x.reshape(BH, N, *last2)
    raw = dict(dv=rs(outs[0], (C, D)), dkdec=rs(outs[1], (C, D)), dbdec=rs(outs[2], (C, D)),
               dqh_s=rs(outs[3], (C, D)), dah_extra=rs(outs[4], (C, D)),
               dAqk=rs(outs[5], (C, C)), dAqb=rs(outs[6], (C, C)),
               dAab=rs(outs[7], (C, C)), dAak=rs(outs[8], (C, C)))
    Sin = mx.stack([cache["S_in"][n] for n in range(N)], axis=1)
    return _bwd2_and_leaf(R, W, K, A, B, C, N, raw, dS_used, Sin)


# === S3.4b-v ФАЗА 2: лёгкий state-scan (dS[n]) ================================
_prescan_cache = {}


def _prescan_kernel(C, D):
    if (C, D) in _prescan_cache:
        return _prescan_cache[(C, D)]
    CD, CC, DD = C * D, C * C, D * D
    src = f"""
    uint bh=thread_position_in_grid.y; uint lid=thread_index_in_threadgroup;
    const device float* Q=q+bh*{CD}; const device float* DO=dout+bh*{CD};
    const device float* GC=gc+bh*{CD}; const device float* AQB=Aqb+bh*{CC};
    device float* o_dv2l=dv2_local+bh*{CD}; device float* o_qhdo=qhdo+bh*{DD};
    threadgroup float qh[{CD}], dtg[{CD}];
    for(uint e=lid;e<{CD};e+=32u){{ dtg[e]=DO[e]; qh[e]=Q[e]*exp(GC[e]); }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto TG=[](threadgroup float*p,int c,int r){{return tensor<threadgroup float,dextents<int,2>,tensor_inline>(p,dextents<int,2>(c,r));}};
    auto DV=[](const device float*p,int c,int r){{return tensor<device float,dextents<int,2>,tensor_inline>((device float*)p,dextents<int,2>(c,r));}};
    #define RUN(L,R,O) {{ auto lf=(L); auto rg=(R); auto ds=(O); auto cc=op.get_destination_cooperative_tensor<decltype(lf),decltype(rg),float>(); op.run(lf,rg,cc); cc.store(ds); }}
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{C},true,false); tensor_ops::matmul2d<d,execution_simdgroup> op; RUN(DV(AQB,{C},{C}), TG(dtg,{D},{C}), DV(o_dv2l,{D},{C})) }}  // A_qb^T@do
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({D},{D},{C},true,false); tensor_ops::matmul2d<d,execution_simdgroup> op; RUN(TG(qh,{D},{C}), TG(dtg,{D},{C}), DV(o_qhdo,{D},{D})) }}  // qh^T@do
    """
    k = mx.fast.metal_kernel(name=f"dplr_prescan_{C}_{D}",
        input_names=["q", "dout", "gc", "Aqb"], output_names=["dv2_local", "qhdo"],
        header=_HDR, source=src)
    _prescan_cache[(C, D)] = k
    return k


_dscan_cache = {}


def _dscan_kernel(C, D, N):
    if (C, D, N) in _dscan_cache:
        return _dscan_cache[(C, D, N)]
    CD, DD = C * D, D * D
    src = f"""
    uint bh=thread_position_in_grid.y; uint lid=thread_index_in_threadgroup;
    const device float* DV2L=dv2_local+bh*{N*CD}; const device float* BDEC=bdec+bh*{N*CD};
    const device float* QHDO=qhdo+bh*{N*DD}; const device float* WM=wmat+bh*{N*CD};
    const device float* LAST=last+bh*{N*D};
    device float* OUT=dS_out+bh*{N*DD}; device float* WD=wd+bh*{DD};
    threadgroup float dS[{DD}], dv2[{CD}];
    for(uint e=lid;e<{DD};e+=32u) dS[e]=0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto TG=[](threadgroup float*p,int c,int r){{return tensor<threadgroup float,dextents<int,2>,tensor_inline>(p,dextents<int,2>(c,r));}};
    auto DV=[](const device float*p,int c,int r){{return tensor<device float,dextents<int,2>,tensor_inline>((device float*)p,dextents<int,2>(c,r));}};
    #define RUN(L,R,O) {{ auto lf=(L); auto rg=(R); auto ds=(O); auto cc=op.get_destination_cooperative_tensor<decltype(lf),decltype(rg),float>(); op.run(lf,rg,cc); cc.store(ds); }}
    #define BAR threadgroup_barrier(mem_flags::mem_threadgroup);
    for(int t={N}-1;t>=0;t--){{
        for(uint e=lid;e<{DD};e+=32u) OUT[t*{DD}+e]=dS[e];          // dS_used[t] = carry до апдейта
        BAR
        {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{D},false,false); tensor_ops::matmul2d<d,execution_simdgroup> op; RUN(DV(BDEC+t*{CD},{D},{C}), TG(dS,{D},{D}), TG(dv2,{D},{C})) }}  // bdec@dS
        BAR
        for(uint e=lid;e<{CD};e+=32u) dv2[e]+=DV2L[t*{CD}+e];        // + A_qb^T@do
        BAR
        {{ constexpr auto d=tensor_ops::matmul2d_descriptor({D},{D},{C},true,false); tensor_ops::matmul2d<d,execution_simdgroup> op; RUN(DV(WM+t*{CD},{D},{C}), TG(dv2,{D},{C}), DV(WD,{D},{D})) }}  // wmat^T@dv2
        BAR
        for(uint e=lid;e<{DD};e+=32u){{ uint kk=e/{D}; dS[e]=LAST[t*{D}+kk]*dS[e]+QHDO[t*{DD}+e]+WD[e]; }}
        BAR
    }}
    """
    k = mx.fast.metal_kernel(name=f"dplr_dscan_{C}_{D}_{N}",
        input_names=["dv2_local", "bdec", "qhdo", "wmat", "last"],
        output_names=["dS_out", "wd"], header=_HDR, source=src)
    _dscan_cache[(C, D, N)] = k
    return k


def dS_scan_metal(R, W, K, V, A, B, do, C, cache):
    """Лёгкий dS[n] для всех чанков: prescan (parallel) + scan (sequential). [BH,N,D,D]."""
    BH, T, D = R.shape; N = T // C; NB = BH * N
    gc_f = mx.cumsum(mx.log(W).reshape(BH, N, C, D), axis=2)
    gc_last = gc_f[:, :, -1]; last = mx.exp(gc_last)                 # [BH,N,D]
    dec = mx.exp(gc_last[:, :, None, :] - gc_f)
    bdec = B.reshape(BH, N, C, D) * dec                             # [BH,N,C,D]
    wmat = mx.stack([cache["wmat"][n] for n in range(N)], axis=1)   # [BH,N,C,D]
    # prescan параллельно
    f = lambda x: x.reshape(BH, N, C, D).reshape(NB, C, D)
    Am = mx.stack([cache["Am"][n] for n in range(N)], axis=1)
    pk = _prescan_kernel(C, D)
    dv2l, qhdo = pk(inputs=[f(R), f(do), gc_f.reshape(NB, C, D), Am[:, :, 1].reshape(NB, C, C)],
                    grid=(32, NB, 1), threadgroup=(32, 1, 1),
                    output_shapes=[(NB, C, D), (NB, D, D)], output_dtypes=[mx.float32] * 2)
    dv2l = dv2l.reshape(BH, N, C, D); qhdo = qhdo.reshape(BH, N, D, D)
    # scan последовательно
    sk = _dscan_kernel(C, D, N)
    dS_out, _ = sk(inputs=[dv2l, bdec, qhdo, wmat, last],
                   grid=(32, BH, 1), threadgroup=(32, 1, 1),
                   output_shapes=[(BH, N, D, D), (BH, D, D)], output_dtypes=[mx.float32] * 2)
    return dS_out


def _fast_bwd_given_cache(R, W, K, V, A, B, do, C, cache):
    """Backward по ГОТОВОМУ кэшу (без рекомпьюта форварда): scan + parallel-KB + bwd2 + leaf."""
    BH, T, D = R.shape; N = T // C; NB = BH * N
    dS_used = dS_scan_metal(R, W, K, V, A, B, do, C, cache)
    gc_f = mx.cumsum(mx.log(W).reshape(BH, N, C, D), axis=2)
    f = lambda x: x.reshape(BH, N, C, D).reshape(NB, C, D)
    Am = mx.stack([cache["Am"][n] for n in range(N)], axis=1)
    s5 = lambda i: Am[:, :, i].reshape(NB, C, C)
    st = lambda key: mx.stack([cache[key][n] for n in range(N)], axis=1).reshape(NB, C, D)
    Sin_nb = mx.stack([cache["S_in"][n] for n in range(N)], axis=1).reshape(NB, D, D)
    kb = _kb_kernel(C, D)
    outs = kb(inputs=[f(R), f(K), f(V), f(B), gc_f.reshape(NB, C, D), f(do),
                      st("u"), st("wmat"), st("v2"), Sin_nb, dS_used.reshape(NB, D, D),
                      s5(0), s5(1), s5(2), s5(3)],
              grid=(32, NB, 1), threadgroup=(32, 1, 1),
              output_shapes=[(NB, C, D)] * 5 + [(NB, C, C)] * 4 + [(NB, D, D)] * 2,
              output_dtypes=[mx.float32] * 11)
    rs = lambda x, l2: x.reshape(BH, N, *l2)
    raw = dict(dv=rs(outs[0], (C, D)), dkdec=rs(outs[1], (C, D)), dbdec=rs(outs[2], (C, D)),
               dqh_s=rs(outs[3], (C, D)), dah_extra=rs(outs[4], (C, D)),
               dAqk=rs(outs[5], (C, C)), dAqb=rs(outs[6], (C, C)),
               dAab=rs(outs[7], (C, C)), dAak=rs(outs[8], (C, C)))
    Sin = mx.stack([cache["S_in"][n] for n in range(N)], axis=1)
    return _bwd2_and_leaf_fused(R, W, K, A, B, C, N, raw, dS_used, Sin)


def dplr_bwd_metal_bh_fast(R, W, K, V, A, B, do, C):
    """S3.4b-v ФИНАЛ (recompute-форвард): forward_v2 + _fast_bwd_given_cache."""
    _, cache = dplr_forward_metal_save_v2(R, W, K, V, A, B, C)
    return _fast_bwd_given_cache(R, W, K, V, A, B, do, C, cache)


_leaf_cache = {}


def _leaf_kernel(C, D):
    if (C, D) in _leaf_cache:
        return _leaf_cache[(C, D)]
    CD, DD = C * D, D * D
    src = f"""
    uint bh=thread_position_in_grid.y; uint lid=thread_index_in_threadgroup;
    #define IN(p) (p + bh*{CD})
    const device float* R=IN(r); const device float* K=IN(k); const device float* Av=IN(a);
    const device float* Bv=IN(b); const device float* Wv=IN(w); const device float* GC=IN(gc);
    const device float* GK=IN(gk); const device float* DQHH=IN(dqh_h); const device float* DKH=IN(dkh);
    const device float* DBH=IN(dbh); const device float* DAHH=IN(dah_h); const device float* DQHS=IN(dqh_s);
    const device float* DAHE=IN(dah_extra); const device float* DKD=IN(dkdec); const device float* DBD=IN(dbdec);
    const device float* SU=Su+bh*{DD}; const device float* SIN=Sin+bh*{DD};
    device float* ODR=dr+bh*{CD}; device float* ODW=dw+bh*{CD}; device float* ODK=dk+bh*{CD};
    device float* ODA=da+bh*{CD}; device float* ODB=db+bh*{CD};
    threadgroup float dgc[{CD}], dec[{CD}], ddec[{CD}], dgkb[{CD}], redD[{D}];
    // шаг 1: элементвайз; dr,dk,db,da финальны; dgc/dec/ddec/dgkb в threadgroup
    for(uint e=lid;e<{CD};e+=32u){{
        uint d=e%{D};
        float g=GC[e], gkv=GK[e];
        float Eg=exp(g), Eng=exp(-g), Eag=exp(g-gkv);
        float qh=R[e]*Eg, kh=K[e]*Eng, bhv=Bv[e]*Eng, ah=Av[e]*Eag;
        float dqh=DQHH[e]+DQHS[e], dah=DAHH[e]+DAHE[e];
        float dkh=DKH[e], dbh=DBH[e];
        float glast=GC[({C}-1)*{D}+d];
        float dc=exp(glast-g); dec[e]=dc;
        float dkd=DKD[e], dbd=DBD[e];
        ODR[e]=dqh*Eg;
        ODA[e]=dah*Eag;
        ODK[e]=dkh*Eng + dkd*dc;
        ODB[e]=dbh*Eng + dbd*dc;
        float dde=dkd*K[e]+dbd*Bv[e]; ddec[e]=dde;
        dgc[e]=dqh*qh - dkh*kh - dbh*bhv + dah*ah - dde*dc;
        dgkb[e]=-dah*ah;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // шаг 2: dgc_last[k] = last[k]*Σ_v Su*Sin + Σ_c ddec*dec
    for(uint kk=lid;kk<{D};kk+=32u){{
        float s1=0.0f; for(uint v=0;v<{D};v++) s1+=SU[kk*{D}+v]*SIN[kk*{D}+v];
        float s2=0.0f; for(uint c=0;c<{C};c++) s2+=ddec[c*{D}+kk]*dec[c*{D}+kk];
        redD[kk]=exp(GC[({C}-1)*{D}+kk])*s1 + s2;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // шаг 3: добавить dgc_last в последнюю строку времени
    for(uint kk=lid;kk<{D};kk+=32u) dgc[({C}-1)*{D}+kk]+=redD[kk];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // шаг 4: реверс-cumsum по C, dw = (dgkb + revcumsum(dgc))/W  (каждый поток — колонка d)
    for(uint d=lid;d<{D};d+=32u){{
        float acc=0.0f;
        for(int c={C}-1;c>=0;c--){{ acc+=dgc[c*{D}+d]; uint e=c*{D}+d; ODW[e]=(dgkb[e]+acc)/Wv[e]; }}
    }}
    """
    k = mx.fast.metal_kernel(name=f"dplr_leaf_{C}_{D}",
        input_names=["r", "k", "a", "b", "w", "gc", "gk", "dqh_h", "dkh", "dbh", "dah_h",
                     "dqh_s", "dah_extra", "dkdec", "dbdec", "Su", "Sin"],
        output_names=["dr", "dw", "dk", "da", "db"], header=_HDR, source=src)
    _leaf_cache[(C, D)] = k
    return k


def _bwd2_and_leaf_fused(R, W, K, A, B, C, N, raw, Su, Sin):
    """L1: bwd2 (батч BH*N) + leaf одним Metal-ядром (вместо MLX-элементвайзов)."""
    BH = R.shape[0]; T = R.shape[1]; D = R.shape[2]; NB = BH * N
    gk_f = mx.log(W); gc_f = mx.cumsum(gk_f.reshape(BH, N, C, D), axis=2)
    r4 = lambda x: x.reshape(BH, N, C, D).reshape(NB, C, D)
    fcc = lambda x: x.reshape(NB, C, C)
    gc_nb = gc_f.reshape(NB, C, D); gk_nb = gk_f.reshape(NB, C, D)
    k2 = _bwd2_bh_kernel(C, D)
    dqh_h, dkh, dbh, dah_h = k2(
        inputs=[r4(R), r4(K), r4(A), r4(B), gc_nb, gk_nb,
                fcc(raw["dAqk"]), fcc(raw["dAqb"]), fcc(raw["dAab"]), fcc(raw["dAak"])],
        grid=(32, NB, 1), threadgroup=(32, 1, 1),
        output_shapes=[(NB, C, D)] * 4, output_dtypes=[mx.float32] * 4)
    fl = lambda x: x.reshape(NB, C, D)
    lk = _leaf_kernel(C, D)
    dr, dw, dk, da, db = lk(
        inputs=[r4(R), r4(K), r4(A), r4(B), r4(W), gc_nb, gk_nb,
                dqh_h, dkh, dbh, dah_h, fl(raw["dqh_s"]), fl(raw["dah_extra"]),
                fl(raw["dkdec"]), fl(raw["dbdec"]), Su.reshape(NB, D, D), Sin.reshape(NB, D, D)],
        grid=(32, NB, 1), threadgroup=(32, 1, 1),
        output_shapes=[(NB, C, D)] * 5, output_dtypes=[mx.float32] * 5)
    o = lambda x: x.reshape(BH, T, D)
    dv = o(raw["dv"].reshape(NB, C, D))
    return o(dr), o(dw), o(dk), dv, o(da), o(db)


# === S3.4b-v L2 (форвард-расщеп): параллельное WY-ядро ========================
# Am(masked), u, wmat (carry-независимы) + o_base=A_qk@v, Sbase=(k*dec)^T@v. Grid BH*N.
_wy_cache = {}


def _wy_kernel(C, D):
    if (C, D) in _wy_cache:
        return _wy_cache[(C, D)]
    CD, CC, DD = C * D, C * C, D * D
    src = f"""
    uint bh=thread_position_in_grid.y; uint lid=thread_index_in_threadgroup;
    const device float* Q=q+bh*{CD}; const device float* K=k+bh*{CD}; const device float* V=v+bh*{CD};
    const device float* AL=alpha+bh*{CD}; const device float* BE=beta+bh*{CD};
    const device float* GC=gc+bh*{CD}; const device float* GK=gk+bh*{CD};
    device float* AMo=Am_out+bh*{4*CC}; device float* Uo=u_out+bh*{CD}; device float* WMo=wmat_out+bh*{CD};
    device float* OBo=o_base+bh*{CD}; device float* SBo=s_base+bh*{DD};
    threadgroup float arena[{6*CD}]; threadgroup float Am[{4*CC}];
    threadgroup float* s0=arena+0u*{CD}; threadgroup float* s1=arena+1u*{CD};
    threadgroup float* s2=arena+2u*{CD}; threadgroup float* s3=arena+3u*{CD};
    threadgroup float* s4=arena+4u*{CD};
    auto TG=[](threadgroup float*p,int c,int r){{return tensor<threadgroup float,dextents<int,2>,tensor_inline>(p,dextents<int,2>(c,r));}};
    auto DV=[](device float*p,int c,int r){{return tensor<device float,dextents<int,2>,tensor_inline>(p,dextents<int,2>(c,r));}};
    #define RUN(L,R,O) {{ auto lf=(L); auto rg=(R); auto ds=(O); auto cc=op.get_destination_cooperative_tensor<decltype(lf),decltype(rg),float>(); op.run(lf,rg,cc); cc.store(ds); }}
    #define BAR threadgroup_barrier(mem_flags::mem_threadgroup);
    // hats: s0=qh, s1=kh, s2=bh, s3=ah
    for(uint e=lid;e<{CD};e+=32u){{ float g=GC[e],em=exp(-g),ep=exp(g);
        s0[e]=Q[e]*ep; s1[e]=K[e]*em; s2[e]=BE[e]*em; s3[e]=AL[e]*exp(g-GK[e]); }}
    BAR
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{C},{D},false,true); tensor_ops::matmul2d<d,execution_simdgroup> op; RUN(TG(s0,{D},{C}),TG(s1,{D},{C}),TG(Am+0u*{CC},{C},{C})) }}
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{C},{D},false,true); tensor_ops::matmul2d<d,execution_simdgroup> op; RUN(TG(s0,{D},{C}),TG(s2,{D},{C}),TG(Am+1u*{CC},{C},{C})) }}
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{C},{D},false,true); tensor_ops::matmul2d<d,execution_simdgroup> op; RUN(TG(s3,{D},{C}),TG(s2,{D},{C}),TG(Am+2u*{CC},{C},{C})) }}
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{C},{D},false,true); tensor_ops::matmul2d<d,execution_simdgroup> op; RUN(TG(s3,{D},{C}),TG(s1,{D},{C}),TG(Am+3u*{CC},{C},{C})) }}
    BAR
    for(uint e=lid;e<{CC};e+=32u){{ uint i=e/{C},j=e%{C};
        if(j> i){{ Am[0u*{CC}+e]=0; Am[1u*{CC}+e]=0; }} if(j>=i){{ Am[2u*{CC}+e]=0; Am[3u*{CC}+e]=0; }} }}
    BAR
    for(uint e=lid;e<{4*CC};e+=32u) AMo[e]=Am[e];
    // u = (I-A_ab)^-1 @ (A_ak@v) ; forward subst
    for(uint e=lid;e<{CD};e+=32u) s1[e]=V[e]; BAR
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{C},false,false); tensor_ops::matmul2d<d,execution_simdgroup> op; RUN(TG(Am+3u*{CC},{C},{C}),TG(s1,{D},{C}),TG(s2,{D},{C})) }}  // A_ak@v
    BAR
    for(uint i=1;i<{C};i++){{ for(uint dd=lid;dd<{D};dd+=32u){{ float acc=s2[i*{D}+dd];
        for(uint n=0;n<i;n++) acc+=Am[2u*{CC}+i*{C}+n]*s2[n*{D}+dd]; s2[i*{D}+dd]=acc; }} BAR }}
    for(uint e=lid;e<{CD};e+=32u) Uo[e]=s2[e]; BAR    // u
    // wmat = (I-A_ab)^-1 @ ah
    for(uint e=lid;e<{CD};e+=32u) s3[e]=exp(GC[e]-GK[e])*AL[e]; BAR
    for(uint i=1;i<{C};i++){{ for(uint dd=lid;dd<{D};dd+=32u){{ float acc=s3[i*{D}+dd];
        for(uint n=0;n<i;n++) acc+=Am[2u*{CC}+i*{C}+n]*s3[n*{D}+dd]; s3[i*{D}+dd]=acc; }} BAR }}
    for(uint e=lid;e<{CD};e+=32u) WMo[e]=s3[e]; BAR   // wmat
    // o_base = A_qk@v
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({C},{D},{C},false,false); tensor_ops::matmul2d<d,execution_simdgroup> op; RUN(TG(Am+0u*{CC},{C},{C}),TG(s1,{D},{C}),DV(OBo,{D},{C})) }}
    BAR
    // Sbase = (k*dec)^T @ v
    for(uint e=lid;e<{CD};e+=32u){{ uint d=e%{D}; s4[e]=K[e]*exp(GC[({C}-1)*{D}+d]-GC[e]); }} BAR
    {{ constexpr auto d=tensor_ops::matmul2d_descriptor({D},{D},{C},true,false); tensor_ops::matmul2d<d,execution_simdgroup> op; RUN(TG(s4,{D},{C}),TG(s1,{D},{C}),DV(SBo,{D},{D})) }}
    """
    kern = mx.fast.metal_kernel(name=f"dplr_wy_{C}_{D}",
        input_names=["q", "k", "v", "alpha", "beta", "gc", "gk"],
        output_names=["Am_out", "u_out", "wmat_out", "o_base", "s_base"],
        header=_HDR, source=src)
    _wy_cache[(C, D)] = kern
    return kern


_fscan_cache = {}


def _fscan_kernel(C, D, N):
    if (C, D, N) in _fscan_cache:
        return _fscan_cache[(C, D, N)]
    CD, CC, DD = C * D, C * C, D * D
    src = f"""
    uint bh=thread_position_in_grid.y; uint lid=thread_index_in_threadgroup;
    const device float* R=r+bh*{N*CD}; const device float* BE=beta+bh*{N*CD};
    const device float* GC=gc+bh*{N*CD}; const device float* U=u+bh*{N*CD};
    const device float* WM=wmat+bh*{N*CD}; const device float* OB=o_base+bh*{N*CD};
    const device float* SB=s_base+bh*{N*DD}; const device float* AQB=Aqb+bh*{N*CC};
    device float* OO=o_out+bh*{N*CD}; device float* SIo=S_in_out+bh*{N*DD}; device float* WD=wd+bh*{DD};
    threadgroup float S[{DD}], v2[{CD}], qo[{CD}], t1[{CD}];
    for(uint e=lid;e<{DD};e+=32u) S[e]=0.0f; threadgroup_barrier(mem_flags::mem_threadgroup);
    auto TG=[](threadgroup float*p,int c,int r){{return tensor<threadgroup float,dextents<int,2>,tensor_inline>(p,dextents<int,2>(c,r));}};
    auto DV=[](const device float*p,int c,int r){{return tensor<device float,dextents<int,2>,tensor_inline>((device float*)p,dextents<int,2>(c,r));}};
    #define RUN(L,R_,O) {{ auto lf=(L); auto rg=(R_); auto ds=(O); auto cc=op.get_destination_cooperative_tensor<decltype(lf),decltype(rg),float>(); op.run(lf,rg,cc); cc.store(ds); }}
    #define FF(L,R_,O,M,Nn,Kk) {{ constexpr auto d=tensor_ops::matmul2d_descriptor(M,Nn,Kk,false,false); tensor_ops::matmul2d<d,execution_simdgroup> op; RUN(L,R_,O) }}
    #define BAR threadgroup_barrier(mem_flags::mem_threadgroup);
    for(int t=0;t<{N};t++){{
        for(uint e=lid;e<{DD};e+=32u) SIo[t*{DD}+e]=S[e];                 // S_in[t]
        BAR
        FF(DV(WM+t*{CD},{D},{C}), TG(S,{D},{D}), TG(t1,{D},{C}), {C},{D},{D})   // wmat@S
        BAR
        for(uint e=lid;e<{CD};e+=32u) v2[e]=U[t*{CD}+e]+t1[e];            // v2 = u + wmat@S
        BAR
        FF(DV(AQB+t*{CC},{C},{C}), TG(v2,{D},{C}), TG(t1,{D},{C}), {C},{D},{C})  // A_qb@v2
        BAR
        for(uint e=lid;e<{CD};e+=32u) OO[t*{CD}+e]=OB[t*{CD}+e]+t1[e];    // o = o_base + A_qb@v2
        for(uint e=lid;e<{CD};e+=32u) qo[e]=R[t*{CD}+e]*exp(GC[t*{CD}+e]); // qh
        BAR
        FF(TG(qo,{D},{C}), TG(S,{D},{D}), TG(t1,{D},{C}), {C},{D},{D})    // qh@S
        BAR
        for(uint e=lid;e<{CD};e+=32u) OO[t*{CD}+e]+=t1[e];               // o += qh@S
        for(uint e=lid;e<{CD};e+=32u){{ uint d=e%{D}; qo[e]=BE[t*{CD}+e]*exp(GC[t*{CD}+({C}-1)*{D}+d]-GC[t*{CD}+e]); }}  // bdec
        BAR
        {{ constexpr auto d=tensor_ops::matmul2d_descriptor({D},{D},{C},true,false); tensor_ops::matmul2d<d,execution_simdgroup> op; RUN(TG(qo,{D},{C}), TG(v2,{D},{C}), DV(WD,{D},{D})) }}  // bdec^T@v2
        BAR
        for(uint e=lid;e<{DD};e+=32u){{ uint kk=e/{D}; S[e]=exp(GC[t*{CD}+({C}-1)*{D}+kk])*S[e]+SB[t*{DD}+e]+WD[e]; }}
        BAR
    }}
    """
    kern = mx.fast.metal_kernel(name=f"dplr_fscan_{C}_{D}_{N}",
        input_names=["r", "beta", "gc", "u", "wmat", "o_base", "s_base", "Aqb"],
        output_names=["o_out", "S_in_out", "wd"], header=_HDR, source=src)
    _fscan_cache[(C, D, N)] = kern
    return kern


def dplr_forward_metal_save_v2(R, W, K, V, A, B, C):
    """S3.4b-v L2: форвард расщеплён — параллельное WY-ядро (BH*N) + лёгкий fscan
    (внутр. цикл, carry S). Возврат как dplr_forward_metal_save: o[BH,T,D] + кэш."""
    BH, T, D = R.shape; N = T // C; NB = BH * N
    gk = mx.log(W); gc = mx.cumsum(gk.reshape(BH, N, C, D), axis=2)
    f = lambda x: x.reshape(BH, N, C, D).reshape(NB, C, D)
    wy = _wy_kernel(C, D)
    Am, u, wmat, o_base, s_base = wy(
        inputs=[f(R), f(K), f(V), f(A), f(B), gc.reshape(NB, C, D), gk.reshape(NB, C, D)],
        grid=(32, NB, 1), threadgroup=(32, 1, 1),
        output_shapes=[(NB, 4, C, C), (NB, C, D), (NB, C, D), (NB, C, D), (NB, D, D)],
        output_dtypes=[mx.float32] * 5)
    g = lambda x, l2: x.reshape(BH, N, *l2)
    Am = g(Am, (4, C, C)); u4 = g(u, (C, D)); wmat4 = g(wmat, (C, D))
    ob4 = g(o_base, (C, D)); sb4 = g(s_base, (D, D))
    fs = _fscan_kernel(C, D, N)
    o_out, S_in_out, _ = fs(
        inputs=[R.reshape(BH, N, C, D), B.reshape(BH, N, C, D), gc, u4, wmat4, ob4, sb4, Am[:, :, 1]],
        grid=(32, BH, 1), threadgroup=(32, 1, 1),
        output_shapes=[(BH, N, C, D), (BH, N, D, D), (BH, D, D)], output_dtypes=[mx.float32] * 3)
    # v2[n] = u + wmat@S_in[n] (нужно для кэша backward) — параллельно (1 матмул-батч)
    v2 = u4 + wmat4 @ S_in_out
    o = o_out.reshape(BH, T, D)
    cache = dict(S_in=[S_in_out[:, n] for n in range(N)], Am=[Am[:, n] for n in range(N)],
                 u=[u4[:, n] for n in range(N)], wmat=[wmat4[:, n] for n in range(N)],
                 v2=[v2[:, n] for n in range(N)])
    return o, cache


# === S3.4b-v L3: arrays-путь (без stack/slice roundtrip кэша) =================
def _fwd_v2_arrays(R, W, K, V, A, B, C):
    """forward v2, но возвращает кэш как [BH,N,...] массивы (без dict/списков)."""
    BH, T, D = R.shape; N = T // C; NB = BH * N
    gk = mx.log(W); gc = mx.cumsum(gk.reshape(BH, N, C, D), axis=2)
    f = lambda x: x.reshape(BH, N, C, D).reshape(NB, C, D)
    wy = _wy_kernel(C, D)
    Am, u, wmat, o_base, s_base = wy(
        inputs=[f(R), f(K), f(V), f(A), f(B), gc.reshape(NB, C, D), gk.reshape(NB, C, D)],
        grid=(32, NB, 1), threadgroup=(32, 1, 1),
        output_shapes=[(NB, 4, C, C), (NB, C, D), (NB, C, D), (NB, C, D), (NB, D, D)],
        output_dtypes=[mx.float32] * 5)
    g = lambda x, l2: x.reshape(BH, N, *l2)
    Am = g(Am, (4, C, C)); u4 = g(u, (C, D)); wmat4 = g(wmat, (C, D))
    ob4 = g(o_base, (C, D)); sb4 = g(s_base, (D, D))
    fs = _fscan_kernel(C, D, N)
    o_out, S_in_out, _ = fs(
        inputs=[R.reshape(BH, N, C, D), B.reshape(BH, N, C, D), gc, u4, wmat4, ob4, sb4, Am[:, :, 1]],
        grid=(32, BH, 1), threadgroup=(32, 1, 1),
        output_shapes=[(BH, N, C, D), (BH, N, D, D), (BH, D, D)], output_dtypes=[mx.float32] * 3)
    v2 = u4 + wmat4 @ S_in_out
    return o_out.reshape(BH, T, D), S_in_out, Am, u4, wmat4, v2


def dS_scan_from_arrays(R, W, K, V, A, B, do, C, wmat_all, Aqb_all):
    BH, T, D = R.shape; N = T // C; NB = BH * N
    gc_f = mx.cumsum(mx.log(W).reshape(BH, N, C, D), axis=2)
    gc_last = gc_f[:, :, -1]; last = mx.exp(gc_last)
    dec = mx.exp(gc_last[:, :, None, :] - gc_f)
    bdec = B.reshape(BH, N, C, D) * dec
    f = lambda x: x.reshape(BH, N, C, D).reshape(NB, C, D)
    pk = _prescan_kernel(C, D)
    dv2l, qhdo = pk(inputs=[f(R), f(do), gc_f.reshape(NB, C, D), Aqb_all.reshape(NB, C, C)],
                    grid=(32, NB, 1), threadgroup=(32, 1, 1),
                    output_shapes=[(NB, C, D), (NB, D, D)], output_dtypes=[mx.float32] * 2)
    dv2l = dv2l.reshape(BH, N, C, D); qhdo = qhdo.reshape(BH, N, D, D)
    sk = _dscan_kernel(C, D, N)
    dS_out, _ = sk(inputs=[dv2l, bdec, qhdo, wmat_all, last],
                   grid=(32, BH, 1), threadgroup=(32, 1, 1),
                   output_shapes=[(BH, N, D, D), (BH, D, D)], output_dtypes=[mx.float32] * 2)
    return dS_out


def _fast_bwd_from_arrays(R, W, K, V, A, B, do, C, S_all, Am_all, u_all, wmat_all, v2_all):
    """Backward по [BH,N,...] кэшу НАПРЯМУЮ — без stack/list. (L3)"""
    BH, T, D = R.shape; N = T // C; NB = BH * N
    dS_used = dS_scan_from_arrays(R, W, K, V, A, B, do, C, wmat_all, Am_all[:, :, 1])
    gc_f = mx.cumsum(mx.log(W).reshape(BH, N, C, D), axis=2)
    f = lambda x: x.reshape(BH, N, C, D).reshape(NB, C, D)
    r3 = lambda x: x.reshape(NB, C, D)
    s5 = lambda i: Am_all[:, :, i].reshape(NB, C, C)
    kb = _kb_kernel(C, D)
    outs = kb(inputs=[f(R), f(K), f(V), f(B), gc_f.reshape(NB, C, D), f(do),
                      r3(u_all), r3(wmat_all), r3(v2_all), S_all.reshape(NB, D, D),
                      dS_used.reshape(NB, D, D), s5(0), s5(1), s5(2), s5(3)],
              grid=(32, NB, 1), threadgroup=(32, 1, 1),
              output_shapes=[(NB, C, D)] * 5 + [(NB, C, C)] * 4 + [(NB, D, D)] * 2,
              output_dtypes=[mx.float32] * 11)
    rs = lambda x, l2: x.reshape(BH, N, *l2)
    raw = dict(dv=rs(outs[0], (C, D)), dkdec=rs(outs[1], (C, D)), dbdec=rs(outs[2], (C, D)),
               dqh_s=rs(outs[3], (C, D)), dah_extra=rs(outs[4], (C, D)),
               dAqk=rs(outs[5], (C, C)), dAqb=rs(outs[6], (C, C)),
               dAab=rs(outs[7], (C, C)), dAak=rs(outs[8], (C, C)))
    return _bwd2_and_leaf_fused(R, W, K, A, B, C, N, raw, dS_used, S_all)
