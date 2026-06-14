"""
wkv7_checkpoint.py — full-sequence Metal kernel
================================================
Forward и backward — по ОДНОМУ GPU-вызову на весь T.
Убирает 16 Python-итераций + mx.eval() и 32 GPU sync-точки.

Ключевые идеи:
  Forward:  один kernel, обрабатывает все T токенов,
            сохраняет h после каждых CHUNK=32 токенов → h_checkpoints[B,H,N,D,D]

  Backward: один kernel, внешний цикл по N чанкам (GPU-side),
            читает h_checkpoints[c] вместо реконструкции с нуля
            → стабильно численно (только 32 шага /w за чанк, не 512)

Результат: 1.73× ускорение vs v2 chunked (T=512, медиана 40 итераций)
"""
import mlx.core as mx

HEAD_SIZE = 64
CHUNK = 32

_fwd_cache: dict = {}
_bwd_cache: dict = {}

def _get_ckpt_fwd(H: int, T: int):
    key = (H, T)
    if key in _fwd_cache: return _fwd_cache[key]
    N = T // CHUNK
    hdr = f"""
constant uint HEAD_SIZE_C = {HEAD_SIZE};
constant uint T_C         = {T};
constant uint CHUNK_C     = {CHUNK};
constant uint N_CHUNKS_C  = {N};
constant uint H_C         = {H};
"""
    src = r"""
    uint dv  = thread_position_in_grid.y;
    uint bhi = thread_position_in_grid.x;
    uint bi  = bhi / H_C, hi = bhi % H_C;
    float h_row[HEAD_SIZE_C];
    uint hb = (bi*H_C+hi)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_row[dk] = h_in[hb+dk];

    for (uint c=0; c<N_CHUNKS_C; c++) {
        for (uint t=0; t<CHUNK_C; t++) {
            uint base = ((bi*T_C + c*CHUNK_C + t)*H_C + hi)*HEAD_SIZE_C;
            float sa = 0;
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) sa += h_row[dk]*a[base+dk];
            sa_out[base+dv] = sa;
            float vv = v[base+dv];
            for (uint dk=0; dk<HEAD_SIZE_C; dk++)
                h_row[dk] = w[base+dk]*h_row[dk] + vv*k[base+dk] + sa*b[base+dk];
            float y = 0;
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) y += h_row[dk]*r[base+dk];
            out[base+dv] = y;
        }
        // Сохраняем h-checkpoint после каждого чанка
        uint ckb = ((bi*H_C+hi)*N_CHUNKS_C + c)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_checkpoints[ckb+dk] = h_row[dk];
    }
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_out[hb+dk] = h_row[dk];
"""
    kern = mx.fast.metal_kernel(
        name=f"wkv7_ckpt_fwd_H{H}_T{T}",
        input_names=["r","w","k","v","a","b","h_in"],
        output_names=["out","h_out","sa_out","h_checkpoints"],
        header=hdr, source=src,
    )
    _fwd_cache[key] = kern
    return kern

def _get_ckpt_bwd(H: int, T: int):
    key = (H, T)
    if key in _bwd_cache: return _bwd_cache[key]
    N = T // CHUNK
    hdr = f"""
constant uint HEAD_SIZE_C = {HEAD_SIZE};
constant uint T_C         = {T};
constant uint CHUNK_C     = {CHUNK};
constant uint N_CHUNKS_C  = {N};
constant uint H_C         = {H};
"""
    src = r"""
    uint dv  = thread_position_in_threadgroup.x;
    uint bhi = threadgroup_position_in_grid.x;
    uint bi  = bhi / H_C, hi = bhi % H_C;

    threadgroup float accum[HEAD_SIZE_C][HEAD_SIZE_C];
    threadgroup float k_sh[HEAD_SIZE_C], v_sh[HEAD_SIZE_C], r_sh[HEAD_SIZE_C];
    threadgroup float w_sh[HEAD_SIZE_C], a_sh[HEAD_SIZE_C], b_sh[HEAD_SIZE_C];
    threadgroup float dy_sh[HEAD_SIZE_C], sa_sh[HEAD_SIZE_C], dsa_sh[HEAD_SIZE_C];

    float C_row[HEAD_SIZE_C], h_row[HEAD_SIZE_C];
    uint hb = (bi*H_C+hi)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) C_row[dk] = d_h_out[hb+dk];

    for (int c=(int)N_CHUNKS_C-1; c>=0; c--) {
        // Загружаем точный h-checkpoint для этого чанка
        uint ckb = ((bi*H_C+hi)*N_CHUNKS_C+(uint)c)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_row[dk] = h_ckpts[ckb+dk];

        for (int t=(int)CHUNK_C-1; t>=0; t--) {
            uint base = ((bi*T_C+(uint)c*CHUNK_C+(uint)t)*H_C+hi)*HEAD_SIZE_C;

            k_sh[dv]=k[base+dv]; v_sh[dv]=v[base+dv]; r_sh[dv]=r[base+dv];
            w_sh[dv]=w[base+dv]; a_sh[dv]=a[base+dv]; b_sh[dv]=b[base+dv];
            dy_sh[dv]=d_out[base+dv]; sa_sh[dv]=sa_fwd[base+dv];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float dy_dv = dy_sh[dv];
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) C_row[dk] += dy_dv*r_sh[dk];

            float dsa_dv=0, dv_val=0;
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) {
                dsa_dv += C_row[dk]*b_sh[dk];
                dv_val  += C_row[dk]*k_sh[dk];
            }
            dv_out[base+dv] = dv_val;
            dsa_sh[dv] = dsa_dv;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk] = dy_dv*h_row[dk];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float dr_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) dr_val+=accum[s][dv];
            dr_out[base+dv] = dr_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float sa_dv=sa_sh[dv], v_dv=v_sh[dv];
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) {
                float hp=(h_row[dk]-v_dv*k_sh[dk]-sa_dv*b_sh[dk])/w_sh[dk];
                accum[dv][dk]=C_row[dk]*hp; h_row[dk]=hp;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float dw_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) dw_val+=accum[s][dv];
            dw_out[base+dv] = dw_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk]=C_row[dk]*v_dv;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float dk_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) dk_val+=accum[s][dv];
            dk_out[base+dv] = dk_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk]=dsa_sh[dv]*h_row[dk];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float da_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) da_val+=accum[s][dv];
            da_out[base+dv] = da_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk]=sa_sh[dv]*C_row[dk];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float db_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) db_val+=accum[s][dv];
            db_out[base+dv] = db_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint dk=0; dk<HEAD_SIZE_C; dk++)
                C_row[dk] = C_row[dk]*w_sh[dk] + dsa_dv*a_sh[dk];
        }
    }
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) dh_in_out[hb+dk] = C_row[dk];
"""
    kern = mx.fast.metal_kernel(
        name=f"wkv7_ckpt_bwd_H{H}_T{T}",
        input_names=["r","w","k","v","a","b","h_ckpts","sa_fwd","d_out","d_h_out"],
        output_names=["dr_out","dw_out","dk_out","dv_out","da_out","db_out","dh_in_out"],
        header=hdr, source=src, atomic_outputs=False,
    )
    _bwd_cache[key] = kern
    return kern


def make_wkv7_checkpoint(B: int, T: int, H: int, D: int = HEAD_SIZE):
    """
    Создаёт функцию wkv7_train использующую checkpoint-kernel.
    Принимает те же аргументы что wkv7_train из wkv7.py.
    """
    assert T % CHUNK == 0, f"T={T} должно делиться на CHUNK={CHUNK}"
    N = T // CHUNK
    h0 = mx.zeros((B, H, D, D))

    @mx.custom_function
    def _fwd(r, w, k, v, a, b, h_in):
        res = _get_ckpt_fwd(H, T)(
            inputs=[x.astype(mx.float32) for x in [r, w, k, v, a, b, h_in]],
            grid=(B*H, D, 1), threadgroup=(1, 1, 1),
            output_shapes=[(B,T,H,D), (B,H,D,D), (B,T,H,D), (B,H,N,D,D)],
            output_dtypes=[mx.float32]*4,
        )
        return res[0], res[1], res[2], res[3]

    @_fwd.vjp
    def _vjp(primals, cotangents, outputs):
        r, w, k, v, a, b, h_in = primals
        d_out, d_h_out, _, _ = cotangents
        _, _, sa_fwd, h_ckpts = outputs
        # mx.eval убран — Metal kernel принимает lazy tensors,
        # mx.compile запрещает eval внутри трансформаций
        res = _get_ckpt_bwd(H, T)(
            inputs=[x.astype(mx.float32) for x in [r, w, k, v, a, b, h_ckpts, sa_fwd, d_out, d_h_out]],
            grid=(B*H*D, 1, 1), threadgroup=(D, 1, 1),
            output_shapes=[(B,T,H,D)]*6 + [(B,H,D,D)],
            output_dtypes=[mx.float32]*7,
        )
        # Приводим градиенты к dtype примала — нужно для bf16/fp16 моделей
        grads = [res[0], res[1], res[2], res[3], res[4], res[5], res[6]]
        return [g.astype(p.dtype) for g, p in zip(grads, primals)]

    def wkv7_train(r, w, k, v, a, b):
        """Drop-in замена для wkv7_train из wkv7.py"""
        out, _, _, _ = _fwd(r, w, k, v, a, b, h0)
        return out

    return wkv7_train
