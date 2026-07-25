"""
wkv7_checkpoint.py — full-sequence Metal kernel
================================================
Forward и backward — по ОДНОМУ GPU-вызову на весь T.
Убирает N Python-итераций + mx.eval() и N GPU sync-точек.

Ключевые идеи:
  Forward:  один kernel, обрабатывает все T токенов,
            сохраняет h после каждых CHUNK=16 токенов → h_checkpoints[B,H,N,D,D]

  Backward: один kernel, внешний цикл по N чанкам (GPU-side),
            читает h_checkpoints[c] вместо реконструкции с нуля
            → стабильно численно (только 16 шагов /w за чанк, не 512;
              32 признан community нестабильным для backward на высокой размерности)

Результат: 1.73× ускорение vs v2 chunked (T=512, медиана 40 итераций)
"""
import mlx.core as mx

HEAD_SIZE = 64
CHUNK = 16

# ── Kernel tuning knobs (measured, see experiments/bench_fwd_variants.py and
#    experiments/bench_bwd_variants.py) ──────────────────────────────────────
# ACC_FWD / ACC_BWD: split each 64-long dot product into N independent
#   accumulators. The kernel is latency-bound on dependent FMA chains, not on
#   ALU throughput or bandwidth (measured: ~0.08 TFLOPS, ~5 GB/s achieved), so
#   breaking the chains is what buys speed. Reassociates the sums, changing
#   results by ~4e-8 -- far under the 1e-5 golden-test bar.
# TILE_BWD: rows of the backward `accum` scratch. 64 = the original single-pass
#   reduction (18.2 KB threadgroup memory); 16 cuts it to 6.2 KB. Worth ~10%.
ACC_FWD = 8
ACC_BWD = 4
TILE_BWD = 16

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
constant uint ACC_C       = {ACC_FWD};
"""
    src = r"""
    // dv индексирует поток ВНУТРИ threadgroup (запуск: grid=(B*H, D),
    // threadgroup=(1, D)), поэтому берём thread_position_in_threadgroup --
    // нужно для стейджинга ниже.
    uint dv  = thread_position_in_threadgroup.y;
    uint bhi = threadgroup_position_in_grid.x;
    uint bi  = bhi / H_C, hi = bhi % H_C;

    // a/w/k/b/r одинаковы для всех 64 потоков threadgroup'а: без стейджинга
    // каждое значение читалось из глобальной памяти 64 раза (по разу на
    // поток). Замер: это стоило ПОЛОВИНЫ времени forward-ядра.
    threadgroup float a_sh[HEAD_SIZE_C], w_sh[HEAD_SIZE_C], k_sh[HEAD_SIZE_C];
    threadgroup float b_sh[HEAD_SIZE_C], r_sh[HEAD_SIZE_C];

    float h_row[HEAD_SIZE_C];
    uint hb = (bi*H_C+hi)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_row[dk] = h_in[hb+dk];

    for (uint c=0; c<N_CHUNKS_C; c++) {
        for (uint t=0; t<CHUNK_C; t++) {
            uint base = ((bi*T_C + c*CHUNK_C + t)*H_C + hi)*HEAD_SIZE_C;

            a_sh[dv]=a[base+dv]; w_sh[dv]=w[base+dv]; k_sh[dv]=k[base+dv];
            b_sh[dv]=b[base+dv]; r_sh[dv]=r[base+dv];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ACC_C независимых аккумуляторов вместо одной цепочки из 64
            // зависимых FMA (ядро latency-bound, см. заметку у ACC_FWD).
            float sacc[ACC_C];
            for (uint i=0; i<ACC_C; i++) sacc[i] = 0.0f;
            for (uint dk=0; dk<HEAD_SIZE_C; dk+=ACC_C)
                for (uint i=0; i<ACC_C; i++) sacc[i] += h_row[dk+i]*a_sh[dk+i];
            float sa = 0.0f;
            for (uint i=0; i<ACC_C; i++) sa += sacc[i];
            sa_out[base+dv] = sa;

            float vv = v[base+dv];
            for (uint dk=0; dk<HEAD_SIZE_C; dk++)
                h_row[dk] = w_sh[dk]*h_row[dk] + vv*k_sh[dk] + sa*b_sh[dk];

            float yacc[ACC_C];
            for (uint i=0; i<ACC_C; i++) yacc[i] = 0.0f;
            for (uint dk=0; dk<HEAD_SIZE_C; dk+=ACC_C)
                for (uint i=0; i<ACC_C; i++) yacc[i] += h_row[dk+i]*r_sh[dk+i];
            float y = 0.0f;
            for (uint i=0; i<ACC_C; i++) y += yacc[i];
            out[base+dv] = y;

            // Барьер перед следующей итерацией: соседний t перезапишет *_sh,
            // пока кто-то ещё читает текущие значения.
            threadgroup_barrier(mem_flags::mem_threadgroup);
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
constant uint ACC_C       = {ACC_BWD};
constant uint TS_C        = {TILE_BWD};
"""
    src = r"""
    uint dv  = thread_position_in_threadgroup.x;
    uint bhi = threadgroup_position_in_grid.x;
    uint bi  = bhi / H_C, hi = bhi % H_C;

    // accum тайлится по индексу потока-источника: [TS_C][64] вместо [64][64].
    // 64x64 float = 16 КБ threadgroup-памяти на группу из всего 64 потоков;
    // TS_C=16 срезает это до 4 КБ (+9 векторов = 6.2 КБ против 18.2 КБ).
    threadgroup float accum[TS_C][HEAD_SIZE_C];
    threadgroup float k_sh[HEAD_SIZE_C], v_sh[HEAD_SIZE_C], r_sh[HEAD_SIZE_C];
    threadgroup float w_sh[HEAD_SIZE_C], a_sh[HEAD_SIZE_C], b_sh[HEAD_SIZE_C];
    threadgroup float dy_sh[HEAD_SIZE_C], sa_sh[HEAD_SIZE_C], dsa_sh[HEAD_SIZE_C];

    float C_row[HEAD_SIZE_C], h_row[HEAD_SIZE_C];
    uint hb = (bi*H_C+hi)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) C_row[dk] = d_h_out[hb+dk];

    // Транспонирующая редукция out[dv] = sum_s VAL_s[dv], тайлами по TS_C
    // потоков-источников. Обе фазы держат все 64 потока занятыми: значения
    // строки считаются из регистров, а суммирование по столбцу ведут все.
    // ACC_C независимых аккумуляторов разрывают цепочку зависимых FMA.
#define TILED_REDUCE(RESULT, WRITE_EXPR)                                      \
    {                                                                         \
        float _acc[ACC_C];                                                    \
        for (uint _i=0; _i<ACC_C; _i++) _acc[_i] = 0.0f;                      \
        for (uint t0 = 0; t0 < HEAD_SIZE_C; t0 += TS_C) {                     \
            if (dv >= t0 && dv < t0 + TS_C) {                                 \
                uint loc = dv - t0;                                           \
                for (uint dk = 0; dk < HEAD_SIZE_C; dk++)                     \
                    accum[loc][dk] = (WRITE_EXPR);                            \
            }                                                                 \
            threadgroup_barrier(mem_flags::mem_threadgroup);                  \
            for (uint s = 0; s < TS_C; s += ACC_C)                            \
                for (uint _i = 0; _i < ACC_C; _i++)                           \
                    _acc[_i] += accum[s+_i][dv];                              \
            threadgroup_barrier(mem_flags::mem_threadgroup);                  \
        }                                                                     \
        RESULT = 0.0f;                                                        \
        for (uint _i=0; _i<ACC_C; _i++) RESULT += _acc[_i];                   \
    }

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

            float _dsa[ACC_C], _dvv[ACC_C];
            for (uint i=0; i<ACC_C; i++) { _dsa[i]=0.0f; _dvv[i]=0.0f; }
            for (uint dk=0; dk<HEAD_SIZE_C; dk+=ACC_C)
                for (uint i=0; i<ACC_C; i++) {
                    _dsa[i] += C_row[dk+i]*b_sh[dk+i];
                    _dvv[i] += C_row[dk+i]*k_sh[dk+i];
                }
            float dsa_dv=0.0f, dv_val=0.0f;
            for (uint i=0; i<ACC_C; i++) { dsa_dv += _dsa[i]; dv_val += _dvv[i]; }
            dv_out[base+dv] = dv_val;
            dsa_sh[dv] = dsa_dv;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // dr использует h_row ДО обновления
            float dr_val;
            TILED_REDUCE(dr_val, dy_sh[dv] * h_row[dk])
            dr_out[base+dv] = dr_val;

            // Обновление h делаем всеми потоками сразу (а не внутри тайла),
            // чтобы к редукции dw оно было завершено у каждого ровно один раз.
            float sa_dv=sa_sh[dv], v_dv=v_sh[dv];
            for (uint dk=0; dk<HEAD_SIZE_C; dk++)
                h_row[dk] = (h_row[dk]-v_dv*k_sh[dk]-sa_dv*b_sh[dk])/w_sh[dk];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float dw_val;
            TILED_REDUCE(dw_val, C_row[dk] * h_row[dk])
            dw_out[base+dv] = dw_val;

            float dk_val;
            TILED_REDUCE(dk_val, C_row[dk] * v_sh[dv])
            dk_out[base+dv] = dk_val;

            float da_val;
            TILED_REDUCE(da_val, dsa_sh[dv] * h_row[dk])
            da_out[base+dv] = da_val;

            float db_val;
            TILED_REDUCE(db_val, sa_sh[dv] * C_row[dk])
            db_out[base+dv] = db_val;

            for (uint dk=0; dk<HEAD_SIZE_C; dk++)
                C_row[dk] = C_row[dk]*w_sh[dk] + dsa_dv*a_sh[dk];

            // The next timestep overwrites the threadgroup arrays above. Wait
            // until every thread has finished reading the current w_sh/a_sh.
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) dh_in_out[hb+dk] = C_row[dk];
#undef TILED_REDUCE
"""
    kern = mx.fast.metal_kernel(
        name=f"wkv7_ckpt_bwd_H{H}_T{T}",
        input_names=["r","w","k","v","a","b","h_ckpts","sa_fwd","d_out","d_h_out"],
        output_names=["dr_out","dw_out","dk_out","dv_out","da_out","db_out","dh_in_out"],
        header=hdr, source=src, atomic_outputs=False,
    )
    _bwd_cache[key] = kern
    return kern


def make_wkv7_checkpoint_with_state(B: int, T: int, H: int, D: int = HEAD_SIZE):
    """
    Create a stateful checkpoint-kernel function.

    The returned callable accepts ``(r, w, k, v, a, b, h_in)`` and returns
    ``(out, h_out)``. Exposing the boundary state makes the full VJP, including
    ``d_h_in``, available for correctness tests and state-tuning use cases.
    """
    assert T % CHUNK == 0, f"T={T} должно делиться на CHUNK={CHUNK}"
    N = T // CHUNK

    @mx.custom_function
    def _fwd(r, w, k, v, a, b, h_in):
        res = _get_ckpt_fwd(H, T)(
            inputs=[x.astype(mx.float32) for x in [r, w, k, v, a, b, h_in]],
            grid=(B*H, D, 1), threadgroup=(1, D, 1),
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

    def wkv7_train_with_state(r, w, k, v, a, b, h_in):
        out, h_out, _, _ = _fwd(r, w, k, v, a, b, h_in)
        return out, h_out

    return wkv7_train_with_state


def make_wkv7_checkpoint(B: int, T: int, H: int, D: int = HEAD_SIZE):
    """
    Создаёт функцию wkv7_train использующую checkpoint-kernel.
    Принимает те же аргументы что wkv7_train из wkv7.py.
    """
    wkv7_train_with_state = make_wkv7_checkpoint_with_state(B, T, H, D)
    h0 = mx.zeros((B, H, D, D))

    def wkv7_train(r, w, k, v, a, b):
        """Drop-in замена для wkv7_train из wkv7.py"""
        out, _ = wkv7_train_with_state(r, w, k, v, a, b, h0)
        return out

    return wkv7_train
