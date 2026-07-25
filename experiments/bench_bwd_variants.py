"""
Backward WKV-7 kernel variants: tiling x ILP.

The forward experiment (bench_fwd_variants.py) found 2.6x: half from staging
broadcast operands in threadgroup memory, half from breaking the 64-deep
dependent FMA chains with split accumulators. The backward kernel already
stages its operands -- but it still has serial reduction chains:

    for (dk) { dsa_dv += C_row[dk]*b_sh[dk]; dv_val += C_row[dk]*k_sh[dk]; }
    ...
    for (s)  RESULT += accum[s][dv];          // x5 reductions per timestep

Each is 64 dependent FMAs. On a latency-bound kernel that is the bottleneck.

Variants (TS = accum tile rows, ACC = accumulator split):
    b0  TS=64  ACC=1   current kernel
    b1  TS=16  ACC=1   tiling only (from bench_bwd_tile.py)
    b2  TS=64  ACC=4   ILP only
    b3  TS=16  ACC=4   both
    b4  TS=16  ACC=8   both, wider

ACC>1 reassociates sums, so results differ by rounding (not bit-exact).

Usage:
    python experiments/bench_bwd_variants.py --B 8 --H 7 --T 512
"""
import argparse
import time

import mlx.core as mx

HEAD_SIZE = 64
CHUNK = 16


def _fwd_kernel(H, T):
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
        uint ckb = ((bi*H_C+hi)*N_CHUNKS_C + c)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_checkpoints[ckb+dk] = h_row[dk];
    }
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_out[hb+dk] = h_row[dk];
"""
    return mx.fast.metal_kernel(
        name=f"bv_fwd_H{H}_T{T}",
        input_names=["r", "w", "k", "v", "a", "b", "h_in"],
        output_names=["out", "h_out", "sa_out", "h_checkpoints"],
        header=hdr, source=src,
    )


def _bwd_kernel(H, T, TS, ACC):
    N = T // CHUNK
    hdr = f"""
constant uint HEAD_SIZE_C = {HEAD_SIZE};
constant uint T_C         = {T};
constant uint CHUNK_C     = {CHUNK};
constant uint N_CHUNKS_C  = {N};
constant uint H_C         = {H};
constant uint TS_C        = {TS};
constant uint ACC_C       = {ACC};
"""
    src = r"""
    uint dv  = thread_position_in_threadgroup.x;
    uint bhi = threadgroup_position_in_grid.x;
    uint bi  = bhi / H_C, hi = bhi % H_C;

    threadgroup float accum[TS_C][HEAD_SIZE_C];
    threadgroup float k_sh[HEAD_SIZE_C], v_sh[HEAD_SIZE_C], r_sh[HEAD_SIZE_C];
    threadgroup float w_sh[HEAD_SIZE_C], a_sh[HEAD_SIZE_C], b_sh[HEAD_SIZE_C];
    threadgroup float dy_sh[HEAD_SIZE_C], sa_sh[HEAD_SIZE_C], dsa_sh[HEAD_SIZE_C];

    float C_row[HEAD_SIZE_C], h_row[HEAD_SIZE_C];
    uint hb = (bi*H_C+hi)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) C_row[dk] = d_h_out[hb+dk];

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

            float dr_val;
            TILED_REDUCE(dr_val, dy_sh[dv] * h_row[dk])
            dr_out[base+dv] = dr_val;

            float sa_dv = sa_sh[dv], v_dv = v_sh[dv];
            for (uint dk=0; dk<HEAD_SIZE_C; dk++)
                h_row[dk] = (h_row[dk] - v_dv*k_sh[dk] - sa_dv*b_sh[dk]) / w_sh[dk];
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

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) dh_in_out[hb+dk] = C_row[dk];
#undef TILED_REDUCE
"""
    return mx.fast.metal_kernel(
        name=f"bv_bwd_H{H}_T{T}_TS{TS}_A{ACC}",
        input_names=["r", "w", "k", "v", "a", "b", "h_ckpts", "sa_fwd", "d_out", "d_h_out"],
        output_names=["dr_out", "dw_out", "dk_out", "dv_out", "da_out", "db_out", "dh_in_out"],
        header=hdr, source=src, atomic_outputs=False,
    )


def run_bwd(kern, ins, B, T, H, D):
    return kern(inputs=ins, grid=(B * H * D, 1, 1), threadgroup=(D, 1, 1),
                output_shapes=[(B, T, H, D)] * 6 + [(B, H, D, D)],
                output_dtypes=[mx.float32] * 7)


def timed(fn, n=5, warm=2):
    for _ in range(warm):
        r = fn(); mx.eval(r)
    t0 = time.time()
    for _ in range(n):
        r = fn(); mx.eval(r)
    return (time.time() - t0) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--H", type=int, default=7)
    ap.add_argument("--T", type=int, default=512)
    args = ap.parse_args()

    B, H, T, D = args.B, args.H, args.T, HEAD_SIZE
    print(f"shape: B={B} H={H} T={T} D={D}  threadgroups={B*H}")

    mx.random.seed(0)
    shape = (B, T, H, D)
    r = mx.random.normal(shape) * 0.1
    w = mx.sigmoid(mx.random.normal(shape)) * 0.5 + 0.5
    k = mx.random.normal(shape) * 0.1
    v = mx.random.normal(shape) * 0.1
    a = mx.random.normal(shape) * 0.1
    b = mx.random.normal(shape) * 0.1
    h_in = mx.zeros((B, H, D, D))
    mx.eval(r, w, k, v, a, b, h_in)

    fwd = _fwd_kernel(H, T)
    out, h_out, sa_fwd, h_ckpts = fwd(
        inputs=[r, w, k, v, a, b, h_in],
        grid=(B * H, D, 1), threadgroup=(1, D, 1),
        output_shapes=[(B, T, H, D), (B, H, D, D), (B, T, H, D),
                       (B, H, T // CHUNK, D, D)],
        output_dtypes=[mx.float32] * 4)
    mx.eval(out, h_out, sa_fwd, h_ckpts)

    d_out = mx.random.normal(shape) * 0.1
    d_h_out = mx.zeros((B, H, D, D))
    mx.eval(d_out, d_h_out)
    ins = [r, w, k, v, a, b, h_ckpts, sa_fwd, d_out, d_h_out]

    variants = [("b0 TS=64 ACC=1 (current)", 64, 1),
                ("b1 TS=16 ACC=1", 16, 1),
                ("b2 TS=64 ACC=4", 64, 4),
                ("b3 TS=16 ACC=4", 16, 4),
                ("b4 TS=16 ACC=8", 16, 8)]

    ref, base_ms = None, None
    print(f"\n  {'variant':<26} {'time ms':>9} {'speedup':>9} {'max err':>11}")
    for label, ts, acc in variants:
        kern = _bwd_kernel(H, T, ts, acc)
        try:
            res = run_bwd(kern, ins, B, T, H, D)
            mx.eval(res)
        except Exception as e:
            print(f"  {label:<26}  FAILED: {type(e).__name__}: {e}")
            continue
        if ref is None:
            ref, err = res, 0.0
        else:
            err = max(mx.abs(x - y).max().item() for x, y in zip(ref, res))
        ms = timed(lambda: run_bwd(kern, ins, B, T, H, D))
        if base_ms is None:
            base_ms = ms
        print(f"  {label:<26} {ms:>9.1f} {base_ms/ms:>8.2f}x {err:>11.2e}")


if __name__ == "__main__":
    main()
