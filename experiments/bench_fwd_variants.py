"""
Forward WKV-7 kernel: where does the time actually go?

The tile experiment (bench_bwd_tile.py) showed the kernel is bound by neither
ALU nor DRAM bandwidth (~0.08 TFLOPS, ~5 GB/s achieved) but by its own serial
structure. Two concrete suspects in `_get_ckpt_fwd`:

  1. REDUNDANT GLOBAL LOADS. threadgroup = 64 threads indexed by dv; every
     thread loops dk=0..63 reading a[base+dk], w[base+dk], k[base+dk],
     b[base+dk], r[base+dk] -- the SAME 64 values, so each is fetched 64x.
     The backward kernel already stages these into threadgroup memory
     (k_sh/v_sh/...); the forward does not.

  2. SERIAL REDUCTION CHAINS. `sa += h_row[dk]*a[dk]` and `y += h_row[dk]*r[dk]`
     are 64 dependent FMAs each. On a latency-bound kernel this is the whole
     ballgame: splitting into K independent accumulators exposes K-way ILP.

Variants:
  v0  current kernel
  v1  v0 + threadgroup staging of w,k,a,b,r  (kills the 64x redundant loads)
  v2  v1 + 4-way split accumulators          (breaks the dependency chains)
  v3  v1 + 8-way split accumulators

v2/v3 reassociate the sums, so they are NOT bit-identical to v0 -- checked
against a tolerance instead. v1 must be bit-exact.

Usage:
    python experiments/bench_fwd_variants.py --B 8 --H 7 --T 512
"""
import argparse
import time

import mlx.core as mx

HEAD_SIZE = 64
CHUNK = 16

_HDR = """
constant uint HEAD_SIZE_C = {HS};
constant uint T_C         = {T};
constant uint CHUNK_C     = {CHUNK};
constant uint N_CHUNKS_C  = {N};
constant uint H_C         = {H};
"""

# ── v0: current kernel, verbatim ──────────────────────────────────────────
_V0 = r"""
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

# ── v1: stage the broadcast operands into threadgroup memory ──────────────
_V1 = r"""
    uint dv  = thread_position_in_threadgroup.y;
    uint bhi = threadgroup_position_in_grid.x;
    uint bi  = bhi / H_C, hi = bhi % H_C;

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

            float sa = 0;
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) sa += h_row[dk]*a_sh[dk];
            sa_out[base+dv] = sa;
            float vv = v[base+dv];
            for (uint dk=0; dk<HEAD_SIZE_C; dk++)
                h_row[dk] = w_sh[dk]*h_row[dk] + vv*k_sh[dk] + sa*b_sh[dk];
            float y = 0;
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) y += h_row[dk]*r_sh[dk];
            out[base+dv] = y;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        uint ckb = ((bi*H_C+hi)*N_CHUNKS_C + c)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_checkpoints[ckb+dk] = h_row[dk];
    }
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_out[hb+dk] = h_row[dk];
"""

# ── v2/v3: v1 + K-way split accumulators (K passed in as ACC) ─────────────
_VSPLIT = r"""
    uint dv  = thread_position_in_threadgroup.y;
    uint bhi = threadgroup_position_in_grid.x;
    uint bi  = bhi / H_C, hi = bhi % H_C;

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
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        uint ckb = ((bi*H_C+hi)*N_CHUNKS_C + c)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_checkpoints[ckb+dk] = h_row[dk];
    }
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_out[hb+dk] = h_row[dk];
"""


def build(name, src, H, T, acc=None):
    hdr = _HDR.format(HS=HEAD_SIZE, T=T, CHUNK=CHUNK, N=T // CHUNK, H=H)
    if acc is not None:
        hdr += f"constant uint ACC_C = {acc};\n"
    return mx.fast.metal_kernel(
        name=f"{name}_H{H}_T{T}" + (f"_A{acc}" if acc else ""),
        input_names=["r", "w", "k", "v", "a", "b", "h_in"],
        output_names=["out", "h_out", "sa_out", "h_checkpoints"],
        header=hdr, source=src,
    )


def run(kern, ins, B, T, H, D):
    return kern(
        inputs=ins,
        grid=(B * H, D, 1), threadgroup=(1, D, 1),
        output_shapes=[(B, T, H, D), (B, H, D, D), (B, T, H, D),
                       (B, H, T // CHUNK, D, D)],
        output_dtypes=[mx.float32] * 4,
    )


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
    ins = [r, w, k, v, a, b, h_in]

    variants = [
        ("v0 current",          build("fwd_v0", _V0, H, T),            None),
        ("v1 +tg staging",      build("fwd_v1", _V1, H, T),            None),
        ("v2 +4-way ILP",       build("fwd_v2", _VSPLIT, H, T, 4),     4),
        ("v3 +8-way ILP",       build("fwd_v3", _VSPLIT, H, T, 8),     8),
    ]

    ref = None
    base_ms = None
    print(f"\n  {'variant':<18} {'time ms':>9} {'speedup':>9} {'max err':>11}")
    for label, kern, _acc in variants:
        try:
            res = run(kern, ins, B, T, H, D)
            mx.eval(res)
        except Exception as e:
            print(f"  {label:<18}  FAILED: {type(e).__name__}: {e}")
            continue
        if ref is None:
            ref, err = res, 0.0
        else:
            err = max(mx.abs(x - y).max().item() for x, y in zip(ref, res))
        ms = timed(lambda: run(kern, ins, B, T, H, D))
        if base_ms is None:
            base_ms = ms
        print(f"  {label:<18} {ms:>9.1f} {base_ms/ms:>8.2f}x {err:>11.2e}")


if __name__ == "__main__":
    main()
