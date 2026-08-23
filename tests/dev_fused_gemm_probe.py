"""Прототип fused dequant+GEMM для sym: корректность + скорость на РЕАЛЬНЫХ
тензорах файла, все четыре формы пресета, fwd и vjp.

Эталон корректности -- ровно то, что делает RwkvqSymLinear.__call__
(каст входа 22.08 + деквант bf16 + плотный матмул). Отличие fused -- только
порядок суммирования, порог relmax 3e-3 (прецедент test_sym_kernel).

    python tests/dev_fused_gemm_probe.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))

import mlx.core as mx                      # noqa: E402
import numpy as np                         # noqa: E402

from rwkv_metal.lora.rwkvq_linear import RwkvqSymLinear  # noqa: E402

PATH = "/tmp/reduction_new.rwkvq"
KEYS = [
    "blocks.0.att.receptance.weight",   # [2048, 2048] @8
    "blocks.0.ffn.key.weight",          # [8192, 2048] @6
    "blocks.0.ffn.value.weight",        # [2048, 8192] @6
    "head.weight",                      # [65536, 2048] @8
]
T = 512


def relmax(a, b):
    mx.eval(a, b)
    a = np.array(a.astype(mx.float32)).astype(np.float64)
    b = np.array(b.astype(mx.float32)).astype(np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def main():
    rs = np.random.RandomState(7)
    for key in KEYS:
        lin = RwkvqSymLinear.from_sidecar(PATH, key)
        sym = lin._sym
        IN, OUT = sym.in_features, sym.out_features
        x = mx.array((rs.randn(T, IN) * 0.5).astype(np.float32)).astype(mx.bfloat16)

        out = sym.gemm_fused(x)
        ref = x @ sym._dequant_w(mx.bfloat16).T
        mx.eval(out, ref)
        r_fwd = relmax(out, ref)

        dy = mx.array((rs.randn(T, OUT) * 0.02).astype(np.float32)).astype(mx.bfloat16)
        dv = sym.gemm_fused_vjp(dy)
        dref = dy @ sym._dequant_w(mx.bfloat16)
        mx.eval(dv, dref)
        r_vjp = relmax(dv, dref)

        flops = 2 * T * IN * OUT
        ts = {"fused": [], "chain": []}
        for _ in range(7):
            for name in ("fused", "chain"):
                if name == "fused":
                    o = sym.gemm_fused(x); mx.eval(o)
                    t0 = time.perf_counter()
                    for _ in range(3):
                        o = sym.gemm_fused(x); mx.eval(o)
                else:
                    o = x @ sym._dequant_w(mx.bfloat16).T; mx.eval(o)
                    t0 = time.perf_counter()
                    for _ in range(3):
                        o = x @ sym._dequant_w(mx.bfloat16).T; mx.eval(o)
                ts[name].append((time.perf_counter() - t0) / 3)
        m_f, m_c = np.median(ts["fused"]), np.median(ts["chain"])
        print(f"{key:36s} [{OUT},{IN}]@{sym.bits} T={T}: "
              f"relmax fwd {r_fwd:.2e} vjp {r_vjp:.2e} | "
              f"fused {m_f*1e3:6.2f} мс ({flops/m_f/1e12:.2f} ТФ) "
              f"chain {m_c*1e3:6.2f} мс, x{m_c/m_f:.2f}")
        assert r_fwd < 3e-3 and r_vjp < 6e-3, "за порогом"  # vjp: K до 8192, шум порядка суммирования


if __name__ == "__main__":
    main()
