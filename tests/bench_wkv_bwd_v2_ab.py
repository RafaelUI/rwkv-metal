"""A/B ЯДРА BACKWARD: v1 (межпотоковые редукции) против v2 (порт Бо Пэна).

Оба ядра зовутся НАПРЯМУЮ на ОДНИХ входах, чередованием с рандомизацией
порядка (законы 1, 24). Это микрозамер одного слоя, и он ЗАВЫШАЕТ абсолюты
против доли в шаге (каждый вызов обрамлён mx.eval, тогда как 24 слоя шага
дают перекрываемую работу) -- закон 29. Арбитр -- сквозной шаг.

    python tests/bench_wkv_bwd_v2_ab.py [T] [раундов]
"""
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))

import mlx.core as mx                      # noqa: E402
import numpy as np                         # noqa: E402

from rwkv_metal.kernel.wkv7_checkpoint import (  # noqa: E402
    _get_ckpt_fwd, _get_ckpt_bwd, _get_ckpt_bwd_v2, CHUNK)

T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
B, H, D = 1, 32, 64
N = T // CHUNK


def main():
    rs = np.random.RandomState(7)

    def a(shape, s=1.0):
        return mx.array((rs.randn(*shape) * s).astype(np.float32))
    w = mx.array((0.545 + 0.455 * rs.rand(B, T, H, D)).astype(np.float32))
    r, k, v = a((B, T, H, D), .5), a((B, T, H, D), .5), a((B, T, H, D), .5)
    aa, bb = a((B, T, H, D), .3), a((B, T, H, D), .3)
    h_in = mx.zeros((B, H, D, D))
    d_out, d_h_out = a((B, T, H, D), .02), a((B, H, D, D), .01)

    fwd = _get_ckpt_fwd(H, T)
    out, h_out, sa_fwd, h_ck = fwd(
        inputs=[r, w, k, v, aa, bb, h_in], grid=(B * H, D, 1),
        threadgroup=(1, D, 1),
        output_shapes=[(B, T, H, D), (B, H, D, D), (B, T, H, D),
                       (B, H, N, D, D)], output_dtypes=[mx.float32] * 4)
    mx.eval(out, h_out, sa_fwd, h_ck)

    ins = [r, w, k, v, aa, bb, h_ck, sa_fwd, d_out, d_h_out]
    kw = dict(grid=(B * H * D, 1, 1), threadgroup=(D, 1, 1),
              output_shapes=[(B, T, H, D)] * 6 + [(B, H, D, D)],
              output_dtypes=[mx.float32] * 7)

    def run_fwd():
        return fwd(inputs=[r, w, k, v, aa, bb, h_in], grid=(B * H, D, 1),
                   threadgroup=(1, D, 1),
                   output_shapes=[(B, T, H, D), (B, H, D, D), (B, T, H, D),
                                  (B, H, N, D, D)],
                   output_dtypes=[mx.float32] * 4)

    VAR = {"bwd v1": lambda: _get_ckpt_bwd(H, T)(inputs=ins, **kw),
           "bwd v2": lambda: _get_ckpt_bwd_v2(H, T)(inputs=ins, **kw),
           "fwd": run_fwd}
    # КОНТРОЛЬ, ЧТО МЕРЯЮТСЯ РАЗНЫЕ ЯДРА: если бы v2 молча звал v1, числа
    # совпали бы, и "выигрыша нет" читалось бы как результат.
    g1, g2 = VAR["bwd v1"](), VAR["bwd v2"]()
    mx.eval(*g1, *g2)
    rel = max(float(mx.max(mx.abs(x1 - x2))) /
              max(float(mx.max(mx.abs(x1))), 1e-30) for x1, x2 in zip(g1, g2))
    assert 0 < rel < 1e-3, "v1 и v2 не различаются (relmax %.2e) -- мерится одно ядро" % rel

    acc = {n: [] for n in VAR}
    rng = np.random.RandomState(23)
    for rnd in range(ROUNDS):
        order = list(VAR)
        rng.shuffle(order)
        for nm in order:
            mx.eval(*VAR[nm]())
            t0 = time.perf_counter()
            for _ in range(3):
                mx.eval(*VAR[nm]())
            acc[nm].append((time.perf_counter() - t0) / 3)

    print("B=%d, T=%d, H=%d, D=%d, раундов %d (relmax v1~v2 %.2e)"
          % (B, T, H, D, ROUNDS, rel))
    m = {n: float(np.median(acc[n])) * 1e3 for n in VAR}
    for n in ("fwd", "bwd v1", "bwd v2"):
        sp = (max(acc[n]) - min(acc[n])) / np.median(acc[n]) * 100
        print("%-8s %7.2f мс (разброс %.1f%%)" % (n, m[n], sp))
    print("v2/v1 = x%.2f, выигрыш %+.2f мс на слой" %
          (m["bwd v1"] / m["bwd v2"], m["bwd v1"] - m["bwd v2"]))
    print("на шаг (24 слоя, 2 fwd + 1 bwd): %.0f -> %.0f мс"
          % (24 * (2 * m["fwd"] + m["bwd v1"]),
             24 * (2 * m["fwd"] + m["bwd v2"])))


if __name__ == "__main__":
    main()
