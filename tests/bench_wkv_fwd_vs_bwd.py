"""ЧТО ИМЕННО СТОИТ 400 мс В WKV: forward, backward, и упираемся ли в занятость.

Записанное «WKV-скан ~400 мс/шаг (12%)» получено сквозной аблацией и не
разделено на forward и backward. Здесь оба ядра checkpoint-пути зовутся
НАПРЯМУЮ (без custom_function, паддинга и astype), чередованием с
рандомизацией порядка (законы 1, 24).

ГЛАВНЫЙ ВОПРОС -- ЗАНЯТОСТЬ. Кернель Бо Пэна параллелится по (H, B), и на
обучении с батчем в десятки блоков хватает. У нас B=1: fwd идёт сеткой
B*H = 32 threadgroup по D=64 потока, то есть 32 блока на 10-ядерный GPU.
Если время при росте B растёт СУБЛИНЕЙНО -- машина недогружена, и рычаг
это параллелизм (по T или по B), а не микрооптимизация математики.
Идеальная сублинейность (время не меняется до какого-то B) означала бы,
что скан на B=1 стоит ровно столько же, сколько на B=4, и весь запас
лежит в раскладке работы, а не в кернеле.

    python tests/bench_wkv_fwd_vs_bwd.py [T] [раундов]
"""
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))

import mlx.core as mx                      # noqa: E402
import numpy as np                         # noqa: E402

from rwkv_metal.kernel.wkv7_checkpoint import (  # noqa: E402
    _get_ckpt_fwd, _get_ckpt_bwd, CHUNK)

T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
H, D = 32, 64                              # 1.5B: n_embd 2048 / head 64
BATCHES = [1, 2, 4, 8]
N = T // CHUNK


def mk(B, seed):
    rs = np.random.RandomState(seed)

    def a(shape, scale=1.0):
        return mx.array((rs.randn(*shape) * scale).astype(np.float32))
    # w в рабочем диапазоне [0.545; 1] -- замерено probe_w_distribution;
    # нули или отрицательные тут дали бы кернель, которого в модели нет
    w = mx.array((0.545 + 0.455 * rs.rand(B, T, H, D)).astype(np.float32))
    return dict(
        r=a((B, T, H, D), 0.5), w=w, k=a((B, T, H, D), 0.5),
        v=a((B, T, H, D), 0.5), a=a((B, T, H, D), 0.3),
        b=a((B, T, H, D), 0.3), h_in=mx.zeros((B, H, D, D)),
        d_out=a((B, T, H, D), 0.02), d_h_out=mx.zeros((B, H, D, D)))


def main():
    print("T=%d, H=%d, D=%d, CHUNK=%d, раундов %d" % (T, H, D, CHUNK, ROUNDS))
    print("%-4s %6s %10s %10s %12s %12s" % (
        "B", "блоков", "fwd мс", "bwd мс", "fwd мс/посл", "bwd мс/посл"))
    rng = np.random.RandomState(23)
    base = {}
    for B in BATCHES:
        d = mk(B, 7)
        fwd, bwd = _get_ckpt_fwd(H, T), _get_ckpt_bwd(H, T)

        def run_fwd(_d=d, _B=B, _f=fwd):
            return _f(inputs=[_d[n] for n in
                              ("r", "w", "k", "v", "a", "b", "h_in")],
                      grid=(_B * H, D, 1), threadgroup=(1, D, 1),
                      output_shapes=[(_B, T, H, D), (_B, H, D, D),
                                     (_B, T, H, D), (_B, H, N, D, D)],
                      output_dtypes=[mx.float32] * 4)

        out, h_out, sa_fwd, h_ckpts = run_fwd()
        mx.eval(out, h_out, sa_fwd, h_ckpts)

        def run_bwd(_d=d, _B=B, _b=bwd, _ck=h_ckpts, _sa=sa_fwd):
            return _b(inputs=[_d["r"], _d["w"], _d["k"], _d["v"], _d["a"],
                              _d["b"], _ck, _sa, _d["d_out"], _d["d_h_out"]],
                      grid=(_B * H * D, 1, 1), threadgroup=(D, 1, 1),
                      output_shapes=[(_B, T, H, D)] * 6 + [(_B, H, D, D)],
                      output_dtypes=[mx.float32] * 7)

        ts = {"fwd": [], "bwd": []}
        for rnd in range(ROUNDS):
            order = ["fwd", "bwd"] if rng.rand() < 0.5 else ["bwd", "fwd"]
            for nm in order:
                fn = run_fwd if nm == "fwd" else run_bwd
                mx.eval(*fn())                        # прогрев
                t0 = time.perf_counter()
                for _ in range(3):
                    mx.eval(*fn())
                ts[nm].append((time.perf_counter() - t0) / 3)
        mf, mb = np.median(ts["fwd"]) * 1e3, np.median(ts["bwd"]) * 1e3
        base.setdefault("f", mf); base.setdefault("b", mb)
        print("%-4d %6d %10.2f %10.2f %12.2f %12.2f" % (
            B, B * H, mf, mb, mf / B, mb / B))
        del d, out, h_out, sa_fwd, h_ckpts
        mx.clear_cache()
    print("\nЕсли мс/последовательность падает с ростом B -- машина на B=1 "
          "НЕДОГРУЖЕНА,\nи рычаг -- раскладка работы, а не математика ядра.")


if __name__ == "__main__":
    main()
