"""РАЗРЫВ 5.9x: тренировочный forward WKV против инференсного, НА ОДНИХ ВХОДАХ.

Записано: тренировочный forward WKV 321 мс против инференсного 54.6 мс на тех
же формах. Оба числа сняты СКВОЗНЫМИ инструментами (аблация шага обучения и
аблация префилла), то есть каждое включает свою обвязку. Здесь оба ядра
зовутся на ОДНОМ И ТОМ ЖЕ входе, чередованием в одном процессе, с
РАНДОМИЗАЦИЕЙ порядка вариантов внутри раунда (закон 24: последовательный
свип рисует эффекты, которых нет; корпус безвентиляторный, закон 25).

Лестница вариантов -- по ОДНОМУ отличию на ступень, поэтому разность соседних
ступеней есть цена ровно этого отличия:

  infer        -- рабочее ядро инференса как есть (2 выхода, одна цепочка FMA)
  infer+sa     -- оно же, но пишет sa_out [B,T,H,D]           (+1 выход)
  infer+sa+ck  -- оно же, но пишет ещё h_checkpoints [B,H,N,D,D] (+1 выход)
  ckpt_acc1    -- тренировочное ядро с ACC_C=1                 (= infer+sa+ck
                  по смыслу; расхождение здесь = разница ИСХОДНИКОВ, а не
                  идей, и служит контролем лестницы)
  ckpt         -- тренировочное ядро как есть (ACC_FWD=8)
  train()      -- обёртка wkv7_train: custom_function, astype, паддинг, h0

ВНИМАНИЕ, В КАКУЮ СТОРОНУ ВРЁТ ИНСТРУМЕНТ (закон 29). Это микрозамер ОДНОГО
слоя: 24 слоя настоящего шага дают независимую работу, которую GPU
перекрывает, а здесь каждый вызов обрамлён mx.eval. Значит абсолюты тут
ЗАВЫШЕНЫ против доли в шаге, а вот ОТНОШЕНИЕ двух ядер на одном входе --
ровно то, ради чего замер и ставится. Арбитром для «сколько это стоит в шаге»
остаётся сквозная аблация.

    python bench_wkv_train_vs_infer_ab.py [T] [раундов] [B] [H]
"""
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))

import mlx.core as mx
import numpy as np

T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 7
B = int(sys.argv[3]) if len(sys.argv) > 3 else 1
H = int(sys.argv[4]) if len(sys.argv) > 4 else 32
D = 64
CHUNK = 16
REPS = 5          # вызовов в бёрсте (окно короткое -- троттлинг, закон 25)


def swap_mb():
    out = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True,
                         env={**os.environ, "LC_ALL": "C"}).stdout
    for tok in out.split():
        if tok.startswith("used"):
            pass
    # формат: total = 2048.00M  used = 736.12M  free = ...
    parts = out.replace("=", " ").split()
    return float(parts[parts.index("used") + 1].rstrip("M"))


# ─────────────────────────── варианты ядер ──────────────────────────────────
# Ступени лестницы собираются ОДНИМ генератором: отличие между ступенями --
# значения флагов, а не переписанный текст. Иначе разность двух ступеней
# включала бы случайные расхождения редакций (закон 27: A/B делается подменой
# одного параметра, а не версией файла).

def _make_kernel(name, acc, write_sa, write_ck, n_chunks):
    hdr = f"""
constant uint HEAD_SIZE_C = {D};
constant uint T_C         = {T};
constant uint CHUNK_C     = {CHUNK};
constant uint N_CHUNKS_C  = {n_chunks};
constant uint H_C         = {H};
constant uint ACC_C       = {acc};
"""
    sa_line = "sa_out[base+dv] = sa;" if write_sa else ""
    ck_block = r"""
        uint ckb = ((bi*H_C+hi)*N_CHUNKS_C + c)*HEAD_SIZE_C*HEAD_SIZE_C
                   + dv*HEAD_SIZE_C;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_checkpoints[ckb+dk] = h_row[dk];
""" if write_ck else ""
    src = r"""
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
            __SA__

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
__CK__
    }
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_out[hb+dk] = h_row[dk];
""".replace("__SA__", sa_line).replace("__CK__", ck_block)

    outs = ["out", "h_out"]
    if write_sa:
        outs.append("sa_out")
    if write_ck:
        outs.append("h_checkpoints")
    return mx.fast.metal_kernel(
        name=name,
        input_names=["r", "w", "k", "v", "a", "b", "h_in"],
        output_names=outs,
        header=hdr, source=src,
    ), outs


def main():
    from rwkv_metal.kernel.wkv7 import wkv7_infer, wkv7_train
    from rwkv_metal.kernel import wkv7_checkpoint as ck

    assert T % CHUNK == 0, "T должно делиться на CHUNK"
    N = T // CHUNK

    rs = np.random.RandomState(0xC0FFEE)
    shape = (B, T, H, D)

    def rnd(scale=1.0):
        return mx.array((rs.randn(*shape) * scale).astype(np.float32))

    # w -- множитель затухания, обязан быть в (0, 1): реконструкция в backward
    # делит на него, а форвард при w>1 расходится. Значения из настоящего
    # диапазона exp(-exp(x)), а не N(0,1).
    r_ = rnd(); k_ = rnd(); v_ = rnd(); a_ = rnd(0.3); b_ = rnd(0.3)
    w_ = mx.array(np.exp(-np.exp(rs.randn(*shape) * 0.5 - 1.0)).astype(np.float32))
    h0 = mx.zeros((B, H, D, D), dtype=mx.float32)
    ins = [r_, w_, k_, v_, a_, b_, h0]
    mx.eval(*ins)

    kern_full = ck._get_ckpt_fwd(H, T)          # НАСТОЯЩЕЕ ядро обучения
    kern_sa, o_sa = _make_kernel("ab_infer_sa", 1, True, False, N)
    kern_sac, o_sac = _make_kernel("ab_infer_sa_ck", 1, True, True, N)
    kern_a1, o_a1 = _make_kernel("ab_ckpt_acc1", 1, True, True, N)  # = sac
    kern_a8, o_a8 = _make_kernel("ab_ckpt_acc8", 8, True, True, N)

    sh_out = [(B, T, H, D), (B, H, D, D), (B, T, H, D), (B, H, N, D, D)]
    launch = dict(grid=(B * H, D, 1), threadgroup=(1, D, 1))

    def v_infer():
        return list(wkv7_infer(r_, w_, k_, v_, a_, b_, h0))

    def v_kern(kern, nout):
        return kern(inputs=ins, output_shapes=sh_out[:nout],
                    output_dtypes=[mx.float32] * nout, **launch)

    VARIANTS = [
        ("infer (как есть)", v_infer),
        ("infer +sa_out", lambda: v_kern(kern_sa, 3)),
        ("infer +sa +ckpts", lambda: v_kern(kern_sac, 4)),
        ("ckpt ACC=1", lambda: v_kern(kern_a1, 4)),
        ("ckpt ACC=8", lambda: v_kern(kern_a8, 4)),
        ("ckpt НАСТОЯЩЕЕ", lambda: v_kern(kern_full, 4)),
        ("wkv7_train()", lambda: [wkv7_train(r_, w_, k_, v_, a_, b_)]),
    ]

    # прогрев: компиляция всех ядер вне окна замера
    for name, fn in VARIANTS:
        mx.eval(*fn())
    mx.eval(*[f() for _, f in VARIANTS][0])

    acc = {n: [] for n, _ in VARIANTS}
    sw0 = swap_mb()
    order = list(range(len(VARIANTS)))
    rng = random.Random(20260817)

    for rnd_i in range(ROUNDS):
        rng.shuffle(order)                      # закон 24
        for idx in order:
            name, fn = VARIANTS[idx]
            t1 = time.perf_counter()
            for _ in range(REPS):
                mx.eval(*fn())
            acc[name].append((time.perf_counter() - t1) / REPS)
        print("  раунд %d (порядок %s)" % (rnd_i, order), flush=True)
    sw1 = swap_mb()

    # контроль равенства: все ступени обязаны давать ОДИН И ТОТ ЖЕ out, иначе
    # сравниваются разные вычисления. ACC=8 переассоциирует суммы, поэтому для
    # него порог, а не равенство.
    ref = v_infer()[0]
    print("\nконтроль выхода против infer:")
    for name, fn in VARIANTS[1:]:
        o = fn()[0]
        d = float(mx.max(mx.abs(o - ref)) / (mx.max(mx.abs(ref)) + 1e-30))
        print("  %-20s relmax %.3e %s" % (name, d, "OK" if d < 1e-5 else "!!!"))

    print("\n%-20s%10s%10s%12s" % ("вариант", "мс", "разброс", "x к infer"))
    base = float(np.median(acc["infer (как есть)"]))
    for name, _ in VARIANTS:
        m = float(np.median(acc[name]))
        sp = (max(acc[name]) - min(acc[name])) / m * 100
        print("%-20s%10.3f%9.1f%%%12.2f" % (name, m * 1e3, sp, m / base))

    print("\nформы: B=%d T=%d H=%d D=%d, N_CHUNKS=%d" % (B, T, H, D, N))
    print("трафик выходов на вызов: out %.1f МБ, sa %.1f МБ, ckpts %.1f МБ"
          % (B * T * H * D * 4 / 1e6, B * T * H * D * 4 / 1e6,
             B * H * N * D * D * 4 / 1e6))
    print("своп %.0f -> %.0f МБ (%s)"
          % (sw0, sw1, "замер валиден" if sw1 - sw0 < 1 else "НЕВАЛИДЕН"))
    print("на 24 слоя: infer %.1f мс, ckpt %.1f мс"
          % (base * 24e3, float(np.median(acc["ckpt НАСТОЯЩЕЕ"])) * 24e3))


if __name__ == "__main__":
    main()
