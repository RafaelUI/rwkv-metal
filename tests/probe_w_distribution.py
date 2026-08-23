"""РАСПРЕДЕЛЕНИЕ w В НАСТОЯЩЕМ ШАГЕ: заходит ли модель в зону, где
обратная реконструкция hp = (h - v·k - sa·b) / w плохо обусловлена.

`test_wkv7_backward.py` сверяет все семь градиентов с einsum-эталоном при
пороге 1e-5, но диапазон `w` в нём выбран СОЗНАТЕЛЬНО без значений около
нуля (~0.58-0.99). Порт кернеля Бо Пэна безопасен по скорости, если модель
тоже не ходит около нуля; если ходит -- это вопрос устойчивости, и сначала
надо расширять гейт.

По ФОРМУЛЕ модели (`rwkv7_x070`, строка помечена "decay"):
    w = exp(-0.606531 * sigmoid(...)),
то есть структурно w ∈ [exp(-0.606531), 1] = [0.5454, 1] -- около нуля
находиться не может. Этот пробник проверяет это ЗАМЕРОМ на обученной 1.5B
во время настоящего forward по корпусу, а не выводом из формулы: формула
не исключает ни вырожденных значений после арифметики в bf16, ни того,
что реальный хвост плотнее теоретической границы.

    python tests/probe_w_distribution.py [окон] [T]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_WIN = int(sys.argv[1]) if len(sys.argv) > 1 else 32
T = int(sys.argv[2]) if len(sys.argv) > 2 else 512
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/sr_qlora_%d.npz" % T)
RWKVQ = os.environ.get("RWKVQ_BASE", "/tmp/reduction_new.rwkvq")

import mlx.core as mx                      # noqa: E402
import rwkv_metal.model.rwkv7_x070 as mod  # noqa: E402

# Порог интереса: ниже него деление на w в реконструкции усиливает
# относительную ошибку эталона сильнее, чем 1/0.5454 ~ 1.83x (лучшая
# граница, достижимая при данной формуле затухания).
FLOOR = float(np.exp(-0.606531))

stats = []          # (layer_id, min, q01, q1, median) на вызов
_orig = mod.wkv7


def wrapped(r, w, k, v, a, b, training=True, state=None, **kw):
    wf = np.asarray(w.astype(mx.float32))   # np.asarray материализует ленивый массив
    q = np.percentile(wf, [0.1, 1, 50])
    stats.append((float(wf.min()), float(q[0]), float(q[1]), float(q[2])))
    return _orig(r, w, k, v, a, b, training=training, state=state, **kw)


mod.wkv7 = wrapped

from bench_qlora_arms import build        # noqa: E402  (тем же путём, что плечи)

d = np.load(CORPUS)
ev = d["eval"][:N_WIN]
print("корпус %s, окон %d, T=%d, теоретический пол w = %.4f"
      % (os.path.basename(CORPUS), N_WIN, T, FLOOR), flush=True)

model, tag = build("reduction")
model._grad_ckpt = True
print("база: %s" % tag, flush=True)

for i in range(N_WIN):
    x = mx.array(ev[i:i + 1, :-1].astype(np.int32))
    y = mx.array(ev[i:i + 1, 1:].astype(np.int32))
    l = model.loss(x, y)
    mx.eval(l)
mod.wkv7 = _orig

a = np.array(stats)
print("\nвызовов кернеля (слой×окон): %d" % len(a))
print("min  по всем вызовам: %.6f" % a[:, 0].min())
print("q0.1 минимум по вызовам: %.6f" % a[:, 1].min())
print("q1   минимум по вызовам: %.6f" % a[:, 2].min())
print("медиана (min/med/max):   %.4f / %.4f / %.4f"
      % (a[:, 3].min(), np.median(a[:, 3]), a[:, 3].max()))
gap = a[:, 0].min() - FLOOR
print("расстояние минимума до теоретического пола: %+.6f" % gap)
n_low = int((a[:, 0] < 0.5).sum())
print("вызовов с min < 0.5 (глубже пола формулы): %d" % n_low)
print("\nВЫВОД: модель %s в опасную область (w около нуля)"
      % ("НЕ заходит" if a[:, 0].min() >= FLOOR - 1e-6 and n_low == 0
         else "ЗАХОДИТ"))
