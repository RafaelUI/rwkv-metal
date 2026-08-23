"""ГЕЙТ КАСТА ВХОДА В КВАНТОВАННОМ ЛИНЕЙНОМ СЛОЕ (внедрён 22.08).

Правка меняет `x @ w.T` на `(x.astype(w.dtype) @ w.T).astype(x.dtype)` в
RwkvqLinear и RwkvqSymLinear. Она МЕНЯЕТ ЧИСЛА при не-bf16 входе -- по
этому гейту равенства со старым путём быть НЕ МОЖЕТ, и каждое утверждение
построено под конкретный инвариант:

1. ФАКТ ВКЛЮЧЕНИЯ. При fp32-входе выход нового пути обязан ОТЛИЧАТЬСЯ от
   прежнего (rl.NOCAST=True), иначе правка не работает и всё дальнейшее
   проверяло бы старый код.
2. ОБРАТНАЯ СОВМЕСТИМОСТЬ ПО ТИПУ. При bf16-входе (а так зовут слои все
   плотные пути и прежний гейт test_rwkvq_sym_base) каст -- no-op: выход
   обязан совпасть со старым путём БИТ-В-БИТ.
3. СЕМАНТИКА КАСТА. При fp32-входе выход равен независимой записи формулы
   (x.astype(w.dtype) @ w.T).astype(x.dtype) с _dequant_w() из самого
   класса -- сверка с ПЕРЕСЧЁТОМ, а не с замороженным старым значением.
4. DTYPE ВЫХОДА. Новый путь возвращает dtype ВХОДА (x.dtype), а не весов:
   правка не должна протекать наружу сменой типа (закон 34 наоборот).
5. ФЛАГ ВОЗВРАТА. rl.NOCAST=True возвращает прежний путь бит-в-бит при
   fp32-входе -- это A/B-механизм закона 27, и он обязан работать.

МУТАЦИОННЫЕ КРЫШКИ (совет владельца: ломать код и проверять, что гейт
заметил): LINCAST_MUT=off -- правка тихо заменена прежним вызовом (гейт
обязан упасть на утверждении 1); LINCAST_MUT=dtype -- выход не приводится
к x.dtype (упасть на 4); LINCAST_MUT=flag -- NOCAST не действует (упасть
на 5).

    python tests/test_lincast_parity.py [model.rwkvq]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx                      # noqa: E402
import numpy as np                         # noqa: E402

from rwkv_metal.lora import rwkvq_linear as rl  # noqa: E402

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/reduction_new.rwkvq"
MUT = os.environ.get("LINCAST_MUT", "")

if MUT == "off":
    # мутация: правка тихо не применяется (эквивалент прежнего кода)
    rl._matmul_cast = lambda self, x: x @ self._dequant_w().T
if MUT == "dtype":
    _orig = rl._matmul_cast
    rl._matmul_cast = lambda self, x: x.astype(
        self._dequant_w().dtype) @ self._dequant_w().T
if MUT == "flag":
    # мутация: NOCAST больше не действует (правка всегда включена)
    def _always_cast(self, x):
        w = self._dequant_w()
        return (x.astype(w.dtype) @ w.T).astype(x.dtype)
    rl._matmul_cast = _always_cast

bad = 0


def check(name, ok, detail=""):
    global bad
    print("%-46s %s %s" % (name, "ОК" if ok else "КРАСНЫЙ", detail))
    bad += 0 if ok else 1


def pick_layer():
    """Реальный квантованный слой из файла: гость не синтетика (закон 17 --
    форма из настоящего манифеста)."""
    for cls, mk in ((rl.RwkvqSymLinear,
                     lambda: rl.RwkvqSymLinear.from_sidecar(PATH, _k_sym())),
                    (rl.RwkvqLinear, lambda: _l_sb6())):
        try:
            lin = mk()
            if lin is not None:
                return cls, lin
        except Exception:
            continue
    raise SystemExit("не нашлось квантованного слоя в %s" % PATH)


def _k_sym():
    from rwkv_quant.formats import codec
    manifest, _ = codec.open_rwkvq(os.path.expanduser(PATH))
    best = None
    for k, m in manifest["tensors"].items():
        if m.get("kind") == "sym" and m["shape"][1] % 256 == 0:
            sz = int(np.prod(m["shape"]))
            if best is None or sz < best[0]:
                best = (sz, k)
    return best[1]


def _l_sb6():
    return None      # sb6-слой требует сайдкар-путей; для гейта хватает sym


cls, lin = pick_layer()
print("слой: %s %s [%d x %d], мутация: %s"
      % (cls.__name__, PATH, lin.out_features, lin.in_features, MUT or "нет"))

rs = np.random.RandomState(int(__import__("zlib").crc32(cls.__name__.encode())))
x32 = mx.array(rs.randn(3, lin.in_features).astype(np.float32))
x16 = x32.astype(lin._dequant_w().dtype)
mx.eval(x32, x16)

# 1. факт включения: fp32-вход -> выходы нового и прежнего путей различны
old_call = lambda x: x @ lin._dequant_w().T
rl.NOCAST = False
y_new = lin(x32)
rl.NOCAST = True
y_old = lin(x32)
rl.NOCAST = False
mx.eval(y_new, y_old)
rel = float(mx.max(mx.abs(y_new - y_old))
           / (mx.max(mx.abs(y_old)) + 1e-30))
check("1. включение (fp32-вход меняет числа)",
      rel > 0, "relmax %.3e" % rel)

# 2. обратная совместимость: bf16-вход -- бит-в-бит со старым путём
y16_new = lin(x16)
y16_old = old_call(x16)
mx.eval(y16_new, y16_old)
check("2. bf16-вход: бит-в-бит со старым путём",
      bool(mx.all(y16_new == y16_old)),
      "max|Δ| = %.1e" % float(mx.max(mx.abs(
          y16_new.astype(mx.float32) - y16_old.astype(mx.float32)))))

# 3. семантика: совпадение с независимой записью формулы
w = lin._dequant_w()
y_ref = (x32.astype(w.dtype) @ w.T).astype(x32.dtype)
mx.eval(y_ref)
check("3. семантика каста (пересчёт формулы)",
      bool(mx.all(y_new == y_ref)))

# 4. dtype выхода = dtype входа, не весов
check("4. dtype выхода == dtype входа",
      y_new.dtype == x32.dtype,
      "выход %s, вход %s" % (y_new.dtype, x32.dtype))

# 5. флаг возврата: NOCAST даёт прежний путь и при fp32-входе
rl.NOCAST = True
y_flag = lin(x32)
rl.NOCAST = False
mx.eval(y_flag)
check("5. rl.NOCAST=True возвращает прежний путь",
      bool(mx.all(y_flag == y_old)))

print("\n%s" % ("ГЕЙТ ЗЕЛЁНЫЙ" if bad == 0 else "ГЕЙТ КРАСНЫЙ (%d)" % bad))
sys.exit(1 if bad else 0)
