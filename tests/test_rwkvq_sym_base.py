"""ГЕЙТ QLoRA-БАЗЫ НА sym-РАСКЛАДКЕ.

Проверяет три вещи, ради которых порт и делался.

1. ФАКТ ВКЛЮЧЕНИЯ. Диспетчер обязан отдать RwkvqSymLinear на sym-ключе и
   RwkvqNativeLinear на sb6-ключе. Без этой проверки гейт был бы зелёным
   и в случае, когда sym-ветка не строится вовсе: прежде такие тензоры
   просто не попадали в манифест загрузчика, и падало это уже у
   потребителя -- KeyError с именем тензора и без диагноза.

2. ЧИСЛА. Выход слоя против `codec.dequant_key` -- нормативной читалки
   формата. Требуется РАВЕНСТВО, а не порог: fp32-деквант бит-в-бит
   совпадает с codec (гейт rwkv-quant/test_sym_dequant_fp32), приведение
   к bf16 детерминировано, матмул тот же самый. Порог здесь означал бы,
   что мы не знаем, что сравниваем.

3. ОТРИЦАТЕЛЬНЫЙ КОНТРОЛЬ. Проверка покрытия обязана РУГАТЬСЯ на ключ,
   которого в файле нет квантованным, и называть раскладки. Гейт, который
   не умеет краснеть, ничего не проверяет.

    python tests/test_rwkvq_sym_base.py [sym.rwkvq] [sb6.rwkvq]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from rwkv_quant.formats import codec  # noqa: E402

from rwkv_metal.lora.add_rwkvq import (  # noqa: E402
    _backend_for, _check_quantized_coverage)
from rwkv_metal.lora.rwkvq_linear import RwkvqSymLinear  # noqa: E402
from rwkv_metal.lora.rwkvq_native import RwkvqNativeLinear  # noqa: E402

SYM = sys.argv[1] if len(sys.argv) > 1 else "/tmp/reduction_new.rwkvq"
SB6 = sys.argv[2] if len(sys.argv) > 2 else "/tmp/champion_v2.rwkvq"
bad = 0


def pick(path, kind, n=2):
    """Самые мелкие ключи КАЖДОЙ битности, а не просто самые мелкие.

    Первая редакция брала n самых мелких -- и все три оказались @8, где
    коды это байты и распаковки нет вовсе. Шестибитный путь с двумя
    битплоскостями и блок-локальным порядком колонок (l0,l1,h0,h1 /
    l2,l3,h2,h3) при этом не исполнялся НИ РАЗУ: гейт был бы зелёным и
    при полностью сломанном декодере нибблов. Голова и emb исключены не
    по важности, а по цене -- полгигабайта транзиента на сверку.
    """
    manifest, _ = codec.open_rwkvq(os.path.expanduser(path))
    by_bits = {}
    for k, m in manifest["tensors"].items():
        if m.get("kind") != kind or m["shape"][1] % 256:
            continue
        by_bits.setdefault(m["bits"], []).append((int(np.prod(m["shape"])), k))
    out = []
    for bits in sorted(by_bits):
        out += [k for _, k in sorted(by_bits[bits])[:n]]
    return out, manifest


# --- 1. диспетчер -----------------------------------------------------------
sym_keys, man_sym = pick(SYM, "sym")
sb6_keys, _ = pick(SB6, "sb6")
print(f"sym-файл: {len(sym_keys)} ключей "
      f"(битности {sorted({man_sym[chr(39)+chr(39)] if False else man_sym['tensors'][k]['bits'] for k in sym_keys})}), "
      f"sb6-файл: {len(sb6_keys)}")
for k in sym_keys:
    got = _backend_for(SYM, k, native=True)
    if got is not RwkvqSymLinear:
        bad += 1
        print(f"   ДИСПЕТЧЕР: {k} -> {got.__name__}, ожидался RwkvqSymLinear")
for k in sb6_keys:
    got = _backend_for(SB6, k, native=True)
    if got is not RwkvqNativeLinear:
        bad += 1
        print(f"   ДИСПЕТЧЕР: {k} -> {got.__name__}, ожидался RwkvqNativeLinear")
print(f"диспетчер: {'ОК' if bad == 0 else 'КРАСНЫЙ'}")

# --- 2. числа ---------------------------------------------------------------
manifest, buf = codec.open_rwkvq(os.path.expanduser(SYM))
rs = np.random.RandomState(3)
print(f"\n{'ключ':<38}{'бит':>5}{'max|dW|':>11}{'max|dY|':>11}")
for k in sym_keys:
    lin = RwkvqSymLinear.from_sidecar(SYM, k)
    ref_w = mx.array(codec.dequant_key(manifest, buf, k)).astype(mx.bfloat16)
    got_w = lin._dequant_w()
    dw = float(mx.abs(got_w.astype(mx.float32)
                      - ref_w.astype(mx.float32)).max())
    x = mx.array(rs.randn(2, lin.in_features).astype(np.float32)
                 ).astype(mx.bfloat16)
    dy = float(mx.abs((lin(x) - x @ ref_w.T).astype(mx.float32)).max())
    bad += (dw != 0.0) or (dy != 0.0)
    print(f"{k:<38}{manifest['tensors'][k]['bits']:>5}{dw:>11.3e}{dy:>11.3e}")
    del lin, ref_w, got_w
    mx.clear_cache()

# --- 3. отрицательный контроль ---------------------------------------------
try:
    _check_quantized_coverage(SYM, {"blocks.0.att.НЕТ-ТАКОГО.weight"})
    print("\nОТРИЦАТЕЛЬНЫЙ КОНТРОЛЬ ПРОВАЛЕН: проверка покрытия промолчала")
    bad += 1
except ValueError as e:
    print(f"\nотрицательный контроль: ругнулась, как и должна --\n   {e}")
# и положительный: настоящие ключи проверку проходят
try:
    _check_quantized_coverage(SYM, set(sym_keys))
    print("положительный контроль: настоящие ключи приняты")
except Exception as e:  # noqa: BLE001
    bad += 1
    print(f"ПОЛОЖИТЕЛЬНЫЙ КОНТРОЛЬ ПРОВАЛЕН: {type(e).__name__}: {e}")

print("\nГЕЙТ ЗЕЛЁНЫЙ" if bad == 0 else f"\nГЕЙТ КРАСНЫЙ ({bad})")
sys.exit(1 if bad else 0)
