"""ГЕЙТ ПРЯМОЙ bf16-ВЕТКИ ДЕКВАНТА sym (23.08).

`RwkvqSymLinear._dequant_w` теперь зовёт кернель с выходом bf16 напрямую,
а прежде было `_dequant_w(mx.float32).astype(mx.bfloat16)`. Правка меняет
ТОЛЬКО точку приведения типа: арифметика кернеля -- тот же fp32-текст, и
оба пути округляют одно и то же fp32-значение до bf16 по ближайшему.
Значит здесь законно требовать РАВЕНСТВО, и на ВСЕХ sym-тензорах
настоящего файла -- включая обе битности (6 и 8), у каждой свой кернель.

Побочно сверяется вовлечение: выходы обязаны существовать и иметь dtype
bf16 (иначе гейт сравнивал бы fp32 с fp32 и был зелёным всегда).

    python tests/test_sym_dequant_bf16.py [rwkvq]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx                      # noqa: E402

from rwkv_quant.formats import codec       # noqa: E402
from rwkv_metal.lora.rwkvq_linear import RwkvqSymLinear  # noqa: E402

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/reduction_new.rwkvq"

manifest, _ = codec.open_rwkvq(os.path.expanduser(PATH))
keys = [k for k, m in manifest["tensors"].items()
        if m.get("kind") == "sym" and m["shape"][1] % 256 == 0]
bits_seen = sorted({manifest["tensors"][k]["bits"] for k in keys})
print("sym-тензоров: %d, битности: %s" % (len(keys), bits_seen))
assert bits_seen == [6, 8], "в файле ожидались ОБЕ битности -- иначе гейт " \
    "не исполняет одну из веток кернеля (закон 31)"

bad = 0
for k in keys:
    lin = RwkvqSymLinear.from_sidecar(PATH, k)
    new = lin._dequant_w()                              # прямая bf16
    old = lin._sym._dequant_w(mx.float32).astype(mx.bfloat16)
    mx.eval(new, old)
    if new.dtype != mx.bfloat16:
        print("%s: выход %s, ожидался bfloat16 -- вовлечение НЕ подтверждено"
              % (k, new.dtype))
        bad += 1
        continue
    if not bool(mx.all(new == old)):
        d = int(mx.sum(new != old))
        rel = float(mx.max(mx.abs(new.astype(mx.float32)
                                  - old.astype(mx.float32)))
                   / (mx.max(mx.abs(old.astype(mx.float32))) + 1e-30))
        print("%s: РАСХОЖДЕНИЕ %d элементов, relmax %.3e" % (k, d, rel))
        bad += 1
    del lin, new, old
    mx.clear_cache()

print("\n%s: %d/%d тензоров бит-в-бит"
      % ("ГЕЙТ ЗЕЛЁНЫЙ" if bad == 0 else "ГЕЙТ КРАСНЫЙ", len(keys) - bad,
         len(keys)))
sys.exit(1 if bad else 0)
