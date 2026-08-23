"""ГЕЙТ fused dequant+GEMM для sym (приоритет 1, 23.08).

Пять утверждений:
  1. ФАКТ ВКЛЮЧЕНИЯ: счётчик FUSED_CALLS растёт, а _dequant_w на fused-пути
     НЕ вызывается (иначе «чистый результат» читался бы у пути, который
     вовсе не проверялся -- та же ловушка, что у квантованных LoRA-веток).
  2. ПОБИТОВОСТЬ ЦЕПOЧКИ ЗНАЧЕНИЙ: fwd/vjp на представителях ВСЕХ четырёх
     форм пресета (обе битности) против старого пути, relmax < 3e-3 (fwd) /
     6e-3 (vjp@6 при K=8192 -- порядок суммирования; прецедент порога --
     test_sym_kernel).
  3. СКВОЗНОЙ ШАГ: лосс и градиенты адаптеров против старого пути
     (фактически совпали бит-в-бит; порог запасный).
  4. ФЛАГ-РЕВЕРС: RWKVQ_NOFUSED (rl.NOFUSED) возвращает прежний путь.
  5. МУТАЦИОННЫЕ КРЫШКИ: RWKVQ_FUSED_MUT=off (тихо уйти на старый путь)
     роняет контролем включения; =zero (нулевые выходы) -- паритетом.
     Гонять с КОПИЕЙ выходного пути, не с боевым JSON (урок 22.08).

    python tests/test_sym_gemm_fused.py [rwkvq]
"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))

import mlx.core as mx                      # noqa: E402
import mlx.nn as nn                        # noqa: E402
import numpy as np                         # noqa: E402
from mlx.utils import tree_flatten         # noqa: E402

from rwkv_metal.lora.rwkvq_linear import RwkvqSymLinear  # noqa: E402
from rwkv_metal.lora import rwkvq_linear as rl            # noqa: E402

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/reduction_new.rwkvq"
KEYS = [
    ("blocks.0.att.receptance.weight", 3e-3, 6e-3),   # [2048,2048]@8
    ("blocks.0.ffn.key.weight", 3e-3, 6e-3),          # [8192,2048]@6
    ("blocks.0.ffn.value.weight", 3e-3, 6e-3),        # [2048,8192]@6
    ("head.weight", 3e-3, 6e-3),                      # [65536,2048]@8
]
T = 512


def relmax(a, b):
    mx.eval(a, b)
    a = np.array(a.astype(mx.float32)).astype(np.float64)
    b = np.array(b.astype(mx.float32)).astype(np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def kernel_parity():
    rs = np.random.RandomState(7)
    for key, tf, tv in KEYS:
        lin = RwkvqSymLinear.from_sidecar(PATH, key)
        sym = lin._sym
        IN, OUT = sym.in_features, sym.out_features
        x = mx.array((rs.randn(T, IN) * 0.5).astype(np.float32)).astype(mx.bfloat16)
        r = relmax(sym.gemm_fused(x), x @ sym._dequant_w(mx.bfloat16).T)
        assert r < tf, f"{key}: fwd relmax {r:.2e} > {tf}"
        dy = mx.array((rs.randn(T, OUT) * 0.02).astype(np.float32)).astype(mx.bfloat16)
        rv = relmax(sym.gemm_fused_vjp(dy), dy @ sym._dequant_w(mx.bfloat16))
        assert rv < tv, f"{key}: vjp relmax {rv:.2e} > {tv}"
        print(f"  {key:34s} fwd {r:.2e} vjp {rv:.2e} -- ок")


def model_step():
    from rwkv_metal.lora import load_rwkvq_model
    model, _, _ = load_rwkvq_model(PATH, rank=16, verbose=False)
    model._grad_ckpt = True
    rs = np.random.RandomState(5)
    x = mx.array(rs.randint(1, 60000, size=(1, T)).astype(np.int32))
    y = mx.array(rs.randint(1, 60000, size=(1, T)).astype(np.int32))

    def loss_fn(m, a, b):
        return m.loss(a, b).astype(mx.float32)

    dq_calls = [0]
    orig_dq = rl.RwkvqSymLinear._dequant_w

    def counted_dq(self):
        dq_calls[0] += 1
        return orig_dq(self)

    def run(nofused):
        dq_calls[0] = 0
        c0 = rl.FUSED_CALLS
        rl.RwkvqSymLinear._dequant_w = counted_dq
        rl.NOFUSED = nofused
        try:
            loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
            mx.eval(loss, grads)
        finally:
            rl.RwkvqSymLinear._dequant_w = orig_dq
            rl.NOFUSED = False
        vec = np.concatenate([np.array(g.astype(mx.float32)).ravel()
                              for _, g in tree_flatten(grads)])
        return float(loss), vec, rl.FUSED_CALLS - c0, dq_calls[0]

    # 1+3: включение и паритет сквозного шага
    lo, go, _, dq_o = run(nofused=True)
    lf, gf_, cf, dq_f = run(nofused=False)
    assert cf > 200, f"fused-путь не включился (вызовов {cf})"
    assert dq_f == 0, (
        f"на fused-пути деквант вызван {dq_f} раз -- мерялся не тот путь "
        f"(старый путь для сравнения: {dq_o} вызовов)")
    lr_ = abs(lf - lo) / lo
    gr = np.linalg.norm(gf_ - go) / np.linalg.norm(go)
    print(f"  шаг: лосс {lo:.6f} -> {lf:.6f} (rel {lr_:.2e}), "
          f"||dg||/||g|| {gr:.2e}, fused {cf} вызовов, деквантов {dq_f}")
    assert lr_ < 1e-3, f"лосс разошёлся на {lr_:.2e}"
    assert gr < 5e-3, f"градиенты разошлись на {gr:.2e}"

    # 4: флаг-реверс возвращает прежний путь
    _, _, cf2, dq2 = run(nofused=True)
    assert cf2 == 0 and dq2 == dq_o, "NOFUSED не вернул прежний путь"

    # 5: мутационные крышки
    mut = os.environ.get("RWKVQ_FUSED_MUT")
    if mut == "off":
        rl._sym_fused_call_orig = rl._sym_fused_call

        def _mut(self, x):
            return rl._matmul_cast(self, x)   # тихо уходим на старый путь
        rl.RwkvqSymLinear.__call__ = _mut
        _, _, cf3, dq3 = run(nofused=False)
        rl.RwkvqSymLinear.__call__ = rl._sym_fused_call_orig
        if cf3 > 0 and dq3 == 0:
            print(f"  МУТАЦИЯ off: контроль включения ПРОМОЛЧАЛ "
                  f"(fused {cf3}, деквантов {dq3}) -- ГЕЙТ НЕ ЧУВСТВИТЕЛЕН")
            raise AssertionError("мутация off не поймана контролем включения")
        print(f"  МУТАЦИЯ off ПОЙМАНА контролем включения "
              f"(fused {cf3}, деквантов {dq3} при {dq_o} у старого пути)")
        raise SystemExit(0)
    if mut == "zero":
        calls = [0]
        orig = rl._build_sym_fused_fn

        def _z(sym):
            def _f(xb):
                calls[0] += 1
                return mx.zeros((xb.shape[0], sym.out_features), mx.bfloat16)
            _f.vjp = orig(sym).vjp
            return _f
        # СБРОС КЕША ОБЯЗАТЕЛЕН: _fused_fn построен на первом fused-прогоне
        # и живёт на модуле, поэтому подмена одной лишь фабрики
        # _build_sym_fused_fn НИЧЕГО не мутирует -- мутационный тест тихо
        # мерил здоровый код (поймано 23.08 вечером: "zero не поймана").
        def _reset_fused_cache():
            n = 0
            for _, m in model.named_modules():
                if isinstance(m, rl.RwkvqSymLinear):
                    m._fused_fn = None
                    n += 1
            return n

        rl._build_sym_fused_fn = _z
        nreset = _reset_fused_cache()
        assert nreset > 0, "не найдено ни одного RwkvqSymLinear -- мутация пуста"
        try:
            lz, _, cfz, _ = run(nofused=False)
        finally:
            rl._build_sym_fused_fn = orig
            _reset_fused_cache()
        assert cfz > 0, "мутация zero: fused-путь не включился"
        assert calls[0] > 0, (
            f"мутация zero НЕ ПРИМЕНИЛАСЬ: заглушка не звалась ни разу "
            f"(сброшено кешей {nreset}, fused-вызовов {cfz})")
        drift = abs(lz - lo) / lo
        if drift < 1e-3:
            print(f"  МУТАЦИЯ zero: лосс не шелохнулся (rel {drift:.2e}, "
                  f"заглушка звалась {calls[0]} раз) -- ГЕЙТ НЕ ЧУВСТВИТЕЛЕН")
            raise AssertionError("мутация zero не поймана паритетом")
        print(f"  МУТАЦИЯ zero ПОЙМАНА паритетом лосса "
              f"(rel {drift:.2e}, заглушка звалась {calls[0]} раз)")
        raise SystemExit(0)


def main():
    print("1-2. паритет кернелей по формам:")
    kernel_parity()
    print("3-5. сквозной шаг, флаг, мутации:")
    model_step()
    print("ГЕЙТ ЗЕЛЁНЫЙ")


if __name__ == "__main__":
    main()
