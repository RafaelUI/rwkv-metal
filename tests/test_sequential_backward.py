"""ГЕЙТ ОБРАТНОГО ПРОХОДА ПО БЛОКАМ (lora/sequential.py).

Режим меняет только то, что живёт в памяти одновременно; градиент обязан
остаться тем же, что у `nn.value_and_grad` по всему графу. Проверяется по
КАЖДОМУ адаптеру:

1. БЕЗ КУСКОВ -- ПОБИТНО автоград: те же операции в том же порядке
   (плотная 0.1B + LoRA: все слои; только верхние; батч 2).
2. С КУСКАМИ лосса -- в допуске: меняется порядок сложения в голове, а
   котангенты идут через адаптеры в bf16 (решётка 2^-8).
3. КВАНТОВАННАЯ БАЗА (.rwkvq, QLoRA): градиент идёт через свой VJP проекций.
4. `finetune` идёт этим путём по умолчанию, не идёт при выключателе, и
   обучение даёт те же адаптеры.
5. После счёта в модели -- значения адаптеров, а не трассеры: лосс тот же.

МУТАЦИОННЫЕ КРЫШКИ (SEQ_MUT=...): правка вносится в ИСХОДНИК модуля перед
его исполнением, гейт обязан упасть.
  dv      -- котангент по v_first не копится по слоям
  v0      -- котангент по v_first в слой 0 не доходит
  scale   -- лосс нормируется без батча
  order   -- куски котангента головы склеены в обратном порядке
  input   -- котангент по входу не доходит до нижнего блока
  restore -- параметры блока подставляются не свои
  off     -- finetune игнорирует режим
  strideall -- запись с шагом хранит все входы (память не делится)
  segv    -- при пересчёте отрезка слои выше нулевого не получают v_first
             (у слоя 0 переданный v_first ничего не меняет: блок решает по
             номеру, что v_first рождается в нём, -- такой мутант равносилен)
  segin   -- пересчитанный отрезок записывает не свои входы
  budget  -- кусок лосса снова на строку, а не на батч

    python tests/test_sequential_backward.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx                      # noqa: E402
import mlx.nn as nn                        # noqa: E402
import numpy as np                         # noqa: E402
from mlx.utils import tree_flatten         # noqa: E402

import rwkv_metal.lora.sequential as seq_orig          # noqa: E402
import rwkv_metal.lora.finetune                        # noqa: E402,F401
ft = sys.modules["rwkv_metal.lora.finetune"]           # имя пакета занято функцией finetune
from rwkv_metal.lora import add_lora, load_rwkvq_model  # noqa: E402
from rwkv_metal.model.convert import load_pretrained    # noqa: E402

MUT = os.environ.get("SEQ_MUT", "")
PTH = os.path.expanduser(os.environ.get(
    "RWKV_PTH", "~/Develop/SwiftRWKV/.testdata/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth"))
RWKVQ = os.path.expanduser(os.environ.get(
    "RWKVQ_BASE", "~/Develop/WKV-kvant/files_0110/0p1b_reduction.rwkvq"))

MUTATIONS = {
    "dv": ("dv = g[1] if dv is None else dv + g[1]", "dv = g[1]"),
    "v0": ("cotangents.append(dv if dv is not None else mx.zeros_like(v_first))",
           "cotangents.append(mx.zeros_like(v_first))"),
    "scale": ("scale = mx.array(1.0 / (B * T), dtype=mx.float32)",
              "scale = mx.array(1.0 / T, dtype=mx.float32)"),
    "order": ("mx.concatenate(pieces, axis=1)", "mx.concatenate(pieces[::-1], axis=1)"),
    "input": ("need_input = l > first", "need_input = l > first + 1"),
    "restore": ("blk.update(tree_unflatten(list(zip(ks, ins[n_data:]))))",
                "blk.update(tree_unflatten(list(zip(ks, ins[n_data:][::-1]))))"),
    "strideall": ("if l >= first and (l - first) % stride == 0:", "if l >= first:"),
    "segv": ("h, _ = model.blocks[j](h, v_first if j > 0 else None)", "h, _ = model.blocks[j](h, None)"),
    "segin": ("        block_input[j + 1] = h", "        block_input[j + 1] = block_input[s]"),
    "budget": ("return max(1, loss_chunk // max(1, batch))", "return loss_chunk"),
}


def load_seq():
    if MUT not in MUTATIONS:
        return seq_orig
    src = open(seq_orig.__file__, encoding="utf-8").read()
    a, b = MUTATIONS[MUT]
    assert src.count(a) == 1, f"мутация {MUT}: подстрока встречается {src.count(a)} раз"
    m = types.ModuleType("seq_mutated")
    m.__file__ = seq_orig.__file__
    exec(compile(src.replace(a, b), seq_orig.__file__, "exec"), m.__dict__)
    return m


seq = load_seq()


def liven(model, seed=99):
    """B адаптеров -- не ноль: при нулевом B градиент по A тождественно нуль."""
    rs = np.random.RandomState(seed)
    upd = []
    for k, v in tree_flatten(model.trainable_parameters()):
        if k.endswith("lora_b"):
            upd.append((k, mx.array(rs.normal(size=v.shape).astype(np.float32) * 0.05).astype(v.dtype)))
    from mlx.utils import tree_unflatten
    model.update(tree_unflatten(upd))
    mx.eval(model.parameters())


def batch(B, T, seed=7):
    rs = np.random.RandomState(seed)
    ids = rs.randint(1, 60000, size=(B, T + 1)).astype(np.int32)
    return mx.array(ids[:, :-1]), mx.array(ids[:, 1:])


def reference(model, x, y):
    model._grad_ckpt = False
    loss, grads = nn.value_and_grad(model, lambda m, a, b: m.loss(a, b).astype(mx.float32))(model, x, y)
    mx.eval(loss, grads)
    return loss.item(), dict(tree_flatten(grads))


def compare(what, model, x, y, chunk, bound, stride=1):
    loss_ref, g_ref = reference(model, x, y)
    loss, grads = seq.value_and_grad(model, x, y, chunk, trace_stride=stride)
    mx.eval(loss, grads)
    g = dict(tree_flatten(grads))
    assert set(g) == set(g_ref), (what, set(g) ^ set(g_ref))
    worst = 0.0
    for k in sorted(g_ref):
        r = g_ref[k].astype(mx.float32)
        scale = mx.abs(r).max().item()
        assert scale > 0, f"{what}: эталонный градиент {k} нулевой -- сверять нечего"
        e = mx.abs(r - g[k].astype(mx.float32)).max().item() / scale
        worst = max(worst, e)
        assert e <= bound, f"{what}: {k} разошёлся на {e:.2e} (допуск {bound:.0e})"
    assert abs(loss.item() - loss_ref) <= abs(loss_ref) * 1e-5, (what, loss.item(), loss_ref)
    # 5. в модели остались значения: лосс тот же
    again = model.loss(x, y).astype(mx.float32).item()
    assert abs(again - loss_ref) <= abs(loss_ref) * 1e-6, (what, again, loss_ref)
    print(f"  {what}: лосс {loss.item():.6f} против {loss_ref:.6f}; "
          f"худший градиент {worst:.1e} ({len(g_ref)} параметров)")


def dense(layers=None):
    model = load_pretrained(PTH, verbose=False)[0]
    mx.random.seed(11)             # A адаптеров случайна: два прогона обязаны стартовать одинаково
    add_lora(model, rank=4, alpha=8.0, layers=layers)
    liven(model)
    return model


def main():
    print(f"обратный проход по блокам, мутация: {MUT or 'нет'}")
    # 1. без кусков -- побитно
    m = dense()
    assert seq.supported(m)
    compare("плотная, все слои", m, *batch(1, 48), chunk=0, bound=0.0)
    compare("плотная, батч 2", m, *batch(2, 32), chunk=0, bound=0.0)
    # 2. куски
    compare("плотная, лосс кусками по 16", m, *batch(2, 48), chunk=16, bound=1e-2)
    compare("плотная, неровные куски по 20", m, *batch(1, 48), chunk=20, bound=1e-2)
    # запись с шагом: побитно тот же градиент, в том числе отрезок со слоем 0
    for stride in (2, 3, 5):
        compare(f"плотная, все слои, запись с шагом {stride}", m, *batch(2, 32), chunk=0, bound=0.0,
                stride=stride)
    x2, _ = batch(1, 32)
    for first, stride, want in ((0, 3, [0, 3, 6, 9]), (9, 2, [9, 11]), (4, 1, list(range(4, 12)))):
        _, bi, _ = seq.forward_trace(m, x2, first, stride)
        assert sorted(bi) == want, f"запись с шагом {stride} от {first}: {sorted(bi)} вместо {want}"
    # кусок лосса -- бюджет позиций на весь батч
    for args, want in (((256, 8, 512), 32), ((256, 1, 512), 256), ((16, 2, 48), 8), ((0, 4, 48), 48),
                       ((3, 8, 48), 1), ((None, 2, 40), 40)):
        got = seq.loss_piece_positions(*args)
        assert got == want, f"loss_piece_positions{args} = {got}, ожидалось {want}"
    compare("плотная, бюджет лосса 32 на батч 4", m, *batch(4, 32), chunk=32, bound=1e-2)
    del m
    m = dense(layers=[9, 10, 11])
    compare("плотная, верхние слои", m, *batch(1, 48), chunk=0, bound=0.0)
    compare("плотная, верхние слои, запись с шагом 2", m, *batch(1, 48), chunk=0, bound=0.0, stride=2)
    del m
    mx.clear_cache()
    # 3. квантованная база
    q = load_rwkvq_model(RWKVQ, rank=4, alpha=8.0, verbose=False, layers=[8, 9, 10, 11])[0]
    liven(q)
    compare(".rwkvq, QLoRA на верхних слоях", q, *batch(1, 32), chunk=16, bound=1e-2)
    compare(".rwkvq, QLoRA, запись с шагом 3", q, *batch(1, 32), chunk=0, bound=0.0, stride=3)
    del q
    mx.clear_cache()

    # 4. finetune: умолчание, выключатель, тот же результат
    def run(sequential):
        model = dense()
        x, y = batch(1, 32)
        cfg = ft.LoRAConfig(lr=1e-3, max_steps=3, log_every=0, cache_limit_gb=0,
                            adapter_path="/tmp/_seq_test_adapters.safetensors")
        assert cfg.sequential_backward is True, "умолчание"
        cfg.sequential_backward = sequential
        cfg.loss_chunk_tokens = 0
        if MUT == "off":
            cfg.sequential_backward = False
        n = seq_orig.CALLS
        res = ft.finetune(model, [(x, y)], cfg)
        return res["final_loss"], dict(tree_flatten(model.trainable_parameters())), seq_orig.CALLS - n
    l_seq, p_seq, n_seq = run(True)
    l_ref, p_ref, n_ref = run(False)
    assert n_seq == 3, f"finetune не пошёл обратным проходом по блокам ({n_seq} вызовов)"
    assert n_ref == 0, "выключатель не действует"
    worst = max(mx.abs(p_seq[k].astype(mx.float32) - p_ref[k].astype(mx.float32)).max().item()
                for k in p_ref)
    assert abs(l_seq - l_ref) <= abs(l_ref) * 1e-5 and worst == 0.0, (l_seq, l_ref, worst)
    print(f"  finetune: лосс {l_seq:.6f} против {l_ref:.6f}, адаптеры после 3 шагов {worst:.1e}")

    # режим не берёт модель с обучаемыми параметрами вне блоков
    plain = load_pretrained(PTH, verbose=False)[0]
    assert not seq.supported(plain), "полное дообучение -- не дело этого режима"
    print("ГЕЙТ ПРОЙДЕН")


if __name__ == "__main__":
    main()
