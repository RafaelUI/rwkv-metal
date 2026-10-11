"""ГЕЙТ ЭМБЕДДЕРА ОБРАТНЫМ ПРОХОДОМ ПО БЛОКАМ (embedding/sequential.py).

Режим меняет только то, что живёт в памяти одновременно; градиент обязан
остаться тем же, что у `nn.value_and_grad` по всему графу. Проверяется по
КАЖДОМУ обучаемому параметру, во всех наборах обучаемого:

1. ОДНА ПАЧКА -- побитно автоград (те же операции в том же порядке).
2. ТРИПЛЕТ (пачки разной длины), вся база: emb и ln0 под блоками, ln_out и
   голова над ними; голова языковой модели -- нулевой градиент.
3. Только голова; верхние слои (низ и основание заморожены).
4. Адаптеры на плотной базе и на .rwkvq (QLoRA) -- в допуске решётки bf16:
   у автограда вклады пачек складываются в котангенте bf16, здесь -- в fp32.
5. Задачи: retrieval, sts (обе стороны), пары, zero-shot классификация.
6. После счёта в модели -- значения, а не трассеры: лосс тот же.
7. `finetune_embedding` идёт этим путём по умолчанию, не идёт при
   выключателе и под GradCache; три шага дают те же параметры.
8. Лосс, лезущий в модель мимо `embed`, получает внятный отказ.

МУТАЦИОННЫЕ КРЫШКИ (SEQ_MUT=...): правка вносится в ИСХОДНИК модуля перед
его исполнением, гейт обязан упасть.
  sum     -- градиенты пачек не складываются (остаётся последняя)
  top     -- при повторе все пачки получают выход тела первой
  cot     -- все пачки получают котангент первой
  above   -- параметры над телом не подставляются под трассировку
  ln_out  -- вектор по выходу тела берётся без ln_out
  unused  -- неучаствующему параметру -- не ноль
  stem    -- котангент по входу нижнего блока не считается, основание
             получает чужой
  first   -- при обучаемом основании нижний блок пропущен
  dv, v0  -- котангент по v_first не копится / не доходит до слоя 0
  input   -- котангент по входу не доходит до блока над нижним обучаемым
  off     -- finetune_embedding игнорирует режим
  gc      -- режим перехватывает шаг под GradCache

    python tests/test_embedding_sequential.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx                      # noqa: E402
import mlx.nn as nn                        # noqa: E402
import numpy as np                         # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten   # noqa: E402

import rwkv_metal.lora.sequential as body_orig        # noqa: E402
import rwkv_metal.embedding.sequential as seq_orig    # noqa: E402
import rwkv_metal.embedding.train as train_orig       # noqa: E402
from rwkv_metal.embedding import tasks                # noqa: E402
from rwkv_metal.embedding.loss import triplet_pool_loss   # noqa: E402
from rwkv_metal.lora import add_lora, load_rwkvq_model     # noqa: E402
from rwkv_metal.model.convert import load_pretrained       # noqa: E402

MUT = os.environ.get("SEQ_MUT", "")
PTH = os.path.expanduser(os.environ.get(
    "RWKV_PTH", "~/Develop/SwiftRWKV/.testdata/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth"))
RWKVQ = os.path.expanduser(os.environ.get(
    "RWKVQ_BASE", "~/Develop/WKV-kvant/files_0110/0p1b_reduction.rwkvq"))

# имя -> (модуль, было, стало)
MUTATIONS = {
    "sum": ("body", "gk = grads[name] + gk", "gk = gk"),
    "top": ("seq", "model.embed_from_top(self._replay[k], pool_idx)",
            "model.embed_from_top(self._replay[0][:, :self._replay[k].shape[1]], pool_idx)"),
    "cot": ("seq", "block_input, v_first, g[k], body, first",
            "block_input, v_first, g[0][:, :g[k].shape[1]], body, first"),
    "above": ("seq", "        put(ins[n:])\n", "        pass\n"),
    "ln_out": ("train", "self._pool(self.base.ln_out(top), pool_idx)", "self._pool(top, pool_idx)"),
    "unused": ("seq", "grads[name] = mx.zeros_like(v)", "grads[name] = mx.ones_like(v)"),
    "stem": ("body", "need_input = l > first or stem_needed", "need_input = l > first"),
    "first": ("body", "    if stem_keys(model):\n        return 0", "    if stem_keys(model):\n        return 1"),
    "dv": ("body", "dv = g[1] if dv is None else dv + g[1]", "dv = g[1]"),
    "v0": ("body", "cotangents.append(dv if dv is not None else mx.zeros_like(v_first))",
           "cotangents.append(mx.zeros_like(v_first))"),
    "input": ("body", "need_input = l > first", "need_input = l > first + 1"),
    "off": ("train", "use_seq = cfg.sequential_backward and", "use_seq = False and"),
    "gc": ("train", "use_seq = cfg.sequential_backward and not use_gc and", "use_seq = cfg.sequential_backward and"),
}
ORIG = {"body": body_orig, "seq": seq_orig, "train": train_orig}


def load_modules():
    """(тело, эмбеддер, обучение) -- один из них с правкой, остальные проведены к нему."""
    if not MUT:
        return body_orig, seq_orig, train_orig
    assert MUT in MUTATIONS, f"неизвестная мутация {MUT}"
    which, a, b = MUTATIONS[MUT]
    orig = ORIG[which]
    src = open(orig.__file__, encoding="utf-8").read()
    assert src.count(a) == 1, f"мутация {MUT}: подстрока встречается {src.count(a)} раз"
    m = types.ModuleType(orig.__name__ + "_mutated")
    m.__file__ = orig.__file__
    m.__package__ = orig.__package__
    exec(compile(src.replace(a, b), orig.__file__, "exec"), m.__dict__)
    mods = dict(ORIG)
    mods[which] = m
    if which == "body":            # эмбеддер зовёт тело через своё имя _body
        s = types.ModuleType("seq_rewired")
        s.__file__, s.__package__ = seq_orig.__file__, seq_orig.__package__
        exec(compile(open(seq_orig.__file__, encoding="utf-8").read(), seq_orig.__file__, "exec"), s.__dict__)
        s._body = m
        mods["seq"] = s
    if which in ("body", "seq"):   # обучение зовёт эмбеддер через своё имя _seq
        t = types.ModuleType("train_rewired")
        t.__file__, t.__package__ = train_orig.__file__, train_orig.__package__
        exec(compile(open(train_orig.__file__, encoding="utf-8").read(), train_orig.__file__, "exec"), t.__dict__)
        t._seq = mods["seq"]
        mods["train"] = t
    return mods["body"], mods["seq"], mods["train"]


body, seq, train = load_modules()
EmbeddingModel = train.EmbeddingModel


def liven(model, seed=99):
    """B адаптеров и fc2 головы -- не ноль: иначе часть градиентов тождественно нуль."""
    rs = np.random.RandomState(seed)
    upd = []
    for k, v in tree_flatten(model.trainable_parameters()):
        if k.endswith("lora_b") or k == "head.fc2.weight":
            upd.append((k, mx.array(rs.normal(size=v.shape).astype(np.float32) * 0.05).astype(v.dtype)))
    model.update(tree_unflatten(upd))
    mx.eval(model.parameters())


def group(B, T, seed):
    rs = np.random.RandomState(seed)
    idx = mx.array(rs.randint(1, 60000, size=(B, T)).astype(np.int32))
    pool = mx.array(rs.randint(T // 2, T, size=(B,)).astype(np.int32))
    return idx, pool


def triplet(B=3, lengths=(36, 20, 28), seed=5):   # первая пачка -- длиннейшая (мутации top / cot)
    out = ()
    for i, T in enumerate(lengths):
        out += group(B, T, seed + i)
    return out


def classification(B=2, K=3, seed=17):
    a_idx, a_pool = group(B, 24, seed)
    c_idx, c_pool = group(B * K, 12, seed + 1)
    mask = mx.array(np.array([[1, 1, 1], [1, 1, 0]], dtype=np.float32))
    return (a_idx, a_pool, c_idx.reshape(B, K, 12), c_pool.reshape(B, K), mask,
            mx.array(np.array([2, 0], dtype=np.int32)))


def single_loss(model, batch, temperature):
    """Одна пачка: лосс по векторам без взаимодействия с другими пачками."""
    idx, pool, w = batch
    return (model.embed(idx, pool).astype(mx.float32) * w).sum() / temperature


def reference(model, batch, loss):
    model.base._grad_ckpt = False
    value, grads = nn.value_and_grad(model, lambda m, b: loss(m, b, 0.05).astype(mx.float32))(model, batch)
    mx.eval(value, grads)
    return value.item(), dict(tree_flatten(grads))


def compare(what, model, batch, loss, bound, zero=("base.head.",)):
    assert seq.supported(model), what
    loss_ref, g_ref = reference(model, batch, loss)
    n = seq.CALLS
    value, grads = seq.value_and_grad(model, batch, loss, 0.05)
    mx.eval(value, grads)
    assert seq.CALLS == n + 1
    g = dict(tree_flatten(grads))
    assert set(g) == set(g_ref), (what, sorted(set(g) ^ set(g_ref))[:6])
    worst, live = 0.0, 0
    for k in sorted(g_ref):
        assert g[k].dtype == g_ref[k].dtype and g[k].shape == g_ref[k].shape, (
            f"{what}: {k} -- {g[k].dtype} {g[k].shape} против {g_ref[k].dtype} {g_ref[k].shape}")
        r = g_ref[k].astype(mx.float32)
        scale = mx.abs(r).max().item()
        mine = g[k].astype(mx.float32)
        if k.startswith(zero):
            assert scale == 0 and mx.abs(mine).max().item() == 0, f"{what}: {k} обязан быть нулём"
            continue
        assert scale > 0, f"{what}: эталонный градиент {k} нулевой -- сверять нечего"
        e = mx.abs(r - mine).max().item() / scale
        worst = max(worst, e)
        live += 1
        assert e <= bound, f"{what}: {k} разошёлся на {e:.2e} (допуск {bound:.0e})"
    assert abs(value.item() - loss_ref) <= abs(loss_ref) * 1e-5, (what, value.item(), loss_ref)
    # 6. в модели остались значения: лосс тот же
    again = loss(model, batch, 0.05).astype(mx.float32).item()
    assert abs(again - loss_ref) <= abs(loss_ref) * 1e-6, (what, again, loss_ref)
    print(f"  {what}: лосс {value.item():.6f} против {loss_ref:.6f}; "
          f"худший градиент {worst:.1e} ({live} параметров)")


def dense(mode, fp32=True):
    """Плотная 0.1B. fp32 -- чтобы сверять с автоградом вплотную: в родном
    bf16 сумма вкладов пачек расходится с ним на решётку (2^-8), и за ней не
    видно ошибок мельче."""
    base = load_pretrained(PTH, verbose=False)[0]
    if fp32 and mode != "lora":
        base.set_dtype("float32")
    mx.random.seed(11)
    if mode == "lora":
        add_lora(base, rank=4, alpha=8.0)
    model = EmbeddingModel(base)
    if mode == "head":
        base.freeze()
    elif mode == "top":
        base.emb.freeze(); base.ln0.freeze(); base.head.freeze()
        for blk in base.blocks[:9]:
            blk.freeze()
    liven(model)
    return model


TIGHT = float(os.environ.get("SEQ_TIGHT", "2e-5"))     # fp32-счёт, другой порядок сложения пачек
LATTICE = 2e-2                                        # счёт в bf16: решётка 2^-8 на вклад пачки


def main():
    print(f"эмбеддер обратным проходом по блокам, мутация: {MUT or 'нет'}")
    # ── вся база ──
    m = dense("full")
    idx, pool = group(2, 40, 3)
    w = mx.array(np.random.RandomState(1).normal(size=(2, m.base.config.n_embd)).astype(np.float32))
    compare("вся база, одна пачка", m, (idx, pool, w), single_loss, bound=0.0)
    compare("вся база, retrieval", m, triplet(), tasks.retrieval_loss, bound=TIGHT)
    compare("вся база, sts", m, triplet(seed=9), tasks.sts_loss, bound=TIGHT)
    compare("вся база, классификация", m, classification(), tasks.classification_loss, bound=TIGHT)
    idx2, pool2 = group(2, 24, 4)
    compare("вся база, пары", m, (idx, pool, idx2, pool2), tasks.pair_loss, bound=TIGHT)
    del m
    mx.clear_cache()
    m = dense("full", fp32=False)
    print(f"  (родной тип плотной базы: {m.base.blocks[0].cmix.key.weight.dtype})")
    compare("вся база в родном типе, retrieval", m, triplet(), tasks.retrieval_loss, bound=LATTICE)

    # 8. лосс мимо embed
    def sneaky(model, batch, temperature):
        return model.base.body(batch[0]).sum()
    try:
        seq.value_and_grad(m, (idx, pool), sneaky, 0.05)
        raise AssertionError("лосс мимо embed не отклонён")
    except AttributeError as e:
        assert "sequential_backward" in str(e)
    del m
    mx.clear_cache()

    # ── только голова; верхние слои ──
    m = dense("head")
    assert body.first_layer(m.base) is None
    compare("только голова", m, triplet(), tasks.retrieval_loss, bound=TIGHT, zero=())
    del m
    m = dense("top")
    assert body.first_layer(m.base) == 9
    compare("верхние три слоя", m, triplet(), tasks.retrieval_loss, bound=TIGHT, zero=())
    del m
    mx.clear_cache()

    # ── адаптеры ──
    m = dense("lora")
    compare("LoRA, одна пачка", m, (idx, pool, w), single_loss, bound=0.0, zero=())
    compare("LoRA, retrieval", m, triplet(), tasks.retrieval_loss, bound=LATTICE, zero=())
    del m
    mx.clear_cache()
    q = load_rwkvq_model(RWKVQ, rank=4, alpha=8.0, verbose=False, layers=[8, 9, 10, 11])[0]
    m = EmbeddingModel(q)
    liven(m)
    compare(".rwkvq, QLoRA на верхних слоях", m, triplet(), tasks.retrieval_loss, bound=LATTICE, zero=())
    del m, q
    mx.clear_cache()

    # 7. finetune_embedding
    def run(sequential, chunk=0):
        model = dense("top")
        cfg = train.EmbedTrainConfig(lr=1e-3, max_steps=3, log_every=0, cache_limit_gb=0,
                                     checkpoint_path="/tmp/_emb_seq_test.safetensors")
        assert cfg.sequential_backward is True, "умолчание"
        cfg.sequential_backward = sequential
        cfg.gradcache_chunk = chunk
        n = seq.CALLS
        res = train.finetune_embedding(model, [triplet()], cfg, tasks.retrieval_loss,
                                       gradcache_spec=tasks.RETRIEVAL_GC if chunk else None)
        return res["final_loss"], dict(tree_flatten(model.trainable_parameters())), seq.CALLS - n
    l_seq, p_seq, n_seq = run(True)
    l_ref, p_ref, n_ref = run(False)
    _, _, n_gc = run(True, chunk=2)
    assert n_seq == 3, f"finetune_embedding не пошёл обратным проходом по блокам ({n_seq} вызовов)"
    assert n_ref == 0, "выключатель не действует"
    assert n_gc == 0, "под GradCache режим обязан молчать"
    worst = max(mx.abs(p_seq[k].astype(mx.float32) - p_ref[k].astype(mx.float32)).max().item()
                / max(mx.abs(p_ref[k].astype(mx.float32)).max().item(), 1e-12) for k in p_ref)
    # Adam делит градиент на его же масштаб: расхождение 1e-7 в почти нулевой
    # компоненте даёт шаг порядка lr, так что параметры сходятся лишь до lr
    # (градиенты сверены выше вплотную; здесь -- что путь тот и обучение то же).
    assert abs(l_seq - l_ref) <= abs(l_ref) * 1e-3 and worst <= 5e-3, (l_seq, l_ref, worst)
    print(f"  finetune_embedding: лосс {l_seq:.6f} против {l_ref:.6f}, параметры после 3 шагов {worst:.1e}")
    print("ГЕЙТ ПРОЙДЕН")


if __name__ == "__main__":
    main()
