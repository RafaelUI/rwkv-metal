"""Обратный проход ПО БЛОКАМ: градиент адаптеров при памяти одного блока.

Зачем. Блочный чекпоинт (`grad_checkpoint`) держит между проходами только
входы блоков, но весь обратный проход остаётся ОДНИМ ленивым графом MLX, и
порядок его вычисления выбирает обход графа, а не мы: пересчитанные
активации нескольких блоков оказываются живыми сразу. Замерено на 1.5B
(QLoRA, sym-база, батч 1, ранг 16): пик MLX растёт на ~1.9 ГБ на каждые 1024
токена при ~0.2 ГБ входов блоков -- T=2048 пикует 5.4 ГБ, T=4096 в 16 ГБ не
входит.

Здесь очередь задана явно. Прямой проход без градиента сохраняет вход
каждого блока; затем ln_out, голова и лосс (кусками по `loss_chunk` позиций,
каждый -- свой vjp и eval), затем блоки сверху вниз, и после каждого блока
граф ВЫЧИСЛЯЕТСЯ (eval вне трассировки), так что живёт один блок. Математика
та же, что у автограда по всему графу: цепное правило по блокам, котангент
по v_first копится со всех слоёв и уходит в слой 0, где v_first рождается.
Без кусков градиент ПОБИТНО равен `nn.value_and_grad` (тот же счёт в том же
порядке); с кусками меняется порядок сложения в голове.

Цена по времени та же, что у блочного чекпоинта: каждый блок считается
дважды. Порт SwiftRWKV/Sources/RWKVGen/Training/SequentialBackward.swift.
"""
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

# Счётчик вызовов: градиент побитно равен автограду, и по числам режим не
# виден -- тесты смотрят сюда.
CALLS = 0


def supported(model) -> bool:
    """Все обучаемые параметры принадлежат блокам (LoRA / QLoRA), а модель --
    со списком `blocks`. Иначе -- обычный путь."""
    if not hasattr(model, "blocks") or not hasattr(model, "ln_out"):
        return False
    names = [k for k, _ in tree_flatten(model.trainable_parameters())]
    return bool(names) and all(k.startswith("blocks.") for k in names)


# ───────────────────────────────────────────────────────────────────────
#  Общая часть: тело модели. Что стоит НАД телом (ln_out, голова, лосс),
#  решает вызывающий: языковая модель (value_and_grad ниже) или эмбеддер
#  (rwkv_metal/embedding/sequential.py). Имена градиентов -- относительно
#  модели-тела: `blocks.N.…`, `ln0.…`, `emb.…`.
# ───────────────────────────────────────────────────────────────────────

def layer_keys(model):
    """Плоские имена обучаемых параметров по блокам."""
    return [[k for k, _ in tree_flatten(b.trainable_parameters())] for b in model.blocks]


def stem_keys(model):
    """Обучаемые параметры основания под блоками: [(имя модуля, [имена])]."""
    out = []
    for name in ("emb", "ln0"):
        ks = [k for k, _ in tree_flatten(getattr(model, name).trainable_parameters())]
        if ks:
            out.append((name, ks))
    return out


def first_layer(model):
    """Нижний блок, через который идёт градиент; None -- тело не обучается."""
    if stem_keys(model):
        return 0
    with_params = [l for l, ks in enumerate(layer_keys(model)) if ks]
    return with_params[0] if with_params else None


def _add(grads: dict, name: str, gk: mx.array) -> mx.array:
    """Градиент пачки прибавляется к уже накопленному (первый -- кладётся как есть)."""
    if name in grads:
        gk = grads[name] + gk
    grads[name] = gk
    return gk


def forward_trace(model, x: mx.array, first: Optional[int]):
    """Прямой проход без градиента. Возвращает (выход последнего блока -- до
    ln_out, входы блоков начиная с `first`, v_first)."""
    blocks = model.blocks
    if first is None:
        first = len(blocks)
    h = model.ln0(model.emb(x))
    block_input = {}
    v_first = None
    for l, blk in enumerate(blocks):
        if l >= first:
            block_input[l] = h
        h, v_first = blk(h, v_first)
        mx.eval(h, v_first)
    return h, block_input, v_first


def backward_blocks(model, block_input: dict, v_first, dx: mx.array, grads: dict,
                    first: Optional[int], x: Optional[mx.array] = None):
    """Блоки сверху вниз от котангента `dx` на выходе последнего блока.
    Градиенты ПРИБАВЛЯЮТСЯ к `grads` (несколько пачек на один шаг); входы
    блоков освобождаются по ходу. `x` (токены) нужен, только если обучается
    основание -- ln0 / emb."""
    if first is None:
        return
    blocks = model.blocks
    L = len(blocks)
    keys = layer_keys(model)
    stem = stem_keys(model)
    stem_needed = bool(stem)
    dv = None                      # накопленный котангент по v_first
    for l in range(L - 1, first - 1, -1):
        blk = blocks[l]
        ks = keys[l]
        current = dict(tree_flatten(blk.trainable_parameters()))
        values = [current[k] for k in ks]
        xin = block_input[l]
        has_v = l > 0
        # У нижнего обучаемого блока котангенты по входу никому не нужны: ниже
        # него нет параметров. Вход тогда -- константа замыкания.
        need_input = l > first or stem_needed
        primals = []
        if need_input:
            primals.append(xin)
            if has_v:
                primals.append(v_first)
        n_data = len(primals)
        primals += values
        outputs_v = not has_v      # слой 0 сам рождает v_first

        def f(*ins, blk=blk, ks=ks, xin=xin, has_v=has_v, need_input=need_input,
              n_data=n_data, outputs_v=outputs_v):
            if ks:
                blk.update(tree_unflatten(list(zip(ks, ins[n_data:]))))
            xi = ins[0] if need_input else xin
            vi = (ins[1] if need_input else v_first) if has_v else None
            xo, vo = blk(xi, vi)
            return [xo, vo] if outputs_v else [xo]

        cotangents = [dx]
        if outputs_v:
            cotangents.append(dv if dv is not None else mx.zeros_like(v_first))
        g = mx.vjp(f, primals, cotangents)[1]
        if ks:                     # в модели -- значения, а не трассеры
            blk.update(tree_unflatten(list(zip(ks, values))))
        ready = []
        if need_input:
            dx = g[0]
            ready.append(dx)
            if has_v:
                dv = g[1] if dv is None else dv + g[1]
                ready.append(dv)
        for k, gk in zip(ks, g[n_data:]):
            ready.append(_add(grads, f"blocks.{l}.{k}", gk))
        mx.eval(ready)
        del block_input[l]

    if stem_needed:
        assert x is not None, "обучается основание: нужны токены пачки"
        mods = [(getattr(model, name), name, ks) for name, ks in stem]
        values = [dict(tree_flatten(m.trainable_parameters()))[k] for m, _, ks in mods for k in ks]

        def put(ins):
            at = 0
            for m, _, ks in mods:
                m.update(tree_unflatten(list(zip(ks, ins[at:at + len(ks)]))))
                at += len(ks)

        def f0(*ins):
            put(ins)
            return [model.ln0(model.emb(x))]

        g = mx.vjp(f0, values, [dx])[1]
        put(values)
        ready = []
        names = [f"{name}.{k}" for _, name, ks in mods for k in ks]
        for name, gk in zip(names, g):
            ready.append(_add(grads, name, gk))
        mx.eval(ready)


def value_and_grad(model, x: mx.array, y: mx.array, loss_chunk: Optional[int] = 256):
    """(лосс, дерево градиентов) -- как `nn.value_and_grad(model, loss)(model, x, y)`
    для `model.loss`, обратным проходом по блокам."""
    global CALLS
    CALLS += 1
    blocks = model.blocks
    L = len(blocks)
    B, T = x.shape

    # обучаемые параметры по блокам: плоские имена и значения
    keys = [[k for k, _ in tree_flatten(b.trainable_parameters())] for b in blocks]
    with_params = [l for l in range(L) if keys[l]]
    assert with_params, "нет обучаемых параметров в блоках"
    first = with_params[0]

    # ── 1. прямой проход без градиента: входы обучаемых блоков ──
    h, block_input, v_first = forward_trace(model, x, first)

    # ── 2. ln_out, голова и лосс: по кускам ──
    hn = model.ln_out(h)
    mx.eval(hn)
    scale = mx.array(1.0 / (B * T), dtype=mx.float32)
    step = loss_chunk if loss_chunk and loss_chunk > 0 else T
    total = mx.array(0.0, dtype=mx.float32)
    pieces = []
    for lo in range(0, T, step):
        hi = min(lo + step, T)
        targets = y[:, lo:hi].reshape(-1)

        def piece(hc, targets=targets):
            logits = model.head(hc)
            V = logits.shape[-1]
            return nn.losses.cross_entropy(logits.reshape(-1, V), targets,
                                           reduction="sum").astype(mx.float32)

        value, g = mx.vjp(piece, [hn[:, lo:hi]], [scale])
        mx.eval(value[0], g[0])
        total = total + value[0]
        pieces.append(g[0])
    dh = pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=1)
    dx = mx.vjp(lambda a: model.ln_out(a), [h], [dh])[1][0]
    loss = total * scale
    mx.eval(dx, loss)
    del pieces, hn

    # ── 3. блоки сверху вниз ──
    grads = {}
    backward_blocks(model, block_input, v_first, dx, grads, first)
    return loss, tree_unflatten(list(grads.items()))
