"""Эмбеддер обратным проходом ПО БЛОКАМ: тот же градиент при памяти одного
блока одной пачки.

Зачем. `nn.value_and_grad` по всему графу держит активации всех слоёв всех
пачек (якоря, положительные, отрицательные) до конца обратного прохода; под
`grad_checkpoint` обратный проход остаётся одним ленивым графом, и порядок его
вычисления выбирает обход MLX (см. rwkv_metal/lora/sequential.py). Здесь
очередь задана явно:

  1. каждая пачка идёт прямым проходом БЕЗ градиента; хранятся только входы
     блоков, через которые пойдёт градиент;
  2. всё, что над телом -- ln_out, пулинг, голова, лосс, -- считается по
     выходам последнего блока всех пачек сразу (лосс видит весь батч), и
     берётся его градиент по этим выходам и по параметрам над телом;
  3. блоки идут сверху вниз, пачка за пачкой, с вычислением после каждого
     блока; градиенты пачек складываются.

Работает для любого набора обучаемых параметров: только голова, верхние слои,
вся база (включая ln0 / emb под блоками и ln_out над ними), адаптеры LoRA /
QLoRA -- на плотной базе и на .rwkvq. Цена -- каждый блок считается дважды.

Лосс задаётся так же, как для обычного пути: `compute_loss(model, batch,
temperature)`, обращающийся к модели ТОЛЬКО через `model.embed(idx, pool_idx)`
(все задачи из `rwkv_metal.embedding.tasks` таковы). Функция вызывается
дважды с подставной моделью: первый раз `embed` пишет прямой проход пачки,
второй -- возвращает вектор из уже посчитанного выхода тела. Требование то же,
что у GradCache: `compute_loss` детерминирован и зовёт `embed` оба раза в
одном порядке.

Порт SwiftRWKV: EmbeddingObjective.sequentialValueAndGrad + SequentialBody.
"""
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from ..lora import sequential as _body

# Счётчик вызовов -- для тестов: по числам режим от автограда почти не отличим.
CALLS = 0

_ABOVE = ("head", "base.ln_out")       # модули над телом, чьи параметры обучаются
_BODY = ("base.blocks.", "base.ln0.", "base.emb.")
_UNUSED = ("base.head.",)              # голова языковой модели в эмбеддинге не участвует


def supported(model) -> bool:
    """Модель -- `EmbeddingModel` над базой со списком блоков и новым
    token-shift, и каждый обучаемый параметр лежит там, где этот путь умеет
    считать его градиент."""
    base = getattr(model, "base", None)
    if base is None or not all(hasattr(base, a) for a in ("blocks", "ln0", "ln_out", "emb")):
        return False
    if not hasattr(model, "embed_from_top") or getattr(base, "legacy_token_shift", False):
        return False
    names = [k for k, _ in tree_flatten(model.trainable_parameters())]
    known = _BODY + _UNUSED + ("head.", "base.ln_out.")
    return bool(names) and all(k.startswith(known) for k in names)


class _Stage:
    """Подставная модель для `compute_loss`: наружу -- только `embed`."""

    def __init__(self, model, first):
        self._model = model
        self._first = first
        self.traces = []               # (idx, pool_idx, входы блоков, v_first)
        self.tops = []                 # выходы последнего блока, по пачкам
        self._replay = None            # список трассеров выходов тела
        self._at = 0

    def begin_replay(self, tops):
        self._replay, self._at = tops, 0

    def embed(self, idx, pool_idx):
        model = self._model
        if self._replay is None:       # запись: прямой проход без градиента
            top, block_input, v_first = _body.forward_trace(model.base, idx, self._first)
            self.traces.append((idx, pool_idx, block_input, v_first))
            self.tops.append(top)
            return model.embed_from_top(top, pool_idx)
        k = self._at                   # повтор: вектор из готового выхода тела
        self._at += 1
        assert k < len(self.traces) and self.traces[k][0].shape == idx.shape, (
            "compute_loss обязан звать model.embed оба раза в одном порядке и с теми же пачками")
        return model.embed_from_top(self._replay[k], pool_idx)

    __call__ = embed

    def __getattr__(self, name):
        if name == "head":             # параметры головы -- под трассировкой, можно
            return getattr(self._model, name)
        raise AttributeError(
            f"обратный проход по блокам: compute_loss обращается к model.{name}, а путь видит "
            f"только model.embed(idx, pool_idx). Для такого лосса выключите "
            f"EmbedTrainConfig.sequential_backward.")


def value_and_grad(model, batch, compute_loss, temperature: float = 0.05):
    """(лосс, дерево градиентов) -- как
    `nn.value_and_grad(model, lambda m, b: compute_loss(m, b, temperature))(model, batch)`."""
    global CALLS
    CALLS += 1
    base = model.base
    first = _body.first_layer(base)

    # ── 1. прямой проход каждой пачки без градиента ──
    stage = _Stage(model, first)
    compute_loss(stage, batch, temperature)
    n = len(stage.traces)
    assert n > 0, "compute_loss ни разу не вызвал model.embed"

    # ── 2. над телом: ln_out, пулинг, голова, лосс ──
    mods = []                          # (модуль, префикс имени, [имена])
    for prefix in _ABOVE:
        m = model
        for part in prefix.split("."):
            m = getattr(m, part)
        ks = [k for k, _ in tree_flatten(m.trainable_parameters())]
        if ks:
            mods.append((m, prefix, ks))
    values = [dict(tree_flatten(m.trainable_parameters()))[k] for m, _, ks in mods for k in ks]
    names = [f"{prefix}.{k}" for _, prefix, ks in mods for k in ks]

    def put(ins):
        at = 0
        for m, _, ks in mods:
            m.update(tree_unflatten(list(zip(ks, ins[at:at + len(ks)]))))
            at += len(ks)

    def above(*ins):
        put(ins[n:])
        stage.begin_replay(ins[:n])
        return [compute_loss(stage, batch, temperature).astype(mx.float32)]

    value, g = mx.vjp(above, stage.tops + values, [mx.array(1.0, dtype=mx.float32)])
    put(values)                        # в модели -- значения, а не трассеры
    assert stage._at == n, "compute_loss при повторе вызвал model.embed другое число раз"
    mx.eval(value[0], g)
    grads = dict(zip(names, g[n:]))
    stage.tops = []

    # ── 3. блоки сверху вниз, пачка за пачкой ──
    body = {}
    for k, (idx, _, block_input, v_first) in enumerate(stage.traces):
        _body.backward_blocks(base, block_input, v_first, g[k], body, first, x=idx)
    for name, gk in body.items():
        grads[f"base.{name}"] = gk

    # обучаемое, но не участвующее в эмбеддинге (голова языковой модели при
    # полном дообучении): нулевой градиент, как у автограда
    for name, v in tree_flatten(model.trainable_parameters()):
        if name not in grads:
            assert name.startswith(_UNUSED), f"градиент {name} не посчитан"
            grads[name] = mx.zeros_like(v)
    return value[0], tree_unflatten(list(grads.items()))
