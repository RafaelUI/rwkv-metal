"""
rwkv_metal.model.initial_state
==============================
Начальное состояние как подключаемый объект: обученное (state-tuning),
свёрнутый промпт или одно поверх другого.

RWKV -- RNN: вместо нулей последовательность может стартовать с любого
состояния. Обученное состояние (state-tuning) и состояние после прогона
промпта -- ОДИН И ТОТ ЖЕ объект, `RWKVState` с батчем 1:

    wkv         [L, 1, H, S, S]  fp32
    tmix_shift  [L, 1, 1, D]
    cmix_shift  [L, 1, 1, D]

У обученного состояния по умолчанию обучается только матрица WKV (как в
RWKV-PEFT), сдвиги остаются нулями; у состояния после промпта заполнены все
три. В файле всегда все три -- потребителю не нужно знать, откуда объект.

Как подключается. `attach_state` кладёт в каждый блок подмодуль `state0`; блок
берёт из него h_in и сдвиги ТОЛЬКО когда вызывающий их не передал. Поэтому:

    model.body(idx)                  -> старт с подключённого состояния
    model.states(prompt)             -> подключённое -> прогон промпта -> кэш
    model.body(idx, state=cached)    -> явное состояние, подключённое не участвует

то есть склейка «обученное -> системный промпт -> кэш» -- это обычное
продолжение, отдельного кода у неё нет. Состояние [1, ...] транслируется на
батч ВНУТРИ блока, так что vjp блока сам суммирует градиент по строкам.

Обучение. `state0.wkv` -- параметр своего блока, и обратный проход по блокам
(`lora/sequential.py`, `embedding/sequential.py`) подхватывает его как любой
другой параметр блока: в памяти один блок. Подключённое нулевое состояние
счёта не меняет -- выход побитно тот же, что без него.

Файл. Отдельный safetensors: три тензора и метаданные -- контракт (слоёв, D,
голов, размер головы), что обучалось, и ОТПЕЧАТОК БАЗЫ. Состояние имеет смысл
только на той базе, на которой получено: на чужой оно даёт правдоподобный, но
другой результат. Контракт несовместим -- отказ всегда; отпечаток другой --
отказ, пока вызывающий явно не разрешил (`allow_other_base=True`).

Отпечаток (`base_fingerprint`) -- sha256 по контракту, малым тензорам базы
(нормировки, миксы token-shift, k_k / k_a / r_k) и по строке плотного веса
двух проекций (первой и последней), всё приведено к bf16: одна и та же база,
загруженная в bf16 и в fp32, даёт один отпечаток. Плотная модель, её
REDUCTION и её COMPRESSION -- ТРИ РАЗНЫЕ базы (малые тензоры у двух пресетов
общие, их различают проекции). Отпечаток -- базы, а не адаптеров поверх неё.
Счёт воспроизводится в SwiftRWKV бит в бит, файлы между библиотеками
переносимы.
"""
import hashlib
from typing import Iterable, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from .state import RWKVState

FORMAT = "rwkv-state"
VERSION = "1"
TENSORS = ("wkv", "tmix_shift", "cmix_shift")

# Малые тензоры блока, по которым считается отпечаток базы. Порядок -- часть
# формата: тот же список в SwiftRWKV.
_FP_BLOCK = ("ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias",
             "tmix.x_r", "tmix.x_w", "tmix.x_k", "tmix.x_v", "tmix.x_a", "tmix.x_g",
             "tmix.k_k", "tmix.k_a", "tmix.r_k", "tmix.ln_x.weight", "tmix.ln_x.bias",
             "cmix.x_k")
_FP_STEM = ("ln0.weight", "ln0.bias", "ln_out.weight", "ln_out.bias")


class InitialState(nn.Module):
    """Состояние одного слоя, батч 1. Параметр блока: `blocks.N.state0.*`."""

    def __init__(self, wkv: mx.array, tmix_shift: mx.array, cmix_shift: mx.array):
        super().__init__()
        self.wkv = wkv                  # [H, S, S] fp32 -- всегда
        self.tmix_shift = tmix_shift    # [1, 1, D]
        self.cmix_shift = cmix_shift    # [1, 1, D]


def _blocks(model):
    blocks = getattr(model, "blocks", None)
    if not blocks or not hasattr(blocks[0], "tmix") or not hasattr(blocks[0].tmix, "H"):
        raise TypeError("начальное состояние подключается к RWKV7X070 (нужны model.blocks[i].tmix)")
    return blocks


def contract(model) -> dict:
    """Что обязано совпасть у состояния и модели."""
    blocks = _blocks(model)
    t = blocks[0].tmix
    return {"n_layer": len(blocks), "n_embd": t.H * t.S, "n_head": t.H, "head_size": t.S}


def zero_state(model) -> RWKVState:
    c = contract(model)
    L, H, S, D = c["n_layer"], c["n_head"], c["head_size"], c["n_embd"]
    return RWKVState(wkv=mx.zeros((L, 1, H, S, S), dtype=mx.float32),
                     tmix_shift=mx.zeros((L, 1, 1, D), dtype=mx.float32),
                     cmix_shift=mx.zeros((L, 1, 1, D), dtype=mx.float32))


def _check(model, state: RWKVState):
    c = contract(model)
    L, H, S, D = c["n_layer"], c["n_head"], c["head_size"], c["n_embd"]
    want = {"wkv": (L, 1, H, S, S), "tmix_shift": (L, 1, 1, D), "cmix_shift": (L, 1, 1, D)}
    for name, shape in want.items():
        got = tuple(getattr(state, name).shape)
        if got != shape:
            raise ValueError(f"состояние не от этой модели: {name} {got}, у модели {shape} "
                             f"(слоёв {L}, D {D}, голов {H}, размер головы {S}; батч обязан быть 1)")


def attach_state(model, state: Optional[RWKVState] = None,
                 train: Iterable[str] = ()) -> None:
    """Подключить состояние: модель стартует с него всюду, где вызывающий не
    передал своё. `state=None` -- нули (заготовка под обучение).

    train -- что из состояния обучается: () -- ничего (инференс), ("wkv",) --
    матрица WKV (state-tuning по умолчанию), ("wkv", "tmix_shift",
    "cmix_shift") -- ещё и сдвиги. Остальная модель не трогается: заморозить
    базу -- дело вызывающего (`prepare_state_tuning`).
    """
    blocks = _blocks(model)
    if state is None:
        state = zero_state(model)
    _check(model, state)
    train = tuple(train)
    for name in train:
        if name not in TENSORS:
            raise ValueError(f"train: {name!r}; есть {TENSORS}")
    for l, blk in enumerate(blocks):
        s0 = InitialState(wkv=state.wkv[l, 0].astype(mx.float32),
                          tmix_shift=state.tmix_shift[l].astype(mx.float32),
                          cmix_shift=state.cmix_shift[l].astype(mx.float32))
        s0.freeze()
        if train:
            s0.unfreeze(keys=list(train))
        blk.state0 = s0
    mx.eval([b.state0.parameters() for b in blocks])


def detach_state(model) -> Optional[RWKVState]:
    """Убрать подключённое состояние; возвращает его (или None)."""
    st = attached_state(model)
    for blk in _blocks(model):
        if "state0" in blk:
            del blk["state0"]
    return st


def attached_state(model) -> Optional[RWKVState]:
    """Подключённое состояние как `RWKVState` с батчем 1 (или None)."""
    blocks = _blocks(model)
    if "state0" not in blocks[0]:
        return None
    return RWKVState(
        wkv=mx.stack([b.state0.wkv[None] for b in blocks], axis=0),
        tmix_shift=mx.stack([b.state0.tmix_shift for b in blocks], axis=0),
        cmix_shift=mx.stack([b.state0.cmix_shift for b in blocks], axis=0),
    ).eval()


def trained_parts(model) -> Tuple[str, ...]:
    blocks = _blocks(model)
    if "state0" not in blocks[0]:
        return ()
    names = {k for k, _ in tree_flatten(blocks[0].state0.trainable_parameters())}
    return tuple(n for n in TENSORS if n in names)


def prepare_state_tuning(model, init: Optional[RWKVState] = None,
                         train: Iterable[str] = ("wkv",)) -> int:
    """Заморозить ВСЮ модель и подключить обучаемое состояние (с нулей либо с
    `init` -- например, с состояния после промпта). Возвращает число
    обучаемых чисел."""
    model.freeze()
    attach_state(model, init, train=train)
    return sum(v.size for _, v in tree_flatten(model.trainable_parameters()))


def fold_prompt(model, ids, state: Optional[RWKVState] = None) -> RWKVState:
    """Свернуть промпт в состояние (батч 1). Старт -- `state`, а без него
    подключённое состояние (или нули): «обученное -> промпт -> кэш»."""
    ids = mx.array(ids).reshape(1, -1)
    return model.states(ids, state=state).stop_gradient().eval()


# ── Отпечаток базы ──────────────────────────────────────────────────────

def fingerprint_names(n_layer: int):
    names = list(_FP_STEM)
    for l in range(n_layer):
        names += [f"blocks.{l}.{k}" for k in _FP_BLOCK]
    return names


def fingerprint_rows(n_layer: int):
    """Проекции, из плотного веса которых в отпечаток идёт строка 0."""
    return ["blocks.0.tmix.r_proj", f"blocks.{n_layer - 1}.cmix.value"]


def base_fingerprint(model) -> str:
    """sha256 (hex) по контракту и малым тензорам базы в bf16 -- см. шапку."""
    c = contract(model)
    flat = dict(tree_flatten(model.parameters()))
    h = hashlib.sha256()
    h.update(f"rwkv-state-base-v1\n{c['n_layer']},{c['n_embd']},{c['n_head']},{c['head_size']}\n".encode())
    for name in fingerprint_names(c["n_layer"]):
        if name not in flat:
            raise KeyError(f"отпечаток базы: у модели нет тензора {name}")
        a = flat[name].astype(mx.bfloat16).astype(mx.float32).reshape(-1)
        h.update(name.encode() + b"\0")
        h.update(np.asarray(a, dtype="<f4").tobytes())
    # Малые тензоры у REDUCTION и COMPRESSION одной модели ОДНИ И ТЕ ЖЕ (16 бит
    # в обоих пресетах) -- пресеты различают большие матрицы. Берётся по одной
    # строке плотного (деквантованного) веса двух проекций: первой и последней.
    from ..lora.lora import LoRALinear, dense_weight
    for name in fingerprint_rows(c["n_layer"]):
        mod = model
        for part in name.split("."):
            mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
        if isinstance(mod, LoRALinear):        # отпечаток -- базы, не адаптеров
            mod = mod.linear
        a = dense_weight(mod, mx.float32)[0].astype(mx.bfloat16).astype(mx.float32).reshape(-1)
        h.update(name.encode() + b"\0")
        h.update(np.asarray(a, dtype="<f4").tobytes())
    return h.hexdigest()


# ── Файл ────────────────────────────────────────────────────────────────

def save_state(path: str, state: RWKVState, model, kind: str = "tuned",
               trained: Iterable[str] = ("wkv",), note: str = "") -> dict:
    """Записать состояние отдельным файлом. `model` -- база, на которой оно
    получено: из неё берутся контракт и отпечаток.

    kind:    "tuned" (обучено), "prompt" (свёрнутый промпт), "tuned+prompt".
    trained: что обучалось (для "prompt" -- пусто).
    note:    свободная строка (имя базы, пресет) -- для человека, не проверяется.
    """
    _check(model, state)
    meta = {"format": FORMAT, "version": VERSION, "kind": kind,
            "trained": ",".join(trained), "base_fingerprint": base_fingerprint(model),
            "note": note}
    meta.update({k: str(v) for k, v in contract(model).items()})
    arrays = {"wkv": state.wkv.astype(mx.float32),
              "tmix_shift": state.tmix_shift.astype(mx.float32),
              "cmix_shift": state.cmix_shift.astype(mx.float32)}
    mx.save_safetensors(path, arrays, metadata=meta)
    return meta


class StateBaseMismatch(ValueError):
    """Состояние получено на другой базе."""


def load_state(path: str, model=None, allow_other_base: bool = False) -> Tuple[RWKVState, dict]:
    """Прочитать файл состояния. С `model` проверяются контракт (несовпадение
    -- отказ всегда) и отпечаток базы (несовпадение -- `StateBaseMismatch`,
    если не `allow_other_base`)."""
    arrays, meta = mx.load(path, return_metadata=True)
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path}: не файл состояния (format={meta.get('format')!r})")
    if meta.get("version") != VERSION:
        raise ValueError(f"{path}: версия формата {meta.get('version')!r}, знаем {VERSION!r}")
    missing = [t for t in TENSORS if t not in arrays]
    if missing:
        raise ValueError(f"{path}: нет тензоров {missing}")
    state = RWKVState(wkv=arrays["wkv"], tmix_shift=arrays["tmix_shift"],
                      cmix_shift=arrays["cmix_shift"])
    if model is not None:
        c = contract(model)
        for k, v in c.items():
            if meta.get(k) != str(v):
                raise ValueError(f"{path}: {k}={meta.get(k)} в файле, {v} у модели -- состояние от другой архитектуры")
        _check(model, state)
        fp = base_fingerprint(model)
        if meta.get("base_fingerprint") != fp and not allow_other_base:
            raise StateBaseMismatch(
                f"{path}: состояние получено на другой базе (отпечаток {meta.get('base_fingerprint', '')[:12]}…, "
                f"у модели {fp[:12]}…; примечание файла: {meta.get('note', '')!r}). "
                "На чужой базе оно даёт другой результат; allow_other_base=True -- подключить всё равно.")
    return state, meta
