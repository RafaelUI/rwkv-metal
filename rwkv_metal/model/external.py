"""
rwkv_metal.model.external
=========================
Загрузка внешних RWKV-7 чекпоинтов, лежащих в формате «каталог с
`config.json` + `model.safetensors` (+ `tokenizer*.json`)» — типичная
раскладка для моделей, переведённых из `flash-linear-attention`
(метаданные `origin: fla-remap`).

Отличие от `load_pretrained()`: тот принимает официальный `.pth` x070 и
переименовывает тензоры (`att.w1` → `tmix.w_lora_A` и т.д.). Здесь имена
уже совпадают с `RWKV7X070`, потому что fla-remap использует ту же
раскладку, — конвертация имён не нужна.

Целевая архитектура — именно `RWKV7X070`, а НЕ from-scratch `RWKV7`:
fla реализует официальную RWKV-7, а `RWKV7` отличается от неё по шести
пунктам (tanh в iclr, gate наружу, LayerNorm вместо GroupNorm по головам,
порядок ln_x/bonus, межблочный перенос token-shift). Имена и формы тензоров
у них при этом совпадают, так что ошибиться архитектурой можно молча: модель
загрузится «успешно» и будет выдавать правдоподобный, но испорченный выход.
Проверено измеримо на ru60m (40 русских отрывков): та же самая раскладка
весов даёт PPL 254 в `RWKV7` и PPL 42.3 в `RWKV7X070`, при 16000 у
случайных весов — см. `tools/verify_local_checkpoint.py`.

Ранги low-rank блоков определяются по самим весам, а не по config.json
(fla считает их своей формулой, и в конфиг они обычно не попадают).
"""
import json
import os
from typing import Optional, Tuple

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .rwkv7_x070 import RWKV7X070
from .rwkv7 import RWKV7
from ..pretrain.config import PretrainConfig

_KINDS = ("w", "a", "v", "g")


def infer_lora_ranks(weights: dict, default: int = 64) -> dict:
    """Ранги low-rank блоков tmix, снятые с фактических форм в чекпоинте."""
    ranks = {}
    for kind in _KINDS:
        r = default
        for k, v in weights.items():
            if k.endswith(f"tmix.{kind}_lora_A.weight"):
                r = int(v.shape[0])
                break
        ranks[kind] = r
    return ranks


def load_local_rwkv7(model_dir: str, arch: str = "x070", verbose: bool = True,
                     strict: bool = True) -> Tuple[object, PretrainConfig]:
    """model_dir: каталог с `config.json` и `model.safetensors`.

    arch: "x070" (по умолчанию — официальная архитектура, то, что выдаёт
    fla-remap) или "scratch" (`RWKV7`, только если чекпоинт действительно
    обучался в rwkv-metal). Выбор архитектуры не проверяется формами —
    проверяйте перплексией, `tools/verify_local_checkpoint.py`.

    strict=True роняет загрузку при любом расхождении имён или форм:
    молчаливо недогруженный чекпоинт даёт правдоподобные, но бессмысленные
    эмбеддинги, и заметить это потом гораздо дороже.
    """
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        raw = json.load(f)

    weights = dict(mx.load(os.path.join(model_dir, "model.safetensors")))
    ranks = infer_lora_ranks(weights)

    cfg = PretrainConfig(
        n_layer=raw["n_layer"],
        n_embd=raw["n_embd"],
        vocab_size=raw["vocab_size"],
        head_size=raw.get("head_size", 64),
        ctx_len=raw.get("ctx_len", 512),
    )

    if arch == "x070":
        model = RWKV7X070(cfg, ranks=ranks)
    elif arch == "scratch":
        model = RWKV7(cfg)
    else:
        raise ValueError(f"неизвестная arch: {arch!r} (ожидалось 'x070' или 'scratch')")

    params = dict(tree_flatten(model.parameters()))

    # fla хранит cmix.x_k как [D], модель держит [1,1,D] — та же математика,
    # другая раскладка.
    for k in list(weights):
        if k.endswith("cmix.x_k") and weights[k].ndim == 1:
            weights[k] = weights[k].reshape(1, 1, -1)

    missing = sorted(set(params) - set(weights))
    extra = sorted(set(weights) - set(params))
    mismatch = [(k, tuple(weights[k].shape), tuple(params[k].shape))
                for k in sorted(set(params) & set(weights))
                if tuple(weights[k].shape) != tuple(params[k].shape)]

    if verbose:
        print(f"[load_local_rwkv7] {raw.get('model_name', model_dir)} as {arch}: "
              f"n_layer={cfg.n_layer} n_embd={cfg.n_embd} vocab={cfg.vocab_size} "
              f"head_size={cfg.head_size}")
        print(f"  lora ranks {ranks} | tensors: ckpt {len(weights)}, model {len(params)} "
              f"| missing {len(missing)}, extra {len(extra)}, mismatched {len(mismatch)}")

    if strict and (missing or mismatch):
        raise ValueError(
            "чекпоинт не сходится с моделью:\n"
            f"  missing:  {missing[:10]}\n"
            f"  mismatch: {mismatch[:10]}\n"
            f"  extra:    {extra[:10]}"
        )

    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model, cfg


def load_local_tokenizer(model_dir: str):
    """Ищет tokenizer*.json в каталоге модели и возвращает BPETokenizer."""
    from ..tokenizer import BPETokenizer
    cfg_path = os.path.join(model_dir, "config.json")
    name: Optional[str] = None
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            name = (json.load(f).get("tokenizer") or {}).get("path")
    if not name:
        cands = [f for f in os.listdir(model_dir)
                 if f.startswith("tokenizer") and f.endswith(".json")]
        if not cands:
            raise FileNotFoundError(f"нет tokenizer*.json в {model_dir}")
        name = sorted(cands)[0]
    return BPETokenizer(os.path.join(model_dir, name))
