"""
rwkv_metal.pretrain.config
==========================
Конфигурация предобучения RWKV-7.

Пример использования:
    from rwkv_metal.pretrain import PretrainConfig

    cfg = PretrainConfig(
        # Архитектура
        n_layer    = 18,
        n_embd     = 256,
        vocab_size = 21248,

        # Данные
        train_data = "data/train.bin",
        val_data   = "data/val.bin",

        # Сколько обучать
        max_tokens = 3_000_000_000,
    )

    # Или использовать готовый пресет:
    cfg = PretrainConfig.from_preset("25m")
    # эквивалентно функции preset("25m")
"""

from dataclasses import dataclass
from typing import Literal, Optional
import math


@dataclass
class PretrainConfig:
    # ── Архитектура ───────────────────────────────────────────────────────────
    n_layer:    int = 12
    n_embd:     int = 768
    vocab_size: int = 21248
    head_size:  int = 64          # размер головы внимания; n_embd должно делиться на head_size

    @property
    def n_head(self) -> int:
        return self.n_embd // self.head_size

    # ── Данные ────────────────────────────────────────────────────────────────
    train_data:  str           = "data/train.bin"   # .bin (uint16) или .txt
    val_data:    str           = "data/val.bin"
    tokenizer:   Optional[str] = None               # путь к tokenizer.json; нужен если данные .txt
    ctx_len:     int           = 512
    batch_size:  int           = 8
    grad_accum:  int           = 1                  # gradient accumulation; эфф. batch = batch_size × grad_accum

    # ── Сколько обучать (одно из двух; max_steps имеет приоритет) ─────────────
    max_steps:  Optional[int] = None
    max_tokens: Optional[int] = 3_000_000_000

    def resolve_max_steps(self) -> int:
        """Возвращает max_steps: явный или вычисленный из max_tokens."""
        if self.max_steps is not None:
            return self.max_steps
        if self.max_tokens is not None:
            tokens_per_step = self.batch_size * self.ctx_len * self.grad_accum
            return math.ceil(self.max_tokens / tokens_per_step)
        raise ValueError("Укажи max_steps или max_tokens")

    # ── Оптимизатор ───────────────────────────────────────────────────────────
    lr:           float = 1.5e-3
    lr_min:       float = 1e-4
    lr_schedule:  Literal["cosine", "linear", "constant"] = "cosine"
    warmup_steps: int   = 200
    weight_decay: float = 0.0
    beta1:        float = 0.9
    beta2:        float = 0.95
    adam_eps:     float = 1e-18
    grad_clip:    float = 1.0

    # ── Железо ────────────────────────────────────────────────────────────────
    dtype:           Literal["bfloat16", "float32"] = "bfloat16"
    grad_checkpoint: bool = False     # экономит память (~2.5× меньше) за счёт ~15% скорости

    # ── Чекпоинты ─────────────────────────────────────────────────────────────
    checkpoint_dir: str  = "checkpoints"
    resume:         bool = True        # автоматически продолжить с последнего чекпоинта
    save_every:     int  = 500
    save_best_only: bool = False       # сохранять только если val loss улучшился

    # ── Валидация и логи ──────────────────────────────────────────────────────
    eval_every:   int  = 500
    eval_batches: int  = 20
    log_every:    int  = 50

    # ── Weights & Biases ──────────────────────────────────────────────────────
    wandb:         bool = False
    wandb_project: str  = "rwkv-metal"
    wandb_run:     Optional[str] = None   # None = автоимя

    # ── Валидация конфига ─────────────────────────────────────────────────────
    def __post_init__(self):
        assert self.n_embd % self.head_size == 0, (
            f"n_embd ({self.n_embd}) должно делиться на head_size ({self.head_size})"
        )
        assert self.grad_accum >= 1
        assert self.ctx_len > 0 and self.batch_size > 0
        if self.max_steps is None and self.max_tokens is None:
            raise ValueError("Укажи max_steps или max_tokens")

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "PretrainConfig":
        """Создать конфиг из готового пресета (см. PRESETS) с переопределениями.

        Эквивалентно функции preset(name, **overrides).
        """
        return preset(name, **overrides)

    def summary(self) -> str:
        """Человекочитаемое резюме конфига."""
        n_params = self._estimate_params()
        eff_batch = self.batch_size * self.grad_accum
        max_steps = self.resolve_max_steps()
        lines = [
            f"{'─'*50}",
            f"  Модель:    {self.n_layer}L × {self.n_embd}d  (~{n_params/1e6:.1f}M параметров)",
            f"  Vocab:     {self.vocab_size}  |  ctx: {self.ctx_len}",
            f"  Batch:     {self.batch_size} × grad_accum {self.grad_accum} = {eff_batch} эфф.",
            f"  Обучение:  {max_steps:,} шагов  (~{max_steps*eff_batch*self.ctx_len/1e9:.2f}B токенов)",
            f"  LR:        {self.lr} → {self.lr_min}  ({self.lr_schedule}, warmup {self.warmup_steps})",
            f"  dtype:     {self.dtype}  |  grad_ckpt: {self.grad_checkpoint}",
            f"  Данные:    {self.train_data}",
            f"{'─'*50}",
        ]
        return "\n".join(lines)

    def _estimate_params(self) -> int:
        """Приблизительный подсчёт параметров модели."""
        D, V, L = self.n_embd, self.vocab_size, self.n_layer
        emb   = V * D                          # embedding
        head  = V * D                          # lm head
        block = (
            4 * D * D +                        # r/k/v/o proj
            3 * (D * 64 + 64 * D) +            # w/a/g lora (A+B)
            D * 64 + 64 * D +                  # v lora
            D * (4 * D) + (4 * D) * D +        # cmix key/value
            4 * D                              # layernorms
        )
        return emb + head + L * block


# ── Готовые пресеты ───────────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {
    # Маленькие модели — для экспериментов и телефонов
    "25m": dict(
        n_layer=18, n_embd=256, head_size=64,
        ctx_len=512, batch_size=18,
        lr=1.5e-3, warmup_steps=200,
    ),
    "50m": dict(
        n_layer=24, n_embd=384, head_size=64,
        ctx_len=512, batch_size=12,
        lr=1.2e-3, warmup_steps=300,
    ),
    # Средние
    "170m": dict(
        n_layer=24, n_embd=768, head_size=64,
        ctx_len=1024, batch_size=4,
        lr=6e-4, warmup_steps=500,
    ),
    "430m": dict(
        n_layer=24, n_embd=1024, head_size=64,
        ctx_len=1024, batch_size=2, grad_accum=4,
        lr=4e-4, warmup_steps=500,
    ),
}


def preset(name: str, **overrides) -> PretrainConfig:
    """
    Создать конфиг из пресета с возможностью переопределить любые поля.

    Пример:
        cfg = preset("25m", train_data="data/en/train.bin", vocab_size=21248)
    """
    if name not in PRESETS:
        raise ValueError(f"Неизвестный пресет '{name}'. Доступны: {list(PRESETS)}")
    params = {**PRESETS[name], **overrides}
    return PretrainConfig(**params)
