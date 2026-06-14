"""
rwkv_metal.lora.finetune
========================
High-level LoRA / QLoRA fine-tuning for RWKV-7, encoding the validated recipe
from the project notes:

    - forward via mlx.nn.utils.checkpoint(block)   (memory + speed)
    - nn.value_and_grad(model, loss_fn)            (respects freeze(); NOT mx.value_and_grad)
    - NO mx.compile                                (5x slower for LoRA on big models)
    - mx.set_cache_limit(int(1.5e9))               (stops swap under free RAM)
    - QLoRA: quantize only the big frozen matrices (cmix.key/value, head, emb)
    - lr=1e-4, alpha=16, clip=1.0 for pretrained bases (aggressive lr diverges)
    - large effective batch via grad accumulation (low micro-batch)

Quick start:
    import rwkv_metal as rk
    from rwkv_metal.lora import LoRAConfig, finetune, quantize_base_model

    model = ...                                   # an RWKV7 / RWKV7X070 with loaded weights
    quantize_base_model(model, bits=4)            # QLoRA: 4-bit big frozen matrices
    model, info = rk.add_lora(model, rank=16, alpha=16.0,
                              quantize_base=4, layers=range(12, 24))

    cfg = LoRAConfig(lr=1e-4, grad_accum=8, max_steps=2000, grad_checkpoint=True)
    finetune(model, batches, cfg)                 # batches: iterable of (x, y) mx.arrays
"""

import time
from dataclasses import dataclass
from typing import Iterable, Optional, Callable, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .lora import save_lora


# ── Big frozen matrices to quantize for QLoRA ───────────────────────────────
# Low-rank matrices (w/a/g/v) have ranks not divisible by group_size and carry
# little memory; quantizing them hurts dynamics. Quantize only the big ones.
BIG_QUANT_TARGETS = ("cmix.key", "cmix.value", "head", "emb")


@dataclass
class LoRAConfig:
    # ── Optimization ──────────────────────────────────────────────────────
    lr:           float = 1e-4          # pretrained bases: keep small (2e-3 diverges)
    grad_clip:    float = 1.0
    weight_decay: float = 0.0
    beta1:        float = 0.9
    beta2:        float = 0.95
    adam_eps:     float = 1e-8

    # ── Schedule ──────────────────────────────────────────────────────────
    max_steps:    int = 1000
    grad_accum:   int = 1               # effective batch via accumulation; keep micro-batch low
    warmup_steps: int = 0

    # ── Memory recipe ─────────────────────────────────────────────────────
    grad_checkpoint:  bool  = True      # nn.utils.checkpoint per block (big RAM + speed win)
    cache_limit_gb:   float = 1.5       # mx.set_cache_limit; <=0 disables

    # ── Logging / checkpoints ─────────────────────────────────────────────
    log_every:    int = 10
    save_every:   int = 0               # 0 = only at end
    adapter_path: str = "lora_adapters.safetensors"


def quantize_base_model(model, bits: int = 4, group_size: int = 64,
                        targets: Tuple[str, ...] = BIG_QUANT_TARGETS):
    """Quantize ONLY the big frozen matrices (QLoRA) — in place.

    Quantizes layers whose parameter path ends with one of `targets`
    (cmix.key/value, head, emb by default). Low-rank tmix matrices are left in
    full precision on purpose (their ranks aren't multiples of group_size and
    quantizing them degrades the in-context dynamics).

    Call this BEFORE add_lora(..., quantize_base=bits) so the LoRA target
    projections (r/k/v/o_proj) get quantized by add_lora, and the remaining big
    matrices get quantized here.
    """
    def predicate(path, module):
        return hasattr(module, "to_quantized") and any(
            path.endswith(t) for t in targets
        )

    nn.quantize(model, group_size=group_size, bits=bits,
                class_predicate=predicate)
    mx.eval(model.parameters())
    return model


def _make_loss_fn(cfg: LoRAConfig):
    """Build the language-modeling loss (fp32 reduction for bf16/QLoRA models)."""
    def loss_fn(model, x, y):
        return model.loss(x, y).astype(mx.float32)

    return loss_fn


def _lr_at(step: int, cfg: LoRAConfig) -> float:
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    return cfg.lr


def finetune(model, batches: Iterable[Tuple[mx.array, mx.array]], cfg: LoRAConfig,
             on_step: Optional[Callable] = None):
    """Fine-tune LoRA adapters on `model` using the validated recipe.

    Args:
        model:   an RWKV7 / RWKV7X070 already wrapped with add_lora(...). Only
                 the adapters should be trainable (add_lora freezes the base).
        batches: iterable yielding (x, y) pairs of mx.array token ids [B, T].
                 Provide enough items for `max_steps * grad_accum` micro-steps;
                 it's fine to cycle a small dataset.
        cfg:     LoRAConfig.
        on_step: optional callback(step, loss_float, peak_gb) for custom logging.

    Returns:
        dict with final loss, steps, and adapter_path.
    """
    if cfg.cache_limit_gb and cfg.cache_limit_gb > 0:
        mx.set_cache_limit(int(cfg.cache_limit_gb * 1e9))

    # Enable per-block gradient checkpointing inside the model body.
    if hasattr(model, "_grad_ckpt"):
        model._grad_ckpt = cfg.grad_checkpoint

    loss_fn = _make_loss_fn(cfg)

    opt = optim.AdamW(
        learning_rate=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )

    # nn.value_and_grad respects freeze() (only adapters get gradients).
    grad_fn = nn.value_and_grad(model, loss_fn)

    it = iter(batches)

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(batches)
            return next(it)

    GB = 1e9
    last_loss = float("nan")
    t0 = time.time()
    print(f"LoRA fine-tune | steps={cfg.max_steps} | grad_accum={cfg.grad_accum} "
          f"| grad_ckpt={cfg.grad_checkpoint}")
    print("─" * 60)

    for step in range(cfg.max_steps):
        opt.learning_rate = _lr_at(step, cfg)

        if cfg.grad_accum == 1:
            x, y = next_batch()
            loss, grads = grad_fn(model, x, y)
            grads, norm = optim.clip_grad_norm(grads, max_norm=cfg.grad_clip)
            opt.update(model, grads)
            mx.eval(loss, model.state, opt.state)
            last_loss = loss.item()
        else:
            # Gradient accumulation — eval after each micro-step (avoid lazy blowup).
            x0, y0 = next_batch()
            total_loss, total_grads = grad_fn(model, x0, y0)
            mx.eval(total_loss, total_grads)
            for _ in range(1, cfg.grad_accum):
                xi, yi = next_batch()
                li, gi = grad_fn(model, xi, yi)
                mx.eval(li, gi)
                total_loss = total_loss + li
                total_grads = _tree_add(total_grads, gi)
                mx.eval(total_grads)
            total_grads = _tree_scale(total_grads, 1.0 / cfg.grad_accum)
            total_loss = total_loss / cfg.grad_accum
            grads, norm = optim.clip_grad_norm(total_grads, max_norm=cfg.grad_clip)
            opt.update(model, grads)
            mx.eval(total_loss, model.state, opt.state)
            last_loss = total_loss.item()

        peak = mx.get_peak_memory() / GB
        if on_step is not None:
            on_step(step, last_loss, peak)

        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.max_steps - 1):
            dt = time.time() - t0
            print(f"  step {step:5d} | loss {last_loss:.4f} | "
                  f"grad_norm {norm.item():.3f} | peak {peak:.2f} GB | {dt:.1f}s")

        if cfg.save_every and step > 0 and step % cfg.save_every == 0:
            save_lora(model, cfg.adapter_path)

    save_lora(model, cfg.adapter_path)
    print("─" * 60)
    print(f"Done. final loss {last_loss:.4f} | adapters -> {cfg.adapter_path}")
    return {"final_loss": last_loss, "steps": cfg.max_steps,
            "adapter_path": cfg.adapter_path}


# ── small tree helpers (avoid importing whole utils namespace) ──────────────

def _tree_add(a, b):
    from mlx.utils import tree_map
    return tree_map(lambda x, y: x + y, a, b)


def _tree_scale(a, s):
    from mlx.utils import tree_map
    return tree_map(lambda x: x * s, a)
