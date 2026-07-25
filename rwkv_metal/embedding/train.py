"""
rwkv_metal.embedding.train
==========================
Contrastive fine-tuning trainer for embeddings, built on `nn.value_and_grad`
(freeze()-aware -- like `rwkv_metal.lora.finetune`, NOT `mx.value_and_grad`)
so the SAME trainer works unmodified for:

  - full fine-tune (nothing frozen -- default recipe for small models,
    matches EmbeddingRWKV's own sft_curriculum recipe: --freeze_rwkv 0)
  - frozen-bottom-layers partial FT (call model.base.blocks[i].freeze() for
    i < k before finetune_embedding())
  - LoRA / QLoRA on top of a frozen (optionally 4/8-bit quantized via
    quantize_base_model) base -- add_lora(embedding_model.base, ...) first.
    This is the path that matters once models get too big to full-FT on
    Apple Silicon (a future 1.4B/4B embedding model): QLoRA never
    materializes full-precision optimizer state for the frozen base, only
    for the small adapters + head.

The pooling head is a SIBLING of the base model (EmbeddingModel.head), never
a submodule of it -- add_lora()/quantize_base_model() call model.freeze() on
the base tree, which would silently freeze the head too if it lived inside
that tree.
"""
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from .heads import EmbeddingHead
from .gradcache import gradcache_value_and_grad


def _l2_normalize(x: mx.array) -> mx.array:
    return x / mx.sqrt((x * x).sum(axis=-1, keepdims=True) + 1e-12)


class EmbeddingModel(nn.Module):
    """base (RWKV7 / RWKV7X070, optionally LoRA/QLoRA-wrapped) + pooling head."""

    def __init__(self, base, dim: int = None, head_hidden: int = None):
        super().__init__()
        self.base = base
        dim = dim or base.config.n_embd
        self.head = EmbeddingHead(dim, head_hidden)

    def embed(self, idx: mx.array, pool_idx: mx.array) -> mx.array:
        h = self.base.body(idx)  # [B, T, D]
        pooled = mx.take_along_axis(h, pool_idx.reshape(-1, 1, 1), axis=1).squeeze(1)  # [B, D]
        return _l2_normalize(self.head(pooled))

    def __call__(self, idx: mx.array, pool_idx: mx.array) -> mx.array:
        return self.embed(idx, pool_idx)


@dataclass
class EmbedTrainConfig:
    # ── Optimization ──────────────────────────────────────────────────────
    lr: float = 2e-5
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.99
    adam_eps: float = 1e-8
    temperature: float = 0.05

    # ── Schedule ──────────────────────────────────────────────────────────
    max_steps: int = 1000
    grad_accum: int = 1
    warmup_steps: int = 0
    lr_schedule: str = "cosine"   # "cosine" | "linear" | "constant" (post-warmup decay)
    lr_min: float = 0.0

    # ── Memory recipe ─────────────────────────────────────────────────────
    grad_checkpoint: bool = True
    cache_limit_gb: float = 1.5
    gradcache_chunk: int = 0      # >0 + a GradCacheSpec => GradCache; rows per chunk

    # ── Logging / checkpoints ─────────────────────────────────────────────
    log_every: int = 10
    save_every: int = 0
    checkpoint_path: str = "embedding_model.safetensors"


def _lr_at(step: int, cfg: EmbedTrainConfig) -> float:
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps

    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(max(progress, 0.0), 1.0)

    if cfg.lr_schedule == "cosine":
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    elif cfg.lr_schedule == "linear":
        decay = 1.0 - progress
    else:  # "constant"
        decay = 1.0

    return cfg.lr_min + (cfg.lr - cfg.lr_min) * decay


def save_embedding_model(model: EmbeddingModel, path: str):
    """Saves TRAINABLE parameters only: head always; base params too if
    full-FT/unfrozen-layers, or just LoRA adapters if the base is
    LoRA/QLoRA-wrapped. Load back with model.update(tree_unflatten(...))."""
    mx.save_safetensors(path, dict(tree_flatten(model.trainable_parameters())))


def finetune_embedding(model: EmbeddingModel, batches: Iterable, cfg: EmbedTrainConfig,
                        compute_loss: Callable, on_step: Optional[Callable] = None,
                        gradcache_spec=None):
    """batches: iterable yielding whatever `compute_loss(model, batch)` expects
    -- see rwkv_metal.embedding.tasks (retrieval_loss / sts_loss /
    classification_loss) and the matching batchers in
    rwkv_metal.embedding.dataset (TripletBatcher / ClassificationBatcher).

    Curriculum training: call this once per task, sequentially, with that
    task's batcher + compute_loss (matches EmbeddingRWKV's own sft_curriculum
    staging -- one code path, run three times).

    gradcache_spec: optional GradCacheSpec (tasks.RETRIEVAL_GC / tasks.STS_GC).
    When given together with cfg.gradcache_chunk > 0, the step runs under
    GradCache: activation memory is bounded by the chunk size while the loss
    still sees the whole batch, so the contrastive negative pool can be far
    larger than activations would otherwise allow. Mathematically identical
    to the eager path (verified to ~3e-8), just a different memory schedule.
    """
    if cfg.cache_limit_gb and cfg.cache_limit_gb > 0:
        mx.set_cache_limit(int(cfg.cache_limit_gb * 1e9))

    if hasattr(model.base, "_grad_ckpt"):
        model.base._grad_ckpt = cfg.grad_checkpoint

    use_gc = gradcache_spec is not None and cfg.gradcache_chunk > 0

    def loss_fn(m, batch):
        return compute_loss(m, batch, cfg.temperature)

    opt = optim.AdamW(learning_rate=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                       eps=cfg.adam_eps, weight_decay=cfg.weight_decay)

    # nn.value_and_grad respects freeze() -- only trainable leaves get grads.
    # (mx.value_and_grad would differentiate the whole tree and ignore freeze().)
    eager_grad_fn = nn.value_and_grad(model, loss_fn)

    def grad_fn(m, batch):
        if not use_gc:
            return eager_grad_fn(m, batch)
        chunks = gradcache_spec.split(batch, cfg.gradcache_chunk)
        return gradcache_value_and_grad(
            m, chunks, gradcache_spec.embed,
            lambda fields: gradcache_spec.loss(fields, cfg.temperature),
        )

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
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    n_total = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Embedding fine-tune | steps={cfg.max_steps} | grad_accum={cfg.grad_accum} "
          f"| trainable {n_train/1e6:.2f}M / {n_total/1e6:.2f}M "
          f"({100*n_train/max(1,n_total):.2f}%)")
    print("-" * 60)

    for step in range(cfg.max_steps):
        opt.learning_rate = _lr_at(step, cfg)

        if cfg.grad_accum == 1:
            batch = next_batch()
            loss, grads = grad_fn(model, batch)
            grads, norm = optim.clip_grad_norm(grads, max_norm=cfg.grad_clip)
            opt.update(model, grads)
            mx.eval(loss, model.state, opt.state)
            last_loss = loss.item()
        else:
            batch = next_batch()
            total_loss, total_grads = grad_fn(model, batch)
            mx.eval(total_loss, total_grads)
            for _ in range(1, cfg.grad_accum):
                batch = next_batch()
                li, gi = grad_fn(model, batch)
                mx.eval(li, gi)
                total_loss = total_loss + li
                total_grads = tree_map(lambda a, b: a + b, total_grads, gi)
                mx.eval(total_grads)
            total_grads = tree_map(lambda g: g / cfg.grad_accum, total_grads)
            total_loss = total_loss / cfg.grad_accum
            grads, norm = optim.clip_grad_norm(total_grads, max_norm=cfg.grad_clip)
            opt.update(model, grads)
            mx.eval(total_loss, model.state, opt.state)
            last_loss = total_loss.item()

        peak = mx.get_peak_memory() / GB
        if on_step is not None:
            on_step(step, last_loss, peak)

        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.max_steps - 1):
            print(f"  step {step:5d} | loss {last_loss:.4f} | "
                  f"grad_norm {norm.item():.3f} | peak {peak:.2f} GB")

        if cfg.save_every and step > 0 and step % cfg.save_every == 0:
            save_embedding_model(model, cfg.checkpoint_path)

    save_embedding_model(model, cfg.checkpoint_path)
    print("-" * 60)
    print(f"Done. final loss {last_loss:.4f} -> {cfg.checkpoint_path}")
    return {"final_loss": last_loss, "steps": cfg.max_steps,
            "checkpoint_path": cfg.checkpoint_path}
