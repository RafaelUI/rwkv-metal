"""
rwkv_metal.pretrain.trainer
===========================
Цикл предобучения RWKV-7.

Пример:
    from rwkv_metal.pretrain import pretrain, preset

    pretrain(preset("25m",
        train_data = "data/train.bin",
        val_data   = "data/val.bin",
        vocab_size = 21248,
        max_tokens = 3_000_000_000,
    ))
"""

import os
import math
import time

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from .config import PretrainConfig
from .dataset import load_dataset


# ── LR schedule ──────────────────────────────────────────────────────────────

def _lr_schedule(step: int, cfg: PretrainConfig) -> float:
    max_steps = cfg.resolve_max_steps()

    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps

    progress = (step - cfg.warmup_steps) / max(1, max_steps - cfg.warmup_steps)
    progress = min(progress, 1.0)

    if cfg.lr_schedule == "cosine":
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    elif cfg.lr_schedule == "linear":
        decay = 1.0 - progress
    else:  # constant
        decay = 1.0

    return cfg.lr_min + (cfg.lr - cfg.lr_min) * decay


# ── Train step builders ───────────────────────────────────────────────────────

def _make_step_simple(model, optimizer, cfg: PretrainConfig):
    """Быстрый путь: один шаг без grad accumulation, mx.compile."""
    state = [model.state, optimizer.state]
    grad_clip = cfg.grad_clip

    def _step(x, y):
        def loss_fn(m, x, y):
            return m.loss(x, y).astype(mx.float32)
        loss, grads = mx.value_and_grad(loss_fn)(model, x, y)
        grads, norm = optim.clip_grad_norm(grads, max_norm=grad_clip)
        optimizer.update(model, grads)
        return loss, norm

    return mx.compile(_step, inputs=state, outputs=state)


def _make_step_accum(model, optimizer, cfg: PretrainConfig):
    """Grad accumulation: накапливаем градиенты grad_accum микро-шагов.

    КРИТИЧНО: mx.eval после каждого микро-шага и после накопления, иначе
    lazy-граф растёт неограниченно → OOM (см. HANDOFF).
    """
    grad_accum = cfg.grad_accum
    grad_clip  = cfg.grad_clip
    micro_state = [model.state]

    def _micro(x, y):
        def loss_fn(m, x, y):
            return m.loss(x, y).astype(mx.float32)
        return mx.value_and_grad(loss_fn)(model, x, y)

    compiled_micro = mx.compile(_micro, inputs=micro_state)

    def _step(xs, ys):
        total_loss, total_grads = compiled_micro(xs[0], ys[0])
        mx.eval(total_loss, total_grads)

        for i in range(1, grad_accum):
            loss_i, grads_i = compiled_micro(xs[i], ys[i])
            mx.eval(loss_i, grads_i)
            total_loss  = total_loss + loss_i
            total_grads = tree_map(lambda a, b: a + b, total_grads, grads_i)
            mx.eval(total_grads)

        total_grads = tree_map(lambda g: g / grad_accum, total_grads)
        total_loss  = total_loss / grad_accum
        grads, norm = optim.clip_grad_norm(total_grads, max_norm=grad_clip)
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state)
        return total_loss, norm

    return _step


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _ckpt_path(cfg: PretrainConfig, tag: str = "latest") -> str:
    name = f"rwkv7_{cfg.n_layer}l{cfg.n_embd}d_{tag}.npz"
    return os.path.join(cfg.checkpoint_dir, name)


def _step_path(cfg: PretrainConfig) -> str:
    return _ckpt_path(cfg, "latest").replace(".npz", ".step")


def _save(model, path: str):
    model.save_weights(path)


def _load_checkpoint(model, cfg: PretrainConfig) -> int:
    path = _ckpt_path(cfg, "latest")
    step_file = _step_path(cfg)
    if os.path.exists(path):
        model.load_weights(path)
        start_step = 0
        if os.path.exists(step_file):
            start_step = int(open(step_file).read().strip())
        print(f"  Продолжаем с шага {start_step:,}")
        return start_step
    return 0


# ── Eval ──────────────────────────────────────────────────────────────────────

def _eval(model, dataset, cfg: PretrainConfig) -> float:
    losses = []
    for i in range(cfg.eval_batches):
        x, y = dataset.batch(cfg.batch_size, i)
        loss = model.loss(x, y).astype(mx.float32)
        mx.eval(loss)
        losses.append(loss.item())
    return sum(losses) / len(losses)


# ── W&B ───────────────────────────────────────────────────────────────────────

def _init_wandb(cfg: PretrainConfig):
    try:
        import wandb
        wandb.init(
            project = cfg.wandb_project,
            name    = cfg.wandb_run,
            config  = {k: v for k, v in cfg.__dict__.items()
                       if not k.startswith("_")},
        )
        return wandb
    except ImportError:
        print("  [!] wandb не установлен: pip install wandb")
        return None


# ── Главная функция ───────────────────────────────────────────────────────────

def pretrain(cfg: PretrainConfig):
    """
    Запускает предобучение RWKV-7 по конфигу.

    Args:
        cfg: PretrainConfig с параметрами обучения
    """
    from ..model.rwkv7 import RWKV7, init_weights

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    max_steps = cfg.resolve_max_steps()

    print(cfg.summary())

    # ── Данные ───────────────────────────────────────────────────────────────
    print("Загрузка данных:")
    train_ds = load_dataset(cfg.train_data, cfg.ctx_len, cfg.tokenizer)
    val_ds   = load_dataset(cfg.val_data,   cfg.ctx_len, cfg.tokenizer)

    # Проверяем OOV токены
    if hasattr(train_ds, "validate"):
        check = train_ds.validate(cfg.vocab_size)
        if not check["ok"]:
            print(f"\n  [!] Предупреждение: найдены OOV токены!")
            for issue in check["issues"]:
                print(f"      {issue}")
            print(f"  Max токен: {check['max_token']}, vocab_size: {cfg.vocab_size}")
            print(f"  Это вызовет NaN при обучении. Исправь vocab_size или перетокенизируй.\n")

    # ── Модель ───────────────────────────────────────────────────────────────
    print("Инициализация модели...")
    model = RWKV7(cfg)
    model._grad_ckpt = cfg.grad_checkpoint

    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"  Параметры: {n_params/1e6:.1f}M")

    # ── Checkpoint / resume ──────────────────────────────────────────────────
    start_step = 0
    if cfg.resume:
        start_step = _load_checkpoint(model, cfg)

    if start_step == 0:
        print("  Инициализация весов RWKV-7...")
        model = init_weights(model)

    # ── dtype ────────────────────────────────────────────────────────────────
    # Разработчик выбирает precision через cfg.dtype: "bfloat16" | "float32".
    # bf16: -50% памяти на веса/Adam, ~+10% скорость; критичные накопления (loss,
    # WKV-рекуррентность) всё равно в fp32 -> это mixed-precision, не "чистый" bf16.
    model.set_dtype(cfg.dtype)
    print(f"  Модель в {cfg.dtype}")

    # ── Оптимизатор ──────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        learning_rate = cfg.lr,
        betas         = (cfg.beta1, cfg.beta2),
        eps           = cfg.adam_eps,
        weight_decay  = cfg.weight_decay,
    )

    # ── Train step ───────────────────────────────────────────────────────────
    if cfg.grad_accum == 1:
        train_step = _make_step_simple(model, optimizer, cfg)
    else:
        train_step = _make_step_accum(model, optimizer, cfg)

    # ── W&B ──────────────────────────────────────────────────────────────────
    wb = _init_wandb(cfg) if cfg.wandb else None

    # ── Цикл обучения ────────────────────────────────────────────────────────
    eff_batch = cfg.batch_size * cfg.grad_accum
    print(f"\nОбучение | steps={max_steps:,} | batch={cfg.batch_size}"
          + (f" × accum={cfg.grad_accum} (эфф.={eff_batch})" if cfg.grad_accum > 1 else "")
          + " | mx.compile ✓")
    print("─" * 60)

    t0       = time.time()
    best_val = float("inf")
    losses   = []

    for step in range(start_step, max_steps):
        optimizer.learning_rate = _lr_schedule(step, cfg)

        if cfg.grad_accum == 1:
            x, y = train_ds.batch(cfg.batch_size, step)
            loss, norm = train_step(x, y)
            mx.eval(loss, norm)
        else:
            xs = [train_ds.batch(cfg.batch_size, step * cfg.grad_accum + i)[0]
                  for i in range(cfg.grad_accum)]
            ys = [train_ds.batch(cfg.batch_size, step * cfg.grad_accum + i)[1]
                  for i in range(cfg.grad_accum)]
            loss, norm = train_step(xs, ys)
            mx.eval(loss, norm)

        losses.append(loss.item())

        # ── Лог ──────────────────────────────────────────────────────────────
        if (step + 1) % cfg.log_every == 0:
            dt    = time.time() - t0
            avg   = sum(losses[-cfg.log_every:]) / cfg.log_every
            tok_s = eff_batch * cfg.ctx_len * cfg.log_every / dt
            t0    = time.time()
            lr    = optimizer.learning_rate
            print(f"step {step+1:7,} | loss {avg:.4f} | lr {lr:.2e} "
                  f"| norm {norm.item():.2f} | {tok_s:.0f} tok/s")
            if wb:
                wb.log({"train/loss": avg, "train/lr": lr,
                        "train/grad_norm": norm.item(), "train/tok_s": tok_s},
                       step=step + 1)

        # ── Валидация ────────────────────────────────────────────────────────
        if (step + 1) % cfg.eval_every == 0:
            val_loss = _eval(model, val_ds, cfg)
            mark     = " ← best" if val_loss < best_val else ""
            print(f"  val loss: {val_loss:.4f}{mark}")
            if wb:
                wb.log({"val/loss": val_loss}, step=step + 1)

            if val_loss < best_val:
                best_val = val_loss
                _save(model, _ckpt_path(cfg, "best"))

        # ── Сохранение ───────────────────────────────────────────────────────
        if (step + 1) % cfg.save_every == 0:
            if not cfg.save_best_only:
                _save(model, _ckpt_path(cfg, "latest"))
                open(_step_path(cfg), "w").write(str(step + 1))
            print(f"  checkpoint: шаг {step+1:,}")

    # ── Финал ────────────────────────────────────────────────────────────────
    _save(model, _ckpt_path(cfg, "latest"))
    open(_step_path(cfg), "w").write(str(max_steps))
    print(f"\nГотово. Лучший val loss: {best_val:.4f}")
    print(f"Чекпоинты: {cfg.checkpoint_dir}/")
    if wb:
        wb.finish()
