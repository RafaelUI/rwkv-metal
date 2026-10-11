"""
rwkv_metal.lora.state_tune
==========================
State-tuning: база заморожена целиком, обучается только начальное состояние
(`model/initial_state.py`) -- по умолчанию матрица WKV [H, S, S] на слой, как в
RWKV-PEFT. 0.1B -- 0.59 М чисел, 0.4B -- 1.57 М, 1.5B -- 3.15 М: на порядок-два
меньше, чем верхние слои, и файл не требует слияния с базой.

Цена шага -- как у адаптеров на ВСЕХ слоях: состояние входит в каждый слой,
начиная с нулевого, поэтому градиент идёт через всю глубину (замороженного
низа на быстрых ядрах тут нет). Память -- одного блока: `state0.wkv` -- параметр
своего блока, и шаг идёт обратным проходом по блокам (`lora/sequential.py`).

    from rwkv_metal.lora import StateTuneConfig, state_tune
    state_tune(model, batches, StateTuneConfig(max_steps=500, state_path="task.state.safetensors"))

    # потом, на той же базе:
    state, meta = load_state("task.state.safetensors", model)
    attach_state(model, state)          # model.body / embed / states стартуют с него

Работает над плотной базой и над .rwkvq (`load_rwkvq_model(path, rank=0)`):
градиент через сжатые проекции идёт их собственным VJP.
"""
import math
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ..model.initial_state import (attached_state, prepare_state_tuning, save_state,
                                   trained_parts)
from ..model.state import RWKVState
from . import sequential as _seq


@dataclass
class StateTuneConfig:
    # Шаг -- ЗАМЕРЕННЫЙ, а не перенесённый: с шагом 1 -> 0.01, который принято
    # называть для state-tuning, состояние у нас разрушает модель (g1d 0.1B,
    # русская проза, 100 шагов 4 x 256, лосс на отложенных окнах 3.228: шаг
    # 1e-2 -- 3.374, 3e-3 -- 3.307, 1e-3 -- 3.197, 3e-4 -- 3.176). Adam двигает
    # КАЖДОЕ из сотен тысяч чисел состояния на ~lr за шаг, а полезный сдвиг
    # мал: при шаге 3e-4 за 100 шагов max |s| = 0.018.
    lr:           float = 3e-4
    lr_final:     float = 3e-5          # косинус от lr к lr_final
    warmup_steps: int = 10
    max_steps:    int = 500
    grad_clip:    float = 0.0           # 0 -- без отсечения
    beta1:        float = 0.9
    beta2:        float = 0.99
    adam_eps:     float = 1e-8

    train:        Tuple[str, ...] = ("wkv",)    # ещё: "tmix_shift", "cmix_shift"
    sequential_backward: bool = True
    loss_chunk_tokens:   int = 256
    cache_limit_gb:      float = 0.25

    log_every:    int = 20
    state_path:   Optional[str] = "state.safetensors"
    note:         str = ""              # в метаданные файла: имя базы, пресет


def lr_at(step: int, cfg: StateTuneConfig) -> float:
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    span = max(1, cfg.max_steps - cfg.warmup_steps)
    t = min(1.0, (step - cfg.warmup_steps) / span)
    return cfg.lr_final + 0.5 * (cfg.lr - cfg.lr_final) * (1.0 + math.cos(math.pi * t))


def state_tune(model, batches: Iterable[Tuple[mx.array, mx.array]], cfg: StateTuneConfig,
               init: Optional[RWKVState] = None, on_step: Optional[Callable] = None) -> dict:
    """Обучить начальное состояние языковой модели на парах (x, y) [B, T].

    База замораживается ЦЕЛИКОМ (в том числе адаптеры, если они есть). `init`
    -- стартовое состояние (например, после промпта); None -- нули.
    Возвращает {"final_loss", "steps", "state" (RWKVState, батч 1), "state_path"}.
    Состояние остаётся подключённым к модели.
    """
    if cfg.cache_limit_gb and cfg.cache_limit_gb > 0:
        mx.set_cache_limit(int(cfg.cache_limit_gb * 1e9))
    n = prepare_state_tuning(model, init=init, train=cfg.train)
    if hasattr(model, "_grad_ckpt"):
        model._grad_ckpt = False

    if cfg.sequential_backward and _seq.supported(model):
        def grad_fn(m, x, y):
            return _seq.value_and_grad(m, x, y, cfg.loss_chunk_tokens)
    else:
        grad_fn = nn.value_and_grad(model, lambda m, x, y: m.loss(x, y).astype(mx.float32))

    opt = optim.Adam(learning_rate=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.adam_eps)
    it = iter(batches)

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(batches)
            return next(it)

    print(f"State-tuning | steps={cfg.max_steps} | обучается {','.join(cfg.train)}: {n/1e6:.2f}M чисел")
    print("-" * 60)
    last, t0 = float("nan"), time.time()
    for step in range(cfg.max_steps):
        opt.learning_rate = lr_at(step, cfg)
        x, y = next_batch()
        loss, grads = grad_fn(model, x, y)
        if cfg.grad_clip and cfg.grad_clip > 0:
            grads, _ = optim.clip_grad_norm(grads, max_norm=cfg.grad_clip)
        opt.update(model, grads)
        mx.eval(loss, model.trainable_parameters(), opt.state)
        last = loss.item()
        if on_step is not None:
            on_step(step, last, mx.get_peak_memory() / 1e9)
        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.max_steps - 1):
            print(f"  step {step:5d} | loss {last:.4f} | lr {opt.learning_rate.item():.4f} "
                  f"| peak {mx.get_peak_memory()/1e9:.2f} GB | {time.time()-t0:.1f}s")
    print("-" * 60)

    state = attached_state(model)
    if cfg.state_path:
        save_state(cfg.state_path, state, model, kind="tuned" if init is None else "tuned+prompt",
                   trained=trained_parts(model), note=cfg.note)
        print(f"Done. final loss {last:.4f} | state -> {cfg.state_path}")
    return {"final_loss": last, "steps": cfg.max_steps, "state": state,
            "state_path": cfg.state_path}
