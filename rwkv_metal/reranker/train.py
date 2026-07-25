"""
rwkv_metal.reranker.train
=========================
Обучение головы реранкера на кэше состояний.

База заморожена, поэтому шаг обучения не трогает её вовсе: батч — это набор
готовых состояний из `StateCache`, а вперёд и назад идёт только голова (один
или несколько RWKV-блоков на одном токене). Отсюда несколько следствий,
которые стоит держать в голове при подборе гиперпараметров:

  * шаг дешёвый, батчи можно брать большими (сотни пар) — упирается в память
    состояний, а не в вычисления;
  * эпоха проходится за секунды, так что эпох имеет смысл делать много;
  * весь риск переобучения лежит на голове: при 7-8 М параметров и десятках
    тысяч пар это реально, поэтому есть held-out и ранняя остановка по нему.
"""
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from .encode import StateCache
from .loss import mixed_loss


@dataclass
class RerankTrainConfig:
    lr: float = 3e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.99
    adam_eps: float = 1e-8
    grad_clip: float = 1.0

    batch_size: int = 32          # запросов в батче (пар = batch_size × n_cand)
    epochs: int = 4
    warmup_frac: float = 0.05
    lr_schedule: str = "cosine"   # cosine | linear | constant
    lr_min: float = 0.0

    loss_alpha: float = 1.0       # 1 = чистый listwise, 0 = чистый BCE
    temperature: float = 1.0

    compile: bool = True          # mx.compile на шаг обучения
    eval_every: int = 0           # шагов между оценками held-out (0 = только в конце)
    log_every: int = 20
    checkpoint_path: str = "reranker_head.safetensors"
    keep_best: bool = True        # сохранять веса лучшего held-out, а не последние
    seed: int = 0


def _lr_at(step: int, total: int, cfg: RerankTrainConfig) -> float:
    warm = max(1, int(total * cfg.warmup_frac))
    if step < warm:
        return cfg.lr * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    p = min(max(p, 0.0), 1.0)
    if cfg.lr_schedule == "cosine":
        decay = 0.5 * (1.0 + math.cos(math.pi * p))
    elif cfg.lr_schedule == "linear":
        decay = 1.0 - p
    else:
        decay = 1.0
    return cfg.lr_min + (cfg.lr - cfg.lr_min) * decay


def batch_states(cache: StateCache, rows: np.ndarray):
    """rows: [b] индексы примеров → (states [b·C, ...] mx, labels [b] mx, C).

    Гейтеринг идёт в numpy, в MLX попадает только батч. Кэш при этом может
    быть memmap и вообще не находиться в памяти.
    """
    pairs = cache.pair_index[rows]                 # [b, C]
    b, C = pairs.shape
    states = cache.gather(pairs.reshape(-1))       # [b*C, n_src, H, S, S]
    return states, mx.array(cache.labels[rows]), C


def batch_scores(head, cache: StateCache, rows) -> mx.array:
    """Скоры [b, C] для примеров `rows`."""
    rows = np.asarray(rows if not isinstance(rows, mx.array) else np.array(rows))
    states, _, C = batch_states(cache, rows)
    return head(states).reshape(-1, C)


def evaluate(head, cache: StateCache, batch_size: int = 64) -> dict:
    """Метрики ранжирования на кэше: MRR, Recall@k, nDCG@10, лосс."""
    n = cache.n_samples
    C = cache.n_cand
    ranks = []
    losses = []
    for start in range(0, n, batch_size):
        rows = np.arange(start, min(start + batch_size, n))
        states, labels, C = batch_states(cache, rows)
        scores = head(states).reshape(-1, C)
        losses.append(float(mixed_loss(scores, labels, 1.0, 1.0)) * len(rows))
        s = np.array(scores.astype(mx.float32))
        lab = cache.labels[rows]
        gold = s[np.arange(len(lab)), lab]
        # Ранг со средним по связкам: 1 + (строго больших) + (равных-1)/2.
        # Без этого необученная голова (все скоры равны) получила бы ранг 1 и
        # MRR = 1.0 — метрика показывала бы идеал там, где модель не знает
        # ничего.
        greater = (s > gold[:, None]).sum(axis=1)
        ties = (s == gold[:, None]).sum(axis=1) - 1
        ranks.append(1 + greater + ties / 2.0)
    ranks = np.concatenate(ranks)
    mrr = float((1.0 / ranks).mean())
    ranks_int = np.ceil(ranks)
    return {
        "mrr": mrr,
        "recall@1": float((ranks_int <= 1).mean()),
        "recall@3": float((ranks_int <= 3).mean()),
        "recall@5": float((ranks_int <= min(5, C)).mean()),
        "ndcg@10": float((1.0 / np.log2(ranks + 1) * (ranks <= 10)).mean()),
        "loss": float(sum(losses) / n),
        "n": int(n),
        "n_cand": int(C),
    }


def train_reranker(reranker, train_cache: StateCache,
                   eval_cache: Optional[StateCache] = None,
                   cfg: RerankTrainConfig = None,
                   on_step: Optional[Callable] = None,
                   save_extra: Optional[dict] = None) -> dict:
    """save_extra: дополнительные метаданные в чекпоинт головы — сюда стоит
    класть контракт подачи текста (`RerankerInference.serving_metadata()`)."""
    cfg = cfg or RerankTrainConfig()
    head = reranker.head
    rng = np.random.default_rng(cfg.seed)

    n = train_cache.n_samples
    steps_per_epoch = max(1, n // cfg.batch_size)
    total = steps_per_epoch * cfg.epochs

    n_train = sum(v.size for _, v in tree_flatten(head.parameters()))
    print(f"Реранкер: обучаемых {n_train/1e6:.2f}M | пар в кэше {train_cache.n_pairs} "
          f"| примеров {n} × {train_cache.n_cand} кандидатов | "
          f"{cfg.epochs} эпох × {steps_per_epoch} шагов")
    print("-" * 64)

    opt = optim.AdamW(learning_rate=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                      eps=cfg.adam_eps, weight_decay=cfg.weight_decay)

    C = train_cache.n_cand

    def loss_fn(h, states, labels):
        return mixed_loss(h(states).reshape(-1, C), labels,
                          cfg.loss_alpha, cfg.temperature)

    grad_fn = nn.value_and_grad(head, loss_fn)

    def eager_step(states, labels):
        loss, grads = grad_fn(head, states, labels)
        grads, gnorm = optim.clip_grad_norm(grads, max_norm=cfg.grad_clip)
        opt.update(head, grads)
        return loss, gnorm

    # mx.compile: шаг головы — это сотни мелких операций на одном токене, то
    # есть время уходит в диспатч, а не в арифметику. Компиляция сливает их в
    # один граф. Формы фиксированы (последний неполный батч отбрасывается),
    # поэтому перекомпиляций нет. `inputs`/`outputs` обязаны включать
    # состояние головы и оптимизатора — иначе обновления весов останутся
    # снаружи графа и шаг просто не будет ничего менять.
    if cfg.compile:
        _state = [head.state, opt.state, mx.random.state]
        step_fn = mx.compile(eager_step, inputs=_state, outputs=_state)
    else:
        step_fn = eager_step

    history = []
    best = {"mrr": -1.0}
    best_params = None
    step = 0
    t0 = time.time()

    for epoch in range(cfg.epochs):
        order = rng.permutation(n)
        for si in range(steps_per_epoch):
            rows = order[si * cfg.batch_size:(si + 1) * cfg.batch_size]
            states, labels, _ = batch_states(train_cache, rows)
            opt.learning_rate = _lr_at(step, total, cfg)
            loss, gnorm = step_fn(states, labels)
            mx.eval(loss, head.parameters(), opt.state)
            lv = float(loss)
            history.append({"step": step, "epoch": epoch, "loss": lv,
                            "lr": opt.learning_rate.item() if hasattr(opt.learning_rate, "item")
                            else float(opt.learning_rate)})
            if on_step is not None:
                on_step(step, lv)
            if cfg.log_every and step % cfg.log_every == 0:
                print(f"  эпоха {epoch} шаг {step:5d} | loss {lv:.4f} | "
                      f"grad {float(gnorm):.3f} | lr {history[-1]['lr']:.2e} | "
                      f"пик {mx.get_peak_memory()/1e9:.2f} ГБ", flush=True)
            step += 1

            if (eval_cache is not None and cfg.eval_every
                    and step % cfg.eval_every == 0):
                m = evaluate(head, eval_cache)
                print(f"    held-out: MRR {m['mrr']:.3f} R@1 {m['recall@1']:.3f} "
                      f"loss {m['loss']:.4f}")
                if cfg.keep_best and m["mrr"] > best["mrr"]:
                    best = m
                    best_params = {k: mx.array(v) for k, v in
                                   tree_flatten(head.parameters())}

        if eval_cache is not None:
            m = evaluate(head, eval_cache)
            print(f"  эпоха {epoch} завершена | held-out MRR {m['mrr']:.3f} "
                  f"R@1 {m['recall@1']:.3f} nDCG@10 {m['ndcg@10']:.3f} "
                  f"loss {m['loss']:.4f}", flush=True)
            if cfg.keep_best and m["mrr"] > best["mrr"]:
                best = m
                best_params = {k: mx.array(v) for k, v in
                               tree_flatten(head.parameters())}

    if cfg.keep_best and best_params is not None:
        from mlx.utils import tree_unflatten
        head.update(tree_unflatten(list(best_params.items())))
        mx.eval(head.parameters())
        print(f"  восстановлены веса лучшей эпохи (held-out MRR {best['mrr']:.3f})")

    reranker.save_head(cfg.checkpoint_path, extra=save_extra)
    print("-" * 64)
    print(f"Готово за {time.time()-t0:.0f}s → {cfg.checkpoint_path}")

    return {"history": history, "best": best, "steps": step,
            "seconds": time.time() - t0, "checkpoint_path": cfg.checkpoint_path}
