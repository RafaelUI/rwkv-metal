"""
rwkv_metal.pretrain.cli
=======================
Командная строка для предобучения RWKV-7.

Установленный пакет создаёт команду:
    rwkv-metal-pretrain --preset 25m --train_data data/train.bin --val_data data/val.bin

Или напрямую:
    python -m rwkv_metal.pretrain.cli --preset 25m ...

Примеры:
    # Пресет с переопределением путей к данным
    rwkv-metal-pretrain --preset 25m \
        --train_data data/train.bin \
        --val_data   data/val.bin \
        --vocab_size 21248

    # Сырой текст (токенизация на лету)
    rwkv-metal-pretrain --preset 25m \
        --train_data data/wiki.txt \
        --tokenizer  tokenizer/tokenizer.json

    # Полный контроль через все флаги
    rwkv-metal-pretrain \
        --n_layer 18 --n_embd 256 --vocab_size 21248 \
        --train_data data/train.bin --val_data data/val.bin \
        --max_tokens 3_000_000_000 \
        --lr 1.5e-3 --batch_size 18 --ctx_len 512 \
        --dtype bfloat16 --checkpoint_dir checkpoints/

    # Заранее токенизировать текст в .bin
    rwkv-metal-pretrain --tokenize \
        --input    data/wiki.txt \
        --tokenizer tokenizer/tokenizer.json \
        --output   data/train.bin
"""

import argparse
import sys

from .config import PretrainConfig, preset as make_preset, PRESETS
from .trainer import pretrain
from .dataset import tokenize_to_bin


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rwkv-metal-pretrain",
        description="RWKV-7 pretraining on Apple Silicon (MLX)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Режим токенизации
    p.add_argument("--tokenize", action="store_true",
                   help="Токенизировать .txt → .bin и выйти")
    p.add_argument("--input",  default=None, help="Входной .txt (для --tokenize)")
    p.add_argument("--output", default=None, help="Выходной .bin (для --tokenize)")

    # Пресет
    p.add_argument("--preset", default=None,
                   choices=list(PRESETS),
                   help="Пресет модели (можно переопределить любой параметр)")

    # Архитектура
    arch = p.add_argument_group("Архитектура")
    arch.add_argument("--n_layer",    type=int,   default=None)
    arch.add_argument("--n_embd",     type=int,   default=None)
    arch.add_argument("--vocab_size", type=int,   default=None)
    arch.add_argument("--head_size",  type=int,   default=None)

    # Данные
    data = p.add_argument_group("Данные")
    data.add_argument("--train_data", default=None)
    data.add_argument("--val_data",   default=None)
    data.add_argument("--tokenizer",  default=None,
                      help="tokenizer.json (нужен если данные .txt)")
    data.add_argument("--ctx_len",    type=int, default=None)
    data.add_argument("--batch_size", type=int, default=None)
    data.add_argument("--grad_accum", type=int, default=None)

    # Сколько обучать
    train = p.add_argument_group("Обучение")
    train.add_argument("--max_steps",  type=int,   default=None)
    train.add_argument("--max_tokens", type=int,   default=None)

    # Оптимизатор
    opt = p.add_argument_group("Оптимизатор")
    opt.add_argument("--lr",           type=float, default=None)
    opt.add_argument("--lr_min",       type=float, default=None)
    opt.add_argument("--lr_schedule",  default=None,
                     choices=["cosine", "linear", "constant"])
    opt.add_argument("--warmup_steps", type=int,   default=None)
    opt.add_argument("--weight_decay", type=float, default=None)
    opt.add_argument("--grad_clip",    type=float, default=None)

    # Железо
    hw = p.add_argument_group("Железо")
    hw.add_argument("--dtype", default=None, choices=["bfloat16", "float32"])
    hw.add_argument("--grad_checkpoint", action="store_true", default=None,
                    help="Gradient checkpointing (меньше RAM, медленнее)")

    # Чекпоинты
    ckpt = p.add_argument_group("Чекпоинты")
    ckpt.add_argument("--checkpoint_dir", default=None)
    ckpt.add_argument("--no_resume",      action="store_true",
                      help="Начать с нуля даже если есть чекпоинт")
    ckpt.add_argument("--save_every",     type=int,  default=None)
    ckpt.add_argument("--save_best_only", action="store_true", default=None)

    # Логи
    log = p.add_argument_group("Логи")
    log.add_argument("--eval_every",   type=int, default=None)
    log.add_argument("--eval_batches", type=int, default=None)
    log.add_argument("--log_every",    type=int, default=None)
    log.add_argument("--wandb",        action="store_true", default=None)
    log.add_argument("--wandb_project", default=None)
    log.add_argument("--wandb_run",     default=None)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    # ── Режим токенизации ─────────────────────────────────────────────────────
    if args.tokenize:
        if not args.input or not args.tokenizer or not args.output:
            print("Для --tokenize нужны --input, --tokenizer, --output")
            sys.exit(1)
        result = tokenize_to_bin(args.input, args.tokenizer, args.output)
        print(f"\nГотово: {result['train_tokens']:,} train + {result['val_tokens']:,} val токенов")
        return

    # ── Собираем переопределения из CLI ──────────────────────────────────────
    overrides = {}
    fields = [
        "n_layer", "n_embd", "vocab_size", "head_size",
        "train_data", "val_data", "tokenizer", "ctx_len", "batch_size", "grad_accum",
        "max_steps", "max_tokens",
        "lr", "lr_min", "lr_schedule", "warmup_steps", "weight_decay", "grad_clip",
        "dtype", "checkpoint_dir", "save_every", "eval_every", "eval_batches", "log_every",
        "wandb_project", "wandb_run",
    ]
    for f in fields:
        v = getattr(args, f, None)
        if v is not None:
            overrides[f] = v

    bool_flags = {
        "grad_checkpoint": args.grad_checkpoint,
        "save_best_only":  args.save_best_only,
        "wandb":           args.wandb,
    }
    for k, v in bool_flags.items():
        if v:
            overrides[k] = True

    if args.no_resume:
        overrides["resume"] = False

    # ── Создаём конфиг ────────────────────────────────────────────────────────
    if args.preset:
        cfg = make_preset(args.preset, **overrides)
    else:
        # Без пресета — нужны обязательные поля
        required = ["train_data", "val_data"]
        for r in required:
            if r not in overrides:
                print(f"Ошибка: укажи --{r} или используй --preset")
                sys.exit(1)
        cfg = PretrainConfig(**overrides)

    pretrain(cfg)


if __name__ == "__main__":
    main()
