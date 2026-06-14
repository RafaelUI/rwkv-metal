"""
rwkv_metal.pretrain.dataset
===========================
Загрузка данных для предобучения.

Поддерживает два формата:
  - .bin  (uint16, токенизированный заранее) — быстро, рекомендуется
  - .txt  (сырой текст) — токенизируется на лету через tokenizer.json

Пример подготовки .bin заранее:
    from rwkv_metal.pretrain import tokenize_to_bin
    tokenize_to_bin("data/wiki.txt", "tokenizer.json", "data/train.bin")
"""

import os
import numpy as np
import mlx.core as mx
from typing import Optional, Tuple


class BinDataset:
    """
    Датасет из токенизированного .bin файла (uint16).

    Файл читается через np.memmap — не грузится в RAM целиком,
    батчи читаются с диска по мере нужды.
    """

    def __init__(self, path: str, ctx_len: int):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл данных не найден: {path}")
        self.data    = np.memmap(path, dtype=np.uint16, mode="r")
        self.ctx_len = ctx_len
        self.n       = len(self.data)
        size_gb      = self.n * 2 / 1e9
        print(f"  {os.path.basename(path)}: {self.n/1e6:.1f}M токенов ({size_gb:.2f} GB)")

    def batch(self, batch_size: int, step: int) -> Tuple[mx.array, mx.array]:
        """Возвращает (x, y) батч для шага step."""
        stride = self.ctx_len + 1
        starts = [
            (step * batch_size + i) * stride % (self.n - stride)
            for i in range(batch_size)
        ]
        x = mx.array(np.stack([
            self.data[s : s + self.ctx_len].astype(np.int32)
            for s in starts
        ]))
        y = mx.array(np.stack([
            self.data[s + 1 : s + self.ctx_len + 1].astype(np.int32)
            for s in starts
        ]))
        return x, y

    def validate(self, vocab_size: int) -> dict:
        """
        Проверяет данные на OOV токены и другие проблемы.
        Читает выборку из разных мест файла.
        """
        sample_size = min(10_000_000, self.n)
        # Берём равномерную выборку из разных мест
        indices = np.linspace(0, self.n - sample_size, 10, dtype=int)
        issues = []
        max_token = 0

        for idx in indices:
            chunk = np.array(self.data[idx : idx + sample_size // 10])
            chunk_max = int(chunk.max())
            max_token = max(max_token, chunk_max)
            oov = int((chunk >= vocab_size).sum())
            if oov > 0:
                issues.append(f"OOV токены в позиции ~{idx:,}: {oov} шт, max={chunk_max}")

        return {
            "ok":        len(issues) == 0,
            "max_token": max_token,
            "issues":    issues,
        }

    def __len__(self) -> int:
        return self.n


class TextDataset:
    """
    Датасет из сырого .txt файла с токенизацией на лету.

    Менее эффективен чем BinDataset — для небольших корпусов
    или быстрого старта без предварительной токенизации.
    """

    def __init__(self, path: str, ctx_len: int, tokenizer_path: str):
        path           = os.path.expanduser(path)
        tokenizer_path = os.path.expanduser(tokenizer_path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл данных не найден: {path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Токенизатор не найден: {tokenizer_path}")

        try:
            from tokenizers import Tokenizer
            self.tok = Tokenizer.from_file(tokenizer_path)
        except ImportError:
            raise ImportError(
                "Для .txt данных нужна библиотека tokenizers: "
                "pip install tokenizers"
            )

        print(f"  Токенизация {os.path.basename(path)} на лету...")
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()

        ids = self.tok.encode(text).ids
        self.data    = np.array(ids, dtype=np.uint16)
        self.ctx_len = ctx_len
        self.n       = len(self.data)
        print(f"  {os.path.basename(path)}: {self.n/1e6:.1f}M токенов")

    def batch(self, batch_size: int, step: int) -> Tuple[mx.array, mx.array]:
        stride = self.ctx_len + 1
        starts = [
            (step * batch_size + i) * stride % (self.n - stride)
            for i in range(batch_size)
        ]
        x = mx.array(np.stack([
            self.data[s : s + self.ctx_len].astype(np.int32)
            for s in starts
        ]))
        y = mx.array(np.stack([
            self.data[s + 1 : s + self.ctx_len + 1].astype(np.int32)
            for s in starts
        ]))
        return x, y

    def __len__(self) -> int:
        return self.n


def load_dataset(path: str, ctx_len: int, tokenizer: Optional[str] = None):
    """
    Автоматически выбирает BinDataset или TextDataset по расширению файла.

    Args:
        path:      путь к .bin или .txt файлу
        ctx_len:   длина контекста
        tokenizer: путь к tokenizer.json (нужен только для .txt)
    """
    path = os.path.expanduser(path)
    ext  = os.path.splitext(path)[1].lower()

    if ext == ".bin":
        return BinDataset(path, ctx_len)
    elif ext in (".txt", ".text"):
        if tokenizer is None:
            raise ValueError(
                "Для .txt файлов нужен tokenizer. "
                "Передай tokenizer='path/to/tokenizer.json' "
                "или заранее токенизируй данные через tokenize_to_bin()"
            )
        return TextDataset(path, ctx_len, tokenizer)
    else:
        raise ValueError(
            f"Неподдерживаемый формат файла: {ext}. "
            "Используй .bin (рекомендуется) или .txt"
        )


def tokenize_to_bin(
    input_path:     str,
    tokenizer_path: str,
    output_path:    str,
    val_ratio:      float = 0.005,
    doc_delimiter:  Optional[int] = 2,
    batch_size:     int = 2000,
) -> dict:
    """
    Токенизирует .txt файл и сохраняет как .bin (uint16).

    Args:
        input_path:     путь к .txt файлу (документы разделены пустой строкой)
        tokenizer_path: путь к tokenizer.json
        output_path:    путь к выходному .bin (val сохраняется рядом как _val.bin)
        val_ratio:      доля документов в val (каждый 1/val_ratio документ)
        doc_delimiter:  токен-разделитель между документами (None = без разделителя)
        batch_size:     документов на один encode_batch

    Returns:
        dict с train_tokens, val_tokens, vocab_size
    """
    try:
        from tokenizers import Tokenizer
    except ImportError:
        raise ImportError("pip install tokenizers")

    input_path     = os.path.expanduser(input_path)
    tokenizer_path = os.path.expanduser(tokenizer_path)
    output_path    = os.path.expanduser(output_path)
    val_path       = output_path.replace(".bin", "_val.bin")

    tok = Tokenizer.from_file(tokenizer_path)
    tok.no_truncation()
    tok.no_padding()
    vocab_size = tok.get_vocab_size()

    def doc_stream(path):
        buf = []
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip() == "":
                    if buf:
                        yield "".join(buf).strip()
                        buf = []
                else:
                    buf.append(line)
        if buf:
            yield "".join(buf).strip()

    val_every   = max(1, round(1.0 / val_ratio))
    train_ids   = []
    val_ids     = []
    train_total = 0
    val_total   = 0
    doc_idx     = 0

    print(f"Токенизация {os.path.basename(input_path)} → {os.path.basename(output_path)}")
    print(f"  vocab_size={vocab_size}, val каждый {val_every}-й документ")

    gen = doc_stream(input_path)
    while True:
        # Набираем батч
        docs = []
        for _ in range(batch_size):
            try:
                d = next(gen)
                if d:
                    docs.append(d)
            except StopIteration:
                break
        if not docs:
            break

        encs = tok.encode_batch(docs, is_pretokenized=False)
        for enc in encs:
            if not enc.ids:
                continue
            ids = enc.ids
            if doc_delimiter is not None:
                ids = ids + [doc_delimiter]
            arr = np.array(ids, dtype=np.uint16)

            if doc_idx % val_every == 0:
                val_ids.append(arr)
                val_total += len(arr)
            else:
                train_ids.append(arr)
                train_total += len(arr)
            doc_idx += 1

        if doc_idx % 10000 == 0:
            print(f"  ...{doc_idx:,} документов, train {train_total/1e6:.1f}M токенов")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.concatenate(train_ids).tofile(output_path)
    np.concatenate(val_ids).tofile(val_path)

    print(f"\n  train.bin: {train_total:,} токенов ({train_total*2/1e9:.2f} GB)")
    print(f"  val.bin:   {val_total:,} токенов")

    return {
        "train_tokens": train_total,
        "val_tokens":   val_total,
        "vocab_size":   vocab_size,
        "train_path":   output_path,
        "val_path":     val_path,
    }
