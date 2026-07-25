"""
Тонкая обёртка над HuggingFace `tokenizers` для BPE-словарей, которыми
пользуются from-scratch чекпоинты rwkv-metal (в отличие от официальных
World-весов с их 65536-словарём).

Интерфейс намеренно совпадает с `WorldTokenizer`: `.encode(str) -> List[int]`
и `.decode(List[int]) -> str`, чтобы весь код в `rwkv_metal.embedding`
(батчеры, eval, Embedder) работал с любым из двух токенизаторов без правок.
"""
from typing import List, Optional


class BPETokenizer:
    """path: путь к tokenizer.json в формате HuggingFace `tokenizers`.

    pad_id/eos_id читаются из словаря спецтокенов, если они там есть.
    `terminator_id` — то, что embedding-код добавляет в конец текста перед
    пулингом; по умолчанию `<eos>`, а если его нет — `<pad>`, а если нет и
    его — 0.
    """

    def __init__(self, path: str, add_special_tokens: bool = False):
        try:
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(
                "BPETokenizer требует пакет `tokenizers` (pip install tokenizers)"
            ) from e
        self._tok = Tokenizer.from_file(path)
        self.path = path
        self.add_special_tokens = add_special_tokens
        self.vocab_size = self._tok.get_vocab_size()

        self.pad_id = self._id_of("<pad>")
        self.eos_id = self._id_of("<eos>")
        self.bos_id = self._id_of("<bos>")
        self.unk_id = self._id_of("<unk>")

    def _id_of(self, token: str) -> Optional[int]:
        i = self._tok.token_to_id(token)
        return i

    @property
    def terminator_id(self) -> int:
        for cand in (self.eos_id, self.pad_id):
            if cand is not None:
                return cand
        return 0

    def encode(self, s: str) -> List[int]:
        return self._tok.encode(s, add_special_tokens=self.add_special_tokens).ids

    def decode(self, tokens: List[int]) -> str:
        return self._tok.decode(list(tokens), skip_special_tokens=False)
