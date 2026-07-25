"""
rwkv_metal.reranker.rerank
==========================
Инференс реранкера: два режима.

Без индекса (`score` / `rank`)
    Каждая пара считается целиком: O(L_doc + L_query) на кандидата. Просто,
    ничего не хранит, годится когда документы каждый раз новые.

С индексом (`build_index` / `score_indexed`)
    Состояние префикса «Instruct + Document» считается один раз и хранится.
    Дальше запрос стоит O(L_query) на кандидата — на 0.1B это 3.5 мс против
    73 мс, то есть примерно 20×. Цена — память: полное состояние это
    L·H·S·S·4 байта, для 0.1B ~2.4 МБ на документ (bf16 — 1.2 МБ). Индекс на
    тысячу документов это 2.4 ГБ, поэтому он для «горячего» подмножества
    (например, top-100 от эмбеддера), а не для всего корпуса.

Важно: индекс валиден только для той же базы, того же шаблона и той же
инструкции, с которыми строился — состояние это функция ровно от префикса.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

from ..model.state import RWKVState
from .data import DEFAULT_INSTRUCT, PairTemplate
from .encode import _batch_ids, _encode_prefix_ids, _encode_suffix_ids


@dataclass
class DocIndex:
    """Кэш состояний префиксов. state.batch == len(docs)."""
    state: RWKVState
    docs: List[str]
    instruct: str
    doc_first: bool

    def nbytes(self) -> int:
        return self.state.nbytes()

    def __len__(self) -> int:
        return len(self.docs)


class RerankerInference:
    def __init__(self, reranker, tokenizer,
                 template: PairTemplate = None,
                 instruct: str = DEFAULT_INSTRUCT,
                 max_doc_tokens: int = 384,
                 max_query_tokens: int = 96,
                 terminator: Optional[int] = 0):
        self.model = reranker
        self.tok = tokenizer
        self.template = template or PairTemplate()
        self.instruct = instruct
        self.max_doc_tokens = max_doc_tokens
        self.max_query_tokens = max_query_tokens
        self.terminator = terminator

    # ── Прямой путь ──────────────────────────────────────────────────────
    def score(self, query: str, docs: Sequence[str], instruct: str = None,
              batch_size: int = 8) -> np.ndarray:
        """Скоры [len(docs)]. Больше — релевантнее. Величина сырая (логит),
        сравнима внутри одного запроса; для абсолютной шкалы используй
        sigmoid и обучение с ненулевым BCE-членом."""
        instruct = instruct or self.instruct
        out = []
        for start in range(0, len(docs), batch_size):
            part = docs[start:start + batch_size]
            seqs = [
                _encode_prefix_ids(self.tok, self.template, instruct, d,
                                   self.max_doc_tokens)
                + _encode_suffix_ids(self.tok, self.template, d, query,
                                     self.max_query_tokens, self.max_doc_tokens,
                                     self.terminator)
                for d in part
            ]
            idx, mask, end_idx = _batch_ids(seqs)
            st = self.model.encode(idx, mask=mask, end_idx=end_idx)
            s = self.model.score_states(self.model.select(st))
            mx.eval(s)
            out.append(np.array(s.astype(mx.float32)))
        return np.concatenate(out)

    def rank(self, query: str, docs: Sequence[str], top_k: int = None,
             instruct: str = None, batch_size: int = 8
             ) -> List[Tuple[int, float]]:
        """[(индекс документа, скор)] по убыванию скора."""
        s = self.score(query, docs, instruct=instruct, batch_size=batch_size)
        order = np.argsort(-s)
        if top_k is not None:
            order = order[:top_k]
        return [(int(i), float(s[i])) for i in order]

    # ── Путь с индексом ──────────────────────────────────────────────────
    def build_index(self, docs: Sequence[str], instruct: str = None,
                    batch_size: int = 8, dtype=None,
                    verbose: bool = False) -> DocIndex:
        """Посчитать и сохранить состояния префиксов документов.

        dtype: mx.bfloat16 ополовинит память индекса. Рекуррентная часть
        (wkv) всё равно останется fp32 — от её точности зависит совпадение
        продолжения со сплошным проходом.
        """
        instruct = instruct or self.instruct
        if not self.template.doc_first:
            raise ValueError(
                "Индекс имеет смысл только при doc_first=True: при обратном "
                "порядке префикс зависит от запроса и кэшировать нечего."
            )
        parts = []
        for start in range(0, len(docs), batch_size):
            chunk = docs[start:start + batch_size]
            seqs = [_encode_prefix_ids(self.tok, self.template, instruct, d,
                                       self.max_doc_tokens) for d in chunk]
            idx, mask, end_idx = _batch_ids(seqs)
            st = self.model.encode(idx, mask=mask, end_idx=end_idx)
            if dtype is not None:
                st = st.astype(dtype)
            st.eval()
            parts.append(st)
            if verbose:
                print(f"  индекс {start + len(chunk)}/{len(docs)}", flush=True)
        state = RWKVState.concat(parts) if len(parts) > 1 else parts[0]
        return DocIndex(state=state, docs=list(docs), instruct=instruct,
                        doc_first=self.template.doc_first)

    def score_indexed(self, query: str, index: DocIndex,
                      doc_ids: Sequence[int] = None,
                      batch_size: int = 32) -> np.ndarray:
        """Скоры запроса против документов индекса (всех или подмножества)."""
        ids = list(range(len(index))) if doc_ids is None else list(doc_ids)
        q_ids = _encode_suffix_ids(self.tok, self.template, "", query,
                                   self.max_query_tokens, self.max_doc_tokens,
                                   self.terminator)
        out = []
        for start in range(0, len(ids), batch_size):
            part = ids[start:start + batch_size]
            sub = index.state[mx.array(np.array(part, dtype=np.int32))]
            idx, mask, end_idx = _batch_ids([q_ids] * len(part))
            st = self.model.encode(idx, mask=mask, end_idx=end_idx, state=sub)
            s = self.model.score_states(self.model.select(st))
            mx.eval(s)
            out.append(np.array(s.astype(mx.float32)))
        return np.concatenate(out)

    def rank_indexed(self, query: str, index: DocIndex, top_k: int = None,
                     doc_ids: Sequence[int] = None, batch_size: int = 32
                     ) -> List[Tuple[int, float]]:
        ids = list(range(len(index))) if doc_ids is None else list(doc_ids)
        s = self.score_indexed(query, index, doc_ids=ids, batch_size=batch_size)
        order = np.argsort(-s)
        if top_k is not None:
            order = order[:top_k]
        return [(int(ids[i]), float(s[i])) for i in order]
