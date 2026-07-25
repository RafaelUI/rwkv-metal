"""
rwkv_metal.reranker.data
========================
Подготовка cross-encoder примеров для реранкера.

Ключевое решение — порядок в шаблоне. Документ идёт ДО запроса:

    Instruct: {инструкция}
    Document: {документ}
    Query: {запрос}

Причина не стилистическая. RWKV — RNN, состояние после префикса зависит
только от префикса. Поставив документ первым, мы делаем состояние
«Instruct + Document» кэшируемым: посчитали один раз на документ, а каждый
следующий запрос стоит только своих токенов. Поставив запрос первым, мы бы
эту возможность выбросили. Кэш индексируется парой (инструкция, документ),
а инструкций в корпусе единицы, так что практически это кэш по документу.

Отсюда же берётся почти бесплатное расширение набора негативов: префиксы
документов внутри шага уже посчитаны, поэтому спарить запрос со ВСЕМИ
документами шага стоит только хвостов запроса. См. `build_candidates`.

Формат исходных данных — LitRetrieval: {anchor, positive, negative, task},
где anchor уже содержит «Instruct: ...\\nQuery: ...».
"""
import json
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_INSTRUCT = "Given a search query, retrieve relevant passages that answer the query"

_ANCHOR_RE = re.compile(r"^Instruct:\s*(.*?)\s*\nQuery:\s*(.*)$", re.S)


def parse_anchor(anchor: str) -> Tuple[str, str]:
    """«Instruct: X\\nQuery: Y» → (X, Y). Без префикса — весь текст запрос."""
    m = _ANCHOR_RE.match(anchor)
    if m:
        return m.group(1), m.group(2)
    return DEFAULT_INSTRUCT, anchor


@dataclass
class PairTemplate:
    """Шаблон пары. Разбит на префикс (кэшируемый) и суффикс.

    doc_first=True  → «Instruct/Document/Query»: префикс = инструкция +
                      документ, кэшируется по (инструкция, документ).
    doc_first=False → «Instruct/Query/Document»: кэшировать нечего, но
                      документ читается уже зная запрос. Оставлено для
                      честного сравнения, умолчание — True.
    """
    doc_first: bool = True

    def prefix(self, instruct: str, doc: str, query: str) -> str:
        """Кэшируемая часть. При doc_first=False от документа не зависит
        ничего, кроме инструкции, — кэшировать нечего, и это видно прямо в
        сигнатуре результата."""
        if self.doc_first:
            return f"Instruct: {instruct}\nDocument: {doc}\n"
        return f"Instruct: {instruct}\n"

    def suffix(self, instruct: str, doc: str, query: str) -> str:
        if self.doc_first:
            return f"Query: {query}"
        return f"Query: {query}\nDocument: {doc}"

    def full(self, instruct: str, doc: str, query: str) -> str:
        return self.prefix(instruct, doc, query) + self.suffix(instruct, doc, query)


@dataclass
class RerankSample:
    """Один запрос со своим набором кандидатов.

    doc_ids:  индексы документов в общем пуле (порядок = порядок кандидатов)
    label:    индекс правильного документа внутри doc_ids
    hard_neg: индекс МАЙНЕННОГО hard-негатива внутри doc_ids (None, если его
              в строке не было). Нужен для честной метрики: случайный документ
              из литературного корпуса отличить легко, а вот выбранный
              майнером — нет, и общий MRR по восьми кандидатам это различие
              размывает.
    """
    instruct: str
    query: str
    doc_ids: List[int]
    label: int
    hard_neg: Optional[int] = None


def load_rows(path: str, task: str = "retrieval", limit: Optional[int] = None,
              seed: int = 0, lang: Optional[str] = None) -> List[Dict]:
    """Потоковое чтение jsonl с reservoir sampling (Algorithm R).

    Файл LitRetrieval — 2.6 ГБ; в памяти живёт только `limit` строк, и это
    честная равномерная подвыборка, а не «первые N».
    """
    rng = random.Random(seed)
    cyr = re.compile(r"[а-яёА-ЯЁ]")
    lat = re.compile(r"[a-zA-Z]")
    res: List[Dict] = []
    seen = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if task is not None and obj.get("task") != task:
                continue
            if lang is not None:
                s = obj.get("positive", "")[:600]
                row_lang = "ru" if len(cyr.findall(s)) > len(lat.findall(s)) else "en"
                if row_lang != lang:
                    continue
            if limit is None:
                res.append(obj)
                continue
            seen += 1
            if len(res) < limit:
                res.append(obj)
            else:
                j = rng.randint(0, seen - 1)
                if j < limit:
                    res[j] = obj
    return res


def build_candidates(rows: Sequence[Dict], n_candidates: int = 8,
                     seed: int = 0) -> Tuple[List[str], List[RerankSample]]:
    """Строки LitRetrieval → (пул документов, примеры).

    Кандидаты каждого запроса:
      1. его позитив,
      2. его hard-негатив (уже отобранный майнером в самом датасете),
      3. остальные — случайные документы из пула других строк.

    Пул документов общий и дедуплицированный: документ, встретившийся у
    нескольких запросов, кодируется один раз. Именно поэтому «добавить
    негативов» здесь почти ничего не стоит — платим только за хвост запроса.
    """
    rng = random.Random(seed)
    pool: List[str] = []
    doc2id: Dict[str, int] = {}

    def add(doc: str) -> int:
        i = doc2id.get(doc)
        if i is None:
            i = len(pool)
            doc2id[doc] = i
            pool.append(doc)
        return i

    prepared = []
    for row in rows:
        instruct, query = parse_anchor(row["anchor"])
        pos_id = add(row["positive"])
        neg = row.get("negative")
        neg_id = add(neg) if neg else None
        prepared.append((instruct, query, pos_id, neg_id))

    samples: List[RerankSample] = []
    n_pool = len(pool)
    for instruct, query, pos_id, neg_id in prepared:
        cand = [pos_id]
        if neg_id is not None and neg_id != pos_id:
            cand.append(neg_id)
        guard = 0
        while len(cand) < n_candidates and guard < 50 * n_candidates:
            guard += 1
            j = rng.randrange(n_pool)
            if j not in cand:
                cand.append(j)
        if len(cand) < n_candidates:
            raise ValueError(
                f"не удалось набрать {n_candidates} различных кандидатов: "
                f"в пуле всего {n_pool} документов. Уменьши --candidates или "
                f"возьми больше строк."
            )
        # позиция позитива перемешивается, чтобы модель не выучила «нулевой
        # кандидат всегда правильный» (при listwise-лоссе это реальный риск)
        order = list(range(len(cand)))
        rng.shuffle(order)
        shuffled = [cand[i] for i in order]
        samples.append(RerankSample(
            instruct=instruct, query=query, doc_ids=shuffled,
            label=shuffled.index(pos_id),
            hard_neg=(shuffled.index(neg_id)
                      if neg_id is not None and neg_id != pos_id else None),
        ))
    return pool, samples


def truncate_tokens(tokenizer, text: str, max_tokens: int) -> List[int]:
    """Токенизировать и обрезать. Обрезаем ХВОСТ документа: начало пассажа
    обычно информативнее, а состояние всё равно копит слева направо."""
    ids = tokenizer.encode(text)
    return ids[:max_tokens] if len(ids) > max_tokens else ids


def split_train_eval(samples: Sequence[RerankSample], n_eval: int,
                     seed: int = 0) -> Tuple[List[RerankSample], List[RerankSample]]:
    """Непересекающийся held-out по ЗАПРОСАМ (документы пул делят — это
    нормально: оценивается ранжирование, а не запоминание документов)."""
    idx = list(range(len(samples)))
    random.Random(seed).shuffle(idx)
    ev = [samples[i] for i in idx[:n_eval]]
    tr = [samples[i] for i in idx[n_eval:]]
    return tr, ev
