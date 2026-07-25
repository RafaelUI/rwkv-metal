# EmbeddingRWKV в rwkv-metal — состояние работ

Порт идей [howard-hou/EmbeddingRWKV](https://github.com/howard-hou/EmbeddingRWKV)
на MLX/Apple Silicon поверх `rwkv-metal`. Всё перечисленное ниже реализовано,
запущено и проверено на реальных данных.

---

## 1. Инференс: получение векторов

`rwkv_metal/embedding/embed.py`

- `Embedder(model, tokenizer, terminator=0, pooling="last")`, `embed_texts(...)`,
  `cosine_similarity_matrix(...)`.
- Работает поверх готового `model.body(idx)` — векторной проекции в словарь не
  требуется, голова языковой модели не задействована.
- Терминатор — токен `0` (зарезервированный/несопоставленный id в World-вокабе);
  пулинг берётся с его позиции, затем L2-нормировка.
- Каждый текст считается отдельным проходом без паддинга — короткие и длинные
  последовательности не загрязняют state друг друга.

**Проверено** (`tools/test_embedding_smoke.py`, `rwkv7-g1d-0.1b.pth`, 21 чанк из
`test.txt`): без какого-либо дообучения тексты чисто кластеризуются и по языку,
и по домену. Русские агро-тексты — sim 0.98–0.99 между собой; английские
wiki-статьи 0.97–0.99 и отдельным кластером английская физика; сербские 0.89–0.94
отдельно от всех. Загрузка модели 2.9 с, эмбеддинг 21 чанка 3.4 с.

---

## 2. Обучение

### Голова
`rwkv_metal/embedding/heads.py` — `EmbeddingHead`: zero-init residual
(`fc2.weight = 0`), то есть на нулевом шаге тождественна и не ломает геометрию
предобученной модели.

> **Важно:** голова — *сиблинг* базовой модели (`EmbeddingModel.base` / `.head`),
> а не её подмодуль. `add_lora()` / `quantize_base_model()` вызывают
> `model.freeze()` на дереве базы, и голова внутри него оказалась бы тихо
> заморожена.

### Данные
`rwkv_metal/embedding/dataset.py`

- `load_triplets_jsonl(path, task=..., limit=...)` — потоковое чтение с
  reservoir sampling (Algorithm R). Файл `retrieval_literature/train.jsonl` —
  2.6 ГБ / 554 096 строк; в памяти живёт только `limit` строк, и это честная
  случайная подвыборка, а не «первые N».
- `encode_batch` — **право**-паддинг (не лево, в отличие от CUDA-конвенции
  EmbeddingRWKV): RWKV-7 каузальна, паддинг после позиции пулинга физически не
  может на неё повлиять, маски не нужны.
- `PairBatcher` (пары), `TripletBatcher` (retrieval/sts),
  `ClassificationBatcher` (zero-shot).
- `parse_classification_candidates` — пул эмоций парсится из инструкции в самом
  anchor'е. Проверено на 20 000 строк: **у каждой строки свой уникальный набор
  из 7 меток**, общего словаря нет, поэтому паддинг + маска.

Состав датасета: retrieval 186 949 / classification 185 667 / sts 181 480.
Длина anchor: retrieval ~210 симв. (запрос), sts/classification ~1700 симв.
(медиана), до ~9000 максимум.

### Функции потерь
`rwkv_metal/embedding/loss.py`

- `info_nce_loss` — симметричный, только in-batch негативы.
- `triplet_pool_loss(anchor, positive, negative, symmetric)` — **пул негативов**:
  кандидаты = `concat(все positive батча, все hard-negative батча)`, то есть
  каждый anchor контрастится и со своим hard-negative, и с чужими — бесплатно.
  `symmetric=False` для retrieval (запрос и документ не симметричны),
  `symmetric=True` для sts (документ ↔ документ).
- `zero_shot_classification_loss` — без обучаемой головы классификации: пул
  меток эмбеддится как обычный текст и сравнивается по косинусу; маскированные
  логиты, т.к. K различается по строкам.

### Тренер
`rwkv_metal/embedding/train.py`

- `EmbeddingModel`, `EmbedTrainConfig`, `finetune_embedding(model, batches, cfg,
  compute_loss, gradcache_spec=None)`.
- Построен на **`nn.value_and_grad`** (уважает `freeze()`), а не на
  `mx.value_and_grad` как основной pretrain-тренер. Благодаря этому **один и тот
  же цикл** без изменений работает для:
  - full fine-tune (ничего не заморожено) — рецепт по умолчанию, совпадает с
    `--freeze_rwkv 0` у самих авторов EmbeddingRWKV;
  - заморозки нижних слоёв;
  - LoRA / QLoRA поверх замороженной (и при желании 4-битной) базы — путь для
    будущих 1.4B/4B, где full-FT на Apple Silicon уже не влезет.
- `compute_loss` — параметр, поэтому куррикулум = три последовательных вызова.
- LR-schedule: warmup + cosine/linear decay (`lr_schedule`, `lr_min`).
- Grad accumulation, клиппинг, gradient checkpointing, лог пиковой памяти.

### Привязка задач
`rwkv_metal/embedding/tasks.py` — `pair_loss`, `retrieval_loss`, `sts_loss`,
`classification_loss`, плюс спеки `RETRIEVAL_GC` / `STS_GC` для GradCache.

---

## 3. GradCache

`rwkv_metal/embedding/gradcache.py` — точный (не приближённый) двухпроходный
GradCache: no-grad forward с кэшированием эмбеддингов → loss и dL/dE на полном
батче → перечанковый backward с seed'ом.

Реализационная деталь: `mx.vjp` в MLX принимает только плоские списки массивов,
не pytree, поэтому фаза 3 идёт через тождество
`d/dθ Σ(embed(chunk) · stop_grad(dL/dE)) == VJP`, что позволяет пройти через
`nn.value_and_grad` и сохранить freeze-совместимость.

**Точность** (fp32, batch 8): при `chunk = batch` (1 чанк) расхождение по loss
ровно `0.0`, по градиентам `1.5e-8` — в точности уровень шума eager-vs-eager.
При 4 чанках `1.3e-4` — чистый порядок суммирования. Для контраста: обычный
grad accumulation на тех же чанках отклоняется на `3.9` (≈400%), потому что
ломает контрастивную математику.

**Память** (bf16, пассажи 800 симв., chunk=4):

| batch | eager | GradCache |
|---|---|---|
| 8 | 3.43 ГБ | 3.15 ГБ |
| 16 | 4.50 ГБ | 3.17 ГБ |
| 32 | 7.00 ГБ | 3.20 ГБ |
| 48 | 9.68 ГБ | 3.20 ГБ |

Цена ~30% времени. Loss совпадает до 4 знаков на каждом размере.

> Замеры — через `mx.get_peak_memory()`, т.е. пик аллокатора MLX, **не** RSS
> процесса. Характер (линейный рост против плоского) реальный, абсолютные
> значения к RSS не приводятся.

Классификация спеки не имеет намеренно: пул кандидатов там свой у каждой строки,
больший батч не даёт больше негативов, экономить нечего.

---

## 4. Оценка

`rwkv_metal/embedding/eval.py`

- `evaluate_retrieval` — MRR, Recall@k, nDCG@10; true positive ранжируется
  против пула из всех positive + всех hard-negative eval-среза.
- `evaluate_sts_pairwise` — pairwise accuracy (`cos(a,pos) > cos(a,neg)`).
  **Это не Spearman-корреляция:** датасет даёт только бинарные пары, без
  градуированных оценок сходства.
- `evaluate_classification` — zero-shot top-1 по собственному пулу меток строки.

---

## 5. Результаты прогонов

Куррикулум на реальном LitRetrieval, 0.1B, 500 строк на задачу, 30 шагов,
held-out по 100 строк (непересекающийся: reservoir на train+eval, затем разрез).

| стадия | метрика | до | после |
|---|---|---|---|
| retrieval | MRR | 0.047 | **0.71** |
| | Recall@1 | 0.00 | **0.61** |
| | Recall@10 | 0.11 | **0.87** |
| | nDCG@10 | 0.045 | **0.75** |
| sts | pairwise acc | 0.97 | 0.98 |
| | зазор sim(pos)−sim(neg) | 0.22 | **0.51** |
| classification | top-1 acc (7 классов) | 0.15 | **0.32** |

Структурная проверка LoRA: тот же тренер, база обёрнута `add_lora(rank=8)` →
1.77M / 192.8M обучаемых (0.92%), loss 1.52 → 1.30 за 10 шагов. Ни строчки в
тренере не менялось.

---

## 6. Инструменты

- `tools/test_embedding_smoke.py` — инференс, кластеризация по языкам/доменам
- `tools/test_embedding_train_smoke.py` — full-FT и LoRA
- `tools/test_embedding_curriculum_smoke.py` — три стадии + eval до/после
- `tools/test_gradcache.py` — эквивалентность и память
- `tools/test_gradcache_accum_baseline.py` — GradCache против grad accumulation

---

## 7. Что дальше

1. **Настоящий прогон эмбеддера** — пока только 30-шаговые смоуки на 500
   строках. Датасета 554K хватит надолго; куррикулум уже собран.
2. ~~**Реранкер**~~ — **сделано**, см. `docs/reranker.md`. Состояние прокинуто
   через `RWKV7X070.body(idx, state=..., mask=..., end_idx=..., return_state=...)`,
   тип `RWKVState` (wkv + token-shift обоих миксов). Голова читает состояние
   базы одним обучаемым токеном; на LitRetrieval held-out MRR 0.222 (случайно)
   → 0.491 (сырой эмбеддер) → **0.977** (реранкер), а против майненного
   hard-негатива 0.493 → **0.960**. Обучение идёт по кэшу состояний, эпоха —
   секунды.

   Попутно выяснилось: в from-scratch архитектуре `RWKV7` межблочный перенос
   token-shift (`x_prev = x[:, -1:]` после каждого блока) отдаёт блоку i+1
   ПОСЛЕДНИЙ токен чанка как «предыдущий» — то есть будущее относительно
   позиции 0. Замер: смена только последнего токена входа меняет скрытые
   состояния на позициях 0..T-2 на 0.30, в x070 — ровно 0.0. Это утечка
   будущего при teacher-forcing и причина, по которой state-API сделан
   только для x070.
3. Мульти-задачные головы (`[CLS]`/`[STS]`/`[RETR]`) — сейчас одна.
4. Фильтрация false negatives в батче, обучаемая температура, Matryoshka.
5. Реранкер поверх НЕзамороженной базы (LoRA): ломает кэш состояний, нужен
   онлайновый тренер.

---

## Приложение: изменения в WKV-ядре

В `rwkv_metal/kernel/wkv7_checkpoint.py` добавлены константы
`ACC_FWD=8, ACC_BWD=4, TILE_BWD=16`:

- forward: стейджинг `a/w/k/b/r` в threadgroup-память (устраняет 64-кратно
  дублированные глобальные чтения) + разбиение цепочек зависимых FMA на
  независимые аккумуляторы → **2.58x** изолированно;
- backward: тайлинг `accum` (18.2 → 6.2 КБ threadgroup) + ILP → **1.31x**;
- golden-тесты проходят (`tests/test_wkv7_backward.py`, погрешность 3.7e-8
  против допуска 1e-5).

**Сквозного эффекта нет** (шаг 1225 → 1201 мс, ~2%): WKV — лишь ~32% шага, и
шаг упирается в общий разделяемый ресурс. Установка `ACC_FWD=1, ACC_BWD=1,
TILE_BWD=64` возвращает в точности исходное поведение.

Разбор узкого места и тупики зафиксированы в `experiments/`:
`roofline_audit.py`, `bench_fwd_variants.py`, `bench_bwd_variants.py`,
`bench_bwd_tile.py`, `wy_3slab.py`.
