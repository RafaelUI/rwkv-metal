# HANDOFF — оптимизация WKV-7 ядра (rwkv-metal)

Состояние на конец сессии. Цель: ускорить on-device обучение и убрать
числовую слабость backward. Читать это в начале новой сессии.

## Железо
- MacBook Air **M4 base**, GPU 8 ядер, **16 ГБ** unified (бюджет процесса ~12 ГБ).
- Пик ПСП **120 ГБ/с**; пик ~**3.4 TFLOPS fp32** (8-ядерный).

## Что изучаем
- rwkv-metal (Python/MLX), предобучение RWKV-7.
- Конфиг: **12L, n_embd 256, head_size 64, vocab 32000, ctx 512, B≈20–24**, dtype bf16.
- Реальный чекпоинт: `/Users/s/Develop/rwkvNeuro/checkpoints/rwkv7_12l256d_best.npz`.

## Замеры (B20, M4, host убран если не сказано иное)
- Базовая пропускная: **~6800–7500 tok/s**, **~1400–1840 ms/step** (растёт с батчем).
- Instruments: **30 ГБ/с (~25% пика)**, ALU **57%**, дыры в таймлайне.
- tok/s **плоский** по B20/24/27 → не parallelism-bound; serial/occupancy-bound.
- Attribution (2 прогона): body(fwd WKV+matmul) ~345 мс; +head+CE ~405 мс (**CE+голова всего ~60 мс**); full step ~1425 мс →
  **backward+opt ~1020 мс = 72%**, forward 28%.
- WKV-**backward** ≈ 3–4 GFLOP за ~850 мс ⇒ **<1% пика**. Матмулы проекций в том же шаге — near-peak.
- Loop overlap (prefetch + async_eval): только **+4%** (6741→6988) ⇒ хост НЕ узкое место.
- Forward dispatch `(1,1,1)→(1,D,1)`: **+7%** (6798→7290), **бит-идентично**. ПРИМЕНЕНО.

## Числовая находка (важно)
- Checkpoint-backward реконструирует `h_{t-1}=(h_t − v⊗k − sa⊗b)/w` **делением на w** → нестабильно при малом w.
- grad-паритет (verify_grad, режим w∈0.55–1): `dk,dv,db` точны (~1e-7); `dr,dw,da` ~**1e-3**.
- w-свип (verify_grad_w): `dr/dw/da` взрываются при w↓: w0.995→5e-7; w0.747→1.4e-3; w0.496→**1.4e4**; w0.270→**3.6e16**. `dk` всегда точен.
- НО формула модели: `w = exp(-0.606531*sigmoid(...)) ∈ (0.5455, 1.0)` — жёсткий пол **0.5455** (=exp(-exp(-0.5))). Катастрофы НЕТ (обрыв ниже 0.5).
- Реальное распределение w (из ckpt, capture_w): **бимодальное** — p0=0.545, **p50=0.609**, mean=0.711, **~48% каналов прижаты к полу 0.545**, остальные ~1.0.
- Вывод: `~1e-3` на `dr/dw/da` бьёт по **~половине каналов** (быстрозабывающие) постоянно. Для Adam терпимо (сходится), но улучшаемо.

## Диагноз
- Стена = **backward WKV-кернел** (72%, <1% пика). Не forward, не softmax, не RAM, не батч.
- Причина: скалярная последовательная рекуррентность + barrier-bound backward (`accum[64][64]`=16 КБ threadgroup-памяти на тредгруппу ⇒ occupancy ~1–2 группы/ядро; ~10 барьеров/токен × 512).
- Forward уже поправлен по occupancy (+7%). Backward — наивный по структуре, хотя кооперативный.

## Потолок (Амдал)
Эффективная часть шага (forward+matmul+CE+opt ~400 мс) — пол. Даже идеальный WKV-backward (850→~50–100 мс) ⇒ шаг 1400→~600 мс ⇒ **×2–2.4** (≈6800→~15k tok/s). Не ×10, но и не ×7%.

## ПЛАН
- **S0 (DONE):** RWKV-7 = **DPLR delta rule**. `chunk_rwkv7(r,k,v,a,b,w,log_w)` → `chunk_dplr_delta_rule(q=r, k=k, v=v, a=a, b=b, gk=log_w)`.
  Устойчивость = **лог-пространство decay** (cumsum log_w, НЕ деление на cumprod w). WY/UT для дельта-части. Backward **рекомпьютит** h (без деления).
- **S1:** DPLR чанковый forward на чистом MLX (порт FLA chunk_dplr) → forward-паритет vs рекуррентный реф (~1e-5) → autograd → verify_grad_w (ждём dr/dw/da ~1e-6 при ЛЮБОМ w).
- **S2:** замер скорости MLX-версии (может быть и быстрее, и медленнее — мерить).
- **S3:** кастомный Metal `simdgroup_matrix` чанковый backward. Валидация vs S1 + grad-харнесс. Цель: WKV-bwd <1%→~10% пика ⇒ шаг ×2–2.4.
- **S4 (потом):** forward-WKV в матмул-форму (добивка на 345 мс body).

## Ключевой код (rwkv-metal)
- `rwkv_metal/kernel/wkv7.py`: `CHUNK=32`; entry `wkv7(...,training=True)`; `wkv7_train`→checkpoint; `_wkv7_chunk_metal`/`_get_fwd` (L~208) и `_get_infer` (L~302) ВСЁ ЕЩЁ `(1,1,1)` — TODO поправить (ускорит инференс/Synapse). `wkv7_train_py` = эталон einsum.
- `rwkv_metal/kernel/wkv7_checkpoint.py`: `make_wkv7_checkpoint` — боевой fwd+bwd обучения. **fwd dispatch L185 СЕЙЧАС `(1,D,1)`** (наша правка). bwd L200 `(D,1,1)` + `accum[64][64]` 16КБ + реконструкция делением на w (L~130) — ЦЕЛЬ S3.
- `rwkv_metal/model/rwkv7.py`: `w = exp(-0.606531*sigmoid(w_lora_B(tanh(w_lora_A(xw)))))` (L~94). `model.body`, `model.loss`. tmix зовёт `wkv7(...,training=True)` глобально.
- `rwkv_metal/pretrain/trainer.py`: `_make_step_simple` (compiled). Цикл: per-step `.item()` + синхронный `ds.batch()` → +4% доступно (prefetch+async_eval). Низкий приоритет.
- `rwkv_metal/pretrain/dataset.py`: `BinDataset` (np.memmap).

## Тест-харнесс (experiments/) — ГАРДЫ, не удалять
- `verify_fwd.py [save|compare]` — forward-паритет Metal vs Python; bit-identical check.
- `verify_grad.py` — grad-паритет Metal vjp vs autograd.
- `verify_grad_w.py` — **свип по w** (главный страж устойчивости).
- `capture_w.py` — реальное распределение w из ckpt.
- `bench_kernel.py` — чистая пропускная ядра (host убран), memory-safe.
- `bench_attrib.py` — раскладка fwd/bwd/CE.
- `bench_loop_overlap.py` — baseline vs prefetch+async_eval.
- Запуск длинных: `nohup .venv/bin/python -u experiments/X.py < /dev/null > /tmp/x.log 2>&1 &` (шелл-лимит 10с — детач+поллинг). memory-safe: лимит mx + кэп «шагов в полёте».

## Применённые изменения
- `wkv7_checkpoint.py` L185: fwd threadgroup `(1,1,1)→(1,D,1)` (+7%, bit-identical). Откат: `git checkout`.
- `experiments/*.py` — новые файлы (харнесс + бенчи).
- (Отдельно, в ~/Develop/SwiftRWKV — Swift-порт: поправлены устаревшие комменты lnX/LoRAFinetune/PartialFinetune + rename gen_residentMemoryMB. К rwkv-metal не относится.)

## Референсы
- Songlin Yang, DeltaNet-II (чанковый WY/UT): https://sustcsonglin.github.io/blog/2024/deltanet-2
- FLA: `fla/ops/rwkv7/chunk.py` (враппер) → `fla/ops/generalized_delta_rule/dplr/chunk.py` (`chunk_dplr_delta_rule`). Лог-decay: `chunk_local_cumsum(g, scale=RCP_LN2)`. Backward: `recompute_w_u_fwd`+`chunk_*_fwd_h`.
- RWKV-7 paper: arXiv 2503.14456 (§8 про ядро; чанк 16, два ядра bf16/fp32).
- M4 + Metal 4.1 (Rigel): tensor-ops лоуерятся в `simdgroup_matrix`, без скрытого матрично го блока и без ANE-роутинга; fp8 эмулируется. Значит потолок скорости = simdgroup_matrix, 4.1 сверху не даёт.

## Следующий конкретный шаг
S1: склонировать FLA локально, прочитать `chunk_dplr_delta_rule` (и торч-референс/наив из тестов), портировать DPLR-чанк в MLX по стадиям с проверкой паритета на каждой.

## S1 — DONE (результаты)
Порт DPLR-чанка в чистый MLX + валидация. Файлы: `experiments/dplr_mlx.py`
(`dplr_recurrence_mlx` stage0, `dplr_chunkwise_mlx` stage1), `experiments/verify_dplr.py`
(forward-паритет), `experiments/verify_grad_w.py` (grad-свип, 3 колонки).

Контракт паритета к `wkv7_train_py` (3 ловушки, учтены):
1. `scale=1.0` (RWKV-7 НЕ скейлит r на d_k**-0.5, в отличие от DeltaNet-наива).
2. `gk = log(w)` — потребляем ЛОГ-decay (в эталоне w — реальный decay, h*=w).
3. layout: эталон h[b,h,s,d] (s=value,d=key) == порт S[b,h,d,m] транспонирован; [B,T,H,D]↔head-first.

Forward-паритет (fp32, vs wkv7_train_py): stage0 1.6e-7; chunk=16 2.0e-7; chunk=32 3.9e-7.
  → chunk=32 даёт ~2× ошибки vs 16 (шире динам. диапазон лог-decay + хуже обусловлена UT-инверсия).

Grad-устойчивость (rel к ИСТИНЕ = autograd рекуррентности; свип по равномерному w, chunk=16):
        w   |  боевой vjp |  новый чанк
      0.995 |     3.4e-7  |    4.1e-7
      0.747 |     4.4e-4  |    3.8e-7
      0.496 |     7.1e+1  |    3.9e-7
      0.270 |     1.1e+10 |    2.9e-7   (все grad конечны)
  Боевой checkpoint-bwd (реконструкция делением на w) расходится уже к w~0.75 и
  взрывается под floor; новый лог-путь держит плато ~1e-6 на ЛЮБОМ w. С учётом
  capture_w (~48% каналов у floor 0.545) боевой постоянно несёт ~1e-3 на половине
  каналов. Числовая слабость закрыта.

ОРАКУЛ: правда = autograd рекуррентности (НЕ боевой vjp — он и есть сломанное).
  Боевой vjp в свипе — контестант, наглядно уступающий.

Хвост: grad-свип гонялся на chunk=16; догнать на chunk=32.
Дальше: S2 — замер скорость/память нового MLX-чанка vs боевое ядро (см. ниже).

## S2 — DONE (замер скорость/память)
Файл: `experiments/bench_dplr.py` (B=20,T=512,H=4,D=64,fp32; один WKV-слой).
ВАЖНО про единицы: бенч мерит ИЗОЛИРОВАННУЮ WKV-операцию, НЕ шаг обучения.
  tok/s тут = B*T/время одной операции — НЕ сравнивать с tok/s полной модели
  (там +12 слоёв, проекции, FFN, LM-голова vocab32k, CE, Adam). Память 0.18ГБ —
  рабочий набор ядра на случайных тензорах, НЕ модель (та забивает 12ГБ).
  Изоляция завышает пропускную ~2× vs in-situ (handoff attrib: ~71мс/слой bwd).

tok/s + пик памяти (изолированный слой):
                       ms     tok/s     peak
  battle  fwd        5.56    1.84M    0.14 GB
  battle  fwd+bwd   35.11    292k     0.18 GB
  chunk16 fwd       75.78    135k     0.71 GB
  chunk16 fwd+bwd  311.81     33k     2.46 GB
  chunk32 fwd      138.09     74k     1.52 GB
  chunk32 fwd+bwd  468.00     22k     4.28 GB

ВЕРДИКТ: новый MLX-чанк как написан — НЕ скоростной выигрыш. ~9× медленнее
  (fwd+bwd) и ~13–24× прожорливее боевого Metal-ядра. Причина в сигнатуре памяти:
  векторная сборка A-матриц материализует [B,H,N,C,C,D] и делает поэлементное
  произведение+сумму по D → bandwidth-bound редукция, simdgroup_matrix НЕ задействован.
  Стабильную in-exp форму разменяли на матмул-форму. MLX = оракул, НЕ ядро.

grad-стабильность на боевом шейпе (нормы dr/dw/da):
   w=0.90  battle 7363/8327/833   chunk 7363/8327/833  (идентично до бита)
   w=0.50  battle 83230/96357/9457  chunk 3749/2146/215
   w=0.27  battle 1.2e13/2.3e13/2.3e12  chunk 3380/1737/174  (все конечны)
  Подтверждает S1 на боевом шейпе: боевой bwd гонит мусор ниже w~0.75; новый
  убывает плавно (физично). Точность = большой плюс. Скорость плюса требует S3.

РЕШЕНИЕ: путь A (прямо в S3 Metal). MLX-оракул готов и достаточен для валидации.
  Путь B (матмул-факторизованный MLX) отклонён: переоткрывает overflow exp(-gc),
  вряд ли побьёт 5.56мс боевого fwd. Берёмся за S3.

## S3 — НАЧАТО: карта возможностей Metal-бэкенда (через mx.fast.metal_kernel)
Эмпирические probe (mlx 0.31.2, M4, Metal 4.1). mx.fast.metal_kernel компилит
ТОЛЬКО тело MSL; сигнатуру (сырые `device float*` буферы) генерит MLX сам.
Хостовые API Metal 4.1 (MTL4Compiler/CommandBuffer/Allocator, untracked+ручные
барьеры) — НЕдоступны (их держит MLX). Релевантна только девайс-сторона.

ПОЛ (работает, проверено): `simdgroup_float8x8` (#include <metal_simdgroup_matrix>).
  Тайлы строго 8×8, half/bfloat/float. Ручной тайлинг. Probe 8×8 matmul — точно.
  D=64/C=16/C=32 кратны 8 → раскладывается чисто. НАДЁЖНЫЙ бэкенд для S3.

ПОТОЛОК (компилит+запускается, НО layout неверный): MPP `tensor_ops::matmul2d`
  (#include <metal_tensor> + <MetalPerformancePrimitives/...>, namespace mpp).
  - Хедеры компилятся. Ограничение: M или N кратно 16 (C∈{16,32},D=64 — ок).
  - float×float→float (Metal4); bf16 нужен OS26.1+. Есть аппаратные
    reduce_rows/reduce_columns на cooperative_tensor (идеально под суммы по D в DPLR)
    и multiply_accumulate в дескрипторе.
  - ПРОБЛЕМА: операнды A/B должны быть host-bound / origin-shifted / shader-allocated.
    `tensor_inline` (обёртка сырого device*) в этот список НЕ входит → matmul2d
    читает layout неверно (probe A=I дал перетасованный B, err=255). Host-bound
    тензоры через MLX недоступны (нет MTL4-биндинга).
  - ВОЗМОЖНЫЙ обход (не проверен): shader-allocated тензоры — скопировать вход
    device→threadgroup кооперативно с нужным layout, обернуть как shader-allocated
    (2.22.2.7), matmul в cooperative_tensor, store. Даёт контроль layout, но +код.

РЕШЕНИЕ ПО БЭКЕНДУ S3 — открыто (ждёт steer):
  A) simdgroup_float8x8 — проверено, предсказуемо, ручной тайлинг 8×8.
  B) MPP matmul2d через shader-allocated обход — современно, гибкие 16-кратные
     размеры, аппаратные редукции; но layout-обход надо ещё решить (probe).
  Рекомендация: строить S3 на A (надёжно), MPP оставить как оптимизацию после
  рабочего baseline. Probe-скрипты: /tmp/probe_metal.py, /tmp/probe_mpp_matmul.py.

## S3 — MPP-путь РЕШЁН (коррекция предыдущей секции)
Баг был НЕ в несовместимости — а в layout. Ключ из доки 2.22.2.7:
  - `tensor_inline` == shader-allocated тензор (ВХОДИТ в список операндов matmul2d).
  - `strides[0]` ВСЕГДА = 1 (контиг-измерение = dim0). Я ставил {K,1} → нарушение.
  - Фикс: packed-конструктор `tensor<device float,dextents<int,2>,tensor_inline>(ptr,
    dextents<int,2>(cols,rows))` — extents в порядке (cols,rows) даёт strides[0]=1.
  - device element_type + device-указатель валиден (копия в threadgroup НЕ нужна).

Probe /tmp/probe_mpp_fix.py — БИТ-В-БИТ:
  tl=false tr=false -> A@B   = 0.00e0
  tl=false tr=true  -> A@B.T = 0.00e0
  tl=true  tr=false -> A.T@B = 0.00e0
  Флаги transpose_left/right в matmul2d_descriptor(M,N,K,tl,tr) работают даром.

ИТОГ S3-бэкенд: путь B (MPP matmul2d) ПРОВЕРЕН и предпочтителен. Гибкие 16-кратные
  размеры (C∈16/32, D=64 — ок), даровые транспозиции (в bwd их много), аппаратные
  reduce_rows/columns под суммы по D, multiply_accumulate под накопление S.
  fp32 float->float работает сейчас; bf16->float требует OS26.1+ (проверить отдельно,
  валидацию вести в fp32 против MLX-оракула, потом bf16).
  simdgroup_float8x8 остаётся рабочим запасным полом.

## S3.1 — DONE (фьюзед конструкция A-матриц на MPP)
Файлы: experiments/s3_dplr_kernel.py (compute_amats), experiments/verify_s3.py.
Один threadgroup = один simdgroup (32 потока) на чанк. На чипе:
  qhat=q*exp(gc), khat=k*exp(-gc), bhat=beta*exp(-gc), ahat=alpha*exp(gc-gk)
  → 4 матмула matmul2d (tr=true): A_qk=qhat@khat^T, A_qb=qhat@bhat^T,
    A_ab=ahat@bhat^T, A_ak=ahat@khat^T. 4 матмула+4 выхода в одном кернеле,
    C++17-лямбды в MSL работают.
Паритет vs MLX-оракул (стабильная форма), C=16,D=64,fp32, БИТ-В-БИТ:
  A_qk 7.2e-7, A_qb 1.5e-8, A_ab 5.6e-9, A_ak 1.5e-7. Свёртка decay численно чиста.
Layout matmul2d (закреплено): операнд [C,D] row-major → threadgroup tensor_inline
  extents (D,C); выход [C,C] → device tensor_inline extents (C,C); desc(C,C,D,false,true).
Маски (j<=i / j<i) пока в харнессе — в кернел на S3.2/3.3.
ЗАМЕТКА память: 4 hat-буфера * C*D*4B. C=16 -> 16KB (ок). C=32 -> 32KB = предел
  threadgroup M4 без запаса под scratch matmul. На C=32 секвенировать матмулы,
  держать <=3 буфера резидентно (khat,bhat + один left).
Дальше: S3.2 — UT-инверсия A_ab в кернеле.

## S3.2 — DONE (треугольный solve в кернеле, без явного A_inv)
Файлы: s3_dplr_kernel.py (trisolve), verify_s3.py (main_s32).
Решаем (I - A_ab) X = RHS построчной подстановкой:
  X[i,:] = RHS[i,:] + Σ_{n<i} A_ab[i,n]·X[n,:]   (C серийных шагов, барьер между строк,
  d-измерение параллельно по 32 потокам). A_ab и X в threadgroup.
Под обе правые части: u (RHS=A_ak@v), wmat (RHS=exp(gc-gk)·alpha).
Паритет vs A_inv@RHS (эталон A_inv = ряд Неймана (I-A_ab)^-1), C=16,D=64,fp32:
  u 1.2e-7, wmat 1.5e-8. БИТ-В-БИТ.
Решение метода: построчно (для C=16 цепочка всего 16 шагов). На C=32 при
  необходимости — блочный TRSM 16×16 (связь через matmul2d). A_ab подаётся
  уже маскированной (j<i); маска в кернел — на S3.3.
Дальше: S3.3 — межчанковая рекуррентность S + полная сборка forward + маски в кернеле.

## S3.3a — DONE (маска A-матриц в кернеле)
Файлы: s3_dplr_kernel.py (compute_amats_masked), verify_s3.py (main_s33a).
Store cooperative_tensor -> threadgroup даёт row-major (A[i,j] по i*C+j) — проверено
маской по i=e/C, j=e%C. Маска в threadgroup: Aqk/Aqb зануляют j>i, Aab/Aak зануляют j>=i.
Паритет с МАСКИРОВАННЫМ оракулом напрямую (без harness-маски): A_qk 7.2e-7, A_qb 1.5e-8,
  A_ab 5.6e-9, A_ak 1.5e-7. БИТ-В-БИТ.
Дальше: S3.3b — сборка выхода одного чанка при S=0 (apply-матмулы [C,C]@[C,D] tr=false:
  RHS_u=A_ak@v, solve u, o=A_qk@v + A_qb@u), паритет vs wkv7_train_py на одном чанке.
  Потом S3.3c — межчанковый цикл S + B*H. Затем S3.4 backward.

## S3.3b — DONE (фьюзед forward одного чанка при S=0)
Файлы: s3_dplr_kernel.py (`chunk_fwd_s0`, `_chunk_s0_kernel`), verify_s3.py (`main_s33b`).
Один threadgroup = один simdgroup = один чанк. Полный фьюз фаза A → apply:
  фаза A: hats=qhat/khat/bhat/ahat (свёртка decay) → 4× matmul2d(tr=true) → Am[4*C*C]
    (qk/qb/ab/ak) → маски в threadgroup (qk/qb: j>i=0; ab/ak: j>=i=0).
  фаза B (S=0): RHS_u=A_ak@v → in-kernel trisolve (I-A_ab)u=RHS_u построчно →
    o = A_qk@v + A_qb@u.  (v2=u т.к. wmat@S=0; o3=(q·e^gc)@S=0).
АРЕНА переиспользована: hats[4*C*D] мертвы после фазы A → v(0)/u(1)/o(2)/tmp(3).
  Am[4*C*C] резидентны весь kernel. Бюджет C=16: 16KB+4KB=20KB (<32KB M4). coop-tensor
  аккумулятор в регистрах симдгруппы (НЕ threadgroup) — подтверждено, scratch не нужен.
Новое vs S3.3a: apply-матмул второй формы [C,C]@[C,D] tr=false →
  desc(C,D,C,false,false); left=A[C,C] extents(C,C); right/out=[C,D] extents(D,C).
  (закреплено probe /tmp/probe_mpp_fix.py: A@B = left(K,M),right(N,K),out(N,M)).
Паритет (C=16,D=64,fp32, S=0), max_abs по 4 сидам:
  vs боевой wkv7_train_py : 1.4–1.9e-6  (|o|max~5–7)
  vs MLX-оракул (N=1)     : 1.3–2.6e-6
  БИТ-В-БИТ (fp32 ~7 ULP). Регресс S3.1/3.2/3.3a цел.
ЗАМЕТКА C=32: hats 32KB + Am 16KB = 48KB > 32KB. Секвенировать (держать <=3 hat-буфера,
  apply-выходы переиспользуют слоты) — на S3.3c когда подключим межчанк.
Дальше: S3.3c — межчанковый цикл S (v2=u+wmat@S, o3=(q·e^gc)@S, апдейт
  S=S·e^gc_last + (k·decay)^T@v + (beta·decay)^T@v2) + размотка B*H; паритет на T=2..4 чанка.
  Затем S3.4 — backward.

## S3.3c — DONE (межчанковый цикл S + размотка B*H)
Файлы: s3_dplr_kernel.py (`_step_kernel`/`chunk_step`/`dplr_forward_metal`),
  verify_s3.py (`main_s33c`). + probe /tmp/probe_mpp_tl.py, /tmp/probe_mixed_space.py.
Архитектура: один threadgroup=один simdgroup=один (b,h) на ОДНОМ чанке; грид по BH
  (`grid=(32,BH,1)`, `bh=thread_position_in_grid.y`). Драйвер гоняет N чанков
  последовательно (межчанк-зависимость S — серийная), внутри чанка BH параллельны.
  S-elementwise рекур (`S=S·e^gc_last + t1 + t2`) на ХОСТЕ (тривиально); все GEMM
  (вкл. S-апдейтные t1,t2) — в кернеле. Полный фьюз N-цикла — позже (оптимизация).
Кернел на чанк: фаза A→masked Am; solve u (RHS=A_ak@v); solve wmat (RHS=e^(gc-gk)·α);
  v2=u+wmat@S; o=A_qk@v+A_qb@v2+(q·e^gc)@S; t1=(k·decay)^T@v; t2=(β·decay)^T@v2,
  decay[i,d]=exp(gc_last[d]-gc[i,d]). qhat (фаза A) переиспользован под o3=(q·e^gc)@S.
Формы matmul (все probe-проверены, бит-в-бит):
  A_xy=xhat@yhat^T  desc(C,C,D,F,T)   операнды [C,D] ext(D,C), out[C,C] ext(C,C)
  apply A@x         desc(C,D,C,F,F)   left[C,C] ext(C,C), right/out[C,D] ext(D,C)
  x@S               desc(C,D,D,F,F)   left[C,Dk] ext(D,C), S[Dk,Dv] ext(D,D) [tg×DEVICE — ок]
  kd^T@v            desc(D,D,C,T,F)   left[C,D] ext(D,C), right[C,D] ext(D,C), out[D,D] ext(D,D)
  → tl=true контрактит по ПЕРВОЙ оси (C); смешанные threadgroup×device операнды РАБОТАЮТ (0.00e0).
Арена: arena[6*C*D] (s0=qhat,s1=v,s2=u→v2,s3=wmat,s4=scratch,s5=oacc) + Am[4*C*C].
  C=16: 24KB+4KB=28KB (<32KB M4). C=32 НЕ влезет (48KB+) → секвенс/блочно (TODO).
Паритет (C=16,D=64,fp32; B*H до 6; N=2/3/4; 4 сида):
  vs боевой wkv7_train_py : 1.9–3.8e-6   vs MLX-оракул : 0.95–2.6e-6   (|o|max до ~12)
  БИТ-В-БИТ. Полный forward DPLR на Metal закрыт.
Дальше: S3.4 — BACKWARD. Это и есть цель S3 (боевой bwd = 72% шага, <1% пика +
  числовая слабость). Оракул правды для grad = autograd рекуррентности (НЕ боевой vjp).
  План: vjp по чанковому forward; recompute h в лог-пространстве (БЕЗ деления на w);
  grad-свип verify_grad_w на боевом шейпе ждём dr/dw/da ~1e-6 при ЛЮБОМ w (как S1/S2).
  Затем бенч скорости vs боевое Metal-ядро (цель шага ×2–2.4).

## S3.3c — ЗАМЕРЫ (скорость / память / многошаговый grad)
Файлы: experiments/bench_s33c.py, experiments/verify_grad_steps.py.

### Многошаговая grad-стабильность (verify_grad_steps.py)
25 шагов Adam, B2T64H4D64, θ→w=exp(-0.606531·sigmoid θ) бимодально (~50% у пола 0.545).
Единая траектория (шаг по ИСТИНЕ=autograd рекуррентности), на каждом шаге rel-ошибка
grad боевого Metal vjp и нового MLX-чанка против истины (dr/dθ/da):
  боевой Metal vjp : 1.8e-3 … 1.8e-2  (гуляет, всплески)
  новый лог-чанк   : 3e-7 … 3e-6      (плато, стабильно)
ВАЖНО: при ФИЗИЧЕСКОМ w (пол 0.545) ОБА пути конечны и обучаются (loss идентичен до
  ~4 знаков) — боевой НЕ взрывается. Взрыв verify_grad_w (3.6e16) был при w=0.27
  НИЖЕ пола, чего модель не достигает. Т.е. многошаговая СТАБИЛЬНОСТЬ боевого в норме;
  разрыв — в ТОЧНОСТИ grad (~1e-3 vs ~1e-6 на ~половине каналов). Это и чинит S3.4.

### Скорость / память forward (bench_s33c.py, B20T512H4D64, один слой)
                                    ms      tok/s     peak
  battle fwd (scalar recur)       5.78    1.77M     0.14 GB
  MLX oracle chunk16 fwd         80.75    127k      0.70 GB
  S3.3c step-kernel fwd (driver) 33.58    305k      0.16 GB   per-chunk dispatch+eval
  battle fwd+bwd                 34.50    297k      0.18 GB
  isolated one chunk_step        0.766 ms (×32 чанков ≈ 24.5 ms чистого ядра/слой)
ВЫВОДЫ:
- vs MLX-оракул: step-кернел ×2.4 быстрее и ×4.4 легче — переход оракул→матмул-ядро
  оправдан (оракул был bandwidth-bound).
- Драйвер 33.6мс = ~24.5мс ядро + ~9мс хост (per-chunk dispatch+eval). Фьюз N-цикла
  в один кернел уберёт ~9мс (планировалось).
- НО vs battle forward (5.78мс) step-кернел ×4 МЕДЛЕННЕЕ. battle forward = быстрый
  стриминговый scalar-recur; его боль — BACKWARD (fwd+bwd 34.5мс ⇒ bwd ~28мс).
  Матмул-формаforward материализует ~10 матмулов/чанк → дороже на forward.
СТРАТЕГИЧЕСКИЙ ВЫВОД ДЛЯ S3.4: DPLR-матмул-форма выигрывает там, где battle катастрофичен
  (BACKWARD, <1% пика), и проигрывает на forward (battle уже near-peak/дёшев). Кандидат:
  НЕ заменять forward на матмул-форму (или только по необходимости для интермедиатов),
  а нацелить матмул-форму на BACKWARD. Бюджет: forward-ядро ~24мс уже съедает много;
  чтобы побить battle fwd+bwd=34.5мс, bwd должен влезть в ~10мс поверх — туго, поэтому
  важны (а) фьюз N-цикла, (б) рекомпьют forward внутри bwd дёшево, (в) не дублировать
  A-матрицы/solve. Перемерить in-situ в 12L-шаге (изоляция завышает ~2×).
Дальше: S3.4 — backward на матмул-форме (recompute h в лог-пространстве, БЕЗ деления на w),
  валидация vs истина (autograd рекуррентности) на свипе w, затем бенч fwd+bwd vs battle.

## S3.4 — НАЧАТО: backward (цель всего S3)
Стратегия (как S1): вывести VJP → доказать в чистом MLX → порт в Metal.

## S3.4a-i — DONE (аналитический backward одиночного чанка S=0, в MLX)
Файл: experiments/dplr_bwd_mlx.py (chunk_fwd_s0_mlx, chunk_bwd_s0_mlx, _inv_neumann).
Матмул-форма: всё = (транспонированные) матмулы + транспонированный треуг. solve
  (M^T x=du, back-subst) + elementwise + revcumsum. БЕЗ деления на w. (I-A_ab)^{-1}
  рядом Неймана (точно для строго-нижней нильпотентной A_ab).
VJP (do): dA_qk=le(do@v^T), dv=A_qk^T@do, dA_qb=le(do@u^T), du=A_qb^T@do;
  dc=(I-A_ab)^{-T}@du, dA_ab=lt(dc@u^T), dA_ak=lt(dc@v^T), dv+=A_ak^T@dc;
  dqh=dA_qk@kh+dA_qb@bh, dkh=dA_qk^T@qh+dA_ak^T@ah, dbh=dA_qb^T@qh+dA_ab^T@ah,
  dah=dA_ab@bh+dA_ak@kh; dq=dqh·e^gc,dk=dkh·e^-gc,dβ=dbh·e^-gc,dα=dah·e^(gc-gk);
  dgc=dqh·qh-dkh·kh-dbh·bh+dah·ah, dgk=-dah·ah; dgk+=cumsum(dgc,reverse); dw=dgk/w; dr=dq.
Валидация (C=16,D=64, свип w вкл. 0.27 НИЖЕ пола):
  vs autograd(MLX-fwd): dr/dk/dv = 0.0 бит-в-бит; dw/da/db 1e-8..9e-7
  vs ИСТИНА(recurrence autograd): ВСЁ 2e-7..9e-7 на ВСЕХ w (вкл. 0.27, где боевой=3.6e16)
  → вывод верен, лог-форма точна везде. Числовая слабость backward закрыта по математике.
Примечание (mlx 0.31.2): mx.linalg.inv не на GPU (нужен CPU-stream) → ряд Неймана;
  mx.flip нет → mx.cumsum(...,reverse=True) для revcumsum.
Дальше: S3.4a-ii — порт в Metal. Новый примитив = транспонированный trisolve (M^T x=du,
  back-subst i=C-1..0: x[i]=du[i]+Σ_{n>i}A_ab[n,i]·x[n]). Остальные матмулы — формы
  S3.3 (с транспозициями, даровыми в matmul2d). Потом S3.4b — межчанк bwd (carry dS назад).

## S3.4a-ii — primitive DONE + архитектура порта решена (kernel ещё не собран)
Транспонированный trisolve в Metal — ГОТОВ и проверен (probe /tmp/probe_trisolve_T.py):
  back-subst i=C-1..0: x[i]=du[i]+Σ_{n>i}A_ab[n,i]·X[n], d-ось ∥ по 32 потокам.
  vs (I-A_ab)^{-T}@du: max_abs 2.6e-5. Все остальные bwd-матмулы — формы S3.3
  (F,T / T,F / F,F), уже probe-проверены.

БЮДЖЕТ (почему НЕ один fused kernel): наивный fused single-chunk bwd ~56KB threadgroup
  (hats16 + Am4 + dA4 + work32) > 32KB M4. Forward влезал (28KB) — у bwd живой набор больше.

РЕШЕНИЕ — 2 Metal-кернела + дешёвый elementwise-хвост на хосте:
  STAGE-1 (A-grads): recompute hats→Am→u (как S3.3b); затем dA_qk=le(do@v^T),
    dv=A_qk^T@do, dA_qb=le(do@u^T), du=A_qb^T@do, dc=trisolve_T(A_ab,du),
    dA_ab=lt(dc@u^T), dA_ak=lt(dc@v^T), dv+=A_ak^T@dc.
    После Am hats мертвы → переиспуем под work(v,u,do,du,dc,dv). Арена 6*CD+Am=28KB (ОК).
    Выход в device: dA_qk,dA_qb,dA_ab,dA_ak [C,C]×4 + dv [C,D].
  STAGE-2 (hats-grads): recompute hats; dA_* как DEVICE-операнды матмулов →
    dqh=dA_qk@kh+dA_qb@bh, dkh=dA_qk^T@qh+dA_ak^T@ah, dbh=dA_qb^T@qh+dA_ab^T@ah,
    dah=dA_ab@bh+dA_ak@kh. Выход dqh,dkh,dbh,dah [C,D]×4 в device.
  ХОСТ (MLX, дёшево): dq=dqh·e^gc; dk=dkh·e^-gc; dβ=dbh·e^-gc; dα=dah·e^(gc-gk);
    dgc=dqh·qh-dkh·kh-dbh·bh+dah·ah; dgk=-dah·ah+cumsum(dgc,reverse); dw=dgk/w; dr=dq.
  Валидация stage1+stage2+хвост vs chunk_bwd_s0_mlx (S3.4a-i) и vs истина.
Потом: S3.4b — межчанк bwd (dS назад по чанкам: o3=qhat@S и v2=u+wmat@S дают dS-вклад;
  обратный цикл n=N-1..0 с carry dS; формы S-апдейта транспонируются). Затем бенч
  fwd+bwd vs battle (цель ×2–2.4 на полном шаге; перемерить in-situ в 12L).

## S3.4a-ii — DONE (Metal single-chunk backward S=0)
Файлы: s3_dplr_kernel.py (_bwd1_kernel, _bwd2_kernel, chunk_bwd_s0_metal),
  verify_s3.py (main_s34a). Реализован staged-дизайн из плана (2 кернела + хост-хвост).
STAGE-1 (_bwd1): recompute hats→Am→u; dA_qk=le(do@v^T), dv1=A_qk^T@do, dA_qb=le(do@u^T),
  du=A_qb^T@do, dc=trisolve_T(A_ab,du), dA_ab=lt(dc@u^T), dA_ak=lt(dc@v^T), dv+=A_ak^T@dc.
  Арена 6*CD+Am=28KB. Выход: dA_qk,dA_qb,dA_ab,dA_ak[C,C] + dv[C,D] (device).
STAGE-2 (_bwd2): recompute hats; dA_* как DEVICE-операнды; dqh=dA_qk@kh+dA_qb@bh,
  dkh=dA_qk^T@qh+dA_ak^T@ah, dbh=dA_qb^T@qh+dA_ab^T@ah, dah=dA_ab@bh+dA_ak@kh. Арена 24KB.
ХОСТ: dq/dk/dβ/dα из dqh.. ⊙ e^±gc; dgc=dqh·qh-dkh·kh-dbh·bh+dah·ah;
  dgk=-dah·ah+cumsum(dgc,reverse); dw=dgk/w; dr=dq.
Формы matmul: F,T (dA=…@…^T + маска), T,F (A^T@x, tl=true и для [C,C]-left ОК),
  F,F (A@x), dev×tg смешанные — все probe-проверены ранее. Маски le/lt на device-выходе.
Грабли: `{-T}` в КОММЕНТЕ MSL парсился как поле f-строки → NameError; убрал скобки.
Валидация (main_s34a; C=16,D=64; 3 сида; w=model/0.747/0.545/0.270):
  vs MLX-hand-bwd : max rel 2.8e-7
  vs ИСТИНА(recurrence autograd): max rel 9.5e-7  — на ВСЕХ w, ВКЛ. 0.270 ниже пола.
  БИТ-В-БИТ. Числовая слабость backward закрыта и в Metal. Регресс S3.1–S3.4a цел.
Дальше: S3.4b — межчанк backward. Forward: o имеет o3=qhat@S и v2=u+wmat@S (вклад S).
  Обратный цикл n=N-1..0 с carry dS: на чанке n принять dS (от n+1), добавить вклад
  в do-эффекты, выдать dS для n-1 и grad-вклады в w/k/v/β через S-апдейт-ветви
  (S'=S·e^gc_last + (k·dec)^T@v + (β·dec)^T@v2). Транспонировать S-апдейт-матмулы.
  Валидация vs истина на T=2..4 чанка, B*H. Затем бенч fwd+bwd vs battle (in-situ 12L).

## S3.4b-i — DONE (межчанк backward в MLX, carry dS — математика доказана)
Файлы: experiments/dplr_bwd_chunk_mlx.py (chunk_fwd_seq c кэшем + chunk_bwd_seq),
  verify_s34b.py. Один head [T,D]; gc — ВНУТРИчанковый cumsum; межчанк-decay
  несут last/dec через S. Расширяет S3.4a ветвями состояния.
VJP межчанка (обратный цикл n=N-1..0, carry dS = adjoint ПРОИЗВЕДЁННОГО чанком S):
  S-апдейт (S'=last*S + (k*dec)^T@v + (b*dec)^T@v2), транспонирован:
    dS_in += last[:,None]*dS;  dv += (k*dec)@dS;  dv2 += (b*dec)@dS;
    dkdec=v@dS^T; dbdec=v2@dS^T;  dgc_last += last*(dS*S_in).sum(m)
  o=A_qk@v+A_qb@v2+qh@S_in:  dA_qk=le(do@v^T); dv+=A_qk^T@do; dA_qb=le(do@v2^T);
    dv2+=A_qb^T@do;  dqh += do@S_in^T (o3);  dS_in += qh^T@do
  v2=u+wmat@S_in:  du=dv2;  dwmat=dv2@S_in^T;  dS_in += wmat^T@dv2
  u-solve:  dc=M_inv^T@du; dA_ab=lt(dc@u^T); dA_ak=lt(dc@v^T); dv+=A_ak^T@dc
  wmat-solve (ОБЩИЙ M_inv, rhs_w=ah):  dcw=M_inv^T@dwmat; dA_ab+=lt(dcw@wmat^T);
    dah_extra=dcw
  hats: dqh+=dA_qk@kh+dA_qb@bh; dkh=dA_qk^T@qh+dA_ak^T@ah; dbh=dA_qb^T@qh+dA_ab^T@ah;
    dah=dA_ab@bh+dA_ak@kh+dcw
  leaves: dq=dqh*e^gc; dk=dkh*e^-gc(+dkdec*dec); db=dbh*e^-gc(+dbdec*dec); da=dah*e^(gc-gk)
    dgc=dqh*qh-dkh*kh-dbh*bh+dah*ah - ddec*dec  (ddec=dkdec*k+dbdec*b)
    dgc_last += (ddec*dec).sum(i);  dgc[-1]+=dgc_last;  dgk=-dah*ah+cumsum(dgc,rev)
    dw=dgk/w; dr=dq.  carry: dS=dS_in (в чанк n-1).
Валидация (verify_s34b; C=16,D=64; N=2/3/4; 3 сида; w=model/0.747/0.545/0.270):
  forward vs dplr_recurrence_mlx : 2.4e-7 .. 4.1e-7
  grad    vs ИСТИНА(autograd рек.): 4.4e-7 .. 1.4e-6  — на ВСЕХ w, ВКЛ. 0.270 ниже пола
    (где боевой vjp = 3.6e16). Числовая слабость закрыта и для межчанка. PASS.
Дальше: S3.4b-ii — порт в Metal. Новизна vs S3.4a-ii (single-chunk):
  (1) ВТОРОЙ trisolve_T под dwmat (общий A_ab; уже есть примитив из S3.4a-ii);
  (2) S-матмулы с device-S [Dk,Dv]: do@S_in^T (desc F,T), qh^T@do→dS, wmat^T@dv2→dS,
      (k*dec)@dS, (b*dec)@dS, v@dS^T, v2@dS^T — формы dev×tg уже probe-проверены (S3.3c);
  (3) carry dS device-резидентный между чанк-диспатчами (драйвер гонит n=N-1..0
      последовательно, как forward в S3.3c гонит n=0..N-1); BH параллельны в гриде.
  Бюджет: bwd-арена + dS[Dk,Dv]=16KB device (не threadgroup). Stage-split как S3.4a-ii
  (stage1 A-grads+solves, stage2 hats-grads), S-ветви добавить в stage1 (нужен S_in,
  v2, wmat резидентно). Валидация vs chunk_bwd_seq (S3.4b-i) и vs истина на T=2..4, BH.
  Затем БЕНЧ fwd+bwd vs battle (цель шага x2-2.4; перемерить in-situ в 12L).

## S3.4b-i — ДОБАВЛЕНО: многошаговая стабильность + замер reference
Файлы: experiments/verify_grad_steps_b.py (степ-тест АНАЛИТИЧЕСКОГО chunk_bwd_seq,
  не autograd), /tmp/probe_mem.py (замер reference).
Многошаг (B2T64H4D64 C16, N=4, 25 шагов Adam, θ бимодально ~пол 0.545, шаг по ИСТИНЕ):
  аналитический межчанк vs истина : 3e-7 .. 6.8e-6 (плато, all-finite ВСЕ шаги)
  боевой Metal vjp vs истина      : 1.8e-3 .. 1.8e-2 (гуляет, всплески)
  → стабильность подтверждена по ТРАЕКТОРИИ, не только single-shot. PASS.
Грабли (важно для Metal-обвязки): MLX `arr.at[int, :, int].add(x)` РАСКЛАДЫВАЕТ НЕВЕРНО
  (rel ~0.97) — смешанная int/slice/int индексация с двумя скалярами. ЧТЕНИЕ срезов
  ок. Сборку B*H делать stack+reshape+transpose ([B*H,T,D]->[B,H,T,D]->(0,2,1,3)),
  НЕ scatter. (Single-head fwd/bwd были идеальны; баг был только в сборке гарды.)
Замер reference (НЕ показатель Metal): chunk_bwd_seq 18мс/вызов 0.004ГБ vs battle vjp
  2.26мс 0.003ГБ. Это питон-оракол (цикл 8гол×4чанка, полный кэш, Нейман), НЕ ядро.
  Реальные скорость/память = порт S3.4b-ii + in-situ 12L. Ориентир: chunk_step 0.766мс/чанк
  (S3.3c); цель bwd ~10мс поверх fwd чтобы побить battle fwd+bwd=34.5мс.

## S3.4b-ii — ГОТОВО: межчанковый backward на Metal (carry dS, BH-батч)
Подход (микрогард→полный, по решению steer): сперва изолировать формы на одном
чанке со СЛУЧАЙНЫМИ S_in/dS, затем многочанк, затем BH. Лестница вся PASS.
Файлы (experiments/):
  dplr_bwd_one_mlx.py        — per-chunk fwd/bwd ОДНОГО чанка (факторизация цикла),
                               оракул для Metal-микрогарда.
  verify_chunk_one.py        — MLX hand-bwd vs autograd изолир. chunk-fwd (vjp do,dS).
  verify_chunk_one_metal.py  — Metal per-chunk (KB+_bwd2+хост) vs оракул.
  verify_s34b_metal.py       — полный Metal межчанк vs истина & vs chunk_bwd_seq.
  verify_s34b_metal_bh.py    — BH-батч vs поголовный single-head & vs истина.
  /tmp/probe_bwd_forms.py    — probe новых форм (см. ниже).
s3_dplr_kernel.py (+214 стр): _kb_kernel (град-ядро KB), _bwd2_bh_kernel (BH-aware
  hats-grads), _fwd_intermediates_mlx[_bh], chunk_bwd_one_metal[_bh],
  dplr_bwd_metal[_bh].
Разбивка диспатчей = PATH A (по steer: B = «впихнуть невпихуемое», риск упора S-ветвей
  в threadgroup). Per-chunk backward:
  KA (fwd-рекомпьют Am,u,wmat,v2) — пока MLX (_fwd_intermediates_mlx), ПОЗДНИЙ перф-своп.
  KB (град-ядро) — почти всё device-операнды, threadgroup только dc,dcw,dv2,scratch+Tcc
    (~17KB → 2+ группы/ядро, занятость с запасом; НЕ жмёмся к 32KB-потолку).
  _bwd2 / _bwd2_bh (hats-grads) — переиспользован.
  хост — leaves, decay-ветвь, dgc_last, fold, revcumsum, dw=dgk/w, dS_in сборка.
Новые формы матмулов (probe_bwd_forms.py, оба rel=0.00e0):
  F,T с N=D → out[C,D] (dv2@S_in^T, do@S_in^T, v@dS^T, dwmat); ПОЛНОСТЬЮ-device операнды
  (dkdec=v@dS^T). Остальное — формы из step/bwd1/bwd2.
Ловушки порта (закрыты): (1) op.run требует LVALUE операндов — TG()/DV() врапперы
  привязывать к именованным локальным в RUN-макросе, не передавать временными.
  (2) входные dA* — const device float*, кастовать в макросе, не объявлять mutable.
  (3) dS_in копит 3 вклада (last*dS + qh^T@do + wmat^T@dv2) — два матмула в ОТДЕЛЬНЫЕ
  device-буферы dSin_a/dSin_b, сумма на хосте (избегаем device-аккумуляции).
  (4) du=dv2 должен быть ПОЛНЫМ (S-ветвь bdec@dS + o-ветвь A_qb^T@do) ДО u-solve.
Валидация (C=16,D=64,fp32):
  per-chunk MLX hand vs autograd : 3.6e-7 .. 9.6e-7 (вкл. w=0.270)
  per-chunk Metal vs оракул      : 1.8e-7 .. 3.2e-7 (dS_in бит-в-бит 0.0e0)
  полный N=2/3/4 vs истина       : 4.4e-7 .. 1.4e-6 ; vs chunk_bwd_seq 1.5e-7 .. 2.8e-7
  BH=6 батч vs поголовный        : 0.0e0 (бит-в-бит) ; vs истина 6.5e-7 .. 1.4e-6
  ВСЁ PASS на w=model/0.747/0.545/0.270. Числовая слабость закрыта и для Metal-межчанка.
Дальше (S3.4b-iii / опт):
  (1) KA как Metal-кернел (заменить _fwd_intermediates_mlx) — это step-кернел минус
      o/S-апдейт-выходы + экспорт Am,u,wmat,v2; и forward должен СОХРАНЯТЬ граничные
      S_in[n] для всех n (сейчас берём из MLX chunk_fwd_seq-кэша).
  (2) Фьюз обратного N-цикла в один драйвер (снять per-chunk host-overhead ~9мс/слой,
      как в S3.3c); carry dS device-резидентно.
  (3) БЕНЧ fwd+bwd vs battle (цель побить 34.5мс) + in-situ 12L.
  (4) ОТЛОЖЕНО: fp16/bf16-storage (ополовинит арену) — ломает fp32-контракт и рискует
      вернуть слабость на каналах у пола; делать ОТДЕЛЬНО под verify_grad_w-гардой,
      bf16->float требует OS26.1+, half для matmul2d не подтверждён.

## S3.4b-iii — ГОТОВО: SAVE-форвард (KA для rwkv-metal); MLX убран из тяжёлого пути
РЕШЕНИЕ по KA (steer): по таргету — SAVE в rwkv-metal (скорость на маке),
  RECOMPUTE в SwiftRWKV (память on-device iOS). Важно: «дыры на графике» из
  прошлого опыта — это СТАРЫЙ боевой vjp со слабостью у пола (S3.4b-i: 1.8e-3..1.8e-2),
  НЕ артефакт пересчёта. KA-рекомпьют здесь = тот же ДЕТЕРМИНИРОВАННЫЙ Metal-кернел,
  бит-в-бит с форвардом (нет RNG/dropout) → дыр не вносит. Т.е. save↔recompute —
  чистый размен скорость↔память без штрафа по стабильности. (Оговорка: save двигает
  память вверх → против потолка батча/утилизации; если упрёшься — флипнуть в recompute
  и на маке.)
Реализация SAVE (s3_dplr_kernel.py, +~150 стр):
  _step_save_kernel — КОПИЯ проверенного _step_kernel (S3.3c) + сторы Am/u/wmat/v2
    в device-выходы (бюджет threadgroup НЕ меняется; u сохраняется ДО перезаписи в v2).
  dplr_forward_metal_save — форвард + сбор per-chunk кэша (S_in,Am,u,wmat,v2).
  chunk_bwd_one_metal_bh_saved / dplr_bwd_metal_bh_saved — backward на СОХРАНЁННЫХ
    интермедиатах; ни _fwd_intermediates_mlx, ни chunk_fwd_seq (MLX) больше не нужны.
  (RECOMPUTE-путь dplr_bwd_metal_bh оставлен — основа KA для SwiftRWKV.)
Файлы: verify_save_fwd.py, verify_s34b_saved.py.
Валидация (C=16,D=64,BH=6,fp32; N=2/3/4; w=model/0.545/0.270):
  SAVE-форвард: o бит-в-бит vs оригинальный Metal-форвард (0.0e0);
    Am/u/wmat/v2 vs MLX 1.4e-7..2.1e-7 (демонстрация «дыр нет»).
  Полный fwd(SAVE)+bwd vs ИСТИНА: 6.5e-7..1.5e-6; vs recompute-путь 1.5e-7..2.0e-7. PASS.
Тяжёлый путь (fwd+bwd) теперь ЦЕЛИКОМ на Metal; на хосте только дешёвая leaf-сборка.
Дальше: (1) фьюз обратного N-цикла в один драйвер (снять per-chunk host-overhead);
  (2) БЕНЧ fwd(SAVE)+bwd vs battle (34.5мс) + in-situ 12L; (3) SwiftRWKV: KA-recompute
  из dplr_bwd_metal_bh (kernel-порт _fwd_intermediates → Metal); (4) ОТЛОЖ. fp16/bf16.

## S3.4b-iii БЕНЧ (B2T64H4D64 C16, M4, fp32) — bench_s34b.py
Глобальный логгер: скорость (изоляция стадий), peak-память, статы+корректность градиентов.
СКОРОСТЬ (median мс): ours fwd(SAVE) 2.46 (0.615/чанк) + bwd 3.45 (0.862/чанк) = 5.91;
  battle fwd+bwd(vjp) 4.50 → ours/battle = 0.76x (ПОКА МЕДЛЕННЕЕ).
РАЗБИВКА bwd: KB(град-ядро) 2.34мс 68% | bwd2(hats) 1.03мс 30% | host+carry 0.08мс 2.3%.
  ⇒ ВРЕМЯ В KERNEL'АХ, не в обвязке. Фьюз N-цикла бесполезен (overhead 2.3%). Гипотеза
    про per-chunk dispatch-overhead ОПРОВЕРГНУТА бенчем.
ПАМЯТЬ peak: ours fwd 5.1 / bwd 8.8 MB ; battle 5.0 MB (наш +кэш save, на этом размере ок).
ГРАДИЕНТЫ: dr/dw/dk/dv/da/db все finite; rel vs истина 2.8e-7..4.8e-7.
ВАЖНО: ориентир «34.5мс» УСТАРЕЛ — реальный battle fwd+bwd = 4.5мс. Цель: побить 4.5.
Узкое место и рычаги (по приоритету):
  (1) C=16 < sweet-spot matmul2d (тайл 16 — мин.); battle CHUNK=32, N вдвое меньше,
      матмулы крупнее. C=32 у нас НЕ влезет: арена 6*CD=49KB>32KB → нужна D-split/
      прямоуг. геометрия (обсуждалось). Вероятно крупнейший выигрыш.
  (2) KB сейчас почти все операнды DEVICE (выбор под бюджет) → возможно bandwidth-bound
      на мелких матмулах; threadgroup-запас в KB ~15KB свободно → застейджить горячие
      операнды (do,v,v2) в threadgroup.
  (3) слить KB+bwd2 в одно ядро (убрать повторный recompute hats в bwd2 — он их строит
      заново; KB их тоже частично считает) — сократить дубль и dispatch.
  (4) fp16/bf16-storage (отложено) ополовинит арену → откроет C=32 без D-split.
МЕТОДИКА: timed=median/50 (10 warmup), eval-барьеры; ours fwd+bwd = СУММА двух eval
  (верхняя оценка, без overlap); battle = один value_and_grad. Стадии меряны изолированно.

## S3.4b-iii РЕШЕНИЕ по chunk-size: НЕ наращивать C (прокси + bf16-бюджет)
Прокси /tmp/probe_chunksize.py (изолир. ядра, одинаковый объём токенов, нормир.):
  construct(F,T→[C,C]): C16 0.578 / C32 0.769 → 0.75x ХУЖЕ
  apply(F,F):           C16 0.603 / C32 0.539 → 1.12x
  trisolve(серийный):   C16 1.118 / C32 1.216 → 0.92x ХУЖЕ
ПРИЧИНА (жёсткая, не эмпирика): Σ construct = N·C²·D = (T/C)·C²·D = T·C·D — ЛИНЕЙНА по C.
  C=32 → 2× construct-FLOPs/токен, C=48 → 3×. construct ДОМИНИРУЕТ (4 в fwd, 4 dA в KB).
  Утилизация тайла не компенсирует. ⇒ больший C делает дорогое дороже.
bf16-бюджет threadgroup (потолок 32768): C48 step=54KB/bwd2=36KB ВСЁ РАВНО OVER;
  C32 step=РОВНО 32KB (смерть занятости). OS 27.0 — bf16→float по API ок, но смысла нет.
ВЕРДИКТ: C наращивать не через что (не влезает И медленнее). bf16 — только как occupancy-
  твик при ФИКСИРОВАННОМ C=16 (step 28→14KB), отдельно, под verify_grad_w (риск пола).
РЕАЛЬНЫЙ путь к паритету (battle 4.5мс): эффективность ядер при C=16 — стейджинг горячих
  операндов в KB + слияние KB+bwd2 (убрать дубль recompute hats).
ДИАГНОЗ-ГИПОТЕЗА (требует замера): KB 2.34мс/(N4·BH8=32 инст.) = ~73мкс/чанк-инстанс на
  ~16 матмулов [C16] + 2 trisolve + ~15 elementwise + ~20 барьеров. Для крошечных C16-матмулов
  это в основном OVERHEAD/латентность (1 simdgroup=32 потока на чанк; при BH=8 всего 8
  simdgroup'ов на dispatch → GPU недогружен), не FLOP'ы. battle, вероятно, параллельнее.

## S3.4b-iv АРХ-ВЫВОД из FLA (flash-linear-attention, локально rwkv-mlx/flash-linear-attention)
Перф-референс комьюнити = FLA (НЕ Bo Peng rwkv-lm — тот research/correctness). chunk_rwkv7
  → chunk_dplr_delta_rule (fla/ops/generalized_delta_rule/dplr/). ДЕФОЛТ chunk_size=16
  ("в практике достаточно") — подтверждает наш C=16 и прокси (больший C хуже).
КЛЮЧ — РАЗДЕЛЕНИЕ sequential/parallel (мы это упустили):
  • SEQUENTIAL, лёгкое: только state-scan. fwd chunk_dplr_fwd_kernel_h и bwd
    chunk_dplr_bwd_kernel_dhu: grid=(NK,NV,N*H) — тайлы СОСТОЯНИЯ [K,V]×(B*H), НЕ чанки;
    внутри ОДНОГО ядра `for i_t in range(NT)` (fwd) / reversed (bwd), состояние b_h/b_dh
    в регистрах, носится между итерациями (b_h*=exp2(g_last); b_h+=b_hc). Лёгкие [BK,BV]-матмулы.
  • PARALLEL по чанкам, тяжёлое: chunk_A_bwd / chunk_o_bwd: grid=(NK, NT, B*H) — NT В ГРИДЕ,
    все чанки независимо. Здесь dA-матрицы, выход, dv, dq/dk/da/db/dw. Нет послед. зависимости.
  Декупл dv: chunk_o_bwd даёт chunk-local b_dv (parallel); dhu добавляет state-связь
    b_dv2=b_dv+b_bg@b_dh и протаскивает dh. Т.е. тяжёлое — параллельно, связь с состоянием —
    в лёгком послед. scan.
НАША ОШИБКА: KB лепит ВСЕ 16 матмулов ВНУТРЬ последовательного carry-цикла → сериализует
  тяжёлую работу → деградация с N. (Замеры Алексея, холодный чип: при N=1 мы БЫСТРЕЕ battle
  1.1–1.4x; N=2 → 0.62x; N=32 → 0.68x. Само per-chunk ядро эффективно, губит сериализация.)
  Также подтвердилось: lever#1 (батч leaf-сборки) НЕЙТРАЛЕН (0.94–1.05x) — host-tail не был
  стоимостью (мои 42% — термал-артефакт). bottleneck = per-chunk DISPATCH (launch overhead × N).
ПЛАН РЕСТРУКТУРА backward (по FLA), реальный lever:
  1. state-scan ядро dS[n] (mirror dhu): ОДНО ядро, внутр. реверс-цикл по чанкам, dS в
     threadgroup/регистрах, тайл по [K,V] на грид; лёгкая рекуррентность
     dS_in=last*dS + qh^T@do + wmat^T@dv2, dv2=dv_local+bdec@dS. Только это последовательно.
  2. parallel-over-chunk grad-ядра (mirror chunk_A_bwd/o_bwd): grid с NT — dA, dqh_s, dv_local,
     dkdec/dbdec, и пр., все чанки разом (заодно чинит occupancy: NT*BH*тайлы заполняют GPU).
  3. leaf-сборка (уже batched).
  Ожидание: снимает деградацию с N (тяжёлое параллельно), при этом N=1-перевес сохраняем.
ОТЛОЖЕНО к замерам: Xcode Instruments / xctrace (Metal System Trace) для GPU-тайминга ядер
  (честнее wall-clock). bitnet-rwkv-lm (1.58-бит обучение) — ортогонально перфу, в бэклог идей
  для on-device (BitNet ternary), не сейчас.

## S3.4b-v ПЛАН реструктуры backward (FLA-style; следовать по фазам)
Принцип: адъоинт ЛИНЕЕН по (do,dS). Всё зависящее от dS — только через лёгкий послед.
  scan дающий dS[n]; остальное (тяжёлые матмулы) — ПАРАЛЛЕЛЬНО по чанкам, читая dS[n].
  Математика НЕ меняется (chunk_bwd_one доказан) — переносим операции по зависимости.
Порядок: parallel-grad ПЕРВЫМ (по нашей методике «потребитель на доверенном producer»):
  KB уже instance-параметризован → почти бесплатно дать ему все чанки в гриде.

ФАЗА 1 — parallel grad (KB по всем чанкам разом):
  1.1 KB instance-параметризован (grid.y=инстанс, оффсеты bh) — ДА, тело не трогаем.
  1.2 throwaway: dS[n] всех чанков из текущего проверенного послед. цикла (временный producer).
  1.3 диспатч KB ОДИН раз grid=(32, BH*N, 1); inputs flatten [BH,N,...]→[BH*N,...] (contiguous,
      free): q,k,v,beta,gc,do,u,wmat,v2,S_in,Aqk..Aak + dS[n]. Все чанки параллельно.
  1.4 валидация бит-в-бит vs dplr_bwd_metal_bh_batched.
  1.5 замер parallel-KB vs sequential-loop (основной перф-выигрыш). bwd2 уже батчем — оставить.

ФАЗА 2 — лёгкий state-scan (заменить throwaway producer dS[n]):
  2.1 ядро dS-scan: внутр. реверс-цикл по чанкам, dS[D,D] в threadgroup, grid по BH
      (тайл [K,V] позже если нужно). Per chunk (2 матмула): dv2 = (A_qb^T@do)_local + bdec@dS;
      dS = last*dS + (qh^T@do)_local + wmat^T@dv2. Выход dS[n] для всех n.
  2.2 local-прекомп (A_qb^T@do, qh^T@do) — parallel pre-pass (carry-независимы).
  2.3 валидация dS[n] vs throwaway (бит-в-бит).

ФАЗА 3 — сборка + замер:
  3.1 scan→dS[n]→parallel-KB→bwd2(batch)→leaf(batch).
  3.2 сквозная валидация verify_s34b_saved бит-близко vs истина.
  3.3 ЗАМЕР: Instruments (Metal System Trace, xctrace) + холодный чип; fwd+bwd vs battle
      по N=1/2/8/32. Ожидание: деградация с N снята (тяжёлое параллельно), N=1-перевес цел.
Прим. по термалу (Алексей): троттлинга в прошлых замерах НЕ было (чип тёплый при немногих
  долгих память-ёмких операциях). Разброс — методика (прогрев/Python-граф/lazy eval), не железо.

## S3.4b-v ФАЗА 1 ГОТОВО — parallel-KB (KB одним диспатчем по BH*N)
s3_dplr_kernel.py: dplr_bwd_metal_bh_parallel, _bwd2_and_leaf (общий хвост),
  _dS_carry_seq (ВРЕМЕННЫЙ producer dS[n], Фаза 2 заменит). verify_parallel.py, /tmp/bench_kb.py.
КОРРЕКТНОСТЬ: parallel vs batched бит-в-бит 0.0e0 (BH8/32, N4/32).
ИЗОЛИР. ЗАМЕР KB (min-of-many, прогрев25, индикативно — не Instruments):
  BH32 N4: seq(4 disp) 2.39 / par(1 disp) 0.98 → 2.43x
  BH8  N8: 1.32 / 0.84 → 1.57x ;  BH8 N32: 3.25 / 1.88 → 1.73x
  ⇒ тяжёлые матмулы распараллелены, GPU был недогружен даже при BH=32. Лес из FLA верен.
ВАЖНО: полный dplr_bwd_metal_bh_parallel пока ДВОИТ KB-работу (throwaway _dS_carry_seq гоняет
  полный KB ради dS[n] + параллельный KB ради градиентов). Сквозной выигрыш — только после Фазы 2.

## S3.4b-v ФАЗЫ 2-3 ГОТОВО — лёгкий scan + полный быстрый backward (КОРРЕКТНО)
s3_dplr_kernel.py: _prescan_kernel (parallel A_qb^T@do, qh^T@do по BH*N), _dscan_kernel
  (последовательный, внутр. реверс-цикл, dS[D,D] в threadgroup ~20KB, 2 матмула/чанк:
  bdec@dS + wmat^T@dv2, wd через device-scratch), dS_scan_metal (драйвер), dplr_bwd_metal_bh_fast
  (scan→parallel-KB→bwd2-батч→leaf-батч). verify_scan.py, verify_fast.py, bench_fast.py.
КОРРЕКТНОСТЬ: scan dS[n] vs throwaway послед. 0..4.7e-8; full fast vs истина 1.44e-6
  (вкл. w=0.270), vs прежний путь 8.5e-8. PASS. Только scan последователен (лёгкий).
ИЗОЛИР. lever (чисто, min): parallel-KB 1.57-2.43x. Сквозной wall-clock — НЕДОСТОВЕРЕН
  (чип горячий от множества прогонов; battle гулял 2.78/3.08/4.66 на одном конфиге).
  ⇒ авторитетный замер ТОЛЬКО на холодном чипе / Instruments (experiments/bench_fast.py).
ДОВЕСКИ (реальные, не шум): (1) dplr_bwd_metal_bh_fast делает mx.stack N кэш-записей/вызов —
  форвард должен ВОЗВРАЩАТЬ уже сложенные [BH,N,...] (а не списки). (2) ФОРВАРД сам всё ещё
  N послед. диспатчей — тот же FLA-расщеп (forward state-scan отдельно, intra-chunk параллельно)
  = следующий крупный lever.
ДАЛЬШЕ: (a) холодный/Instruments замер fast vs battle по N=1/2/8/32; (b) пре-стек кэшей в
  форварде; (c) FLA-расщеп ФОРВАРДА (parallel intra-chunk + light state-scan); (d) затем
  in-situ 12L и снова vs battle.

## S3.4b-v ДИАГНОЗ Instruments (Metal System Trace, /Users/s/rwkv.trace) — LAUNCH-BOUND
Счётчики M4: Compute Shader Launch Limiter НАСЫЩЕН весь прогон; ALU/F32 Utilization НИЗКИЕ;
  Total Occupancy средняя; Read/Write BW средние (не насыщены). GPU Channel Summary: Compute
  9294 диспатча, min ядро 4.25µs, avg CPU→GPU latency 886µs.
ВЕРДИКТ: bottleneck = ЧИСЛО ДИСПАТЧЕЙ (+ CPU-кодирование команд), НЕ compute/BW. Тысячи
  крошечных ядер, GPU голодает на запусках. (Прошлое «батч leaf нейтрален» по wall-clock —
  артефакт горячего чипа; счётчик показывает: крошечные диспатчи И ЕСТЬ стоимость.)
  Прим.: shader-profiler-interval в шаблоне пуст; MLX не метит энкодеры именами ядер →
  по именам не разложить, но счётчиков достаточно.
ИСТОЧНИКИ ДИСПАТЧЕЙ в fast (по убыванию): leaf-сборка ~20-30 крошечных MLX-элементвайзов;
  форвард 16 послед. step_save; mx.stack кэша/вызов; scan(2)+KB(1)+bwd2(1) — мало.
ПЛАН (cut dispatch count — counter-justified):
  L1. leaf-сборку → ОДНО Metal-ядро (вместо ~25 MLX-диспатчей). Самый чистый выигрыш по
      числу запусков. Параллельно по BH*N.
  L2. FLA-расщеп ФОРВАРДА: light state-scan (как dscan) + parallel intra-chunk → вместо 16
      послед. step_save. Убирает 16 запусков → ~2-3.
  L3. форвард возвращает ПРЕД-сложенные [BH,N,...] кэши (убрать mx.stack/вызов).
  Цель: с ~60-80 диспатчей/итер до ~6-8. Тогда launch-limiter отпустит.

## S3.4b-v L1 ГОТОВО — leaf-сборка в одно Metal-ядро
s3_dplr_kernel.py: _leaf_kernel (вся leaf-математика: элементвайз + 2 редукции (Su*Sin по Dv,
  ddec*dec по C) + реверс-cumsum по времени; threadgroup ~17KB; параллельно по BH*N) +
  _bwd2_and_leaf_fused (bwd2-батч + leaf-ядро). dplr_bwd_metal_bh_fast переключён на fused.
КОРРЕКТНОСТЬ: full fast vs истина 1.44e-6 (вкл. w=0.270), vs прежний путь 8.5e-8. PASS.
ЭФФЕКТ: ~20-30 крошечных MLX-элементвайзов leaf → 1 Metal-ядро. Бэквард grad/leaf теперь
  ~4 ядра (KB + bwd2 + leaf + scan(prescan+dscan)) вместо ~30 диспатчей.
ОСТАВШИЕСЯ источники диспатчей: форвард 16 step_save (L2); mx.stack кэша x5 в fast (L3);
  gc/gk MLX (2, мелочь). Следующий доминант — почти наверняка форвард (L2).
ПРОВЕРКА: повторить трейс (launch-limiter должен заметно просесть); затем L2.

## S3.4b-v L2 ГОТОВО — форвард-расщеп (FLA-style): параллельный WY + лёгкий fscan
По анализу зависимостей форвард расщеплён как backward: тяжёлый WY-реп (Am,u,wmat +
  trisolve'ы) + o_base=A_qk@v + s_base=(k*dec)^T@v — carry-НЕзависимы → параллельно по BH*N;
  только state-рекуррентность (v2=u+wmat@S, o+=A_qb@v2+qh@S, S=last*S+s_base+bdec^T@v2) —
  последовательна, в одном fscan-ядре (внутр. цикл t=0..N-1, S[D,D] в threadgroup, bdec^T@v2
  через device-scratch).
s3_dplr_kernel.py: _wy_kernel (параллельный WY+o_base+s_base), _fscan_kernel (внутр.цикл),
  dplr_forward_metal_save_v2. dplr_bwd_metal_bh_fast переключён на forward v2.
  verify_wy.py, verify_fwd_v2.py.
КОРРЕКТНОСТЬ: WY-выходы vs MLX 1.3e-7; forward v2 (o+весь кэш) vs прежний форвард ~5e-8
  (бит-в-бит при N=2, вкл. w=0.270); full fast vs истина 1.44e-6, vs прежний путь 8.5e-8. PASS.
ЭФФЕКТ: форвард 16 послед. step_save → 2 ядра (wy + fscan). Суммарно Metal-ядер в fast теперь
  7 (wy, fscan, prescan, dscan, KB, bwd2, leaf) — каждое ОДИН диспатч (параллельный или
  внутр.-цикл) — вместо ~60-80 запусков до L1/L2. Прямо бьёт в launch-bound.
ОСТАЛОСЬ (MLX-glue, мелочь): gc/gk (log+cumsum), v2-матмул, stack'и кэш-списков в fast (L3).
ПРОВЕРКА: перезаписать трейс с counter set = Performance Limiters — launch-limiter должен
  заметно отпустить (оценить эффект L1+L2 разом). L3 (убрать stack'и, держать [BH,N,...]) —
  мелкий довесок если glue всплывёт.

## S3.4b-v ДИАГНОЗ после L1+L2 (Performance Limiters counter set) — launch-bound СНЯТ
Замер Алексея (BH32 T256 и BH8 T512): Total Occupancy 98-100% (было низко!); launch-limiter
  больше НЕ доминант. Топ-лимитеры теперь LLC 42% + MMU 38-40% + ALU 39% — сопоставимы, единого
  доминанта нет. L1 cache limiter 13%. ⇒ ВЫШЛИ ИЗ launch-starvation В здоровый, насыщенный
  memory/cache-leaning режим. L1+L2 сработали как задумано (~60-80 диспатчей → 7 ядер).
СЛЕДУЮЩИЕ ЛЕВЕРЫ (memory-обоснованы, инкрементальны — не один большой 2x):
  • стейджинг горячих операндов в threadgroup (KB/leaf/scan device-operand-heavy → LLC-трафик;
    в KB ~15KB threadgroup свободно; do/v/v2/S_in) — срезать LLC 42%.
  • меньше буферов → MMU 38-40%: L3 (кэш в [BH,N,...], убрать mx.stack/вызов — каждый плодит
    буфер), возможно слить KB+bwd2.
NB: нужны wall-clock мс (bench_fast: saved/fast/battle) чтобы знать положение vs battle 4.5мс —
  счётчики дают «здоровую утилизацию», но не «обгон battle». Ждём числа.

## S3.4b-v ВЕХА — wall-clock после L1+L2 (холодный чип, Алексей)
BH=8 T=512 N=32:  fast=9.45-9.47  battle=9.32-9.33  → fast/battle=0.98-0.99x ПАРИТЕТ;
  vs saved (17.7-34.2, шумит) = 1.88-3.61x. На ДЛИННЫХ N реструктура = паритет с battle.
BH=32 T=256 N=16: fast=13.51  battle=9.63  → 0.71x; vs saved 13.75 = 1.02x (≈, т.к. saved при
  BH=32 уже занят 32 simdgroup/диспатч).
ИНТЕРПРЕТАЦИЯ: fast выигрывает где saved был задушен (low BH и/или high N). При high BH+mid N
  saved уже ок → паритет fast≈saved, и 0.71x vs battle. fast СТАБИЛЕН (9.45-9.47) vs saved
  шумит (17-34) — меньше диспатчей. Прогресс перф-работы: старт 0.31-0.53x → теперь 0.71-0.99x.
ОТСТАВАНИЕ при BH=32 = LLC 42%/MMU 38%: battle почти не растёт со временем (9.33→9.63 при 2x
  инстансов), мы линейны → battle эффективнее по памяти (регистры/threadgroup vs наши device-
  round-trip'ы). ЛЕЧИТСЯ: стейджинг операндов в threadgroup (LLC) + L3 буферы (MMU).
СТАТУС: паритет на длинном контексте достигнут. Дожать BH=32 — стейджинг/L3 (вероятно обгон).
ДАЛЬШЕ: (a) стейджинг операндов KB/leaf/scan + L3; ИЛИ (b) сразу in-situ 12L (раз паритет на
  длинном N есть) + сквозной тренинг-степ vs battle.

## S3.4b-v IN-SITU 12L (РЕАЛЬНЫЙ замер) — отрезвляющий результат
Конфиг: PretrainConfig() = 12L, d=768, H=12, S=64, vocab=21248, B=8, T=512.
  wkv7 на [8,512,12,64] → BH=96, N=32 (C=16). Battle CHUNK=32 (N=16). Модель идентична
  (passthrough база одна), отличается только wkv7-ядро.
Числа (12L fwd+bwd step, median мс): passthrough 1648 | battle 2351 | ours 3301.
  wkv7 вклад: battle 703 (30% шага) | ours 1654 (50%). шаг ours/battle=0.712x;
  ТОЛЬКО-wkv7 ours/battle=0.425x.
ВАЖНО / признание: прошлый «паритет» был при BH=8 — НЕПРЕДСТАВИТЕЛЬНО. Реальная модель BH=96.
  Тренд по BH подтвердился до конца: BH8 паритет → BH32 0.71x → BH96 0.425x (LLC/MMU memory-
  bound растёт с BH; battle эффективнее по памяти). wkv7=30% шага (не доминанта, но значимо):
  довести ядро до battle = шаг 3301→2351 (1.4x).
НЕЧЕСТНОСТЬ к нам: in-situ vjp РЕКОМПЬЮТИТ форвард (forward 2×), а наш rwkv-metal дизайн = SAVE
  (форвард 1×). Завышает наше время на ~один форвард. Надо протащить кэш через custom_function
  outputs (save), пере-замерить — честное число.
ПЛАН: (1) custom_function с SAVE (кэш через outputs, без рекомпьюта в vjp) → честный in-situ.
  (2) затем memory-staging при BH=96: стейджинг do/v/v2/S_in в threadgroup KB (LLC), сократить
  буферы/стек (MMU). Цель — закрыть 0.425x→паритет при РЕАЛЬНОМ BH=96 (не BH=8).

## S3.4b-v IN-SITU SAVE vs RECOMPUTE (BH=80, N=32; d=256 vocab32000 B=20) — ЧИСТО
12L fwd+bwd step (median, min≈median — стабильно): passthrough 1074 | battle 1634.
  ours SAVE 1976 (wkv7 902, 46%) | ours RECOMPUTE 2256 (wkv7 1183, 52%).
SAVE убирает ~281ms (лишний форвард) → ядро save на ~24% быстрее recompute. Подтверждает
  дизайн: SAVE для rwkv-metal (мак), RECOMPUTE для SwiftRWKV (iOS) — с цифрой.
СТАНДИНГ (save, реальный BH=80): ядро ours/battle=0.62x; ШАГ ours/battle=0.827x;
  wkv7=34% шага. (0.425x ранее = recompute+BH96; честная цифра 0.62x.)
ЦЕНА разрыва: довести ядро до battle → шаг 1976→~1634 = ~17-21% ускорение претрейн-шага.
ОСТАТОК = LLC/MMU memory-bound: стейджинг do/v/v2/S_in в threadgroup KB + меньше буферов.
  Инкрементально, потолок неопределён (0.8x..паритет).
ИТОГ перф-работы: старт 0.31-0.53x → save@BH80 шаг 0.827x / ядро 0.62x. Корректность на каждом
  шаге (vs истина 1.44e-6 вкл w=0.270). Полный fwd+bwd на Metal, save-режим, custom_function
  drop-in в реальную 12L-модель работает.
РЕШЕНИЕ (открыто): дожимать ядро (стейджинг, ~20% шага) ИЛИ принять (корректное save-ядро +
  основа SwiftRWKV-recompute; battle пока быстрее для rwkv-metal-претрейна).
