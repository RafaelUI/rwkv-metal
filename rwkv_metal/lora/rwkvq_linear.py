"""
QLoRA-база на РОДНОМ формате rwkv-quant (.rwkvq, gw_mode="sb6"), а не на
стоковом mlx.nn.quantize.

ОДНА СТАДИЯ ВМЕСТО ДВУХ (05.08). Раньше здесь было написано, что
.rwkvq требует torch, а потому нужен промежуточный сайдкар: export_mlx
(в venv rwkv-quant, где torch есть) перекладывал файл в
*.rwkvq_mlx.safetensors + .json, и уже его читал этот модуль. Оба
утверждения устарели:

  - `rwkv_quant.formats.codec` читает контейнер и строит K3-интерлив на
    numpy, без torch (гейт rwkv-quant/tests/test_torch_free_import.py);
  - `load_sidecar` принимает и .rwkvq напрямую, и прежний сайдкар.

Сайдкар остался НЕОБЯЗАТЕЛЬНЫМ кешем: он экономит построение интерлива
при загрузке, но стоит отдельного файла и +45 МБ на 2.9B. Что оба пути
дают бит-в-бит одно и то же, проверяет tests/dev_rwkvq_direct.py
(453.5M элементов на 2.9B, расхождений нет).

Dense-вес по-прежнему восстанавливается НА ЛЕТУ при каждом forward
(транзиент, не кешируется -- смысл QLoRA: база должна жить в памяти
сжатой). Для ИНФЕРЕНСА этот размен другой, см. rwkvq_native.py.

Точность: dequant делается в float32 (НЕ float16, в отличие от
GwQuantLinear._dequant_w(), который держит математику в half ради
скорости GEMM-префилла и даёт ~18% расхождений на 1 бит бф16-мантиссы) --
здесь нужен бит-в-бит с rwkv_quant.formats.reader._dequantize_gw_sb6,
т.к. REDUCTION-пресет откалиброван именно под эту математику
(ppl 11.4438, "деградация около нуля" -- QLoRA-база должна её сохранять,
не добавлять свой источник шума поверх калибровки).
Сверено бит-в-бит с PyTorch-референсом на реальном reduction_v2.rwkvq
(tests/dev/mlx_dequant_precise_fast.py в rwkv-quant) -- 0 расхождений.

_dequant_w() использует fused Metal-кернель (rwkvq_kernel.py) -- один
launch вместо ~8 отдельных MLX-операций. Замерено на реальных тензорах
(tests/dev_check_rwkvq_fused_kernel.py): 3.8-7.1x быстрее композитного
порта, бит-в-бит идентичен. _dequant_w_slow() оставлен как медленный,
чисто-MLX референс для отладки/кросс-проверки при подозрении на баг в
кернеле.
"""
import json
import os

import mlx.core as mx
import mlx.nn as nn
from .rwkvq_kernel import dequant_dense

# ЕДИНЫЙ СПИСОК КВАНТОВАННЫХ РАСКЛАДОК. Их перечисляют ТРИ места: этот
# загрузчик, диспетчер бэкендов (add_rwkvq._backend_for) и конвертер
# (model/convert.py), который по нему решает, ставить заглушку или
# деквантовать. Пока список был один ("sb6"), расхождение было
# невозможно; с приходом sym конвертер отстал -- и молча разворачивал всю
# базу в плотный bf16 (145 тензоров, +5.4 ГБ пика, время сборки x5) при
# том, что логиты оставались верными до бита. Поэтому список ровно один.
QUANTIZED_KINDS = ("sb6", "sym")

# КАСТ ВХОДА В КВАНТОВАННОМ ЛИНЕЙНОМ СЛОЕ (внедрено 22.08, решение
# владельца). Прежний `x @ w.T` без приведения позволял fp32-выходу ядра
# WKV тащить за собой fp32-копию деквантованной матрицы и fp32-матмул на
# каждом слое -- 12-27% тренировочного шага (bench_wkv_fp32_tail_ab).
# Правка делает ровно то, что инференсный путь rwkv-quant
# (`quant_model._matmul`) делает с самого начала. Гейты: ppl на обоих
# масштабах нейтрален (+0.0044%/+0.0024%, не значимо), KL-сдвиг от bf16
# +3% против +23% у CAST_WKV_OUTPUT (probe_cast_kl). Гейт правки --
# tests/test_lincast_parity.py. RWKVQ_NOCAST=1 (или rl.NOCAST = True)
# возвращает прежнее поведение для A/B В ОДНОМ ПРОЦЕССЕ (закон 27);
# флаг читается на каждый вызов.
NOCAST = os.environ.get("RWKVQ_NOCAST") == "1"

# FUSED dequant+GEMM для sym (23.08, приоритет 1). Заменяет пару
# «деквант-кернель bf16 + плотный матмул» ОДНИМ кернелем: без плотного
# транзиента и с половиной запусков. Градиент -- через mx.custom_function
# (mx.fast-кернель непрозрачен для автограда, без VJP адаптеры ниже слоя
# молча получили бы НУЛЕВЫЕ градиенты -- ловушка хуже падения).
#
# ИЗМЕРЕНО (bench_fused_gemm_ab, чередование, своп-контроль): лосс и
# градиенты БИТ-В-БИТ прежнему пути, пик шага 3272 -> 2976 МБ (-296),
# но СКОРОСТЬ 2728 -> 2818 мс (-3.3%, валидный прогон; второй прогон
# -5.2% при росте свопа -- самопомечен невалидным). Оценка «~200 мс/шаг»
# из NEXT_SESSION НЕ подтвердилась: fused-GEMM идёт на 1.9-2.5 ТФ против
# 2.3-2.8 у плотного матмула, и это съедает всю экономию декванта.
# Поэтому УМОЛЧАНИЕ ВЫКЛ (opt-in RWKVQ_FUSED=1), включение -- решение
# владельца: -300 МБ пика против -3..5% шага. Рантайм-реверс для A/B в
# одном процессе: rl.NOFUSED = True/False (закон 27). FUSED_CALLS --
# счётчик включения для гейтов/бенчей: «правка не применилась» обязана
# быть видимой, а не читаться как «правка чиста».
FUSED_ENABLED = os.environ.get("RWKVQ_FUSED") == "1"
NOFUSED = not FUSED_ENABLED
FUSED_CALLS = 0


def _matmul_cast(self, x):
    if NOCAST:
        return x @ self._dequant_w().T
    w = self._dequant_w()
    return (x.astype(w.dtype) @ w.T).astype(x.dtype)


def _sym_fused_call(self, x):
    """__call__ RwkvqSymLinear с fused-путем для T >= 64.

    Семантика -- ПОБИТНО прежней строки `(x.astype(bf16) @ W.T).astype(x.dtype)`
    в части точности: кернель округляет веса до bf16 тем же текстом, что
    деквант-кернель, копит во float, выход -- bf16; меняется только ПОРЯДОК
    суммирования (гейт пороговый, relmax ~ 3e-3)."""
    global FUSED_CALLS
    lead = x.shape[:-1]
    x2d = x.reshape(-1, self.in_features)
    T = x2d.shape[0]
    sym = self._sym
    if (not NOFUSED) and T >= sym.GEMM_FUSED_MIN_T and sym.gemm_fused_ok:
        if self._fused_fn is None:
            self._fused_fn = _build_sym_fused_fn(sym)
        xb = x2d if x2d.dtype == mx.bfloat16 else x2d.astype(mx.bfloat16)
        out = self._fused_fn(xb).reshape(*lead, self.out_features)
        FUSED_CALLS += 1
        return out if x.dtype == mx.bfloat16 else out.astype(x.dtype)
    return _matmul_cast(self, x)


def _build_sym_fused_fn(sym):
    @mx.custom_function
    def _f(xb):
        return sym.gemm_fused(xb)

    @_f.vjp
    def _f_vjp(primals, cotangent, output):
        # VJP прежнего пути: dX = dY.astype(bf16) @ W (порядок суммирования
        # другой, семантика та же); котангенты только по x -- буферы базы
        # константы и градиента не требуют
        return (sym.gemm_fused_vjp(cotangent.astype(mx.bfloat16)),)

    return _f

_SIDECAR_CACHE = {}


def _load_rwkvq_direct(path: str):
    """`.rwkvq` -> те же (arrays, manifest), что даёт сайдкар.

    Сайдкар больше не нужен: `rwkv_quant.formats.codec` умеет и читать
    контейнер, и строить K3-интерлив, и всё это на numpy без torch
    (проверено гейтом test_torch_free_import). Раньше промежуточный файл
    был обязателен только потому, что интерлив умел строить лишь
    export_mlx через torch.

    K3 строится для ВСЕХ sb6-тензоров сразу, а не лениво по ключу: это
    ровно та память, которую занимал сайдкар (1.8 ГБ на 2.9B), и
    усложнять ради отложенной сборки нечего.
    """
    from rwkv_quant.formats import codec

    manifest, buf = codec.open_rwkvq(path)
    arrays, tensors, syms = {}, {}, {}
    for key, meta in manifest["tensors"].items():
        kind = meta.get("kind")
        if kind not in QUANTIZED_KINDS:
            continue

        def b(field):
            return buf.get(f"{key}::{field}")

        if kind == "sym":
            # Q6_K-раскладка нынешнего пресета REDUCTION. Интерлив НЕ
            # собирается здесь заново: берётся SymQuantLinear из
            # rwkv-quant (закон 23 -- параллельные реализации расходятся
            # ровно тогда, когда правку вносят в одну из них). Он
            # torch-free, поэтому импорт сюда законен.
            from rwkv_quant.backends.metal.quant_linear_sym import (
                SymQuantLinear)
            try:
                syms[key] = SymQuantLinear.from_buffers(
                    shape=tuple(meta["shape"]), bits=meta["bits"],
                    qs=b("gw_qs"), d=b("gw_d"), codes=b("codes"),
                    codes_packed=b("codes_packed"),
                    qh=b("gw_qh"), qh2=b("gw_qh2"))
            except AssertionError:
                # Кернель требует IN кратным суперблоку 256. Пропускаем
                # МОЛЧА сознательно: если этот ключ и правда нужен,
                # ругнётся проверка покрытия в add_rwkvq -- и назовёт
                # причину, а не просто имя тензора.
                continue
            tensors[key] = dict(meta)
            continue

        qblk, qsqm, ddm, xbits = codec.sb6_to_k3(
            b("codes_packed"), b("gw_qsqm"), b("gw_d"), b("gw_dm"),
            shape=tuple(meta["shape"]), gs=meta["gw_gs"], sb=meta["gw_sb"],
            nb=meta.get("n_blocks"), qh=b("gw_qh"), qh2=b("gw_qh2"))
        arrays[f"{key}::qblk"] = mx.array(qblk)
        arrays[f"{key}::qsqm"] = mx.array(qsqm)
        arrays[f"{key}::ddm"] = mx.array(ddm)
        # xbits в манифесте .rwkvq нет: там он выводится из наличия
        # битплоскостей, а сайдкар хранил его явно. Проставляем, чтобы
        # потребители не различали источник.
        tensors[key] = dict(meta, xbits=xbits)
    mx.eval(*arrays.values())
    # Объекты кладутся ПОСЛЕ mx.eval: их буферы уже отевалуированы внутри
    # from_buffers, а сами они не mx.array и в eval попасть не должны.
    for key, lin in syms.items():
        arrays[f"{key}::sym"] = lin
    return arrays, dict(manifest, tensors=tensors)


def load_sidecar(path: str):
    """Квантованная база: `.rwkvq` НАПРЯМУЮ либо прежний сайдкар.

    path -- либо путь к `.rwkvq`, либо сайдкар без суффикса
    .safetensors/.json (как его кладёт export_mlx). Различается по
    расширению и по наличию файлов, чтобы прежние вызовы работали без
    правок."""
    if path in _SIDECAR_CACHE:
        return _SIDECAR_CACHE[path]
    if path.endswith(".rwkvq") or not os.path.exists(path + ".safetensors"):
        arrays, manifest = _load_rwkvq_direct(path)
    else:
        arrays = mx.load(path + ".safetensors")
        with open(path + ".json") as f:
            manifest = json.load(f)
    _SIDECAR_CACHE[path] = (arrays, manifest)
    return arrays, manifest


class RwkvqLinear(nn.Module):
    """Frozen linear поверх sb6-квантованного тензора из .rwkvq.
    y = x @ W^T, W восстанавливается на лету (не хранится dense)."""

    def __init__(self, qblk, qsqm, ddm, shape, gw_gs, gw_sb, xbits):
        super().__init__()
        OUT, IN = shape
        self.out_features, self.in_features = OUT, IN
        self.NB, self.NSB = IN // gw_gs, IN // (gw_gs * gw_sb)
        self._gw_sb = gw_sb
        self.xbits = xbits
        self.qblk = qblk
        self.qsqm = qsqm
        self.ddm = ddm
        self.freeze()

    @classmethod
    def from_sidecar(cls, sidecar_path: str, key: str):
        arrays, manifest = load_sidecar(sidecar_path)
        meta = manifest["tensors"][key]
        return cls(
            arrays[f"{key}::qblk"], arrays[f"{key}::qsqm"], arrays[f"{key}::ddm"],
            tuple(meta["shape"]), meta["gw_gs"], meta["gw_sb"], meta["xbits"],
        )

    def _dequant_w(self) -> mx.array:
        w32 = dequant_dense(self.qblk, self.qsqm, self.ddm,
                             self.out_features, self.in_features,
                             gw_sb=self._gw_sb, xbits=self.xbits)
        return w32.astype(mx.bfloat16)

    def _dequant_w_slow(self) -> mx.array:
        """Медленный чисто-MLX путь (~8 отдельных операций) -- держим для
        отладки/кросс-проверки фьюз-кернеля, не для использования в forward."""
        OUT, IN, NB, NSB = self.out_features, self.in_features, self.NB, self.NSB
        blk = self.qblk.reshape(OUT, NB, 16 + 4 * self.xbits)
        cb = blk[:, :, :16]
        q = mx.concatenate([cb & 0xF, cb >> 4], axis=2).astype(mx.float32)
        if self.xbits >= 1:
            qh = blk[:, :, 16:20].reshape(OUT, IN // 8)
            bits = (qh[..., None] >> mx.arange(8, dtype=mx.uint8)) & 1
            q = q + bits.reshape(OUT, NB, 32).astype(mx.float32) * 16.0
        if self.xbits >= 2:
            qh2 = blk[:, :, 20:24].reshape(OUT, IN // 8)
            bits2 = (qh2[..., None] >> mx.arange(8, dtype=mx.uint8)) & 1
            q = q + bits2.reshape(OUT, NB, 32).astype(mx.float32) * 32.0

        sm = self.qsqm.reshape(OUT, NB, 2)
        qs = sm[:, :, 0].astype(mx.float32)
        # qm уже хранится как int8 со сдвигом -31, применённым при экспорте
        # (см. GwQuantLinear.__init__: qm_np = unpack6(...) - 31, .to(int8));
        # здесь только реинтерпретация байта, БЕЗ повторного сдвига.
        qm = mx.view(sm[:, :, 1], mx.int8).astype(mx.float32)

        dd = self.ddm.reshape(OUT, NSB, 2)
        d = dd[:, :, 0].astype(mx.float32)
        dm = dd[:, :, 1].astype(mx.float32)
        sb = NB // NSB
        d_c = mx.repeat(d, sb, axis=1)
        dm_c = mx.repeat(dm, sb, axis=1)

        scale = (qs * d_c).astype(mx.float16).astype(mx.float32)
        scale = mx.maximum(scale, 1e-8)
        mn = (qm * dm_c).astype(mx.float16).astype(mx.float32)

        w = q * scale.reshape(OUT, NB, 1) + mn.reshape(OUT, NB, 1)
        return w.reshape(OUT, IN).astype(mx.bfloat16)

    def __call__(self, x):
        return _matmul_cast(self, x)


class RwkvqSymLinear(nn.Module):
    """Frozen linear поверх sym-квантованного тензора (Q6_K-раскладка).

    ПОЧЕМУ НЕ ЧЕРЕЗ РОДНОЙ quantized_matmul, как sb6. У sym блок ШЕСТНАДЦАТЬ,
    а `mx.quantized_matmul` принимает group_size только 32, 64 и 128 --
    это проверено перебором, а не выведено
    (rwkv-quant/tests/probe_prefill_affine.py). Точного репака не
    существует, а неточный означал бы ДРУГУЮ квантованную базу под тем же
    именем -- ровно тот сорт подмены, от которого в проекте есть отдельный
    закон. Поэтому здесь путь RwkvqLinear: деквант в плотную на каждый
    forward, база в памяти живёт сжатой.

    ЦЕНА ЭТОГО ВЫБОРА РЕАЛЬНА И ЕЁ НАДО МЕРИТЬ, А НЕ СЧИТАТЬ: sb6-база на
    родном матмуле плотную матрицу не материализует вовсе, а эта --
    материализует на каждый вызов, и автограду она нужна ещё и в backward
    (градиент по x есть dY @ W). Закрывается это своим VJP с ПЕРЕСЧЁТОМ
    декванта вместо хранения; здесь этого нет сознательно -- сначала
    паритет с sb6-путём, потом отдельная правка с отдельным замером.

    Интерлив и кернель НЕ ДУБЛИРУЮТСЯ: берётся SymQuantLinear из
    rwkv-quant. Деквант считается в fp32, где он бит-в-бит совпадает с
    нормативным `codec.dequant_sym` (гейт
    rwkv-quant/tests/test_sym_dequant_fp32.py), и приводится к bf16 ровно
    там же, где это делает sb6-путь -- чтобы две базы отличались
    раскладкой, а не точностью математики.
    """

    def __init__(self, sym):
        super().__init__()
        # Имя с подчёркиванием -- НЕ параметр модуля: буферы базы заморожены
        # и в parameters() им делать нечего (иначе оптимизатор увидел бы
        # квантованную базу как обучаемое).
        self._sym = sym
        self.out_features = sym.out_features
        self.in_features = sym.in_features
        self.bits = sym.bits
        self._fused_fn = None
        self.freeze()

    @classmethod
    def from_sidecar(cls, sidecar_path: str, key: str):
        arrays, _ = load_sidecar(sidecar_path)
        lin = arrays.get(f"{key}::sym")
        if lin is None:
            raise KeyError(
                f"{key}: sym-буферов нет. Либо тензор в файле не sym, либо "
                f"его IN не кратен суперблоку 256 и кернель его не берёт.")
        return cls(lin)

    def _dequant_w(self) -> mx.array:
        # Прямой bf16-выход кернеля (23.08): прежде было
        # `_dequant_w(mx.float32).astype(mx.bfloat16)`, и astype-перекладка
        # стоила 30% цепочки декванта (68.6 мс на проход по модели, два
        # прохода на шаг с чекпоинтингом). Бит-в-бит с прежней цепочкой --
        # гейт tests/test_sym_dequant_bf16.py на всех 146 sym-тензорах.
        return self._sym._dequant_w(mx.bfloat16)

    def __call__(self, x):
        return _sym_fused_call(self, x)
