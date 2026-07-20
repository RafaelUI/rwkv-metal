"""
QLoRA поверх нативного .rwkvq-квантования (сайдкар из
rwkv_quant.formats.export_mlx), а не стокового mlx.nn.quantize.

Соответствие имён: rwkv-metal (tmix.r_proj/k_proj/v_proj/o_proj,
cmix.key/value, head) <-> официальные ключи .rwkvq (см.
model/convert.py::convert() -- та же официальная схема имён, т.к.
REDUCTION/COMPRESSION калибровались на официальном x070-чекпоинте).

emb.weight ПОКА не подключён: nn.Embedding делает gather по индексам, а
не x@W^T; RwkvqLinear рассчитан на Linear-семантику. Полный dense-декванд
65536x2048 таблицы ради лукапа нескольких строк -- пустая трата, нужен
отдельный indexed-gather-путь (см. NEXT_SESSION/задачи) -- не сделан.

load_lora_rwkvq_model() -- предпочтительный вход: грузит .pth и сразу
подменяет proj/cmix/head на RwkvqLinear ДО материализации их случайной
bf16-инициализации (см. model/convert.py::load_pretrained_partial
pre_materialize_hook) -- экономит пик памяти ЗАГРУЗКИ, не только
тренировки.
"""
from .lora import LoRALinear, TMIX_TARGETS, _unfreeze_adapters, _param_stats
from .rwkvq_linear import RwkvqLinear
import mlx.core as mx

_TMIX_KEY = {"r_proj": "receptance", "k_proj": "key", "v_proj": "value", "o_proj": "output"}


def _replace_targets_with_rwkvq(model, sidecar_path: str, rank: int, alpha: float,
                                 dropout: float, tmix_targets, quantize_cmix: bool,
                                 quantize_head: bool, layers) -> list:
    """setattr-замена proj/cmix/head на RwkvqLinear/LoRALinear. Без
    freeze()/eval() -- вызывающий код решает, когда их делать (для
    load_lora_rwkvq_model это откладывается до конца, после материализации
    остального bf16, чтобы был один проход eval, а не два)."""
    n_layer = len(model.blocks)
    sel = set(range(n_layer)) if layers is None else set(i % n_layer for i in layers)
    wrapped = []

    for li, blk in enumerate(model.blocks):
        if li not in sel:
            continue
        for name in tmix_targets:
            mod = getattr(blk.tmix, name, None)
            if mod is None:
                continue
            key = f"blocks.{li}.att.{_TMIX_KEY[name]}.weight"
            base = RwkvqLinear.from_sidecar(sidecar_path, key)
            setattr(blk.tmix, name, LoRALinear(rank=rank, alpha=alpha,
                                                dropout=dropout, base_module=base))
            wrapped.append(f"tmix.{name}")

        if quantize_cmix:
            for name in ("key", "value"):
                key = f"blocks.{li}.ffn.{name}.weight"
                setattr(blk.cmix, name, RwkvqLinear.from_sidecar(sidecar_path, key))

    if quantize_head:
        model.head = RwkvqLinear.from_sidecar(sidecar_path, "head.weight")

    return wrapped


def add_lora_rwkvq(model, sidecar_path: str, rank: int = 16, alpha: float = 32.0,
                    dropout: float = 0.0, tmix_targets=TMIX_TARGETS,
                    quantize_cmix: bool = True, quantize_head: bool = True,
                    layers=None):
    """Как _replace_targets_with_rwkvq, но на УЖЕ полностью загруженной
    (bf16) модели -- удобно для экспериментов, но платит полный пик
    загрузки .pth. Для тренировки с нуля предпочтительнее
    load_lora_rwkvq_model (ленивая загрузка)."""
    wrapped = _replace_targets_with_rwkvq(model, sidecar_path, rank, alpha, dropout,
                                           tmix_targets, quantize_cmix, quantize_head, layers)
    model.freeze()
    _unfreeze_adapters(model)
    mx.eval(model.parameters())

    info = _param_stats(model)
    info["wrapped_per_block"] = sorted(set(wrapped))
    info["num_lora_adapters"] = len(wrapped)
    info["quantize_cmix"] = quantize_cmix
    info["quantize_head"] = quantize_head
    return model, info


def _rwkvq_skip_keys(pth_path, tmix_targets, quantize_cmix, quantize_head, layers):
    """Официальные ключи .pth, которые НЕ нужно материализовывать из
    bf16 -- они всё равно заменяются RwkvqLinear. n_layer -- дёшево, без
    чтения тела тензоров (см. convert._load_pth_lazy)."""
    from ..model.convert import _load_pth_lazy
    z_lazy = _load_pth_lazy(pth_path)
    n_layer = 1 + max(int(k.split(".")[1]) for k in z_lazy if k.startswith("blocks."))
    sel = set(range(n_layer)) if layers is None else set(i % n_layer for i in layers)
    skip = set()
    for i in sel:
        for name in tmix_targets:
            skip.add(f"blocks.{i}.att.{_TMIX_KEY[name]}.weight")
        if quantize_cmix:
            skip.add(f"blocks.{i}.ffn.key.weight")
            skip.add(f"blocks.{i}.ffn.value.weight")
    if quantize_head:
        skip.add("head.weight")
    return skip, n_layer


def load_lora_rwkvq_model(pth_path, sidecar_path, rank: int = 16, alpha: float = 32.0,
                           dropout: float = 0.0, tmix_targets=TMIX_TARGETS,
                           quantize_cmix: bool = True, quantize_head: bool = True,
                           layers=None, config=None, verbose: bool = True):
    """Загрузка + QLoRA-обвязка ЗА ОДИН заход. proj/cmix/head заменяются
    на RwkvqLinear СРАЗУ после конструирования скелета модели -- ДО того,
    как их случайная bf16-инициализация (полного размера, из
    RWKV7X070.__init__) успевает материализоваться, и ДО того, как
    начинается загрузка+материализация РЕАЛЬНЫХ bf16-данных для
    остальных ~650 тензоров. См. model.convert.load_pretrained_partial.

    Возвращает (model, config, info) -- info как из add_lora_rwkvq.
    """
    from ..model.convert import load_pretrained_partial
    skip_keys, _ = _rwkvq_skip_keys(pth_path, tmix_targets, quantize_cmix, quantize_head, layers)

    wrapped_holder = []

    def hook(m):
        wrapped_holder.extend(_replace_targets_with_rwkvq(
            m, sidecar_path, rank, alpha, dropout, tmix_targets,
            quantize_cmix, quantize_head, layers))

    model, cfg = load_pretrained_partial(pth_path, skip_keys, config=config, verbose=verbose,
                                          pre_materialize_hook=hook)
    if model is None:
        raise RuntimeError("load_pretrained_partial: конверсия не чистая (см. лог выше)")

    model.freeze()
    _unfreeze_adapters(model)
    mx.eval(model.parameters())

    info = _param_stats(model)
    info["wrapped_per_block"] = sorted(set(wrapped_holder))
    info["num_lora_adapters"] = len(wrapped_holder)
    info["quantize_cmix"] = quantize_cmix
    info["quantize_head"] = quantize_head
    return model, cfg, info
