"""torch-free загрузчик RWKV .pth (torch zip-serialization) → MLX-массивы.

load_pth() материализует ВСЁ сразу (как раньше). _load_pth_lazy() -- то же
самое, но каждый тензор -- _LazyTensor (shape известен из pickle-метаданных
БЕСПЛАТНО, реальное чтение+decompress байтов из zip откладывается до
.materialize()). Нужно для load_pretrained_partial(): пропустить дорогое
чтение+decompress ГИГАБАЙТОВ данных под тензоры, которые всё равно тут же
заменяются сжатыми (rwkv_metal.lora.add_lora_rwkvq) -- иначе платим полный
пик памяти загрузки (~2.3x размера файла, см. NEXT_SESSION QLoRA-заметки)
за данные, которые тут же выбрасываются.
"""
import io, zipfile, pickle
import numpy as np
import mlx.core as mx
from collections import OrderedDict

# имя класса storage -> (numpy dtype | 'bf16', itemsize)
_DT = {
    'FloatStorage': (np.float32, 4), 'HalfStorage': (np.float16, 2),
    'BFloat16Storage': ('bf16', 2), 'DoubleStorage': (np.float64, 8),
    'LongStorage': (np.int64, 8), 'IntStorage': (np.int32, 4),
    'ByteStorage': (np.uint8, 1), 'BoolStorage': (np.bool_, 1),
}

class _Stor:
    def __init__(self, dt, isz, key, numel):
        self.dt, self.isz, self.key, self.numel = dt, isz, key, numel


class _LazyTensor:
    """Развёрнутая torch-тензор-ссылка: shape известен сразу (из pickle),
    сами байты читаются из zip и декодируются в mx.array только в
    .materialize() (закешировано после первого вызова)."""
    __slots__ = ("_zf", "_prefix", "_storage", "_offset", "_size", "_cached")

    def __init__(self, zf, prefix, storage, storage_offset, size):
        self._zf, self._prefix, self._storage = zf, prefix, storage
        self._offset, self._size = storage_offset, size
        self._cached = None

    @property
    def shape(self):
        return tuple(self._size)

    def reshape(self, *shape):
        # convert() дёргает .reshape/.T на МЕЛКИХ тензорах (k_k/k_a/r_k,
        # w/a/g/v-lora) -- те никогда не в skip-наборе, материализация
        # тут дешёвая и ожидаемая (не большой proj/cmix/head).
        return self.materialize().reshape(*shape)

    @property
    def T(self):
        return self.materialize().T

    def materialize(self) -> mx.array:
        if self._cached is None:
            raw = self._zf.read(f"{self._prefix}/data/{self._storage.key}")
            n = 1
            for s in self._size: n *= s
            start = self._offset * self._storage.isz
            sub = raw[start:start + n * self._storage.isz]
            if self._storage.dt == 'bf16':
                arr = mx.array(np.frombuffer(sub, np.uint16).copy()).view(mx.bfloat16)
            else:
                arr = mx.array(np.frombuffer(sub, self._storage.dt).copy())
            self._cached = arr.reshape(tuple(self._size)) if self._size else arr
        return self._cached


def _load_pth_lazy(path) -> dict:
    """dict[str, _LazyTensor] -- дёшево (только pickle-метаданные, без
    чтения содержимого storage-файлов из zip)."""
    zf = zipfile.ZipFile(path)
    prefix = zf.namelist()[0].split('/')[0]

    def rebuild(storage, storage_offset, size, stride, *a):
        return _LazyTensor(zf, prefix, storage, storage_offset, size)

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name in _DT:
                return ('STOR',) + _DT[name]
            if mod == 'torch._utils' and name in ('_rebuild_tensor_v2', '_rebuild_tensor'):
                return rebuild
            if mod == 'collections' and name == 'OrderedDict':
                return OrderedDict
            if mod == 'torch' and name == 'Size':
                return tuple
            try:
                return super().find_class(mod, name)
            except Exception:
                return lambda *a, **k: None
        def persistent_load(self, pid):
            assert pid[0] == 'storage', pid[0]
            _, dt, isz = pid[1]
            return _Stor(dt, isz, str(pid[2]), pid[4])

    return U(io.BytesIO(zf.read(f"{prefix}/data.pkl"))).load()


def load_pth(path):
    lazy = _load_pth_lazy(path)
    return {k: v.materialize() for k, v in lazy.items()}


def convert(z, n_layer, H, S):
    """официальные имена x070 -> наши (rwkv7_x070). bf16 сохраняется."""
    T = lambda a: a.T  # транспонирование low-rank
    out = {
        'emb.weight': z['emb.weight'],
        'ln0.weight': z['blocks.0.ln0.weight'], 'ln0.bias': z['blocks.0.ln0.bias'],
        'ln_out.weight': z['ln_out.weight'], 'ln_out.bias': z['ln_out.bias'],
        'head.weight': z['head.weight'],
    }
    for i in range(n_layer):
        b = f'blocks.{i}.'; att = b + 'att.'; ffn = b + 'ffn.'; P = b + 'tmix.'
        out[b + 'ln1.weight'] = z[b + 'ln1.weight']; out[b + 'ln1.bias'] = z[b + 'ln1.bias']
        out[b + 'ln2.weight'] = z[b + 'ln2.weight']; out[b + 'ln2.bias'] = z[b + 'ln2.bias']
        for x in ('x_r', 'x_w', 'x_k', 'x_v', 'x_a', 'x_g'):
            out[P + x] = z[att + x]
        out[P + 'k_k'] = z[att + 'k_k'].reshape(H, S)
        out[P + 'k_a'] = z[att + 'k_a'].reshape(H, S)
        out[P + 'r_k'] = z[att + 'r_k'].reshape(H, S)
        out[P + 'r_proj.weight'] = z[att + 'receptance.weight']
        out[P + 'k_proj.weight'] = z[att + 'key.weight']
        out[P + 'v_proj.weight'] = z[att + 'value.weight']
        out[P + 'o_proj.weight'] = z[att + 'output.weight']
        out[P + 'w_lora_A.weight'] = T(z[att + 'w1']); out[P + 'w_lora_B.weight'] = T(z[att + 'w2'])
        out[P + 'w_lora_B.bias'] = z[att + 'w0'].reshape(-1)
        out[P + 'a_lora_A.weight'] = T(z[att + 'a1']); out[P + 'a_lora_B.weight'] = T(z[att + 'a2'])
        out[P + 'a_lora_B.bias'] = z[att + 'a0'].reshape(-1)
        out[P + 'g_lora_A.weight'] = T(z[att + 'g1']); out[P + 'g_lora_B.weight'] = T(z[att + 'g2'])
        if i > 0:
            out[P + 'v_lora_A.weight'] = T(z[att + 'v1']); out[P + 'v_lora_B.weight'] = T(z[att + 'v2'])
            out[P + 'v_lora_B.bias'] = z[att + 'v0'].reshape(-1)
        out[P + 'ln_x.weight'] = z[att + 'ln_x.weight']; out[P + 'ln_x.bias'] = z[att + 'ln_x.bias']
        out[b + 'cmix.x_k'] = z[ffn + 'x_k']
        out[b + 'cmix.key.weight'] = z[ffn + 'key.weight']
        out[b + 'cmix.value.weight'] = z[ffn + 'value.weight']
    return out


def load_pretrained(pth_path, config=None, verbose=True):
    """Load official RWKV-7 x070 weights (.pth) into an RWKV7X070 model.

    torch-free: reads the torch zip/pickle directly into MLX arrays (bf16 kept).
    Architecture (n_layer/D/H/S/vocab) is inferred from the checkpoint; pass an
    explicit `config` to override.

    Args:
        pth_path: path to the official BlinkDL/rwkv-7-world .pth file.
        config:   optional PretrainConfig; if None, inferred from the weights.
        verbose:  print conversion diagnostics.

    Returns:
        (model, config). model is an RWKV7X070 with weights loaded; config is the
        PretrainConfig used to build it. Returns (None, config) if conversion is
        not clean (missing/extra/mismatched keys) so problems fail loudly.
    """
    import os
    from mlx.utils import tree_flatten, tree_unflatten
    from ..pretrain.config import PretrainConfig
    from .rwkv7_x070 import RWKV7X070

    pth_path = os.path.expanduser(pth_path)
    z = load_pth(pth_path)
    n_layer = 1 + max(int(k.split('.')[1]) for k in z if k.startswith('blocks.'))
    V, D = z['emb.weight'].shape
    H, S = z['blocks.0.att.r_k'].shape
    if verbose:
        print(f"checkpoint config: n_layer={n_layer} D={D} H={H} S={S} vocab={V}")

    if config is None:
        config = PretrainConfig(
            n_layer=n_layer, n_embd=D, vocab_size=V, head_size=S,
            ctx_len=512, batch_size=1,
            train_data="", val_data="", max_steps=1,
        )

    m = RWKV7X070(config)
    conv = convert(z, n_layer, H, S)

    model_keys = set(k for k, _ in tree_flatten(m.parameters()))
    conv_keys = set(conv.keys())
    miss = model_keys - conv_keys
    extra = conv_keys - model_keys
    md = dict(tree_flatten(m.parameters()))
    bad = [(k, tuple(conv[k].shape), tuple(md[k].shape))
           for k in (model_keys & conv_keys)
           if tuple(conv[k].shape) != tuple(md[k].shape)]

    if verbose:
        print(f"model keys: {len(model_keys)}, converter keys: {len(conv_keys)}")
        print(f"  missing in converter: {len(miss)} {sorted(miss)[:5]}")
        print(f"  extra in converter:   {len(extra)} {sorted(extra)[:5]}")
        print(f"  shape mismatches:     {len(bad)} {bad[:5]}")

    if miss or extra or bad:
        print("!!! conversion is NOT clean - aborting")
        return None, config

    m.update(tree_unflatten(list(conv.items())))
    mx.eval(m.parameters())
    return m, config


def load_pretrained_partial(pth_path, skip_official_keys, config=None, verbose=True, pre_materialize_hook=None):
    """Как load_pretrained(), но НЕ читает+декодирует байты storage'ей для
    тензоров, чей ОФИЦИАЛЬНЫЙ ключ (att.receptance/key/value/output.weight,
    ffn.key/value.weight, head.weight -- см. convert()) есть в
    skip_official_keys. Для них подставляется дешёвый mx.zeros-заглушка
    (сразу же безусловно заменяется вызывающим кодом -- напр.
    rwkv_metal.lora.add_lora_rwkvq -- на RwkvqLinear поверх сжатого
    сайдкара; заглушка никогда не используется в forward).

    Мотивация: load_pretrained грузит ВЕСЬ .pth (пик RSS ~2.3x размера
    файла из-за zip-decompress, замерено -- 3GB файл -> 6.9GB пик) даже
    если 60-70% этих тензоров тут же выбрасываются под QLoRA-замену.
    Здесь этот пик платится только за тензоры, которые ДЕЙСТВИТЕЛЬНО
    остаются в bf16 (emb, LayerNorm, w/a/v/g_lora, токен-шифты).
    """
    import os
    from mlx.utils import tree_flatten, tree_unflatten
    from ..pretrain.config import PretrainConfig
    from .rwkv7_x070 import RWKV7X070

    pth_path = os.path.expanduser(pth_path)
    z_lazy = _load_pth_lazy(pth_path)  # дёшево: только pickle-метаданные
    n_layer = 1 + max(int(k.split('.')[1]) for k in z_lazy if k.startswith('blocks.'))
    V, D = z_lazy['emb.weight'].shape
    H, S = z_lazy['blocks.0.att.r_k'].shape
    if verbose:
        print(f"checkpoint config: n_layer={n_layer} D={D} H={H} S={S} vocab={V}")

    if config is None:
        config = PretrainConfig(
            n_layer=n_layer, n_embd=D, vocab_size=V, head_size=S,
            ctx_len=512, batch_size=1,
            train_data="", val_data="", max_steps=1,
        )

    m = RWKV7X070(config)
    conv_lazy = convert(z_lazy, n_layer, H, S)  # дешёвый rename, БЕЗ чтения байт

    # Валидация ключей -- ДО хука: хук меняет структуру дерева параметров
    # (proj/cmix/head -> LoRALinear/RwkvqLinear, другие leaf-имена), сверять
    # нужно с ОРИГИНАЛЬНОЙ x070-архитектурой, иначе miss/extra ложно сработают.
    model_keys = set(k for k, _ in tree_flatten(m.parameters()))
    conv_keys = set(conv_lazy.keys())
    miss = model_keys - conv_keys
    extra = conv_keys - model_keys
    if verbose:
        print(f"model keys: {len(model_keys)}, converter keys: {len(conv_keys)}")
        print(f"  missing in converter: {len(miss)} {sorted(miss)[:5]}")
        print(f"  extra in converter:   {len(extra)} {sorted(extra)[:5]}")
    if miss or extra:
        print("!!! conversion is NOT clean - aborting")
        return None, config

    if pre_materialize_hook is not None:
        # Заменить skip-подмодули (случайная bf16-инициализация из
        # RWKV7X070.__init__, полный размер) на RwkvqLinear СРАЗУ --
        # до того, как ниже начнётся материализация РЕАЛЬНЫХ bf16-данных
        # для остальных 650 тензоров. Иначе случайные заглушки и реальные
        # данные временно сосуществуют в памяти одновременно (замерено:
        # это дало peak ХУЖЕ, чем вообще без ленивой загрузки).
        pre_materialize_hook(m)

    skip_official_keys = set(skip_official_keys)
    n_skipped = n_materialized = 0
    conv = {}
    for key, val in conv_lazy.items():
        official = _internal_to_official(key, n_layer)
        if official in skip_official_keys:
            n_skipped += 1
            continue  # НЕ материализуем -- модуль будет заменён целиком
        conv[key] = val.materialize() if hasattr(val, "materialize") else val
        n_materialized += 1

    bad = [(k, tuple(conv[k].shape), tuple(dict(tree_flatten(m.parameters()))[k].shape))
           for k in conv if tuple(conv[k].shape) != tuple(dict(tree_flatten(m.parameters()))[k].shape)]
    if bad:
        print(f"!!! shape mismatches: {bad[:5]} - aborting")
        return None, config

    if verbose:
        print(f"materialized {n_materialized} tensors, skipped (lazy) {n_skipped}")

    m.update(tree_unflatten(list(conv.items())))
    mx.eval(m.parameters())
    return m, config


def _internal_to_official(key: str, n_layer: int) -> str:
    """Обратная карта convert(): внутреннее имя (tmix.k_proj.weight и т.п.)
    -> официальное (blocks.N.att.key.weight) -- ТОЛЬКО для ключей, которые
    convert() копирует БЕЗ трансформации (проекции/cmix/head; для w/a/v/g
    lora и k_k/k_a/r_k есть .T/.reshape, они никогда не в skip-наборе,
    сюда не попадают)."""
    if key == "head.weight":
        return "head.weight"
    parts = key.split(".")
    if len(parts) >= 4 and parts[0] == "blocks":
        i, rest = parts[1], ".".join(parts[2:])
        m = {
            "tmix.r_proj.weight": f"blocks.{i}.att.receptance.weight",
            "tmix.k_proj.weight": f"blocks.{i}.att.key.weight",
            "tmix.v_proj.weight": f"blocks.{i}.att.value.weight",
            "tmix.o_proj.weight": f"blocks.{i}.att.output.weight",
            "cmix.key.weight": f"blocks.{i}.ffn.key.weight",
            "cmix.value.weight": f"blocks.{i}.ffn.value.weight",
        }
        return m.get(rest, "")
    return ""


def save_converted(pth_path, out_path):
    """Convert an official .pth and save as an MLX safetensors file."""
    from mlx.utils import tree_flatten
    m, _ = load_pretrained(pth_path)
    if m is None:
        return False
    mx.save_safetensors(out_path, dict(tree_flatten(m.parameters())))
    print("saved ->", out_path)
    return True
