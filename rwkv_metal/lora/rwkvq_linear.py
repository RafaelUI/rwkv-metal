"""
QLoRA-база на РОДНОМ формате rwkv-quant (.rwkvq, gw_mode="sb6"), а не на
стоковом mlx.nn.quantize.

Почему отдельный модуль: .rwkvq требует torch для чтения (rwkv_quant/
formats/reader.py), а rwkv-metal намеренно torch-free в рантайме (см.
model/convert.py). Поэтому конвертация в две стадии:
  1) rwkv_quant.formats.export_mlx (venv rwkv-quant, есть torch) -- один раз
     переупаковывает .rwkvq в *.rwkvq_mlx.safetensors + .json манифест,
     используя K3-интерлив-буферы (qblk/qsqm/ddm) из
     backends/metal/quant_linear_gw.py::GwQuantLinear -- те же буферы,
     что и в проверенном бит-в-бит инференс-пути.
  2) этот модуль (venv rwkv-metal, БЕЗ torch) грузит сайдкар через
     mx.load + json и восстанавливает dense-вес НА ЛЕТУ при каждом
     forward (транзиент, не кешируется -- смысл QLoRA: база должна жить
     в памяти в сжатом виде, а не dense).

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
import mlx.core as mx
import mlx.nn as nn
from .rwkvq_kernel import dequant_dense

_SIDECAR_CACHE = {}


def load_sidecar(path: str):
    """path -- без суффикса .safetensors/.json (как передан в export_mlx)."""
    if path in _SIDECAR_CACHE:
        return _SIDECAR_CACHE[path]
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
        w = self._dequant_w()
        return x @ w.T
