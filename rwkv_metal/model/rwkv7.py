import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import checkpoint as _nn_checkpoint
from ..kernel.wkv7 import wkv7


def l2_norm(x):
    # sqrt(sum(x^2) + eps) безопасен в backward в отличие от max(norm, eps)
    return x / mx.sqrt((x * x).sum(axis=-1, keepdims=True) + 1e-12)


class RWKV_Tmix_x070(nn.Module):
    def __init__(self, config, layer_id: int):
        super().__init__()
        D  = config.n_embd
        H  = config.n_head
        S  = config.head_size
        self.H, self.S = H, S
        self.layer_id  = layer_id

        # Token shift lerp
        self.x_r = mx.zeros((1, 1, D))
        self.x_w = mx.zeros((1, 1, D))
        self.x_k = mx.zeros((1, 1, D))
        self.x_v = mx.zeros((1, 1, D))
        self.x_a = mx.zeros((1, 1, D))
        self.x_g = mx.zeros((1, 1, D))

        # Per-head scale параметры
        self.k_k = mx.ones((H, S))   # ключ нормировки (ones чтобы l2_norm не получала zeros)
        self.k_a = mx.zeros((H, S))  # смешивание ключа с iclr
        self.r_k = mx.zeros((H, S))  # бонусный член на выходе

        # Low-rank decay
        self.w_lora_A = nn.Linear(D, 64,  bias=False)
        self.w_lora_B = nn.Linear(64, D,  bias=False)

        # ICLR (in-context learning rate)
        self.a_lora_A = nn.Linear(D, 64,  bias=False)
        self.a_lora_B = nn.Linear(64, D,  bias=True)

        # Value first смешивание (для слоёв > 0)
        if layer_id > 0:
            self.v_lora_A = nn.Linear(D, 64,  bias=False)
            self.v_lora_B = nn.Linear(64, D,  bias=True)

        # Gate
        self.g_lora_A  = nn.Linear(D, 64,  bias=False)
        self.g_lora_B  = nn.Linear(64, D,  bias=False)

        # Проекции
        self.r_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)

        # Нормализация выхода
        self.ln_x = nn.LayerNorm(D)

    def __call__(self, x, x_prev, v_first):
        B, T, D = x.shape
        H, S = self.H, self.S

        # Token shift
        xx = mx.concatenate([x_prev, x[:, :-1]], axis=1) - x
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        # Проекции
        r = self.r_proj(xr).reshape(B, T, H, S)
        k = self.k_proj(xk).reshape(B, T, H, S)
        v = self.v_proj(xv).reshape(B, T, H, S)

        # Gate через low-rank
        gate = mx.sigmoid(self.g_lora_B(nn.gelu(self.g_lora_A(xg))))

        # V-first: смешиваем value с первым слоём
        if self.layer_id == 0:
            v_first = v
        else:
            vv = mx.sigmoid(self.v_lora_B(self.v_lora_A(xv))).reshape(B, T, H, S)
            v  = v + (v_first - v) * vv

        # ICLR через low-rank + sigmoid
        iclr = mx.sigmoid(
            self.a_lora_B(nn.tanh(self.a_lora_A(xa)))
        ).reshape(B, T, H, S)

        # Decay: sigmoid → exp(-0.606531 * sigmoid)
        w = mx.sigmoid(
            self.w_lora_B(nn.tanh(self.w_lora_A(xw)))
        ).reshape(B, T, H, S).astype(mx.float32)
        w = mx.exp(-0.606531 * w).astype(x.dtype)

        # kk = l2_norm(k * k_k) — нормированный ключ
        kk = l2_norm(k * self.k_k)

        # Модифицированный ключ: k * (1 + (iclr - 1) * k_a)
        k = k * (1.0 + (iclr - 1.0) * self.k_a)

        # a = -kk, b = kk * iclr  (дельта-правило)
        a = -kk
        b = kk * iclr

        # WKV-7
        out, _ = wkv7(r, w, k, v, a, b, training=True)  # [B, T, H, S]

        # Бонусный член: прямое взаимодействие r, k, v
        bonus = (r * k * self.r_k).sum(axis=-1, keepdims=True) * v
        out   = (out + bonus).reshape(B, T, D)

        out = self.ln_x(out)
        return self.o_proj(out * gate), v_first


class RWKV_CMix_x070(nn.Module):
    def __init__(self, config):
        super().__init__()
        D = config.n_embd
        self.x_k   = mx.zeros((1, 1, D))
        self.key   = nn.Linear(D, D * 4, bias=False)
        self.value = nn.Linear(D * 4, D, bias=False)

    def __call__(self, x, x_prev):
        xx = mx.concatenate([x_prev, x[:, :-1]], axis=1) - x
        xk = x + xx * self.x_k
        return self.value(nn.relu(self.key(xk)) ** 2)



def init_weights(model):
    """
    Инициализация весов RWKV-7 по правилам Bo Peng.
    Без этого NaN на первом шаге гарантирован.
    """
    import math
    n_layer = model.config.n_layer
    n_embd  = model.config.n_embd

    for i, block in enumerate(model.blocks):
        tmix = block.tmix

        # LoRA B матрицы → нули
        # Это делает все динамические параметры нейтральными на старте
        tmix.w_lora_B.weight = mx.zeros_like(tmix.w_lora_B.weight)
        tmix.a_lora_B.weight = mx.zeros_like(tmix.a_lora_B.weight)
        tmix.g_lora_B.weight = mx.zeros_like(tmix.g_lora_B.weight)
        if hasattr(tmix, 'v_lora_B'):
            tmix.v_lora_B.weight = mx.zeros_like(tmix.v_lora_B.weight)

        # k_proj: демпфирование для стабильности WKV
        scale = 0.1
        tmix.k_proj.weight = tmix.k_proj.weight * scale

        # r_proj и v_proj: масштаб по глубине сети
        depth_scale = 1.0 / math.sqrt(n_layer)
        tmix.r_proj.weight = tmix.r_proj.weight * depth_scale
        tmix.v_proj.weight = tmix.v_proj.weight * depth_scale

    # Выходная голова: малый масштаб
    vocab_scale = 1.0 / math.sqrt(n_embd)
    model.head.weight = model.head.weight * vocab_scale

    mx.eval(model.parameters())
    return model

class RWKVBlock(nn.Module):
    def __init__(self, config, layer_id: int):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.ln2  = nn.LayerNorm(config.n_embd)
        self.tmix = RWKV_Tmix_x070(config, layer_id)
        self.cmix = RWKV_CMix_x070(config)

    def __call__(self, x, x_prev, v_first):
        h, v_first = self.tmix(self.ln1(x), x_prev, v_first)
        x = x + h
        x = x + self.cmix(self.ln2(x), x_prev)
        return x, v_first


class RWKV7(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config    = config
        self._train    = True
        self._grad_ckpt = False  # gradient checkpointing по блокам
        self.emb     = nn.Embedding(config.vocab_size, config.n_embd)
        self.ln0     = nn.LayerNorm(config.n_embd)
        self.blocks  = [RWKVBlock(config, i) for i in range(config.n_layer)]
        self.ln_out  = nn.LayerNorm(config.n_embd)
        self.head    = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def set_dtype(self, dtype):
        """Convert all model parameters to the given dtype.

        dtype: "bfloat16" | "float32" | mx.Dtype.
            bf16  -> 2x smaller weights/optimizer-state, ~+10% speed.
            fp32  -> max precision (default of MLX init).
        Critical reductions (loss/cross-entropy, WKV recurrence) stay in fp32
        internally regardless of this setting, so bf16 is mixed-precision.
        """
        from mlx.utils import tree_map
        if isinstance(dtype, str):
            dtype = {"bfloat16": mx.bfloat16, "bf16": mx.bfloat16,
                     "float32": mx.float32, "fp32": mx.float32}[dtype]
        self.update(tree_map(
            lambda x: x.astype(dtype) if isinstance(x, mx.array) else x,
            self.parameters(),
        ))
        mx.eval(self.parameters())
        return self

    def body(self, idx):
        """Run everything except the lm head; returns hidden states [B, T, D].

        Useful when you want hidden states without the (large) vocab projection,
        e.g. custom loss heads or feature extraction.
        """
        B, T    = idx.shape
        x       = self.ln0(self.emb(idx))
        # x_prev must match x.dtype, otherwise silent fp32/bf16 cast in token-shift
        x_prev  = mx.zeros((B, 1, self.config.n_embd), dtype=x.dtype)
        v_first = None
        for block in self.blocks:
            if self._grad_ckpt:
                x, v_first = _nn_checkpoint(block)(x, x_prev, v_first)
            else:
                x, v_first = block(x, x_prev, v_first)
            x_prev     = x[:, -1:]
        return self.ln_out(x)

    def __call__(self, idx):
        return self.head(self.body(idx))

    def loss(self, idx, targets):
        logits  = self(idx)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T)
        ).mean()
