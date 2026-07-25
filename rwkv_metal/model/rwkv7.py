import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import checkpoint as _nn_checkpoint
from ..kernel.wkv7 import wkv7, wkv7_step
from .state import RWKVState, gather_last


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

    def __call__(self, x, x_prev, v_first, h_in=None, mask=None,
                 return_state=False):
        """x_prev: [B, 1, D] — «предыдущий» токен для позиции 0, или None
        (нулевой паддинг). Смотри `RWKV7.body` про то, почему None здесь
        теперь норма, а не заглушка."""
        B, T, D = x.shape
        H, S = self.H, self.S

        # Token shift
        if x_prev is None:
            x_prev = mx.zeros_like(x[:, :1])
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

        # Маска паддинга: пад-позиции делаются no-op для рекуррентности
        # (w=1, k=0, b=0 → h_next = h). См. RWKV_Tmix_x070 в rwkv7_x070.py.
        k_wkv, b_wkv = k, b
        if mask is not None:
            m = mask.reshape(B, T, 1, 1).astype(w.dtype)
            w = w * m + (1.0 - m)
            k_wkv = k_wkv * m
            b_wkv = b_wkv * m

        # WKV-7
        if T == 1:
            if h_in is None:
                h_in = mx.zeros((B, H, S, S), dtype=mx.float32)
            out, h_out = wkv7_step(r, w, k_wkv, v, a, b_wkv, h_in)
        elif return_state or h_in is not None:
            out, h_out = wkv7(r, w, k_wkv, v, a, b_wkv, training=True,
                              state=h_in, return_state=True)
        else:
            out, h_out = wkv7(r, w, k_wkv, v, a, b_wkv, training=True)

        # Бонусный член: прямое взаимодействие r, k, v
        bonus = (r * k * self.r_k).sum(axis=-1, keepdims=True) * v
        out   = (out + bonus).reshape(B, T, D)

        out = self.ln_x(out)
        y = self.o_proj(out * gate)
        if return_state:
            return y, v_first, h_out
        return y, v_first


class RWKV_CMix_x070(nn.Module):
    def __init__(self, config):
        super().__init__()
        D = config.n_embd
        self.x_k   = mx.zeros((1, 1, D))
        self.key   = nn.Linear(D, D * 4, bias=False)
        self.value = nn.Linear(D * 4, D, bias=False)

    def __call__(self, x, x_prev=None):
        if x_prev is None:
            x_prev = mx.zeros_like(x[:, :1])
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

    def __call__(self, x, v_first, h_in=None, mask=None, tmix_prev=None,
                 cmix_prev=None, end_idx=None, return_state=False):
        """Каждый микс шифтует СВОЙ вход с нулевым паддингом на позиции 0.

        Раньше здесь был один общий `x_prev`, приходивший снаружи как выход
        ПРЕДЫДУЩЕГО блока на последней позиции чанка, — то есть будущее
        относительно позиции 0. См. `RWKV7.body`.
        """
        x1 = self.ln1(x)
        if return_state:
            h, v_first, h_out = self.tmix(x1, tmix_prev, v_first, h_in=h_in,
                                          mask=mask, return_state=True)
        else:
            h, v_first = self.tmix(x1, tmix_prev, v_first, h_in=h_in, mask=mask)
        x = x + h
        x2 = self.ln2(x)
        x = x + self.cmix(x2, cmix_prev)
        if return_state:
            return x, v_first, h_out, gather_last(x1, end_idx), gather_last(x2, end_idx)
        return x, v_first

    def _legacy_call(self, x, x_prev, v_first):
        """Прежнее поведение с межблочным переносом token-shift.

        Оставлено только чтобы можно было воспроизвести чекпоинты, обученные
        до исправления. Для нового обучения не используй: `x_prev` здесь —
        последний токен чанка из предыдущего блока, то есть будущее для
        позиции 0.
        """
        h, v_first = self.tmix(self.ln1(x), x_prev, v_first)
        x = x + h
        x = x + self.cmix(self.ln2(x), x_prev)
        return x, v_first


class RWKV7(nn.Module):
    def __init__(self, config, legacy_token_shift: bool = False):
        super().__init__()
        self.config    = config
        self._train    = True
        self._grad_ckpt = False  # gradient checkpointing по блокам
        # см. RWKV7.body — прежний межблочный перенос token-shift подглядывал
        # в будущее; True только для воспроизведения старых чекпоинтов
        self.legacy_token_shift = legacy_token_shift
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

    def body(self, idx, state=None, mask=None, end_idx=None, return_state=False):
        """Run everything except the lm head; returns hidden states [B, T, D].

        Сигнатура и семантика — те же, что у `RWKV7X070.body`: `state`
        (RWKVState) продолжает последовательность, `mask` помечает реальные
        токены right-padded батча, `end_idx` говорит, где снимать token-shift.

        Про legacy_token_shift
        ----------------------
        До этого здесь был межблочный перенос: `x_prev = x[:, -1:]` после
        каждого блока, и блок i+1 получал в качестве «предыдущего токена»
        ПОСЛЕДНИЙ токен чанка — будущее относительно позиции 0. Замер:
        смена только последнего токена входа меняла скрытые состояния на
        позициях 0..T-2 на 0.30 (у x070 — ровно 0.0). То есть при
        teacher-forcing модель на первой позиции каждого слоя подглядывала
        в конец окна, а на инференсе такого токена нет — расхождение между
        обучением и применением, плюс невозможность определить продолжение
        с состояния.

        Теперь каждый микс шифтует свой вход с нулевым паддингом, как в
        официальной x070. `legacy_token_shift=True` возвращает прежнее
        поведение — только чтобы воспроизвести уже обученные этим кодом
        чекпоинты, для нового обучения смысла нет.
        """
        B, T = idx.shape
        x = self.ln0(self.emb(idx))
        v_first = None

        if self.legacy_token_shift:
            if state is not None or return_state or mask is not None:
                raise NotImplementedError(
                    "legacy_token_shift=True несовместим с состоянием: "
                    "межблочный перенос token-shift делает продолжение "
                    "принципиально несовпадающим со сплошным проходом."
                )
            x_prev = mx.zeros((B, 1, self.config.n_embd), dtype=x.dtype)
            for block in self.blocks:
                x, v_first = block._legacy_call(x, x_prev, v_first)
                x_prev = x[:, -1:]
            return self.ln_out(x)

        if not return_state and state is None:
            for block in self.blocks:
                if self._grad_ckpt:
                    x, v_first = _nn_checkpoint(block)(x, v_first)
                else:
                    x, v_first = block(x, v_first)
            return self.ln_out(x)

        wkvs, tshifts, cshifts = [], [], []
        for i, block in enumerate(self.blocks):
            h_in = None if state is None else state.wkv[i]
            tprev = None if state is None else state.tmix_shift[i]
            cprev = None if state is None else state.cmix_shift[i]
            if return_state:
                x, v_first, h_out, ts, cs = block(
                    x, v_first, h_in=h_in, mask=mask, tmix_prev=tprev,
                    cmix_prev=cprev, end_idx=end_idx, return_state=True)
                wkvs.append(h_out); tshifts.append(ts); cshifts.append(cs)
            else:
                x, v_first = block(x, v_first, h_in=h_in, mask=mask,
                                   tmix_prev=tprev, cmix_prev=cprev)

        h = self.ln_out(x)
        if return_state:
            return h, RWKVState.stack(wkvs, tshifts, cshifts)
        return h

    def states(self, idx, mask=None, end_idx=None, state=None) -> RWKVState:
        """Только состояние на конце последовательности."""
        _, st = self.body(idx, state=state, mask=mask, end_idx=end_idx,
                          return_state=True)
        return st

    def __call__(self, idx):
        return self.head(self.body(idx))

    def loss(self, idx, targets):
        logits  = self(idx)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T)
        ).mean()
