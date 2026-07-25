"""
RWKV-7 "Goose" x070 — архитектурно ТОЧНАЯ версия для загрузки официальных весов.

Отличия от model/rwkv7.py (эталонная from-scratch версия) — приведены к официальному
x070 (BlinkDL/RWKV-LM RWKV-v7, сверено с ChatRWKV rwkv_pip RWKV_x070_TMix):
  1. decay w: + w0 (bias w_lora_B) внутри sigmoid
  2. iclr a:  без tanh  → sigmoid(a0 + a_lora_B(a_lora_A(xa)))
  3. gate g:  sigmoid ВНУТРИ, линейно наружу → g_lora_B(sigmoid(g_lora_A(xg)))
  4. ln_x:    GroupNorm по головам (num_groups=H, eps=64e-5), не LayerNorm
  5. порядок: WKV → ln_x(GroupNorm) → +bonus → *g  (а не +bonus → ln_x)
  6. token-shift: нулевой паддинг на t=0 в каждом блоке, БЕЗ межблочного переноса
                  (cmix шифтует свой вход)
  + low-rank ranks вычисляются из D по официальной формуле (можно переопределить).

WKV-ядро (model.wkv7.wkv7) и cmix (squared-relu FFN) совпадают с официальными.
lora.py работает без изменений (tmix.{r,k,v,o}_proj — те же имена).
"""

import math
import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import checkpoint as _nn_checkpoint
from ..kernel.wkv7 import wkv7, wkv7_step
from .state import RWKVState, gather_last


def l2_norm(x):
    return x / mx.sqrt((x * x).sum(axis=-1, keepdims=True) + 1e-12)


def lora_ranks(D: int) -> dict:
    """Официальная формула x070 для low-rank размерностей."""
    f = lambda c, p: max(32, int(round((c * (D ** p)) / 32) * 32))
    return {"w": f(1.8, 0.5), "a": f(1.8, 0.5), "v": f(1.3, 0.5), "g": f(0.6, 0.8)}


def _token_shift(x, x_prev=None):
    # prev[t] = x[t-1]; prev[0] = x_prev или 0 (обучение)
    pad = mx.zeros_like(x[:, :1]) if x_prev is None else x_prev
    return mx.concatenate([pad, x[:, :-1]], axis=1)


class RWKV_Tmix_x070(nn.Module):
    def __init__(self, config, layer_id: int, ranks: dict = None):
        super().__init__()
        D = config.n_embd
        H = config.n_head
        S = config.head_size
        self.H, self.S = H, S
        self.layer_id = layer_id
        r = ranks or lora_ranks(D)

        # Token-shift lerp
        self.x_r = mx.zeros((1, 1, D))
        self.x_w = mx.zeros((1, 1, D))
        self.x_k = mx.zeros((1, 1, D))
        self.x_v = mx.zeros((1, 1, D))
        self.x_a = mx.zeros((1, 1, D))
        self.x_g = mx.zeros((1, 1, D))

        # Per-head параметры (хранятся как (H,S); официал (D,) → reshape при конвертации)
        self.k_k = mx.ones((H, S))
        self.k_a = mx.zeros((H, S))
        self.r_k = mx.zeros((H, S))

        # Low-rank: decay (w0 = bias w_lora_B)
        self.w_lora_A = nn.Linear(D, r["w"], bias=False)
        self.w_lora_B = nn.Linear(r["w"], D, bias=True)
        # iclr (a0 = bias a_lora_B), БЕЗ tanh в forward
        self.a_lora_A = nn.Linear(D, r["a"], bias=False)
        self.a_lora_B = nn.Linear(r["a"], D, bias=True)
        # value-residual (слои >0), v0 = bias v_lora_B
        if layer_id > 0:
            self.v_lora_A = nn.Linear(D, r["v"], bias=False)
            self.v_lora_B = nn.Linear(r["v"], D, bias=True)
        # gate: sigmoid внутри, без bias (нет g0)
        self.g_lora_A = nn.Linear(D, r["g"], bias=False)
        self.g_lora_B = nn.Linear(r["g"], D, bias=False)

        # Проекции
        self.r_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)

        # GroupNorm по головам (как официальный F.group_norm(num_groups=H, eps=64e-5))
        self.ln_x = nn.GroupNorm(H, D, eps=64e-5, affine=True, pytorch_compatible=True)

    def __call__(self, x, v_first, x_prev=None, h_in=None, mask=None,
                 return_state=False):
        """mask: [B, T] (1 — реальный токен, 0 — паддинг) либо None.

        Пад-позиции делаются НЕЙТРАЛЬНЫМИ для рекуррентности: w←1, k←0, b←0,
        откуда h_next = 1·h + v·0ᵀ + sa·0ᵀ = h. Это точное решение проблемы
        right-padding'а: состояние строки замирает на её последнем реальном
        токене и не зависит ни от числа пад-токенов, ни от соседей по батчу.
        (Оригинальный EmbeddingRWKV вместо этого паддит слева и мирится с тем,
        что пад-токены всё же проходят через рекуррентность.)

        Выход на пад-позициях остаётся мусорным, но модель каузальна: он может
        попасть только на пад-позиции следующих слоёв, до реальных токенов не
        доходит.
        """
        B, T, D = x.shape
        H, S = self.H, self.S

        xx = _token_shift(x, x_prev) - x
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.r_proj(xr).reshape(B, T, H, S)
        k = self.k_proj(xk).reshape(B, T, H, S)
        v = self.v_proj(xv).reshape(B, T, H, S)

        # gate: sigmoid ВНУТРИ, линейно наружу
        g = self.g_lora_B(mx.sigmoid(self.g_lora_A(xg)))

        # a (iclr): БЕЗ tanh, sigmoid(a0 + B(A(xa)))
        a = mx.sigmoid(self.a_lora_B(self.a_lora_A(xa))).reshape(B, T, H, S)

        # decay: exp(-exp(-0.5) * sigmoid(w0 + B(tanh(A(xw)))))
        w = self.w_lora_B(mx.tanh(self.w_lora_A(xw)))
        w = mx.exp(-0.606531 * mx.sigmoid(w.astype(mx.float32))).astype(x.dtype)
        w = w.reshape(B, T, H, S)

        kk = l2_norm(k * self.k_k)
        k = k * (1.0 + (a - 1.0) * self.k_a)

        if self.layer_id == 0:
            v_first = v
        else:
            vv = mx.sigmoid(self.v_lora_B(self.v_lora_A(xv))).reshape(B, T, H, S)
            v = v + (v_first - v) * vv

        # WKV-7: a_kernel = -kk, b_kernel = kk * a
        k_wkv, b_wkv = k, kk * a
        if mask is not None:
            m = mask.reshape(B, T, 1, 1).astype(w.dtype)
            w = w * m + (1.0 - m)   # w=1 на паддинге → состояние не затухает
            k_wkv = k_wkv * m       # обнуляет v·kᵀ
            b_wkv = b_wkv * m       # обнуляет sa·bᵀ

        if T == 1:
            # Checkpoint-ядро требует T кратного 16, то есть один токен стоил
            # бы 16 шагов. Разворачиваем шаг в обычные MLX-операции: та же
            # математика, autograd бесплатно, в 16 раз меньше работы. Это путь
            # реранкера (один токен-зонд поверх состояния) и пошагового
            # инференса.
            if h_in is None:
                h_in = mx.zeros((B, H, S, S), dtype=mx.float32)
            out, h_out = wkv7_step(r, w, k_wkv, v, -kk, b_wkv, h_in)
        elif return_state or h_in is not None:
            out, h_out = wkv7(r, w, k_wkv, v, -kk, b_wkv, training=True,
                              state=h_in, return_state=True)
        else:
            out, h_out = wkv7(r, w, k_wkv, v, -kk, b_wkv, training=True)

        # Порядок официала: ln_x (GroupNorm) ДО bonus
        # ln_x per-token: канон RWKV-7 — F.group_norm(x.view(B*T, C), H).
        # reshape(B, T, D) подал бы [B,T,D] в MLX GroupNorm, и та усреднила бы
        # по ВСЕМ T внутри головы (утечка будущего при teacher-forcing).
        out = self.ln_x(out.reshape(B * T, D)).reshape(B, T, H, S)
        bonus = (r * k * self.r_k).sum(axis=-1, keepdims=True) * v
        out = (out + bonus).reshape(B, T, D)

        y = self.o_proj(out * g)
        if return_state:
            return y, v_first, h_out
        return y, v_first


class RWKV_CMix_x070(nn.Module):
    def __init__(self, config):
        super().__init__()
        D = config.n_embd
        self.x_k = mx.zeros((1, 1, D))
        self.key = nn.Linear(D, D * 4, bias=False)
        self.value = nn.Linear(D * 4, D, bias=False)

    def __call__(self, x, x_prev=None):
        xx = _token_shift(x, x_prev) - x   # шифтует СВОЙ вход, нулевой паддинг t=0
        xk = x + xx * self.x_k
        return self.value(nn.relu(self.key(xk)) ** 2)


class RWKVBlock(nn.Module):
    def __init__(self, config, layer_id: int, ranks: dict = None):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.tmix = RWKV_Tmix_x070(config, layer_id, ranks)
        self.cmix = RWKV_CMix_x070(config)

    def __call__(self, x, v_first, h_in=None, mask=None, tmix_prev=None,
                 cmix_prev=None, end_idx=None, return_state=False):
        """Быстрый путь (без состояния) — ровно как раньше: token-shift
        стартует с нуля в каждом блоке, межблочного переноса нет.

        Путь с состоянием возвращает (x, v_first, h_out, tmix_shift, cmix_shift),
        где сдвиги сняты с позиции end_idx (или последней, если end_idx=None) —
        именно они нужны, чтобы продолжение совпало со сплошным проходом.
        """
        x1 = self.ln1(x)
        if return_state:
            h, v_first, h_out = self.tmix(x1, v_first, x_prev=tmix_prev,
                                          h_in=h_in, mask=mask, return_state=True)
        else:
            h, v_first = self.tmix(x1, v_first, x_prev=tmix_prev, h_in=h_in,
                                   mask=mask)
        x = x + h
        x2 = self.ln2(x)
        x = x + self.cmix(x2, x_prev=cmix_prev)
        if return_state:
            return x, v_first, h_out, gather_last(x1, end_idx), gather_last(x2, end_idx)
        return x, v_first


class RWKV7X070(nn.Module):
    def __init__(self, config, ranks: dict = None):
        super().__init__()
        self.config = config
        self._grad_ckpt = False  # per-block gradient checkpointing (set by finetune)
        self.ranks = ranks or lora_ranks(config.n_embd)
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.ln0 = nn.LayerNorm(config.n_embd)
        self.blocks = [RWKVBlock(config, i, self.ranks) for i in range(config.n_layer)]
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def set_dtype(self, dtype):
        """Convert all parameters to the given dtype ("bfloat16"|"float32"|mx.Dtype).

        Sensitive reductions (loss, WKV recurrence) stay fp32 internally, so
        bf16 is mixed-precision. Note: quantized (QLoRA) layers are unaffected.
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
        """Всё кроме lm-головы; возвращает скрытые состояния [B, T, D].

        state:        RWKVState — продолжить с этого состояния (или None).
        mask:         [B, T] — 1 у реальных токенов, 0 у right-паддинга.
                      Без неё пад-токены попадают в рекуррентность и портят
                      финальное состояние (на сами скрытые состояния реальных
                      позиций они, будучи справа, не влияют никогда).
        end_idx:      [B] — позиция последнего реального токена; нужна только
                      при return_state, чтобы снять token-shift там, а не в
                      конце паддинга.
        return_state: вернуть (h, RWKVState) вместо h.

        Путь с состоянием намеренно не оборачивается в gradient checkpointing:
        он существует ради инференса и обучения НАД замороженной базой, где
        пересчёт активаций ничего не экономит.
        """
        x = self.ln0(self.emb(idx))
        v_first = None

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
        """Только состояние на конце последовательности, без скрытых состояний.

        Это вход реранкера: он читает свёрнутую в state историю пары
        (документ, запрос), а не потокенные активации.
        """
        _, st = self.body(idx, state=state, mask=mask, end_idx=end_idx,
                          return_state=True)
        return st

    def __call__(self, idx):
        return self.head(self.body(idx))

    def loss(self, idx, targets):
        logits = self(idx)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V), targets.reshape(B * T)
        ).mean()
