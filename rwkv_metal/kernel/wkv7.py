"""
wkv7.py — публичный слой WKV-7 ядра (forward / backward / inference)
===================================================================
Единая точка входа к WKV-7 на Metal:

    wkv7(r, w, k, v, a, b, training=True)        → (out, state)
    wkv7_train(r, w, k, v, a, b)                 → out      (обучение, autograd)
    wkv7_infer(r, w, k, v, a, b, state)          → (out, new_state)
    wkv7_train_py(r, w, k, v, a, b)              → out      (Python fallback, отладка)

Все тензоры имеют форму [B, T, H, D], где D == HEAD_SIZE (64).
Обучение использует checkpoint-kernel (один fwd + один bwd вызов на весь T).
"""
import mlx.core as mx

from .wkv7_checkpoint import make_wkv7_checkpoint, make_wkv7_checkpoint_with_state

HEAD_SIZE = 64
CHUNK     = 16

# ─────────────────── Fallback: чистый Python einsum (для отладки) ────────────

def _wkv7_chunk_py(r, w, k, v, a, b, h):
    B, T, H, D = r.shape
    outs = []
    for t in range(T):
        r_t = r[:, t]; w_t = w[:, t]; k_t = k[:, t]
        v_t = v[:, t]; a_t = a[:, t]; b_t = b[:, t]
        sa  = mx.einsum("bhsd,bhd->bhs", h, a_t)
        sab = mx.einsum("bhs,bhd->bhsd", sa, b_t)
        vk  = mx.einsum("bhs,bhd->bhsd", v_t, k_t)
        h   = h * w_t[:, :, None, :] + vk + sab
        y   = mx.einsum("bhsd,bhd->bhs", h, r_t)
        outs.append(y)
    return mx.stack(outs, axis=1), h

def wkv7_train_py_with_state(r, w, k, v, a, b, h_in):
    """Einsum reference with an explicit initial state and returned final state."""
    B, T, H, D = r.shape
    h = h_in
    outs = []
    for start in range(0, T, CHUNK):
        end = min(start + CHUNK, T); cl = end - start
        rc,wc,kc,vc,ac,bc = (x[:,start:end] for x in (r,w,k,v,a,b))
        if cl < CHUNK:
            pad = CHUNK - cl
            def p(x, val=0.0):
                return mx.pad(x,[(0,0),(0,pad),(0,0),(0,0)],constant_values=val)
            rc=p(rc);wc=p(wc,1.0);kc=p(kc);vc=p(vc);ac=p(ac);bc=p(bc)
        out_c, h = _wkv7_chunk_py(rc,wc,kc,vc,ac,bc,h)
        outs.append(out_c[:,:cl])
    return mx.concatenate(outs, axis=1), h


def wkv7_train_py(r, w, k, v, a, b):
    B, _, H, D = r.shape
    out, _ = wkv7_train_py_with_state(
        r, w, k, v, a, b, mx.zeros((B, H, D, D))
    )
    return out

# ─────────────────── Metal training kernels (fwd + bwd) ─────────────────────
#
# Почанковые ядра (_get_fwd / _get_bwd и обёртка _wkv7_chunk_metal) убраны
# 12.10: их никто не вызывал с перехода обучения на checkpoint-ядро, а
# обратное держало 18.7 КБ общей памяти группы (accum[64][64] + девять
# векторов) при пределе Metal 16 КБ -- под валидацией шейдеров такой запуск
# не проходит. Боевые ядра обучения -- в wkv7_checkpoint.py (плитка 16, 6.2 КБ).

# Checkpoint kernel: один fwd + один bwd вызов на весь T
# 1.73× быстрее chunked v2, численно точнее (stable reconstruction per chunk)
_ckpt_cache: dict = {}
_ckpt_state_cache: dict = {}


def _pad_to_chunk(r, w, k, v, a, b):
    """Дополняет T до кратного CHUNK так, чтобы добавленные шаги были no-op.

    w=1, k=v=a=b=0  =>  h_next = 1*h + v*kᵀ + sa*bᵀ = h. Состояние на выходе
    не меняется, поэтому h_out после паддинга равен h_out без него.
    """
    T = r.shape[1]
    if T % CHUNK == 0:
        return r, w, k, v, a, b, T
    pad = CHUNK - (T % CHUNK)

    def p(x, val=0.0):
        return mx.pad(x, [(0, 0), (0, pad), (0, 0), (0, 0)], constant_values=val)

    return p(r), p(w, 1.0), p(k), p(v), p(a), p(b), T + pad


def wkv7_train(r, w, k, v, a, b):
    B, T, H, D = r.shape
    r, w, k, v, a, b, T_pad = _pad_to_chunk(r, w, k, v, a, b)

    key = (B, T_pad, H, D)
    if key not in _ckpt_cache:
        _ckpt_cache[key] = make_wkv7_checkpoint(B, T_pad, H, D)

    out = _ckpt_cache[key](r, w, k, v, a, b)
    return out[:, :T]


def wkv7_train_with_state(r, w, k, v, a, b, h_in=None):
    """Как wkv7_train, но с явным начальным состоянием и возвратом конечного.

    h_in: [B, H, D, D] или None (нулевое состояние).
    Возвращает (out [B,T,H,D], h_out [B,H,D,D]).

    Дифференцируемо по h_in тоже — VJP ядра отдаёт dh_in, что и позволяет
    учить поверх состояния (реранкер) или тюнить само состояние.
    """
    B, T, H, D = r.shape
    if h_in is None:
        h_in = mx.zeros((B, H, D, D), dtype=mx.float32)
    r, w, k, v, a, b, T_pad = _pad_to_chunk(r, w, k, v, a, b)

    key = (B, T_pad, H, D)
    if key not in _ckpt_state_cache:
        _ckpt_state_cache[key] = make_wkv7_checkpoint_with_state(B, T_pad, H, D)

    out, h_out = _ckpt_state_cache[key](r, w, k, v, a, b, h_in)
    return out[:, :T], h_out


def wkv7_step(r, w, k, v, a, b, h_in):
    """Один шаг рекуррентности на чистых MLX-операциях (T == 1).

    Ядро-checkpoint требует T кратного CHUNK=16, то есть один токен стоил бы
    16 шагов. Реранкер прогоняет ровно один обучаемый токен на блок, поэтому
    здесь дешевле и точнее развернуть шаг в обычные операции: autograd MLX
    даёт градиенты и по параметрам, и по h_in без отдельного backward-ядра.

    r..b: [B, 1, H, D] (либо [B, H, D]). h_in: [B, H, D, D] (h[s, d]).
    Возвращает (out [B, 1, H, D], h_out [B, H, D, D]).
    """
    squeeze = (r.ndim == 4)
    if squeeze:
        r, w, k, v, a, b = (x[:, 0] for x in (r, w, k, v, a, b))
    h = h_in.astype(mx.float32)
    r, w, k, v, a, b = (x.astype(mx.float32) for x in (r, w, k, v, a, b))
    # sa[s] = sum_d h[s,d] * a[d]
    sa = (h * a[:, :, None, :]).sum(axis=-1)                      # [B,H,D]
    h = (h * w[:, :, None, :]
         + v[:, :, :, None] * k[:, :, None, :]
         + sa[:, :, :, None] * b[:, :, None, :])
    out = (h * r[:, :, None, :]).sum(axis=-1)                     # [B,H,D]
    if squeeze:
        out = out[:, None]
    return out, h

# ─────────────────── Inference: Metal kernel ────────────────────────────────

_infer_cache = {}

def _get_infer_kernel(H: int, T: int = None):
    # T -- число шагов, зашиваемое в кернель константой (кеш по (H, T)).
    # По умолчанию CHUNK -- прежнее поведение. T=1 для single-token decode
    # убирает CHUNKx лишней работы (раньше недостающие шаги паддились
    # no-op'ами и весь чанк считался целиком).
    if T is None: T = CHUNK
    key = (H, T)
    if key in _infer_cache: return _infer_cache[key]
    hdr = f"""
constant uint HEAD_SIZE_C = {HEAD_SIZE};
constant uint CHUNK_C     = {T};
constant uint H_C         = {H};
"""
    body = r"""
    // dv -- индекс потока ВНУТРИ threadgroup: запуск идёт threadgroup=(1, D),
    // одна группа на пару (b, h), ради стейджинга ниже.
    uint dv   = thread_position_in_threadgroup.y;
    uint bhi  = threadgroup_position_in_grid.x;
    uint bi   = bhi / H_C; uint hi = bhi % H_C;

    // a/w/k/b/r на шаге t одинаковы для всех 64 потоков группы, и без
    // стейджинга каждое значение читалось из глобальной памяти 64 раза
    // (2048 потоков x T x 5 строк x 256 байт = 1.34 ГБ на вызов при 29 МБ
    // полезного трафика). Ровно тот же приём и та же причина, что уже
    // стояли в forward-ядре обучения, -- сюда они не доехали. Порядок
    // суммирования НЕ меняется, выход бит-в-бит прежний
    // (tests/test_wkv_infer_parity.py в rwkv-quant).
    threadgroup float a_sh[HEAD_SIZE_C], w_sh[HEAD_SIZE_C], k_sh[HEAD_SIZE_C];
    threadgroup float b_sh[HEAD_SIZE_C], r_sh[HEAD_SIZE_C];

    float h_row[HEAD_SIZE_C];
    uint h_base = (bi*H_C+hi)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_row[dk] = h_in[h_base+dk];

    for (uint t=0; t<CHUNK_C; t++) {
        uint base = ((bi*CHUNK_C+t)*H_C+hi)*HEAD_SIZE_C;
        a_sh[dv]=a[base+dv]; w_sh[dv]=w[base+dv]; k_sh[dv]=k[base+dv];
        b_sh[dv]=b[base+dv]; r_sh[dv]=r[base+dv];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float sa = 0.0f;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) sa += h_row[dk]*a_sh[dk];
        float v_dv = v[base+dv];
        for (uint dk=0; dk<HEAD_SIZE_C; dk++)
            h_row[dk] = w_sh[dk]*h_row[dk] + v_dv*k_sh[dk] + sa*b_sh[dk];
        float y = 0.0f;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) y += h_row[dk]*r_sh[dk];
        out[base+dv] = y;

        // Следующий шаг перезапишет *_sh, пока кто-то ещё читает текущие.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_out[h_base+dk] = h_row[dk];
"""
    kern = mx.fast.metal_kernel(
        name=f"wkv7_infer_{H}_{T}",
        input_names=["r","w","k","v","a","b","h_in"],
        output_names=["out","h_out"],
        header=hdr, source=body,
    )
    _infer_cache[key] = kern
    return kern

def wkv7_infer(r, w, k, v, a, b, h):
    B, T, H, D = r.shape
    assert D == HEAD_SIZE, f"HEAD_SIZE mismatch: got {D}, expected {HEAD_SIZE}"
    assert T >= 1, "T must be >= 1"
    inputs = [x.astype(mx.float32) for x in [r,w,k,v,a,b,h]]
    res = _get_infer_kernel(H, T)(
        inputs=inputs,
        grid=(B*H, D, 1), threadgroup=(1, D, 1),
        output_shapes=[(B,T,H,D), (B,H,D,D)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return res[0], res[1]

# ─────────────────── Публичный API ──────────────────────────────────────────

def wkv7(r, w, k, v, a, b, training=True, state=None, return_state=False):
    """Единая точка входа.

    training=True, state=None, return_state=False  → (out, None), h0 = 0
    training=True, state и/или return_state        → (out, h_out), дифференцируемо по state
    training=False                                  → (out, h_out) на inference-ядре
    """
    if not training:
        return wkv7_infer(r, w, k, v, a, b, state)
    if state is None and not return_state:
        return wkv7_train(r, w, k, v, a, b), None
    return wkv7_train_with_state(r, w, k, v, a, b, state)
