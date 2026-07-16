"""
dplr_mlx.py — порт DPLR delta-rule (RWKV-7) в чистый MLX.

Строится по стадиям S1 (handoff). Контракт паритета к боевому
`wkv7_train_py` / `_wkv7_chunk_py` (rwkv_metal.kernel.wkv7):

  layout : [B, T, H, D], D == HEAD_SIZE.  H heads, key-dim == value-dim == D.
  w      : РЕАЛЬНЫЙ decay в (0.5455, 1.0) (как в модели, h *= w).
           Здесь decay живёт в ЛОГ-пространстве: gk = log(w).
  scale  : 1.0 (RWKV-7 НЕ скейлит r на d_k**-0.5, в отличие от DeltaNet-наива).
  state  : эталон держит h[b,h,s,d] (s=value, d=key); здесь S[b,h,d,m]
           (d=key, m=value) == h транспонированный в плоскости key/value.

Стадии:
  stage 0 — dplr_recurrence_mlx : по-токенная рекуррентность (этот файл).
  stage 1..3 — чанковый алгоритм (добавляются ниже по мере валидации).
"""
import mlx.core as mx


def dplr_recurrence_mlx(r, w, k, v, a, b, scale: float = 1.0):
    """
    Stage 0. По-токенная DPLR-рекуррентность в лог-пространстве decay.

    Все входы [B, T, H, D]; w — реальный decay (НЕ лог). Возврат o [B, T, H, D].
    Должно совпадать с wkv7_train_py(r,w,k,v,a,b) до ~1e-5 (fp32).
    """
    B, T, H, D = r.shape
    gk = mx.log(w)                      # лог-decay (ловушка #2: gk=log(w))
    q = r * scale                       # scale=1.0 (ловушка #1: r не скейлится)

    S = mx.zeros((B, H, D, D), dtype=r.dtype)   # S[b,h, d=key, m=value]
    outs = []
    for t in range(T):
        q_t = q[:, t]                   # [B,H,D] key
        k_t = k[:, t]                   # [B,H,D] key
        v_t = v[:, t]                   # [B,H,D] value
        a_t = a[:, t]                   # alpha, key
        b_t = b[:, t]                   # beta,  key
        gk_t = gk[:, t]                 # [B,H,D] key

        # sa[m] = S_d S[d,m]*alpha[d]   (контракция по key-оси)
        sa = mx.einsum("bhdm,bhd->bhm", S, a_t)
        # kv[d,m] = k[d]*v[m] + b[d]*sa[m]
        kv = mx.einsum("bhd,bhm->bhdm", k_t, v_t) + mx.einsum("bhd,bhm->bhdm", b_t, sa)
        # decay по key-оси d (axis -2), затем апдейт
        S = S * mx.exp(gk_t)[..., :, None] + kv
        # o[m] = S_d q[d]*S[d,m]
        o_t = mx.einsum("bhd,bhdm->bhm", q_t, S)
        outs.append(o_t)

    return mx.stack(outs, axis=1)       # [B,T,H,D]


def dplr_chunkwise_mlx(r, w, k, v, a, b, scale: float = 1.0, chunk_size: int = 16):
    """
    Stage 1. Чанковый DPLR (порт FLA dplr_chunkwise) в чистый MLX.

    Входы [B,T,H,D], w — реальный decay. Возврат o [B,T,H,D].
    Лог-decay внутри exp(diff), diff<=0 под маской (без факторизации → без overflow).
    UT-инверсия A_ab переписана функционально (без in-place).
    """
    # → head-first [B,H,L,D]
    r, w, k, v, a, b = (mx.transpose(x, (0, 2, 1, 3)) for x in (r, w, k, v, a, b))
    B, H, L, Dk = r.shape
    Dv = v.shape[-1]
    C = chunk_size
    assert L % C == 0
    N = L // C

    q = r * scale
    gk = mx.log(w)

    def chunks(x):
        return x.reshape(B, H, N, C, x.shape[-1])
    q, k, v, alpha, beta, gk = map(chunks, (q, k, v, a, b, gk))
    gc = mx.cumsum(gk, axis=-2)                      # [B,H,N,C,D]

    iidx = mx.arange(C)[:, None]
    jidx = mx.arange(C)[None, :]
    le = (jidx <= iidx)                              # j<=i  [C,C]
    lt = (jidx < iidx)                               # j<i
    NEG = -1e30

    # pairwise diffs over (i,j,d)
    diff = gc[..., :, None, :] - gc[..., None, :, :]                 # [B,H,N,C,C,D]
    diff_le = mx.where(le[..., None], diff, NEG)
    attn_le = mx.exp(diff_le)
    diff_lt = mx.where(lt[..., None],
                       gc[..., :, None, :] - gk[..., :, None, :] - gc[..., None, :, :],
                       NEG)
    attn_lt = mx.exp(diff_lt)

    qe = q[..., :, None, :]
    ae = alpha[..., :, None, :]
    kj = k[..., None, :, :]
    bj = beta[..., None, :, :]
    A_qk = mx.sum(qe * kj * attn_le, axis=-1)        # [B,H,N,C,C]
    A_qb = mx.sum(qe * bj * attn_le, axis=-1)
    A_ab = mx.sum(ae * bj * attn_lt, axis=-1)        # strictly lower
    A_ak = mx.sum(ae * kj * attn_lt, axis=-1)

    # UT-инверсия: forward substitution, функционально.
    # orig[i] — исходная строка i (множитель); rows[n] (n<i) — уже обновлённые.
    orig = [A_ab[..., i, :] for i in range(C)]       # each [B,H,N,C]
    rows = list(orig)
    for i in range(1, C):
        acc = orig[i]
        for n in range(i):
            acc = acc + orig[i][..., n:n + 1] * rows[n]
        rows[i] = acc
    A_inv = mx.stack(rows, axis=-2) + mx.eye(C)      # [B,H,N,C,C]

    u = A_inv @ (A_ak @ v)                           # [B,H,N,C,Dv]
    wmat = A_inv @ (mx.exp(gc - gk) * alpha)         # [B,H,N,C,Dk]

    S = mx.zeros((B, H, Dk, Dv), dtype=r.dtype)
    outs = []
    for n in range(N):
        q_i = q[:, :, n]; k_i = k[:, :, n]; v_i = v[:, :, n]
        beta_i = beta[:, :, n]; gc_i = gc[:, :, n]
        v2 = u[:, :, n] + wmat[:, :, n] @ S          # [B,H,C,Dv]
        o1 = A_qk[:, :, n] @ v_i
        o2 = A_qb[:, :, n] @ v2
        o3 = (q_i * mx.exp(gc_i)) @ S
        outs.append(o1 + o2 + o3)
        decay = mx.exp(gc_i[:, :, -1:, :] - gc_i)    # [B,H,C,Dk]
        last = mx.exp(gc_i[:, :, -1, :])[..., :, None]   # [B,H,Dk,1]
        S = S * last \
            + mx.transpose(k_i * decay, (0, 1, 3, 2)) @ v_i \
            + mx.transpose(beta_i * decay, (0, 1, 3, 2)) @ v2

    o = mx.stack(outs, axis=2).reshape(B, H, L, Dv)  # [B,H,L,Dv]
    return mx.transpose(o, (0, 2, 1, 3))             # [B,T,H,Dv]
