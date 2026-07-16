"""
dplr_bwd_mlx.py — S3.4a: аналитический backward одиночного чанка (S=0) в чистом MLX.
Цель — ДОКАЗАТЬ вывод VJP перед портом в Metal. Матмул-форма (всё = матмулы +
транспонированный треуг. solve + elementwise), лог-пространство, БЕЗ деления на w.

Forward (S=0), входы [C,D], q=r, gk=log w, gc=cumsum(gk,0):
  qh=q·e^gc, kh=k·e^-gc, bh=β·e^-gc, ah=α·e^(gc-gk)
  A_qk=le(qh@kh^T) A_qb=le(qh@bh^T) A_ab=lt(ah@bh^T) A_ak=lt(ah@kh^T)
  RHS_u=A_ak@v;  u=(I-A_ab)^{-1}RHS_u;  o=A_qk@v + A_qb@u

Backward (do):
  dA_qk=le(do@v^T); dv=A_qk^T@do; dA_qb=le(do@u^T); du=A_qb^T@do
  dc=(I-A_ab)^{-T}@du; dA_ab=lt(dc@u^T)
  dA_ak=lt(dc@v^T); dv+=A_ak^T@dc
  dqh=dA_qk@kh+dA_qb@bh; dkh=dA_qk^T@qh+dA_ak^T@ah
  dbh=dA_qb^T@qh+dA_ab^T@ah; dah=dA_ab@bh+dA_ak@kh
  dq=dqh·e^gc; dk=dkh·e^-gc; dβ=dbh·e^-gc; dα=dah·e^(gc-gk)
  dgc=dqh·qh - dkh·kh - dbh·bh + dah·ah;  dgk=-dah·ah
  dgk += revcumsum(dgc);  dw=dgk/w;  dr=dq
"""
import mlx.core as mx


def _masks(C):
    ii = mx.arange(C)[:, None]; jj = mx.arange(C)[None, :]
    return (jj <= ii), (jj < ii)


def _inv_neumann(A_ab, C):
    """(I - A_ab)^{-1} рядом Неймана; точно для строго-нижней нильпотентной A_ab."""
    A_inv = mx.eye(C); P = mx.eye(C)
    for _ in range(C - 1):
        P = P @ A_ab
        A_inv = A_inv + P
    return A_inv


def chunk_fwd_s0_mlx(r, w, k, v, a, b):
    """Forward одного чанка [C,D], S=0. Возврат o [C,Dv]."""
    C = r.shape[0]
    le, lt = _masks(C)
    gk = mx.log(w); gc = mx.cumsum(gk, axis=0); q = r
    qh = q * mx.exp(gc); kh = k * mx.exp(-gc)
    bh = b * mx.exp(-gc); ah = a * mx.exp(gc - gk)
    A_qk = mx.where(le, qh @ kh.T, 0.0)
    A_qb = mx.where(le, qh @ bh.T, 0.0)
    A_ab = mx.where(lt, ah @ bh.T, 0.0)
    A_ak = mx.where(lt, ah @ kh.T, 0.0)
    A_inv = _inv_neumann(A_ab, C)
    u = A_inv @ (A_ak @ v)
    return A_qk @ v + A_qb @ u


def chunk_bwd_s0_mlx(r, w, k, v, a, b, do):
    """Аналитический backward. Возврат (dr,dw,dk,dv,da,db), все [C,D]."""
    C = r.shape[0]
    le, lt = _masks(C)
    gk = mx.log(w); gc = mx.cumsum(gk, axis=0); q = r
    Eg = mx.exp(gc); Eng = mx.exp(-gc); Eag = mx.exp(gc - gk)
    qh = q * Eg; kh = k * Eng; bh = b * Eng; ah = a * Eag
    A_qk = mx.where(le, qh @ kh.T, 0.0)
    A_qb = mx.where(le, qh @ bh.T, 0.0)
    A_ab = mx.where(lt, ah @ bh.T, 0.0)
    A_ak = mx.where(lt, ah @ kh.T, 0.0)
    M_inv = _inv_neumann(A_ab, C)
    RHS_u = A_ak @ v
    u = M_inv @ RHS_u

    dA_qk = mx.where(le, do @ v.T, 0.0)
    dv = A_qk.T @ do
    dA_qb = mx.where(le, do @ u.T, 0.0)
    du = A_qb.T @ do
    dc = M_inv.T @ du
    dA_ab = mx.where(lt, dc @ u.T, 0.0)
    dA_ak = mx.where(lt, dc @ v.T, 0.0)
    dv = dv + A_ak.T @ dc

    dqh = dA_qk @ kh + dA_qb @ bh
    dkh = dA_qk.T @ qh + dA_ak.T @ ah
    dbh = dA_qb.T @ qh + dA_ab.T @ ah
    dah = dA_ab @ bh + dA_ak @ kh

    dq = dqh * Eg; dk = dkh * Eng; db = dbh * Eng; da = dah * Eag
    dgc = dqh * qh - dkh * kh - dbh * bh + dah * ah
    dgk = -dah * ah
    # revcumsum по оси времени: dgk[j] += sum_{i>=j} dgc[i]
    dgk = dgk + mx.cumsum(dgc, axis=0, reverse=True)
    dw = dgk / w
    dr = dq
    return dr, dw, dk, dv, da, db
