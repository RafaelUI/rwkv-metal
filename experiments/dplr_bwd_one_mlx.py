"""
dplr_bwd_one_mlx.py — S3.4b-ii: per-chunk fwd/bwd одного чанка с СОСТОЯНИЕМ.

Факторизация тела цикла chunk_fwd_seq/chunk_bwd_seq (dplr_bwd_chunk_mlx.py) в
изолированную функцию одного чанка [C,D] с входным S_in и выходным S_out.
Назначение: ОРАКУЛ для Metal-микрогарда (S3.4b-ii). Проверяется здесь же против
autograd изолированного chunk-forward (vjp с котангентами do, dS) — это
независимая истина для per-chunk математики (carry dS + все S-ветви).

Все формы соответствуют GEMM'ам, которые порт делает на Metal:
  o-ветвь, v2=u+wmat@S_in, u-solve, wmat-solve (общий M_inv), S-update,
  hats-grads, decay-ветвь. БЕЗ деления на w.
"""
import mlx.core as mx


def _masks(C):
    ii = mx.arange(C)[:, None]; jj = mx.arange(C)[None, :]
    return (jj <= ii), (jj < ii)


def _inv_neumann(A_ab, C):
    A_inv = mx.eye(C); P = mx.eye(C)
    for _ in range(C - 1):
        P = P @ A_ab
        A_inv = A_inv + P
    return A_inv


def chunk_fwd_one(r, w, k, v, a, b, S_in, C):
    """Один чанк [C,D] с состоянием. Возврат (o[C,Dv], S_out[Dk,Dv], cache)."""
    le, lt = _masks(C)
    gk = mx.log(w)
    gc = mx.cumsum(gk, axis=0)
    Eg = mx.exp(gc); Eng = mx.exp(-gc); Eag = mx.exp(gc - gk)
    qh = r * Eg; kh = k * Eng; bh = b * Eng; ah = a * Eag
    A_qk = mx.where(le, qh @ kh.T, 0.0)
    A_qb = mx.where(le, qh @ bh.T, 0.0)
    A_ab = mx.where(lt, ah @ bh.T, 0.0)
    A_ak = mx.where(lt, ah @ kh.T, 0.0)
    M_inv = _inv_neumann(A_ab, C)
    u = M_inv @ (A_ak @ v)
    wmat = M_inv @ ah
    v2 = u + wmat @ S_in
    o = A_qk @ v + A_qb @ v2 + qh @ S_in
    gc_last = gc[-1]
    last = mx.exp(gc_last)
    dec = mx.exp(gc_last[None, :] - gc)
    S_out = last[:, None] * S_in + (k * dec).T @ v + (b * dec).T @ v2
    cache = dict(S_in=S_in, u=u, wmat=wmat, v2=v2, M_inv=M_inv,
                 A_qk=A_qk, A_qb=A_qb, A_ab=A_ab, A_ak=A_ak,
                 qh=qh, kh=kh, bh=bh, ah=ah, gc=gc, gk=gk,
                 Eg=Eg, Eng=Eng, Eag=Eag, dec=dec, last=last,
                 k=k, v=v, b=b, w=w)
    return o, S_out, cache


def chunk_bwd_one(r, w, k, v, a, b, S_in, dS, do, C):
    """
    Backward одного чанка. Входы: чанк [C,D], S_in[Dk,Dv], dS[Dk,Dv]=adjoint S_out,
    do[C,Dv]=adjoint o. Возврат (dr,dw,dk,dv,da,db, dS_in[Dk,Dv]=carry в n-1).
    """
    Dk = r.shape[-1]; Dv = v.shape[-1]
    le, lt = _masks(C)
    _, _, c = chunk_fwd_one(r, w, k, v, a, b, S_in, C)
    u = c["u"]; wmat = c["wmat"]; v2 = c["v2"]; M_inv = c["M_inv"]
    A_qk = c["A_qk"]; A_qb = c["A_qb"]; A_ab = c["A_ab"]; A_ak = c["A_ak"]
    qh = c["qh"]; kh = c["kh"]; bh = c["bh"]; ah = c["ah"]
    gc = c["gc"]; gk = c["gk"]; Eg = c["Eg"]; Eng = c["Eng"]; Eag = c["Eag"]
    dec = c["dec"]; last = c["last"]; k_i = k; v_i = v; b_i = b

    dqh = mx.zeros((C, Dk)); dgc_last = mx.zeros((Dk,))
    dv_n = mx.zeros((C, Dv)); dv2 = mx.zeros((C, Dv))
    dS_in = mx.zeros((Dk, Dv))

    # S-update: S_out = last*S_in + (k*dec)^T@v + (b*dec)^T@v2
    kdec = k_i * dec; bdec = b_i * dec
    dS_in = dS_in + last[:, None] * dS
    dv_n = dv_n + kdec @ dS
    dv2 = dv2 + bdec @ dS
    dkdec = v_i @ dS.T
    dbdec = v2 @ dS.T
    dgc_last = dgc_last + last * (dS * S_in).sum(axis=1)

    # o = A_qk@v + A_qb@v2 + qh@S_in
    dA_qk = mx.where(le, do @ v_i.T, 0.0)
    dv_n = dv_n + A_qk.T @ do
    dA_qb = mx.where(le, do @ v2.T, 0.0)
    dv2 = dv2 + A_qb.T @ do
    dqh = dqh + do @ S_in.T
    dS_in = dS_in + qh.T @ do

    # v2 = u + wmat@S_in
    du = dv2
    dwmat = dv2 @ S_in.T
    dS_in = dS_in + wmat.T @ dv2

    # u-solve: u = M_inv @ (A_ak@v)
    dc = M_inv.T @ du
    dA_ab = mx.where(lt, dc @ u.T, 0.0)
    dA_ak = mx.where(lt, dc @ v_i.T, 0.0)
    dv_n = dv_n + A_ak.T @ dc
    # wmat-solve: wmat = M_inv @ ah (общий M_inv)
    dcw = M_inv.T @ dwmat
    dA_ab = dA_ab + mx.where(lt, dcw @ wmat.T, 0.0)
    dah_extra = dcw

    # hats grads
    dqh = dqh + dA_qk @ kh + dA_qb @ bh
    dkh = dA_qk.T @ qh + dA_ak.T @ ah
    dbh = dA_qb.T @ qh + dA_ab.T @ ah
    dah = dA_ab @ bh + dA_ak @ kh + dah_extra

    # hats -> leaves
    dq = dqh * Eg; dk_n = dkh * Eng; db_n = dbh * Eng; da_n = dah * Eag
    dgc = dqh * qh - dkh * kh - dbh * bh + dah * ah
    dgk = -dah * ah

    # decay-ветвь
    dk_n = dk_n + dkdec * dec
    db_n = db_n + dbdec * dec
    ddec = dkdec * k_i + dbdec * b_i
    dgc = dgc - ddec * dec
    dgc_last = dgc_last + (ddec * dec).sum(axis=0)

    dgc = dgc.at[-1].add(dgc_last)
    dgk = dgk + mx.cumsum(dgc, axis=0, reverse=True)
    dw_n = dgk / w
    dr_n = dq
    return dr_n, dw_n, dk_n, dv_n, da_n, db_n, dS_in
