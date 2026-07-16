"""
dplr_bwd_chunk_mlx.py — S3.4b: аналитический МЕЖЧАНКОВЫЙ backward (carry dS) в чистом MLX.

Цель — ДОКАЗАТЬ вывод VJP полного чанкового DPLR (N чанков, межчанк-состояние S)
ПЕРЕД портом в Metal. Расширяет S3.4a (single-chunk, S=0) ветвями состояния:
  forward даёт вклад S через  o3 = qh @ S  и  v2 = u + wmat @ S,
  S-апдейт                    S' = last*S + (k*dec)^T@v + (b*dec)^T@v2.
Backward обратным циклом n=N-1..0 несёт dS назад и транспонирует S-апдейт-матмулы.

Один head [T, D] (B*H разматывается снаружи). Лог-decay, БЕЗ деления на w.
gc = ВНУТРИчанковый cumsum gk; межчанк-decay несёт last/dec через S.
Истина для grad = autograd рекуррентности (dplr_recurrence_mlx).
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


def chunk_fwd_seq(r, w, k, v, a, b, C):
    T, Dk = r.shape
    Dv = v.shape[-1]
    assert T % C == 0
    N = T // C
    le, lt = _masks(C)
    q = r
    gk_full = mx.log(w)

    S = mx.zeros((Dk, Dv), dtype=r.dtype)
    outs = []; cache = []
    for n in range(N):
        sl = slice(n * C, (n + 1) * C)
        q_i, k_i, v_i = q[sl], k[sl], v[sl]
        a_i, b_i = a[sl], b[sl]
        gk = gk_full[sl]
        gc = mx.cumsum(gk, axis=0)
        Eg = mx.exp(gc); Eng = mx.exp(-gc); Eag = mx.exp(gc - gk)
        qh = q_i * Eg; kh = k_i * Eng; bh = b_i * Eng; ah = a_i * Eag
        A_qk = mx.where(le, qh @ kh.T, 0.0)
        A_qb = mx.where(le, qh @ bh.T, 0.0)
        A_ab = mx.where(lt, ah @ bh.T, 0.0)
        A_ak = mx.where(lt, ah @ kh.T, 0.0)
        M_inv = _inv_neumann(A_ab, C)
        u = M_inv @ (A_ak @ v_i)
        wmat = M_inv @ ah
        v2 = u + wmat @ S
        o = A_qk @ v_i + A_qb @ v2 + qh @ S
        gc_last = gc[-1]
        last = mx.exp(gc_last)
        dec = mx.exp(gc_last[None, :] - gc)
        S_in = S
        S = last[:, None] * S + (k_i * dec).T @ v_i + (b_i * dec).T @ v2
        outs.append(o)
        cache.append(dict(S_in=S_in, u=u, wmat=wmat, v2=v2, M_inv=M_inv,
                          A_qk=A_qk, A_qb=A_qb, A_ab=A_ab, A_ak=A_ak,
                          qh=qh, kh=kh, bh=bh, ah=ah, gc=gc, gk=gk,
                          Eg=Eg, Eng=Eng, Eag=Eag, dec=dec, last=last,
                          k=k_i, v=v_i, b=b_i, sl=sl))
    o = mx.concatenate(outs, axis=0)
    return o, cache


def chunk_bwd_seq(r, w, k, v, a, b, do, C):
    T, Dk = r.shape
    Dv = v.shape[-1]
    N = T // C
    le, lt = _masks(C)
    _, cache = chunk_fwd_seq(r, w, k, v, a, b, C)

    dr = mx.zeros((T, Dk)); dw = mx.zeros((T, Dk))
    dk = mx.zeros((T, Dk)); dv = mx.zeros((T, Dv))
    da = mx.zeros((T, Dk)); db = mx.zeros((T, Dk))

    dS = mx.zeros((Dk, Dv), dtype=r.dtype)
    for n in range(N - 1, -1, -1):
        c = cache[n]
        S_in = c["S_in"]; u = c["u"]; wmat = c["wmat"]; v2 = c["v2"]; M_inv = c["M_inv"]
        A_qk = c["A_qk"]; A_qb = c["A_qb"]; A_ab = c["A_ab"]; A_ak = c["A_ak"]
        qh = c["qh"]; kh = c["kh"]; bh = c["bh"]; ah = c["ah"]
        gc = c["gc"]; gk = c["gk"]; Eg = c["Eg"]; Eng = c["Eng"]; Eag = c["Eag"]
        dec = c["dec"]; last = c["last"]; k_i = c["k"]; v_i = c["v"]; b_i = c["b"]
        do_n = do[c["sl"]]

        dqh = mx.zeros((C, Dk)); dgc_last = mx.zeros((Dk,))
        dv_n = mx.zeros((C, Dv)); dv2 = mx.zeros((C, Dv))
        dS_in = mx.zeros((Dk, Dv))

        # S-апдейт: S' = last*S_in + (k*dec)^T@v + (b*dec)^T@v2
        kdec = k_i * dec; bdec = b_i * dec
        dS_in = dS_in + last[:, None] * dS
        dv_n = dv_n + kdec @ dS
        dv2 = dv2 + bdec @ dS
        dkdec = v_i @ dS.T
        dbdec = v2 @ dS.T
        dgc_last = dgc_last + last * (dS * S_in).sum(axis=1)

        # o = A_qk@v + A_qb@v2 + qh@S_in
        dA_qk = mx.where(le, do_n @ v_i.T, 0.0)
        dv_n = dv_n + A_qk.T @ do_n
        dA_qb = mx.where(le, do_n @ v2.T, 0.0)
        dv2 = dv2 + A_qb.T @ do_n
        dqh = dqh + do_n @ S_in.T
        dS_in = dS_in + qh.T @ do_n

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
        dq = dqh * Eg
        dk_n = dkh * Eng
        db_n = dbh * Eng
        da_n = dah * Eag
        dgc = dqh * qh - dkh * kh - dbh * bh + dah * ah
        dgk = -dah * ah

        # decay-ветвь
        dk_n = dk_n + dkdec * dec
        db_n = db_n + dbdec * dec
        ddec = dkdec * k_i + dbdec * b_i
        dgc = dgc - ddec * dec
        dgc_last = dgc_last + (ddec * dec).sum(axis=0)

        # fold gc_last в gc[-1], затем gc->gk
        dgc = dgc.at[-1].add(dgc_last)
        dgk = dgk + mx.cumsum(dgc, axis=0, reverse=True)
        dw_n = dgk / w[c["sl"]]
        dr_n = dq

        sl = c["sl"]
        dr = dr.at[sl].add(dr_n); dw = dw.at[sl].add(dw_n)
        dk = dk.at[sl].add(dk_n); dv = dv.at[sl].add(dv_n)
        da = da.at[sl].add(da_n); db = db.at[sl].add(db_n)

        dS = dS_in

    return dr, dw, dk, dv, da, db
