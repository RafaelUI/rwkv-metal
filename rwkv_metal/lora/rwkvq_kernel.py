"""
Fused Metal-кернель: sb6-деквант (packed qblk/qsqm/ddm -- та же K3-интерлив
раскладка, что в rwkv_quant/backends/metal/quant_linear_gw.py::GwQuantLinear)
-> dense [OUT, IN] в ОДИН launch, без промежуточных MLX-операций.

Почему не переиспользуем K3 GEMV/GEMM-кернели из rwkv_quant напрямую:
1) они forward-only заточены под инференс (dot-product с x, simd_sum
   редукция по threadgroup) -- нам нужен просто dequant, без матмула
   (матмул после отдаём mx.matmul -- сами авторы rwkv-quant для N>=128
   токенов используют ИМЕННО dequant-to-dense+mx.matmul, а не свои
   N-батч GEMV-кернели, см. GEMM_MIN_BATCH_NB=128 в quant_linear_gw.py --
   то есть наш путь архитектурно совпадает с их же выводом, узкое место
   было только в МЕДЛЕННОЙ Python-реализации самого dequant, не в
   выборе dequant+matmul вместо fused-GEMV).
2) rwkv_quant тянет torch на импорте -- rwkv-metal принципиально
   torch-free в рантайме (см. model/convert.py).

Раскладка (см. rwkv_quant/formats/schema.py -- источник истины):
  qblk[row,blk]  = 16Б кодов (block-local split-ниббл, pack_nib_block,
                   gs=32) [+ 4Б qh (bit4, pack_bitplane) [+ 4Б qh2 (bit5)]]
  qsqm[row,blk]  = uchar2 (qs, qm+31 как int8) на блок
  ddm[row,sblk]  = half2 (d, dm) на суперблок (sblk = blk // gw_sb)
  w = code * half(qs*d) + half(qm*dm), ФИНАЛЬНЫЙ combine в float32
      (не half!) -- бит-в-бит с rwkv_quant.formats.reader._dequantize_gw_sb6
      (сверено: tests/dev_check_rwkvq_linear.py, 0 расхождений).
"""
import mlx.core as mx

_KERNEL_CACHE = {}


def _get_dequant_kernel(IN: int, OUT: int, xbits: int, gw_sb: int):
    assert xbits in (0, 1, 2)
    assert IN % 32 == 0
    key = (IN, OUT, xbits, gw_sb)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    NB = IN // 32
    NSB = NB // gw_sb
    SU = 4 + xbits  # слов uint32 на блок в qblk

    hdr = f"""
constant uint IN_C  = {IN};
constant uint OUT_C = {OUT};
constant uint NB_C  = {NB};
constant uint NSB_C = {NSB};
constant uint SB_C  = {gw_sb};
constant uint SU_C  = {SU};
constant uint XBITS = {xbits};
constant uint TOTAL = {OUT * NB};
"""

    qh_body = """
    if (XBITS >= 1) {
        uint hb = qb[4];
        for (uint c = 0; c < 32; c++) nib[c] |= uchar(((hb >> c) & 1u) << 4);
    }
""" if xbits >= 1 else ""

    qh2_body = """
    if (XBITS >= 2) {
        uint hb2 = qb[5];
        for (uint c = 0; c < 32; c++) nib[c] |= uchar(((hb2 >> c) & 1u) << 5);
    }
""" if xbits >= 2 else ""

    body = """
    uint idx = thread_position_in_grid.x;
    if (idx >= TOTAL) return;
    uint row = idx / NB_C;
    uint blk = idx % NB_C;

    device const uint* qb = (device const uint*)qblk + (row * NB_C + blk) * SU_C;
    thread uchar nib[32];
    for (uint w = 0; w < 4; w++) {
        uint word = qb[w];
        for (uint b = 0; b < 4; b++) {
            uchar byte = uchar((word >> (b * 8)) & 0xFFu);
            uint j = w * 4 + b;
            nib[j]      = byte & 0xFu;
            nib[j + 16] = (byte >> 4) & 0xFu;
        }
    }
""" + qh_body + qh2_body + """
    uchar2 sm = ((device const uchar2*)qsqm)[row * NB_C + blk];
    half2  dd = ((device const half2*)ddm)[row * NSB_C + blk / SB_C];
    half s  = (half)((float)sm.x * (float)dd.x);
    half mn = (half)((float)as_type<char>(sm.y) * (float)dd.y);
    float sf = float(s);
    float mf = float(mn);

    device float* orow = out + row * IN_C + blk * 32;
    for (uint c = 0; c < 32; c++) {
        orow[c] = float(nib[c]) * sf + mf;
    }
"""

    kern = mx.fast.metal_kernel(
        name=f"rwkvq_dequant{4 + xbits}_{IN}_{OUT}",
        input_names=["qblk", "qsqm", "ddm"],
        output_names=["out"],
        header=hdr,
        source=body,
    )
    _KERNEL_CACHE[key] = kern
    return kern


def dequant_dense(qblk: mx.array, qsqm: mx.array, ddm: mx.array,
                   OUT: int, IN: int, gw_sb: int, xbits: int) -> mx.array:
    """packed sb6 (K3-интерлив) -> dense [OUT, IN] float32, один launch."""
    NB = IN // 32
    kern = _get_dequant_kernel(IN, OUT, xbits, gw_sb)
    TG = 256
    total = OUT * NB
    n_groups = (total + TG - 1) // TG
    out = kern(
        inputs=[qblk, qsqm, ddm],
        grid=(n_groups * TG, 1, 1), threadgroup=(TG, 1, 1),
        output_shapes=[(OUT, IN)],
        output_dtypes=[mx.float32],
    )[0]
    return out
