# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-head prefill attention with 2D block loads on Intel XPU."""

import os
import torch

from vllm.triton_utils import tl, triton

from . import attention_triton


@triton.jit
def _rel_attention(
    Q,
    KV,
    Rel,
    BlockTable,
    SeqLens,
    CuSeqlensQ,
    Out,
    Lse,
    softmax_scale,
    num_tokens,
    num_kv_rows,
    k_lo_tok,
    k_hi_tok,
    stride_bt,
    stride_out_t,
    stride_out_h,
    stride_lse_h,
    stride_lse_t,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    PAGE: tl.constexpr,
    KV_W: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    HEAD_ROWS: tl.constexpr,
    HEAD_COLS: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    E_PAD: tl.constexpr = E + 256
    LOG2E: tl.constexpr = 1.4426950408889634
    t0 = tl.program_id(0) * BLOCK_M
    head = tl.program_id(1)
    req = tl.program_id(2)
    kv_head = head // G
    q_start = tl.load(CuSeqlensQ + req)
    q_len = tl.load(CuSeqlensQ + req + 1) - q_start
    if t0 >= q_len:
        return
    seqlen_k = tl.load(SeqLens + req)
    k_limit = tl.where(k_hi_tok < 0, seqlen_k, tl.minimum(seqlen_k, k_hi_tok))
    past = seqlen_k - q_len
    tok = t0 + tl.arange(0, BLOCK_M)
    pos = past + tok
    pos_hi = past + tl.minimum(t0 + BLOCK_M, q_len) - 1
    n_lo = k_lo_tok // BLOCK_N
    if WINDOW > 0:
        n_lo = tl.maximum(n_lo, tl.maximum(past + t0 - (WINDOW - 1), 0) // BLOCK_N)
    n_hi = (tl.minimum(pos_hi + 1, k_limit) + BLOCK_N - 1) // BLOCK_N

    qd = tl.make_tensor_descriptor(Q, [num_tokens, H * D], [H * D, 1], [BLOCK_M, D])
    kvd = tl.make_tensor_descriptor(KV, [num_kv_rows, KV_W], [KV_W, 1], [BLOCK_N, D])
    reld = tl.make_tensor_descriptor(
        Rel, [num_tokens + 128, H * E_PAD], [H * E_PAD, 1], [BLOCK_M, BLOCK_N]
    )
    q = qd.load([q_start + t0, head * D])
    # BLOCK_M divides the 128-row shear tile, so one rectangle covers its bias.
    tile_end = tl.minimum((t0 // 128 + 1) * 128, q_len)
    rel_base = (past + tile_end + 127) // 128 * 128 - E_PAD
    rel_col = head * E_PAD - rel_base

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    bt = BlockTable + req.to(tl.int64) * stride_bt
    for n in range(n_lo, n_hi):
        k0 = n * BLOCK_N
        keys = k0 + tl.arange(0, BLOCK_N)
        page = tl.load(bt + k0 // PAGE)
        row = page * ROWS_PER_BLOCK + kv_head * HEAD_ROWS + k0 % PAGE
        col = kv_head * HEAD_COLS
        k = kvd.load([row, col]).to(q.dtype)
        scores = tl.dot(q, tl.trans(k)) * (softmax_scale * LOG2E)
        dist = pos[:, None] - keys[None, :]
        visible = (dist >= 0) & (keys[None, :] < k_limit)
        if WINDOW > 0:
            visible &= dist < WINDOW
        bias = reld.load([q_start + t0, rel_col + k0])
        scores += tl.where(visible & (dist < E), bias.to(tl.float32) * LOG2E, 0.0)
        scores = tl.where(visible, scores, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(scores - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = kvd.load([row, col + D]).to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
        m_i = m_new

    empty = l_i == 0.0
    l_safe = tl.where(empty, 1.0, l_i)
    o = tl.where(empty[:, None], 0.0, acc / l_safe[:, None])
    lse = tl.where(empty, float("-inf"), (m_i + tl.log2(l_safe)) * 0.6931471805599453)
    token = (q_start + tok).to(tl.int64)
    offs_d = tl.arange(0, D)
    tl.store(
        Out + token[:, None] * stride_out_t + head * stride_out_h + offs_d[None, :],
        o,
        mask=(tok < q_len)[:, None],
    )
    tl.store(Lse + head * stride_lse_h + token * stride_lse_t, lse, mask=tok < q_len)


def _kv_layout(
    key: torch.Tensor, value: torch.Tensor
) -> tuple[int, int, int, int] | None:
    """Return the 2D page/head strides and row width of an interleaved KV cache."""
    _, page, heads, d = key.shape
    s0, s1, s2, s3 = key.stride()
    if s3 != 1 or value.stride() != key.stride() or not s1 or s0 % s1:
        return None
    if value.data_ptr() != key.data_ptr() + d * key.element_size():
        return None
    # Both layouts put V in the D columns immediately following K.
    if s2 % s1 == 0 and s2 // s1 * (heads - 1) + page <= s0 // s1 and 2 * d <= s1:
        return s0 // s1, s2 // s1, 0, s1  # HND
    if s2 >= 2 * d and (heads - 1) * s2 + 2 * d <= s1 and s0 == page * s1:
        return page, 0, s2, s1  # NHD
    return None


def rel_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    rel_logits: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    softmax_scale: float,
    window: int,
    num_splits: int,
    mode: int,
    k_lo_tok: int = 0,
    k_hi_tok: int = -1,
) -> None:
    """Write the prefill attention output and natural-log LSE."""
    tokens, heads, head_dim = q.shape
    blocks, page, kv_heads, _ = key_cache.shape
    bm, bn = int(os.getenv("INK2_PF_BLOCK_M", "64")), int(os.getenv("INK2_PF_BLOCK_N", "16"))
    layout = _kv_layout(key_cache, value_cache)
    if (
        layout is None
        or page % bn
        or num_splits != 1
        or mode != attention_triton.MODE_OUT_LSE
        or q.dtype != torch.bfloat16
        or rel_logits.dtype != torch.bfloat16
        or key_cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
        or value_cache.dtype != key_cache.dtype
        or not (q.is_contiguous() and rel_logits.is_contiguous())
    ):
        attention_triton.rel_attention(
            q,
            key_cache,
            value_cache,
            out,
            lse,
            rel_logits,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_query_len=max_query_len,
            softmax_scale=softmax_scale,
            window=window,
            num_splits=num_splits,
            mode=mode,
        )
        return
    rows_per_block, head_rows, head_cols, width = layout
    grid = (triton.cdiv(max_query_len, bm), heads, cache_seqlens.shape[0])
    _rel_attention[grid](
        q,
        key_cache,
        rel_logits,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        out,
        lse,
        softmax_scale,
        tokens,
        blocks * rows_per_block,
        k_lo_tok,
        k_hi_tok,
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        lse.stride(0),
        lse.stride(1),
        H=heads,
        G=heads // kv_heads,
        D=head_dim,
        E=rel_logits.shape[2] - 256,
        PAGE=page,
        KV_W=width,
        ROWS_PER_BLOCK=rows_per_block,
        HEAD_ROWS=head_rows,
        HEAD_COLS=head_cols,
        WINDOW=window,
        BLOCK_M=bm,
        BLOCK_N=bn,
        num_warps=int(os.getenv("INK2_PF_NUM_WARPS", "8")),
        num_stages=int(os.getenv("INK2_PF_NUM_STAGES", "2")),
        grf_mode="256",
    )
