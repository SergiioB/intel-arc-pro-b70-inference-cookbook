# SPDX-License-Identifier: Apache-2.0
"""CUTLASS Xe2 grouped-GEMM MoE path for native NVFP4 and MXFP8 experts on Intel XPU."""

import torch
from vllm.triton_utils import tl, triton
from vllm_xpu_kernels import fused_moe_interface as fmi

from . import _inkling_nvfp4_xe2 as nvfp4_ext


@triton.jit
def _gather_sorted(
    X,
    Order,
    Out,
    routes,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if row < routes:
        route = tl.load(Order + row).to(tl.int64)
        token = route // TOPK
        cols = tl.arange(0, BLOCK)
        for k0 in tl.static_range(0, K, BLOCK):
            c = k0 + cols
            v = tl.load(X + token * K + c, mask=c < K, other=0)
            tl.store(Out + row * K + c, v, mask=c < K)


@triton.jit
def _scatter_weighted_combine(
    DownSorted,
    InvOrder,
    Weights,
    IsPadding,
    Out,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = True
    if IsPadding is not None:
        live = not tl.load(IsPadding + token)
    acc = tl.zeros((BLOCK,), tl.float32)
    mask = live & (col < H)
    for slot in tl.static_range(TOPK):
        route = token * TOPK + slot
        srow = tl.load(InvOrder + route).to(tl.int64)
        val = tl.load(DownSorted + srow * H + col, mask=mask, other=0).to(tl.float32)
        w = tl.load(Weights + route, mask=live, other=0)
        acc += val * w
    tl.store(Out + token * H + col, acc.to(Out.dtype.element_ty), mask=col < H)


def cutlass_experts(
    x: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    is_padding: torch.Tensor | None,
    w13_w: torch.Tensor,
    w13_s: torch.Tensor,
    w13_esc: torch.Tensor | None,
    w2_w: torch.Tensor,
    w2_s: torch.Tensor,
    w2_esc: torch.Tensor | None,
    nvfp4: bool,
    num_experts: int,
    gate_cap: float,
    up_cap: float,
) -> torch.Tensor:
    """Graph-capture-safe CUTLASS Xe2 grouped-GEMM forward for NVFP4 and MXFP8."""
    tokens, hidden = x.shape
    if tokens == 0:
        return torch.empty_like(x)
    routes, topk = ids.numel(), ids.shape[1]

    if routes <= 256:
        counts, order, inv_order, atomic2 = nvfp4_ext.moe_route_sort(
            ids.contiguous(), is_padding, num_experts
        )
        atm0, atm1 = atomic2[0:1], atomic2[1:2]
    else:
        if is_padding is not None:
            ids = ids.masked_fill(is_padding[:, None], num_experts)
        flat_ids = ids.reshape(-1).to(torch.int64)
        flat_ids = torch.where(flat_ids < 0, num_experts, flat_ids)
        order = torch.argsort(flat_ids, stable=False)
        inv_order = torch.empty_like(order)
        inv_order.scatter_(0, order, torch.arange(routes, dtype=torch.int64, device=x.device))
        counts_all = torch.zeros((num_experts + 1,), dtype=torch.int32, device=x.device)
        counts_all.scatter_add_(
            0, flat_ids, torch.ones((routes,), dtype=torch.int32, device=x.device)
        )
        counts = counts_all[:num_experts]
        atm0 = atm1 = None

    sorted_x = x.new_empty((routes, hidden))
    _gather_sorted[(routes,)](
        x, order, sorted_x, routes, hidden, topk, 256, num_warps=4
    )

    n13 = w13_w.shape[1] if nvfp4 else w13_w.shape[2]
    inter = n13 // 2
    gate_up = x.new_empty((routes, n13))
    if nvfp4:
        nvfp4_ext.nvfp4_grouped_gemm_xe2(
            sorted_x, w13_w, w13_s, w13_esc, gate_up, counts, n13, hidden, num_experts,
            atm0,
        )
    else:
        fmi.cutlass_grouped_gemm_xe2(
            sorted_x, w13_w, w13_s, None, gate_up, counts, n13, hidden, num_experts
        )

    act = nvfp4_ext.fused_swiglu_interleaved(gate_up, gate_cap, up_cap)

    down_sorted = x.new_empty((routes, hidden))
    if nvfp4:
        nvfp4_ext.nvfp4_grouped_gemm_xe2(
            act, w2_w, w2_s, w2_esc, down_sorted, counts, hidden, inter, num_experts,
            atm1,
        )
    else:
        fmi.cutlass_grouped_gemm_xe2(
            act, w2_w, w2_s, None, down_sorted, counts, hidden, inter, num_experts
        )

    out = torch.empty_like(x)
    block_c = 256
    _scatter_weighted_combine[(tokens, triton.cdiv(hidden, block_c))](
        down_sorted, inv_order, weights, is_padding, out, hidden, topk, block_c, num_warps=4
    )
    return out
