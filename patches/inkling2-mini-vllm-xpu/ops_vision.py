# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable PyTorch XPU ops for the Inkling2 vision encoder.

Matches the exact arithmetic and bf16 rounding boundaries of
`vllm.models.inkling2.nvidia.ops.vision` without CUDA/PTX inline assembly so the
same code runs on Intel XPU and CPU reference tests.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F


@functools.cache
def _rope_inv_freq(head_dim: int, device: torch.device) -> torch.Tensor:
    values = [10000.0 ** (-dim / head_dim) for dim in range(0, head_dim, 4)]
    return torch.tensor(values, dtype=torch.float32, device=device)


def gather_patches(
    pixels: torch.Tensor,
    offsets: torch.Tensor,
    patch_size: int,
    mean: tuple[float, float, float],
    clip_std: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    """Extract `[num_patches, patch_size*patch_size*3]` bf16 patches normalized
    through CLIP (bf16-rounded) -> PE [-1, 1] (bf16-rounded)."""
    _, side, _, channels = pixels.shape
    assert channels == 3 and pixels.is_contiguous()
    p = patch_size
    features = p * p * 3
    flat = pixels.reshape(-1)

    feat_idx = torch.arange(features, device=pixels.device, dtype=torch.int64)
    py = feat_idx // (p * 3)
    rem = feat_idx % (p * 3)
    rel_idx = py * (side * 3) + rem
    src_idx = offsets.to(torch.int64)[:, None] + rel_idx[None, :]

    val = flat[src_idx].to(torch.float32)
    ch = feat_idx % 3
    mean_t = torch.tensor(mean, dtype=torch.float32, device=pixels.device)[ch]
    clip_std_t = torch.tensor(clip_std, dtype=torch.float32, device=pixels.device)[ch]
    std_t = torch.tensor(std, dtype=torch.float32, device=pixels.device)[ch]

    val = ((val / 255.0 - mean_t) / clip_std_t).to(torch.bfloat16).to(torch.float32)
    val = 2.0 * (val * std_t + mean_t) - 1.0
    return val.to(torch.bfloat16)


def _source_coords(
    index: torch.Tensor, target: torch.Tensor, source: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    coord = (float(source) * (index + 0.5)) / target - 0.5
    coord = coord.clamp(0.0, float(source - 1))
    src0 = coord.floor().to(torch.int64)
    src1 = (src0 + 1).clamp(max=source - 1)
    w1 = coord - src0.to(torch.float32)
    return src0, src1, 1.0 - w1, w1


def embed_tokens(
    patch_embeds: torch.Tensor,
    patch_rows: torch.Tensor,
    position_grid: torch.Tensor,
    meta: torch.Tensor,
    class_embedding: torch.Tensor,
    cls_position_embedding: torch.Tensor,
    ln_pre: torch.nn.LayerNorm,
    ln_1: torch.nn.LayerNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build `(hidden, ln_1(hidden))` where `hidden = ln_pre(patch + pos)`."""
    src_h, src_w, dim = position_grid.shape
    grid_flat = position_grid.reshape(src_h * src_w, dim).to(torch.float32)

    y = meta[:, 0]
    is_cls = y < 0
    x = meta[:, 1].to(torch.float32)
    height = meta[:, 2].to(torch.float32).clamp(min=1.0)
    width = meta[:, 3].to(torch.float32).clamp(min=1.0)

    y0, y1, wy0, wy1 = _source_coords(y.clamp(min=0).to(torch.float32), height, src_h)
    x0, x1, wx0, wx1 = _source_coords(x.clamp(min=0), width, src_w)

    v00 = grid_flat[y0 * src_w + x0]
    v01 = grid_flat[y0 * src_w + x1]
    v10 = grid_flat[y1 * src_w + x0]
    v11 = grid_flat[y1 * src_w + x1]

    top = v00 * wx0[:, None] + v01 * wx1[:, None]
    bottom = v10 * wx0[:, None] + v11 * wx1[:, None]
    pos = (top * wy0[:, None] + bottom * wy1[:, None]).to(torch.bfloat16).to(torch.float32)

    hidden = patch_embeds[patch_rows.to(torch.int64)].to(torch.float32) + pos
    cls_vec = class_embedding.to(torch.float32) + cls_position_embedding.to(torch.float32)
    hidden = torch.where(is_cls[:, None], cls_vec[None, :], hidden)

    hidden = hidden.to(torch.bfloat16)
    hidden = F.layer_norm(hidden, (dim,), ln_pre.weight, ln_pre.bias, ln_pre.eps)
    normed = F.layer_norm(hidden, (dim,), ln_1.weight, ln_1.bias, ln_1.eps)
    return hidden.contiguous(), normed.contiguous()


def apply_rotary_qk_(qkv: torch.Tensor, meta: torch.Tensor, num_heads: int) -> None:
    """In-place interleaved 2D RoPE on the q and k thirds of `qkv [T, 3D]`."""
    num_tokens, width = qkv.shape
    head_dim = width // (3 * num_heads)
    half_pairs = head_dim // 4
    inv_freq = _rope_inv_freq(head_dim, qkv.device)

    # First half of pairs rotate by x + 1, second half by y + 1 (CLS: -1 + 1 = 0).
    pos_x = (meta[:, 1] + 1).to(torch.float32)
    pos_y = (meta[:, 0] + 1).to(torch.float32)
    ang_x = pos_x[:, None] * inv_freq[None, :]
    ang_y = pos_y[:, None] * inv_freq[None, :]
    angles = torch.cat((ang_x, ang_y), dim=-1)  # [T, head_dim // 2]
    cos = angles.cos()[:, None, :]  # [T, 1, head_dim // 2]
    sin = angles.sin()[:, None, :]

    qk = qkv[:, : 2 * num_heads * head_dim].view(num_tokens, 2 * num_heads, head_dim // 2, 2)
    even = qk[..., 0].to(torch.float32)
    odd = qk[..., 1].to(torch.float32)
    rot_even = even * cos - odd * sin
    rot_odd = odd * cos + even * sin
    qk[..., 0] = rot_even.to(qkv.dtype)
    qk[..., 1] = rot_odd.to(qkv.dtype)


def scaled_residual_norm_(
    hidden: torch.Tensor,
    update: torch.Tensor,
    gamma: torch.Tensor,
    norm: torch.nn.LayerNorm | None,
) -> torch.Tensor | None:
    """`hidden += (update * gamma).to(bf16)` in place; returns `norm(hidden)`."""
    scaled = (update.to(torch.float32) * gamma.to(torch.float32)).to(torch.bfloat16)
    hidden.copy_((hidden.to(torch.float32) + scaled.to(torch.float32)).to(torch.bfloat16))
    if norm is None:
        return None
    dim = hidden.shape[-1]
    return F.layer_norm(hidden, (dim,), norm.weight, norm.bias, norm.eps).contiguous()
