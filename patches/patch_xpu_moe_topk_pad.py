#!/usr/bin/env python3
"""Pad unsupported MoE top-k to the next kernel-supported value (vLLM 0.31.0 XPU).

vllm_xpu_kernels remap_hidden_states only instantiates TopK in {1,2,4,6,7,8,10,12,16}
(csrc/moe/remap_hidden_states.cpp, upstream main 19c6ddca). Nemotron 3.5 Bolt routes 9
experts per token, Super VL 22: both raise "Unsupported TopK value" at the first forward.

Fix: before the fused kernel, append slots that repeat the row's last expert id with
router weight 0.0. The extra expert outputs are multiplied by exactly 0 and add +-0 to the
weighted sum, so the result is unchanged; the cost is the extra expert rows (9 -> 10: +11%
routed-expert work). Temporary until the kernel gets a TopK=9/22 specialization.
"""
import sys
from pathlib import Path

F = Path("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/experts/xpu_moe.py")
MARK = "B70_TOPK_PAD"
ANCHOR = (
    "        if self.fused_moe_impl is None:\n"
    "            topk = topk_ids.size(-1)\n"
)
INJECT = (
    "        # B70_TOPK_PAD: kernel supports TopK in {1,2,4,6,7,8,10,12,16} only\n"
    "        _b70_k = topk_ids.size(-1)\n"
    "        _b70_ok = (1, 2, 4, 6, 7, 8, 10, 12, 16)\n"
    "        if _b70_k not in _b70_ok:\n"
    "            _b70_pad = next(v for v in _b70_ok if v > _b70_k) - _b70_k\n"
    "            topk_ids = torch.cat(\n"
    "                [topk_ids, topk_ids[:, -1:].expand(-1, _b70_pad)], dim=-1\n"
    "            ).contiguous()\n"
    "            topk_weights = torch.cat(\n"
    "                [topk_weights, topk_weights.new_zeros(topk_weights.size(0), _b70_pad)],\n"
    "                dim=-1,\n"
    "            ).contiguous()\n"
) + ANCHOR


def main() -> int:
    src = F.read_text()
    if MARK in src:
        print(f"[{MARK}] already installed")
        return 0
    n = src.count(ANCHOR)
    if n != 1:
        print(f"[{MARK}] FATAL: anchor count {n} != 1 in {F}", file=sys.stderr)
        return 2
    if "import torch\n" not in src:
        print(f"[{MARK}] FATAL: torch not imported in {F}", file=sys.stderr)
        return 3
    F.write_text(src.replace(ANCHOR, INJECT, 1))
    print(f"[{MARK}] patched {F.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
