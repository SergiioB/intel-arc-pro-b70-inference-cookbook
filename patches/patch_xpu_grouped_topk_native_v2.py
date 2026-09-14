#!/usr/bin/env python3
"""Patch vLLM grouped_topk: native XPU implementation + force-eager graph break.

v2 of the graph-safe grouped-topk fix. The v1 native implementation (repeated
argmax + one-hot masks, device-native, no CPU sync) is DETERMINISTIC in full
eager mode (verified: 3 identical temp-0 seeded requests byte-identical), but
non-deterministic when the whole model forward is torch-compiled (the compiled
kernels introduce nondeterminism at temp 0). This version additionally applies
`torch.compiler.disable` on XPU so the router executes eager even inside a
compiled model — a graph break at the router, keeping the rest compiled.

Verified on B70 (2026-08-12):
- eager + v1 native router: deterministic replay PASSES.
- compiled + v1: replay FAILS (nondeterministic).
- graphs + v1: 93.17/87.37 t/s but replay FAILS.

This v2 targets: compiled/graph mode deterministic + fast.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE = "vllm.model_executor.layers.fused_moe.router.grouped_topk_router"
MARKER = "# B70_XPU_GRAPH_SAFE_GROUPED_TOPK_V2"

DECORATOR_OLD = '''@torch.compile(
    dynamic=True,
    backend=current_platform.simple_compile_backend,
    options=maybe_disable_graph_partition(current_platform.simple_compile_backend),
)
def grouped_topk(
'''
DECORATOR_NEW = '''@(
    torch.compiler.disable
    if current_platform.is_xpu()
    else torch.compile(
        dynamic=True,
        backend=current_platform.simple_compile_backend,
        options=maybe_disable_graph_partition(
            current_platform.simple_compile_backend
        ),
    )
)
def grouped_topk(
'''
BODY_ANCHOR = '''    if (
        envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK
'''
BODY_PATCH = '''    # B70_XPU_GRAPH_SAFE_GROUPED_TOPK_V2
    if current_platform.is_xpu():
        assert hidden_states.size(0) == gating_output.size(0), (
            "Number of tokens mismatch"
        )
        if scoring_func == "softmax":
            original_scores = torch.softmax(gating_output, dim=-1)
        elif scoring_func == "sigmoid":
            original_scores = gating_output.sigmoid()
        else:
            raise ValueError(f"Unsupported scoring function: {scoring_func}")

        scores = original_scores
        if e_score_correction_bias is not None:
            scores = scores + e_score_correction_bias.unsqueeze(0)

        num_token, num_experts = scores.shape
        experts_per_group = num_experts // num_expert_group
        grouped_scores = scores.view(
            num_token, num_expert_group, experts_per_group
        )

        # Top-2 per group (matches the reference topk(2).sum() formula).
        first_group_max = grouped_scores.max(dim=-1)
        first_group_mask = torch.nn.functional.one_hot(
            first_group_max.indices, num_classes=experts_per_group
        ).bool()
        second_group_max = grouped_scores.masked_fill(
            first_group_mask, float("-inf")
        ).max(dim=-1).values
        group_scores = first_group_max.values + second_group_max

        selected_group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        remaining_group_scores = group_scores
        for _ in range(topk_group):
            group_ids = remaining_group_scores.argmax(dim=-1)
            group_one_hot = torch.nn.functional.one_hot(
                group_ids, num_classes=num_expert_group
            ).bool()
            selected_group_mask = selected_group_mask | group_one_hot
            remaining_group_scores = remaining_group_scores.masked_fill(
                group_one_hot, float("-inf")
            )

        expert_mask = (
            selected_group_mask.unsqueeze(-1)
            .expand(num_token, num_expert_group, experts_per_group)
            .reshape(num_token, num_experts)
        )
        remaining_scores = scores.masked_fill(~expert_mask, float("-inf"))
        selected_ids = []
        selected_weights = []
        for _ in range(topk):
            expert_ids = remaining_scores.argmax(dim=-1)
            expert_one_hot = torch.nn.functional.one_hot(
                expert_ids, num_classes=num_experts
            )
            selected_ids.append(expert_ids)
            selected_weights.append(
                (original_scores * expert_one_hot).sum(dim=-1)
            )
            remaining_scores = remaining_scores.masked_fill(
                expert_one_hot.bool(), float("-inf")
            )

        topk_ids = torch.stack(selected_ids, dim=-1)
        topk_weights = torch.stack(selected_weights, dim=-1)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(
                dim=-1, keepdim=True
            )
        if routed_scaling_factor != 1.0:
            topk_weights = topk_weights * routed_scaling_factor
        return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

    if (
        envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK
'''


def paths() -> list[Path]:
    candidates: list[Path] = []
    spec = importlib.util.find_spec(MODULE)
    if spec is not None and spec.origin is not None:
        candidates.append(Path(spec.origin))
    candidates.extend(
        [
            Path("/workspace/vllm/vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py"),
            Path("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py"),
        ]
    )
    return list(dict.fromkeys(path for path in candidates if path.is_file()))


def main() -> None:
    targets = paths()
    if not targets:
        raise SystemExit(f"cannot locate {MODULE}")
    for path in targets:
        text = path.read_text()
        if MARKER in text:
            print(f"[B70] grouped_topk v2 already applied: {path}")
            continue
        changed = False
        if DECORATOR_NEW not in text:
            if text.count(DECORATOR_OLD) != 1:
                raise SystemExit(f"decorator anchor mismatch in {path}")
            text = text.replace(DECORATOR_OLD, DECORATOR_NEW, 1)
            changed = True
        if BODY_PATCH not in text:
            if text.count(BODY_ANCHOR) != 1:
                raise SystemExit(f"body anchor mismatch in {path}")
            text = text.replace(BODY_ANCHOR, BODY_PATCH, 1)
            changed = True
        if changed:
            path.write_text(text)
            print(f"[B70] patched grouped_topk v2 (native + torch.compiler.disable): {path}")
        else:
            print(f"[B70] grouped_topk v2 already applied: {path}")


if __name__ == "__main__":
    main()
