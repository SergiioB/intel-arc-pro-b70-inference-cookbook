---
title: "Qwen3.8 GDN mixed-split v5 (2026-08-18)"
type: research
date: 2026-08-18
status: final
---

# Qwen3.8 GDN mixed-split v5 (2026-08-18)

PROVISIONAL — mixed concurrent probe + C1 n=3 screen. Not a public headline.

## Root cause (unchanged)

XPU fused `torch.ops._xpu_C.gdn_attention` host-checks
`!(num_spec_decodes > 0 && num_prefills + num_decodes > 0)`. CUDA already
splits this in `QwenGDNLinearAttention._forward_core`. Tracker:
[vllm-xpu-kernels#510](https://github.com/vllm-project/vllm-xpu-kernels/issues/510).
Writeup: `research/2026/vllm-xpu-mtp-agentic-crash-pr-20260816.md`.

## Fix

`scripts/tmp/vllm-xpu-causalconv-patch/patch_gdn_mixed_split_v5.py`

- Homogeneous batch: one fused call (C1 MTP / no-spec unchanged).
- Mixed: `index_select` each group, two fused calls with compact
  `arange` indices and `num_actual_tokens = group_count`, then
  `index_copy_` **both** `core_attn_out` and `z`. Idle optional side is
  `None` (binding is `Tensor?`).
- Must patch **every** `_xpu_ops.py`. First live probe only patched
  `/workspace/vllm`; EngineCore imported
  `/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py` and still
  died on the exclusive check.

CPU unit test: `test_gdn_mixed_split_v5.py` (16/16).

## Evidence

| Cell | Result | Notes |
|---|---|---|
| Autoresearch C1 n=3 (`a2d9018`) | prose 83.77 / code 81.1 / agentic 38.76 / prefill 1734.7 | vs keep 83.49 / 80.94 / 38.57 / 1720.1. Speed-flat. Sequential agentic never mixed a step. |
| Concurrent mix, site-packages unpatched | EngineDead, same `causal_conv1d` string | Confirms import-path bug. |
| Concurrent mix, both copies patched | decode 54 tok + prefill 32 tok completed; health 200 | Log: `[B70] GDN mixed split prefill=1 decode=0 spec=1 n_ns=1187 n_sp=5` (48 hits). |

Raw: `results/2026/08/qwen38-gdn-mixed-split-v5-20260818/`.

C1 p512/g128 **83.77** is n=3 median, cache-off, 230 W cap, client
post-first. Not a replacement for Run 40 n=5 **83.7**. Mixed probe is a
correctness smoke, not a tok/s claim.

## Not claimed

- Token/task-quality parity of the split vs isolated groups.
- 64K–128K gather cost.
- DSpark as a served cell (zero cells on this image). A 2.6G leftover
  dir is on `/mnt/<models-disk>/Qwen3.8-27B-DSpark-vLLM` after the remount —
  do not relaunch it inside this loop.

## Production / research closeout (2026-08-18)

- `scripts/tmp/qwen38-pi-serve.sh` launches MTP4 + this overlay
  (`NOSPEC=0`, `GPU_UTIL=0.88`). Serving-script change, not a speed keep.
- Autoresearch last keep: `8924799`. Standing `experiment.py` is
  `mtp4-split-standing`. Config-space exhausted.
- `/mnt/<models-disk>` was empty because `vg_data` is partial and systemd
  skipped `lv_models`. Remounted; boot unit `vg-data-lv-models.service`.
  Catalog ThinkingCap/Qwen3.6 GGUFs are still absent on B70 and Rock 5B.
