---
title: "vLLM / Intel XPU — PR readiness after Nemotron DFlash (2026-08-13)"
type: research
date: 2026-08-13
status: final
---

# vLLM / Intel XPU — PR readiness after Nemotron DFlash (2026-08-13)

Local derived image `vllm-openai-xpu:1da0a954-det0123` is **not** a pullable
prerequisite. PRs must target public source. DFlash success does **not**
require submitting the draft converter or the local GPTQ tool.

## Submit first

### 1. Grouped-GEMM atomic init `at::empty` → `at::zeros`

| | |
|---|---|
| Repo | `intel/vllm-xpu-kernels` (or current Intel kernel repo) |
| File | `csrc/xpu/grouped_gemm/xe_2/grouped_gemm_xe2_interface.hpp` |
| Type | One-word bug fix |
| Why | Persistent-block scheduler atomic counter is uninitialized. SYCL does not guarantee `group_id==0` runs first. Graph replay makes the dirty counter systematic → non-deterministic MoE GEMM. |
| Proven | Nemotron GPTQ graph: 8/8 temp-0 match after the fix (`results/2026/08/nemotron-gptq-det-a-grade-20260812T2236Z/`). Zero perf cost vs 93.00 non-det. |
| Strip | Host paths, local image tag, Nemotron-only wording. Add a unit/repro that the buffer is zero before GEMM. |
| Risk | Low. Tile-disjoint GEMM; this is scheduler-counter init only. |

### 2. XPU graph-safe `grouped_topk` (native v2)

| | |
|---|---|
| Repo | `vllm-project/vllm` |
| File | `vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py` |
| Type | Platform guard |
| Why | Unconditional `@torch.compile` on XPU emits out-of-range expert ids (`0 <= tmp0 < 128`). Blocks graph capture for grouped-topk MoE (NemotronH, DeepSeek-V2/V3). |
| Proven | Enables PIECEWISE+FULL graphs; Nemotron no-spec 93/87 and DFlash 186.6. Full-eager + this router is deterministic. |
| Strip | `torch.compiler.disable` on XPU + native argmax/mask body. Keep CUDA `@torch.compile`. Add a device test, not a B70 script. |
| Risk | Medium. Must match reference `topk(2).sum()` group score and unbiased gather weights. |

### 3. Muse paged-decode tuple

| | |
|---|---|
| Repo | `vllm-xpu-kernels` |
| File | `csrc/xpu/attn/kernel_configs/paged_decode_default.conf` |
| Type | One-line config |
| Why | Missing `16,128,64,false,true,false` aborts Muse-Glimmer text decode after load. |
| Proven | Muse INT4 text screen. Independent of Nemotron DFlash. |
| Strip | Just the tuple + a one-line comment. Can ride with PR 1. |

## Submit after cleanup (optional / small)

### 4. B70 Mamba SSU JSON (B8/W4)

| | |
|---|---|
| Repo | `vllm` (SSU config tree) |
| Type | Device config |
| Why | Default B4 generates 1024 tiny programs/layer. B8/W4 halves that. |
| Proven | Small (~1%) on Nemotron p512; mainly removes the default-config warning. |
| Risk | Device-specific. Do **not** copy to B60/PVC without a retune. Filename must match `device_name` exactly. |

`VLLM_XPU_ENABLE_XPU_GRAPH=1` and `--async-scheduling` are flags, not PRs.

## Do not PR

| Item | Why |
|---|---|
| NemotronH MTP prenorm / concat patches | 0% draft acceptance remains. Root cause is still bad draft logits. |
| Retracted V1-bypass GitHub issue | Wrong diagnosis. Do not reopen. |
| W4A8 dense kernel | Discarded: diverges from W4A16 by word 15. |
| Local GPTQ / NVFP4→BF16 converters | Artifacts, not engine bugs. Publish weights + conversion notes. |
| Symlink-merge converter pattern | Can corrupt the source index. Document as an anti-pattern only. |
| Expert-id clamps / compiled-mode “fixes” | Hide the real assert. Eager or native router only. |
| Local det image tag as a requirement | Not pullable. Cite public digest + patches. |
| N-gram as production spec | Temp-0 nondeterministic on this stack. |

## Suggested PR order

1. kernels: `at::zeros` (+ Muse tuple if the same tree).
2. vLLM: XPU `grouped_topk` guard + native body + tests.
3. vLLM: B70 SSU JSON (separate, optional).
4. Stop. DFlash itself is already in vLLM; XPU needed a **BF16 draft** and the two runtime patches above.

Draft write-ups (issue / fix / repro, not opened on GitHub yet):
`research/prs/20260813-README.md`. Local kernel tree at
`de8eae16` still has `at::empty` and lacks `16,128,64,false,true,false`.
The GLM `false,false,false` sibling is already upstream.

## What DFlash proved (engine, not a PR)

Official `method=dflash` works on this XPU stack once the draft is BF16 (or
otherwise XPU-loadable), leftover NVFP4 metadata is gone, graphs + native
router are on, and production is pinned off for the campaign. Native MTP is
still 0%. Do not bundle MTP speculation into these PRs.
