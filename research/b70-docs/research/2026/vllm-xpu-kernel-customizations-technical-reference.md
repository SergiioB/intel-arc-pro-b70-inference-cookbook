---
title: "B70 vLLM XPU Kernel Customizations — Complete Technical Reference"
type: research
date: 2026-08-12
status: final
---

# B70 vLLM XPU Kernel Customizations — Complete Technical Reference

**Purpose**: Exact, line-level documentation of every code change made to achieve
the Nemotron XPU path on Intel Arc Pro B70. Cite **isolated DFlash n=5** for
the current decode headline (186.61 t/s at p2048/g128) and the no-spec graph
n=5 card (93.00 / 87.25). The 10,349 figure is a **no-spec n=3 cold input
rate** (p8192/g128 TTFT), not isolated engine prefill and not the DFlash
campaign — see `research/2026/nemotron-cold-input-vs-10k.md`. Each change has the
file path, original code, patched code, rationale, and impact — sufficient to
create upstream PRs. PR split: `research/2026/vllm-xpu-pr-readiness-20260813.md`.

**Date**: 2026-08-12/13
**Hardware**: Intel Arc Pro B70 (Xe2/Battlemage, 32 GB GDDR6, 608 GB/s, 256 XMX, 32 Xe-cores)
**Base image**: `vllm/vllm-openai-xpu@sha256:1da0a95485455f08588c11080b9718992fd7d434c6a965d74654903a9d999c57`
**vLLM version**: `0.26.1rc1.dev668+g3ee2df303`
**vllm-xpu-kernels version**: `0.1.12.3` (source revision `de8eae16cdf2d864c86a203d9431b9bcdb42fb3c`)

---

## Change 1: Grouped-GEMM atomic buffer initialization fix (CRITICAL)

**Target repo**: `vllm-xpu-kernels` (Intel's XPU kernel library)
**PR type**: Bug fix (one-word change)
**Severity**: Critical — causes non-deterministic output in ALL XPU graph-captured MoE inference

### File
`csrc/xpu/grouped_gemm/xe_2/grouped_gemm_xe2_interface.hpp`, line ~226

### Original code
```cpp
  at::Tensor atomic_buffer =
      at::empty({static_cast<long>(1)}, ptr_A.options().dtype(at::kInt));
```

### Patched code
```cpp
  at::Tensor atomic_buffer =
      at::zeros({static_cast<long>(1)}, ptr_A.options().dtype(at::kInt));
```

### Root cause

The persistent block scheduler in `libgrouped_gemm_xe_2.so` distributes MoE GEMM
work across Xe2 work-groups using a **global atomic counter**. The counter buffer
is allocated with `at::empty` — **uninitialized memory**.

The kernel design assumes `group_id == 0` clears the counter to zero before other
groups read it:
```cpp
// Inside grouped_gemm_xe2.hpp kernel:
if (group_id == 0) {
    atm.store(0);  // Clear the counter
}
// ... all groups then atomicAdd to get their tile assignment
```

**SYCL provides no guarantee that `group_id == 0` launches first.** If any other
group executes its `cutlass::atomicAdd(atomic_buffer, 1)` before group 0 clears
the buffer, it reads leftover garbage from the allocator → inflated counter values
→ work-groups fetch out-of-bounds tile indices → compute writes to wrong output
tiles → **garbage MoE GEMM output**.

**In eager mode**: PyTorch allocates new memory each call. Over repeated calls,
memory gets reused, sometimes retaining a dirty counter from the previous launch.
This produces **intermittent** non-determinism (depends on allocator state).

**In XPU graph mode**: The memory address is captured and **fixed** for the
graph's lifetime. The buffer **guarantees** a dirty carry-over state on every
replay → **systematic** non-determinism on every graph-captured decode step.

### Fix

`at::zeros` allocates AND zeroes the buffer. PyTorch queues a `fill_(0)` kernel
that executes strictly before the GEMM kernel. In XPU graph mode, the graph
cleanly captures the `fill_` kernel → the counter is always zero at GEMM start.

### Verification

- **Without fix**: 8 identical temperature-0/seed=42 requests produced 2–4
  distinct outputs (divergence at contested near-tie tokens; both continuations
  coherent). Evidence: `~/B70-DOCS/results/nemotron-gptq-graph-n5-20260812T2045Z/`.
- **With fix**: 8/8 byte-identical outputs. Evidence:
  `~/B70-DOCS/results/nemotron-gptq-det-a-grade-20260812T2236Z/`.
- **Performance**: Zero cost (within 0.4% noise: 93.00 non-det vs 92.65 det).
- **The mathematical reduction inside the GEMM is tile-disjoint** (no Split-K
  atomics), so once the scheduler counter is fixed, the output is deterministic.

### Build

Rebuild `vllm-xpu-kernels` from source with this one-word change. Build
environment: official XPU nightly image + host oneAPI mounted read-only.
```bash
docker run --rm --entrypoint /bin/bash \
  -e MAX_JOBS=3 -e VLLM_VERSION_OVERRIDE=0.1.12.3 \
  -v /opt/intel:/opt/intel:ro \
  -v /path/to/vllm-xpu-kernels:/workspace/vllm-xpu-kernels \
  vllm/vllm-openai-xpu@sha256:1da0a954... \
  -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
       cd /workspace/vllm-xpu-kernels && python setup.py build_ext --inplace'
```
Incremental rebuild after the one-word change: 12 actions (~5 min).
Full rebuild from clean: 229 actions (~2.5 h at -j3).

---

## Change 2: Native graph-safe grouped-topk router for XPU

**Target repo**: `vLLM` (model_executor layer)
**PR type**: Platform-specific fix (XPU guard, CUDA unaffected)
**Severity**: High — blocks XPU graph capture for all grouped-topk MoE models (NemotronH, DeepSeek-V2/V3, etc.)

### File
`vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`

### Problem

The `grouped_topk` function has an **unconditional** `@torch.compile` decorator
(line 75):
```python
@torch.compile(
    dynamic=True,
    backend=current_platform.simple_compile_backend,
    options=maybe_disable_graph_partition(current_platform.simple_compile_backend),
)
def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    ...
)
```

On XPU, the compiled Inductor/Triton kernel for the group-score normalization
produces **out-of-range expert IDs** in the final gather:
```
Assertion `index out of bounds: 0 <= tmp0 < 128` failed.
```
(Source: retained Inductor kernel at
`/mnt/<models-disk>/inductor-cache/vv/cvvhmtiqivc2c3jhlx4gwpbq6t4ril6e3gfhpncu4bpm6lbbxu76.py:35`)

The compiled kernel gathers `scores[topk_ids]` where `topk_ids` contains values
outside `[0, num_experts)`. This is an XPU-specific Inductor codegen bug (similar
to vLLM's own note about PowerPC argmax under torch.compile).

The **base model** never executes this compiled router because its MoE routing is
fused inside the native XPU WNA16 kernel (`XPUExpertsWNA16`, non-monolithic). But
the **MTP draft** and any model that falls back to the torch MoE routing path
hits the compiled crash.

### Patch (v2 — `patch_xpu_grouped_topk_native_v2.py`)

Two changes in the same file:

**Change 2a: Disable `@torch.compile` on XPU**
```python
# Original:
@torch.compile(
    dynamic=True,
    backend=current_platform.simple_compile_backend,
    options=maybe_disable_graph_partition(current_platform.simple_compile_backend),
)
def grouped_topk(

# Patched:
@(
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
```

`torch.compiler.disable` creates a graph break at the router: the router runs
eager while the rest of the model stays compiled and graph-captured. This avoids
the Inductor codegen bug without disabling compilation for the entire model.

**Change 2b: Native XPU grouped_topk body** (inserted before the existing
`if envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK:` line)

The native implementation replaces `aten.topk` with repeated `argmax` + one-hot
masks, staying entirely on-device:

```python
if current_platform.is_xpu():
    assert hidden_states.size(0) == gating_output.size(0)
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
    grouped_scores = scores.view(num_token, num_expert_group, experts_per_group)

    # Top-2 per group using max + masked_fill (matches reference topk(2).sum())
    first_group_max = grouped_scores.max(dim=-1)
    first_group_mask = torch.nn.functional.one_hot(
        first_group_max.indices, num_classes=experts_per_group
    ).bool()
    second_group_max = grouped_scores.masked_fill(
        first_group_mask, float("-inf")
    ).max(dim=-1).values
    group_scores = first_group_max.values + second_group_max

    # Select top-k groups via repeated argmax
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

    # Mask to selected groups
    expert_mask = (
        selected_group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, experts_per_group)
        .reshape(num_token, num_experts)
    )
    remaining_scores = scores.masked_fill(~expert_mask, float("-inf"))

    # Select top-k experts via repeated argmax
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
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)
```

### Correctness

The native implementation matches the **reference formula** exactly:
- Group scores: `max(group) + second_max(group)` = reference `topk(2, dim=-1)[0].sum(dim=-1)`
- Expert selection: repeated argmax with masking = reference `topk(k, dim=-1)`
- Weights: `original_scores.gather(1, topk_ids)` = reference gather from unbiased scores
- Renormalization: identical

Verified: full-eager mode with native router produces **deterministic** replay
(8/8 byte-identical temp-0/seed requests). The router math is exonerated as the
source of non-determinism.

### Impact

Enables PIECEWISE + FULL XPU graph capture for models using `use_grouped_topk=True`
(NemotronH puzzle-MoE, DeepSeek-V2/V3). Without this patch, graph capture crashes
at the compiled grouped_topk Inductor kernel.

### Alternatives considered

- **CPU-routing fallback** (`patch_xpu_grouped_topk_cpu.py`): computes grouped
  routing on CPU. Correct and deterministic, but the XPU→CPU sync breaks graph
  capture (`RuntimeError: wait method cannot be used for an event associated
  with a command graph`). Not viable for graph mode.
- **Sort-based variant** (`patch_xpu_grouped_topk_native_sort.py`): uses full
  sort + equality masks. Also graph-safe but more complex. The argmax variant
  is simpler and sufficient.

---

## Change 3: B70 Mamba2 SSU tuning configuration

**Target repo**: `vLLM` (device-specific config)
**PR type**: New config file
**Severity**: Low (performance optimization, not a bug fix)

### File (new)
`vllm/model_executor/layers/mamba/ops/configs/selective_state_update/headdim=64,dstate=128,device_name=Intel(R)_Arc(TM)_Pro_B70_Graphics,cache_dtype=float32.json`

### Content
```json
{
  "BLOCK_SIZE_M": 8,
  "num_warps": 4
}
```

### Rationale

The Mamba2 selective state update (SSU) kernel uses a persistent block scheduler
with two tuning knobs: `BLOCK_SIZE_M` (tile height) and `num_warps`. The config
file is selected by `(headdim, dstate, device_name, cache_dtype)` and effective
batch `batch × nheads`.

The default fallback (`BLOCK_SIZE_M=4, num_warps=4`) generates 1,024 tiny Triton
programs per Mamba layer (ceil(64/4) × batch × 64 heads = 16 × 1 × 64 = 1024).
Across NemotronH's 23 Mamba layers: **23,552 Triton programs per token** —
excessive small-kernel launch overhead on Xe2.

The tuned config (`BLOCK_SIZE_M=8`) halves the program count to 512 per layer.

### Tuning sweep results (p512/g128, n=3, 150W)

| Config | p512 t/s | p8192 t/s |
|---|---|---|
| B4/W1 (fallback) | 21.61 | 21.90 |
| **B8/W4 (selected)** | **22.42** | **21.96** |
| B16/W4 | 21.92 | 22.28 |

The gain is small because SSU is only 0.36 ms/token (profiled). The primary
benefit is eliminating the "Using default Mamba SSU config" warning and
providing a starting point for further tuning.

### Other Xe2 devices

Each device needs its own config file. The B70's config should not be applied
to Arc B60 or data-center PVC without re-tuning.

---

## Change 4: Muse-Glimmer paged-decode kernel tuple

**Target repo**: `vllm-xpu-kernels` (kernel config)
**PR type**: Config addition (one line)
**Severity**: Medium — blocks Muse-Glimmer-30B text-only serving on XPU

### File
`csrc/xpu/attn/kernel_configs/paged_decode_default.conf`

### Added line
```
16,128,64,false,true,false
```

### Format
`qgroup,headsize,pagesize,causal,local,sink`

### Rationale

Muse-Glimmer-30B uses attention with qgroup=16 (16 query heads per KV head
group), headsize=128, pagesize=64, causal=false, local=true, sink=false.
This exact configuration was not in the compiled paged-decode kernel
specializations, causing:
```
RuntimeError: Paged decode kernel tuple not compiled for this configuration.
16,128,64,false,true,false
```

The error occurs after successful weight load, WNA16 dispatch, and KV-cache
allocation — during the first attention warmup call.

### Build impact

Adding the tuple triggers compilation of one additional paged-decode template
instantiation (`paged_decode_kernel_template_q16_h128_p64_ftf.cpp`). This adds
~30 seconds to the full rebuild.

### Full rebuild with all changes

The same `vllm-xpu-kernels` rebuild includes BOTH Change 1 (at::zeros) and
Change 4 (paged-decode tuple). Build command:
```bash
docker run --rm --entrypoint /bin/bash \
  -e MAX_JOBS=3 -e VLLM_VERSION_OVERRIDE=0.1.12.3 \
  -e VLLM_PAGED_DECODE_CONFIG=/workspace/vllm-xpu-kernels/csrc/xpu/attn/kernel_configs/paged_decode_default.conf \
  -v /opt/intel:/opt/intel:ro \
  -v /path/to/vllm-xpu-kernels:/workspace/vllm-xpu-kernels \
  vllm/vllm-openai-xpu@sha256:1da0a954... \
  -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
       cd /workspace/vllm-xpu-kernels && python setup.py build_ext --inplace'
```

---

## Change 5: NemotronH MTP prenorm hidden state (UNDER INVESTIGATION)

**Target repo**: `vLLM` (model-specific code)
**PR type**: Model bug fix
**Status**: Patched but did NOT resolve 0% MTP draft acceptance — additional issues remain

### File
`vllm/model_executor/models/nemotron_h.py`

### Change
In `NemotronHModel.forward()`, before `self.norm_f(hidden_states, residual)`:
```python
# Save pre-norm hidden state for MTP draft
self._mtp_target_hidden_states = (
    hidden_states + residual if residual is not None else hidden_states
)
hidden_states, _ = self.norm_f(hidden_states, residual)
```

In `NemotronHForCausalLM`, add:
```python
def get_mtp_target_hidden_states(self) -> torch.Tensor:
    return self.model._mtp_target_hidden_states
```

### Rationale
NemotronH applies `norm_f` (final RMSNorm) before passing hidden states to the
MTP draft. The draft's fusion layer applies its own `hnorm` → double-normalized
input → corrupted draft attention. The fix saves the pre-norm state and exposes
it via the standard `get_mtp_target_hidden_states()` hook.

### Status
Applied but 0% acceptance persists. The root cause of the 0% is likely deeper
(NaN/Inf in draft logits, position mismatch, or other computational issue).
Requires proper instrumentation from inside the GPU worker process to diagnose.

---

## Change 6: NemotronH MTP concat order (UNDER INVESTIGATION)

**Target repo**: `vLLM` (model-specific code)
**Status**: Patched but did NOT resolve 0% MTP draft acceptance

### File
`vllm/model_executor/models/nemotron_h_mtp.py`

### Change
In `NemotronHMTPAttentionDecoderLayer.forward()` and
`NemotronHMTPMoEDecoderLayer.forward()`:
```python
# Original:
fused = torch.cat([inputs_embeds_normed, previous_hidden_states_normed], dim=-1)

# Patched:
fused = torch.cat([previous_hidden_states_normed, inputs_embeds_normed], dim=-1)
```

### Rationale
MTP/EAGLE fusion layers typically concatenate in a specific order matching the
training-time `eh_proj` weight matrix. If the order is reversed, the projection
multiplies the wrong halves of the fused vector → uncorrelated logits.

### Status
Applied alongside Change 5 but 0% acceptance persists. The correct order needs
verification against the training configuration.

---

## Summary: what goes into which PR

### PR 1: `vllm-xpu-kernels` — Grouped-GEMM atomic buffer fix + paged-decode tuple

**Repo**: Intel's `vllm-xpu-kernels` (separate from vLLM)
**Files changed**:
1. `csrc/xpu/grouped_gemm/xe_2/grouped_gemm_xe2_interface.hpp` — `at::empty` → `at::zeros` (Change 1)
2. `csrc/xpu/attn/kernel_configs/paged_decode_default.conf` — add `16,128,64,false,true,false` (Change 4)

**Impact**: Fixes non-deterministic XPU graph-captured MoE output (all models,
not just NemotronH). Enables Muse-Glimmer-30B text-only serving.

### PR 2: `vLLM` — Native graph-safe grouped-topk for XPU

**Repo**: `vllm-project/vllm`
**File changed**: `vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py` (Change 2)
**Impact**: Enables XPU graph capture for grouped-topk MoE models. CUDA path
unaffected (original `@torch.compile` preserved via `if not is_xpu()` guard).

### PR 3: `vLLM` — B70 Mamba SSU config

**Repo**: `vllm-project/vllm`
**File added**: `vllm/model_executor/layers/mamba/ops/configs/selective_state_update/headdim=64,dstate=128,device_name=Intel(R)_Arc(TM)_Pro_B70_Graphics,cache_dtype=float32.json` (Change 3)
**Impact**: Removes "Using default Mamba SSU config" warning for B70. Small
performance improvement (~1%).

### NOT ready for PR: NemotronH MTP fixes (Changes 5–6)

Patched but 0% acceptance persists. Root cause of bad draft logits not yet
identified. Do NOT submit until the issue is fully diagnosed and resolved.

---

## Reproduction

### A-grade deterministic decode config (92.65 t/s)

```bash
# Pull the deterministic image (or build from source with Changes 1+4)
docker pull vllm-openai-xpu:1da0a954-det0123
# Apply Change 2 (native router) and Change 3 (SSU config) at runtime:
docker run ... vllm-openai-xpu:1da0a954-det0123 \
  -lc "python /patch_xpu_grouped_topk_native_v2.py; \
       cp ssu-b70-b8w4/*.json .../configs/selective_state_update/; \
       exec vllm serve /model --dtype bfloat16 --quantization gptq \
       --max-model-len 16384 --max-num-seqs 1 --max-num-batched-tokens 8192 \
       --gpu-memory-utilization 0.90 --no-enable-prefix-caching \
       --async-scheduling \
       --host 0.0.0.0 --port 8000"
# Env: VLLM_XPU_ENABLE_XPU_GRAPH=1
```

### Verification commands

```bash
# Decode speed (p512/g128, n=3, C1, cache off, 150W)
python3 b70-nemotron-gptq-eager-screen.py \
  --prompts validated-prompts.json --model served-name \
  --root http://127.0.0.1:8001

# Deterministic replay (8 identical temp-0 calls must be byte-identical)
for i in $(seq 1 8); do
  curl -s http://127.0.0.1:8001/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"served-name","prompt":"Return exactly this: SMOKE","max_tokens":32,"temperature":0.0,"seed":42}'
done | sort -u | wc -l  # Must be 1
```

### Evidence paths

| Artifact | Path |
|---|---|
| Profile (950 launches/token root cause) | `results/2026/08/nemotron-gptq-builtin-profile-20260812T171658Z-671822/` |
| A-grade n=3 decode screen | `results/2026/08/nemotron-gptq-det-a-grade-20260812T2236Z/` |
| n=5 supplementary | `results/2026/08/nemotron-gptq-graph-n5-20260812T2045Z/` |
| Eager baseline | `results/2026/08/nemotron-gptq-latest-eager-correctness-screen-20260812T131758Z-440538/` |
| Muse text-only screen | `results/2026/08/muse-glimmer-int4-screen-20260812T192845Z-835270/` |
| Deterministic wheel | `/mnt/<models-disk>/vxk-paged-decode-0.1.12.3/src/vllm-xpu-kernels/dist/vllm_xpu_kernels-0.1.12.3-cp312-cp312-linux_x86_64.whl` |
| Deterministic image | `vllm-openai-xpu:1da0a954-det0123` (Docker ID 506899080f10) |
| Patches | `scripts/tmp/patch_xpu_grouped_topk_native_v2.py`, `scripts/tmp/ssu-b70-b8w4/` |
| HF target | `SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym` |
| HF DFlash draft | `SergiioB/Nemotron-3.5-Lightning-30B-A3B-DFlash-BF16` |
| Isolated DFlash n=5 | `results/2026/08/nemotron-dflash-bf16-n7-n5-20260813T082203Z-1407994/` |
| Cookbook | `intel-arc-pro-b70-inference-cookbook/docs/nemotron35-30a3/NEMOTRON-DFLASH-B70.md` |

---

## Acknowledgments

- **Profile-driven root cause**: the 950-launches/token diagnosis was confirmed
  by an XPU torch profile (`urEnqueueKernelLaunchWithArgsExp`: 32.2 ms/token CPU
  vs 9.45 ms/token GPU), conducted by the `b70-vllm-server-specialist` subagent.
- **Grouped-GEMM atomic race**: found by the `b70-vllm-server-specialist`
  subagent after ruling out Inductor/Triton/PyTorch determinism knobs.
- **Native router**: implemented by the specialist after the compiled
  grouped_topk crashed with `tmp0 < 128` on XPU.
- **Architectural review**: the `b70-architect` subagent corrected the MTP
  "V1 bypass" diagnosis (it was an instrumentation failure, not a real bypass).
