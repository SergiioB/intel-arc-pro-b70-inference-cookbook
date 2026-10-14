---
title: "NemotronH MTP speculator: 0% draft acceptance on Intel XPU — V1 execution path bypasses standard sam"
type: research
date: 2026-08-13
status: final
---

# NemotronH MTP speculator: 0% draft acceptance on Intel XPU — V1 execution path bypasses standard sampling chain

> **RETRACTED diagnosis (keep as history).** The V1-bypass claim was wrong —
> drafts run and the rejection sampler is called; acceptance is 0% because
> draft logits are uncorrelated. Do not reopen the GitHub issue.
> **Working spec path (2026-08-13):** official DFlash, isolated n=5, 186.61 t/s
> at p2048/g128. Canonical HF: `SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym`
> + `SergiioB/Nemotron-3.5-Lightning-30B-A3B-DFlash-BF16`.

## Summary

Native MTP speculative decoding for Nemotron-3.5-Lightning-30B-A3B produces **0% draft acceptance** on Intel Arc Pro B70 (Xe2/Battlemage) via vLLM XPU nightly. After extensive debugging (9+ GPU cycles, 3 identified bugs patched, 7 instrumentation attempts), we found the V1 speculator's execution path **completely bypasses** the standard sampling/verification chain (`_process_engine_step`, `post_step`, `sample`, `RejectionSampler.__call__`). Drafts are proposed (spec counters increment) but never verified.

## Environment

- **Hardware**: Intel Arc Pro B70 (32 GB GDDR6, Xe2/Battlemage, 256 XMX, `xe` driver)
- **Image**: `vllm/vllm-openai-xpu@sha256:1da0a95485455f08588c11080b9718992fd7d434c6a965d74654903a9d999c57`
- **vLLM**: `0.26.1rc1.dev668+g3ee2df303`
- **vllm-xpu-kernels**: `0.1.12.3` (with grouped-GEMM `at::zeros` fix — see below)
- **Model**: Nemotron-3.5-Lightning-30B-A3B, local symmetric GPTQ INT4 G64 conversion (base model from `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`; canonical publish `SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym`)
- **Config**: `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`, `--enforce-eager`, `--no-enable-prefix-caching`

## Three identified bugs (all patched, still 0% acceptance)

### 1. Double-norming of hidden states
`NemotronHModel.forward` applies `norm_f` (final RMSNorm) to hidden states BEFORE passing to the MTP draft. The draft's `NemotronHMTPAttentionDecoderLayer` applies its OWN `hnorm` → double-normalized → garbage draft attention input.

**Fix**: `patch_nemotron_prenorm_mtp.py` — saves pre-norm state (`hidden_states + residual`) and exposes via `get_mtp_target_hidden_states()`.

### 2. Concatenation order mismatch
The MTP fusion layer concatenates `[inputs_embeds, hidden_states]` but MTP/EAGLE architectures may train with reversed order.

**Fix**: `patch_concat_order.py` — swaps to `[hidden_states, inputs_embeds]`.

### 3. Non-gated MoE w13 initialization
NemotronH uses a non-gated MoE (no `gate_proj`). The fused w13 mapping in `routed_experts.py` warns `"Unexpected gate/up projection names: up_proj, ."` and skips fused mapping. The XPU WNA16 expert backend handles this correctly (its own loading path), but the torch FusedMoE path (used by unquantized drafts) leaves expert weights uninitialized.

## The critical finding: V1 execution path bypass

After patching all three bugs and testing across 5 GPU-cycle combinations (INT4/BF16 × prenorm/concat/control — ALL 0% acceptance), we instrumented the V1 pipeline with file-based logging (baked into a custom Docker image, confirmed via grep):

| Method | File | Reached? |
|---|---|---|
| `EngineCore.__init__` | `engine/core.py` | ✅ (file write confirmed) |
| `_process_engine_step` | `engine/core.py:1442` | ❌ NEVER executed |
| `post_step` | `engine/core.py:615` | ❌ NEVER executed |
| `step` | `engine/core.py:583` | ❌ NEVER executed |
| `model_runner.sample` | `model_runner.py:1227` | ❌ NEVER executed |
| `RejectionSampler.__call__` | `rejection_sampler.py:233` | ❌ NEVER executed |
| `rejection_sample` | `rejection_sampler_utils.py:922` | ❌ NEVER executed |
| `DraftTokensHandler.get_draft_tokens` | `spec_decode/utils.py` | ❌ NEVER executed |

**All patches confirmed present in source (grep verified) and the EngineCore subprocess loads from `/workspace/vllm/` (confirmed via `__init__` file write).** Yet NONE of the inference-time methods execute during MTP spec decode.

The spec counters DO increment (`spec_decode_num_drafts_total: 31`, `spec_decode_num_draft_tokens_total: 63`, `spec_decode_num_accepted_tokens_total: 0`), confirming the speculator IS running — but through a completely non-standard execution path.

## Bonus: grouped-GEMM atomic race fix (already applied)

During this investigation, we found and fixed a **one-word bug** in `vllm-xpu-kernels` that caused non-deterministic output in XPU graph mode:

**File**: `csrc/xpu/grouped_gemm/xe_2/grouped_gemm_xe2_interface.hpp`
```diff
- at::Tensor atomic_buffer = at::empty({static_cast<long>(1)}, ...);
+ at::Tensor atomic_buffer = at::zeros({static_cast<long>(1)}, ...);
```

The persistent block-scheduler's atomic counter was allocated with `at::empty` (uninitialized). SYCL doesn't guarantee `group_id == 0` launches first, so other groups `atomicAdd` into garbage → inflated counters → out-of-bounds tiles → non-deterministic MoE output. In XPU graph mode the dirty buffer is captured and guaranteed to poison every replay.

**Impact**: This fix enabled deterministic XPU graph capture for NemotronH, raising decode from 21.8 → 92.65 t/s (4.25×) with byte-identical temperature-0 replay. The fix has zero performance cost.

## Request

1. **Investigate the V1 MTP speculator execution path** — why does the standard `_process_engine_step` → `step_fn` → `post_step` chain not execute during MTP spec decode? What alternative path does the speculator use?
2. **Verify the three identified bugs** — are they real issues on other platforms (CUDA)?
3. **Consider the `at::zeros` grouped-GEMM fix** for upstream merge (it affects all XPU users, not just NemotronH)

## Repro

```bash
docker pull vllm/vllm-openai-xpu@sha256:1da0a95485455f08588c11080b9718992fd7d434c6a965d74654903a9d999c57
# Apply patches: patch_nemotron_prenorm_mtp.py, patch_xpu_grouped_topk_native_v2.py
# (available at https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/tree/main/patches)
vllm serve /model --dtype bfloat16 --quantization gptq \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --enforce-eager \
  --no-enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
# Send any completion request → check /metrics:
# spec_decode_num_accepted_tokens_total = 0 (should be >0)
```
