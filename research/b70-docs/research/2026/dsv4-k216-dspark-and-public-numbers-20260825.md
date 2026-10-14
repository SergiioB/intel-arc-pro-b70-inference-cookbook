---
title: "K216 decode next steps: DSpark / public numbers / kernels (2026-08-25)"
type: research
date: 2026-08-25
status: final
---

# K216 decode next steps: DSpark / public numbers / kernels (2026-08-25)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE**

Local evidence:

- decode ladder `results/2026/08/dsv4-k216-20260825Tdual2/`
- prefill ubatch `results/2026/08/dsv4-k216-20260825T0707Z-prefill/`
- this host: llama.cpp SYCL `build-sycl-latest-qw38` commit `1692f9e50`, dual B70, REAP-K216 UD-IQ3_XXS

## How good is ~9 tg / ~105 pp?

Local C1, n=3 screen, engine timings, 230 W during campaign:

- best decode: L5 `-ncmoe 15 -ts 65,35` p512/g128 median **8.87** predicted tok/s — **SUPERSEDED 2026-09-21: 9.25 clean / 9.58 patched**
- best prefill: L3 `-b 8192 -ub 1024 -c 8448` p8192/g1 median **105.1** prompt tok/s

### Closest public match (same artifact family)

[heath0xFF REAP-K216 GGUF card](https://huggingface.co/heath0xFF/DeepSeek-V4-Flash-0731-REAP-K216-GGUF): NVIDIA DGX Spark / GB10, llama.cpp CUDA `b10223`, full CUDA offload, Q8_0 K/V, `-b 1024 -ub 512`, short smokes **not** full-window:

- short: **33.2 pp / 15.9 gen**
- 32K–131K configured-context smokes: **44–48 pp / 16.3–16.8 gen**

Comparison class: `directional_only` (CUDA vs SYCL, unified GB10 vs dual 32 GB + CPU MoE, Q8 KV vs f16 MLA KV, different prompt shapes). Decode we are slower (~9 vs ~16). Prefill we are not obviously worse on dedicated long cells (~105 vs their short-prompt 33–48); do not headline that as a win.

### Not comparable headlines

- Unsloth: **60 → 120 tok/s on NVIDIA B200** with DSpark ([guide](https://unsloth.ai/docs/models/deepseek-v4)). Different class of GPU; workload unspecified.
- Unsloth “up to 2× decode” / 1.9× at `--spec-draft-n-max 3`: vendor claim, no raw samples.
- No public 4090/5090/Mac/Arc B70 tok/s table found for Flash-0731.

### Same-host dense models (also not a multiplier)

Architecture audit batch-1 llama.cpp: Qwen 27B Q5 MTP ~25 tg / 613 pp; Gemma 31B Q4 MTP ~24–27 tg. Flash is a 284B-class MoE with ~8–13B active, split across two cards, IQ3_XXS, no spec. ~9/105 is a **fits-and-runs** result, not a strong speed result, and is expected rather than a measurement bug.

## DSpark / MTP / DFlash2 on this binary

This commit **does** implement `--spec-type draft-dspark|draft-mtp|draft-dflash` in `libllama-common.so` (not the tiny `llama-server` stub). Flags live in `common/arg.cpp`. `llama-server --help` needs oneAPI `setvars` (`libsvml.so`).

Installed target GGUF has **no** `nextn` / `mtp` / `dspark` tensors. `deepseek4.expert_count = 216`, `expert_used_count = 6`. In-place `--spec-type draft-mtp` without `-md` cannot work.

No DeepSeek-V4 draft exists under `/mnt/<models-disk>`. The Nemotron DFlash GGUF is the wrong architecture and vocab.

Unsloth ships `dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf` (~10.9 GB) for the **unpruned 256-expert** Flash-0731, not REAP-K216. Pairing that draft with this 216-expert target is a compatibility experiment, not a documented recipe. Failure modes: vocab/layer-id/hidden-state mismatch, load error, or load-but-garbage acceptance.

DFlash2 is **not** a named llama.cpp mode here. Lightning Indexer is already a SYCL op for DSA attention; it is not speculative decode.

Architect E0 decode ceilings if a **compatible** GPU-resident draft actually works on L3 (not a promise):

- fused IQ3 `MUL_MAT_ID` MMVQ only: ~9.6–11 t/s
- MTP: ~10–13 t/s typical, ~15 stretch
- DSpark: ~11–16 t/s typical, ~18 stretch
- **not 30 t/s**

## Kernel leftover (rebuild, not flags)

1. Add IQ3_XXS to fused SYCL `ggml_sycl_mul_mat_vec_q_id()` so decode stops D2H expert-IDs + `stream->wait()` (`ggml-sycl.cpp` MUL_MAT_ID, `mmvq.cpp` quant dispatch). Decode-only; prefill needs grouped multi-row IQ3 GEMM.
2. Do **not** flip `ggml_sycl_supports_mmq()`; IQ3_XXS MMQ does not exist.
3. `GGML_SYCL_ENABLE_GRAPH=1` remains 0% (dual device + `MUL_MAT_ID`).
4. Optional tiny flag screen only: L5 `-ncmoe 14/15/16` with GPU0-heavy splits; L3 p8192 `-ub 1536/2048`. Expected 0–10%. Do not retry 46/54, 50/50, row-split.

## Recommendation

Stop broad flag-tuning. Next decode experiment is a **DSpark load probe** of Unsloth Q8_0 against REAP-K216 on L3, documented as compatibility-unknown. If it refuses to load or acceptance is junk, remaining decode work is the IQ3 fused MMVQ kernel or converting an official unpruned target (104 GB; does not fit beside K216 on this `/mnt/<models-disk>` 43 GB free).
