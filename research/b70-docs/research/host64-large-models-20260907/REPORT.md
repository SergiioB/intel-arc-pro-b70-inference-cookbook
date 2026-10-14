---
doc_id: DOC-HOST64-CAPACITY-20260907
doc_type: research_report
title: "64 GiB Host RAM Impact: GLM-5.3-Flash & Qwen3.8-Flash-Next Capacity Ceilings"
lifecycle: active
created: 2026-09-07
canonical_for:
  - host64-capacity-research
supersedes: []
---

# 64 GiB Host RAM Impact: GLM-5.3-Flash & Qwen3.8-Flash-Next Capacity Ceilings

**Date:** 2026-09-07  
**Status:** PROVISIONAL — verified mathematically from upstream/remote artifacts; hardware offline.  
**Hardware target:** Two Intel Arc Pro B70 32 GB (64 GB total VRAM) + 64 GiB DDR4 Host RAM.

## Executive Summary

The upgrade to 64 GiB host RAM eliminates the `blocked_host_ram` constraint on PP2 / vLLM multi-process overhead, but it **does not magically fit 320B-class models on 64 GB of VRAM**.

- **GLM-5.3-Flash (320B total, 18B active):** Needs ~112 GiB just to hold the UD-IQ3_XXS weights. It **cannot fit** in 64 GB VRAM. It can run on `llama.cpp` using heavy CPU offload (approx. 56 GiB on host RAM), but performance will be severely bottlenecked (projected < 4 tok/s).
- **Qwen3.8-Flash-Next (125B total + 51B N-gram, 6B active):** The UD-IQ3_XXS fits (76.3 GiB total = 56 GiB VRAM + 20.3 GiB Host RAM for N-gram). The 64 GiB host RAM upgrade makes this **the premier target for the B70 pair**.

This document outlines the exact math, GGUF artifacts, PRs required, and the `llama.cpp` vs `vLLM` execution routes.

---

## 1. Capacity Arithmetic with 64 GiB Host RAM

### The B70 Constraints

*   **VRAM:** 2 × 32,656 MiB = ~63.7 GiB total.
    *   Maximum safe weight budget across both GPUs: **~56.0 GiB** (reserving ~7.7 GiB for KV cache, context, and driver overhead).
*   **Host RAM:** 64 GiB.
    *   Maximum safe budget for system overhead + vLLM/XCCL or `llama.cpp` offload: **~48.0 GiB** (reserving 16 GiB for OS/desktop).
*   **Total System Memory Budget (Weights + Cache):** ~104.0 GiB (56 VRAM + 48 RAM).

### GLM-5.3-Flash (320B)

*   **Architecture:** Hybrid linear/sparse-attention MoE. 320B parameters, 18B active. Context 1M.
*   **Artifacts:** `6block/GLM-5.3-Flash-GGUF` (self-quantized imatrix), `unsloth/GLM-5.3-Flash-GGUF`.
*   **Sizes:**
    *   `UD-IQ3_XXS`: 112.6 GiB
    *   `UD-IQ4_XS`: 155.8 GiB
*   **Fit Analysis:**
    *   `UD-IQ3_XXS` (112.6 GiB) minus 56 GiB VRAM = **56.6 GiB must be offloaded to CPU RAM**.
    *   56.6 GiB exceeds the safe 48.0 GiB host RAM budget (though it might barely fit if the OS is stripped to 7.4 GiB).
    *   **Performance Warning:** Offloading 56 GiB of a 320B MoE model across the PCIe Gen3 x8 bus to CPU RAM (DDR4-2133) will destroy performance. As seen on DeepSeek-V4 (`research/dsv4-k216-request-to-output-pipeline-20260825.md`), CPU-MoE dispatch over PCIe x8 yields < 4 tok/s.
    *   **Verdict:** Blocked for high-speed serving. Can run as a proof-of-concept `llama.cpp` CPU-offload experiment, but it is not a "fast" model for this hardware.

### Qwen3.8-Flash-Next (125B + 51B N-gram)

*   **Architecture:** Gated DeltaNet + Qwen Sparse Attention (QSA). 125B parameters, 6B active. Includes a massive 51B parameter N-gram table (PLE).
*   **Artifacts:** `unsloth/Qwen3.8-Flash-Next-GGUF`.
*   **Sizes:**
    *   `UD-IQ3_XXS`: 76.3 GiB
    *   `UD-IQ4_XS`: 87.2 GiB
*   **The N-gram Advantage:**
    *   The `per_layer_token_embd` (PLE) table is ~48.2 GiB in BF16 (shape `[160, 320001536]`).
    *   In the `UD-IQ3_XXS` GGUF, this table is quantized as type 20 (`IQ4_XS`), taking exactly **18.78 GiB** (verified via remote byte header inspection).
    *   **Crucially, the PLE table runs on the host CPU in `llama.cpp`.**
*   **Fit Analysis (`UD-IQ3_XXS`):**
    *   Total Size: 76.3 GiB.
    *   PLE Table (stays in Host RAM): 18.8 GiB.
    *   Remaining weights for VRAM: 76.3 - 18.8 = **57.5 GiB**.
    *   This is slightly above the 56.0 GiB ideal VRAM budget, requiring a tiny bit of CPU offload (1.5 GiB), but is immensely practical compared to GLM-5.3.
*   **Verdict:** **Strong Go.** The 64 GiB RAM upgrade makes this model fit beautifully (56 GiB in VRAM, 20 GiB in Host RAM).

---

## 2. Upstream Kernel Blockers (llama.cpp)

Both of these models use brand new, highly customized architectures. They require specific `llama.cpp` PRs to function on Intel GPUs.

### For Qwen3.8-Flash-Next

1.  **Merged Support:** The base architecture (Qwen4Exp) was merged in `ggml-org/llama.cpp#27742`.
2.  **MTP Support:** PR `#28243` adds NextN/MTP draft head support (1.3-2x speedup). *Open*.
3.  **Intel SYCL IQ MoE Bug:** PR `#28476` ("SYCL: Add IQ type handling for MoE"). Running IQ3_XXS on Intel SYCL is currently broken/slow because `mul_mat_vec_q_moe` falls back to host serialization for IQ quants. **This PR is mandatory before benchmarking Qwen3.8-Flash-Next on the B70.**
4.  **Expert Limit Bug:** PR `#28501` ("vulkan: raise the hoisted row-id limit for mul_mat_id from 256 to 512 experts"). Qwen3.8 has 512 experts. Code paths hardcoded to 256 experts will drop to un-hoisted slow paths.

### For GLM-5.3-Flash

1.  **Architecture Support:** PR `#27773` or `#27754`. *Open*. The model uses KDA linear attention + DSA laters.
2.  **MTP Support:** PR `#27917`. *Open*.

---

## 3. vLLM XPU Path

vLLM XPU is the canonical serving engine for the B70 (`AGENTS.md`).

*   **Qwen3.8-Flash-Next:** vLLM has an official recipe (`https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next`). It requires `VLLM_PLE_CPU_OFFLOAD=1` to push the N-gram table to RAM. However, the recipe explicitly notes: *"offload currently only runs on Nvidia devices"*. The Intel XPU backend does not yet support PLE CPU offload.
*   **GLM-5.3-Flash:** No vLLM recipe available yet.

**Conclusion:** `llama.cpp` (with SYCL) is the only viable day-zero path for Qwen3.8-Flash-Next on the B70s until Intel ports PLE CPU offload to the XPU backend.

---

## 4. Next Actions (When Host Returns)

1.  **Compile custom `llama.cpp`:**
    *   Checkout `master`.
    *   Apply PR `#28476` (SYCL IQ type handling for MoE). This is non-negotiable for IQ3 performance.
    *   Apply PR `#28243` (MTP).
2.  **Download Qwen3.8-Flash-Next:**
    *   Target: `unsloth/Qwen3.8-Flash-Next-GGUF`
    *   Files: `UD-IQ3_XXS` shards + `mmproj-BF16.gguf` + `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`.
3.  **Launch (llama.cpp split):**
    *   `llama-server -m Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf -mmproj mmproj-BF16.gguf --draft-model mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf -ngl 99 -sm layer -dev SYCL0,SYCL1 -c 8192 -fa on`
    *   *Note:* The PLE table (`per_layer_token_embd.weight`) is forced to CPU by `llama.cpp` automatically.
4.  **Update `AGENTS.md` and `LEDGER.md`:** Document that GLM-5.3-Flash is blocked for high-performance serving due to total size (>112 GiB), but Qwen3.8-Flash-Next is unblocked by the 64 GiB RAM upgrade.
