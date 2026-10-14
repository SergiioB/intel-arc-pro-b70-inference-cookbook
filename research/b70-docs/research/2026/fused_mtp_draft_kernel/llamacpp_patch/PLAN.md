---
title: "llama.cpp SYCL Fused MTP Patch Plan"
type: research
date: 2026-09-05
status: final
---

> QUARANTINED — INVALID, UNIMPLEMENTED PROTOTYPE. No compilation, parity, or performance result exists. See QUARANTINE.md.

# llama.cpp SYCL Fused MTP Patch Plan

To integrate our `fused_mtp_step_sycl` kernel into `llama.cpp` for the Qwopus GGUF model, we need to modify three areas:

1.  **`ggml-sycl.cpp`**: 
    - Inject the SYCL device kernel (`ggml_sycl_fused_mtp_draft_step`).
    - Expose the C wrapper: `void ggml_sycl_fused_mtp(ggml_tensor * dst)`.
2.  **`ggml-backend-sycl.cpp`**: 
    - Catch the custom fused operation inside `ggml_backend_sycl_compute()`.
3.  **`llama.cpp` (Graph Builder)**:
    - In `build_qwen2_mtp()` or the MTP graph building logic, instead of appending sequential `ggml_norm`, `ggml_mul_mat`, and `ggml_argmax` operations, append our new custom operator (`GGML_OP_FUSED_MTP_DRAFT_STEP`).

## Current Status
We are drafting the unified `.patch` file that will apply cleanly against `llama.cpp` `master` (or the specific checkout on the B70).
