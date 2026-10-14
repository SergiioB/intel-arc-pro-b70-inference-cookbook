# SGLang Dense Xe2 Custom Kernel for Qwen3.8-27B

## The Strategy: The "1-Expert" Shortcut
SGLang possesses hand-written, highly optimized SYCL kernels specifically for the B70 (Xe2 architecture), bypassing the generic `oneDNN` black box used by vLLM. 

However, they only wrote these for Mixture of Experts (MoE) models (`moe_grouped_mm_nt_xe20_w4a16`).

**The Hack:** A dense matrix multiplication is mathematically identical to an MoE grouped matrix multiplication where the number of experts is `1`, and all rows are assigned to Expert 0.

By passing `num_experts = 1` and `rows_per_expert = [gemm_m]` to SGLang's MoE XMX kernel, we can physically execute Dense INT4 layers using their bleeding-edge DPAS hardware templates without rewriting the C++ execution pipeline.

## Implementation Steps
1. **Compile SGLang-Kernel-XPU:** Build the raw `sgl-kernel-xpu` library on the B70 workstation using Intel oneAPI 2025.3.
2. **Benchmark the Kernel:** Write a Python test script (`bench_dense_xe2.py`) that feeds Qwen3.8's exact layer dimensions (e.g., K=3584, N=18944) into the `moe_grouped_mm_nt_xe20_w4a16` op with `E=1`. Compare this directly against vLLM's `int4_gemm_w4a16` (oneDNN).
3. **Patch SGLang Python:** If the benchmark proves SGLang's XMX kernel is faster, we patch `sglang/srt/layers/quantization/w4a16_gptq.py` to route all dense linear layers through this 1-expert MoE kernel.
4. **GDN Fusion:** (Phase 2) Write a dedicated SYCL kernel for the Gated DeltaNet state updates to unblock the prefill ceiling.