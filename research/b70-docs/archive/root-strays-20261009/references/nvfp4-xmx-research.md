# NVFP4 vs INT4 on Intel Arc Pro B70 (Xe2)

## Architectural Reality
NVIDIA Blackwell has native FP4 tensor cores (`sm_100`). It computes `FP4 × FP4 = FP32` directly in silicon.
Intel Arc Pro B70 (Xe2) has XMX engines with DPAS instructions supporting `INT8, INT4, FP16, BF16`. **It does not have native FP4 matrix hardware.**

Because there is no physical `dpas.f4` instruction, any "FP4" or "NVFP4" model running on a B70 is actually a memory-bandwidth optimization (W4A16), not a W4A4 hardware execution. The execution path is:
1. Load 4-bit packed weights and FP8 (E4M3) block scales.
2. Unpack E2M1 to FP16 in registers (software dequantization using SIMD FMA).
3. Issue a standard `dpas` instruction using the decompressed BF16 weights and BF16 activations.

## The Benchmark Verdict
For dense 27B models on the B70, **GPTQ INT4 (W4A16) is architecturally faster than NVFP4.**
- GPTQ INT4 uses the native Intel `XPUwNa16` INT4 kernel. It achieves 83.7 t/s decode on vLLM.
- NVFP4 W4A16 forces register-level unpacking, wasting execution ports and generating overhead. 
- NVFP4 W4A4 requires activation quantization to FP4 in software, destroying both speed and quality.

## Where to Tinker for Absolute Maximum B70 Speed
Do not try to force FP4 math onto an integer/FP16 architecture. To push the B70 to its absolute limits for Qwen3.8-27B, optimize these three areas:
1. **Tune the `XPUwNa16LinearKernel`**: Remove fallback branches in `vllm-xpu-kernels` and hardcode the matrix tile sizes for Qwen3.8's exact dimensions.
2. **Gated DeltaNet (GDN) Kernels**: Qwen3.8 spends massive time in recurrent state convolutions where the XMX engines sit idle. Write a fused SYCL kernel for the GDN updates.
3. **MTP Speculative Decoding**: Optimize the MTP tree-acceptance kernel on XPU to evaluate 4 tokens for every single 14GB weight-read from GDDR6.
