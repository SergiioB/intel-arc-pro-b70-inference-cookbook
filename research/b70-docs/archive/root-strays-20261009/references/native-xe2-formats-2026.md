# Xe2 Native Matrix Formats: Qwen3.8-27B on B70

## 1. The Verdict: GPTQ INT4 (W4A16)
GPTQ INT4 (W4A16) is the absolute highest-speed choice for dense 27B models on the B70. NVFP4 requires software dequantization that burns register cycles, and W8A8 halves decode speed while causing quality cliffs.

## 2. B70 Hardware Reality (DPAS Instructions)
The B70 XMX engines execute matrix math using the DPAS (Dot Product Accumulate Systolic) instruction.
- **Native support:** `INT8`, `INT4`, `FP16`, `BF16`.
- **Missing:** `FP4` (NVFP4), `FP8` (arrives in Xe3).

Because B70 lacks `dpas.f4`, any NVFP4 or FP8 format must be unpacked in software to FP16 or BF16 before hitting the matrix engines. This wastes register space, execution ports, and power. To get the absolute highest speed, you must feed DPAS what it speaks natively.

## 3. The Bandwidth Math (Why INT4 Wins)
LLM decode speed is bound by memory bandwidth, not compute TOPS.
- B70 bandwidth: 608 GB/s.
- Qwen3.8-27B at INT4: ~14 GB weights. Theoretical max decode = 43 tokens/s (raw memory limit).
- Qwen3.8-27B at INT8: ~28 GB weights. Theoretical max decode = 21 tokens/s.

INT4 maximizes the memory bus. W4A16 (INT4 weights, FP16 activations) uses the native Intel `XPUwNa16` kernel, keeping memory footprint small while protecting activation precision to avoid the W4A8 quality failure you saw.

## 4. Where to Tinker for Speed
If we are going to rebuild or tinker with kernels to maximize B70 throughput, do not build an NVFP4 dequantizer. Focus on these three native XMX/Xe2 optimizations:

1. **Tune `XPUwNa16LinearKernel`:** Open `vllm-xpu-kernels` and strip out generic fallback branches for the INT4 GEMM. Hardcode the tile sizes for Qwen3.8's exact dimensions on Xe2.
2. **GDN Recurrent Kernels:** Qwen3.8 has 48 Gated DeltaNet layers. XMX sits idle during recurrent convolutions. Writing a fused SYCL kernel for the GDN state updates will yield more speed than changing the matrix format.
3. **MTP Speculative Decoding:** MTP is the only way to beat the memory bandwidth wall. Optimize the Intel XPU MTP tree-acceptance kernel so we evaluate 4 tokens per 14GB weight-read.