# Intel Arc B70 Inference Optimization Strategy (Updated based on "Inference Engineering")

This document synthesizes operational plans for the Intel Arc B70 workstation, grounded in Philip Kiely's "Inference Engineering".

## 1. Engine Selection and Tooling
- **Current State:** The B70 relies on llama.cpp (SYCL) and OpenVINO. 
- **Strategic Shift:** We must evaluate TensorRT-LLM and vLLM paradigms. While TensorRT-LLM is highly NVIDIA-specific (and requires Hopper/Blackwell to shine with fused kernels/FP8), the **vLLM architecture** natively supports Intel GPUs. 
- **Action:** Transition B70 inference experiments towards Intel-optimized vLLM forks (Intel Extension for PyTorch) for production traffic.

## 2. Prefill vs Decode Bottlenecks
- **Prefill (Compute-Bound, TTFT):** Focus heavily on Intel's XMX engine. Ensure SYCL/OpenVINO kernels are dispatching matrix multiplications (GEMM) directly to XMX. 
- **Decode (Memory-Bound, TPS):** The Arc B70 has constrained memory bandwidth compared to H100s. We must aggressively deploy **KV Cache Quantization** (FP8/INT8) and enforce PagedAttention (native in vLLM) to prevent memory fragmentation during high-batch decode.

## 3. Advanced Inference Techniques
- **Speculative Decoding:** Given the B70's high compute capabilities but limited memory bandwidth, speculative decoding is the primary unlock for TPS. 
  - **Action:** Test EAGLE (hidden-state draft token models) or n-gram lookahead decoding for our coding/scripting pipelines on the B70. Avoid "Draft-Target" speculation as holding two models in B70 VRAM is prohibitive.
- **Prefix Caching:** For repetitive tasks (agent system prompts, code context), enable Prefix Caching to skip the compute-bound prefill phase entirely for matching context.
- **Quantization:** Standardize around FP8 (E4M3) for weights and KV cache. Avoid INT4/INT8 integer quantization for sensitive workloads unless absolutely required for VRAM, as they lack the dynamic range of floating point and degrade attention mechanism accuracy.

## 4. Scaling the B70 Architecture
- **Disaggregation:** Not viable on a single B70 instance. If we cluster the B70 with other GPUs (e.g., RTX laptops), we should treat the B70 as the compute-heavy **Prefill Engine** (handling context) and stream the KV cache to the RTX nodes acting as **Decode Engines**, adapting NVIDIA Dynamo's disaggregation patterns.
- **Client Code Overhead:** B70 inference pipelines should migrate to **WebSockets** or **gRPC** for continuous connections instead of REST HTTP to avoid TLS handshake overhead in low-latency loops.