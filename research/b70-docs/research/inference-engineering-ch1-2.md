# Inference Engineering (Philip Kiely) - Chapters 1 & 2 Notes

## 1. Prerequisites (Scale & Specialization)
- **Scale and Specialization:** Transition from shared API endpoints (pay-per-token) to dedicated inference deployments once you need control over latency, model quality, rate limits, and have sufficient scale to optimize costs.
- **AI-Native Apps:** Different use cases have different requirements (e.g., chat needs low TTFT; agents need high overall latency control; offline processing favors throughput).
- **Latency vs. Throughput:** Online apps prioritize latency (TTFT / ITL). Offline tasks prioritize throughput (TPS).
- **Latency Metrics:** Focus on P90, P95, and P99 percentiles instead of means, as inference latency distributions are right-skewed and long-tail outliers ruin user experience.
- **End-to-End Metrics:** Always measure end-to-end user latency (including network and queuing) alongside pure on-GPU inference latency.
- **Model Selection:** Use the smallest, easiest-to-run model that meets requirements. Smaller models drastically reduce latency and cost. Fine-tuning and distillation are valid approaches to shrink model sizes for specific domains.

## 2. Models (Architecture and Bottlenecks)
- **Autoregressive Generation (LLMs):** Generate one token at a time. Inference has two phases:
  - **Prefill (Compute-Bound):** Processes the prompt, generates the KV cache. Metric: TTFT.
  - **Decode (Memory-Bound):** Autoregressively generates tokens using the KV cache. Metric: TPS.
- **Arithmetic Intensity:** The ratio of operations to bytes transferred from memory. Prefill has high arithmetic intensity (compute bound); decode has low arithmetic intensity (memory bandwidth bound).
- **Attention Optimizations:** Attention scales quadratically with context length. The KV cache makes decode linear time but consumes massive VRAM.
  - **Hardware/Kernel Implementations:** FlashAttention reduces VRAM reads/writes by fusing kernels and utilizing GPU SRAM efficiently. PagedAttention fragments KV cache into non-contiguous blocks, avoiding memory fragmentation and waste.
  - **Algorithmic Changes:** Sliding window attention, linear attention, and new architectures like state-space models (e.g., Mamba) seek to break the quadratic scaling of attention.
- **Mixture of Experts (MoE):** Adds sparsity. For example, a 235B MoE model might only activate 22B parameters per token. Great for single-batch local latency, but under high concurrency (large batches), nearly all experts load anyway, reducing the sparsity benefits unless Expert Parallelism is used.
- **Image/Video Generation:** Uses iterative denoising (diffusion). Compute-bound across the board (both phases). Video is incredibly expensive due to 3D attention (X, Y, Time).