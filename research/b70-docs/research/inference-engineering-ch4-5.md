# Inference Engineering (Philip Kiely) - Chapters 4 & 5 Notes

## 4. Software
- **CUDA and Kernels:** Inference code is compiled down to CUDA kernels. Kernel fusion (combining operations like matmul+bias+activation into a single kernel) avoids costly round-trips to GPU memory (VRAM).
- **Frameworks:** PyTorch is standard. PyTorch compilation (`torch.compile`) works but can't fuse custom plugins like FlashAttention or DeepGEMM.
- **Inference Engines:**
  - **vLLM:** Broadest hardware/model support. Easiest to use out-of-the-box. Good performance.
  - **SGLang:** Strong community support, particularly optimized for DeepSeek/Qwen MoE models. Uses a flexible frontend language. Great throughput.
  - **TensorRT-LLM:** Hardest to use, highest performance. NVIDIA's engine, optimized specifically for NVIDIA GPUs (Hopper/Blackwell). Supports all major optimization techniques.
- **NVIDIA Dynamo:** Distributed orchestration system for high-scale LLM inference (100B+ models, huge traffic). It manages KV cache routing, dynamic disaggregation, and multi-node parallelism.
- **Benchmarking:** Must shadow real traffic. Focus on sequence length, batching, and input parameters. Profile with Nsight (NSys/NCU) or PyTorch Profiler only if writing custom engines.

## 5. Techniques
- **Quantization:**
  - Prefill gets faster (compute-bound = more FLOPS on lower precision).
  - Decode gets faster (memory-bound = half the bytes to move).
  - Use FP8 (E4M3) or MXFP8 on Hopper/Blackwell. Integer formats (INT8/INT4) lack the dynamic range for quality unless heavily dynamic/block-quantized (like GGUF for local).
  - Sensitivity: Attention > KV Cache > Activations > Weights. Never quantize attention equations (like softmax) aggressively.
- **Speculative Decoding:**
  - Improves TPS (decode), but not TTFT (prefill). Uses spare compute at low batch sizes to generate draft tokens and validate them in parallel.
  - *Draft-Target:* Uses a smaller draft model. High overhead.
  - *EAGLE:* Uses hidden states to generate draft tokens. The recommended technique for high token acceptance.
  - *N-gram / Lookahead:* Dictionary-based matching. Incredible for Code Completion.
- **Caching (KV Cache Re-use):**
  - Prefix caching re-uses KV values for shared prompts (e.g., system prompts, RAG context). Drops TTFT drastically. Novel tokens must be placed at the END of the prompt.
  - KV Cache can be offloaded to CPU RAM (G2) or SSDs (G3/G4) via NVIDIA Dynamo's KVBM. Cache-aware routing directs traffic to nodes that hold the relevant prefix.
- **Model Parallelism:**
  - *Tensor Parallelism (TP):* Splits layers across GPUs. Best for low latency on dense models. Communication heavy (use NVLink).
  - *Expert Parallelism (EP):* Splits experts across GPUs in MoE models. Best for throughput. Less communication overhead, scales to InfiniBand (multi-node).
  - *Pipeline Parallelism (PP):* Splits model depth across nodes. Use for multi-node deployments alongside TP.
- **Disaggregation:**
  - Separates Prefill (compute-bound) and Decode (memory-bound) onto different GPU nodes.
  - Good for 100B+ models with 100M+ tokens/day traffic, especially with long input sequences.
  - Dynamo allows dynamic xPyD ratios (e.g., 5 Prefill to 3 Decode workers) based on traffic.