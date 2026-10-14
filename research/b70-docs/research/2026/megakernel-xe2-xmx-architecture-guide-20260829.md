---
title: "Megakernels, Persistent Fusion, and Xe2/XMX Architecture Guide"
type: research
date: 2026-08-29
status: final
---

# Megakernels, Persistent Fusion, and Xe2/XMX Architecture Guide

**Date:** 2026-08-29  
**Status:** Active Research Authority  
**Applies to:** Intel Arc Pro B70 (Xe2 Battlemage, 32 Xe-cores, 32 GB GDDR6), SYCL, ESIMD, vLLM XPU, llama.cpp SYCL

---

## 1. Executive Summary & The Megakernel Thesis

Standard large language model (LLM) inference runtimes (such as PyTorch, Hugging Face, vLLM, and vanilla llama.cpp) execute a token forward pass by launching hundreds of small, isolated GPU kernels. For a typical 32-layer transformer model, a single token generation step requires approximately **200 to 300+ separate kernel launches** (RMSNorm $\to$ QKV GEMV $\to$ RoPE $\to$ KV-Cache update $\to$ Attention $\to$ Out-projection $\to$ Residual Add $\to$ RMSNorm $\to$ Gate/Up GEMV $\to$ SwiGLU $\to$ Down GEMV $\to$ Residual Add, across all layers).

In low-batch / single-sequence decode ($M=1$ token generation), this multi-kernel paradigm hits two devastating performance ceilings:

1. **Host & Driver Launch Overhead (The Time Gap):**
   - Each kernel launch carries $1.5\text{–}3\,\mu\text{s}$ of CPU driver serialization, queue submission, and GPU front-end scheduling latency.
   - $280\text{ launches} \times 2\,\mu\text{s} = \mathbf{560\,\mu\text{s}}$ of dead latency per token before accounting for any actual tensor math. If hardware capability could theoretically decode a token in $1\,\text{ms}$ ($1,000\text{ tok/s}$), launch overhead alone destroys $>50\%$ of attainable throughput.
2. **Intermediate Activation Memory Roundtrips (The Bandwidth Waste):**
   - During $M=1$ decode with hidden dimension $d = 4096$, the active activation vector is merely $4096 \times 2\text{ bytes} = \mathbf{8\text{ KB}}$ (in FP16/BF16) or $16\text{ KB}$ (in FP32).
   - This 8 KB vector fits comfortably inside on-chip registers (GRF) or Shared Local Memory (SLM).
   - In naive multi-kernel execution, this 8 KB vector is repeatedly written out to global VRAM (GDDR6) at the end of each kernel and re-read at the start of the next, saturating memory controllers with redundant activation traffic rather than streaming model weights and KV cache.

### The Megakernel Solution
A **Megakernel** fuses the entire transformer forward pass (or a large multi-layer block) into a **single persistent GPU kernel launch**. The thread grid stays resident on the hardware execution units across layers, maintains the residual activation state in on-chip SRAM/registers (GRF / SLM), and synchronizes internally via low-latency barriers or instruction-interpreter loops without yielding control to the CPU host.

---

## 2. Hardware Architecture Mapping: NVIDIA SM vs. Intel Xe2 Battlemage

Understanding how to translate CUDA-centric megakernel concepts (such as those in *Grokking Megakernels* by Infatoshi or *ThunderKittens* by Stanford Hazy Research) to Intel Arc Pro B70 requires mapping execution and memory primitives accurately.

| Architecture Component | NVIDIA Hopper / Blackwell | Intel Arc Pro B70 (Xe2 Battlemage) | Implications for B70 Megakernels |
|---|---|---|---|
| **Compute Units** | 132–144 SMs | **32 Xe-cores** (Render Slices) | Persistent grid should launch exactly 32 work-groups (or an exact multiple matching hardware thread occupancy). |
| **Vector Units** | 128 FP32 ALUs / SM | **256 Vector Engines (VE)** (8 VEs / Xe-core) | SIMD16 / SIMD32 execution width. High vector compute density per Xe-core. |
| **Matrix Engines** | 4 Tensor Cores / SM (WGMMA, HMMA) | **256 XMX Engines** (8 XMX engines / Xe-core) | Systolic `dpas` (Dot Product Accumulate Systolic) instructions for INT4, INT8, FP16, BF16. |
| **Register File (GRF)** | 64K 32-bit registers / SM (~256 KB) | **128 to 256 registers / thread** (up to 8 KB per hardware thread) | Massive General Register File (GRF). Can store entire 8 KB activation vector in registers across layer steps. |
| **On-Chip Shared Memory** | 228 KB Shared Memory / L1 per SM | **64 KB Shared Local Memory (SLM)** per Xe-core + dedicated L1 | 64 KB SLM is plenty for an 8 KB activation vector, LayerNorm statistics, and double-buffered weight tiles. |
| **L2 Cache** | 50 MB to 128 MB | **16 MB to 32 MB Unified L2** | High-bandwidth L2 cache. Frequently accessed tokens, router weights, and norm biases hit L2 without DRAM latency. |
| **Memory Subsystem** | 80 GB HBM3 (~3.3 TB/s) | **32 GB GDDR6** (256-bit bus, **~608 GB/s** — 256-bit × 19 Gbps; CORRECTED 2026-09-29: the earlier 512–576 figure is B580-class) | Bandwidth-limited on decode: streaming weights must achieve $\ge 80\text{–}90\%$ of theoretical GDDR6 peak. |
| **Low-level Programming** | PTX, SASS, CUDA C++, CuTe, ThunderKittens | **ESIMD (`sycl::ext::intel::esimd`)**, Intel DPC++ / SYCL, oneAPI | ESIMD gives direct register allocation, 2D block load messages, and DPAS intrinsics without compiler spills. |

---

## 3. The Six Architectural Pillars for B70 Megakernels

### Pillar I: The Persistent Work-Group Grid
Instead of launching separate kernels for each layer:
1. Launch a single grid containing **32 work-groups** (1 work-group pinned to each physical Xe-core) or a small multiple (e.g. 64 work-groups for 2 threads/Xe-core).
2. The work-groups enter a persistent execution loop across all $L$ layers:
   ```text
   for (int layer = 0; layer < num_layers; ++layer) {
       rms_norm_in_slm(residual_state, normalized_state);
       gemv_qkv_xmx(normalized_state, q, k, v, layer_weights);
       rope_and_kv_append(q, k, v, kv_cache, layer);
       paged_flash_attention(q, kv_cache, attn_out);
       gemv_out_xmx(attn_out, proj_out, layer_weights);
       vector_add(residual_state, proj_out);
       ...
       grid_barrier(); // inter-workgroup sync
   }
   ```
3. Eliminates all intermediate host-driver kernel submission latency ($500+\,\mu\text{s}$ saved per token).

### Pillar II: Register & SLM Activation Residence
In decode ($M=1$), the residual stream vector ($8\text{ KB}$ in FP16) never leaves on-chip storage:
- **Zero VRAM Roundtrips:** Only model weights and KV-cache history are read from GDDR6.
- Intermediate projections (Q, K, V, MLP intermediate states) are produced in registers, written to SLM, consumed by the next operation, and overwritten in-place.
- This frees up maximum GDDR6 memory bus bandwidth for weight streaming.

### Pillar III: ESIMD (`sycl::ext::intel::esimd`) vs. Implicit SYCL
Standard SYCL uses LLVM compiler auto-vectorization. For deep fused kernels, compiler heuristics frequently result in:
- Unintended register spills to scratch space;
- Suboptimal instruction scheduling;
- Inefficient unaligned memory accesses.

**ESIMD** provides explicit SIMD vector types and hardware instructions:
- `esimd::block_load_2d`: Directly maps to hardware 2D block read messages with specific cache hints (e.g. cache in L1/L2 or bypass L1).
- `esimd::xmx::dpas`: Directly issues Dot Product Accumulate Systolic instructions to XMX units.
- Explicit register management: Forces variables into GRF (General Register Files) with zero spill risk.

### Pillar IV: Sub-Group Specialization on Xe-Cores
Each Xe-core contains 8 Vector Engines and 8 XMX engines. Within a work-group executing on an Xe-core:
- **Producer Sub-Groups (Prefetching):** 1–2 sub-groups issue asynchronous 2D block load commands to fetch upcoming layer weights from GDDR6 into SLM buffers (double-buffering / pipelining).
- **Consumer Sub-Groups (Compute):** 6–7 sub-groups read weights and activation tiles from SLM and feed them into the XMX systolic pipeline (`dpas`).
- Hides memory access latency behind ongoing matrix-vector multiplication.

### Pillar V: DPAS Quantization Precision Alignment
The XMX systolic array natively supports:
- `INT4` (with INT8 / FP16 scaling)
- `INT8`
- `FP16` / `BF16`

**The Quantization Trap:** Formats like `IQ3_XXS`, `Q3_K`, or non-uniform quants lack native XMX DPAS instructions on Xe2. In SYCL, they force scalar dequantization loops back into FP16 on Vector Engines before math can occur.
- **Rule for Megakernels:** All megakernel weight representations **must** be aligned with native XMX formats (e.g., **GPTQ INT4 / AWQ INT4 / INT8** or native **FP16**).

### Pillar VI: Multi-GPU Placement & Topology Guard
In *Grokking Megakernels* Ch. 9, multi-GPU megakernels are designed for 8xH100 NVLink clusters using integrated `all_reduce` inside the kernel loop.

**The Host Reality on Dual-B70:**
- Our host ~~(ASUS B450-F) connects GPU 0 via PCIe Gen3 **x8** (~7.1 GB/s) and GPU 1 via PCIe Gen3 **x4** (~3.6 GB/s) with **no peer-to-peer (P2P)** memory access (all inter-GPU traffic bounces through host DDR4-2133)~~ **SUPERSEDED 2026-09-29** (register row 8): the host is the **ASUS ROG STRIX X570-F** (since 2026-08-27) with two CPU-attached PCIe **x8/x8** slots, and peer P2P **copy** capability is VERIFIED (RL-059: ~11.1–11.9 GB/s both-way, checksum-clean, 2.02× host-staged) — only **ATOMICS** are absent. See `research/2026/x570f-dual-b70-pcie-topology-20260830.md`.
- **Hard Rule:** **Never implement an inter-card Tensor-Parallel (TP) Megakernel across PCIe Gen3.** The $3.6\text{ GB/s}$ DDR4 bounce will completely stall the persistent GPU grid.
- **Correct Dual-Card Deployment:** Deploy **independent single-card megakernel engines** pinned to `ZE_AFFINITY_MASK=0` and `ZE_AFFINITY_MASK=1` to handle independent concurrent requests.

---

## 4. Decision Matrix: When to Use Megakernels vs. Standard Runtimes

| Criteria | Standard Engine (vLLM / llama.cpp) | Custom Megakernel / Deep Fusion |
|---|---|---|
| **Workload Phase** | Prefill (Prompt Processing, large $M$, compute-bound) | **Decode ($M=1$, low batch, latency-critical)** |
| **Model Size** | Large models ($\ge 27\text{B}\text{–}70\text{B}$) requiring multi-card offload or hybrid CPU split | **Small to Medium models ($0.5\text{B}\text{–}8\text{B}$)** fitting entirely in 32 GB VRAM |
| **Batch Size** | High batch ($B \ge 16\text{–}64$, throughput-oriented) | Low batch ($B = 1\text{–}4$, minimal latency per user) |
| **Serving Dynamic** | Multi-tenant dynamic prefix caching, arbitrary context lengths | Dedicated ultra-low latency agentic loops / speculative draft models |
| **Quantization** | GGUF k-quants, CPU-MoE hybrid offload | **Native INT4 / INT8 / FP16 XMX-aligned weights** |

---

## 5. Immediate Actionable Roadmap for B70-DOCS

See full model-by-model technical implementation plan: [`research/2026/megakernel-implementation-roadmap-20260829.md`](megakernel-implementation-roadmap-20260829.md).

1. **Eliminate MoE Host-Sync in DeepSeek-V4 & Ornith-1.5 (`ggml-sycl`):**
   - Fuse the MoE gating router + expert GEMV dispatch directly on-device in SYCL to eliminate 84+ blocking `stream->wait()` and D2H expert-ID transfers per token (`research/2026/dsv4-k216-request-to-output-pipeline-20260825.md`). Target: `recompute-required` (was $8.87 \to 20+\text{ tok/s}$ off the superseded 8.87 baseline — re-derive from 9.25 clean / 9.58 patched, and note the device-ID repair alone measured only +3.54 %, gate failed).
2. **Fuse Mamba2 SSD Block & MoE Router in Nemotron-3.5-Lightning-30B-A3B:**
   - Fuse the 16 operations across each of the 46 Mamba2 layers into a single work-group launch per layer (eliminating 736 driver dispatches per token). Keep selective scan states in GRF/SLM. Target: $186.6 \to 230+\text{ tok/s}$.
3. **Fuse MTP Speculative Draft Step in Qwen3.8-27B (`vllm-xpu-kernels`):**
   - Fuse RMSNorm + 1-layer MTP GEMV + LM Head projection + sampling into 1 persistent launch. Cut draft iteration time from $1.5\text{ ms} \to 0.25\text{ ms}$. Target: $102.6 \to 120+\text{ tok/s}$.
