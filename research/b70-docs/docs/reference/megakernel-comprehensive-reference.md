---
title: "Megakernels, Persistent Fusion, and Hardware Architecture"
status: living
type: reference
date: 2026-09-15
---

---
id: b70-ref-megakernel-comprehensive-reference
title: "Megakernels, Persistent Fusion, and Hardware Architecture: The Comprehensive Reference Manual"
document_type: reference
status: canonical
version: "1.0"
effective_date: 2026-08-29
last_reviewed: 2026-08-29
authority: docs/methods/benchmark-protocol.md
scope: Definitive architectural reference manual for Megakernels, Persistent Fusion, ESIMD, and XMX on Intel Arc Pro B70 and modern GPUs
---

# Megakernels, Persistent Fusion, and Hardware Architecture
## The Comprehensive Reference Manual for Intel Arc Pro B70 & Modern Inference Engines

---

## 1. Executive Summary & Foundational Physics

### 1.1 The Classical Multi-Kernel Paradigm
Modern LLM inference engines (such as PyTorch, vLLM, SGLang, and vanilla llama.cpp) decompose the forward pass of a transformer into a sequential directed acyclic graph (DAG) of small, isolated GPU kernels. For a 32-layer transformer model, a single token generation step requires approximately **200 to 300+ distinct GPU kernel launches**:

$$\text{RMSNorm} \to \text{QKV GEMV} \to \text{RoPE} \to \text{KV-Append} \to \text{FlashAttention} \to \text{Out-Proj} \to \text{Add} \to \text{RMSNorm} \to \text{Gate/Up} \to \text{SwiGLU} \to \text{Down} \to \text{Add}$$

Repeated across all 32 layers, this produces $\sim 280$ kernel invocations per token.

### 1.2 The Two Fatal Walls of Single-Sequence / Low-Batch Decode ($M=1$)

#### Wall 1: The CPU-Driver Launch Latency Barrier
- Launching any GPU kernel from the host CPU incurs $1.5\text{–}3.0\,\mu\text{s}$ of driver overhead (command buffer encoding, ring buffer submission, PCIe MMIO registers, and GPU front-end command streamer decoding).
- **The Calculation:**
  $$280\text{ launches} \times 2.0\,\mu\text{s} = \mathbf{560\,\mu\text{s}}\text{ of dead time per token}$$
- If a lightweight model could theoretically decode a token in $1.0\text{ ms}$ ($1,000\text{ tok/s}$), **$>50\%$ of the entire execution budget is wasted on driver scheduling gaps**, leaving the execution units (EUs / SMs) completely idle between operations.

#### Wall 2: The Activation Memory Round-Trip Tax
- In single-token decode ($M=1$) for a model with hidden dimension $d = 4096$:
  $$\text{Activation Vector Size} = 4096 \times 2\text{ bytes (FP16)} = \mathbf{8\text{ KB}}$$
- An 8 KB vector easily resides in on-chip hardware storage (registers or SRAM / SLM).
- However, in a multi-kernel runtime:
  1. Kernel $A$ writes the 8 KB activation vector across the memory crossbar to global VRAM (GDDR6/HBM);
  2. Kernel $B$ launches and reads that same 8 KB vector back from global VRAM.
- Across 280 kernel transitions, **hundreds of megabytes/sec of memory bus bandwidth are consumed by redundant activation round-trips**, competing with the critical streaming of model weights.

### 1.3 The Megakernel Definition
A **Megakernel** fuses an entire transformer forward pass (or multi-layer sub-networks) into a **single persistent GPU kernel launch**. The thread grid remains resident on the hardware execution units across all layers, maintains the residual activation state permanently in on-chip registers (GRF) or Shared Local Memory (SLM), and synchronizes internally using fast hardware/software barriers without returning control to the CPU host.

```
TRADITIONAL MULTI-KERNEL EXECUTION (280+ launches per token):
Host CPU: [Launch K1]──(gap)──[Launch K2]──(gap)──[Launch K3]──(gap)──...──[Launch K280]
           │                   │                   │                        │
GPU DRAM:  ▼ Write 8KB         ▼ Read/Write 8KB    ▼ Read/Write 8KB         ▼ Final Token
           [==== VRAM ====]    [==== VRAM ====]    [==== VRAM ====]         [==== VRAM ====]

PERSISTENT MEGAKERNEL EXECUTION (1 launch per token / generation):
Host CPU: [Single Persistent Launch] ──────────────────────────────────────────────> [Token Ready]
GPU Units: ┌────────────────────────────────────────────────────────────────────────┐
           │ Layer 0 (Norm->GEMV->Attn->MLP) -> Layer 1 -> ... -> Layer N-1         │
           │ All intermediate activations stay resident in On-Chip GRF / 64 KB SLM  │
           │ Only Model Weights & KV Cache stream from GDDR6                        │
           └────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Deep Hardware Architecture Comparison: NVIDIA vs. Intel Xe2 Battlemage

To translate CUDA-centric concepts (*Grokking Megakernels*, *ThunderKittens*, *FlashDecoding*) into Intel Arc Pro B70, we map hardware execution primitives directly:

| Architectural Metric | NVIDIA Hopper / Blackwell (H100 / RTX 5090) | Intel Arc Pro B70 (Xe2 Battlemage G31) | Engineering Implication for B70 |
|---|---|---|---|
| **Compute Execution Units** | 132–144 Streaming Multiprocessors (SM) | **32 Xe-cores** (Render Slices) | Persistent grid should launch exactly **32 work-groups** (1 per Xe-core) or a small multiple (e.g. 64). |
| **Vector Hardware** | 128 FP32 ALUs / SM (Warp size 32) | **256 Vector Engines (VE)** (8 VEs / Xe-core, SIMD16) | High SIMD16 vector compute density per Xe-core. |
| **Matrix Hardware** | 4 Tensor Cores / SM (`WGMMA` / `HMMA`) | **256 XMX Engines** (8 XMX engines / Xe-core) | Systolic `dpas` (Dot Product Accumulate Systolic) instructions for INT4, INT8, FP16, BF16. |
| **General Register File (GRF)** | 64K 32-bit registers / SM (~256 KB) | **128 to 256 registers / hardware thread** (up to **8 KB register space per thread**) | **Huge Register File Advantage:** Can store the entire 8 KB activation vector directly in registers across layers. |
| **On-Chip Shared Memory** | 228 KB Shared Memory / L1 cache per SM | **64 KB Shared Local Memory (SLM)** per Xe-core + dedicated L1 | 64 KB SLM easily holds the 8 KB hidden state, LayerNorm statistics, and double-buffered weight tiles. |
| **L2 Unified Cache** | 50 MB (H100) to 128 MB (B200) | **16 MB to 32 MB Unified L2** | High-bandwidth on-die L2 cache. Frequently accessed tokens, router weights, and norm biases hit L2 without DRAM latency. |
| **Memory Bandwidth** | ~3.3 TB/s (HBM3) | **~512–576 GB/s** (256-bit GDDR6, 32 GB) | Memory-bandwidth limited in decode: weight streaming must achieve $\ge 80\text{–}90\%$ of theoretical GDDR6 peak. |
| **Bare-Metal Programming** | PTX, SASS, CuTe, ThunderKittens | **ESIMD (`sycl::ext::intel::esimd`)**, Intel DPC++ / SYCL | ESIMD gives direct register allocation, 2D block load messages, and DPAS intrinsics without compiler auto-vectorization spills. |

---

## 3. When to Consider Megakernels: The Decision Framework

Megakernels are **not** a universal silver bullet. Understanding when to implement a persistent megakernel requires analyzing the **arithmetic intensity** and **operation topology** of the target workload:

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │       When Should You Build / Use a Megakernel?        │
                                  └───────────────────────────┬────────────────────────────┘
                                                              │
                 ┌────────────────────────────────────────────┴────────────────────────────────────────────┐
                 │                                                                                         │
                 ▼                                                                                         ▼
┌─────────────────────────────────────────────────────────┐               ┌─────────────────────────────────────────────────────────┐
│ HIGH-YIELD OPPORTUNITIES (Consider Megakernels)         │               │ ANTI-PATTERNS & LOW ROI (Avoid Megakernels)             │
├─────────────────────────────────────────────────────────┤               ├─────────────────────────────────────────────────────────┤
│ 1. Pure MoE Architectures (DeepSeek-V4, Ornith):        │               │ 1. Compute-Bound Prefill ($M \gg 1$, batch > 16):       │
│    • Hundreds of tiny GEMVs with host sync stalls.      │               │    • Large GEMMs already saturate GPU compute units;    │
│ 2. Hybrid SSM / Mamba2 Architectures (Nemotron-3.5):    │               │      launch overhead is $< 0.1\%$ of total time.         │
│    • 700+ sequential kernel dispatches per token.       │               │ 2. Large Dense Base Models without Speculation (27B):   │
│ 3. Speculative Draft Heads (MTP / DFlash / Eagle):      │               │    • 98.4% of time is spent streaming 14 GB weights;   │
│    • Micro-graph draft steps add 5-10ms overhead.       │               │      base GEMV megakernel only saves ~1.5%.             │
│ 4. Small Models ($0.5\text{B}\text{–}3\text{B}$) in C1 Decode:           │ 3. Non-Native Quant Formats (IQ3_XXS, Q3_K):             │
│    • Launch overhead accounts for 30-50% of latency.    │               │    • Lack systolic DPAS; scalar unpacking stalls ALU.   │
│ 5. Latency-Critical Agentic / Real-Time Tool Loops.     │               │ 4. Cross-Card TP over Copy-Only PCIe Links.             │
└─────────────────────────────────────────────────────────┘               └─────────────────────────────────────────────────────────┘
```

---

## 4. The Seven Core Implementation Patterns for Intel Xe2/XMX

### Pattern 1: The Persistent Work-Group Grid
Instead of launching separate kernels for each layer:
1. Launch a single grid containing exactly **32 work-groups** (1 work-group pinned to each physical Xe-core).
2. The work-groups enter a persistent execution loop across all $L$ layers:
   ```cpp
   // Host launch:
   q.submit([&] (sycl::handler &cgh) {
       cgh.parallel_for(sycl::nd_range<1>(32 * 256, 256), [=] (sycl::nd_item<1> item) {
           // Persistent thread loop across layers
           for (int layer = 0; layer < num_layers; ++layer) {
               // Execute layer operations inside on-chip registers / SLM
               ...
               // Synchronize across all 32 work-groups
               grid_barrier(item);
           }
       });
   });
   ```
3. Eliminates all intermediate host-driver kernel submission latency ($500+\,\mu\text{s}$ saved per token).

### Pattern 2: On-Chip Residual Stream Residence (Zero-VRAM Roundtrips)
In single-token decode ($M=1$), the residual stream vector ($8\text{ KB}$ in FP16) **never touches global VRAM**:
- **Registers (GRF):** Store the active hidden state vector directly in GRF.
- **Shared Local Memory (SLM):** Store LayerNorm statistics, intermediate activation projections, and attention key-value tile buffers.
- **GDDR6 Memory Bus:** Saturated 100% with streaming layer weights and reading KV cache history.

### Pattern 3: Sub-Group Specialization (Producer vs. Consumer on Xe-Cores)
Each Xe-core contains 8 Vector Engines and 8 XMX engines. Within a single work-group executing on an Xe-core:
- **Producer Sub-Groups (Prefetching):** 1–2 sub-groups execute asynchronous 2D block load commands (`esimd::block_load_2d`) to stream upcoming layer weights from GDDR6 into SLM double-buffers.
- **Consumer Sub-Groups (Compute):** 6–7 sub-groups read weights and activation tiles from SLM and drive the systolic XMX `dpas` pipeline.
- Completely overlaps memory load latency with matrix arithmetic.

```
       Xe-Core (Work-Group of 8 Sub-Groups)
 ┌──────────────────────────────────────────────────────────┐
 │ Sub-Group 0 (Producer):                                  │
 │   esimd::block_load_2d (GDDR6 -> SLM Buffer B)           │
 ├──────────────────────────────────────────────────────────┤
 │ Sub-Groups 1–7 (Consumers):                              │
 │   XMX dpas Matrix Compute (SLM Buffer A -> GRF Residual) │
 └──────────────────────────────────────────────────────────┘
```

### Pattern 4: Bare-Metal Register Control via ESIMD (`sycl::ext::intel::esimd`)
Standard DPC++ SYCL relies on LLVM compiler auto-vectorization. For deep fused kernels, compiler heuristics frequently spill registers to global memory.
**ESIMD provides explicit hardware control:**
- `esimd::simd<T, N>`: Binds variables directly to hardware registers in the General Register File (GRF).
- `esimd::block_load_2d`: Directly generates hardware 2D block read messages.
- `esimd::xmx::dpas`: Directly issues Dot Product Accumulate Systolic instructions to XMX hardware.

### Pattern 5: On-Device MoE Routing & Expert GEMV Fusion
- In standard MoE runtimes, the GPU router copies top-$K$ expert IDs back to host CPU, triggers a blocking `stream->wait()`, and loops from the CPU to launch individual tiny GEMVs ($84+$ blocking host waits per token).
- **The Megakernel Pattern:**
  1. The router computes routing logits on GPU and writes top-$K$ indices into device VRAM/SLM.
  2. The same kernel immediately loops over the top-$K$ experts **on-device**, executing Gate, Up, SwiGLU, and Down projections in SLM without returning to host CPU.
  3. Enables Level Zero command graph capture (`zeCommandListCreate`).

### Pattern 6: Fused Hybrid SSM / Mamba2 Blocks
- Models like Nemotron-3.5 execute 46 Mamba2 layers with 16–18 separate operations per layer (736 launches per token).
- **The Megakernel Pattern:**
  1. Load input activation ($d=2688$) into SLM.
  2. Compute Input GEMV ($2688 \to 10304$) into $z, xBC, dt$ in-place via XMX `dpas`.
  3. Causal Conv1D ($d_\text{conv}=4$) updates the ring buffer in SLM/registers with zero DRAM traffic.
  4. Selective scan state update ($d_\text{state}=128$) executes in GRF registers.
  5. SwiGLU gating + Grouped RMSNorm executes in SLM.
  6. Output GEMV ($4096 \to 2688$) accumulates directly to the residual stream.
  7. Eliminates 736 driver dispatches per token!

### Pattern 7: Fused Speculative Draft Heads (MTP / DFlash)
- Speculative decoding draft heads (e.g. MTP4 or DFlash) launch separate micro-graphs per draft step, adding $1.5\text{ ms}$ per draft token ($6\text{ ms}$ total).
- **The Megakernel Pattern:**
  1. Fuse RMSNorm + 1-layer MTP projection + LM Head GEMV + Top-1 sampling into **1 persistent launch**.
  2. The proposed token ID is written directly to the draft sequence buffer in device memory.
  3. Draft iteration latency drops from $1.5\text{ ms} \to \mathbf{0.25\text{ ms}}$.

---

## 5. Precision & Quantization: The Systolic DPAS Alignment Law

The XMX systolic array on Intel Xe2 natively accelerates:
- **`INT4`** (with INT8 / FP16 scaling)
- **`INT8`**
- **`FP16` / `BF16`**

```
NATIVE DPAS PIPELINE (Single-Cycle Systolic Multiply-Accumulate):
[INT4 / INT8 Weights] ──> [XMX Systolic Array (dpas)] ──> [FP32/FP16 Accumulator in GRF]

NON-NATIVE SCALAR UNPACK TRAP (Destroys Megakernel Performance):
[IQ3_XXS / Q3_K] ──> [Vector Engine Unpack Loop] ──> [Convert to FP16] ──> [oneMKL / GEMV]
```

### The Systolic Law for Megakernels:
Never use non-native quantization formats (such as `IQ3_XXS` or `Q3_K`) in persistent megakernel pipelines on Intel Arc B70. Non-native formats lack hardware DPAS support, forcing scalar unpack loops on Vector Engines that stall the entire persistent grid. **Stick strictly to INT4 (GPTQ/AWQ), INT8, or FP16/BF16.**

### Rate-Class Law (measured 2026-09-15 — see `research/2026/xmx-dpas-rate-classes-20260915.md`):

Sub-8-bit **operands** are native, but sub-8-bit does **not** buy extra
throughput. Xe2 DPAS has exactly two integer/float rate classes on B70:

| DPAS class | Rate (measured, ngen DPAS-guaranteed) | Peak @2.8 GHz |
|---|---:|---:|
| BF16/FP16 × same | 170 TFLOPS | 183.5 TFLOPS |
| any ≤8-bit int mix with an 8-bit operand (8×8, 8×4, 8×2, 4×8) | 363 TOPS | 367 TOPS |

- INT4 weights (W4A8) **tie** INT8 in math rate — they are a
  **weight-bandwidth/capacity lever** (≈½ weight bytes → decode win), never
  a compute multiplier. There is **no NVFP4-style sub-byte TOPS doubling** on
  Xe2. ~~(Pure 4×4 geometry is unmeasured and irrelevant for LLM serving.)~~ **CORRECTED 2026-09-29** (register row 7): pure s4×s4 was measured 2026-09-21 — 33.91 vs 16.96 TOPS = **2.00× work per instruction** (issue-limited harness, absolutes are floors; `research/2026/xmx-dpas-rate-classes-20260915.md`, `results/2026/09/b70-improvement-probes-20260920/dpas-rate-probe/`). It remains true that no W4A4 *serving path* exists — the rate class is real, the application is absent (decode is bandwidth-bound).
- **Battlemage has no native FP8 DPAS** — FP8 W8A16 paths dequant to BF16
  and land in the half-rate class. INT4 W4A8 dominates FP8-W8A16 on both
  axes (rate and bytes) on this hardware.
- Consequence for kernel choice: compute-bound prefill gains nothing from
  INT4 vs INT8; bandwidth-bound decode gains, bounded well under 2×
  end-to-end (KV/activations/scales don't shrink).

---

## 6. Multi-GPU Realities: Topology Constraints (premise corrected 2026-10-07, RL-059)

In *Grokking Megakernels* Chapter 9, multi-GPU megakernels are demonstrated across 8xH100 systems connected via 900 GB/s NVLink.

### The B70 Host Topology Reality:
- X570-F host (since 2026-08-27): GPU 0 and GPU 1 both CPU-attached **x8/x8**; direct-peer P2P **copy** VERIFIED ~11–12 GB/s with **ATOMICS absent** (RL-059) (`ADR-0008`; measured H2D ≈ 8.9/7.9 GB/s probe-limited — treat as floors; `lspci` LnkSta misreports these endpoints). Full authority: `research/2026/x570f-dual-b70-pcie-topology-20260830.md`.
- Copy-based cross-card collectives are possible over the peer path; atomics-dependent collectives bounce through host memory or fail.

### The Multi-GPU Rule:
1. **Never attempt an inter-card Tensor-Parallel (TP) Megakernel across the X570-F PCIe link.** (Corrected 2026-10-07, RL-059: peer P2P copy is verified at ~11–12 GB/s with ATOMICS absent; the retired B450-F's "PCIe Gen3 x4 / 3.6 GB/s" figure is not this host.) A persistent grid needs fine-grained synchronization a copy-only, atomics-absent link cannot provide, and every barrier still crosses PCIe.
2. **Deploy Independent Per-Card Megakernel Engines:**
   - GPU 0 (`ZE_AFFINITY_MASK=0`): Pinned persistent megakernel pipeline.
   - GPU 1 (`ZE_AFFINITY_MASK=1`): Independent pinned persistent megakernel pipeline.
   - Handles independent concurrent user streams with zero inter-card communication overhead.

---

## 7. Comprehensive Model Diagnostic & Optimization Map

| Target Model | Architecture Class | Primary Bottleneck | Optimization Action | Authority Document |
|---|---|---|---|---|
| **DeepSeek-V4 REAP-K216** | Pure MoE (216 experts) | 84+ blocking host D2H syncs per token | On-device router + Fused `MUL_MAT_ID` | [`research/2026/dsv4-k216-request-to-output-pipeline-20260825.md`](../../research/2026/dsv4-k216-request-to-output-pipeline-20260825.md) |
| **Nemotron-3.5-30B** | Hybrid Mamba2 + MoE (128 experts) | 736 Mamba2 kernel dispatches per token | Fused Mamba2 SSD block in GRF/SLM | [`models/nemotron/nemotron-3.5-lightning-30b-a3b/serving.md`](../../models/nemotron/nemotron-3.5-lightning-30b-a3b/serving.md) |
| **Qwen3.8-27B** | Dense 27B + MTP4 / DFlash | Base decode is memory-bound; MTP draft steps add $6\text{ ms}$ | Fused MTP draft head + W8A16 linear reroute | [`research/2026/qwen38-fp8-tp2-speed-attack-and-ceilings-20260829.md`](../../research/2026/qwen38-fp8-tp2-speed-attack-and-ceilings-20260829.md) |
| **Ornith-1.5 / Qwen3.6** | MoE (A3B) | Gating router host synchronization | Port on-device MoE router to AutoRound INT4 | [`models/ornith/ornith-1.5-35b-a3b/README.md`](../../models/ornith/ornith-1.5-35b-a3b/README.md) |
| **General Strategy** | All Models | Architecture classification | Follow deterministic triage playbook | [`docs/methods/architecture-diagnostic-playbook.md`](../methods/architecture-diagnostic-playbook.md) |
