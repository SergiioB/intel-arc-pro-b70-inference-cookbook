---
title: "Megakernel & Persistent Fusion Implementation Roadmap: Host Production Target Guide"
type: research
date: 2026-08-29
status: final
---

# Megakernel & Persistent Fusion Implementation Roadmap: Host Production Target Guide

**Date:** 2026-08-29  
**Status:** Actionable Engineering Roadmap  
**Applies to:** Intel Arc Pro B70 (Xe2 Battlemage, 32 Xe-cores, 32 GB GDDR6), SYCL, ESIMD, llama.cpp SYCL, vLLM XPU

---

## 1. Executive Summary & Model Target Focus

We focus strictly on the production and research targets deployed on this host:

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │      Host Model Targets for Persistent Fusion          │
                                  └───────────────────────────┬────────────────────────────┘
                                                              │
                 ┌────────────────────────────────────────────┴────────────────────────────────────────────┐
                 │                                                                                         │
                 ▼                                                                                         ▼
┌─────────────────────────────────────────────────────────┐               ┌─────────────────────────────────────────────────────────┐
│ Track 1: Pure & Hybrid MoE Architectures                │               │ Track 2: Large Dense + Speculative (MTP/DFlash)         │
│ • DeepSeek-V4 REAP-K216 (43 layers, 216 experts, top-6) │               │ • Qwen3.8-27B (27B dense, GPTQ INT4 / FP8, MTP4)        │
│ • Nemotron-3.5-Lightning-30B-A3B (46 Mamba2 + 7 MoE)    │               │                                                         │
│ • Ornith-1.5-35B-A3B & Qwen3.6-35B-A3B                  │               │                                                         │
├─────────────────────────────────────────────────────────┤               ├─────────────────────────────────────────────────────────┤
│ Bottleneck: 84+ D2H syncs, 700+ tiny GEMVs, 736 Mamba2  │               │ Bottleneck: Base model is GDDR6-bandwidth bound; MTP    │
│ kernel dispatches per token                             │               │ draft step micro-graphs add 1.5ms per draft token       │
│                                                         │               │                                                         │
│ Solution: On-device MoE router + Fused Mamba2 SSD block │               │ Solution: Fused MTP Draft Step (Norm + GEMV + Top-1)    │
│ Target: 2x to 3x decode throughput gain                 │               │ Target: Lift from 102.6 tok/s -> 120-125+ tok/s         │
└─────────────────────────────────────────────────────────┘               └─────────────────────────────────────────────────────────┘
```

---

## 2. Track 1A: Nemotron-3.5-Lightning-30B-A3B (Hybrid Mamba2-Transformer MoE)

### Architecture Profile
- **Total Parameters:** 32.91B (23.68 GiB Q4_K_M / GPTQ INT4)
- **Active Parameters per token:** ~3.5B (A3B)
- **Structure (53 total blocks):**
  - **46 Mamba2/SSM layers:** 16–18 individual kernel dispatches per layer (input proj $\to$ split $\to$ causal conv1d $\to$ SiLU $\to$ SSM selective scan $\to$ skip connection $\to$ SwiGLU gating $\to$ grouped RMSNorm $\to$ output proj).
    - **Dead overhead:** $46\text{ layers} \times 16\text{ dispatches} = \mathbf{736\text{ kernel launches per token}}$ purely in Mamba blocks!
  - **7 Attention + MoE layers:** 128 experts total (6 routed + 1 shared expert active per layer).
  - **Speculative Layer:** DFlash / MTP draft head.

### The Megakernel Intervention for Nemotron-3.5

#### Step 1: Fused Mamba2 SSM Block Kernel (`fused_mamba2_ssd_xe2`)
In `vllm-xpu-kernels` / `ggml-sycl`:
- Fuse all 16 operations of the Mamba2 block into a single persistent work-group launch per layer:
  1. Input activation ($d=2688$) is loaded into SLM once.
  2. Input GEMV ($2688 \to 10304$) computes $z, xBC, dt$ in-place using XMX `dpas`.
  3. Depthwise causal conv ($d_\text{conv}=4$) updates the ring buffer in SLM/registers with zero VRAM traffic.
  4. Selective scan state update ($d_\text{state}=128$, $n_\text{group}=8$) executes in GRF registers.
  5. SwiGLU gating + Grouped RMSNorm executes in SLM.
  6. Output GEMV ($4096 \to 2688$) accumulates directly to the residual stream.
- **Impact:** Eliminates 736 driver dispatches per token; keeps SSM state in on-chip SRAM.

#### Step 2: On-Device Router for the 7 MoE Layers
- Gating router evaluates 128 experts on GPU; selects top-6 routed + 1 shared expert.
- Expert GEMVs execute on-device without copying expert IDs back to host DDR4.

### Benchmark & Validation Plan
```bash
# Verify vLLM XPU Nemotron-3.5 with fused Mamba2 + DFlash
python3 -m vllm.entrypoints.openai.api_server \
  --model /mnt/<models-disk>/hf/nemotron-3.5-30b-gptq \
  --speculative-config '{"method":"dflash"}' \
  --gpu-memory-utilization 0.90 \
  --port 8000 &

# Measure C1 post-first latency (n=5)
python3 scripts/tmp/b70-nemotron-prefill-budget-client.py --port 8000 --n-samples 5
```
- **Target Gate:** Push Nemotron-3.5 C1 post-first decode from **$186.6\text{ tok/s} \to \mathbf{220\text{–}250+\text{ tok/s}}$**.

---

## 3. Track 1B: DeepSeek-V4-Flash REAP-K216 (Large 216-Expert MoE)

### Architecture Profile
- **Total Parameters:** 43 layers, 216 routed experts + 1 shared expert per layer.
- **Active per token:** 6 routed experts + 1 shared expert.
- **Current Bottleneck (`research/2026/dsv4-k216-request-to-output-pipeline-20260825.md`):**
  - Gating router computes 216 logits on GPU.
  - Runtime D2H-copies 24 bytes of expert IDs back to CPU and triggers a blocking `stream->wait()`.
  - Host CPU executes a serial loop to launch 6 individual tiny GEMVs per projection ($43 \times 2 = 86$ host waits per token on GPU-routed layers).
  - Current decode speed is throttled at **$9.25\text{ tok/s}$ clean / $9.58\text{ tok/s}$ patched** (re-measured 2026-09-21; 8.87 superseded).

### The Megakernel Intervention for DeepSeek-V4

#### Step 1: On-Device Gating & Fused MoE Dispatch in `ggml-sycl.cpp`
- Eliminate D2H copy of `top_k_indices`.
- Implement `mul_mat_id_fused_moe_sycl`:
  ```cpp
  // Keep expert IDs on device and execute expert GEMVs inside GPU work-groups
  void mul_mat_id_fused_moe_sycl(
      sycl::queue &q,
      const half *act_in,
      const void *gate_weights,
      const void *up_weights,
      const void *down_weights,
      const int32_t *dev_topk_ids,
      const float *dev_topk_weights,
      half *act_out,
      int hidden_dim, int ffn_dim, int top_k)
  {
      q.submit([&] (sycl::handler &cgh) {
          sycl::local_accessor<half, 1> slm_buf(sycl::range<1>(hidden_dim + ffn_dim), cgh);
          cgh.parallel_for(sycl::nd_range<1>(32 * 256, 256), [=] (sycl::nd_item<1> item) {
              // Iterate through top-6 experts directly on-device
              for (int k = 0; k < top_k; ++k) {
                  int exp_id = dev_topk_ids[k];
                  float exp_w = dev_topk_weights[k];
                  // Fused Gate + Up GEMV -> SwiGLU -> Down GEMV in SLM
              }
          });
      });
  }
  ```
- **Impact:** Eliminates all 84+ blocking host roundtrips.

#### Step 2: Enable SYCL Command Graph Capture
- Set `GGML_SYCL_GRAPH=1` in llama.cpp to submit the entire multi-layer decode graph in a single Level Zero command list (`zeCommandListCreate`).

### Benchmark & Validation Plan
```bash
# Build llama.cpp SYCL with on-device MoE patch
source /opt/intel/oneapi/setvars.sh --force
cmake -B build-sycl-moe -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl-moe -j $(nproc)

# Test C1 decode speed (n=5)
./build-sycl-moe/bin/llama-cli \
  -m /mnt/<models-disk>/hf/DeepSeek-V4-Flash-0731-REAP-K216-UD-IQ3_XXS.gguf \
  -p "Explain persistent fusion for MoE on Intel Arc." \
  -n 128 -c 2048 -ngl 99 -ncmoe 0 -fa on -ctk q8_0 -ctv q4_1 -t 8
```
- **Target Gate:** Median decode speed must jump from **$9.25\text{–}9.58\text{ tok/s} \to \mathbf{18\text{–}25+\text{ tok/s}}$** (`recompute-required`: was $8.87 \to 18\text{–}25+$ off the superseded baseline; re-derive the multiplier from the 9.25/9.58 re-measure, and note the device-ID repair alone measured +3.54 %, ≥10 % gate FAILED).

---

## 4. Track 2: Qwen3.8-27B (Dense + MTP4 Speculative Decoding)

### Architecture Profile
- **Base Model:** 27B dense transformer (GPTQ INT4 G128 / native FP8).
- **Decode Reality:**
  - Base decode is $98.4\%$ memory-bandwidth bound ($14\text{ GB}$ weights / token $\implies 31.1\text{ ms / token} \approx 32.9\text{ tok/s}$).
  - Base GEMV megakernel only saves $0.5\text{ ms}$ ($+1.5\%$).
- **The True Lever:** **Multi-Token Prediction (MTP4)**, which runs 4 draft iterations per target verification step.

### The Megakernel Intervention for Qwen3.8-27B

#### Step 1: Fused MTP Draft Step Kernel (`fused_mtp_step_xpu`)
In `vllm-xpu-kernels`:
- The current draft pipeline executes separate launches for `RMSNorm` $\to$ `Gemv MTP Layer` $\to$ `Sampling Argmax` ($1.5\text{ ms}$ per draft token $\times 4 = 6.0\text{ ms}$).
- Fuse the entire draft step into a single persistent kernel:
  1. RMSNorm computed in SLM.
  2. 1-layer MTP projection computed via XMX `dpas`.
  3. LM Head projection + top-p/argmax sampling writing the token ID directly to the draft sequence buffer in device memory.
- Execution time drops from $1.5\text{ ms} \to \mathbf{0.25\text{ ms}}$ per draft token ($1.0\text{ ms}$ total for 4 tokens).

#### Step 2: Compile `vllm-xpu-kernels` with Explicit Launch Bounds
```bash
cd vllm-xpu-kernels
export VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_full.conf
export VLLM_PAGED_DECODE_CONFIG=paged_decode_full.conf
export CXXFLAGS="-O3 -fsycl -fsycl-targets=spir64_gen -Xs '-device bmg' -Xs '-options -cl-intel-reqd-sub-group-size=16'"
pip install . --no-build-isolation
```

### Benchmark & Validation Plan
```bash
# Benchmark Qwen3.8-27B with Fused MTP Head
python3 -m vllm.entrypoints.openai.api_server \
  --model /mnt/<models-disk>/hf/qw38-gptq \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --gpu-memory-utilization 0.90 \
  --port 8000 &

python3 scripts/tmp/qw38-natural-content-bench.py --port 8000 --n-samples 5
```
- **Target Gate:** Lift C1 median post-first throughput from **$102.61\text{ tok/s} \to \mathbf{120\text{–}125+\text{ tok/s}}$**.

---

## 5. Execution Matrix & Priority Table

| Priority | Target Model | Architecture Class | Intervention | Primary Bottleneck Removed | Expected Performance Target |
|:---:|---|---|---|---|:---:|
| **P1** | **DeepSeek-V4 REAP-K216** | Pure MoE (216 experts) | On-device router + `MUL_MAT_ID` | 84+ blocking host D2H syncs/token | **$9.25\text{–}9.58 \to 20+\text{ tok/s}$** (`recompute-required`: was $8.87 \to 20+$ / $2.3\times$ off the superseded baseline — re-derive from the 9.25/9.58 re-measure; the ≥10 % device-ID gate FAILED so the 2.3× premise needs re-basing on the fused-IQ3 kernel, not the sync repair) |
| **P2** | **Nemotron-3.5-Lightning** | Mamba2 + MoE (128 experts) | Fused Mamba2 SSD block + On-device MoE router | 736 Mamba launches + MoE host waits | **$186.6 \to 230+\text{ tok/s}$ ($+25\%$)** |
| **P3** | **Qwen3.8-27B** | Dense + MTP4 | Fused MTP Draft Step + Xe2 launch bounds | 6 ms MTP draft sequence overhead | **$102.6 \to 120+\text{ tok/s}$ ($+17\%$)** |
| **P4** | **Ornith-1.5 / Qwen3.6** | MoE (A3B) | Port Track 1A fused router to AutoRound INT4 | Expert routing host synchronization | **$45 \to 65+\text{ tok/s}$ ($+40\%$)** |
