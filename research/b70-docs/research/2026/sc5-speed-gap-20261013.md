---
title: "SC_5.00bpw_H6 Decode Speed Gap on exl3xpu B70: Root Causes and Levers"
type: research
date: 2026-10-13
status: completed
---

# SC_5.00bpw_H6 Decode Speed Gap on exl3xpu B70: Root Causes and Levers

## 1. Price the Gap: Bytes per Token, Bandwidth Floor, and Removable Overhead

### Artifact Byte Inventory and Active Decode Stream
From `/mnt/<sata-disk>/models-exl3-sc/qwen38-27b-exl3-SC_5.00bpw_H6` and `/mnt/<nvme-disk>/b70-exl3-qwen38-4bpw-20260926`:

- **Total checkpoint size**:
  - SC_5.00bpw_H6: 18,762,684,996 bytes (17.4741 GiB / 18.76 GB)
  - Plain 4.00bpw: 15,722,469,956 bytes (14.6427 GiB / 15.72 GB)
  - Total byte ratio (SC5 / Plain): **1.1934x** (~1.20x)
- **Active decode streamed weights** (excluding `embed_tokens` 2.4134 GiB, which is indexed once per token as a single 10 KiB vector rather than streamed):
  - SC_5.00bpw_H6: 16,219,888,196 bytes (15.1059 GiB / 16.22 GB)
  - Plain 4.00bpw: 13,179,673,156 bytes (12.2745 GiB / 13.18 GB)
  - Active byte ratio (SC5 / Plain): **1.2307x** (~1.23x)

#### Distribution by Bitwidth (K) in Active Streamed Weights
- **SC_5.00bpw_H6** (401 quantized linear tensors + 306 unquantized norms/biases):
  - K=4: 41 tensors, 1.5592 GiB (10.32% of active bytes)
  - K=5: 272 tensors, 10.3454 GiB (68.49% of active bytes)
  - K=6: 66 tensors, 3.0619 GiB (20.27% of active bytes)
  - K=7: 22 tensors, 0.0942 GiB (0.62% of active bytes)
  - Unquantized (RMSNorm, bias): 306 tensors, 0.0452 GiB (0.30% of active bytes)
- **Plain 4.00bpw** (401 quantized linear tensors + 306 unquantized norms/biases):
  - K=4: 400 tensors, 11.3408 GiB (92.39% of active bytes)
  - K=6: 1 tensor (`lm_head`), 0.8885 GiB (7.24% of active bytes)
  - Unquantized: 306 tensors, 0.0452 GiB (0.37% of active bytes)

### Bandwidth Floor vs Measured Performance
Intel Arc Pro B70 has 608 GB/s peak theoretical memory bandwidth. Achievable sustained streaming bandwidth under full-model execution on this hardware is 430–450 GB/s (with ~500 GB/s isolated raw GEMV ceiling).

- **Theoretical bandwidth floor (Plain bytes / SC5 bytes)**:
  - Based on checkpoint byte ratio: $1 / 1.1934 = \mathbf{0.8380\text{x}}$ (~0.838x)
  - Based on active streamed bytes: $1 / 1.2307 = \mathbf{0.8126\text{x}}$ (~0.813x)
- **No-spec decode (M=1, greedy C1, RL-182)**:
  - Plain 4.00bpw: 32.58 tok/s (achieved streaming rate: 429.4 GB/s)
  - SC_5.00bpw_H6: 27.17 tok/s (achieved streaming rate: 440.7 GB/s)
  - Measured speed ratio: $27.17 / 32.58 = \mathbf{0.8339\text{x}}$ (~0.834x)
  - **Conclusion at M=1**: SC5/plain speed ratio (0.834x) is 99.5 % of the *checkpoint-size* ratio (0.838x) and 102.6 % of the *active-byte* ratio (0.813x). At no-spec there is no unexplained SC5-vs-plain gap.
  - **Correction 2026-10-15 (external audit, RL-216):** this is a relative ratio, not bandwidth saturation. Bytes x rate gives a model-derived effective stream of 429.4 / 440.7 GB/s = 71-72 % of the 608 GB/s pin rate (86-88 % of the ~500 GB/s isolated GEMV ceiling). It is not a DRAM-counter measurement. Perfect pin bandwidth would bound plain no-spec at 46.1 tok/s and SC5 at 37.5 tok/s (weight bytes only), ~1.4x above the observed rates before KV, state and overhead.

### MTP3 Speculative Decoding Ratios (Evidence RL-182, RL-185, RL-186)
In MTP3, 3 draft tokens are proposed sequentially ($M=1$ draft forward), followed by 1 batched verification round across the target model ($M=4$ verify forward).

- **Before dispatch fix (image `:20261012`, RL-182)**:
  - Plain MTP3: 81.8 tok/s (post-first)
  - SC5 MTP3: 57.5 tok/s (post-first)
  - Ratio: **0.703x** (a 13% penalty below the 0.83x byte floor).
- **After dispatch fix (image `:20261012c`, RL-185)**:
  - Odd-K (3, 5, 7) and tiny-N K6 at $M \le 4$ rerouted to GEMV MR=4.
  - Plain MTP3: 83.7 tok/s
  - SC5 MTP3: 70.3 tok/s (+23.3% speedup)
  - Ratio: **0.840x** (reaches the 0.838x byte floor).
- **Recipe sampling (T0.7, top_p 0.8, pres 1.5, RL-186)**:
  - Plain MTP3: 63.56 tok/s (p512), 64.34 tok/s (p8192)
  - SC5 MTP3: 56.62 tok/s (p512), 56.82 tok/s (p8192)
  - Ratio: **0.891x** (p512), **0.883x** (p8192).
  - Note: Under sampling, SC5 achieves higher acceptance per round (1.5–1.9 vs 1.4–1.8 drafts accepted) due to lower logit divergence (KL 0.42x prose / 0.16x chat vs BF16), beating the byte bandwidth floor.
- **Single-cell greedy outlier (RL-186)**:
  - Plain: 85.4 tok/s vs SC5: 58.1 tok/s (**0.680x**), reflecting low draft acceptance on that specific cell without warm DVFS ramp.

### Attribution of Round Wall Time and Removable Share
From replay-true attribution in RL-185b on tag c (`mixedk_20261012c`), one verify round takes **46.7 ms wall time**:

| Component | Execution Mode | Calls/Round | Duration (ms) | Share (%) |
|---|---|---:|---:|---:|
| Target verify forward ($M=4$ exl3) | Captured (XPUGraph) | 296 | 36.85 | 78.9% |
| Target `lm_head` + tail calls ($M=4$) | Eager | ~6 | 1.50 | 3.2% |
| MTP draft forward (3 steps $M=1$, all K4 backbone) | Eager | ~19 | 7.80 | 16.7% |
| CPU / graph replay gap / sampler | Host / Eager | — | 0.55 | 1.2% |
| **Total Verify Round** | | | **46.70** | 100.0% |

#### Sizing the Removable Share Above the Hardware Ceiling
1. **Target verify forward**:
   - Streamed weight bytes: 15.42 GB per $M=4$ round.
   - At the raw kernel ceiling (~500 GB/s measured on isolated `DpasKernel`/`GemvKernel` without framework glue): $15.42 \text{ GB} / 500 \text{ GB/s} = \mathbf{30.84\text{ ms}}$.
   - Current captured time is 36.85 ms (418 GB/s blended rate).
   - Verify gap above ceiling: $36.85 - 30.84 = \mathbf{6.01\text{ ms}}$.
2. **Sources of the 6.01 ms verify gap**:
   - Tiny-N starvation ($N=1024$ on `k_proj` and `v_proj`, 32 calls of $5120\times 1024$ K=6/7, 66–76 GB/s, taking 2.05 ms per round; only 64 threads launched across 512 HW threads).
   - Redundant launch and Hadamard overhead in mixed-K fused modules (16 full-attention layers compute `had_in` 3 times per layer).
3. **Draft forward (7.8 ms eager)**:
   - Evaluates K=4 backbone + K=6 draft head (0.58 ms x 3 = 1.74 ms for logits).
   - Attempts to graph-capture the draft failed (RL-187 piecewise capture regressed by 4–9%; RL-188 V2 FULL capture regressed by 13% at p8192).
   - Pruning draft head from K=6 to K=4 recovers ~0.6 ms.
4. **Total removable share**:
   - Removable from verify forward: ~2.5–3.0 ms
   - Removable from draft forward: ~0.6 ms
   - Removable from eager tails: ~0.8 ms
   - **Total removable duration**: **~4.0–4.5 ms** out of 46.7 ms (**~8.6–9.6% of round wall time**).
   - This would raise greedy decode from 70.3 tok/s to ~77–78 tok/s (**~0.92–0.93x of plain 4.00bpw**).

---

## 2. Per-K Kernel Path at M=1 and M=4 in `csrc/exl3_ops.sycl`

### Dispatch Structure in `exl3_gemm_small`
`exl3_gemm_small` governs small-batch execution. In image `:20261012c`:
```cpp
bool use_vec = M <= g_vec_max_m; // g_vec_max_m = 2
// RL-185 reroute:
if (!use_vec && M <= 4 && ((K & 1) || (K == 6 && N <= 2048))) use_vec = true;
```
If `use_vec == true`, dispatch goes to `launch_gemv<K, 2, MR, nt_of(K, MR)>`.
If `use_vec == false`, dispatch goes to `launch_dpas<K, 2, MB, dpas_nt(MB)>`.

### Detailed Per-K Execution

#### 1. At $M=1$ (No-spec decode and sequential draft proposals)
`M <= g_vec_max_m` is `true` for all K. Every op routes to `GemvKernel` with $MR=1$:
- **K=4**: `nt_of(4, 1) = 8` (128 columns per thread). $V=8, G=4, TP=8, GV=32$. Contiguous aligned reads, highly tuned. Achieves 464–542 GB/s.
- **K=5**: `nt_of(5, 1) = 4` (64 columns per thread). $V=32, G=8, TP=4, GV=32$. Loops 32 unrolled steps (`u < 32`). Achieves 446–460 GB/s on wide matrices.
- **K=6**: `nt_of(6, 1) = 4` (64 columns per thread). $V=16, G=16, TP=2, D=3, GV=32$. Planar de-interleaving (`PLANAR=true`). Achieves 474–493 GB/s.
- **K=7**: `nt_of(7, 1) = 4` (64 columns per thread). $V=32, G=8, TP=4, GV=32$. Loops 32 unrolled steps. Achieves 478–522 GB/s.

#### 2. At $M=4$ (MTP3 Target Verification)
- **K=4**:
  - `(4 & 1) == 0` and `4 != 6`, so `use_vec = false`.
  - Routes to `dispatch_dpas<4, 2>(MB=8)`.
  - Uses the hand-tuned `build_k4` in `exl3_esimd.h`: packs VNNI B operands using contiguous 32-lane writes and no scatter moves.
  - Achieves 371–510 GB/s (padding $M=4$ to $MB=8$).
- **K=5**:
  - `(5 & 1) == 1`, so `use_vec = true`.
  - Routes to `dispatch_mr<5, 2>(MR=4)`.
  - `GemvKernel<5, 2, 4, 4>` executes with accumulator $ACC = MR \times NP \times 2 \times GV = 4 \times 1 \times 2 \times 32 = 256$ fp16 elements.
  - Achieves 399–458 GB/s (up from 205–325 GB/s under DPAS).
- **K=6**:
  - If $N \le 2048$ (attention `k_proj`, `v_proj` where $N=1024$): `use_vec = true`. Routes to `GemvKernel<6, 2, 4, 4>`.
  - If $N > 2048$ (e.g. $17408\times 5120$ down-projector): `use_vec = false`. Routes to `DpasKernel<6, 2, 8, 2>` (`PLANAR=true`). Achieves 438–501 GB/s.
- **K=7**:
  - `(7 & 1) == 1`, so `use_vec = true`. Routes to `GemvKernel<7, 2, 4, 4>`.
  - Used only on $N=1024$ `k_proj` and `v_proj`. Severe launch-bound bottleneck: achieves only 133–152 GB/s.

### Missing Tuned Paths
1. **DPAS operand builder (`build<0>`) lacks tuned implementations for K=5 and K=7**:
   - In `exl3_esimd.h:757`, only $K=4$ has a specialized builder (`build_k4`).
   - For $K \in \{5, 7\}$, DPAS falls back to recursive template `build<0>` ($u < 32$), which issues 32 per-lane scatter writes. This was measured in RL-182 to take ~80% of tile compute time at $M \le 4$, crippling DPAS to 205–325 GB/s. RL-185 bypassed this by routing odd-K to GEMV.
2. **GEMV MR=4 lacks high-occupancy split-K for narrow matrices ($N=1024$)**:
   - For $N=1024, NT=4$, there are only $1024 / (16 \times 4) = 16$ column strips.
   - The thread target calculation clamps splits $P$ to 4, launching only $16 \times 4 = 64$ hardware threads across B70's 512 hardware execution threads. The card is 87.5% idle during these 32 calls.

### Fused QKV Mixed-K Handling
In `exl3xpu/vllm_plugin.py` (`0002-vllm-plugin-mixed-k.patch`):
- All 16 full self-attention layers in Qwen3.8-27B have mixed bitwidths across projection members:
  - `q_proj`: K=4 or 5 ($N=12288$)
  - `k_proj`: K=6 or 7 ($N=1024$)
  - `v_proj`: K=6 or 7 ($N=1024$)
- Because `q_proj`, `k_proj`, and `v_proj` differ in K, `layer.exl3_mixed` is set to `True`.
- In `apply()`, execution splits into 3 separate member runs: `layer.exl3_runs = [[0], [1], [2]]`.
- For each member, `torch.ops.exl3xpu_C.linear(...)` is invoked separately, writing into `y2[:, a:b]`:
  1. Each invocation launches `launch_had_in` to sign-flip and block-pad activation `x2`. For 16 layers, `had_in` is redundantly computed 3 times per layer (48 times instead of 16).
  2. Each invocation launches GEMV/DPAS kernel.
  3. Each invocation launches `launch_had_out`.
  4. Each invocation performs a PyTorch slice copy into the output buffer.
- This creates **32 unnecessary linear calls** (96 kernel launches) per verify round.
- Critically, `k_proj` and `v_proj` share **identical K** (both K=6 or both K=7) and identical $N=1024$. The plugin fails to merge them into a single joint $N=2048$ call.

### MTP Draft Head Precision
- In `config.json`, `"mtp_bits": 4`, `"head_bits": 6`.
- The MTP draft backbone (`mtp.layers.0.*` and `mtp.fc`) is quantized at $K=4$.
- However, Qwen3.8-27B does not train an independent MTP lm_head; it shares the target's `lm_head`.
- In `vllm_plugin.py:382-397` (`_build_draft_head`), the draft head slices 512 blocks (65,536 vocabulary tokens) directly from `lm_head`, inheriting `lm.exl3_K = 6` (**6 bpw mul1 codebook**).
- During each draft step, logits are computed eager at $M=1$ with this K=6 head ($N=65536, K_{\text{dim}}=5120$).
- This consumes **0.58 ms per draft step** (1.74 ms across the 3 draft steps), running at 6 bpw rather than 4 bpw.

---

## 3. Ranked Candidate Fixes with Predicted End-to-End Gain and Safe Screen

All candidate fixes comply with AGENTS.md §7: evaluations must occur under active XPU graph capture (`cudagraph_mode=FULL_DECODE_ONLY`).

### Rank 1: Co-merge K and V Projections in Mixed-K QKV (`vllm_plugin.py`)
- **Mechanism**: In `vllm_plugin.py:170-205`, group adjacent checkpoint members that share identical bitwidth ($K_i = K_{i+1}$) into a single run. For `qkv_proj` where `part_Ks` is `[5, 7, 7]` or `[4, 6, 6]`, set `layer.exl3_runs = [[0], [1, 2]]` instead of `[[0], [1], [2]]`.
- **Why it pays**:
  - Eliminates 16 linear calls (48 kernel launches) per verify round.
  - Merges two starved $N=1024$ calls into one $N=2048$ call, doubling strip count from 16 to 32 and lifting throughput from ~133 GB/s toward ~250 GB/s.
  - Pure Python plugin change; zero kernel changes. Slices into `y2` remain bit-exact.
- **Predicted gain**: **1.0–1.5 ms per verify round** (~2.1–3.2% decode speedup; lifts greedy from 70.3 to ~72.2 tok/s).
- **Smallest graph-replay-safe screen**:
  1. Correctness: `python3 tests/check_mixedk.py` (rel-L2 $\le 2\times 10^{-2}$).
  2. Replay screen: Greedy $p512/g128, n=3$ sequential screen under `FULL_DECODE_ONLY` on GPU1. Verify identical output canaries (6/6).

### Rank 2: Occupancy Tuning for Narrow $N \le 2048$ GEMV at $M=4$ (`csrc/exl3_ops.sycl`)
- **Mechanism**: In `exl3_gemm_small` (lines 297–304), for $M=4$ and $N \le 2048$, override the split-K sizing `P` to increase thread parallelism:
  ```cpp
  if (use_vec && M == 4 && N <= 2048) {
      P = std::max(1, std::min(tk, 16)); // force 16 splits instead of 4
  }
  ```
  This increases total threads from $16 \times 4 = 64$ to $16 \times 16 = 256$, saturating 16 of B70's 32 Xe cores instead of only 4.
- **Why it pays**:
  - Directly accelerates the 32 latency-bound $5120\times 1024$ calls (which currently consume 2.05 ms per round at 66–76 GB/s).
- **Predicted gain**: **0.8–1.2 ms per verify round** (~1.7–2.5% decode speedup).
- **Smallest graph-replay-safe screen**:
  1. Microbench: Standalone JIT benchmark on $5120\times 1024$ K=6/7 at $M=4$ verifying no split-K reduction regression.
  2. Graph screen: 3-rep greedy $p512/g128$ screen under graph replay.

### Rank 3: Shared `had_in` for Mixed-K Members (`vllm_plugin.py` / `exl3_gemm_small`)
- **Mechanism**: Expose a two-stage C++ API `exl3_had_in(x, suh) -> xh` and `exl3_gemm_with_xh(xh, trellis, ...)`. In `vllm_plugin.py:apply`, for mixed-K modules, run `had_in` once on input $x$ and reuse the blocked Hadamard activations `xh` across the 2 or 3 member GEMV calls.
- **Why it pays**:
  - Eliminates 32 redundant Hadamard transformations per verify round.
- **Predicted gain**: **0.5–0.7 ms per verify round** (~1.1–1.5% decode speedup).
- **Smallest graph-replay-safe screen**:
  1. Graph capture test: confirm `xh` intermediate tensor allocated during warmup remains valid and capture-safe.
  2. 3-rep paired screen vs tag c baseline.

### Rank 4: Quantize MTP Draft Head at K=4 (`convert_mtp.py` / artifact export)
- **Mechanism**: Re-quantize the 65,536-token shortlist draft head tensor at 4 bpw mul1 (matching `mtp_bits: 4`) instead of inheriting 6 bpw from target `lm_head`.
- **Why it pays**:
  - Cuts draft head streaming volume by 33.3% ($6 \to 4$ bpw).
  - Lowers eager draft logit latency from 0.58 ms to ~0.39 ms per step (saving 0.57 ms across 3 draft steps).
  - Draft head only proposes candidate tokens; target `lm_head` (K=6) performs verification, so generation quality is fully preserved provided acceptance rate is unchanged.
- **Predicted gain**: **~0.6 ms per round** (~1.3% decode speedup).
- **Smallest graph-replay-safe screen**:
  1. Canary battery (6/6 exact match).
  2. Acceptance rate check: ensure accepted tokens per round does not drop below 1.5.

---

## 4. Summary Table of Levers

| Lever | Location | Mechanism | Predicted Gain (ms/rd) | Predicted E2E Gain | Replay-Safe Gate |
|---|---|---|---:|---:|---|
| **1. Merge K+V in QKV** | `vllm_plugin.py` | Co-launch identical-K members ($N=1024 \to 2048$) | 1.0–1.5 ms | +2.1–3.2% | Python-only, `check_mixedk.py` |
| **2. Split-K for narrow N** | `csrc/exl3_ops.sycl` | Increase $P \to 16$ at $M=4, N \le 2048$ | 0.8–1.2 ms | +1.7–2.5% | JIT microbench + p512 screen |
| **3. Shared `had_in`** | `vllm_plugin.py` + C++ | Compute input Hadamard once per fused layer | 0.5–0.7 ms | +1.1–1.5% | Intermediate buffer capture test |
| **4. K=4 draft head** | Model artifact | Quantize draft head shortlist to 4 bpw | ~0.6 ms | ~+1.3% | Acceptance rate $\ge 1.5$ + canaries |
| **Combined Levers 1–4** | Stack | Simultaneous execution of all 4 levers | **~3.2–4.0 ms** | **+7.0–9.0%** | Full recipe A/B (5 pairs, 230 W) |
