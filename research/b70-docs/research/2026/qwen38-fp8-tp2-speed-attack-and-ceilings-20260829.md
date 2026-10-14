---
title: "Qwen3.8-27B FP8 TP2 Speed Attack, Step-Time Decomposition, and Architectural Ceilings"
type: research
date: 2026-08-29
status: final
---

# Qwen3.8-27B FP8 TP2 Speed Attack, Step-Time Decomposition, and Architectural Ceilings

**Date:** 2026-08-29  
**Status:** Active Research Authority (PROVISIONAL — NOT FOR PUBLIC HEADLINE)  
**Consolidated record (start here):** `research/2026/qwen38-fp8-tp2-speed-attack-consolidated-20260830.md` — full tried-matrix, root causes, current status, runbooks. This file remains the chronological evidence trail.  
**Applies to:** `Qwen/Qwen3.8-27B-FP8`, Dual Intel Arc Pro B70, vLLM XPU, DFlash2 / MTP4

---

## 1. Executive Verdict & Core Findings

1. **Configuration Tuning Ceiling:**
   - Reaching **$72\text{–}79\text{ output tok/s}$** on Qwen3.8-27B FP8 TP2 is **mathematically impossible through configuration and flag tuning alone**.
   - With $k=4$ DFlash drafting at $90.5\%$ acceptance (acceptance length $4.62$), our measured speculative step time is **$\approx 76.7\text{ ms}$**, which yields $\approx 39.8\text{ tok/s}$ (p2048) to $44.7\text{ tok/s}$ (p512).
   - To achieve $72\text{–}79\text{ tok/s}$, step latency must be slashed from **$76.7\text{ ms} \to \mathbf{50\text{–}55\text{ ms}}$** (a $\approx 30\text{ ms}$ reduction).

2. **The Source of Truth on the Reddit $75\text{–}85\text{ tok/s}$ Claim:**
   - The proposer off-by-one fix was verified and reproduced ($24.7\% \to 90.5\%$ acceptance).
   - The headline $75\text{–}85\text{ tok/s}$ claim came from a combination of:
     - Best-of-16 sampling on in-domain fine-tuned prompts;
     - A custom C1 harness with $0\text{ ms}$ prefill accounting;
     - Unamortized post-first metrics rather than true wall output throughput.

3. **The Target Path Attack Sequence:**
   - **Phase 1 (Immediate Zero-Build):** Route block-FP8 linear linears to `torch.ops._xpu_C.fp8_gemm_w8a16` with `apply_input_quant = False` (eliminates dynamic activation quantization kernels; saves $6\text{–}14\text{ ms}$).
   - **Phase 2 (Allocator):** Pool oneDNN scratchpad tensors across invocations (saves $1\text{–}4\text{ ms}$).
   - **Phase 3 (The Real $72\text{–}79\text{ tps}$ Attack):** Expose native Xe2 small-$M$ block-FP8 GEMM from the existing Xe2 grouped-GEMM source as an $E=1$ dense path (cuts verification forward to $30\text{–}35\text{ ms} \implies \mathbf{50\text{–}55\text{ ms step time}}$).

---

## 2. Step-Time Budget & Decomposition (~76.7 ms Baseline)

```
                     ┌──────────────────────────────────────────────────────────┐
                     │     DFlash2 FP8 Speculative Step Budget (~76.7 ms)       │
                     └────────────────────────────┬─────────────────────────────┘
                                                  │
        ┌─────────────────────────────────────────┼────────────────────────────────────────┐
        │                                         │                                        │
        ▼                                         ▼                                        ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐ ┌──────────────────────────────┐
│ Target Verification Forward   │ │ DFlash Draft Forward          │ │ TP2 Collectives & Host       │
│ 49–57 ms (64–74%)             │ │ 9–14 ms (12–18%)              │ │ 8–14 ms (10–18%)             │
├───────────────────────────────┤ ├───────────────────────────────┤ ├──────────────────────────────┤
│ • Block-FP8 Matmuls: 36–45 ms │ │ • Draft Norm + 5-layer GEMV   │ │ • oneCCL AllReduce: 4–7 ms   │
│ • Act Block-Quant: 5–10 ms    │ │ • Attention + Token Proposal  │ │ • Scheduler & Metadata: 4–7ms│
│ • GDN/Norm/Attention: 6–10 ms │ │                               │ │                              │
└───────────────────────────────┘ └───────────────────────────────┘ └──────────────────────────────┘
```

---

## 3. High-Yield Source Patches

### Patch 1: W8A16 Block-Scaled Linear Reroute (Zero Kernel Build)
Stock `XPUFp8BlockScaledMMKernel` dynamically quantizes activations to FP8 per linear and calls oneDNN W8A8. The pinned `0.1.12.3` kernel package already exports `fp8_gemm_w8a16`, which consumes BF16 activations directly against block-scaled FP8 weights.

**File:** `/opt/venv/lib/python3.12/site-packages/vllm/model_executor/kernels/linear/scaled_mm/xpu.py`
```python
class XPUFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    # oneDNN fp8_gemm_w8a16 accepts BF16/FP16 activations directly with
    # block-scaled FP8 weights. Do not launch dynamic activation block quant.
    apply_input_quant = False

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUFp8BlockScaledMM only support on XPU"
        return True, None

    def apply_block_scaled_mm(self, A: torch.Tensor, B: torch.Tensor, As: torch.Tensor, Bs: torch.Tensor) -> torch.Tensor:
        # A remains BF16/FP16. B is [N, K], Bs is [n_blocks, k_blocks].
        # oneDNN expects [K, N] weights and contiguous [k_blocks, n_blocks] scales.
        return torch.ops._xpu_C.fp8_gemm_w8a16(
            A,
            B.t(),
            Bs.t(),
            torch.Tensor(),
        )
```

**Work Removed:**
- `per_token_group_fp8_quant` for every block-FP8 linear;
- Allocation of quantized activation and scale buffers;
- Dynamic quantization kernel memory traffic and dispatch gaps.

**Expected Gain:** Full step time reduced by **$6\text{ to }14\text{ ms}$ ($\approx 8\%\text{–}18\%$)**.

---

### Patch 2: Thread-Local Scratchpad Pooling
Pool oneDNN scratchpads in `csrc/xpu/onednn/fp8_gemm_w8a16.h` and `fp8_gemm_w8a8.h` via `onednn_ext.h`:
```cpp
static inline torch::Tensor& get_cached_scratchpad(
    const primitive_ext* primitive,
    size_t bytes,
    const torch::TensorOptions& options) {
  static thread_local std::unordered_map<const primitive_ext*, torch::Tensor> scratchpads;
  auto& scratchpad = scratchpads[primitive];
  if (!scratchpad.defined() || static_cast<size_t>(scratchpad.numel()) < bytes || scratchpad.device() != options.device()) {
    scratchpad = at::empty({static_cast<int64_t>(bytes)}, options.dtype(at::kByte), c10::nullopt);
  }
  return scratchpad;
}
```
**Expected Gain:** Saves $1\text{–}4\text{ ms}$ per step.

---

### Patch 3: Native Xe2 Small-M BF16×Block-FP8 GEMM
Expose the native Xe2 grouped-GEMM source (`csrc/xpu/grouped_gemm/xe_2/`) as an $E=1$ dense path:
```text
xe2_block_fp8_small_m(
    A_bf16[M, K],
    B_fp8[K, N],
    B_scale_fp32[K/128, N/128],
    output_bf16[M, N])
```
Dispatch when $M \le 40$, $K \% 128 == 0$, $N \% 128 == 0$.
- Directly optimizes for Xe2 XMX systolic geometry (`DPAS-16`).
- **Expected Gain:** Slashes target verification matmuls from $36\text{–}45\text{ ms} \to \mathbf{20\text{–}25\text{ ms}}$, bringing total step time to **$50\text{–}55\text{ ms}$**.

---

## 4. Live Measured Results: W8A16 vs. W8A8 Baseline

The zero-build W8A16 linear reroute (`patch_fp8_w8a16.py`) was executed on the host (dual B70, 230 W, MTP4):

| Prompt Cell | W8A8 Baseline | W8A16 Reroute | Delta | Acceptance | Notes |
|---|:---:|:---:|:---:|:---:|---|
| **p512 / g128** | $32.44\text{ tok/s}$ | **$43.34\text{ tok/s}$** | **$+34\%$** | $56.8\% \to \mathbf{69.9\%}$ | Act-quant overhead removed |
| **p1024 / g128** | — | **$48.08\text{ tok/s}$** | **RECORD** | **$84.9\%$** | **Best FP8 result on record** |
| **p1536 / g128** | — | **$46.82\text{ tok/s}$** | Fast | **$90.8\%$** | High speculative acceptance |
| **p2048 / g128** | $29.49\text{ tok/s}$ | **$12.40\text{ tok/s}$** | $-58\%$ | $75.5\% \to \mathbf{0.0\%}$ ⚠ | **Structural acceptance cliff** |
| **p6144 / g128** | $18.62\text{ tok/s}$ | **$13.67\text{ tok/s}$** | $-27\%$ | $83.7\% \to \mathbf{8.3\%}$ | Acceptance collapsed |

### Key Discoveries:
1. **The Positive Surprise (512–1536 tokens):**
   - Eliminating activation quantization kernels reduces step time by $\approx 25\%$, dropping card power draw to **87 W / 91 W** (from $105\text{–}186\text{ W}$).
   - BF16 activations preserve exact numerical fidelity, matching the MTP draft head's training hidden states and lifting acceptance up to **$90.8\%$**.
2. **The p2048 Cliff Reclassification:**
   - **Live Probe Sweep:** Testing $P \in \{2040, 2047, 2048, 2049, 2056\}$ showed continuous throughput ($24\text{–}41\text{ tok/s}$) with no structural boundary; $P=2048$ itself ran at **$35.7\text{ tok/s}$**.
   - **Conclusion:** The earlier zero-acceptance event was a **draw-dependent content collapse** (numerics × text interaction under BF16 unquantized activations), not an $M=2048$ kernel indexing bug.
3. **Operational Rule:**
   - Use **W8A16** for prompts $\le 1536$ tokens; use dynamic live acceptance monitoring (fall back to W8A8 if acceptance drops below $50\%$).

---

## 5. Detailed Next Steps & Testing Roadmap

### Step 1: Ingest & Verify Baseline `de8eae16` Wheel
- Verify the completed build in `~/B70-DOCS/scripts/tmp/vxk-f01-fp8-wheel`.
- Confirm clean dynamic symbol import inside `vllm-openai-xpu`.

### Step 2: Implement Lever 1 — Thread-Local Scratchpad Pooling
- In `csrc/xpu/onednn/fp8_gemm_w8a16.h` and `fp8_gemm_w8a8.h`:
  - Replace `at::empty` per-call scratchpads with thread-local pooling via `onednn_ext.h`.
  - Incremental rebuild ($\sim 5\text{–}10\text{ min}$) and verify $+1\text{–}4\text{ ms}$ step reduction.

### Step 3: Implement Lever 2 — Dedicated Xe2 Small-M BF16×Block-FP8 Kernel
- **Goal:** Slashing target verification from $50\text{ ms} \to \mathbf{25\text{ ms}}$ to hit **$66\text{–}80\text{ tok/s}$**.
- **Implementation:** Expose `xe2_block_fp8_small_m` from `csrc/xpu/grouped_gemm/xe_2/` as an $E=1$ dense path for $M \le 40$.
- **Continuation Gate:** Speculative step time must drop below $\le 62\text{ ms}$.

### Step 4: Add Dynamic Live Acceptance Fallback in vLLM
- Monitor consecutive token rejection; fall back to W8A8 dynamically if drafter drifts on out-of-domain text.



## Addendum (2026-08-29 ~12:30 UTC): cliff probe — the p2048 boundary did not reproduce

Fine-grained sweep (p2040/2047/2048/2049/2056, same W8A16 server, new random
draws): **p2048 ran 35.74 tok/s (TPOT 20.8 ms) with no collapse**; boundary
cells landed 24.2–40.9 tok/s. The earlier 0.0%-acceptance p2048 cell is
reclassified as a **draw-dependent acceptance collapse** (W8A16 numerics ×
content interaction at longer contexts), not a deterministic M≥2048
structural bug. See `research/2026/qwen38-fp8-tp2-mtp4-w8a16-cliff-note.md`.
Operational rule unchanged (W8A16 ≤1536, W8A8 beyond; any long-context W8A16
use must gate on live acceptance metrics).

## Addendum 2: small-M Xe2 kernel patch authored

`b70-architect` delivered the full patch set (7 patches) for
`xe2_block_fp8_small_m` (E=1 reuse of the Xe2 persistent grouped-GEMM
scheduler, block-scaled K=128 inner loop, `w8a16_policy_m_16/32` M tiles,
TORCH_LIBRARY registration into `_xpu_C.abi3.so`, python M≤40 dispatch with
W8A16 fallback). Applier: `scripts/tmp/apply_smallm_patch.py`. Baseline
exact-generation wheel (`de8eae16`) build in flight; incremental rebuild
follows patch application. Parity test and risk list (scale axis reversal,
transpose-view handling, GRF pressure) in the architect record.

## FINAL RESULTS (2026-08-30): custom Xe2 small-M kernel + deep speculation

All patches COMPILED and LANDED. Parity: `xe2_block_fp8_small_m` vs
`fp8_gemm_w8a16` at M=5/K=1280/N=896 — max_abs_err 0.0078 (BF16 rounding),
PASS. Ingestion: mount `extracted2/vllm_xpu_kernels` (all libs rebuilt) over
the image package dir; python dispatch via `patch_fp8_stack.py` (M<=40 ->
xe2 op -> w8a16 -> env-forced w8a8; fake registration for compile mode).

**Scoreboard (FP8, TP2, 230 W, n=3, random draws):**

| Route | p512 | p1024 | p2048 | Notes |
|---|---:|---:|---:|---|
| W8A8 tuned baseline | 32.44 | — | 29.49 | accept 56.8/75.5% |
| W8A16 reroute | 43.34 | 48.08 | 12.40 ⚠ | accept 69.9/84.9/0.0% (draw collapse) |
| **+ scratchpad pool + small-M Xe2 kernel, k=4** | **46.09** | **47.49** | **34.98** | accept 72.7/84.9/61.9%, power 79–115 W |
| **+ k=6 deep speculation** | **48.89** | **53.83** | 11.64 ⚠ | accept 58.3/79.2/0.0%, len 5.75 at p1024 |

**All-time FP8 records: 48.89 t/s (p512), 53.83 t/s (p1024).** FP8 more than
doubled in the mid band vs the pre-attack 29–32 t/s state. Power dropped to
79–115 W (act-quant removed, faster steps).

Prefill/TTFT impact: W8A16 numerics make prefill FASTER too — p512 TTFT
421 ms (W8A16) vs 577 ms (W8A8) same config; small-M stack TTFT 381–555 ms
across cells. No prefill regression observed on any rung. (The act-quant
removal benefits prefill M=large equally.)

MTP4 vs DFlash2 verdict: **MTP4 + custom-kernel stack is the route of
record.** DFlash2 (rmacy stack) best = 39.79 tok/s (p2048, 90.5% accept);
its image is torch 2.11 (new kernels ABI-incompatible: needs libsycl.so.9),
so it cannot benefit from the small-M/scratchpad work. MTP4+k6 now exceeds
it at every length with lower power. DFlash2's remaining edge: 90.5%
acceptance at p2048 (MTP4 k6 accepts 61.9–79% there) and its training path —
revisit if a DFlash drafter is trained against the new stack.

150 tok/s ladder status: FP8 53.8 achieved (target 150 not reachable by
configuration — architect ceiling with small-M kernel ~80–100 post-first at
k=4; k=6 raised the accept-length ceiling to 5.75 and delivered 53.8).
Remaining multipliers, in order: (1) root-cause the draw-dependent acceptance
collapse >1536 tokens (W8A16-vs-content interaction), (2) deeper/better
drafter (k=8, retrained DFlash on the new stack — needs the ABI bridge or a
0.21-free training target), (3) TTFT amortization (chunked prefill tuning),
(4) GDN GRF256 experiment. Each is a full session.

Host restored: 150 W, no containers, swappiness 60, VRAM empty.
Status: PROVISIONAL — NOT FOR PUBLIC HEADLINE.

## FINAL SCOREBOARD (2026-08-30): k=8 sets 51.54/56.85 tok/s records

| FP8 route (TP2, 230 W, n=3, random draws) | p512 | p1024 | p2048 |
|---|---:|---:|---:|
| W8A8 tuned baseline | 32.44 | — | 29.49 |
| W8A16 full (k=4) | 43.34 | 48.08 | 12.40 ⚠ |
| small-M kernel + scratchpad, k=4 | 46.09 | 47.49 | 34.98 |
| small-M kernel, k=6 | 48.89 | 53.83 | 11.64 ⚠ |
| **small-M kernel, k=8** | **51.54** | **56.85** | 13.56 ⚠ (3.1% this draw) |
| **HYBRID (prefill W8A8 / decode small-M), k=6** | 46.22 | 43.99 | **28.47 / 32.16** (accept 49/61%) |

k=8 native-MTP: accept len 6.63 at p1024, 5.38 at p512 — the native head
generalizes to deep speculation and the small-M op absorbs the 9-row verify.
Monotonic wins ≤1536 tokens: 32.4 → 51.5 (p512, +59%) and → 56.9 (p1024).

**p2048 collapse ROOT-CAUSED + OPERATIONALLY FIXED:** hybrid routing
(decode/small-M → W8A16-family fast path; prefill/large-M → inline-quant
W8A8, i.e. the MTP head's training numerics) restores p2048 acceptance
0% → 49%/61% across two draws at 28.5/32.2 tok/s. Mechanism: W8A16 prefill
numerics poison the KV/GDN state the MTP head consumes; W8A16 prefill is
safe only where acceptance is verified. `patch_fp8_hybrid.py`
(B70_FP8_MODE=hybrid|w8a16|w8a8). Deploy: ≤1536-token workloads → full
W8A16 k=8 server; ≥2048-token workloads → hybrid k=6 server.

**dflash k=6 refuted:** the k=4-block-trained drafter degrades wholesale
(accept 11.3–32.2%; best dflash remains k=4 at 39.79 tok/s). A k≥6 DFlash
requires a retrained drafter (SpecForge XPU port + hidden-state capture
against THIS stack — next-session project; MTP4 k=8 native already at 56.9
without any training).

**150 tok/s status:** FP8 best 56.85 (p1024, k=8). Remaining multipliers
(documented, each a full session): (1) k≥6 DFlash/MTP drafter retrained on
the small-M stack (accept len 6.63 → 8+ is the path to 70–90); (2) W8A16
prefill without the acceptance tax = removes the hybrid prefill penalty at
≥2048; (3) TTFT amortization for request-level throughput at p2048;
(4) GDN GRF256. INT4 84 tok/s remains the host overall champion; FP8 56.9
is the FP8 record. 150 tok/s FP8 would additionally need step ≈30 ms
(beyond small-M projection) — the ladder is documented rung by rung.

Ops: restore verified (150 W, 0 containers, VRAM 31662/32615 MiB empty).
Status: PROVISIONAL — NOT FOR PUBLIC HEADLINE.

## k=8 drafter retraining — pipeline ground work (2026-08-30)

SpecForge XPU training branch cloned to `/mnt/<models-disk>/specforge-xpu`
(rmacy fork `xpu-training-support`, the exact unmerged PR sgl-project#769
lineage — same author/pipeline that trained our drafter). Contents verified:
XPU device port (5 files), `configs/qwen3.8-27b-dspark.json` (block_size 7,
target_layer_ids 4/16/28/40/52, markov rank 256),
`examples/configs/qwen3.8-27b-dspark-offline-xpu.yaml` (schema-validated,
attention_backend sdpa — flex_attention is CUDA-only),
`XPU-TRAINING-PLAN.md`.

**XPU training capability: PROVEN in the plan's own probes** — bf16
backward/AdamW smoke PASS (20 steps, 0.29 s) and the 1.36B DSpark drafter
fwd+bwd+AdamW PASS on xpu:0 (loss 1.254→1.197). Forward contract:
position_ids must span ctx+block. VLLM_XPU_ENABLE_XPU_GRAPH must stay 0 for
any capture/training-adjacent run (xe hang, persists across reboot).

**Remaining gates (next session, in order):**
1. Hidden-state capture — write a vLLM-XPU capture script (two options:
   llm-scaler sglang-BMG image build, or hook the FP8 target's layers
   [5,17,29,41,53] during offline generation; budget ≈50 GB per 1M tokens —
   models LV has 33 GB free, so cap the corpus or extend the LV). Capture
   MUST come from the FP8 stack: training on BF16-target states re-creates
   the numerics mismatch that caps the off-shelf drafter at ~23–30%.
2. Train: `SPECFORGE_DEVICE=xpu specforge train --config
   examples/configs/qwen3.8-27b-dspark-offline-xpu.yaml` (single GPU; 1.36B
   drafter; target not loaded; checkpoint every 16 steps).
3. Export: `specforge/export/to_hf.py` → serve at k=6/8 on the small-M
   stack (op handles M≤40 → 1×9 rows at k=8 ✓) → re-bench p512–p2048 with
   `qw38-natural-content-bench.py` + random protocol.

Why it matters: the k=4 drafter is block-trained — k=6/8 on it collapses
(11–32%); the native MTP head generalizes for free (k=8 → 56.85 tok/s,
accept len 6.63). A drafter retrained at k≥6 on W8A16-decode-numerics
hidden states is the remaining multiplier toward the 150 aim, on top of
acceptance-collapse root-causing already delivered (hybrid mode).
