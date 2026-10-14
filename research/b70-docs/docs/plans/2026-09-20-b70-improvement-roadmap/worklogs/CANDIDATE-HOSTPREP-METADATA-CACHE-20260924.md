---
title: "Candidate Pre-Commitment: Host-Prep Metadata Caching for C1 Speculative Decode"
type: research
date: 2026-09-24
status: living
---

# Candidate Pre-Commitment: Host-Prep Metadata Caching for C1 Speculative Decode

**Date:** 2026-09-24  
**Status:** OPEN CANDIDATE (Pre-Screening Protocol Commitment)  
**Author:** omp / Sergio  
**Authority:** `research/2026/exl3-analysis-and-next-inference-levers-20260924.md` (§5) & `docs/methods/benchmark-protocol.md` (§1.9)  
**Target Model:** `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` (local `-nogidx` artifact)  
**Stack:** Single-card B70 (GPU 1), vLLM XPU `f01e24f6…` + overlay, MTP4, graph capture enabled, recommended non-thinking sampling, 230 W cap.

---

## 1. Hypothesis & Quantitative Derivation

- **Observation:** In the uninstrumented Control 1 and Control 2 runs, mean speculative round wall time was ~44.8 ms (73.84–77.14 tok/s post-first, yield ~3.46 tokens/round). The PyTorch profiler measured 10.08 ms of host input preparation (inflated by profiler Python frame hooks). The baseline uninstrumented host prep is estimated at $T_\text{prep} \approx 4.8\text{ ms}$, representing an exclusive round fraction of:
  $$f = \frac{4.8\text{ ms}}{44.8\text{ ms}} \approx 0.107 \quad (10.7\%)$$
- **Mechanism:** In steady-state C1 speculative decode with MTP4 (`num_spec_decodes == 1`, `num_prefills == 0`, `num_decodes == 0`), `vllm/v1/attention/backends/gdn_attn.py(210): build()` executes redundant CPU and GPU allocations on every single round:
  1. `torch.arange(5, dtype=torch.int32, device=device)` allocates a new tensor on device every round.
  2. `torch.empty(0, dtype=torch.int32, device=device)` allocates a new empty tensor on device every round.
  3. `async_tensor_h2d(spec_sequence_masks_cpu)` executes an asynchronous host-to-device copy of `tensor([True])` every round.
  4. Advanced boolean mask indexing `spec_state_indices_tensor = block_table_tensor[spec_sequence_masks_cpu, : self.num_spec + 1]` evaluates a CPU mask on a GPU tensor, triggering PyTorch mask dispatch and allocation.
- **Proposed Optimization (One Variable):**
  Pre-allocate and cache `spec_token_indx`, `non_spec_token_indx`, and `spec_sequence_masks` on device once for the C1 steady-state shape. Replace mask indexing with direct tensor slicing `block_table_tensor[:1, : self.num_spec + 1]` (an instantaneous O(1) view).
- **Amdahl Projection:**
  Assuming a local speedup $s = 2\times$ on host prep (~2.4 ms saved) with zero added overhead ($o = 0$):
  $$S_\text{net} = \frac{1}{(1 - f) + \frac{f}{s} + o} = \frac{1}{(1 - 0.107) + \frac{0.107}{2}} = \frac{1}{0.893 + 0.0535} = \frac{1}{0.9465} \approx 1.0565\times \quad (+5.65\%)$$
- **Risk & Sensitivity:**
  The projected margin over the mandatory $\ge 5\%$ investment gate is narrow (+5.65% vs 5.00%). Because control post-first throughput varied by 3.3 tok/s (77.14 vs 73.84 tok/s, ~4.3% spread) between consecutive runs, arithmetic alone cannot decide adoption. The empirical screen must decide.

---

## 2. Pre-Committed Acceptance Gates

Before this candidate can be promoted to a production serving change or cookbook recommendation, it must satisfy all protocol gates:

1. **Isolated Correctness Gate:**
   - Spec-canary check (`spec-canary-20260923.py`): 8/8 identical greedy outputs between patched and unpatched arms.
   - Greedy byte-identity verification: SHA256 of greedy output must match baseline exactly across three standard prompts.
2. **Counterbalanced A/B Screen (n=3 per arm):**
   - Matched screening at BOTH anchors:
     - Short cell: C1 p512/g128 (calibrated tokens).
     - Long cell: C1 p8192/g128 (calibrated tokens).
   - Order: ABBA schedule within the same server session to isolate thermal drift.
   - Acceptance threshold: $\ge 5.0\%$ representative client post-first throughput gain at the primary anchor without material regression ($< 1.0\%$) at the long anchor.
3. **Finalist Confirmation (n=5 per arm):**
   - If the screen passes, run a fresh n=5 confirmation at both anchors.
   - Screen samples must NEVER be folded into the finalist sample set.
4. **Telemetry & Isolation Invariants:**
   - Empty-GPU gate: visible VRAM $\ge 31,000\text{ MiB}$ on GPU 1 before start.
   - Cooldown: GPU package temperature $\le 55^\circ\text{C}$ before each measured arm.
   - Power cap: fixed 230 W configured cap; dynamic hwmon energy counter sampling (`energy1_input`) across the exact workload window to report measured draw and tokens/J.
   - Prefix cache: zero cache-hit delta bracketed before and after each cell.
   - Cleanup: container destroyed, 150 W power cap restored upon exit.
