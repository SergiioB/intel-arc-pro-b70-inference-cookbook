---
title: "ADR-0009: Xe2 Megakernels, Persistent Fusion, and ESIMD Optimization Policy"
status: final
type: adr
date: 2026-09-15
---

---
id: b70-ref-adr-0009-xe2-megakernel-and-persistent-fusion-policy
title: "ADR-0009: Xe2 Megakernels, Persistent Fusion, and ESIMD Optimization Policy"
document_type: reference
status: canonical
version: "1.0"
effective_date: 2026-08-29
last_reviewed: 2026-08-29
authority: docs/methods/benchmark-protocol.md
scope: Architectural guidelines for Megakernels, Persistent Kernels, and ESIMD on Intel Arc Pro B70 (Xe2)
adr:
  number: 9
  decision_status: accepted
  truth_verdict: ESTABLISHED
  evidence_tier: E2
  validation_state: architecture_analyzed
  publication_state: not_for_public_headline
  correctness_state: not_a_correctness_decision
  operational_state: research_guideline
  supersession_state: current
---

# ADR-0009: Xe2 Megakernels, Persistent Fusion, and ESIMD Optimization Policy

## Context

Standard LLM decode ($M=1$) suffers from driver launch overhead (200–300+ kernel launches per token) and intermediate activation roundtrips across global VRAM (GDDR6). Recent literature (*Grokking Megakernels* by Infatoshi, *ThunderKittens* by Stanford Hazy Research) demonstrates that fusing the full transformer forward pass into a single persistent kernel eliminates launch latency and keeps activations in on-chip SRAM/registers.

We evaluate how to apply persistent megakernel concepts, ESIMD (`sycl::ext::intel::esimd`), sub-group specialization, and kernel fusion to the Intel Arc Pro B70 (Xe2 Battlemage, 32 Xe-cores, 32 GB GDDR6).

See full architecture analysis: [`research/2026/megakernel-xe2-xmx-architecture-guide-20260829.md`](../research/2026/megakernel-xe2-xmx-architecture-guide-20260829.md).

## Decision

1. **Single-Card Persistent Grid Default:**
   - Any custom megakernel exploration must target a single B70 card (32 work-groups matching the 32 physical Xe-cores).
   - Hidden states ($8\text{ KB}$ for $d=4096$ in FP16) must remain resident in General Register Files (GRF) or Shared Local Memory (SLM) across all layer steps.
2. **Strict Prohibition of Cross-Card TP Megakernels (premise corrected 2026-10-07, RL-059):**
   - The original rationale cited the retired B450-F topology ("PCIe Gen3 x8/x4, 3.6 GB/s DDR4 bounce"). On the current X570-F host the links are PCIe Gen4 x8/x8 and direct-peer P2P **copy** is verified at ~11–12 GB/s (checksum-clean, ACCESS flag) with **ATOMICS absent**. The prohibition stands on the corrected premise: a barrier-synchronized persistent grid needs fine-grained synchronization and atomics this link cannot provide, and every barrier still crosses PCIe. Keep the decision; do not cite the old topology numbers.
   - Dual-card serving must use independent megakernel engines per card (`ZE_AFFINITY_MASK=0` and `ZE_AFFINITY_MASK=1`).
3. **Mandatory DPAS Precision Alignment:**
   - Megakernels on B70 must use native XMX systolic `dpas` formats: **INT4 (GPTQ/AWQ)**, **INT8**, or **FP16/BF16**.
   - Non-native quants (such as `IQ3_XXS` or `Q3_K`) that require scalar unpacking loops must **not** be used in persistent megakernel pipelines.
4. **Sub-Group Specialization on Xe-Cores:**
   - Megakernel designs should divide work-group execution into producer sub-groups (async 2D block prefetching into SLM) and consumer sub-groups (systolic XMX `dpas` compute).
5. **Priority Engine Actions:**
   - **llama.cpp SYCL:** Prioritize fusing the MoE gating router and `MUL_MAT_ID` on-device to eliminate the 84+ blocking `stream->wait()` and D2H expert-ID host roundtrips per token (`research/2026/dsv4-k216-request-to-output-pipeline-20260825.md`).
   - **vLLM XPU:** Ensure decode kernels are compiled with explicit sub-group size bounds (`[[intel::reqd_sub_group_size(16)]]`) to prevent GRF register spills.

## Consequences

- Directs future kernel optimization toward eliminating host synchronizations and register spills before attempting complex multi-model pipelines.
- Prevents wasted effort trying to implement cross-card TP megakernels over Gen3 PCIe.
- Establishes clear qualification criteria for when a small model ($0.5\text{B}\text{–}3\text{B}$) qualifies for a dedicated persistent ESIMD prototype.
