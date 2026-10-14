---
title: "ADR-0008: Independent per-card stacks are the dual-B70 default; vLLM TP2 is a gated research path"
status: final
type: adr
date: 2026-08-24
---

---
id: b70-ref-adr-0008-dual-b70-independent-default
title: "ADR-0008: Independent per-card stacks are the dual-B70 default; vLLM TP2 is a gated research path"
document_type: reference
status: canonical
version: "1.0"
effective_date: 2026-08-24
last_reviewed: 2026-08-24
authority: docs/methods/benchmark-protocol.md
scope: Dual-card product choice on b70-host (two Arc Pro B70; board: B450-F at decision time, ASUS ROG STRIX X570-F GAMING dual x8/x8 since 2026-08-27 — see research/2026/x570f-dual-b70-pcie-topology-20260830.md. CORRECTED 2026-10-07: peer P2P COPY is verified at ~11-12 GB/s with ATOMICS absent (RL-059); the decision is unchanged — copy-only collectives stay gated research and independent stacks remain the default)
adr:
  number: 8
  decision_status: accepted
  truth_verdict: PARTIAL
  evidence_tier: E2
  validation_state: topology_confirmed_tp2_externally_reported
  publication_state: not_for_public_headline
  correctness_state: not_a_correctness_decision
  operational_state: independent_stacks_default
  supersession_state: current
---

# ADR-0008: Independent per-card stacks are the dual-B70 default; vLLM TP2 is a gated research path

## Context

Two Arc Pro B70s were installed on an ASUS ROG STRIX B450-F at decision time; the host now runs an ASUS ROG STRIX X570-F GAMING (dual CPU-attached x8/x8, 2026-08-27+). Historical live topology
([`research/2026/b450f-dual-b70-pcie-topology-20260824.md`](../research/2026/b450f-dual-b70-pcie-topology-20260824.md))
shows GPU 0 at PCIe Gen3 x8, GPU 1 at Gen3 x4, ReBAR 32 GB on both, and no
usable GPU P2P (consumer AM4 root ports, no ACS; copies bounce through host
RAM).

> **Correction — 2026-10-07 (RL-059):** on the X570-F, direct-peer P2P **copy** is
> verified (bidirectional, checksum-clean, ~11.1–11.9 GB/s, ACCESS flag) with
> **ATOMICS absent**. The B450-F "no usable GPU P2P" premise above is superseded as a
> capability claim. The decision below is unchanged: copy-based collectives remain
> gated research (`research/PROGRAM.md` RQ-06) and independent stacks remain the default.

“Dual inference” is therefore at least three products:

1. two independent engines, one card each;
2. one vLLM tensor-parallel-2 model across both cards;
3. one llama.cpp layer/tensor split across both cards.

The 2026-08-24 topology note initially treated (2) as not recommended on
physics grounds. The same day, the public recipe
[JP-devv/humble-b70-llm](https://github.com/JP-devv/humble-b70-llm) showed a
working **host-staged collective** TP2 path on the same board class (their board
reported no P2P; second slot may be x4). That is external E0 evidence, not a local campaign.
Details: [`research/2026/humble-b70-llm-dual-tp2-20260824.md`](../research/2026/humble-b70-llm-dual-tp2-20260824.md).

## Decision

1. **Default dual-card serving remains two independent stacks**, one engine
   per GPU via `ZE_AFFINITY_MASK`. This is the launcher and campaign default.
2. **vLLM `--tensor-parallel-size 2` is a gated research path**, not a
   current profile and not a public headline. It may be pursued only as a
   declared campaign that includes the host-staged-collectives / piecewise
   graph-capture patches (or an equivalent upstream merge), empty-GPU gates
   on both cards, `--max-num-seqs 1` until a local Cn gate exists, and
   prefix-cache controls for any cold cell.
3. **Do not run stock Docker vLLM TP2** on this board expecting a 2× win.
   Without host-staged collectives and comms kept outside captured graphs,
   the expected failure is IPC error or graph-capture device loss.
4. **llama.cpp split** remains topology-viable for models that do not fit
   one 32 GB card. It is not a current launcher product.
5. **Never mix product metrics.** Independent-stack C1, independent-stack
   Cn, TP2 C1, and external LocalMaxxing rows are different coordinates.

## Evidence

- Local topology, bandwidth, and P2P absence: topology note §3–§6 (E2 for
  links and placement; diagnostic single-run best-of-5 for H2D/D2H).
- External TP2 existence proof, env, patches, and their two protocols:
  humble-b70-llm research note (E0; their numbers, not this host).
- Current Qwen3.8 single-card route: [`models/qwen/qwen3.8-27b/serving.md`](../models/qwen/qwen3.8-27b/serving.md)
  (Run 48 C1 102.61 tok/s recommended non-thinking; `not_comparable` to
  their TP1/TP2 tables).

## Required wording

Use:

> On this dual-B70 host (B450-F at decision time, X570-F since 2026-08-27), the default is one independent engine per
> card. vLLM TP2 without GPU P2P is an unmeasured local research path whose
> external existence proof uses host-staged collectives; it is not the
> serving default and is not a 2× claim.

Do not use:

> Dual B70 means 2× throughput.
> vLLM TP2 is impossible on this board.
> humble-b70-llm 136.4 / 94.58 tok/s is this host.

## Consequences

- GPU 1 power-off (`b70-gpu1`) remains valid for independent-stack
  operation and is incompatible with a live TP2 server.
- A TP2 experiment occupies both cards and is still “one declared engine.”
- INT8 lm_head from that recipe may be A/B’d on **one** card without
  adopting TP2.
- Local derived patched images stay local-only (`AGENTS.md` §7).
- Compile RAM (~7–12 GB per SYCL job; they recommend 64 GB host RAM) is a
  feasibility gate on this 32 GB DDR4 box.

## Reconsideration triggers

- A protocol-compliant local TP1 vs TP2 campaign on a pinned image/patch
  set with quality gate and energy-aligned power.
- Upstream vLLM/XPU merge of host-staged collectives or working peer IPC
  on a future board.
- A launcher need for a single model that does not fit one 32 GB card.
- Working GPU P2P (not expected on this AM4 board).
