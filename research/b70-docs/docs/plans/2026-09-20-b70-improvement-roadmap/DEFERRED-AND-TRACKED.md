---
type: plan
date: 2026-09-20
phase: DEFERRED
title: "Deferred and tracked work (EXL3, sub-byte DPAS, PP2, TP2)"
status: TRACKED
opened: 2026-09-20
updated: 2026-09-20
ledger: n/a
depends_on: []
---

# Deferred and tracked work

## Status log

| Date | Item | Change | Note / evidence |
|---|---|---|---|
| 2026-09-20 | all | filed as TRACKED | Reopen triggers recorded per item |

---

## 1. EXL3 / Trellis — **DEFERRED (no-go as a fleet project today)**

### Why (the full reasoning)

**1. Nothing exists to run it.** ExLlamaV3 is CUDA-first with partial ROCm support.
There is no SYCL, Level Zero, oneAPI, OpenVINO, vLLM, or GGUF path. Adopting it means
writing the entire trellis kernel stack ourselves: format parsing and reference
reconstruction, Xe2 GEMV and small-M kernels, large-M fused reconstruction/GEMM
kernels, engine loader/packing/dispatch, MoE routed-expert support, speculation/KV
integration, plus the quality gates. Planning estimate: a narrow dense prototype is
several expert-weeks; production dense + MoE is multi-month. Materially larger than
any item on the P1–P5 queue.

**2. It buys no arithmetic on this silicon — and the codes are not integer DPAS
operands anyway.** The measured rate-class law says sub-8-bit adds zero TOPS on Xe2.
More fundamentally, trellis codes select/generate *reconstructed weight values*; the
straightforward Xe2 implementation reconstructs FP16/BF16 tiles and uses the
floating-point DPAS class — the same class W4A16 already occupies. Integer storage
codes ≠ integer matrix operands. There is no native "trellis DPAS" or
indirect-codebook matrix operand; data must be reconstructed and tiled first.

**3. The bandwidth win is bounded and easily eaten.** Against the stipulated 27B,
4.5-effective-bpw baseline:

| Rate | Weight bytes | Ideal stream time @ 608 GB/s |
|---|---:|---:|
| 4.5 bpw | 15.19 GB | 24.98 ms |
| 3.5 bpw | 11.81 GB | 19.43 ms |
| 3.0 bpw | 10.13 GB | 16.65 ms |

At 3.5 bpw the ideal saving is **5.55 ms per full weight pass** — that is the entire
budget for reconstruction, codebook gathers, rotations, and metadata. Unhidden
reconstruction + Hadamard rotations of ~5–15% of baseline time reduce the end-to-end
bound from ~1.22× to ~1.15×/1.03×. KV traffic, activations, draft work, and MTP do
not shrink. A double-digit end-to-end gain is a *plausible successful outcome*, not a
promise.

**4. The fleet has no capacity crisis for it to solve.** The 27B/35B models fit at
INT4 with quality gates passed and a mature, BF16-preserved MTP integration that a
format switch would forfeit. K216 is *not* a miracle case: the artifact is already
IQ3-class (~3 bpw); single-card placement needs ~1 bpw, not 3. The genuine trigger
class is a desirable model just past the 32 GB line — e.g. 70B-class dense needs
~3 bpw ≈ 26 GB for weights alone (KV/metadata/workspaces on top) — which is exactly
EXL3's home turf but does not exist in the current fleet.

**5. The quality praise is real and is harvested more cheaply by P8.** EXL3's
quality-per-bit edge below 4 bpw comes substantially from incoherence processing
(Hadamard-class rotations) plus codebook structure. The rotation half of that idea
can ride on the existing validated W4A16 kernels — that is P8's rotated-GPTQ branch,
at days not months.

### Reopen triggers (ALL must hold)

1. A specific desirable model is capacity-blocked on the current fleet.
2. Its EXL3 representation demonstrates adequate quality at a bitrate that materially
   changes placement.
3. A maintained SYCL backend exists **or** a narrow Xe2 reconstruction/GEMV probe wins
   after reconstruction, rotations, and metadata are counted.
4. Engine integration (and MoE routing where required) is funded.
5. Speculation and quality gates remain viable under the format.

A ROCm implementation alone is not an Intel backend.

---

## 2. Pure sub-byte DPAS (W4A4 / W2A2) — **cheap probe, promoted into P0**

### Why

The rate-class note leaves OPS_PER_CHAN=8 explicitly unmeasured ("do not cite a 2×
sub-byte figure either way without a 4×4 microbench"). One operator microbench with
verified instruction emission closes the question and updates
`research/2026/xmx-dpas-rate-classes-20260915.md`. It is hardware research only: no
quality-viable A4 activation path exists for LLM serving, so a faster 4×4 microbench
must never be sold as a serving route.

---

## 3. PP2 (pipeline-parallel 2, host-RAM collectives) — **deferred/blocked per ledger**

### Why

RL-016 records the blocked state. Reopen only with a specific upstream/runtime change
addressing the recorded failure *and* a capacity-driven use case (a model that fits
2×32 GB but not 1×32 GB and cannot be served as two independent stacks). Do not rerun
stock PP2 as a general speed experiment — the collective and staging costs are the
same class that makes TP2 unattractive on this board.

---

## 4. TP2 — **retain research status quo (ADR-0008)**

### Why

- Both cards are CPU-attached x8 with **no GPU P2P**; host-staged oneCCL AllReduce
  measured 4–7 ms per step (10–18% of a speculative step budget).
- Independent single-card stacks are the default and the demonstrated product
  (ADR-0008); the validated TP2 recipe (worker affinity, SYS_PTRACE, host-staged CCL)
  is a gated research path, not a launcher product.
- Single-request 2× scaling is implausible as a default expectation given collective
  costs; two independent engines usually beat one TP2 server for the fleet's actual
  workloads.
- Cross-card persistent megakernels are banned outright (contract §17).

Any new TP2 comparison must justify capacity or service objectives against two
independent engines.
