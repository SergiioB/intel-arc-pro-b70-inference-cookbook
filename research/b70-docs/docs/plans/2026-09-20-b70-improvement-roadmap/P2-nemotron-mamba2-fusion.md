---
type: plan
date: 2026-09-20
phase: P2
title: "Nemotron incremental Mamba2/SSD fusion"
status: WITHDRAWN
opened: 2026-09-20
updated: 2026-09-20
ledger: RL-TBD-P2
depends_on: [P0]
effort: 5–10 sessions
---

# P2 — Nemotron incremental Mamba2/SSD fusion

## Status log

| Date | Status change | Note / evidence |
|---|---|---|
| 2026-09-20 | filed as PLANNED | Roadmap filed from b70-architect draft |
| 2026-09-21 | PLANNED → RESEARCH — **premise measured and not supported** | Both artifacts were re-downloaded (target + DFlash n7). The recorded image tag `1da0a954-det0123` is absent and unpullable, so the probe ran the local image on GPU 1 at 230 W (deviations recorded). Baseline: **252.30 post-first / 178.55 out-throughput tok/s**, TTFT 0.21 s, 5,220 MiB free after load. Dispatch attribution over 3,338 device events (2,876 kernels): **MoE gemm 41.7 % + `gemm_kernel` 31.3 % = 73 % of device time**, while the **entire Mamba/SSD family the phase proposes to fuse is 8 %** with medians 55–357 µs (i.e. not tiny fragmented launches). With XPU graphs on, the fragmentation the phase budgets 1.01 ms/token (≈19 %) against is already largely paid down, so **fusion cannot deliver the predeclared gain — the phase's own stop rule applies**. Caveats recorded: mixed prefill/decode window; the 736-launches figure not directly counted. Worklog `worklogs/P2-20260921.md`; evidence `results/2026/09/b70-improvement-probes-20260920/p2-nemotron-entry-20260921T102137Z/`. |
| 2026-09-21 | RESEARCH → **WITHDRAWN** (stop rule fired) | P2's premise was that 736 dispatches/token of fragmented Mamba work is the loss class fusion removes, worth 1.01 ms/token of a 5.36 ms token (≈19 %). The trace measures the whole fusable Mamba/SSD family at **8 % of device time** (medians 55–357 µs — not tiny fragmented launches) against **73 % in MoE+GEMM**, and with XPU graph capture on, the launch cost the budget assumes is already paid down. 8 % of 5.36 ms is ≈0.43 ms, well short of the 1.01 ms the phase needs, so **the fusion program cannot deliver its declared gain and is withdrawn with the evidence attached** (the 2026-09-20 filing's own stop rule: "Stop when exposed overhead cannot support the predeclared useful gain"). **Reassignment:** the 73 %-GEMM attribution is P5's entry evidence (grouped/small-M GEMM waste is the measured object, not Mamba fusion). Caveats recorded with the withdrawal: the window mixed prefill and decode, and the 736-launch figure was not itself counted; neither changes the ordering GEMM ≫ fusable family, and any future re-entry must first show a Mamba-family share above ~30 % of device time under a decode-only window. Worklog `worklogs/P2-20260921.md`. |

## Why this decision (the reasoning)

**The thesis: Nemotron is the fleet's throughput flagship being throttled by *measured*
launch fragmentation, and unlike the dense-27B case, a fusion program here has a real
millisecond budget to win.**

1. **The budget works, and that is the whole argument.** 186.61 tok/s = 5.36 ms per
   output token. 230 tok/s = 4.35 ms. The required saving is **1.01 ms/output** — and the
   Mamba blocks issue **736 kernel launches per token** (46 layers × 16–18 dispatches:
   input proj → split → causal conv → SiLU → selective scan → skip → SwiGLU → grouped
   RMSNorm → output proj). Contrast the dense Qwen3.8 case, which is 98.4%
   bandwidth-bound: a base-model GEMV megakernel there buys <1.5%. Fragmented execution
   is exactly the loss class fusion *can* remove, and this model has the most
   fragmentation on the host. This is why Nemotron gets a fusion program and dense 27B
   does not.
2. **Why *incremental*, not the monolithic "all 16 ops in one kernel" sketch.** The old
   roadmap drew one persistent kernel per Mamba layer holding everything in SLM. Three
   corrections forced the redesign:
   - **State size:** an illustrative 4096×128 SSD state is 524,288 elements = 1 MiB FP16
     / 2 MiB FP32 *per layer* — it does not fit one Xe-core's SLM. State must be tiled
     across workgroups with explicit residency and lifetime; "keep the SSM state in
     SRAM" is not a design, it is a slogan.
   - **Rollback semantics:** DFlash speculative decoding can reject at *any* draft
     position; every fused stage must preserve accept/reject state restoration. A
     monolithic kernel makes that a single enormous correctness surface; incremental
     fusion lets each step carry its own parity/rollback gate and be promoted or
     reverted independently.
   - **Attribution:** if the monolith regresses, you cannot tell which fusion hurt. The
     ordering (elementwise → conv/act/state → norm/gating epilogues → projections last)
     is cost-ascending and each increment is end-to-end measurable.
3. **Why the 736-launch figure must be normalized before savings are claimed.** 736 is
   launches per *token*; under speculation there are draft steps and verify steps per
   emitted output. Savings per launch × 736 ≠ savings per output. The P0
   state/dispatch trace converts the launch count into exposed time, which is the only
   number the 1.01 ms budget may be compared against.
4. **Why the MoE router is a separate question.** The 7 MoE layers (128 experts, 6
   routed + 1 shared active) also sync — but the *llama.cpp* D2H router defect must not
   be assumed to exist in the vLLM route (the architect's explicit warning: do not
   transfer the IQ3 diagnosis automatically to native vLLM grouped GEMM). Verify, then
   decide; the router work item lives inside this phase only if the trace shows it.

**Alternatives considered and rejected.**

- *Monolithic persistent Mamba2 kernel (old-roadmap shape).* Rejected for the three
  reasons above; the goal (fewer launches, less state traffic) is reached by the same
  increments with recoverable risk.
- *Base-GEMV megakernel spending on the MoE projections instead.* Wrong loss class —
  the flagship's loss is dispatch fragmentation, not weight streaming.
- *Dropping DFlash to simplify rollback during fusion.* Rejected: DFlash n7 is part of
  the champion configuration; kernel A/B must hold it fixed, not remove it.

**What would change this decision.** P0 trace showing exposed dispatch time < ~1 ms per
output → withdraw the 230 target and shrink the phase to whatever increments still pay;
a spec-freeze or architecture change in Nemotron serving (upstream vLLM landing a fused
Mamba2) → collapse to validation + benchmark of the upstream path.

## Objective

Reduce fragmented Mamba2 execution while preserving recurrent state and DFlash rollback
semantics. Targets an already-fast, useful route with a plausible ~1 ms/output
optimization budget.

## Baseline and E0 target

Baseline: 186.61 tok/s, p2048/g128, GPTQ-INT4 G64 + BF16 DFlash n7, n=5 client
post-first, 150 W configured cap (`registry/model-status.json`,
`models/nemotron/nemotron-3.5-lightning-30b-a3b/README.md`).

- Baseline 5.36 ms/output → 230 tok/s needs 4.35 ms/output → **save 1.01 ms/output**.
- E0 range: ~1.15–1.25× end-to-end. The 736-launch figure must be normalized by actual
  verification steps and emitted outputs before deriving savings.

## Engineering work

1. Inventory exact Mamba block operations and tensor layouts.
2. Specify recurrent-state ownership, reset, copy, update, and rollback behavior.
3. Fuse cheap adjacent elementwise operations first.
4. Fuse causal convolution, activation, and compatible state-update stages.
5. Tile SSD/state operations explicitly.
6. Fuse norm/gating epilogues where numerically safe.
7. Consider projection fusion only after the local stages show end-to-end value.
8. Keep distinct implementations where decode and prefill algorithms differ.
9. Evaluate MoE routing separately; do not assume the vLLM route has llama.cpp's D2H
   defect.

State arithmetic to respect in design: 4096×128 state = 1 MiB FP16 / 2 MiB FP32 per
layer — tile across workgroups; account for registers, memory residency, and lifetime.

## Validation gates

- Operator parity for each fusion increment.
- Chunked vs unchunked prefill; decode continuation from identical state.
- Sequence reset and cross-request isolation.
- Speculative accept/reject rollback, including rejection at every draft position.
- Long-sequence accumulated error against the reference.
- Per-position logits and task quality; no invalid hybrid PPL ranking.

## Benchmark plan

- Primary reproduction: C1 p2048/g128. Standard p512/g128, p8192/g128, p2048/g1,
  p8192/g1. One state-sensitive natural-text workload.
- C2/C4 serving checks only after C1 correctness.
- Fixed DFlash n7 during kernel A/B. n=3 screen, fresh n=5 finals.

## Risks and stop criteria

- Stop after two rollback/parity failures.
- Reject a fast path that changes recurrent semantics or silently changes draft
  acceptance.
- Stop an increment if spills or added memory traffic erase its gain.
- Promote increments independently; no monolithic "all 16 operations" requirement.
- Withdraw the 230 target if measured exposed budget is below ~1 ms/output.

## Deliverables

- `research/2026/nemotron-mamba2-incremental-fusion-<YYYYMMDD>.md`
- `results/2026/<MM>/nemotron-mamba2-fusion-<UTC>/`
- Ledger entry; stateful correctness suite; per-increment attribution.

## Cookbook integration

Update the Nemotron vLLM XPU recipe and engine dossier. Add audited p2048/g128 and
standard-card records. State exact DFlash configuration and whether the optimized path
supports concurrent serving.
