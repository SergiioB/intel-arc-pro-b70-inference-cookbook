---
title: "EXL3 TP2 decision — stock NO-GO, custom TP2 conditional (2026-09-29)"
type: research
date: 2026-09-29
status: provisional
---

# EXL3 TP2 decision (2026-09-29)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Authority-closing synthesis of
four 2026-09-29 reports; no GPU/system/build work performed or authorized.
Inputs: `b70-exl3-tp-feasibility-upstream-20260929.md` (pinned-source audit),
`b70-exl3-tp-feasibility-bound-20260929.md` (adversarial bound),
`b70-exl3-round-attribution-audit-20260929.md` (measurement-class audit),
`b70-exl3-tp-cpu-slice-20260929.md` + `results/2026/09/b70-exl3-tp-cpu-slice-20260929/` (T1 executed).
Reviewed target: `b70-tp2-decode-feasibility-20260929.md`.

## 1. Decision

- **Stock TP2 on the incumbent exl3xpu route: NO-GO — impossible, not slow.**
  The pinned plugin hard-raises `NotImplementedError("exl3: tensor parallel
  sharding not supported yet (use DP)")` in `create_weights` on any sharded
  partition (`vllm_plugin.py:157-162`, verified on the pinned extract). Even
  if it did not, the image pins `dist_backend="xccl"` only
  (`platforms/xpu.py:115`) — the same transport class RL-075 measured at
  −71.4 %/−45.5 % vs matched 1-card control on the GPTQ route.
- **Custom TP2 remains a research path — CONDITIONAL.** Exactly one gate has
  passed: the **representation gate** (T1, PASS 38/38 — on a sampled class
  of two layer-0 `linear_attn` tensor quads, `in_proj_z` K=5120→N=6144 and
  `out_proj` K=6144→N=5120, K=4 CB_MUL1, batch m=8: decoded submatrices are
  bitwise-identical under 128-block-aligned N- and K-slices, column-split
  forward is bitwise, and row-split forward matches within fp32
  reassociation only, max_abs ≤ 9.5e-6). **This is not a format-wide or
  full-model gate:** the mixed-precision families — head (head_bits=6) and
  MTP (mtp_bits=4) — and every other layer and tensor family are unsampled.
  Everything else is unproven: sharded loaders, kernel
  efficiency at half-N, the copy-only collective itself, and — decisive —
  **transport capture** (cross-card waits inside XPU graph replay have never
  been demonstrated on this stack; §20 rule 1 makes capture-on mandatory for
  any champion input).
- **Production route unchanged:** single-card exl3xpu 4bpw @ 63.25 tok/s;
  dual-card default stays independent stacks per ADR-0008. Do not launch any
  TP2 arm.

## 2. Numbers that survive the audit (with their caveats)

| Term | Value | Caveat |
|---|---|---|
| Representative serving round | **47.2 ms** @ 63.25 tok/s, yield 2.954 (p512/g128 n=5, 230 W) | measured wall |
| Diagnostic window | 44.10 ms @ 70.1, yield 3.134 | n=1 favorable sample, `-prof` server; **never pair with 63.25** — the 2.79 yield in the reviewed note is a cross-window artifact, refuted |
| M=4 verify pass | 34.46 ms / 13.126 GB / 381 GB/s | **isolated-op sum at the 180 W as-found cap** (RL-098) — cross-cap vs the 230 W wall; the 2.50 ms "residual" is a subtraction across measurement classes and caps, unbounded in either direction |
| Draft traffic | 3.499 GB/round, of which **2.86 GB = shared 6-bit lm_head read 3×** | measured census + source-verified; draft wall-time (7.14 ms) is an isolated-rate estimate, not measured |
| Collective wire transfer (arithmetic) | 1.74 µs/op (10 KiB × 2 @ ~11.5 GB/s verified P2P copy rate) | bandwidth-derived figure, not a demonstrated floor — launch cost of a captured custom collective is unmeasured; ~98 % of measured cost is fixed overhead |
| Measured barrier | **~102 µs/op** (xccl ladder: 101.7 @4 KB, 104.5 @64 KB, 129.2 @512 KB — flat ⇒ fixed cost) | RL-071, copy-correct but graphs-off |

## 3. Conditional TP2 pricing — additive only, no multiplying ceilings

**Every tok/s and round figure in this section is a conditional model
estimate** — cross-cap arithmetic mixing a 180 W isolated-op sum, a 230 W
serving wall, a measured xccl barrier from a different transport, and
modeled per-card shrink. Nothing below is a measurement or a guaranteed
ceiling; rows labeled "Optimistic model bound" are the model's own
best-case arithmetic ceilings under their stated cross-cap assumptions —
optimistic scenarios conditional on those assumptions, not hard physical
limits or measured caps.

Per-card halving is priced against the real sharding census (row-parallel
families halve K not N; ~6 % shape penalty band 17.4–21.5 ms). Collective
census depends on the draft design (§4). At the **measured** barrier:

| Scenario | Round | tok/s | ×63.25 |
|---|---:|---:|---:|
| TP2, design A, 102 µs barrier | 38.5–41.8 ms | **70.7–76.6** | 1.12–1.21 |
| TP2, 60 µs barrier (RL-091 program-stop threshold) | ~33.7 | 87.7 | 1.39 |
| TP2, 35 µs barrier (aspiration) | ~30.3 | 97.7 | 1.54 |
| Optimistic model bound, measured yield 2.954 | — | **127 @60 µs / 100 @102 µs** | — |
| Optimistic model bound, perfect MTP3 yield 4.0 (never observed) | — | 172 @60 µs / 135 @102 µs | — |

Rules of record:

- **Never multiply ceilings.** The reviewed note's 85–110/110–140/150+
  bands do not survive the model's own accounting: at the note's own
  (refuted) yield premise, 190 tok/s would require a *negative* collective
  budget; at the measured yield it would require ~4.3 µs/op — far below
  the measured ~102 µs barrier, and no demonstrated implementation
  supports it (the 1.74 µs wire figure assumes the verified copy rate; a
  launch floor for a captured custom collective is unmeasured). Stacked
  composition is additive in the model (98 @102 µs, 122 @60 µs).
- The TP2 base case (≤×1.21) sits at or below the TP1 kernel track's
  **own projected ceilings** (had_out patch ~65.3–66.2; M=4 rewrite
  ~79–84, RL-102 — themselves conditional estimates, not measured) while
  requiring new hardware behavior — it must be priced against that, not
  against the incumbent alone.
- **Energy (hypothetical draw assumption, unmeasured):** if two cards
  sustained ~cap-class draw (~400 W assumed, not measured under TP2 load),
  the ×~1.18 modeled rate would imply ~1.7× worse J/token — an estimate,
  not a measured efficiency conclusion. It also converts 2×63.25 tok/s of
  independent-stack capacity into 1×~75 — the ADR-0008 opportunity cost.

## 4. The two draft designs are different decisions, not a contradiction

| | Design A — shard everything | Design B — replicated draft stack |
|---|---|---|
| Collectives/round | **138 allreduces** (128 trunk + 6 draft + 4 embedding) + 1 logits allgather (~2.0 MB) + 3 argmax pair-gathers | **128 allreduces** (trunk only), zero draft/vocab ops |
| Per-card draft+head traffic | halved (3.52 ms) | duplicated (≈4.45 GB/round ⇒ ~7.1 ms) |
| Correctness | draft forward numerics reassociate vs TP1 | replicated draft avoids draft-side reduction reassociation, but end-to-end parity is unproven — hidden states feeding the draft already carry trunk TP reduction reassociation (argmax drafting keeps proposals aligned across ranks; a sampling drafter would need seed-sync) |

Both share the 128 trunk allreduces (2/layer × 64, 10 KiB). Quoting A's
"138" does not refute B's "zero draft collectives"; quoting B's "128" while
keeping A's halved draft bytes double-counts. The verdict direction is the
same under either (B reprices S3 to ~43–46 ms, ~64–69 tok/s).

## 5. New lever found — NOT adopted

`EXL3_DRAFT_VOCAB` + `_build_draft_head` + `_patch_mtp_draft_logits`
(`vllm_plugin.py:237, 271–319, 331–332`, verified pinned): an opt-in pruned
draft head — draft logits over a 128-block-aligned vocab subset while the
target verifies full vocab. RL-090's launch did not set it. It is the only
genuinely new candidate surfaced by these reviews (attacks the 2.86
GB/round shared-head re-read on TP1, and shrinks design B's replicated-head
cost), **but it is not quality/output-preserving by inspection** — it is a
draft-proposal-subset change requiring an acceptance gate + spec-canary
before any serving claim. Do not adopt; record as a gated candidate.
(Bound review notes the orthogonal head_bits 6→4 variant on the same term.)

## 6. Next deciding step — smallest GPU falsifier (nothing launched)

Build and measure a **10 KiB copy-only (flag-slot) collective** on the
verified P2P ACCESS path, **XPU graph capture ON**, 2 ranks:

- **> 60 µs/op ⇒ pre-declared program stop** (RL-091 kill rule): the modeled
  base case stays ≤×1.21, at or below the projected TP1 kernel-track
  improvement, so the TP2 program closes **on economic grounds relative to
  that alternative** — this is a decision gate, not a claim that TP2 is
  physically impossible at higher barrier costs.
- **≤ 35 µs ⇒ conditional ~90-band opens**; 35–60 µs is a narrow
  base-case-only zone still requiring the ADR-0008 matched-control screen
  (graphs ON, spec-canary, pre-declared +20–50 % band).

Caveats of record: the custom flag-slot collective **does not exist in any
tree** (RL-091 Audit C: no spin/poll precedent, reader-side visibility of
peer writes unknown). The O14 harness measured **xccl** collectives
(~102 µs, graphs-off per its own recommendation) — do not assume it
provides the required primitive or capture behavior. The loader work
(relax `vllm_plugin.py:157-162` per partition type, per-rank `shard_of_nb`)
is likewise absent. Both must exist before the falsifier can even run; T1
only proved they are worth building *if* transport lands in-band.
llama.cpp is not a fallback (GGUF-only; `-sm layer` is serial, `-sm row`
segfault-class, `-sm tensor` unmeasured — separate research cell).

## 7. Registrations

Ledger: **RL-103** (stock TP2 impossible — blocked), **RL-104** (T1
representation gate — validated), **RL-105** (custom TP2 bound/attribution
— blocked as priced, conditional), **RL-106** (EXL3_DRAFT_VOCAB candidate —
provisional, not adopted). Campaign: **CAM-2026-EXL3-TP-CPU-SLICE**
(evidence tier E2). model-status production route unchanged; Qwen3.8-27B
TP2 status narrowed to "stock impossible / custom conditional".
