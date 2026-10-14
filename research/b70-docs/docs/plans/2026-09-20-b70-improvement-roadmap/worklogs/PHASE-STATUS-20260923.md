---
title: "Per-phase status note — every P0–P10 entry point after the 2026-09-22/23 night"
type: worklog
date: 2026-09-23
status: living
---

---
doc_id: WL-PHASE-STATUS-20260923
doc_type: worklog
title: Per-phase status note (P0–P10) — no entry point left silently untouched
lifecycle: active
created: 2026-09-23
updated: 2026-09-23
canonical_for:
  - roadmap-phase-status
---

# Per-phase status note — 2026-09-23

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** The close-out discipline: every entry point is
either completed or explicitly refuted/withdrawn with evidence, and no phase is left silently
untouched. Read `OUTCOMES.md` (wins/losses/deciders) and `LEVERS.md` (per-lever verdicts)
first; this note is the phase-by-phase roll call.

| Phase | Status | Verdict + the single decider | Evidence |
|---|---|---|---|
| **P0** baselines & probes | **COMPLETED** (`DONE-P0-baseline-and-probes.md`, RL-030) | Baselines captured and later re-captured; the 8.87 figure it carried is SUPERSEDED by 9.25 clean / 9.58 patched | `research/2026/b70-improvement-probe-baseline-20260920.md` |
| **P1** K216 routed-IQ3 device-ID repair | **WITHDRAWN** (RL-031; L1) | +3.54 % e2e vs a ≥10 % gate. Kept only as the SYCL-graph prerequisite — and since 2026-09-09 **merged upstream** as #28476 / `304665fe7` | `p1-k216-e2e-ab-20260921T081155Z/` |
| **P2** Nemotron Mamba2/SSD fusion | **WITHDRAWN** (RL-032; L2) | Fusable family 8 % of device time vs 73 % MoE+GEMM. Its trace was reassigned to P5/P3 | `p2-nemotron-entry-20260921T102137Z/` |
| **P3** K216 grouped-IQ3 prefill | **RESEARCH — the as-built kernel FAILED its gate (L9); the lever survives via P-3b** | Entry PASSED (ubatch 1.292× at 1024; 2048 saturated, 4096 OOM ⇒ the config lever is exhausted). Attribution COMPLETE (routed op = the whole prefill budget, 25× off dense). The E0a draft measured **1.39× SLOWER** despite perfect parity/zero spills ⇒ scheduling, not math, was the failure. Single decider now: the **IQ3_XXS extension on upstream #29245**, whose tile scheduling is exactly what E0a lacked | `perf-k216-grouped-iq3-20260923T002622Z.txt`, `worklogs/P3-review-pr29245-20260922.md`, `worklogs/ARCHITECT-kernel-spec-20260922e.md` |
| **P4** draft-side INT4 (native-MTP subgraph → the draft-INT4 pair) | **RESEARCH — the campaign's biggest measured win** | +39.6…+48.6 % single-stream (publication-grade n=5, 2026-09-22) and +9.6…+31.4 % measured tokens/J, iso-power. Output-preserving (greedy byte-identical). Gated on XPU graph capture. Published to LocalMaxxing (see the duplicate-ingestion correction). Claims-audit fix list CLOSED | `p4-champion-ab-p8192-n5-…T213957Z/`, `…-p512-n5-…T214938Z/`, `research/2026/qwen38-draft-int4-patch-provenance-20260922.md` |
| **P5** small-M INT4 | **WITHDRAWN — a decision** (L3) | At M=1 the op sits exactly at its weight-read floor (54.78 µs vs ~54 µs). M≈4–16 folded into P3; its padding question is now answered (29.7 % at M=16, synthetic) | `perf-mulmat-mregime.txt`, `perf-k216-grouped-iq3-20260923T002622Z.txt` |
| **P6** long-context / mixed-load gates | **RESEARCH — classification COMPLETE** (RL-033; L8) | OV ≥196K is per-generation growth (one long gen passes, the second fails at 44 MiB free). Verified envelope ≤163,840; ≥196,608 upstream-blocked (#4483). Remaining: the ≥98 K draft-drop function (O5) | `p6-ov-oor-classification-20260921T102903Z-244970/`, `p6-ov-ladder-*` |
| **P7** scheduler / concurrency knee | **RESEARCH — promoted by its own measurement** (O8) | Aggregate 2.79× at B=32, stacking with P4. Operator deprioritised aggregate in favour of single stream ⇒ the depth beyond B=32 is only worth revisiting if the objective changes | `p4-champion-ab-p512-n1-…T151358Z/conc-B*.jsonl` |
| **P8** bounded quant screen | **RESEARCH — AWQ format branch CLOSED** (RL-034; L6) | Every community "AWQ" artifact declares `compressed-tensors`; the format the phase wanted is not what the ecosystem produces. The compressed-tensors W4A16 class serves fine (Q1). A further kill tonight: **verify-head INT4 retired (L10)** | `p8-awq-20260921T115440Z/`, `scripts/tmp/screen_verify_head_int4_divergence.py` |
| **P9** SpecForge drafter | **RESEARCH — designed, un-run** | The Nemotron DFlash transfer is fully specced (harness + 2 ported patches + fixed gates) with an honest prediction of **+7–11 %** = screen-only. A real structural correction landed along the way: **DFlash shares the target's head** (`has_lm_head: false`), the opposite of Qwen MTP's own-head case | `scripts/tmp/p9-nemotron-draftint4-ab-20260923.sh`, `patch9_nemotron_draft_*.py`, `models/nemotron/…/serving.md` |
| **P10** cookbook completeness | **COMPLETED 2026-09-23** | Catalog 26 → **32 records** (OV GenAI + Cascadia added as capability, their zero-record gap closed). Chart renderer written (`cookbook scripts/render-engine-charts.py`, stdlib-only) and the three merged engine charts SUPERSEDED by per-engine SVGs with vision-QA PASS. The stale 60.13 record's supersession banner is now actually applied. All 9 cookbook gates green | `cookbook docs/qwen38-27b/RE-MEASURE-TASKS.md`, `cookbook data/chart-inputs.v1.json`, the three `b70-qwen38-*-20260923.svg` |

## Extra entry points opened and closed tonight (not in P0–P10)

| Entry point | Status | Decider |
|---|---|---|
| Speculation-depth economics (k-sweep, k=3 clamp cost) | **CLOSED** (RL-042) | Depth is flat (k=5/6 ≤1.7 % and negative); the #54785 k=3 clamp adoption is **REVERSED by measurement** (canary PASS at both depths; the clamp costs +5.45 % — `clampk3-20260923T022938Z/`; keep MTP4) |
| Verify-pass (5th head) INT4 (D8) | **RETIRED** (RL-041; L10) | top-1 agreement 72.07 % vs a ≥99 % line; first-divergence 2–3 tokens vs >2000 |
| `ggml_fp32_to_fp16` overflow corruption | **FOUND + dispositioned** | Host-side icpx-contraction bug; device path clean. Draft-only upstream candidate |
| O9 external levers (vllm-xpu-kernels ≥0.1.15 sampler; #29107 IQ3 reorder) | **SCREENS RUNNING** | P11 early signal at p512: stock 77.58 vs upgraded 74.13 median (noisy, 58.5–86.9) — inconclusive until p8192 lands; P12 diff fetched (76 KB) |
| IQ3_XXS extension on #29245 (the P-3b redemption) | **IN PROGRESS** | `fg_stage_a<block_iq3_xxs>` + `fg_block_traits` + launch-switch case landed in `fused-gemm.cpp`; L2 parity evidence captured at 3 shapes (both K216 shapes + tail rows) |
| LocalMaxxing publication | **DONE, with a defect** | 8 records APPROVED — two sets of 4 (duplicate ingestion from a parallel session). Correction recorded; operator decides which set is canonical |

## What is deliberately NOT started (and why)

- **Capture-era flag A/B** (compile × capture × async × cap): fully designed with fixed gates
  (`CAPTURE-ERA-FLAG-AB-PLAN-20260923.md`) including the k=3 clamp factor D that the
  zero-GPU reanalysis already decided. Un-run because the GPU is committed to P11/P12/P-3b
  tonight; it is the next GPU campaign after them.
- **P7 depth beyond B=32** and **cross-card prefill/decode disaggregation**: operator
  deprioritised aggregate/serving-shape work in favour of single-stream levers.
- **P6's ≥98 K draft-drop function (O5)**: real work, but it needs a long-context window and
  is ranked below the prefill kernel and the external lever screens.

## The one-paragraph roll call

Ten phases, every one accounted for: two COMPLETED (P0, P10), three WITHDRAWN with their
killing measurements (P1, P2, P5), four in RESEARCH with a named decider each (P3 via the
#29245 extension, P4 published and claims-clean, P6 classified to its upstream bug, P7
measured and deliberately deprioritised, P8 closed on the format question), and one designed
and un-run (P9). Three extra entry points opened and closed tonight beyond the roadmap
(the k-sweep/κ-clamp economics, the verify-head lever, the FP32→FP16 finding), and three are
live (two external-lever screens, the IQ3_XXS extension, plus the operator-decision item on
the duplicate LocalMaxxing records). Nothing is silently untouched.
