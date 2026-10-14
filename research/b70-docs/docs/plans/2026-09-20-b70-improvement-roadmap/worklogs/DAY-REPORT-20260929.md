---
title: "Day report — breakthrough pass, corrections, and the 3bpw GO (2026-09-29)"
type: worklog
date: 2026-09-29
status: final
---

# DAY REPORT 2026-09-29

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** No commits, no publication, no X-draft (operator dropped it today). Ledger: RL-089…RL-095 filed. Everything below traces to on-disk evidence.

## The one-paragraph answer to "any huge wins?"

Three, honestly earned: **(1) the adopted route was 2.6× faster than we'd recorded** — the EXL3 true token rate is 49.2/63.3 tok/s (150/230 W, n=5, all gates PASS; the old "24.6" counted SSE chunks); **(2) the 30-defect is proven to live in the quantized weights, not the kernel** — both W4A16 export classes invert a +0.40-logit correct-answer margin into ≈−2.9 under exact fp32 on CPU, exonerating the XPU stack and naming margin-aware requant as the path back to fast 4-bit; **(3) 3bpw EXL3 is quality-proven** (battery 6/6 ×2 — the canary that killed W4A16 answers a clean "14"; code tasks 9/10, beating the 4bpw reference) — the K=3 kernel program is GO, with the full patch series authored and statically validated.

## The day's arc

1. **Research audit → breakthrough pass** (`research/2026/b70-breakthrough-architect-20260928.md`, substitute run after the architect's provider death): standing structure (ALWAYS-ON / ALWAYS-TRY / CLOSED — now AGENTS.md §20, graph capture rule #1), 16 first-principles candidates, top-3 probe designs, 13-row corrections register.
2. **Adversarial review** (`b70-breakthrough-review-20260929.md`): 7 of 10 surfaces NEEDS-FIX; B1 demoted (break-even precision 0.96; the only serving-shaped survival window is flat), B10 collapsed to +0.6 % net, B11 re-based to ~1.33–1.49× vs live EXL3, B16 rescoped (12.9 % addendum at deployed 65536), **N1 refuted by its own kill-probe** (fallout: the W10/S2 "2.79× at B=32" aggregates were span-diluted — steady-state implied ~1,050 tok/s; citations corrected). Corrections register applied with dated flags across 11 files; AGENTS.md §16's two wrong clauses fixed; §20 added.
3. **Probe 1 — EXL3 rate + attribution** (RL-090): corrected rate table above; H1 confirmed (chunks = MTP3 rounds); depth-flat 2K→16K; power-limited +25–29 %; attribution: ccs0 ~88 % busy, bandwidth 57–63 % of pin floor — the missing time is INSIDE kernels (**B6 conditional GO**; named-op split unresolved — needs an L0 tracer that survives this fork's oneCCL init). Cleanup verified.
4. **Probe 2A — the 30-defect discriminator** (RL-092): VERDICT `weights`. BF16 base +0.4017 (answers 14; control 60 correct); AutoRound-ext fp32 −2.9730 (30); arm-C auto_gptq fp32 −2.8794 ('3'). ~−3.3-logit systematic margin inversion, both packings, both quantizers, exact fp32 accumulation, no serving stack in the loop. Layer-lens: inversion established mid-network (deepest ≈−7.1 @ ~L60). ADR-0010 upgraded observed→proven; kernel exonerated; margin-requant (Task B, spec'd) inherits the −3.37 target.
5. **Upstream dig** (RL-089): Intel's "sym_int4 output fix" is MoE-router-only — NOT our defect; #53912 decomposed (6 mechanisms; our image predates every merged fix; **#50729** mamba conv-state race is the best zero-hit match); **A3 gate: K=3 (width 48) unsupported in the pinned image** — source `#ifdef` + binary symbol + artifact demand, three-level proof; llama.cpp pinned tree has neither #26689 nor #28796 (both are tree-bump programs, not flag flips).
6. **Engine capability audit** (RL-091): EP has no middle rung (coupled to flag-slot sync); GDN-state exploitation largely in-tree already; flag-slot AMBER (no hard blocker; reader-side visibility is the microbench question); **free fact: the qwen3_5 MTP draft is full-attention-only** (cross-device drafting needs no GDN-state sharing).
7. **Acceptance forensics A1–A5** (RL-093): bimodality is content-class-dependent, NOT universal (27/76 cliff, 38 gradual, 8 flat); the serving EXL3-MTP3 route is FLAT (0.79/0.65/0.46) — B1's premise was a retired-stack artifact; draft logits never logged (probe-3 Tier-2 is the unblock); **yield=1+kα validated within ±2.4 %** (the corrected +6.25 %/0.05 pricing is empirical now).
8. **3bpw quality-first trial** (RL-094): ADVANCED — battery 6/6 ×2, tasks 9/10, replay stable; Triton-path diagnostic 10.25 tok/s (not power-bound) = the kernel program IS the speed case; 3bpw MTP3 acceptance 0.882/0.811/0.337; **D5 load bug found** (ops.py hasattr guard blocks every sub-4bpw artifact on the default backend — upstreamable).
9. **K=3 patch series authored** (RL-095): six patches (P1 build + D2 fail-closed build fix; P1b D1 template repair — the naive flag-flip alone enables ILL-FORMED K=3/K=5 instantiations; P2 strip + sweep spec; P3 owned correctness test on real on-host tensors; P4 A/B spec ≥+10 % decode / ≤5 % prefill; P5 K-aware loader guard, 5 CPU tests pass). **D3 capture-safety is the single open gate** — architect decision in flight (graph-private pre-reservation vs caller-owned workspace).

## Incidents / deviations (disclosed)

- Provider deaths: b70-architect (quota), the general-purpose breakthrough substitute (timeout AFTER writing its file), probe-2A twice, the first kernel-engineer pass, the forensics agent — every one survived via skeleton-first discipline + salvage + resume; one orchestrator relay misattributed armC-bf16 output as arext-fp32 (caught by the specialist, corrected in the record).
- Host reboot ~13:23Z: killed /tmp (the source clone), reset power caps (the trial restored as-found 180/230 W), likely killed several CPU forwards mid-run (the "silent deaths" earlier), and the prior transient dockerd — the trial recreated the isolated dockerd per the documented stack and left it running empty.
- The corrections register (13 rows + review rows 14–15) applied with dated flags, originals preserved; `registry/model-status.json` carries the corrected 63.25 figure with the superseded 24.6 flagged.

## Ready-to-fire queue (next GPU windows, value order)

1. **B6 — in-kernel efficiency at 4bpw** (biggest lever: ~63 → 85–95 tok/s, zero quality risk). Prerequisite: an L0 tracer that survives this fork's oneCCL init (newer pti-gpu in-image, or a refreshed base).
2. **K=3 build + P3 bit-exactness window** — gated on the D3 decision; then the P4 matched A/B (≥+10 % decode vs 63.25; ≤5 % prefill regression).
3. **B11 flag-slot microbench** (60 µs kill threshold; no hard blocker found; the enabling primitive for EP and cheap cross-card sync).
4. **Probe-3 Segments A+B** (acceptance instrumentation on the serving route — the B1/B2/B10 evidence unblock; Tier-2 fields per the design).
5. **Task B margin-requant** (operator-gated): the path back to a fast 4-bit W4A16-class route; first target the −3.37-logit inversion; intact BF16 source on NVMe.

## Evening addendum — the execution windows (2026-09-29 late)

- **K=3 build + P3 window (RL-097)**: build PASS (360 K=3 instantiations, D1 repair live, fail-closed build), default-backend 3bpw load PASS (P5's K-aware routing fixes the trial's fatal bug end-to-end), **P3 FAIL-closed** on a single-cell dense excursion (k=6144→5120, M=8, bf16, seed 1701: abs 0.125 vs ~0.07 bound) → **NO-GO this window, exactly as the gate demands**.
- **Root-cause (appended to PATCH-SERIES.txt, substitute analyst)**: the "intermittency" was partly a harness artifact — the repro scripts re-seeded per cell, so the 50+ reruns never replayed the failing input (proven bit-exact from JSONL). Primary verdict: **deterministic input-dependent bf16 1-ulp boundary flip on one element** (split-K partial ruled out — a partial would smear ~128 columns via FWHT; the failing M=8 path is `DpasKernel<3,2,8,NT=4>`, not the D1-repaired vector path). Disposition: the FAIL stands; a NEW screen with the predeclared amendment `mx ≤ max(0.01+0.05·rms, 1 ulp)` + the deterministic generator-replay repro (n≥200 identical-input reps, ulp instrumentation) must clear it. Zero-tolerance trellis/one-hot gates unaffected.
- **D3 decided (architect-2)**: Option A, graph-private pre-reserved allocation — grounded in the production route already replaying this exact allocation pattern under 10/10 capture; D3a/D3b proof gates defined (not yet run — gated on P3).
- **B6 micro-probe (RL-098)**: in-kernel loss CONFIRMED (M=4 verify pass = 62.6 % of pin, zero engine/host) but **REDIRECTED** — lm_head/in_proj_qkvz/gate_up are at ceiling (≥90 % of achievable); the slack is in mid-N families and a **uniform M=4 DPAS defect (−8–15 points vs M=1 vector GEMV on identical bytes)**. Priced: ×1.26 decode → **63 → ~79 tok/s honest ceiling** (the 85–95 expectation is not reachable by kernel work alone). Next experiment pre-named: M≥3 DPAS-vs-vector A/B on down_proj.
- **Provider toll**: architect-1 + kernel-engineer ×2 died on gpt-6-astra quota; every task recovered via substitutes with zero work lost (six recorded substitutions today).

**Next GPU window menu (all priced, in value order):** (1) B6 DPAS-path fix on down_proj (the A/B patch) — biggest lever, ~+26 % decode; (2) the K=3 repro + predeclared-tolerance re-screen → D3a/D3b → P4 A/B (3bpw's ~77-ceiling stacks on B6's fix); (3) B11 flag-slot microbench; (4) probe-3 Segments A+B. CPU-side anytime: margin-requant (Task B).

## Night close (2026-09-29 late evening)

- **K=3 completion chain (RL-100)**: repro BOUNDARY_FLIP_CONFIRMED (bit-exact on the certified input, 1 bf16 ulp, one element, fp64-anchor 0.0, no race across 200 reps — the "intermittency" was the old probe's reseed defect); re-screen **PASS 7/7 families, 644/644 cells** under the predeclared 1-ulp amendment; **D3a PASS / D3b PASS** (zero post-capture allocator movement; no graph aliasing); **P4: quality PASS (6/6 ×2) but performance FAIL — 42.44 tok/s vs 63.25 = 0.671×** (gate ≥1.10×), TTFT 1.63×, acceptance identical ⇒ pure kernel-speed loss at unswept strip geometry. **3bpw = NO-GO this window; 4bpw route stands.** Reopeners: the P2 strip sweep + B6-class kernel work.
- **B6 MB=4 A/B (RL-101)**: build PASS (new MB=4 instantiations, bit-identical outputs, tail-stabilizing) but **PRIMARY GATE MISS** — +0.41 pts (down_proj) / +4.72 (qkv) vs the ≥5-both gate; ~+0.2 % decode vs the ~26 % target. **Tile padding was not the B6 mechanism.** P6 stays in the series as a correct local patch (tail stabilization, no adoption claim). The 9.07 ms M=4→M=1 gap remains open: split-K partials + had_out reduction cost, DPAS utilization at verify-tile intensity, the P2 strip-geometry sweep. Secondary: MB8 M=4 timings drift ±5 % cross-session while M=1 vector reproduces within 1–2 % — a measurement caveat for the class.
- **Night tally**: three levers killed cleanly by their own pre-declared gates (B1 early-exit, N1 GDN-knee, MB=4 padding), one adopted quietly (P6 tail stabilization), every question closed with a measured verdict. Ledger RL-089…RL-101 (13 entries today).

## Decisions taken (not to be re-litigated)

- Metric discipline: every rate names its counted quantity; chunk-rate ≠ tok/s (claims rule, §20.6). The old campaign's EXL3 speed rows are superseded diagnostics.
- Sub-4bpw route: prebuilt-variant quality-first (done, PASS), K=3 kernel extension (authored), NOT requantize-from-4bpw; SC variants deprioritized (more kernel widths for less savings).
- B1 closed on the serving route; B10 demoted; TP2 stays closed; #50729 base-refresh is a separate single-variable decision, explicitly out of the K=3 program's scope.
- No X-draft today (operator); no commits (operator's call on the ~8 modified + ~10 new files, inventory in `results/2026/09/b70-inventory-20260929/INVENTORY.md`).
