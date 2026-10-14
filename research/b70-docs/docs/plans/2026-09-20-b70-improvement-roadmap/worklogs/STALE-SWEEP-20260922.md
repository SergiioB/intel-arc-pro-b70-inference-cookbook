---
title: "Proactive stale-number sweep — 2026-09-22"
type: worklog
date: 2026-09-22
status: living
---

# Stale-number sweep 2026-09-22 (read-only audit, no fixes applied)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** All current numbers below are provisional
2026-09-21/22 measurements.

Scope: `AGENTS.md` (§2, §16, §17, §18), `docs/`, `research/2026/`,
`models/**/serving.md` + `README.md`, `campaigns/`, `decisions/`,
`registry/*.json`, `README.md`, `DOCUMENT-MAP.md`, `docs/indexes/`,
`docs/plans/2026-09-20-b70-improvement-roadmap/` (all phase files + worklogs +
`OUTCOMES`/`LEVERS`), the public cookbook
(`~/intel-arc-pro-b70-inference-cookbook`), the vendor source
(`~/projects/intel-arc-pro-b70-inference-cookbook`), and the XeCores
site checkout (`~/projects/xmxlab`, `sync:cookbook` in
`package.json`). No GPU workload, docker, or server was run. No repo file was
modified.

Authoritative current numbers used as ground truth: champion draft-INT4 pair
(calibrated p8192/g128 n=5: 55.61 → 80.32 = +44.4 %; p512/g128 n=3:
52.83 → 77.14 = +46.0 %; first n=5: 52.99 → 72.68 = +37.2 %; both-arms-patched
control 80.61/78.27; XPU graph capture is the enabler, off → ≤1.5 % inert,
capture alone ≈1.68×; TTFT 308–315 → 281–290 ms at p512; acceptance
run-variable 0.55–0.69 patched vs 0.53–0.58 baseline); K216 decode 9.25 clean /
9.58 patched (8.87 SUPERSEDED); device-ID repair +3.54 %, ≥10 % gate FAILED;
prefill routed op 28.87 ms/op gate-up (1.79 TFLOPS) / 29.91 ms/op down at n=512
(28.28/30.02 SUPERSEDED); dense MUL_MAT 46.76 TFLOPS (~25× gap); ubatch
512 → 101.20 pp, 1024 → 130.78 (1.292×), 2048 → 131.10 (saturated),
4096 → OOM; Nemotron Mamba2/SSD fusion WITHDRAWN (fusable 8 % vs 73 %
MoE+GEMM), restricted re-measure 252.30 post-first / 178.55 out-throughput with
the `1da0a954-det0123` tag absent; M=1 specialization DEAD (54.78 µs vs ~54 µs
floor); s4×s4 = exactly 2.00× s8×s8 (33.91 vs 16.96); OV envelope ≤163,840
verified, ≥196,608 upstream-blocked per-generation growth (#4483); every
community "AWQ" declares `quant_method=compressed-tensors`; EXL3 NO-GO; W4A8
discarded; AWQ iso-bytes BLOCKED (1.079×: 20.10 vs 18.63 GB); 0.25 ms/draft-step
DEAD (3.39 GB vs 152 MB = 22.3× over).

Severity key: **WRONG-NUMBER** (contradicts a current measurement) /
**SUPERSEDED-METHOD** (method/premise overturned but old wording live) /
**MISSING-QUALIFIER** (number right, required coordinates absent) / **TAXONOMY**
(engine merge or class violation) / **OK** (checked, current).

## Findings table

| # | file:line | stale/under-qualified text (exact quote, trimmed) | what it should say | severity |
|---|---|---|---|---|
| 1 | `AGENTS.md:334` | "PROVISIONAL best 8.87 engine tok/s at L5" (headline position; the 9.25/9.58 re-measure appears only parenthetically later in the same sentence) | Headline decode wall is 9.25 clean / 9.58 patched engine tok/s (re-measured 2026-09-21); 8.87 is superseded | WRONG-NUMBER |
| 2 | `AGENTS.md:334` | "IQ3_XXS is omitted from fused SYCL MUL_MAT_ID MMVQ upstream at 1692f9e50, but the pinned tree … already carries uncommitted device-ID cases" | Correct as qualified in this file, but every downstream copy that drops the pinned-tree clause (findings 3–5) is false for this host | SUPERSEDED-METHOD (downstream copies) |
| 3 | `engines/llamacpp-sycl/README.md:22` | "best 8.87 tok/s — decode wall is the fused-IQ3 MUL_MAT_ID MMVQ gap" | 9.25 clean / 9.58 patched (2026-09-21 re-measure); device-ID repair measured +3.54 %, gate failed, withdrawn as speed lever | WRONG-NUMBER |
| 4 | `engines/llamacpp-sycl/README.md:45` | "IQ3/IQ4-class quants omitted from fused SYCL MUL_MAT_ID MMVQ → per-token D2H expert-ID copies + host GEMV (the K216 wall)" | Unqualified: true upstream at 1692f9e50, FALSE for the pinned tree (device-ID cases present uncommitted); and the op is 5.52× above its weight-read floor — kernel efficiency, not sync | SUPERSEDED-METHOD |
| 5 | `research/2026/dsv4-k216-request-to-output-pipeline-20260825.md:20,173,209,247,286` | "median 8.87 engine tok/s (n=3)" table + "~84 blocking waits/token" + "8.87 tok/s ≈ 113 ms/token" (5 occurrences) | Dated 2026-08-25 note: 8.87 superseded by 9.25/9.58; wait-count framing superseded by the +3.54 % ABBA (gate failed) and the 5.52×-above-floor efficiency reading | WRONG-NUMBER |
| 6 | `research/2026/dsv4-k216-dspark-and-public-numbers-20260825.md:22` | "best decode: L5 … median 8.87 predicted tok/s" | 9.25 clean / 9.58 patched | WRONG-NUMBER |
| 7 | `research/2026/dsv4-k216-buun-moe-cache-20260825.md:25,52,86` | "best no-spec 8.87 engine tok/s (n=3, L5)" / "L5 8.87" / "Compare later C1 p512/g128 n=3 against L5 8.87" | 9.25 clean / 9.58 patched baseline for any new comparison | WRONG-NUMBER |
| 8 | `research/2026/dsv4-k216-prefill-ubatch-20260825.md:27` | L5 row "8.87" | 9.25 clean / 9.58 patched | WRONG-NUMBER |
| 9 | `research/2026/megakernel-xe2-xmx-architecture-guide-20260829.md:131` | "Target: 8.87 → 20+ tok/s" | Baseline 9.25/9.58; target range itself (~9.6–18 tok/s architect range) also overstates the top — re-derive from 9.58 | WRONG-NUMBER |
| 10 | `research/2026/megakernel-implementation-roadmap-20260829.md:97,148,200` | "throttled at 8.87 tok/s" / "jump from 8.87 tok/s → 18–25+ tok/s" / "8.87 → 20+ tok/s (2.3×)" | Baseline 9.25/9.58; the ≥10 % device-ID gate FAILED so the 2.3× premise needs re-basing on the fused-IQ3 kernel, not the sync repair | WRONG-NUMBER |
| 11 | `research/2026/b70-improvement-probe-baseline-20260920.md:28` | "8.87 engine tok/s (n=3)" as the P0 K216 baseline | 9.25 clean / 9.58 patched (re-measured 2026-09-21 during the same phase) | WRONG-NUMBER |
| 12 | `research/LEDGER.md:29` (RL-010 row) | "8.87 tok/s best, needs fused IQ3 MMVQ" | 9.25/9.58; device-ID repair +3.54 %, gate failed, withdrawn as speed lever | WRONG-NUMBER |
| 13 | `docs/plans/2026-09-20-b70-improvement-roadmap/P3-k216-grouped-iq3-prefill.md:23` | "gate/up 28.28 ms/op (1.82 TFLOPS) and down 30.02 ms/op (1.72 TFLOPS) at n=512" | Recaptured baseline 28.87 ms/op (1.79 TFLOPS) gate-up / 29.91 ms/op down (`perf-k216-prefill-n512.txt` as re-run 2026-09-21) | WRONG-NUMBER |
| 14 | `docs/plans/2026-09-20-b70-improvement-roadmap/OUTCOMES.md:24` (W4) | "gate/up 28.28 ms/op (1.82 TFLOPS), down 30.02 ms (1.72 TFLOPS) at n=512" | 28.87 ms/op (1.79 TFLOPS) / 29.91 ms/op | WRONG-NUMBER |
| 15 | `docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/P3-20260921.md:59-60,66` | table "28.28 ms (1.82 TFLOPS)" / "30.02 ms (1.72 TFLOPS)" + "2 × 28.28 + 30.02 = 86.6 ms" | Same recapture (note: the same worklog §"ubatch lever" already cites the recaptured 28.87 — the file contradicts itself) | WRONG-NUMBER |
| 16 | `docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/ARCHITECT-next-levers-20260921c.md:35` | "28.28 ms = 49 GB/s effective" | Recompute from 28.87 ms/op | WRONG-NUMBER |
| 17 | `docs/plans/2026-09-20-b70-improvement-roadmap/P4-qwen38-native-mtp-subgraph.md:25` (first entry) | "TTFT unchanged" (p8192 local-image +11.0 % cell) + "the lever is regime-dependent: inert at short context, worth +11 % at 8192" | "TTFT unchanged" is FALSE at p512 (308–315 → 281–290 ms, claims audit); the long-context-regime reading is CORRECTED by the same file's §28 entry (gate = XPU graph capture, not context length) — but this earlier entry still states the overturned reading without a pointer | SUPERSEDED-METHOD |
| 18 | `docs/plans/2026-09-20-b70-improvement-roadmap/P4-qwen38-native-mtp-subgraph.md:28` | "TTFT unchanged" (p512 +46.0 % cell) | TTFT 308.4–315.4 → 281.0–289.9 ms (≈ −10 %); corrected in the claims audit, still live in the phase headline entry | WRONG-NUMBER |
| 19 | `docs/plans/2026-09-20-b70-improvement-roadmap/P4-qwen38-native-mtp-subgraph.md:27` | "TTFT identical" (fresh n=5 +37.2 % cell) | Same TTFT correction applies; verify per-cell before reusing | MISSING-QUALIFIER |
| 20 | `docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/P4-20260920.md:255,386,417` | "both draft patches are inert at short context" / "the pair is inert at short context" / "TTFT unchanged" | Overturned: inert only with graphs OFF (≤1.5 %); on the champion stack +46.0 % at p512. TTFT changed (see #18) | SUPERSEDED-METHOD |
| 21 | `models/qwen/qwen3.8-27b/serving.md:218` | Greedy k8 p1024/g128 table row "60.13 / 71.06 tok/s" presented as a live speed row | Declared superseded ("superseded by this same-protocol sweep for ranking purposes" per the P10 gap plan) but still live with no supersession annotation in the dossier itself | SUPERSEDED-METHOD |
| 22 | cookbook `data/benchmarks.v1.json:317/332` (`qwen38-fp8-tp2-k8-p1024-g128-c1`, `client_e2e_output_tps` 60.13) | Live catalog record with no supersession annotation | Owner decision pending (annotation wording or re-measure); still served as a current record | SUPERSEDED-METHOD |
| 23 | cookbook `cookbook/docs/qwen38-27b/ENGINE-COMPARISON-CTX-20260914.md:174-177` (via P10-gap-plan citation) | "banked E2 cells (53.42/60.13/33.52) superseded by this same-protocol sweep for ranking purposes" | The sweep cited (73.0@512→57.3@96K, n=3, different Pi-prompt protocol) is NOT like-for-like p1024 — the annotation overstates what supersedes the record | SUPERSEDED-METHOD |
| 24 | `models/qwen/qwen3.8-27b/serving.md:40-42` (draft-INT4 table) | p512/g128 row "52.83 → 77.14 (+46.0 %), n=3" without actual-token note in the table | Actual 447 endpoint tokens (word-count client, uncalibrated); n=3 screen-class only — the n=5 rows carry calibration, this one does not | MISSING-QUALIFIER |
| 25 | `models/qwen/qwen3.8-27b/serving.md:41` | "C1 p8192/g128 (first measurement) 52.99 → 72.68" without actual-token note in the table | Actual 6993 endpoint tokens, uncalibrated (vs 8189 calibrated in the row above); both labels pinned in GAINS worklog but not in the table | MISSING-QUALIFIER |
| 26 | cookbook `cookbook/docs/qwen38-27b/DRAFT-INT4-S-M1.md:74` (working tree, uncommitted §"Independent reproduction") | p512/g128 row "52.83 → 77.14 (+46.0 %), n=3" with no actual-447 note, no TTFT note | Same as #24, plus no mention that TTFT moved 308–315 → 281–290 ms; uncommitted text will publish the defect if committed as-is | MISSING-QUALIFIER |
| 27 | `registry/model-status.json:12` (Qwen3.8 `best_tok_s` 102.61) | "102.61, cell p512/g128, recommended non-thinking Run48 n=5" with no cache-state/GPU coordinates | Later prefix-on champion stack, GPU 0 — must not be blended with cache-off Run 40 or GPU 1 draft-pair arms (`not_comparable`, GAINS Anchor A); fields carry no cache/GPU coordinates | MISSING-QUALIFIER |
| 28 | `registry/model-status.json:54` (Nemotron `best_tok_s` 186.61) | "186.61, p2048/g128, n=5 client post-first 150W" with `open_gate: "Fused Mamba2 SSD block kernel"` | Fusion program WITHDRAWN 2026-09-21 (fusable 8 % vs 73 % MoE+GEMM); open_gate must be re-pointed (small-M/grouped-GEMM reassignment); 186.61 needs the deviation note (252.30/178.55 restricted re-measure, tag absent) | SUPERSEDED-METHOD |
| 29 | `models/STATUS.md:26` (Nemotron row) | "186.61 tok/s p2048/g128, 157.92 p8192/g128 (n=5, 150 W…)" + "Open gate: Fused Mamba2 SSD block kernel (GRF/SLM)" | Same as #28: gate withdrawn, reassigned to P5/P3 | SUPERSEDED-METHOD |
| 30 | `models/STATUS.md:23` (Qwen3.8 row) | "83.7 tok/s p512/g128 (Run 40, 230 W); with recommended sampling 102.61 tok/s Run 48 C1" | Run 40 matrix vs Run 47/48 prefix-on stack blended in one cell without cache-state separation (serving.md separates them; STATUS does not) | MISSING-QUALIFIER |
| 31 | `AGENTS.md:366` (§18 triage table, Hybrid SSM + MoE row) | "Dominant Bottleneck: Sequential Dispatch Lag (736 launches per token)" + "Primary Lever: Fused Mamba2 SSD block kernel in GRF/SLM" | Withdrawn: fusable family is 8 % of device time (medians 55–357 µs, large calls not tiny launches) vs 73 % MoE+GEMM; fragmentation budget already paid down by graph capture | SUPERSEDED-METHOD |
| 32 | `engines/vllm-xpu/README.md:13,47` | "102.61 tok/s p512/g128" with no cache-state/GPU coordinates | Same as #27: later prefix-on stack, GPU 0; not comparable to cache-off arms | MISSING-QUALIFIER |
| 33 | `engines/vllm-xpu/README.md:23-25` | "TP2 holds 57–65 tok/s decode from 32K to 98K" / "45.7 / 32.6 / 34.5 tok/s at 98K/131K/196K" with no metric-phase/n labels in the dossier | Cross-engine sweep cells; metric class (client post-first vs engine tg) and n not stated at the point of use | MISSING-QUALIFIER |
| 34 | `docs/reference/engine-support-matrix.md:44` | "grafted MTP5 speculative decode (79 tok/s @512, single-card champion)" with no metric/n coordinates | Metric phase (client post-first?), n, sampling, and cap absent at the point of use | MISSING-QUALIFIER |
| 35 | `engines/openvino-genai/README.md:43` | "79.5 tok/s @512 (single-card best)" with ≤64K qualifier present | Qualifier present — but no metric-phase (client post-first?) or n stated | MISSING-QUALIFIER |
| 36 | cookbook `cookbook/docs/qwen38-27b/ENGINE-COMPARISON-CTX-20260914.md` (full page) | Single tables/series mixing OpenVINO GenAI, vLLM XPU, and llama.cpp SYCL (e.g. lines 25, 37, 60–62, 77–78, prefill table) | 2026-09-15 chart standard forbids merged engines; page needs per-engine splits from evidence JSONs via a committed renderer | TAXONOMY |
| 37 | cookbook SVGs `b70-qwen38-engine-decode-vs-ctx.svg`, `b70-qwen38-engine-decode-int4-class-20260915.svg` (+ prefill twin), and one moe-int4 SVG (3 merged OV/Cascadia/llama charts per P10 audit) | Series "OpenVINO MTP5 vs vLLM MTP4 vs llama.cpp Q8_0 on one B70 each" in one chart | Same standard violation; cookbook has no committed renderer for the split (only `render-benchmark-catalog.py`), so hand-editing is not an option — recorded, not forced | TAXONOMY |
| 38 | cookbook `cookbook/docs/qwen38-27b/ENGINE-COMPARISON-CTX-20260914.md:8` | "OpenVINO's full-protocol ceiling is 128K" | Superseded: 131,072 and 163,840 full protocols PASS (2026-09-21); verified envelope is ≤163,840, ≥196,608 upstream-blocked | WRONG-NUMBER |
| 39 | cookbook `cookbook/docs/qwen38-27b/ENGINE-COMPARISON-CTX-20260914.md:227,271` | "128K on one card: no config fits" / "full-protocol requests run to 128K and fail at ≥196K" | Same envelope update: 163,840 verified working | WRONG-NUMBER |
| 40 | xmxlab vendored `data/cookbook-b70/docs/benchmark-catalog.md:20` | "60.13 token/s (client e2e output tps)" served to the site | Vendored copy of the stale record (#22); inherits the supersession defect at publish time | SUPERSEDED-METHOD |
| 41 | xmxlab vendored `data/cookbook-b70/docs/qwen38-27b-engine-comparison-ctx-20260914.md:174` + SVGs `b70-qwen38-fp8-tp2-dashboard.svg:23`, `b70-qwen38-fp8-tp2-sampling-collapse.svg:38,80` | "60.13" banked cells and rendered SVG values | Vendored copies of #22–23; SVGs bake the value into pixels (re-render required, not a text edit) | SUPERSEDED-METHOD |
| 42 | xmxlab vendored `data/cookbook-b70/docs/readme.md:33` | "186.61 C1 client post-first at p2048/g128 n=5" with no withdrawal/deviation note | Inherits #28–29; site copy will outlive any B70-DOCS fix unless re-vendored | MISSING-QUALIFIER |
| 43 | cookbook working tree (uncommitted): `cookbook/docs/qwen38-27b/DRAFT-INT4-S-M1.md` + `cookbook/docs/QUANTIZATION-AND-DRAFTERS.md` + 3 power-hygiene lines | New 2026-09-21 draft-INT4 section (80.32/77.14/72.68 table, enabler, output boundary, aggregate note) + AWQ-name + concurrency notes + "230 W cap" fixes — all uncommitted (`M` in git status) | Content itself is current, but (a) the p512 row lacks the actual-447/TTFT qualifiers (#26), (b) nothing is committed or pushed — public HEAD still serves the pre-2026-09-21 text | MISSING-QUALIFIER |
| 44 | `AGENTS.md:340,342` | Architect-range paragraph duplicated verbatim (two identical consecutive lines) | Editorial: delete one copy | MISSING-QUALIFIER |
| 45 | `models/STATUS.md:40` (Qwen3.8 SYCL lane, external cross-check row) | "2× Q4_K_M+MTP2 TP2 64.24" and "1× GPTQ INT4+MTP4 102.61 (vLLM XPU, non-thinking)" in one row | External-lab numbers (mndodd) share a row with the internal vLLM record without a lab/host separation stronger than prose — near-miss for the no-merge rule | TAXONOMY |
| 46 | `docs/operations/recommended-configs.md:31` | Nemotron "n=5 campaign measured 186.61 tok/s at p2048/g128" with no deviation note | Needs the restricted re-measure pointer (252.30/178.55, tag absent, GPU/cap deviations) per the dossier's own §"Restricted re-measure" | MISSING-QUALIFIER |

Checked and current (OK, no finding filed): `models/qwen/qwen3.8-27b/serving.md`
draft-INT4 section (PROVISIONAL label, enabler, acceptance run-variance,
validity limits, both-arms control all present); Nemotron dossier Mamba2
withdrawal + deviation disclosure; P6/OUTCOMES/LEVERS OV envelope + per-generation
growth; XMX rate classes (AGENTS §17, rate note, OV dossier row); P8 AWQ
closure + 1.079× bytes; P5 M=1 withdrawal; P4 §28 attribution correction;
LEVERS S1/S2 + serving.md aggregate-vs-single-stream guard; `DOCUMENT-MAP.md`,
`README.md`, `docs/indexes/`, `docs/methods/`, `docs/governance/`,
`docs/operations/current-host-state.md`, `research/LEDGER.md` (generated view —
8.87 lives only in the generated `research/LEDGER.md:29` rendering of RL-010,
not in `registry/research-ledger.json`), `social-assets/`, patch headers
(C1/TP1-only, greedy-identical scope).

Engine-taxonomy verdict: no B70-DOCS table merges OpenVINO GenAI / Cascadia /
OVMS / llama.cpp SYCL / vLLM XPU into one series — the support matrix keeps
separate rows and the Cascadia row carries the never-merge warning. All live
violations are the cookbook's (findings 36–37) plus the STATUS near-miss (45).

Naked-number scan: no bare tok/s figure without at least a cell was found in
B70-DOCS outside the items above; the systematic defect is missing
*coordinates* (metric phase, C1 vs Cn, cache state, configured cap vs measured
draw, n, PROVISIONAL label), concentrated in engine dossiers and STATUS rows
(findings 27, 30, 32–35). No client-rate-presented-as-engine-rate and no
aggregate-presented-as-single-stream instance was found — every aggregate
number checked travels with its aggregate label.

## Severity counts (46 findings, one severity each)

- WRONG-NUMBER: 17 (findings 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 38, 39)
- SUPERSEDED-METHOD: 12 (findings 2, 4, 17, 20, 21, 22, 23, 28, 29, 31, 40, 41)
- MISSING-QUALIFIER: 14 (findings 19, 24, 25, 26, 27, 30, 32, 33, 34, 35, 42, 43, 44, 46)
- TAXONOMY: 3 (findings 36, 37, 45)
- OK surfaces: listed above

## Top 10 WRONG-NUMBER findings (fix priority)

1. **#18 — P4 phase file still headlines "TTFT unchanged"** on the +46.0 % p512
   cell after the claims audit proved it false (308–315 → 281–290 ms). Highest
   visibility, directly contradicted.
2. **#1 — AGENTS §16 headlines "best 8.87"** for the K216 decode wall; current
   is 9.25 clean / 9.58 patched. The contract is the most-read file in the repo.
3. **#3 — llama.cpp SYCL dossier "best 8.87"** with the wall attributed to the
   old sync framing; same correction as #1 plus the +3.54 % gate-failed verdict.
4. **#13/#14/#15 — the 28.28/30.02 op figures live in three places** (P3 phase
   file, OUTCOMES W4, P3 worklog table) after the 28.87/29.91 recapture; the
   worklog even contradicts itself.
5. **#38/#39 — cookbook comparison page "ceiling is 128K" / "no config fits"**;
   163,840 full protocol passes. Public-facing and wrong.
6. **#5 — K216 pipeline note (5× 8.87 + wait-count framing)**; the note the
   contract cites as the full walk still teaches the overturned mechanism.
7. **#9/#10 — megakernel guide + roadmap baseline and 2.3× target off 8.87**;
   targets derived from a dead baseline.
8. **#11 — P0 probe baseline "8.87 (n=3)"**; the phase's own later re-measure
   supersedes its opening baseline.
9. **#12 — LEDGER RL-010 rendering "8.87 tok/s best"**; ledger readers inherit
   the stale baseline (source: generated view — fix at the RL-010 record or its
   renderer input).
10. **#16 — ARCHITECT worklog "28.28 ms = 49 GB/s"**; derived arithmetic off
    the superseded capture.
