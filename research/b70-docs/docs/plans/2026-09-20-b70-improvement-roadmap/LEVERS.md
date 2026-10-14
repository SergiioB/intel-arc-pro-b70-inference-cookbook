---
title: "LEVERS — every lever measured, with its verdict"
type: plan
date: 2026-09-21
status: final
---

# LEVERS — every lever measured on this fleet, with its verdict

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** One row per lever that was actually tested (or explicitly
closed) in this campaign. Effect sizes are the measured ones; the cell and engine are named because
a number without them is not a number. Evidence paths are relative to the repo root.

Reading rules: aggregate/concurrent numbers are never comparable to single-stream ones; client rates
are never engine rates; a lever marked **neutral** was measured and did not pay, which is a result.

**Qwen3.8-27B decision update — 2026-09-24:** the [corrected research plan](../../../research/2026/exl3-analysis-and-next-inference-levers-20260924.md) ranks only work applicable to the retained dense GPTQ C1 route. D11/D12/D14 and unchanged depth/verify-head retries stay closed. P-3b is an isolated K216 IQ3_XXS MoE win, not a Qwen27 prefill rollout. Rotated low-bit quantization remains compatibility research, not an adopted conversion; H4 remains blocked on this no-P2P topology. Older forward-looking sentences do not override these outcomes.

**CORRECTIONS 2026-09-29** (architect pass, spot-verified; full register in `research/2026/b70-breakthrough-architect-20260928.md` §D): the adopted EXL3 route's "24.6 tok/s" is a chunk rate, not tok/s (probe 1 re-derives); RL-072/F3's "87.7 % GEMM, no hidden waste" is active-kernel share over a ~58.4 ms round with ~18 ms uncovered — not wall share; the α economics are **+6.25 % per +0.05** (counter form 1+4α) or +9.25 % (geometric), not the "+8.3…8.8 %" previously recorded here; the draft byte floor is **3.498 GB/round** incl. fp16 g128 scales (3.39 GB was payload-only) and "at exactly pin rate" is model arithmetic, not measured DRAM traffic; per-position survivals are window/content-dependent and bimodal, not a stable head property. Verdicts unchanged; do not quote the superseded figures.

## Decode — single stream

| # | Lever | Route / cell | Measured effect | Verdict | Evidence |
|---|---|---|---|---|---|
| D1 | **Draft-side INT4** (draft LM head + MTP body quantized to W4A16 g128, target untouched) | Qwen3.8-27B GPTQ-INT4 + MTP4, champion stack, C1 p8192/g128 (calibrated) | **Publication-grade n=5 (2026-09-22): 48.81 → 72.55 tok/s post-first = +48.6 %** (p8192, 8194 actual tokens); **52.61 → 73.43 = +39.6 %** (p512, 517 tokens). Earlier runs: 55.61 → 80.32 (+44.4 %), 52.83 → 77.14 (+46.0 % n=3), 52.99 → 72.68 (+37.2 %); reproduced 80.6/78.3 with both arms patched | **WIN** (claims fix list closed 2026-09-23) | `…/p4-champion-ab-p8192-n5-…-20260922T213957Z/`, `…-p512-n5-…-20260922T214938Z/` |
| D1a | Draft-side INT4 — **measured efficiency** (tokens/J from energy telemetry) | same cells, 230 W configured cap | Draw unchanged (228.7 vs 229.5 W p8192; 224.5 vs 228.6 W p512) while work rises ⇒ **+9.6 % tokens/J at p8192** (0.0596→0.0654), **+31.4 % at p512** (0.1952→0.2566). Window includes 5-request prefill | **WIN** (iso-power; measured draw, not cap) | same roots, `power.json` per arm |
| D2 | Same lever, **output preservation** | same, greedy + seeded sampling + 30-draw distributions | greedy **byte-identical** to the unpatched arm; seeded arms identical; **same modal class on 6/6 prompts** (30 draws each) | **WIN** (quality side) | worklog P4 §16–§18 |
| D3 | XPU **graph capture** (the enabler) | local image, same cell | with capture off the same two patches measured **≤1.5 %**; with capture on **+46 %**. Capture itself: eager 25.40 → compiled 42.76 tok/s = **1.68×** | **WIN** | `p4-draftint4-ab-*` vs `p4-champion-ab-*` |
| D4 | Draft-head INT4 **alone** / draft-MTP INT4 **alone** | local image, p512 | −0.5 % / +0.7 % compiled — individually inert | **NEUTRAL alone** (only the pair pays, and only with capture) | `p4-draftint4-ab-*` |
| D5 | K216 routed-IQ3 **device-resident expert IDs** | K216 L5 dual B70, C1 p512/g128, ABBA n=6/build | clean 9.25 → patched 9.577 engine tok/s = **+3.54 %** | **LOSS vs its ≥10 % gate** (kept: it unblocks SYCL graph capture for `MUL_MAT_ID`) | `p1-k216-e2e-ab-20260921T081155Z/` |
| D6 | MTP depth beyond 4 | recorded | depth screen closed at MTP4 | **LOSS** (no re-run; a retune without a changed implementation is banned) | `registry/model-status.json` |
| D7 | DFlash2 drafter against the INT4 target | recorded | ~25 vs 51 tok/s; selector/eager defects root-caused upstream | **LOSS** | RL-027…RL-029 |
| D8 | Verify-pass (5th) LM-head INT4 | Qwen3.8-27B, real 248320×5120 head vs the exact draft INT4 g128 quantizer (2.37 → 0.61 GiB = −9.0 % bytes, the premise held) | **quality gate FAILED outright**: top-1 agreement **72.07 %** on 4,096 hidden states vs a ≥99 % promotion line (fp16-matmul noise floor 99.80 % ⇒ ~140× more modal flips); top-32 overlap **81.7 %** vs 99.85 %; median first-divergence **2–3 tokens** vs >2000 | **LOSS — RETIRED.** Weight-only INT4 on the verify head moves emissions (it *is* the target distribution), matching the fp8 convention of excluding `lm_head` | `scripts/tmp/screen_verify_head_int4_divergence.py` |
| D9 | **MTP depth k-sweep (γ=3…6)** — zero-GPU reanalysis on tonight's measured-window acceptances (Leviathan speedup formula, c_d=1.5 ms, c_v=28.8 ms) | all four champion cells | **k=4 is within ±1.4 % of optimal at every measured α** (0.4873–0.5676); γ=5/6 strictly −1.5…−6.6 %; only the α=0.45 corner prefers γ=3 (+2.1 %) | **LOSS — killed at zero GPU cost** (wave-2's ≤1.5 % prediction confirmed flat). Real lever: acceptance **+6.25 % per +0.05 α** (counter form 1+4α; geometric per-position +9.25 % — corrected 2026-09-29: the earlier "+8.3…+8.8 %" conflated the two α definitions). Side product priced in the arithmetic: the #54785 clamp looked nearly free — **but MEASURED 2026-09-23: k=3 costs +5.45 % throughput** (k4 75.04 vs k3 71.16 tok/s, n=5/arm) and the canary says the corruption is not even live → **ADOPTION REVERSED, keep k=4** (see D10) | `worklogs/SPEC-ECONOMICS-20260923.md` |
| D10 | **#54785 k=3 clamp + spec-canary** (measured 2026-09-23, champion stack + draft-INT4 pair, C1 p512/g128, n=5/arm) | one variable = `num_speculative_tokens` | Canary: **PASS at BOTH k=3 and k=4** — 8 identical greedy outputs each, and the same hash across depths (`1c650a7f…`): the #54785 corruption class is **NOT live on this stack** (the published greedy-determinism premise holds). Throughput: k4 **75.04** vs k3 71.16 tok/s = the clamp costs **+5.45 %** | **LOSS for the clamp — adoption REVERSED** (its zero-GPU arithmetic said −1.4…+0.5 %; measurement says −5.5 %: draft cost is not a fixed 1.5 ms at k=3). Keep MTP4. The canary (`scripts/tmp/spec-canary-20260923.py`) is adopted as a boot/campaign gate anyway | `results/2026/09/b70-improvement-probes-20260920/clampk3-20260923T022938Z/` |

## Decode — concurrent / serving

| # | Lever | Route / cell | Measured effect | Verdict | Evidence |
|---|---|---|---|---|---|
| S1 | **Concurrency B=1 → 4 → 8** | champion stack, C1 p512/g128, 8 requests/level | baseline config 48.45 → **115.70** (B=4, **+139 %**) → 119.82; draft-pair config 61.86 → **140.30** (+127 %) → **149.79** (B=8). All outputs server-reported at 128 tokens | **WIN** (aggregate, not single-stream) | `p4-champion-ab-p512-n1-…T143215Z|T143819Z/conc-B*.jsonl` |
| S2 | Deep knee (B=16/32) | same, 32 requests/level | **162.74** tok/s at B=16, **172.48** at B=32 vs 61.86 at B=1 = **2.79×**; per-request 72 → 29 tok/s, TTFT ~2.0–2.6 s [CORRECTED 2026-09-29 (N1 review addendum): citation is `…T151528Z`, not `T151358Z` (that dir is a C1 arm with no conc files); the aggregates are **span-diluted** — steady-state implied ~**1,050 tok/s** at B=32 (32×32.8 per-request, T_step 88.7 ms); "2.79×" is an artifact of serial-prefill-contaminated spans, not a saturation measurement — the above-B=32 headroom conclusion survives] | **WIN** (diminishing, not saturated; magnitude corrected 2026-09-29) | `p4-champion-ab-p512-n1-…T151528Z/conc-B*.jsonl` |
| S3 | Long-context interaction with the knee | not tested | — | **OPEN** (P7) | — |
| S4 | Scheduler / speculation economics | not tested | — | **OPEN** (P7, now promoted) | — |

## Prefill

| # | Lever | Route / cell | Measured effect | Verdict | Evidence |
|---|---|---|---|---|---|
| P-1 | **Ubatch 512 → 1024** | K216, p7000/g1, `-c 8448` | 101.20 → **130.78 pp = 1.292×** (criterion fixed at ≥1.15× before measuring) | **WIN** | `p3-prefill-ub-20260921T085017Z/` |
| P-2 | Ubatch beyond 1024 (2048/4096) | K216, p7000/g1 | **2048: +0.2 %** (131.10 vs 130.78 pp) — saturated; **4096: does not load** (`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`) | **LOSS / exhausted** (the whole ubatch win was the 1.292× at 1024) | `p3-prefill-ub-20260921T152807Z/` |
| P-3 | **Grouped-IQ3 kernel, as-built E0a draft** (tile-local recon, no whole-expert FP16 materialization) | K216, isolated op n=512, both shapes | **1.39× SLOWER gate/up (39.36 vs stock 28.37 ms) and 1.30× slower down (39.28 vs 30.13)** against a ≤14.4 ms (≥2×) gate; n=32 2.4× slower; n=1 parity. Correctness fully proven: parity **33,554,432 values / 0 mismatches** vs `dequantize_row_iq3_xxs`, **0 spills** (AOT ocloc `bmg`), 6/6 eval rows NMSE ≤5e-4 | **LOSS as built — the ≥2× gate FAILED.** Kills the 7.25 GB round trip but pays ~34× the 1.141 ms weight floor in exposed latency/ALU. Named causes: ~8.7k non-persistent WGs (spec wanted 32 persistent), per-WG SLM table copies, no RAW staging (SLM 41.19 vs 49.25 KiB ⇒ spec F5 violation), serial recon gather. **Retained as the parity oracle/reference implementation** (`GGML_SYCL_IQ3_GROUPED_EXPERIMENTAL=ON` + `GGML_SYCL_IQ3_GROUPED=1`, fail-open) | `perf-k216-grouped-iq3-20260923T002622Z.txt`, `kernel-iq3-e0a/` |
| P-3a | **M-tile padding histogram** (decides the tile shape) | K216, synthetic uniform top-6, 3072 rows | **29.7 %** waste at M=16, 18.5 % at M=8, 8.9 % at M=4; T_e mean 14.2, range 4–25 | **NEUTRAL knowledge** (confirms spec R4's ≈29.5 %) — 11 points of issue is not the L9 failure cause, but the extension should derive M from real K216 histograms | `perf-k216-grouped-iq3-20260923T002622Z.txt` |
| P-3b | **IQ3_XXS extension on llama.cpp #29245** — the redemption path for the prefill lever after P-3's L9 failure | K216 MUL_MAT_ID IQ3_XXS n=512 | **IMPLEMENTED + ISOLATED-OP GATE PASSED (measured 2026-09-23, same-session ABBA):** `fg_stage_a<block_iq3_xxs>` (`fused-gemm.cpp`) + `fg_block_traits` + launch switches, bit-faithful reconstruct core (Gate 1: 33,554,432 FP16 values 0 mismatches). **GATE 2 MEASURED: fused 9,530 / 9,367 µs (mean 9,448) vs stock 33,929 / 34,041 (mean 33,985) = 3.60× same-session, 3.03× vs the registered 28,867 µs baseline; down 3.70×** — gate demanded ≤14,400 µs (≥2×): **PASSED with 1.5× margin**, distributions fully disjoint (A max ≪ B min). The fused path lands at ~9.4 ms — exactly the architect's projected 8–13 ms window and the E0a spec's 6–9 ms arithmetic, confirming L9 attributed E0a's failure to launch/scheduling structure (RAW staging + persistent WGs), not the datapath. Build note: first measurement attempt hit a CPU-only build (GGML_SYCL=OFF misconfig, 0 SYCL devices) — numbers above are from the correctly rebuilt SYCL build. **Decode no-regression rung PASSED with a bonus (recomputed from the same ABBA perf files): n=1 fused 163–173 µs vs stock 186–196 µs (mean 166.6 vs 192.2) = 13.3 % lower operator latency / 1.154× operator throughput (distributions disjoint: A max 173 < B min 186); n=32 1.47× operator throughput (5.3 vs 7.8 ms).** K216 e2e was canceled when the model was retired; this routed IQ3_XXS path does not dispatch for the dense GPTQ Qwen27 target | **WIN at isolated op (3.60×) — no end-to-end gain established** | `results/2026/09/b70-improvement-probes-20260920/p3b-fg-gate-20260923/` (manifest, perf-A1/A2/B1/B2, rebuild-abba.log) + `perf-k216-iq3xxs-ext-20260923T0115Z.txt` |
| P-4 | Dedicated prefill config `-b 8192 -ub 1024` | K216 | recorded 105.1 pp anchor (different cell from P-1; not comparable) | baseline | `research/2026/dsv4-k216-prefill-ubatch-20260825.md` |
| P-5 | OV long context | OV INT4-GDN8 u4 | 131,072 ✓ and 163,840 ✓ full protocol; **196,608 fails** — second long generation only | **classified** (upstream growth bug, not fleet config) | `p6-ov-oor-classification-*`, `p6-ov-ladder-*` |
| P-6 | Qwen3.6-35B mixed-load `causal_conv1d` | not testable here | artifact not on host | **BLOCKED** | `models/STATUS.md` |

## Quantization / format

| # | Lever | Route | Measured effect | Verdict | Evidence |
|---|---|---|---|---|---|
| Q1 | **Compressed-tensors W4A16** (the format the ecosystem actually ships) | local image, Qwen3.8-27B | loads, serves, 5,075 MiB free after load, 5/6 bounded suite | **WIN** (bounded) | `p8-awq-20260921T115440Z/` |
| Q2 | AWQ **format** route | all five community "AWQ" artifacts | every one declares `quant_method=compressed-tensors`, none declares `awq` | **LOSS / closed** (ecosystem overtaken) | worklog P8 addendum |
| Q3 | EXL3 | recorded | no runtime, BF16-class after reconstruction, ≤1.25× bound | **deferred, no-go** | `DEFERRED-AND-TRACKED.md` |
| Q4 | W4A8 on the dense 27B | recorded | quality divergence, discarded | **LOSS** | RL history |
| Q5 | AWQ at iso-bytes | not available | available artifacts are 1.079× the champion's language-only bytes | **BLOCKED** (needs a local conversion) | worklog P8 |
| Q6 | GPTQ-INT4 G128 (the champion representation) | all routes | the measured baseline everything else is compared against | baseline | `registry/model-status.json` |

## Hardware / rate class / placement

| # | Lever | Measured effect | Verdict | Evidence |
|---|---|---|---|---|
| H1 | **Sub-byte DPAS (OPS_PER_CHAN=8)** | s4×s4 **33.91 vs s8×s8 16.96 TOPS = 2.00×** per instruction at identical shape and chain depth; harness issue-limited, absolutes are floors | **WIN** (knowledge; no A4 serving path exists) | `dpas-rate-probe/FINDINGS.md` |
| H2 | Small-M specialization at M=1 | the op measures **54.78 µs against a ~54 µs weight-read floor** — arithmetic, not waste | **LOSS** (P5's M=1 object withdrawn) | `perf-mulmat-mregime.txt` |
| H3 | Small-M at M≈4–16 | n=8: 8.87 TFLOPS vs ≈17.3 at the floor ⇒ ≈2× of headroom | **OPEN — folded into P3** | `perf-mulmat-mregime.txt` |
| H4 | Dual-card TP2 | no P2P on this board; host-staged collectives required | **BLOCKED** (ADR-0008; independent stacks are the default) | `decisions/adr-0008-*` |
| H5 | Power cap as a speed lever | recorded flat for decode (byte-bound), not re-run here | **NEUTRAL** (not re-measured) | `docs/methods/observability-standard.md` |

## External levers — surveyed online 2026-09-22/23, NOT measured on this host

Ranked by expected payoff/effort; every number here is external and unverified here until it passes
the protocol gates. Source: `worklogs/RESEARCH-online-novel-levers-20260923.md` (all URLs inside).

| # | Lever | External signal | Effort | Verdict |
|---|---|---|---|---|
| X1 | llama.cpp PR **#29107** IQ3 sign/dot rewrite (reordered MMVQ layouts are Q4_K/Q5_K/Q6_K-only — NOT our IQ3_XXS MoE path) | **SCREENED 2026-09-23:** the dot rewrite alone (transplanted `iq_sign_mask`/`iq_sign_bits` + rewritten `vec_dot_iq3_xxs_q8_1`/`vec_dot_iq3_s_q8_1`) measured **−7.1 % at BOTH rows** (n=1 190.05→176.51 µs, n=32 6282→5834 µs; gate/up alone −11.4 %, down −2.8 %), parity clean 6/6+4/4. Below the 10 % promote bar at op level. **The L5 e2e decider then landed (full-port ABBA, n=6/arm engine tg): base 10.579 vs patched 10.587 tok/s = +0.08 % (distributions interleaved), prefill −1.34 % ⇒ REJECT overall (L11).** The external +75…95 % is a dense-model effect (all decode weights reorder); K216's decode is the routed `MUL_MAT_ID` path the PR does not reorder. Do NOT quote the external numbers (wrong workload class) | low (transplant done in `~/llama-iq3-reorder`) | **SCREENED — operator improved, e2e pending de-confliction** | `results/2026/09/b70-improvement-probes-20260920/p12-iq3-reorder-20260923T013116Z/` |
| X2 | vLLM **#53997** graph-safe grammar-bitmask copies; #53989 Qwen fused QK-norm+RoPE+gate dispatch | **#53989 SCREENED 2026-09-23:** corrected n=6/arm screen p512 -10.97%, p8192 +11.68%; ranges overlap at both cells, acceptance shifts -0.0663/+0.102 exceed the ±0.03 guard; 69–78°C measurements violate the ≤55°C cooldown rule. No promotion; dispatch was not traced. Equivalent dispatch merged upstream as **#56096**. | low-medium | **NO PROMOTION / no duplicate PR**; #53997 remains a separate structured-output graph-safety patch question |
| X3 | **vllm-xpu-kernels ≥ v0.1.15** fused top-k/top-p sampler + per-row RNG + compact sampling-mask | **SCREENED 2026-09-23: KILL — it is a REGRESSION on this stack.** 0.1.12.3 vs 0.1.15.4, one variable (wheel bump in-entrypoint, no derived image): p512 77.58→74.13 (**−4.4 %**), p8192 80.81→69.11 (**−14.5 %**), measured-window MTP acceptance 0.617→0.536. Correctness gate PASSED (greedy byte-identity 3×2, identical sha256) so this is a real perf regression, not a numerics break. Prefix-cache delta 0.0 all four cells | low (wheel bump) | **LOSS — KILLED** | `results/2026/09/b70-improvement-probes-20260920/p11-kernels-bump-20260923T004751Z/` |
| X4 | llama.cpp **#28725** SYCL graph record/replay + `SYCL_GRAPH_FORCE_NATIVE_RECORDING=1` | +19.8 % decode on a MoE (external); upstream **excludes MUL_MAT_ID "due to a memcpy+wait"** — i.e. our device-ID repair is the named prerequisite — which is **NOT satisfied as assumed** (corrected 2026-09-29): #28476/`304665fe7` added IQ MMVQ dispatch only; local `ggml-sycl.cpp` still rejects `ids->ne[1] != 1` and the multi-token path keeps `memcpy+wait` (D5's speed-gate result unchanged) | medium | AFTER O3 (same tree) |
| X5 | llama.cpp issue **#27593** cheap screens: `-DGGML_SYCL_F16=ON` (external 3.72× dense prefill), `SYCL_UR_USE_LEVEL_ZERO_V2=0` (+1.4 % tg), `GGML_SYCL_FA_ONEDNN` (long-ctx prefill +49 %) | build/env matrix on K216 | low | **SCREEN** — dense-skewed, keep gates |
| X6 | Draft-side quantization literature (Leviathan losslessness; QSpec W4A4-draft/W4A16-target; QuantSpec) | validates D1's distribution-preservation argument in the write-up | none | **CITE** in the upstream draft |
| — | sglang-xpu on B70 (garbage-output bug), Triton grouped GEMM on XPU, stock dual TP2, dense-model SYCL graphs, waiting for Xe3P | do-not-attempt confirmed in the survey | — | **DO NOT ATTEMPT** (merged into the standing do-not-retry list) |

| X7 | **Cascadia PP2 on Qwen3.8-27B** (same-source 1-stage control) | Functional screen PASS; same synthetic greedy p1692/g128 n=3: PP2 4.267 vs 1-stage 4.278 tok/s = **−0.26%**; prior Qwen2.5-1.5B PP2 was −18%, different model, not comparable. Probe API/relay plaintext; Cascadia binds probe 9101 to `0.0.0.0` even when API/relay are loopback | **No speed win verified; PP2 is not a default recipe**. PP speed-retention claim remains open pending real workload + n=5; independent stacks remain fleet default (ADR-0008) | `worklogs/CASCADIA-PP-qwen38-screen-20260923.md` |

## Acceptance frontier (wave-3 research 2026-09-23) — the ranked α movers

Source: `worklogs/RESEARCH-acceptance-levers-wave3-20260923.md` (every claim URL'd). The economics
(D9) put α at **+6.25 % per +0.05 (counter form 1+4α; geometric +9.25 % — corrected 2026-09-29)** — still an order above any depth or kernel lever left. Per-position survivals are window/content-dependent and **bimodal** (0.98/0.98/0.12/0.09 on the D11 natural-text window vs ~1.00/0.96/0.88/0.88 on a p512 champion window): never a uniform-α decay curve.

| # | Lever | Expected Δα (external basis) | Effort | Verdict |
| D11 | **FastMTP-style retrain of the shipped MTP head** (position sharing, 1,500-sample corpus) | **FAILED; audit-corrected 2026-09-24.** Stock accepted/drafted 438/808 vs retrained 430/856: α **0.5421→0.5023 (−0.0398)**. Nominal C1 p512/g128, sequential same-day servers, n=5 client post-first samples: medians **73.68→71.83 tok/s (−2.51%)**. Positional percentages are marginal accepted-prefix survival per all draft rounds, not conditional accuracy. The older 77.26/α0.5537 comparator is the **D12 finalist greedy control**, not Run 48; its −7.03% comparison is historical only. Raw power/cache/config/thermal completeness limits prevent a publication-grade matched claim | training program | **CLOSED / REJECTED.** Keep stock MTP weights; no small-corpus retraining retry. Failure supports this program's no-go, not an all-model impossibility theorem. Evidence: `results/2026/09/b70-improvement-probes-20260920/d11-stitched-fastmtp1500-probe-20260924T172339Z/` · `worklogs/QWEN27-REPLAN-CLAIMS-20260924.md` |
| D12 | **Probabilistic draft sampling** (`draft_sample_method='probabilistic'`: sample from the draft distribution + full-logit probability-ratio test) | **SCREEN vs FINALIST:** screen (4 windows) measured α +0.0505 and ≈ +9.8 % window tps — but the **fresh n=5 finalist REFUTED it: α −0.0037, tps −3.86 %** (greedy 77.26 / α 0.5537 vs prob 74.28 / α 0.5500, n=5/arm, counterbalanced, both canaries PASS, flag confirmed engaged) | flag-level | **FAILED AT FINALIST — do not adopt; keep greedy drafting.** The screen's win was α noise inside the known 0.48–0.70 run-variable spread: at n=5 windows ±0.05 α is not resolvable without a much larger sample. Acceptance frontier remains stock MTP weights | `results/2026/09/b70-improvement-probes-20260920/d12-finalist-20260923T045006Z/` (screen: `…/d13d12-20260923T043549Z/`) |
| D13 | **Penalty audit** — presence_penalty 1.5 vs 0 under spec decode (the RFC #27691-era worry that logits-processor penalties fight spec verify) | **CLEARED — the penalty HELPS:** α(1.5) 0.546/0.5735 vs α(0) 0.5083/0.5603 across four counter-bracketed windows on one server → window-average **+0.025 for the penalty** (the hypothesized cost would have been ≥ 0.05 the other way) | request-only | **NO RECIPE CHANGE — keep presence_penalty 1.5** (mildly acceptance-positive; plausibly keeps the target out of repetitive tails). Not the source of the run-variable 0.48–0.70 α spread | same run dir |
| D14 | fp8 vs bf16 **draft-KV** (QuantSpec prior: α bit-insensitive to KV precision) | **REJECT — measured 2026-09-23: α went the WRONG way** (fp8 0.5812 → bf16 0.5394 = **−0.042** vs the +0.05 gate; throughput +3.5 % inside overlapping sample ranges 68.7–94.2 vs 63.7–89.8 tok/s, and bf16 KV costs bandwidth too). Both arms spec-canary PASS | **RESOLVED — hypothesis falsified; keep fp8 KV** (champion recipe unchanged). KV precision is not an acceptance lever | `results/2026/09/b70-improvement-probes-20260920/draftkv-20260923T040206Z/` |
| — | Early-exit verification | ∅ on this stack | — | **CLOSED by arithmetic**: ~58 % verify rows skippable at α=0.55 but decode is weight-read-bound (best case +4.8 %, realistic +2 %) |
| — | SPRINTER approximate verification | — | — | **DO NOT ATTEMPT** — lossy, breaks our published output-preservation property |
| — | EAGLE-3 / trees / LayerSkip self-spec | — | — | **BLOCKED**: no Qwen3.8 head/artifact, `parallel_drafting` excludes mtp, no XPU kernels |

**Correction — 2026-09-24:** external acceptance at a different depth, prompt distribution, or sampling policy is not evidence that retraining will help this route. D11 and D12 failed their adoption gates; stock MTP weights and greedy drafting remain selected. Depth stays closed absent a changed mechanism. Per-position counters must be distinguished from conditional acceptance probabilities and aligned to the same measured window before modeling yield.

## 2026-09-25 improvement-night verdicts (post-replan execution)

Champion recipe unchanged; **zero promotions**. The night converted the replan's open questions into
measured verdicts and closed single-card kernel/host hunting: the 44.8 ms decode round is at its
byte budget floor. Full record: [NIGHT-REPORT-20260925](worklogs/NIGHT-REPORT-20260925.md); ledger RL-061…RL-065.

| # | Lever | Measured effect | Verdict | Evidence |
|---|---|---|---|---|
| D15 | Host-prep metadata cache (gdn_attn build() C1 fast path; pre-committed 2026-09-24) | p512 **+1.51 %** / p8192 +3.27 % pooled medians n=6/arm vs the +4 % gate; canary + greedy byte-identity preserved; iso-power 226.5–229.3 W | **LOSS** — the 10.08 ms instrumented share was profiler-inflated; removable share ~1.5–3 % | RL-061 |
| D16 | L-B draft weight-stationary fusion | Draft bucket 5.5 ms / 12.3 % of the round = **the unavoidable 4×-read bytes floor**: re-reads stream at pin rate (3.39 GB payload ÷ 5.5 ms ≈ the 5.57 ms floor) [CORRECTED 2026-09-29: post-pair budget is 3.498 GB/round incl. fp16 g128 scales (0.875 GB/pass × 4) — "at exactly pin rate" is model arithmetic, no DRAM-counter measurement exists] and cannot be cached on-chip across sequentially dependent positions | **CLOSED — physically inapplicable** (mechanism corrected; see NIGHT-REPORT correction block) | RL-062 |
| D17 | L-C exact verify-head kernel | Head 4.2–4.7 ms at its BF16 bytes floor (s ≥ 1.9 implausible) | **CLOSED at floor** | RL-062 |
| X8 | vLLM #56096 fused QK-norm+RoPE+gate | Gate exists only in `qwen3_next.py` + `is_cuda()`; this route is `qwen3_5.py` | **NOT APPLICABLE** — corrects RL-054's "equivalent merged" note; reopen only on an upstream qwen3_5 XPU implementation | RL-063 |
| P-7 | `max-num-batched-tokens` C1 TTFT (p8192/g1) | 4.679 / 4.689 / 4.683 s across 4096/8192/16384 = ±0.25 % | **Non-lever** — keep 8192 | RL-064 |
| H4a | Direct-peer P2P — capability discriminator (the §4.3 second decider) | Bidirectional ACCESS-flag copies **~11.1–11.9 GB/s, checksum-clean**, 2.02× host-staged control; **ATOMICS flag absent** | **CAPABILITY PASS** — TP2 stays blocked pending copy-based collective correctness + a gated matched-control campaign (ADR-0008); honest modeled band ~1.4–1.6× decode | RL-059 |

**Net position:** single-card C1 decode on this route is closed at its byte floor. The remaining live
lines are O14 (copy-based collectives → gated TP2), O13 (record-closing unitrace capture, ≤2–4 ms
expected slack), and O15 (operator thermal-window decision — the ≤55 °C gate is unreachable while a
server is loaded, RL-065, so night deltas are diagnostic-grade only).

## 2026-09-26 final verdicts (follow-on program, RL-070…072)

The confirm/deny pass after the operator-authorized window. Full record: [NIGHT-REPORT2](worklogs/NIGHT-REPORT2-20260926.md).

| # | Lever | Measured effect | Verdict | Evidence |
|---|---|---|---|---|
| F1 | flash-next C1, as placed (RL-055 dual-card split + CPU experts) | **First valid measurement (attempt #14)**: engine tg 21.3–21.9 tok/s, cold input 185–195 tok/s (p512/g128 n=3, calib VERIFIED, coherence PASS, peaks 64/60 °C) | **MEASURED — capacity route, NOT a speed route** (3.5×/9× slower than the qwen27 champion); speed work needs a different placement priced against these anchors | RL-070 |
| F2 | Copy-based cross-card collectives (O14 decider) | **Bitwise-exact** int64 + f32-rel-0.0 allreduce on BOTH transports; 101.7 µs @4 KB … 662.9 µs @4 MB; 128×51 KB serial round = 14.31 ms | **CONFIRMED — TP2 campaign GO.** Pre-declared gate +20…50 % decode, matched 1-card control, graphs-off, backend **xccl**, ADR-0008 | RL-071 |
| F3 | Body-bucket record-close (O13 decider) | Launcher-mode unitrace sees graph-replayed kernels: final round = **87.7 % GEMM (563 × ~1.1 ms), rest ~5 ms — no hidden waste** [CORRECTED 2026-09-29: active-kernel share, not wall share — 553 GEMM events ≈ 35.2 ms of a 58.35–58.59 ms round, ~18 ms uncovered; "no hidden waste" withdrawn as proven] | **CLOSED — single-card kernel hunting over** (byte floor + rate ceiling + visible composition; the composition level subject to the correction — register rows 2/13) | RL-072 |
| F4 | Idle-heat wall (RL-065/069) | PCI cycle executed: 72→44 °C, stale 2800 readout cleared, suspend accrues when unpollled; flash-next then ran cool | **FIXED** — standing rules: check card-node holders, no 1 Hz energy polling on sleeping cards | RL-069/072 |

**Terminal state of the single-card qwen27 program:** every decode and prefill lever is measured
closed. The one open lever on this host is the **gated TP2 campaign** (F2's follow-on, honest band
~1.3–1.5×). T3 (rotated-W4 quality rung) remains spec'd and storage-gated on the operator's
~55 GB BF16 re-fetch decision.

## How to read the whole table

Three levers won big and stack: **D1** (draft-side INT4, +39.6…+48.6 % single-stream, publication-grade
n=5 as of 2026-09-22, and iso-power so also +9.6…+31.4 % measured tokens/J — **D1a**), **D3** (graph
capture, without which D1 is invisible, and worth 1.68× by itself), and **S1** (concurrency,
~2.3–2.4× aggregate). One lever is the biggest unbuilt win: **P-3** (the routed IQ3 prefill operator,
~25× off the card's own dense GEMM), whose contribution shape is now **P-3b** (a minimal IQ3_XXS
extension on the open upstream #29245, which already implements tile-local reconstruction for two
other formats). Everything else either was a measured null, was closed by the ecosystem, or is
blocked on an artifact. The losses are as load-bearing as the wins: each one retired a multi-session
program that the original roadmap had budgeted on arithmetic rather than measurement.
