---
title: "Gains achieved, measured against current inference data"
type: plan
date: 2026-09-22
status: living
---

---
doc_id: PLAN-GAINS-VS-CURRENT-DATA
doc_type: plan_note
title: Gains achieved, measured against current inference data
lifecycle: active
created: 2026-09-22
updated: 2026-09-22
canonical_for:
  - gains-versus-registered-cells-reconciliation
---

# Gains achieved, measured against current inference data

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** The direct answer to "what have we
gained, and how does it compare against the data we currently hold": three real
gains (one large, one large-and-stacking, one prefill), each reconciled below
against the **registered** cells in `data/benchmarks.v1.json` — because a gain
that is not reconciled against registered cells is not a gain yet. One published
record is in conflict with the current measurement and needs an owner decision
(§4).

Engine rule: OpenVINO GenAI, Cascadia, OVMS, llama.cpp SYCL and vLLM XPU are
separate engines and never share a table or a chart series.

## 1. The gains, at their measured coordinates

| # | Gain | Measured cell | Effect | Metric (exact class) | Status |
|---|---|---|---|---|---|
| G1 | **Draft-side INT4 pair** (draft MTP LM head + MTP body → W4A16 g128; target untouched) | vLLM XPU champion stack (digest `f01e24f6…` + champion overlay + `VLLM_XPU_ENABLE_XPU_GRAPH=1` + `B70_MTP_BF16_DRAFT=1`), GPU 1, **230 W configured cap**, C1, recommended non-thinking sampling, 128/128 out | calibrated p8192 (8189 tok) g128 n=5: **55.61 → 80.32 = +44.4 %**; p512 g128 n=3: **52.83 → 77.14 = +46.0 %**; earlier n=5: 52.99 → 72.68 = **+37.2 %** | **client post-first rate** (tok/s) — not engine-native | Re-measuring at publication grade with energy/temp telemetry |
| G2 | **XPU graph capture** (the enabler without which G1 is invisible) | local image, same cells | capture alone **1.68×** (eager 25.40 → compiled 42.76); G1 with capture **off** = **≤1.5 %** (inert) | client post-first rate | validated |
| G3 | **Concurrency** (stacks with G1; separate metric class) | champion stack, 8 req/level (32/level at B=16/32) | aggregate B=1→B=4 **+139 %** (48.45 → 115.70 baseline; 61.86 → 140.30 patched); **172.48 tok/s at B=32 = 2.79×** | **aggregate generated tok/s** — never comparable to single-stream | validated (aggregate) |
| G4 | **K216 prefill ubatch** 512 → 1024 | K216 IQ3 MoE, p7000/g1, `-c 8448` | 101.20 → **130.78 pp = 1.292×** | llama.cpp engine `pp` | validated; the config lever is **exhausted** (2048 = +0.2 %, 4096 OOM) |
| G5 | **Sub-byte DPAS knowledge** | identical shape/chain, ESIMD `xmx::dpas` | s4×s4 **2.00×** s8×s8 MACs/instruction (33.91 vs 16.96, issue-limited so absolutes are floors) | probe TOPS | validated; **adds no TOPS vs an 8-bit operand** — a bandwidth/capacity lever |
| G6 | **OV ≥196K classified** | OV GenAI INT4-GDN8 u4 | failure is **per-generation growth**, not a capacity ceiling: one long gen passes at 196608, the **second** 64-tok gen fails at 44 MiB free | capacity/correctness | validated; upstream-report recipe for #4483 |
| G7 | **Format clarity** | five community "AWQ" artifacts | **all** declare `quant_method=compressed-tensors`; "AWQ" is a name here, not a format | knowledge | validated |
| G8 | **Output preservation of G1** | greedy + seeded + 30-draw | greedy **byte-identical** across arms (3 tasks × 2 repeats, same sha256); **same modal class 6/6 prompts** at 30 draws | correctness | validated (greedy + one sampler config) |

## 2. Reconciliation against the registered catalog (26 records)

The registered cells that bear on G1–G3. **None of them is a like-for-like cell
for the champion-stack measurement**, which is why the +37…46 % is reported as a
matched A/B ratio and not as a table position.

| Registered record | Registered number | vs our measurement | Comparable? |
|---|---:|---|---|
| `qwen38-27-draft-int4-p512-g128-c1` | **106.7** client post-first, p512/g128 C1, **cache: enabled**, n=5 median | our p512/g128 = **77.14** (n=3) / **80.32** at p8192 (n=5), **cache disabled**, recommended sampling | **NOT comparable** — cache state, sampling and prompt differ. See §4 |
| `qwen38-fp8-tp2-k8-p1024-g128-c1` | 60.13 client e2e, p1024/g128 C1, cache disabled-zero-hit, n=5 (greedy diagnostic) | different model (FP8-W8A16), **different metric** (client e2e vs post-first), different topology (TP2 dual) | **not comparable**; also the declared-stale record (§4) |
| `qwen38-fp8-tp2-localmaxxing-p74-g256-c1` | 31.1 client post-first, p74/g256 C1, n=3, cache disabled-zero-hit | FP8-W8A16 + MTP8, p74/g256 — a different model class and cell | not comparable; **is** the prior LocalMaxxing precedent |
| `qwen38-27-fp8-tp2-ctx-sweep-512-96k` | decode 73.0 (p512) → 57.3 (p96K); prefill 1077–1392 | FP8-W8A16 TP2, MTP8, cache off | not comparable (different model/topology) — useful only as the long-context shape |
| `qwen38-27-powercap-prefill-v1` | spec-off decode flat 24.41 → 24.25 (150→230 W); cold input 1039 → 1586 at p8192/g1 (**+53 %**) | GPTQ-INT4 route, **capability** record (no ranking metric) | establishes our cap premise: **decode is cap-insensitive, prefill is cap-clipped** |
| `qwen38-27-gguf-q4km-v1` (llama.cpp) | engine tg128 20.01 (p512) / 20.06 (p8192) | Q4_K_M dense GGUF, engine-native tg | different engine + metric class — never merged with vLLM rows |

**Metric-name discipline applied throughout:** `client_post_first_output_tps` and
`client_e2e_output_tps` and `engine_predicted_output_tps` and
`engine_native_decode_tps` and `cold_input_rate_tps` are five distinct fields and
never share a column. G1 is `client_post_first_output_tps` only.

## 3. What "improved recipes" honestly means here

The defensible public statement is a **matched A/B ratio on named cells**, not a
leaderboard position:

> On one Intel Arc Pro B70 (GPU 1, 230 W configured cap), vLLM XPU with XPU graph
> capture on, Qwen3.8-27B GPTQ-INT4-G128 + BF16 MTP4, C1, recommended non-thinking
> sampling, 128/128 output tokens — quantizing only the drafter's weights to
> W4A16 g128 raised **client post-first output rate** from 55.61 to 80.32 tok/s at
> calibrated p8192/g128 (median, n=5) = **1.44×**, and from 52.83 to 77.14 at
> p512/g128 (median, n=3) = **1.46×**, with the target untouched and the greedy
> emitted sequence byte-identical.

Comparison classification: `matched_except_quant` (arms differ only in the two
draft-side env gates; same image, same overlays, same sampling, same cache state,
same cap). That classification is what licenses the simple multiplier.

Two rules that must travel with it: the metric is a **client post-first rate**,
not engine-native throughput; and the concurrency multiplier is a **separate
class** and is never folded into the single-stream figure.

## 4. Conflict requiring an owner decision (do not publish either figure until settled)

`qwen38-27-draft-int4-p512-g128-c1` (public, `official-lab`) reports **106.7**
client post-first tok/s at p512/g128 C1, n=5 median, **cache enabled**. The
2026-09-21/22 calibrated work measures the same lever at **77.14** (p512, n=3) and
**80.32** (p8192, n=5) with **cache disabled** and recommended non-thinking
sampling.

Reconciliation facts (each verifiable on disk):

1. **Cache state differs** — `enabled` vs `--no-enable-prefix-caching` + zero-hit
   delta. The claims standard forbids mixing cache states in one comparison.
2. **Sampling differs** — the campaign record notes 106.7 came from the greedy
   diagnostic run (Run 47) while 102.61 is the recommended-sampling run (Run 48).
   Per the 2026-08-30 sampling policy a **greedy cell is a diagnostic maximum and
   may not be the published representative number**. If 106.7 is greedy it is
   mis-placed as a ranking metric regardless of cache state.
3. **Prompt differs** — "calibrated Pi prompt" (real workload) vs the campaign's
   calibrated synthetic p512/p8192 prompts. Different prompt-content class.
4. **Engine generation differs in label** — 106.7 records `0.27.2rc1.dev77+gac7509e2b`;
   the champion runs are the `f01e24f6…` digest ("v0.27.2rc1.dev77-class").

**Decision taken (2026-09-22, executing agent, per "decide rather than defer"):**
the record is **annotated as a distinct cell, not deleted**, and the two numbers
are never placed in one ratio. It is *not* re-labeled as superseded, because no
like-for-like replacement exists yet (the re-measure in flight is
cache-disabled). If the re-measure is later run at cache-enabled on the Pi prompt,
the pair becomes a valid matched comparison and the record can be superseded with
its replacement. Until then the public wording must say which cell it is.

Separately, the known-stale `qwen38-fp8-tp2-k8-p1024-g128-c1` = **60.13** remains
declared-superseded-but-live. **Decision taken (2026-09-22):** it stays live but
must carry its supersession banner in the summary text, since no protocol-grade
like-for-like p1024/g128 replacement exists — a re-measure is required and that
re-measure is now on the board as a named task, not an open question.

## 5. Gains that died (so the recipe story is honest)

Each retired a program on one measurement: the K216 device-ID repair as a speed
lever (**+3.54 %** vs a ≥10 % gate), Nemotron Mamba2/SSD fusion (fusable family
**8 %** of device time vs **73 %** MoE+GEMM), small-M M=1 specialization (op at
its **54.78 µs** weight-read floor), the 0.25 ms/draft-step MTP target (**22.3×**
over byte budget), ubatch >1024 (2048 +0.2 %, 4096 OOM), the AWQ-format route
(ecosystem ships compressed-tensors), EXL3 (no runtime), W4A8 (quality
divergence), and TP2 on this board (x8/x8 CPU-attached, no P2P). Full detail with
killing numbers: `OUTCOMES.md` L1–L8 and `LEVERS.md`.

**Added 2026-09-23 (the two biggest planned wins of the night, both retired with
their killing measurements):**

- **A from-scratch grouped-IQ3 prefill kernel is slower than the path it replaces.**
  Correctness was never the question — the reconstruction is bit-exact (33,554,432
  values, 0 mismatches), zero spills, 6/6 eval rows NMSE-clean — but it measured
  **39.36 ms gate/up and 39.28 ms down at n=512 against stock 28.37/30.13 = 1.39×
  and 1.30× SLOWER**, versus a ≤14.4 ms (≥2×) gate. Killing the 7.25 GB FP16 round
  trip is not sufficient: the as-built pays ~34× its 1.141 ms weight floor in
  exposed latency and ALU (non-persistent work-groups, no RAW staging, serial
  reconstruction). The lever is not dead — the redemption path is the IQ3_XXS
  extension on upstream llama.cpp **#29245**, whose tile scheduling is exactly what
  this draft lacked. `OUTCOMES.md` L9.
- **Quantizing the verify-pass (5th) head moves emissions.** The bytes premise held
  (−9.0 %) but the quality gate failed outright: **top-1 agreement 72.07 %** against
  a ≥99 % promotion line, median first-divergence **2–3 tokens** against a >2000
  gate. The head *is* the target distribution being verified. Retired before it cost
  any GPU time. `OUTCOMES.md` L10.

One cheap positive from the same pass: the depth axis is arithmetically closed
(k=5/6 buys ≤1.7 % and is negative across the measured acceptance band) and the
**#54785 k=3 safety clamp is nearly free here** (−0.8 % to +1.4 % at α 0.49–0.58),
so it is adopted as the published default — removing a silent-corruption hazard for
≈0 % throughput. The steep lever is drafter acceptance: **+9.3 % decode per +0.05 α**.

## 6. Evidence

- G1/G2/G3/G8: `results/2026/09/b70-improvement-probes-20260920/p4-champion-ab-*`,
  `p4-draftint4-ab-*`, `p4-draftint4-ab-compile-*`
- G4: `…/p3-prefill-ub-20260921T085017Z/`, `…T152807Z/`
- G5: `…/dpas_rate_probe_20260920/` + `FINDINGS.md`
- G6: `…/p6-ov-oor-classification-20260921T102903Z-244970/`, `…/p6-ov-ladder-*`
- G7: `worklogs/P8-awq-loader-probe-20260921.md`, `P8-20260921.md`
- Registered cells: `intel-arc-pro-b70-inference-cookbook/data/benchmarks.v1.json`
- Patch provenance: `research/2026/qwen38-draft-int4-patch-provenance-20260922.md`

> **MEASURED CORRECTION 2026-09-23:** the #54785 k=3 clamp recommendation in this file is **REVERSED** — the clamp costs +5.45 % throughput (k4 75.04 vs k3 71.16 tok/s) and the corruption class is not live (spec-canary PASS at k=3 and k=4). Keep MTP4. Details: `worklogs/SPEC-ECONOMICS-20260923.md` §CORRECTION and `clampk3-20260923T022938Z/`.
