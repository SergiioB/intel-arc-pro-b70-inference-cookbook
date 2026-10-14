---
title: "Gains achieved vs current inference data — recomputed comparison for LocalMaxxing"
type: plan
date: 2026-09-23
status: analysis
---

# GAINS vs CURRENT — recomputed from raw evidence (2026-09-23)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Read-only inventory task: no benchmarks run, no
models loaded, no GPUs touched, no registry/dossier files edited. Every delta below was
recomputed from raw `comparison.json` / `client.log` JSONL under
`results/2026/09/b70-improvement-probes-20260920/`. Numbers travel results → campaign →
cookbook catalog → XeCores renderer; nothing here is a new claim.

Metric-class rule used throughout: **C1 client post-first** (`out_tokens/(total−ttft)`,
client-side) is never an engine rate; **Cn aggregate** (total server-reported out tokens /
monotonic wall) is never single-stream; **engine_pp / engine_tg** (llama.cpp `timings`) are
engine-native and never vLLM client numbers.

## 1. Recomputed gains table

### G1 — Draft-side INT4 pair, calibrated p8192 C1 (the headline A/B)

| field | value |
|---|---|
| Metric class | C1 client post-first tok/s (NOT engine decode) |
| Requested / actual cell | requested p8192/g128 → **actual 8189 baseline / 8190 draft endpoint prompt tokens** (calibrated; first row of each client.log is the discarded warmup) |
| n | 5 measured + 1 same-shape warmup discarded per arm |
| Baseline median | **55.61** (samples 46.69, 47.75, 55.61, 55.62, 62.31; min 46.69, max 62.31) |
| Draft-pair median | **80.32** (samples 78.08, 80.05, 80.32, 90.61, 93.30; min 78.08, max 93.30) |
| Delta | **+44.4 %** median; distributions disjoint (baseline max 62.31 < draft min 78.08) |
| TTFT | flat at this cell (baseline med 4732.2 ms vs draft 4708.5 ms) |
| Cache | prefix caching off and proven off (`enable_prefix_caching: False` + resident-prefix zero-hit test) |
| Sampling | recommended non-thinking (temp 0.7, top_p 0.8, top_k 20, presence_penalty 1.5, thinking off) = Run 48 regime |
| Stack | champion digest `f01e24f6…` + overlay + `VLLM_XPU_ENABLE_XPU_GRAPH=1` + `B70_MTP_BF16_DRAFT=1`; single variable between arms `B70_DRAFT_LMHEAD_INT4=1,B70_DRAFT_MTP_INT4=1`; MTP4; fresh container per arm; ≥31,000 MiB gate per arm |
| Power | configured cap **230 W**; measured draw **not measured**; no energy/temp telemetry; no cooldown gate (teardown + `sleep 3`); baseline always first (order uncounterbalanced) |
| GPU | **GPU 1** (`0000:0f:00.0`) — recorded champion cells were GPU 0, so arm-vs-arm only |
| Evidence | `results/2026/09/b70-improvement-probes-20260920/p4-champion-ab-p8192-n5-B70_DRAFT_LMHEAD_INT4_1_B70_DRAFT_MTP_INT4_1-20260921T114444Z/` |
| Audit caveats | cell mislabeled in headlines (actual 8189, not 8192); metric must be called "client post-first"; patches anchored to `2c427ef`/vLLM 0.26.1.dev457 but run on `f01e24f6`/0.27.2rc1.dev77; patch headers claim losslessness only for greedy C1/TP1 but runs use stochastic sampling; acceptance scalar warmup-diluted (measured-window: baseline 0.5676 → draft 0.7027 here, but the p512 cell moves the opposite way — acceptance at ±2 pp is noise at these n) |

### G2 — Draft-side INT4 pair, first p8192 n=5 (actual 6993 tokens)

| field | value |
|---|---|
| Metric class | C1 client post-first tok/s |
| Requested / actual cell | requested p8192/g128 → **actual 6993 endpoint tokens both arms** (word-count client, uncalibrated — the audit's defect #1) |
| Baseline median | **52.99** (48.18, 49.22, 52.99, 54.45, 55.73) |
| Draft-pair median | **72.68** (67.49, 69.33, 72.68, 72.85, 78.54) |
| Delta | **+37.2 %**; disjoint (55.73 < 67.49) |
| TTFT | flat (~4.00–4.06 s both arms) |
| Cache / sampling / stack / power / GPU | same as G1 (cache off proven, recommended sampling, champion stack, 230 W configured / not measured, GPU 1) |
| Evidence | `…/p4-champion-ab-p8192-n5-B70_DRAFT_LMHEAD_INT4_1_B70_DRAFT_MTP_INT4_1-20260921T093453Z/` |

### G3 — Draft-side INT4 pair, p512 screen n=3 (actual 447 tokens) — screen-class only

| field | value |
|---|---|
| Metric class | C1 client post-first tok/s |
| Requested / actual cell | requested p512/g128 → **actual 447 endpoint tokens both arms** |
| Baseline median | **52.83** (51.37, 52.83, 60.11); TTFT 308.4–315.4 ms |
| Draft-pair median | **77.14** (65.63, 77.14, 77.26); TTFT 281.0–289.9 ms |
| Delta | **+46.0 %**, but **n=3 cannot back a representative number** (protocol §1.9 lane 2) |
| Extra defects | "TTFT unchanged" is **false** here (all three draft TTFTs below all three baseline TTFTs, ≈ −10 %); measured-window acceptance **down −0.048**, breaching the patch header's own 0.03 gate |
| Evidence | `…/p4-champion-ab-p512-n3-B70_DRAFT_LMHEAD_INT4_1_B70_DRAFT_MTP_INT4_1-20260921T094524Z/` |

### G4 — Repeatability control: both arms patched (no unpatched arm)

Draft_A median **80.61** (74.27–91.15) vs Draft_B median **78.27** (64.14–83.07), n=5+5,
same calibrated cell as G1. Reproduces the ~80 tok/s level with both arms patched ⇒ the G1
delta is not acceptance luck. Evidence: `…/p4-champion-ab-p8192-n5-…T121436Z/`
(`v5_draftpair_A` / `v5_draftpair_B`; the `v5_baseline`/`v5_draftpair` slots are empty n=0).

### G5 — Graph capture: the enabler (local image, NOT the champion stack)

| arm | median post-first | delta vs same-cell baseline |
|---|---|---|
| Eager baseline (graphs off) | 25.40 | — |
| Eager + pair | 25.79 | **+1.5 % (inert)** |
| Compiled baseline (graphs on) | 42.76 | **capture itself = 1.684×** (42.76/25.40) |
| Compiled + LM-head-only | 42.55 | −0.5 % (inert alone) |
| Compiled + MTP-only | 43.04 | +0.7 % (inert alone) |
| Compiled + pair at p8192 (actual ~6993-class) | 42.96 vs 38.71 | +11.0 % (n=3) |

Reading: each patch alone is neutral; the pair pays only with capture on. Evidence:
`…/p4-draftint4-ab-20260921T065451Z/`, `…-compile-20260921T070050Z/`,
`…-compile-B70_DRAFT_MTP_INT4_1-20260921T070844Z/`,
`…-p8192-compile-B70_DRAFT_LMHEAD_INT4_1_B70_DRAFT_MTP_INT4_1-20260921T073451Z/`.

### G6 — Concurrency knee (aggregate, NEVER single-stream)

Method (`scripts/tmp/p4-concurrent-client.py`): B requests in flight, R total, greedy,
ignore_eos, unique nonce per request, one discarded warmup; aggregate =
Σ server-reported out tokens / monotonic wall (`client.log` summary rows — verified by reread,
not by summing per-request rates).

| B (in flight) | baseline-config aggregate (wall) | patched-config aggregate (wall) | patched per-req post-first med |
|---|---|---|---|
| 1 (8 req) | 48.45 (21.137 s) | 61.86 (16.553 s) | 72.36 |
| 2 (8 req) | 69.73 (+44 %) | 85.40 (+38 %) | 64.97 |
| 4 (8 req) | **115.70 (+139 %)** | **140.30 (+127 %)** | 48.54 |
| 8 (8 req) | 119.82 (+147 %) | **149.79 (+142 %)** | 38.92 |
| 16 (32 req, patched only) | — | **162.74** (25.169 s) | 29.16 |
| 32 (32 req, patched only) | — | **172.48** (23.748 s) | 29.75 |

2.39× by B=4 (baseline) / 2.27× (patched); 2.79× by B=32 over patched B=1. Price:
per-request rate 72 → ~29 tok/s; TTFT 0.32 s → ~2 s at B=8, up to ~21 s at B=32.
Evidence: `…T143215Z/v5_baseline/` (baseline) + `…T143819Z/v5_draftpair/` (patched B=1–8)
+ `…T151528Z/v5_draftpair/` (B=16/32). Note: LEVERS S2 cites the deep-knee dir as T151358Z;
the B16/B32 jsonl on disk live under **T151528Z** — use T151528Z.

### G7 — K216 ubatch 512 → 1024 (llama.cpp SYCL engine rate, NOT vLLM)

| field | value |
|---|---|
| Metric class | **engine** prompt_per_second (`engine_pp`), prefill |
| Requested / actual cell | manifest says p8192/g1 → **actual ~6995–7000 prompt tokens** (p7000-class), g1 |
| ub512 median | **101.20** (98.006, 101.20, 101.223) |
| ub1024 median | **130.78** (129.377, 130.779, 131.364) |
| Delta | **1.292×** (criterion ≥1.15× fixed before measuring) |
| ub2048 | 131.10 (+0.2 % — saturated); ub4096 **does not load** (`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`) |
| Placement | `-sm layer -ncmoe 15 -ts 65,35`, dual B70, caps gpu0=180/gpu1=230 (not the contracted 150 W rest) |
| Evidence | `…/p3-prefill-ub-20260921T085017Z/` (win), `…/p3-prefill-ub-20260921T152807Z/` (saturation/OOM) |

### G8 — K216 device-ID repair end-to-end (engine rate; gate FAILED, withdrawn)

ABBA clean-r1/patched-r1/patched-r2/clean-r2, fresh load + teardown + VRAM re-gate +
cooldown ≤55 °C per arm (no retained temperature values). Metric: engine
`predicted_per_second` (`engine_tg`); actual ~447–453 tokens; greedy + `cache_prompt=false`.

Clean median **9.25** (9.093–9.475) → patched median **9.577** (9.446–9.632) = **+3.54 %**,
below the ≥10 % gate ⇒ **P1 WITHDRAWN**; repair kept only as SYCL-graph-capture prerequisite.
Attribution caveat: patched build = whole uncommitted tree (283 insertions, 9 modified + 3
untracked files), not the +38-line mmvq.cpp block alone — publishable today only as
"patched tree vs clean export". Evidence: `…/p1-k216-e2e-ab-20260921T081155Z/`.

### Non-gains confirmed (do not publish as wins)

OV ≥196K failure = second-generation host-memory growth (`CL_OUT_OF_RESOURCES` at 44 MiB
free), 131072 + 163840 full protocols pass — upstream-report shape, not a fleet number.
Compressed-tensors W4A16 = loads/serves (5,075 MiB free, 5/6 bounded suite, eager, local
image) — bounded validation, not a speed claim. All five "AWQ" artifacts declare
`quant_method=compressed-tensors` — the AWQ-format branch is closed, not a route.

## 2. Before/after against registered anchors (WITHOUT editing them)

### Anchor A — `registry/model-status.json`: Qwen3.8-27B `best_tok_s` 102.61, cell p512/g128, Run 48 n=5 recommended non-thinking

- BEFORE: **102.61 tok/s client post-first**, later prefix-on champion stack, GPU 0.
- AFTER (draft-pair on the same anchor): **not measured — no value may be proposed.**
  The G3 screen (77.14, actual 447 tokens, cache-off, GPU 1, n=3) is not the anchor's cell.
- Classification: **`not_comparable`** as a before/after pair. Fails the comparison gate on
  (3) coordinates — actual 447 vs 512 tokens; (5) cache state — prefix-on vs proven-off;
  host/GPU — GPU 0 vs GPU 1 (manifests explicitly disclaim cross-run compare); and repetition
  policy (n=5 vs screen n=3). Metric class matches (both C1 client post-first) and sampling
  matches (recommended non-thinking) — those two gates pass, the other four fail.
- What WOULD update it: fresh n=5 draft-pair run at calibrated 512 tokens, prefix-on
  champion stack, GPU 0, 150 W rest state (missing-evidence item M1).

### Anchor B — `models/qwen/qwen3.8-27b/serving.md` Run 40 matrix (83.7 MTP4 p512/g128 custom post-first, different artifact; 56.3 at exact 131072)

- BEFORE: historical orientation only (different artifact, custom statistic).
- AFTER: not proposed. G1/G2 baselines (55.61 at actual 8189; 52.99 at actual 6993) are
  internal A/B controls on the current nogidx artifact, cache-off, GPU 1 — they must never be
  quoted as champion-cell records.
- Classification: **`not_comparable`** (artifact + cache + GPU + statistic all differ).

### Anchor C — serving.md draft-side INT4 table (55.61 → 80.32 etc.)

- This table already reports the improvement arms; its n=5 p8192 rows (G1/G2) are
  internally **`matched_exact`** arm-vs-arm (same image, overlay, graphs, cache, sampling,
  cap, GPU, matched actual tokens). The p512 n=3 row stays screen-class.
- The dossier's "8,189 of 8,192 tokens" line matches the T114444Z client.log (8189/8190);
  the audit's "6993" refers to the earlier T093453Z run — both labels are now pinned to runs.

### Anchor D — concurrency vs the 102.61 champion (explicit trap)

- 140.30 / 149.79 / 172.48 tok/s are **Cn aggregates**; 102.61 is a **C1 single-stream**
  cell. Classification: **`not_comparable`** by definition (gate rule 2). The only valid
  concurrency claim is the within-campaign multiplier (2.27–2.39× by B=4, 2.79× by B=32),
  always with per-request latency attached.

### Anchor E — K216 numbers (G7/G8) vs any vLLM/LocalMaxxing row

- Classification: **`not_comparable`** across engines. G7 is a llama.cpp SYCL engine
  prefill rate; G8 is a llama.cpp SYCL engine decode rate; neither may enter a `vllm`
  payload or a vLLM before/after. The K216 decode wall stays at the AGENTS §16 provisional
  9.25 clean / 9.577 patched engine tok/s.

## 3. Proposed LocalMaxxing payloads — DRAFT, awaiting fresh n=5 verification

`engineName` lowercase enum value is `vllm`. `hfId` `Qwen/Qwen3.8-27B` verified resolvable
via the HF API (200, `modelId: Qwen/Qwen3.8-27B`); the local GPTQ artifact
(`…-GPTQ-Int4-sym-G128-MTP-BF16-nogidx`) travels in `quantization` + notes, never as `hfId`.

### Payload 1 — C1 decode (the only near-submittable number)

```json
{
  "hfId": "Qwen/Qwen3.8-27B",
  "hardware": {
    "hwClass": "DISCRETE_GPU",
    "gpuName": "Intel Arc Pro B70",
    "gpuCount": 1,
    "vramGb": 32.0,
    "os": "Linux"
  },
  "engineName": "vllm",
  "engineVersion": "0.27.2rc1.dev77+gac7509e2b, kernels 0.1.12.3, container digest sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f plus local stack overlay",
  "quantization": "GPTQ-INT4-G128-sym-local target, BF16 MTP4 draft head; DRAFT-SIDE experimental: draft LM head + MTP body W4A16 g128 (local-only patches, not a public recipe)",
  "backend": "xpu",
  "tokSOut": 80.32,
  "contextLength": 8192,
  "batchSize": 1,
  "notes": "DRAFT - DO NOT SUBMIT pending fresh n=5 verification. C1 single-stream client post-first median 80.32 tok/s, n=5 (max observed 93.30), requested p8192/g128, ACTUAL 8190 endpoint prompt tokens, 128/128 server-reported out tokens every sample, one same-shape warmup discarded. Prefix caching OFF (startup flag plus zero-hit resident-prefix test). Spec: MTP4. Sampling: temp 0.7, top_p 0.8, top_k 20, presence penalty 1.5, thinking off. Graph capture ON (VLLM_XPU_ENABLE_XPU_GRAPH=1) - the draft-side pair is inert without it (<=1.5%). Single-GPU arm-vs-arm A/B; recorded champion cells ran on the other GPU so no cross-run compare is claimed. Power: 230 W configured cap; measured draw not measured; no temp telemetry. Local-only INT4 draft patches, anchor-generation provenance pending. Approved self-reported submission, not independent reproduction.",
  "engineFlags": {
    "commandSnippet": "vllm serve (champion-container digest f01e24f6 + overlay, VLLM_XPU_ENABLE_XPU_GRAPH=1, B70_MTP_BF16_DRAFT=1, B70_DRAFT_LMHEAD_INT4=1, B70_DRAFT_MTP_INT4=1), MTP4, prefix caching off, single GPU",
    "kvCacheDtype": "fp8",
    "flashAttn": true,
    "numParallel": 1
  }
}
```

Pre-submit confirmations still required: server `max_model_len` for `contextLength`
(8192 assumed from the champion scheduler budget — verify in server.log); KV dtype string
(fp8 assumed from the Run 40 matrix — verify); engine/kernels versions from server.log:28.

### Payload 2 — prefill: OMIT unless the platform needs the field

No isolated prefill exists for the vLLM route. The only honest number is the TTFT-derived
**cold input rate** at the G1 cell: 8189 tokens / 4.7322 s = **~1730 tok/s** (baseline arm;
draft arm ~1739). If the field must be filled, use 1730 with this notes append:
"tokSPrefill is the cold input rate from TTFT (endpoint input tokens / client TTFT median),
not isolated prefill." Default recommendation: **omit `tokSPrefill`**. The K216 130.78
engine_pp is a different engine/model path and must not fill a vLLM payload's prefill field.

### Must never be submitted

- 77.14 (n=3 screen), 93.30 or any max without the median beside it, any concurrency
  aggregate (115.70–172.48) in the C1 `tokSOut` field, G7/G8 llama.cpp engine rates in a
  `vllm` payload, 102.61 */* 80.32 multipliers (not_comparable anchors).

## 4. Missing-evidence checklist (claims-audit fix list + P1)

- [ ] **M1.** Tokenizer-calibrated exact 512-token prompts + **fresh n=5** draft-pair A/B at
  p512 (p512 has never had one); recalibrate p8192 to exact 8192 or permanently relabel to
  actual counts (8189/8190 calibrated run; 6993 early run).
- [ ] **M2.** Measured-window acceptance brackets: spec-decode counter snapshots
  before/after the measured phase (after-only cumulative reads are warmup-diluted and
  direction-unstable: +0.003 at p8192 vs −0.048 at p512).
- [ ] **M3.** Champion-generation patch validation: anchor check of both INT4 draft patches
  on vLLM 0.27.2rc1.dev77 + greedy C1/TP1 token-parity spot check patched vs baseline
  (the patch headers' own validity condition; runs used stochastic sampling).
- [ ] **M4.** Energy + named-sensor temperature telemetry over the measured windows, real
  cooldown gate between arms, counterbalanced arm order; separate configured 230 W cap from
  measured draw; restore the contracted 150 W rest state around the campaign.
- [ ] **M5.** P1 single-variable rebuild (clean tree + only the mmvq.cpp device-ID cases),
  same L5 cell, n≥5, retained temperature log, greedy clean-vs-patched token-parity dump —
  else keep the tree-vs-tree wording permanently.
- [ ] **M6.** Confirm pre-submit payload fields from server.log (max_model_len, KV dtype,
  engine/kernel versions) and the sanitized public evidence description (no repo paths in
  any public payload).
