---
title: "FINDINGS — Qwen3.8-27B Quant Format Showdown (Overnight 2026-09-13/14)"
type: research
date: 2026-09-13
status: final
---

# FINDINGS — Qwen3.8-27B Quant Format Showdown (Overnight 2026-09-13/14)

> **PROVISIONAL — NOT FOR PUBLIC HEADLINE**
> Evidence tier: E2 (self-reported with raw evidence). No public surface, registry, or
> catalog may repeat these numbers until the claims audit passes and the user authorizes it.

Campaign: `results/2026/09/qw38-quant-format-showdown-20260913/` · Plan: [PLAN.md](PLAN.md) · Manifest: `manifest.json`
Subject: dense hybrid-GDN **Qwen3.8-27B** (`qwen3_5`; 64 blocks + MTP layer). One B70 (GPU0), 150 W configured cap, serial arms, one engine at a time.

---

## 1. What the dropped plan got right and wrong (b70-architect verdict)

The dropped plan (`~/Downloads/b70-quant-challengers-autoround.md`) proposed a
custom `.b70pack` sidecar ABI with mixed Q4_G128+Q6/Q8, MXFP4, and NVFP4 arms plus custom kernels.

- **Not executable here**: `scripts/b70pack_build.py`, `build-sycl-0717`, and
  `/models/Qwen3.8-Flash-F16.gguf` do not exist; GLM5 Flash has no artifacts on this host; the
  BF16 27B source was a broken symlink farm.
- **Hardware corrections** (architect, 2026-09-13): MXFP4 scales are 8-bit E8M0 (not 2-bit);
  NVFP4 is E2M1 values + E4M3 block scales (not "E4M3 weights"); G128 is not a DPAS panel
  format and a W4A16 kernel path is not native W4A4 DPAS; M≥8 is not a universal DPAS
  crossover; two B70s are two independent 608 GB/s cards (no P2P) — "2x B70 as one engine"
  TP2 remains a separately gated route (ADR-0008).
- **Scale-traffic math**: vs INT4-G128-sym (4.125 bpw with FP16 scales), MXFP4-G32 ≈ +3.0 %
  bytes, NVFP4-G16 ≈ +9.1 % bytes.
- The **feasible version** of its mixed-precision idea (§1.1) is a GGUF `--tensor-type`
  mixture, which tonight's G2 arm tests. The exotic formats (MXFP4/NVFP4) were excluded —
  no loader/kernel route exists on this stack.

## 2. Method (validation-first, per the operator's rule)

No long run or mode commitment was made without a cheap probe first:

1. **Eager anomaly probe (no reload)**: TPOT flat ~96.5–97.2 ms across p128→p8192 (64×
   context range) → per-step overhead, not content/length; server log confirmed
   `--enforce-eager` disabled torch.compile entirely.
2. **Compile-only A/B screen** (one relaunch, one cell, n=3): 23.63 tok/s vs 10.35 eager
   → threshold ≥12.4 passed → commit. (Screen values unretained; the retained final
   pair is 24.41 vs 10.35 = **2.36×** — §3.)
3. **Long-prompt observation** (p8192/g128, n=1): 23.86 tok/s, no pathology → commit full matrix.
4. **B4 MTP4 gate**: n=3 screen ≥15 % over B0 required; measured +59 % → commit full matrix.
5. **Perplexity harness validation**: 4-chunk probe on a known artifact (old Q4_K_M) before
   any arm runs; tool verified (note: `llama-perplexity` in `build-sycl-latest-qw38` segfaults
   at startup; `build-sycl-0804` — same checkout commit — works and is used for all PPL).
6. **Conversion ladder**: serial quantize with MemAvailable guard (abort < 8 GiB), hash
   verification of the source before conversion, dry-run-free but size-logged.

## 3. vLLM serving-route results (fresh n=5, 2026-09-13T20:1x–20:4xZ)

**Canonical-set declaration (claims-audit correction, 2026-09-14):** the campaign tree
contains two B0 spec-off n=5 sets. **`vllm-b0/b0.jsonl` (20:23:29Z, clean host) is
CANONICAL**: 24.41 / 23.89 tok/s. `vllm-b0/b0-final.jsonl` (21:31:29Z, 22.00 / 21.97)
is **SUPERSEDED** — it was measured while the serial CPU conversion chain
(Q8_0/G2/Q5_K_M quantizations, 20:53–21:32Z) ran on the same host, a one-engine
discipline slip whose host-CPU contention explains the ~10 % depression. Both raw sets
retained; all ratios below use the canonical set. A third clean-host confirmation
matrix (`vllm-b0-confirm/`) was run post-audit 2026-09-14T00:1xZ.

Config: pinned image `vllm/vllm-openai-xpu@sha256:f01e24f6…`, GPTQ INT4 sym G128 nogidx
checkpoint (config-verified), TP1/PP1, GPU0 only, BF16 KV, prefix cache **off**
(`enable_prefix_caching: False` in both server configs; per-cell counter values
retained in the JSONL `counters` records — hits delta 0.0 on every cell), compile-only
(torch.compile on, cudagraph NONE), max-model-len 9216, max-num-seqs 8, batched tokens
9216, 150 W. Recommended non-thinking sampling (temp 0.7, top_p 0.8, top_k 20, presence 1.5),
thinking off, `ignore_eos`, unique entropy-first prompts from wikitext corpus (seed recorded),
endpoint token counts verified (rendered + 12 chat-template tokens). All cells: 1 warmup
discarded + 5 measured, all `finish_reason=length`.

| Arm | p2048/g1 cold input | p8192/g1 cold input | p512/g128 post-first | p8192/g128 post-first |
|---|---:|---:|---:|---:|
| **B0 spec-off** (canonical) | 1048 tok/s | 1047 tok/s | **24.41 tok/s** (TPOT 41.0 ms) | **23.89 tok/s** (TPOT 42.0 ms) |
| **B4 MTP4** | 1018 tok/s | 1028 tok/s | **45.72 tok/s** (74.6 ms/step) | **37.11 tok/s** (75.2 ms/step) |

- B4 vs B0 (C1, `matched_except_speculation` within one stack/config; denominators =
  canonical `b0.jsonl`): **+87 %** at p512/g128, **+55 %** at p8192/g128.
- MTP4 acceptance: server-log windows (12 windows, 20:28–20:31Z) show mean accepted
  length **≈3.10** (range 2.51–3.93); per-position acceptance aggregates to
  ≈77 / 59 / 45 / 33 % (draft-weighted from the log windows). The exact counter
  aggregation was not retained — a before/after `/metrics` spec-counter diff is the
  named follow-up; treat these as log-derived diagnostics, not counter-verified values.
- **Metric-semantics note (auditor-relevant)**: under speculative decoding tokens arrive in
  bursts — the B4 requests returned 128 endpoint tokens in 33–49 SSE content deltas. All B4
  decode rates above use **endpoint `completion_tokens` over the client decode window**
  (`(completion−1)/(e2el−ttft)`); the client delta-count field (`client_post_first_tps`)
  undercounts to 13.3–13.4 on the same raw records and must not be used for spec rows —
  and note 45.7 tok/s is endpoint-completion throughput, not interactive SSE streaming
  rate (13.3–13.4 chunks/s). B0 (one token per delta) shows both fields agreeing on its
  canonical set. Verified from `vllm-b4/b4.jsonl` / `vllm-b0/b0.jsonl` raw records.
- Power (energy1_input, card domain): the campaign-window averages (B0 75.7 W, B4 76.2 W,
  eager 96.3 W) include asymmetric idle gaps and are **not** a valid mode-vs-mode power
  comparison (audit finding). Matched active windows from `monitor/all.csv`: eager ≈103.5 W,
  compile B0 ≈103.5 W, B4 ≈97.8 W wall — i.e. eager is ~2.4× slower at similar wall
  power ⇒ ~2.4× worse energy per generated token. Exact per-cell energy bracketing is
  part of the confirmation follow-up.
- **Eager cost finding**: the same B0 config with `--enforce-eager` delivered 10.3–10.4 tok/s
  (TPOT ~97 ms, context-flat). Against the canonical compile set: **2.36×** (24.41/10.35).
  (The intermediate "23.63 n=3 / 2.28×" screen values cited in §2.2 have no retained raw
  samples and are superseded by this retained pair.) `--enforce-eager` is an anti-pattern
  for XPU serving on this build.
- Comparison class vs registered anchors: `directional_only` (registered Run 48 102.61 used a
  prefix-on champion stack; the 49.08 single-card anchor's manifest is partially
  PP2-era/mislabeled). Tonight's 45.72 is ~7 % below that 49.08 anchor — framing is
  "route re-confirmed on fresh n=5 evidence", not a new high-water mark. Do not publish
  cross-config ratios from this campaign.
- **Cookbook reconciliation (morning question, 2026-09-14)**: the cookbook's 102.61
  tok/s (Run 48) is **TP2 on both cards, 230 W configured, prefix-cache-on champion
  stack** — a different cell from tonight's single-card/150 W/cache-off/cold-prefix
  45.72. Per-card arithmetic: 102.61/2 ≈ 51.3 tok/s/card at 230 W warm vs 45.72
  single card at 150 W cold — comparable but ~11 % below per-card throughput at 65 %
  of the power budget; `directional_only` (collective overhead, cache state, cap, and
  statistic all differ). Tonight's +87 % is strictly the within-stack MTP4-vs-spec-off
  multiplier and beats no registered number.

**Power-cap A/B addendum (2026-09-14T04:2x–04:4xZ, user-directed 230 W rerun,
`vllm-230w/`)** — same configs, fresh servers, clean host, n=5, 230 W configured
both cards (measured window draw ≈201–202 W avg GPU0, pkg max 82 °C):

| Cell | 150 W (canonical) | 230 W | Delta |
|---|---:|---:|---:|
| B0 p512/g128 post-first | 24.41 | **24.25** | −0.7 % (cap-insensitive) |
| B0 p8192/g128 post-first | 23.89 | **24.34** | +1.9 % |
| B0 p8192/g1 cold input | 1039–1047 | **1586** | **+53 %** |
| B4 p512/g128 post-first | 45.72 | 41.07 | −10 % (see note) |
| B4 p8192/g128 post-first | 37.11 | **37.96** | +2.3 % |
| B4 p2048/g1 cold input | 1012–1018 | **1403** | **+38 %** |
| B4 p8192/g1 cold input | 1026–1028 | **1493** | **+45 %** |

Power answer for this stack (now controlled): **decode is cap-insensitive**
(launch-gap-bound, draws ~75–110 W regardless); **prefill/cold-input is heavily
clipped at 150 W** (+38–53 % at the stock 230 W cap). The B4 p512 decode delta is
NOT a power effect: the no-spec B0 control is flat at both caps and mean MTP
acceptance was ~unchanged (3.01 vs 3.10) — it sits within the documented
content-driven spec-decode variance (per-cell acceptance not isolated; the
dossier's "acceptance is content, not configuration" warning applies). No new
decode record at 230 W; the win is prefill. Comparison class: `matched_exact`
per pair except cap (fresh server per arm, i.i.d. prompt draws).

**Post-audit clean-host confirmation (2026-09-13T23:51–23:54Z, `vllm-b0-confirm/`):**
same config, third independent n=5 matrix, no concurrent host work, exact power
bracketing:

- p512/g128 endpoint post-first median **24.32** tok/s (TPOT 41.1 ms);
  p8192/g128 **24.21** (TPOT 41.5 ms) — canonical set confirmed within 0.4–0.8 %.
- Cold input p2048/g1 **1034**, p8192/g1 **1036** tok/s.
- Per-cell prefix-cache counter values retained in the JSONL `counters` records
  (hits 0.0 → 0.0 every cell; queries 0.0 → 0.0 — counters inactive with caching off,
  config line is the primary cache-off evidence).
- Measured draw over the exact request window (`power-window.json`): **146.0 W avg
  GPU0** (180 s, n=180 samples), package max 73.0 °C, 150 W configured cap. This
  replaces the idle-contaminated 75.7/76.2/96.3 W campaign-window figures for any
  power statement about this route.

## 4. llama.cpp GGUF format matrix

*(filled in as arms complete — see per-arm dirs `gguf-g0|g2|gr|g1/`)*

Conversion source: HF `Qwen/Qwen3.8-27B` BF16, all files SHA256-verified against HF LFS
metadata before conversion (see `conversions/hash-verify.txt`), converted to F16 GGUF with the
same checkout's converter (`qwen35` arch, 65 blocks; MTP `nextn` tensors preserved in GGUF but
unused by these builds). Quantizer: `llama-quantize` from `build-sycl-latest-qw38`
(commit 1692f9e50). Cells: `pp512+tg128@p512`, `pp8192+tg128@p8192` (r=5), `pp2048` (r=5),
`-b 8192 -ub 4096`, KV q8_0/q4_1, fa on, ngl 99, one GPU (SYCL0). Lane-0 stability gate
(≤15 % spread) passed before each arm's cells.

| Arm | Format | Bytes | pp512 | pp2048 | pp8192 | tg128@p512 | tg128@p8192 | PPL (72 chunks) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GR | Q8_0 (reference) | 29,047,084,352 | 664.09 | 864.56 | 853.08 | **13.62** ±0.01 | **13.54** ±0.03 | 6.6779 ±0.045 |
| G0 | Q4_K_M | 16,810,714,432 | 611.73 | 848.04 | 844.25 | **20.01** ±0.28 | **20.06** ±0.22 | **6.5474** ±0.043 |
| G2 | Q4_K_M + ssm_out/attn_qkv Q6_K, attn_gate Q8_0, token_embd q6_K (48× each over the M-mix) | 18,609,861,952 | 622.71 | 849.02 | 847.49 | **18.85** ±0.21 | **18.72** ±0.16 | 6.6528 ±0.044 |
| G1 | Q5_K_M | 19,535,701,312 | 677.06 | 871.34 | 856.76 | **16.97** ±0.23 | **16.90** ±0.15 | 6.7943 ±0.046 |

Readings (engine-native llama-bench medians, r=5, KV q8_0/q4_1, one B70):

- **Decode tracks weight bytes**: Q4_K_M 20.0 → mixed 18.9 (−5.8 % for +10.7 % bytes) →
  Q5_K_M 17.0 (−15.2 % for +16.2 % bytes) → Q8_0 13.6 tok/s (+72.8 % bytes). The
  GDN-hybrid's always-active stream makes selective up-precision a direct decode tax.
- **Prefill is byte-flat** (844–871 tok/s across 16.8–29 GB): the pp path is
  kernel/compute-bound, not weight-stream-bound, on this build — a format choice costs
  decode but essentially nothing in prefill across this bpw range.
- **Effective streaming bandwidth rises with precision**: Q4_K_M streams ≈336 GB/s
  effective (bytes×rate), Q5_K_M ≈332, Q8_0 ≈396 GB/s — the Q8_0 dequant path
  (trivial scale multiply, no K-quant super-block indirection) utilizes GDDR6 better
  per byte than K-quant formats. Cross-format kernel efficiency, not just size,
  shapes the decode curve.
- **vLLM INT4 W8A16 vs GGUF Q4_K_M** (directional cross-engine, NOT a matched pair —
  different engines/KV/kernels; client post-first vs engine-native tg): 22.0 vs 20.0
  tok/s-class decode; vLLM cold-input 1022–1039 vs llama pp8192 844–853 (different
  metric definitions). The AutoRound/GPTQ route keeps a modest decode edge with
  MTP4 multiplying it to 45.7/37.1.
- Greedy-replay smokes (final state): g0 **PASS** (deterministic, coherent JSON
  emitted; truncated at the 96-token budget before full JSON close — budget, not
  incoherence), gr **PASS** via `/v1/models` readiness gate (deterministic, same
  coherent JSON start). g1/g2 coherent smokes **not validly completed** — the first
  harness version passed an unescaped-quote payload (server 400s), and the re-run
  `cmp` compared two identical error bodies; g0/g2 "PASS" lines in
  `post-arms.log` from that version are **void**. g1/g2 forward-pass evidence =
  their r=5 cells + 72-chunk PPL runs. Tooling lesson (corrected 2026-09-14): our
  readiness gates misused `curl -s` exit status (0 for any HTTP status, incl. 503
  "Loading model") — the passing reruns gate on `-w '%{http_code}' == 200`; see
  the formal correction in §5.

*(IQ4_XS skipped: no imatrix exists and calibration would exceed the night's budget per the
architect ladder — do not run IQ quants without imatrix.)*

## 5. Quality

Perplexity on `wiki.test.raw` (sha256 `aca2f467…`, 1.29 MB wikitext-2 test), `-c 4096`,
72 chunks, one engine at a time, **unquantized KV** (weights-only axis; KV policy must
not pollute the format comparison). Build `build-sycl-0804` (`build-sycl-latest-qw38`'s
llama-perplexity segfaults at startup — recorded). KL divergence is unavailable in this
build; ΔPPL vs the fresh Q8_0 anchor was the planned proxy.

**Result: the perplexity axis came back NON-MONOTONIC in precision and was
ROOT-CAUSED to the harness/architecture interaction (closed 2026-09-14, upstream
issue [ggml-org/llama.cpp#28879](https://github.com/ggml-org/llama.cpp/issues/28879)):**

| Arm | PPL |
|---|---:|
| G0 Q4_K_M | **6.5474** ±0.043 |
| G2 mixed | 6.6528 ±0.044 |
| GR Q8_0 (near-lossless anchor) | 6.6779 ±0.045 |
| G1 Q5_K_M | 6.7943 ±0.046 |

Root-cause evidence (2026-09-14 session, `ppl-f16ref/` + `ppl-chunkgeom/`):

1. **F16 reference (dual-card layer split, same corpus/chunk geometry): 6.5545** —
   the full-precision model scores *worse* than Q4_K_M on **7 of 8 chunks** and is
   statistically identical to Q8_0 (both near-lossless relative to each other, as
   expected — which also controls for the dual-vs-single-card variable, since
   single-GPU Q8_0 matches dual-GPU F16). Full precision cannot legitimately lose
   to 4-bit weights ⇒ the metric is confounded; quant kernels are exonerated (F16
   shows the effect with no dequant kernels).
2. **Chunk-geometry sensitivity (same corpus prefix, ~32k tokens):** Q4_K_M
   6.2867 (c2048×16) → 6.4372 (c4096×8) → 7.1542 (c8192×4); Q8_0 6.3782 → 6.5527 →
   7.7157. PPL **grows** with chunk length (opposite of normal context benefit) and
   the Q8−Q4 gap widens +0.09 → +0.56 — recurrent-state accumulation dominates the
   measurement, not weight precision.
3. **Consequences:** no quality ranking of formats is claimed or needed from PPL on
   this arch/build; do not use `llama-perplexity` to rank quant formats for
   hybrid-GDN models (it ranks Q4_K_M above F16). Correctness evidence that stands:
   finite sane PPL over 72 real-text chunks per arm + deterministic-replay smokes
   (g0, gr). If format quality is ever needed here: per-position logit parity vs
   the F16 reference, not PPL.

**Correction (2026-09-14):** an earlier note blamed "llama-server `/health`
returns 200 before model load completes". Reading the source shows the middleware
correctly 503s `/health` during load; our smoke harness used `curl -s` exit status,
which is 0 for *any* HTTP status including 503, so the gate broke on the first 503.
The fix is on our side: readiness gates must check `-w '%{http_code}' == 200`
(the passing reruns did exactly that). No upstream defect.

Greedy-replay smoke status per arm: g0 PASS, gr PASS (see §4 note); g1/g2 not validly
completed (harness quoting bug — documented; their cells and PPL runs stand as
forward-pass evidence). SYCL reduction-order single-token divergence remains a known
allowed failure mode for replay checks (not observed in the passing pairs).

## 6. OpenVINO branch (parallel research, CPU-inspection only)

b70-architect inspection (no GPU slot consumed; artifacts already exercised on this host
2026-09-11..13 — see `research/2026/qwen38-ov-int4-gdn8-convert-20260912.md`,
`research/2026/ov-cascadia-qwen38-bakeoff-20260911.md`):

1. **"INT4-GDN8" is misnamed.** The local `SergiioB/Qwen3.8-27B-int4-gdn8-ov` IR is
   actually **packed INT4 asym G128 language weights + BF16/FP32 GDN** (ignored-scope
   `.*linear_attn.*`, 11.12 GB BF16 GDN constants) **+ u8 LM head**. Its
   `openvino_config.json` is stale (`dtype=int8, bits=8, group_size=-1,
   ignored_scope=null`) and contradicts the graph XML. The label must not be read as
   "8-bit GDN".
2. **Architecture correction**: Qwen3.8-27B (`qwen3_5`) language stack is **48 GDN +
   16 full-attention layers** (not the dropped plan's 30/10); GDN state 48×(128×128)
   FP32, serialized loop interfaces FP32, ~3 MiB SSM state per layer at batch 1.
3. **Byte census**: local mixed = 23.885 GiB all-BIN (u4 9.47 GB + BF16 11.12 GB +
   u8 1.27 GB + FP16/FP32 aux); official INT8 = 25.903 GiB. Genuine packed u4 exists.
4. **Runner**: `openvino_genai.VLMPipeline` on the 2026.5 nightly venv
   (`.ov-genai-venv`) is the demonstrated route (VLM export). The old `ov_ab.py`
   harness has protocol deficiencies (char≈token/4 approximation, requested-not-actual
   tokens, fixed hwmon5, fixed prompt) — its historical tok/s must not be reused as a
   headline. A matched, corrected text-only engine comparison is the documented next
   step if scheduled; **not worth tonight's serial GPU slot** (architect NO-GO).

## 7. Conclusions (final for this campaign)

1. **Serving route**: on one B70 at 150 W, the registered GPTQ-INT4 vLLM route with MTP4
   (45.7 tok/s at p512/g128, 37.1 at p8192/g128, mean accepted ≈3.1) is the C1 decode
   leader among measured serving routes; spec-off (canonical clean-host set) is
   24.41/23.89. MTP4's +87 %/+55 % confirm the registered product route on fresh
   single-card evidence with recommended sampling and entropy-first prompts
   (`matched_except_speculation` within the same stack). This is a route
   re-confirmation, ~7 % below the registered 49.08 anchor (different stack era) —
   not a new high-water mark.
2. **Eager is a trap**: compile-only is mandatory on this stack (**2.36×** decode on
   the retained pair; similar wall power ⇒ ~2.4× worse energy per token). Execution
   mode must be part of any XPU serving claim.
3. **GGUF format verdict (speed)**: decode tracks streamed bytes (Q4_K_M 20.0 → mixed
   18.9 → Q5_K_M 17.0 → Q8_0 13.6 tok/s); long-prompt prefill is byte-flat
   (pp2048/pp8192 844–871; the pp512 anchor sits at 611–677 in the short-prompt
   launch-bound regime) — format choice prices decode, not long prefill. Q4_K_M
   remains the llama.cpp workhorse for this model; Q8_0 shows the best effective
   streaming (~396 GB/s vs ~336 for Q4_K_M) but cannot overcome its byte count at
   decode.
4. **Mixed-precision rescue (the dropped plan's §1.1, as implementable tonight) is a
   negative on both measured axes**: +10.7 % bytes bought −5.8 % decode AND no quality
   gain (PPL 6.653 vs 6.547, on the confounded axis). The Q4_K_M preset's own mixture
   already allocates precision well for this GDN hybrid; blanket "protect the
   GDN/attn tensors" overrides are not a free lunch.
5. **Quality-by-PPL on hybrid-GDN is closed (root-caused)**: wikitext PPL is
   non-monotonic in precision — the F16 reference itself loses to Q4_K_M on 7/8
   chunks and PPL grows with chunk length — a harness/recurrent-state confounder,
   not a kernel bug (§5; upstream issue
   [ggml-org/llama.cpp#28879](https://github.com/ggml-org/llama.cpp/issues/28879)).
   Do not rank quant formats with `llama-perplexity` on linear-attention hybrids;
   use per-position logit parity if a quality axis is ever required.
6. **OpenVINO**: artifact-label correction + prior feasibility documented (§6); a
   matched OV-vs-vLLM/llama.cpp comparison is the next scheduled branch, not tonight's.
7. **Cross-engine caution**: vLLM-vs-GGUF rows are `directional_cross_engine` only
   (engine, KV implementation, kernels, metric family all differ). The vLLM INT4 route
   keeps a decode edge over same-class GGUF with speculation multiplying it.

## 8. Not attempted tonight (with reasons)

Custom sidecar packs/kernels (nonexistent tooling), MXFP4/NVFP4 (no loader/kernel route),
QAT/distillation (weeks), TP2/dual-engine (ADR-0008 gate; single-card only tonight), KV sweeps,
Cn concurrency (GDN mixed-load crash documented upstream), FP8 arm (checkpoint absent),
128K capacity claims, IQ4_XS (no imatrix), GLM5 Flash / Qwen3.8-Flash-Next fresh arms (no
convertible source on host; Flash-Next evidence reused directionally only).

## 8a. Publication and upstream status (2026-09-14)

- Claims audit: **passed 2026-09-14** (general-purpose auditor fallback after the
  b70-claims-auditor agent type failed on a missing provider); all audit corrections
  applied (canonical B0 set, power rewording, slip disclosure, acceptance
  re-derivation, prefill qualifier).
- **Published to the public cookbook (user-authorized)**: PR
  [SergiioB/intel-arc-pro-b70-inference-cookbook#19](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/pull/19)
  merged — 3 benchmark records (GGUF Q4_K_M / Q5_K_M / Q8_0 matrices) + 2 capability
  records (power-cap prefill finding; hybrid-GDN PPL warning), evidence doc
  commit-pinned at `22bd325`, live raw catalog verified after merge.
- **Upstream issue filed (user-authorized)**:
  [ggml-org/llama.cpp#28879](https://github.com/ggml-org/llama.cpp/issues/28879)
  (llama-perplexity not precision-monotonic on hybrid GDN). Two other candidates were
  investigated and honestly dropped: the `/health` race was our curl-exit misuse
  (§5 correction); the `-mmp` arg drift is already resolved on current master.
- X draft: `X-POST-DRAFT.md` — audit-compliant, **not posted** (needs explicit
  authorization per AGENTS.md §12).

## 9. Cleanup state

Verified 2026-09-13T23:36Z (monitor end) and re-verified 2026-09-14T00:0xZ after the
post-audit confirmation run:

- No `llama-server` / `llama-bench` / vLLM process; no inference container.
- Both cards restored to 150 W (`hwmon5`/`hwmon6` `power1_cap` = 150000000 µW each).
- VRAM gates: GPU0 `visible_avail` 32,113 MiB, GPU1 32,615 MiB (GPU0 delta is the
  desktop's own usage).
- 1 Hz energy/temp monitor stopped; full campaign CSV retained at `monitor/all.csv`.
- All campaign evidence retained under `results/2026/09/qw38-quant-format-showdown-20260913/`
  (vllm-b0/, vllm-b4/, gguf-g0|g2|g1|gr/, ppl-*/, conversions/, monitor/, manifest).
- Artifacts retained on T-FORCE: `quant-showdown-20260913/arms/` (F16 + 4 quants,
  ~139 GB total) and `src/` BF16 (~54.7 GB) — deletion is a user decision.

Protocol slip disclosed (corrected per audit, 2026-09-14): during the serial CPU
conversion chain (20:53–21:32Z) a compile-only no-spec vLLM server was relaunched at
21:20:36Z and the SUPERSEDED `b0-final.jsonl` set (22.00/21.97) was measured against
it at 21:31Z under that host-CPU contention — a one-engine-discipline violation that
also explains the ~10 % depression vs the canonical 20:23Z set. The container was
torn down at ~21:33Z, before the llama.cpp arms began (their successful full-VRAM
loads confirm no resident engine after that point). No other measurement overlapped
any conversion.