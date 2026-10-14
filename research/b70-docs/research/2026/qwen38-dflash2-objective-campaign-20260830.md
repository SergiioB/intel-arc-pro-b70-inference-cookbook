---
title: "Qwen3.8-27B Official DFlash2 Objective Campaign"
status: final
type: research
date: 2026-08-30
---

---
doc_id: DOC-RESEARCH-QWEN38-DFLASH2-OBJECTIVE-20260830
title: Qwen3.8-27B Official DFlash2 Objective Campaign — 2026-08-30
status: provisional
---

# Qwen3.8-27B Official DFlash2 Objective Campaign

**Date:** 2026-08-30 · **Host:** b70-host, dual Arc Pro B70, vLLM XPU TP2, per-worker `ZE_AFFINITY_MASK` + `SYS_PTRACE` · **Status:** PROVISIONAL — NOT FOR PUBLIC HEADLINE
**Campaign record:** [`campaigns/2026/qwen38-dflash2-objective/README.md`](../../campaigns/2026/qwen38-dflash2-objective/README.md) · **Registry:** `CAM-2026-QWEN38-DFLASH2-OBJECTIVE`
**Supersedes the planning assumption** in `research/2026/qwen38-fp8-tp2-speed-attack-consolidated-20260830.md` (retraining before adoption). This document is the decision authority for the DFlash2 route.

---

## 1. Decision

**Prioritize adoption and matched evaluation of the official `incoai/Qwen3.8-27B-DFlash2` checkpoint before writing a hidden-state capture pipeline or retraining DSpark.**

The official release changes the research state: DFlash2 is not merely a better-trained DSpark. It augments parallel DFlash with a top-16 low-rank adjacent-candidate path selector, two-tap dynamic depthwise convolutions (reducing suffix decay), and fully parallel draft computation — only the final path walk is sequential. Adoption of the official artifact precedes retraining; the SpecForge DSpark capture/training route is a diagnostic fallback, not the DFlash2 objective.

## 2. External evidence (E3 — not B70 projections)

All outside numbers below are *reported* by Inco/z-lab on their own hardware and cannot be projected onto the B70 FP8 W8A16 TP2 stack. They set expectations only.

### 2.1 Model card — `incoai/Qwen3.8-27B-DFlash2` (verified 2026-08-30)

- 2B-parameter BF16 safetensors draft model for the target `Qwen/Qwen3.8-27B`; apache-2.0; mirrored at `z-lab/Qwen3.8-27B-DFlash2`. Not a standalone LM — it runs inside a speculative decoding server.
- Speculation block size 8 → **7 draft tokens per verification step**; "lossless: greedy output matches the target model exactly, and sampling preserves its distribution."
- Tagged for both `vllm` and `sglang`; the official vLLM recipe is upstream [PR #52816](https://github.com/vllm-project/vllm/pull/52816) with `method: dflash`.

Published mean acceptance length (completed tokens per verification step, per the model card):

| Task | MTP | DSpark | DFlash2 |
|---|---:|---:|---:|
| GSM8K | 5.02 | 4.36 | 5.46 |
| MATH-500 | 4.72 | 3.92 | 5.28 |
| HumanEval | 3.91 | 3.30 | 4.39 |
| MBPP | 3.99 | 3.51 | 4.79 |
| MT-Bench | 3.74 | 3.01 | 4.10 |
| **Pooled mean** | **4.28** | **3.62** | **4.80** |

Interpretation: 4.80 vs 4.28 native MTP (+0.52) and vs 3.62 community DSpark (+1.18) at block size 8 — significant, but **not** evidence for the previous accepted-length 8–10 projection.

Published throughput (vs autoregressive, one NVIDIA H200, BF16 target, FlashAttention 3, up to 4096 generated tokens): 2.67–3.43× at concurrency 1 (184–236 tok/s), 2.27–2.85× at concurrency 8, and ≤1.45× at concurrency 32. These numbers describe H200/BF16/FA3, not B70 FP8 W8A16 TP2. Do not quote them as B70 expectations.

### 2.2 Implementation — vLLM PR #52816 (verified merged 2026-08-30)

- **Merged** into `vllm-project:main` (author: z-lab; head commit `19c9351904df4c63042671bc67a866ca48dc7d6f`).
- Adds a second DFlash architecture: `DFlash2DraftModel` (registered in `model_executor/models/registry.py`) + `DFlash2Speculator` (`v1/worker/gpu/spec_decode/dflash2/speculator.py`); routed via `method=dflash` when `DFlash2DraftModel` is in the architecture list; forces the V2 model runner (V1 silently degrades to DFlash1).
- Core files: `model_executor/models/qwen3_dflash2.py`, `v1/worker/gpu/spec_decode/dflash2/speculator.py`, `model_executor/models/registry.py`, `v1/worker/gpu/spec_decode/__init__.py`; tests `tests/v1/spec_decode/test_dflash2.py`, `test_config.py::test_dflash2_draft_forces_v2`, `test_dflash_causality.py`, `test_dflash_lookahead.py`, `test_dflash_prepare_inputs.py`.
- Selector: keep the target head's top-K per slot, score transitions `edge(p→c) = ⟨A[p] ⊙ project(h), B[c]⟩ + unary[c]`, walk the best path from the verified anchor (inverse-CDF walk at T>0); sized by `selector_rank` / `selector_top_k` (top-16 per the Inco article). Convolution: grouped dynamic depthwise inside each block, taps zeroed across block boundaries; sized by `conv_kernel_size` / `conv_group_size`.
- **Platform: CUDA-focused** — Triton kernels, FlashInfer radix top-k where available, per-component CUDA graphs. No XPU support in the merged PR; a separate `vllm-ascend` PR (#14533) exists to port DFlash2, implying non-CUDA backends need porting work.
- Follow-up fragility: a `decoder_layer_cls` indirection introduced by the PR and reverted by #52560 broke DFlash2 loading (issue #53428, fix #53435) — any adoption layer must anchor to a tested commit lineage, not the raw HEAD.

### 2.3 Inco article — `inco.ai/blog/dflash2` ("DFlash 2: Keep Drafting Parallel", 2026-08-18)

Verified 2026-08-30 against the article text (see decision log A1):

- *"Averaged across them, it gains 1.05 tokens over DFlash (21%) and 0.48 over DSpark"* — mean acceptance-length advantage, Table 3, **Qwen3.5-4B** (GSM8K, MATH-500, HumanEval, MBPP, MT-Bench; default sampling, lossless rejection).
- *"the selector and the convolution together add only 1.3% to the five-layer DFlash draft–verify cycle latency"* — the 1.3% figure. The intro's "around 1% added cycle latency" is the article's looser restatement.
- Table 4 (Qwen3.8-27B, block size 8, per-request mean acceptance length): **MTP 4.28, DSpark 3.62, DFlash2 4.80** — the basis for the campaign's +0.30 acceptance gate over matched MTP7 is deliberately below this published +0.52 headroom.
- The article also references llama.cpp PR 27342 and Ollama PR 17865 for DFlash2 — noted, but this campaign is scoped to the vLLM XPU route.

These are external, small-model/large-model-but-foreign-hardware relative comparisons, not B70 projections.

## 3. Local B70 history to reconcile (why this campaign exists)

| Date | Record | Result | Reading |
|---|---|---|---|
| 2026-08-19 | `research/2026/qwen38-dflash2-smoke-20260819.md` | NO-GO: 0% acceptance; later causal-fallback 24.7%, non-causal shims 6.8–8.2% | Two independent XPU bugs: **(A)** XPU FA ignores `use_non_causal` (draft is `is_causal: false`, SWA layers) → zero attention; **(B)** draft MLP overflows fp16 → NaN scores when target server runs `--dtype float16`. `--dtype bfloat16` context fixed the scores but OOM'd KV init at the time. The 24.7% cell used the *wrong* causal mask; it measured prefix agreement, not block diffusion. |
| 2026-08-29 | rmacy fix-era, community DSpark drafter (`rwmacy/qwen3.8-27b-dflash-drafter-fp8-b70`), off-by-one proposer fix + ABI-locked rmacy image (vllm 0.21.1.dev0+xpu, torch 2.11) | p2048 **90.5% accept, 39.79 tok/s** (k=4); p512 23.8–26.0 tok/s | The 2026-08-19 24.7% cap was also an **implementation failure** (off-by-one proposer offsets, `dflash.py num_query_per_req = k`), now fixed upstream-era. The 90.5% p2048 cell is a **narrow corrected-stack observation** (rmacy image, community drafter, random-content draw) — not a representative official-DFlash2 result, and the rmacy stack cannot take the custom-kernel work. |
| 2026-08-30 | FP8 route of record: pinned image `f01e24f6` (vLLM 0.27.2rc1.dev77, torch 2.11) + custom wheel (`libsycl.so.9`) + `patch_fp8_stack.py` + Xe2 small-M op + scratchpad pooling + native MTP | **k=8: p512 51.54, p1024 56.85 tok/s** (accept len 6.63); hybrid mode k=6 rescues p2048 (28.5–32.2) | Native MTP on the custom-kernel stack is the incumbent. DFlash2 must beat **matched MTP7** on this exact stack; the incumbent k=8 record is the operational stretch reference. |

**Key stack fact for the DFlash2 re-test:** the current FP8 route runs `--dtype bfloat16` compute, which by 2026-08-19 findings (fix1) makes DFlash2 draft scores finite — blocker (B) is structurally addressed by the BF16 stack the campaign uses. Blockers (A) non-causal attention and the remaining distribution/shape bugs (grouped-conv + RMSNorm recast, RoPE layout, query/context packing, selector walk) are still open gates for this campaign.

## 4. Hypotheses and comparisons

**Primary hypothesis:** on the same Qwen3.8-27B FP8 W8A16 TP2 target, exact prompts, sampling, block depth, cache state, and build, official DFlash2 increases accepted tokens per speculative round enough to improve end-to-end C1 throughput over native MTP, without changing target output semantics.

- **Primary matched comparison:** DFlash2 k=7 vs native MTP k=7. Same p512/g128 and p1024/g128 tuning cells; same target checkpoint (official FP8), W8A16/small-M stack, TP2 placement, tokenizer, sampling, prompt corpus, prefix-cache state, graph-off state, 230 W configured cap.
- **Incumbent comparison (operational, not depth-matched):** DFlash2 k=7 vs the current native-MTP k=8 FP8 record (51.54 / 56.85 tok/s).

## 5. Success gates

1. **Static compatibility:** official checkpoint schema and the PR-52816 implementation are understood, reproducible, and load without silently dropping selector/convolution weights.
2. **Correctness:** deterministic greedy output matches the no-spec target over ≥5 fixed prompts (one natural language, one math/reasoning, one code, one tool/JSON, one adversarial/repetition); sampling is lossless rejection; no NaN/Inf; no proposer-offset failure; no causal fallback.
3. **Acceptance:** held-out mean acceptance length improves by ≥0.30 over matched MTP7 and does not collapse by content or prompt length.
4. **Efficiency:** accepted tokens per second and end-to-end output tok/s improve; acceptance alone is insufficient.
5. **Performance:** fresh n=5 median at p512/g128 and p1024/g128 beats matched MTP7. Stretch target: ≥65 request-level tok/s at p1024/g128 (not an assumed 80–100 tok/s).
6. **Operational:** server stays healthy, cleanup completes, both cards return to empty state, cap returns to 150 W.

## 6. Stop conditions

Stop (any one) and record a completed no-go with the exact failed gate:

- selector or convolution tensors ignored or structurally incompatible;
- no exact PR-52816 implementation adoptable on the pinned XPU generation without an uncontrolled ABI mix;
- deterministic target output diverges;
- non-causal draft attention, selector scores, convolution, or MLP NaN / required the known incorrect causal fallback;
- matched DFlash2 accepted length fails to exceed MTP7 materially;
- added cycle latency consumes the acceptance gain;
- content-dependent acceptance collapse persists across the fixed held-out set;
- GDN, TP2, memory, swap, or device-reset safety gates fail.

A stopped campaign still fulfills the objective if it produces a reproducible, evidence-backed no-go naming the failed gate. It is never presented as an unfinished benchmark.

## 7. Metric formulas

- Optimization metric: **accepted output tokens ÷ complete draft–verify cycle time** (not acceptance percentage alone).
- Mean acceptance length: accepted draft tokens ÷ verification steps over the measured window; report per-sample and window-provenance.
- Matched cells: actual endpoint prompt tokens verified per request (do not relabel); one same-shape warmup discarded per shape; fresh n=5 for finalists; report median, max observed, dispersion.
- Cache guard: ≥64 chars of unique entropy at prompt start, snapshot prefix-cache hits before/after, require zero cache-hit delta.
- Power: `(energy_end_uJ − energy_start_uJ) / elapsed_s / 1e6` per card, aligned to the exact workload interval; configured cap ≠ measured draw.

## 8. Implementation sequence (operational record in the campaign README)

1. This research authority + campaign package (done 2026-08-30).
2. Pin and inspect the official artifact: resolve revision; download metadata/config first; inventory tensor names/shapes/dtypes/parameters (selector, convolution, mask token, RoPE, target layers); record hashes and size; verify ≥10 GiB free reserve before the ~2B-parameter BF16 drafter lands on `/mnt/<models-disk>`; preserve unchanged.
3. Inspect PR #52816 commit lineage vs the pinned XPU vLLM source generation; classify each change as already present / directly applicable / needs XPU adaptation / incompatible; avoid unrelated upstream changes.
4. Build the smallest reproducible adoption layer in `scripts/tmp/`: static checkpoint/loader validator; exact patch/apply script anchored to the reviewed PR-52816 lineage; launch wrapper recording image digest, source commit, patch hashes, environment, target/draft revisions, affinity, graph state, safety gates; controlled client harness for correctness + matched C1. Requirements: keep `method: dflash`; load every selector/convolution tensor, fail closed on missing keys; corrected proposer offsets; numerically correct non-causal draft path; drafter stays BF16 (no inherited FP8 quantization); XPU graphs off on TP2; preserve the W8A16/small-M verifier stack where ABI-compatible — otherwise one coherent build for correctness first, then the custom-kernel variant as a separate build cell (no binary mixing).
5. Static/CPU/XPU unit gates: schema completeness; selector shapes/top-16 behavior; dynamic-convolution boundary at the first draft position; non-causal semantics; proposer offset/query count; greedy target-prefix equality; finite scores/hidden states; XPU device/dtype placement. XPU microtests use one empty B70, no target load, CPU/BF16 reference within declared tolerance.
6. Target-load correctness gate: both-card preflight; one declared TP2 server; order no-spec control → native MTP7 control → DFlash2 k=7. Deterministic greedy parity on the five prompt classes, counter reconciliation, no engine death.
7. Matched n=3 screens: p512/g128, p1024/g128, p2048/g128 (p2048 = numerics/content-stability diagnostic, not a guaranteed W8A16 production route) comparing no-spec, MTP7, MTP8 incumbent, DFlash2 k=7. Record per sample: actual tokens, TTFT/TPOT/e2e tok/s, speculative rounds/proposed/accepted/mean acceptance length, per-position conditional acceptance when exposed, cache counters, exact prompt hash/content class, server/build/artifact identity, configured cap + measured draw + temperatures + VRAM, errors/exclusions/cleanup.
8. Real-finalist confirmation only: fresh n=5 p512/g128 + p1024/g128, one representative real workload, benchmark-claims audit on the internal wording.
9. Retraining decision (only after measurement): if acceptance is FP8-target-state-limited and Inco/z-lab releases the DFlash2 training implementation → separate exact-stack retraining campaign; else SpecForge DSpark capture/training remains the diagnostic fallback. No 500K–1M-token capture and no tens-of-GiB frees during this campaign.

## 9. Decision log

| # | Date | Decision | Basis |
|---|---|---|---|
| A1 | 2026-08-30 | Adopt-first, retrain-later; official checkpoint is the campaign object | Official `incoai/Qwen3.8-27B-DFlash2` published; PR #52816 merged; Inco article (inco.ai/blog/dflash2, 2026-08-18) verified 2026-08-30; prior planning assumption (SpecForge capture first) superseded |
| A2 | 2026-08-30 | DFlash2 k=7 is the primary spec depth; MTP7 matched control; MTP8 incumbent operational reference | Model card block size 8 = 7 draft tokens per step; matched depth requirement from hypothesis |
| A3 | 2026-08-30 | FP8 BF16 stack addresses the 2026-08-19 fp16 draft-MLP NaN blocker; non-causal attention remains the primary open gate | 2026-08-19 fix1 + current `--dtype bfloat16` route |
| A4 | 2026-08-30 | **COMPLETED NO-GO.** Official DFlash2 k=7 does not improve end-to-end C1 throughput over matched MTP7 on the FP8 W8A16 TP2 stack; the campaign concludes per the acceptance/efficiency/performance gates. Decomposition: (1) acceptance is the primary failure — held-out mean accept length ≈0.7 vs MTP7 ≈2.8 (gate: MTP7 + 0.30); (2) draft-path cycle latency adds a further ~1.9× e2e gap (python torch walk + SDPA context-cat shim); (3) the FP8-target-state mismatch hypothesis (draft trained against BF16 target hidden states) is the leading mechanism for the acceptance gap and is now the documented research target; TP2 overhead and kernel ABI separation are NOT implicated (identical verify stack, no ABI mixing). | Decision-log: full gate table and cells in §11; evidence under `results/2026/08/qwen38-dflash2-objective-20260830/` |
| A5 | 2026-08-30 | **DSpark community-drafter follow-up (tests the A4 hypothesis directly).** Evaluate [`Doopeworld/Qwen3.8-27B-DSpark-vLLM`](https://huggingface.co/Doopeworld/Qwen3.8-27B-DSpark-vLLM) (rev `ebaa0919…`, sha256 `9d26d5e6…`, 1.359B params — RadixArk weights + architecture rename to `Qwen3DSparkModel`) on the FP8 W8A16 TP2 stack: this drafter was **trained against the FP8 target's hidden states** (author-measured serving acceptance 2.45–2.79 at k=7 on a B70; per-class 2.18 prose / 4.79 math), which is precisely the target-state fidelity the official DFlash2 lacked. Loads **natively** on the pinned image (no port, no patches — the architecture rename avoids the `DSparkDraftModel`→deepseek_v4 trap). Correctness gate PASS (5/5 coherent-prefix parity, greedy). Greedy acceptance ≈2% (mean len ~0.15) — far below the author's range — in BOTH the anchor-first default layout and a config-forced **1+N fill-in layout** (`sample_from_anchor: false`; verified wiring: `Qwen3DSparkModel`, aux taps (5,17,29,41,53), zero cache hits): **the block-layout hypothesis is REFUTED**. INT4 TP2 target discriminator (same image, W4A16 BF16 activations — the author's served setup) in flight per the architect's decision table. Also established: `draft_sample_method: probabilistic` crashes this pinned build in worker init (`hf_overrides must be a dict` in `get_quant_config` — composed draft `hf_overrides` is callable; author used 0.26.1 where probabilistic worked); the metadata-guarded fix is the existing `scripts/tmp/patch_dspark_quant.py`. | Architect (b70-architect) analysis: ranked root causes — (1) DSpark serving-contract mismatch (proposal positions, Markov first-prev token, aux taps/order), (2) W8A16 capture-vs-served target-state mismatch, (3) draft context-KV reference-attention fallback, (4) TP2 aux-state distribution; and the honest FP8 TP2 ceiling ≈35–60 tok/s (optimistic 65–73) — 80+ is not defensible for a 5-layer DSpark draft on this stack; DSpark (native kernels) is the better target than shimmed DFlash2. |
| A6 | 2026-08-30 | **INT4 discriminator result: the FP8 W8A16 target-state hypothesis is REFUTED.** The same drafter (fill-in layout, greedy, k=7, pinned image) served against the INT4 GPTQ target (W4A16, BF16 activations — the author's own served setup) shows acceptance **1.54%** (62/4018) vs **1.91%** (75/3934) on the FP8 target — statistically identical. Per the architect's decision table, ~2% on the author-matched INT4 target implicates the **DSpark serving contract itself** (vLLM 0.27.2 vs the author's 0.26.1 semantics: proposal positions/Markov first-prev-token boundary/aux-state handling), not target-state numerics and not the block layout. Both targets kept 5/5 greedy coherent-prefix parity (the INT4 "natural/math" mismatches are cross-target greedy differences vs the FP8-captured reference, not speculation breaks; zero cache hits everywhere). **Named next step:** the first-cycle semantic trace of the pinned dspark path vs the SpecForge reference (`/mnt/<models-disk>/specforge-xpu/specforge/modeling/draft/dspark.py`, `first_prev_token_ids=block_output_ids[:, 0]` contract) — compare fc output, context K/V, base logits, Markov-corrected logits, and proposal ids per position for the same prompt. The pinned code reads correct at the anchor/bonus and offset level, so the trace must confirm value-level parity before any further GPU cycles. | Expected-value note: even a fully recovered drafter is bounded at ≈35–60 tok/s honest (architect ceiling); the 80+ FP8 target requires a cycle-time breakthrough (small-M draft kernels) beyond acceptance recovery. |

## 11. Result — completed no-go (2026-08-30)

**Outcome: COMPLETED NO-GO.** The official DFlash2 checkpoint was made to load and serve on the FP8 W8A16 TP2 stack (correctness gate PASS — a first for this host), but its acceptance and throughput lose to matched native MTP7 on every cell. The campaign stops per the acceptance stop rule; no fresh n=5 finalist confirmation (step 7) is warranted.

### 11.1 What was executed (all evidence under `results/2026/08/qwen38-dflash2-objective-20260830/`)

1. Pin + inventory (artifact-pin/): revision `dedf8df6…`, sha256 `67fc76d6…`, 81 tensors 1.924B params, config schema — PASS.
2. PR #52816 classified and ported as a 13-file 3-way merge onto the exact image generation (`ac7509e2b` base); XPU adaptation = torch selector walk + torch SDPA non-causal draft attention + NaN guard (unit-gates/).
3. Static gate 29/29 PASS; unit gates 7/7 PASS (causality, conv boundary, selector math vs independent reference, greedy walk, dup-aware cache fill).
4. Correctness gate: no-spec reference vs MTP7 (4/5 exact + 1 tie at pos 68, logprob-verified) and DFlash2 k=7 (5/5 coherent-prefix parity; the "natural" divergence at pos 90 sits inside the post-EOS degenerate tail where exact special-token ties were measured — `<|im_end|>` vs ` thinking` both −1.0632). Engine dummy-warmup NaN (14 lines) is warmup-scoped; all served requests finite. Zero cache hits on every request. **Correctness PASS.**
5. Matched n=3 screens (greedy, same prompts, zero cache hits):

| Cell (actual prompt) | no-spec | MTP7 | MTP8 (operational) | DFlash2 k=7 |
|---|---:|---:|---:|---:|
| p512 (501) e2e tok/s | 8.92 | **22.14** | 22.02 | 11.12 |
| p512 acceptance | — | 0.400 | 0.360 | **0.101** |
| p1024 (946) e2e tok/s | 8.45 | **21.90** | 20.21 | 11.86 |
| p1024 acceptance | — | 0.412 | 0.343 | **0.122** |
| p2048 (1836) e2e tok/s | 8.02 | **18.86** | 18.37 | 10.94 |
| p2048 acceptance | — | 0.347 | 0.321 | **0.105** |

Mean acceptance length at k=7: DFlash2 ≈ 0.71–0.85 vs MTP7 ≈ 2.2–2.9. The gate required ≥ MTP7 + 0.30. **Acceptance: FAIL (primary).** Efficiency and performance gates also FAIL (accepted tokens/s and e2e both far below MTP7; stretch target 65 tok/s unreached by any cell). Operational gate: PASS (servers healthy, cleanups complete, 150 W restored, cards empty).

### 11.2 Loss decomposition (the "why")

1. **Acceptance is the primary loss** (≈ −2.0 mean accepted length vs MTP7, ~3–4×). The ratio holds across all three prompt lengths — no content-dependent collapse, no length cliff, which rules out the p2048-style structural failure.
2. **Draft-path cycle latency adds a further ~1.9× e2e gap**: the XPU-adapted path runs a python-loop torch walk (7 steps × 16 candidates) plus per-layer dense context-cat SDPA per verify, and the draft itself must run 5 layers × (attention + conv + MLP) per speculative round. Even at MTP7 acceptance, DFlash2 would still trail on step time in this form.
3. **FP8 target-state mismatch is the leading acceptance mechanism**: the official draft was trained against BF16 target hidden states; the W8A16 verifier stack supplies quantized target states at `target_layer_ids`. The non-causal attention and selector/conv numerics were independently verified finite and correct at unit level, so the residual gap is most consistent with target-state distribution shift. **This is the documented next research target**, and it is exactly the case where a DFlash2 trainer (Inco/z-lab release pending) retrained on W8A16-decode hidden states would be the fix — the SpecForge route remains the diagnostic fallback.
4. **NOT implicated**: TP2 overhead (identical across variants), kernel ABI separation (python-level overlay on one coherent image, no binary mixing), prefix cache (zero hits everywhere), GDN/memory/device-safety gates.

### 11.3 Deliverable status

Per §9, the campaign fulfills its objective as a reproducible, evidence-backed no-go naming the failed gates (acceptance, efficiency, performance) and the precise next engineering blocker: **DFlash2 target-state fidelity on FP8 W8A16 TP2** (retrain on W8A16-decode hidden states once the trainer is available) and, secondarily, an XPU-native selector/attention path to cut draft-cycle latency. All results remain PROVISIONAL — NOT FOR PUBLIC HEADLINE pending the repository publication gate.


## 12. 2026-09-14 closure — root cause recovered, every bug fixed, still inferior; the full why

Follow-up campaign: `results/2026/09/qw38-mtp-depth-20260914/` (ledger RL-026..029). This
section supersedes the §11.2 attribution reasoning where they conflict and is the
final word on DFlash2 vs native MTP for this host.

### 12.1 What changed since §11

1. **The K-norm kernel bug** (upstream vLLM PR #56431; kernel fix vxk#579 merged
   2026-09-14 but unreleased): the XPU `rms_norm` kernel applied layer 0's
   K-norm weight to every draft layer in `_normalize_context_k` — the context
   precompute path. Our SDPA shim had already covered the draft-forward path, so
   §11's "correct numerics verified at unit level" was true for forward and
   false for context. Porting the per-layer fallback moved DFlash2 acceptance
   from ≈0.71–0.85 to ≈1.62–2.26 accepted tokens/step (GPTQ-INT4 TP1 target).
2. **The compile-path break** (vLLM issue #56787): DFlash2 could not run under
   torch.compile on XPU at all (inductor dynamic-shape stride assert in the
   custom-op fake kernel on the (5, ctx, 8, 128) context-KV layout). Fixed
   locally by stripping `@support_torch_compile` from `DFlashQwen3Model` +
   `@torch.compiler.disable` on the context path; compiled-target/eager-draft
   serving verified (~1.7× vs enforce-eager).
3. **The selector NaN explained**: all-flat NaN walks occur only in fully-eager
   mode. A 115-window per-position probe of draft hidden states in the compiled
   config shows 100% finite values and zero NaN fires — the second-defect
   suspicion of §11 is retracted; in the working configuration DFlash2 is
   numerically clean.

### 12.2 Final matched numbers (GPTQ-INT4 TP1, 230 W, compiled target, n=3)

| | DFlash2 k=7 (all fixes) | native MTP4 |
|---|---:|---:|
| p512/g128 endpoint post-first | 25.16 tok/s | **50.78 tok/s** |
| p8192/g128 endpoint post-first | 25.30 tok/s | **46.58 tok/s** |
| Mean accepted length | ≈1.4–2.2 | **≈3.3–4.0** |
| Draft cost per cycle | 5-layer 3.85 GB BF16 + selector walk | ~1 layer sharing target embed/head |

### 12.3 Why DFlash2 is inferior here (the general analysis)

1. **Acceptance does not survive target quantization.** The drafter was trained
   against BF16 hidden states; acceptance is conditioned on that distribution.
   Serving a GPTQ-INT4 (or FP8-decode) target shifts the distribution and the
   published 4.8-vs-4.28 advantage (BF16 H200) becomes ~2.0 — *below* the
   native head's ~3.5, which is quantization-exempt by construction (its tensors
   are excluded from quantization in our artifact).
2. **Cycle-cost asymmetry is independent of quantization.** Speculation
   throughput ≈ accepted length ÷ cycle time. DFlash2's denominator is a
   3.85 GB 5-layer draft plus selector walk; MTP4's is one shared-weight layer.
   Break-even would need DFlash2 acceptance ≈2× MTP4's; it measures ≈0.6×.
   Worse numerator AND worse denominator = the measured 2×.
3. **Per-position decay kills deep proposals.** Position-wise acceptance
   collapses to ≈0.11–0.26 by draft positions 5–6 on natural text, so k=7
   proposals mostly return nothing while their draft cost is paid up front.
4. **Hardware/workload scoping (fair to DFlash2):** the design's wins are on
   BF16 targets (matching its training), H200-class verify throughput, and
   batch serving that absorbs cycle cost. None holds on a 32 GB single card at
   C1 with a quantized dense target. The closing is scoped exactly there.

### 12.4 Standing retry condition (unchanged in spirit, now precise)

Reopen only for a drafter **trained on the quantized target's actual hidden
states** (SpecForge-class capture against GPTQ-INT4/FP8-decode serving) that
demonstrates ≥1.5× the native head's acceptance on held-out natural text
*before* any throughput campaign. A port of another BF16-trained artifact — or
any "retry now that the PR merged" impulse — is pre-rejected by this section.

All of §12 remains PROVISIONAL — NOT FOR PUBLIC HEADLINE pending the repository
publication gate; the public-facing summary lives in the cookbook
(`docs/qwen38-27/MTP-DEPTH-DFLASH2-KNORM-20260914.md`, PRs #20/#21).

## 10. Safety and operational constraints

- Before every model load: both-card empty-GPU, process, container, VRAM, hwmon discovery, and 150 W restoration gates (AGENTS.md §4–5, §10). One declared server and one harness only.
- TP2 worker affinity explicit; graph capture disabled on TP2. No swap as model capacity.
- Do not delete `vxk-build`, BF16 shards, or any model artifact during this campaign.
- External metadata/artifact access: HTTPS only, validated destination host; immutable revision + retained hashes.
- On every exit path: stop the exact server/container, restore 150 W, retain cleanup evidence.
- No commit, push, publish, or external submission is implied by campaign completion.