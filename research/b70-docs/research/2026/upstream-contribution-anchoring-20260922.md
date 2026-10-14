---
title: "Upstream contribution anchoring — draft-side INT4 (vLLM) and grouped IQ MoE GEMM (llama.cpp) — 2026-09-22"
type: research
date: 2026-09-22
status: living
---

# Upstream contribution anchoring (2026-09-22 archaeology pass)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Anchoring facts only; no contribution
prose drafted here (operator authorization required before anything leaves this host).
Every claim below carries the full-SHA raw URL it was fetched from and the date
observed (all observations 2026-09-22, UTC, via `gh api` against the public APIs of
`vllm-project/vllm`, `ggml-org/llama.cpp`, `intel/*`, `uxlfoundation/oneDNN`,
`vllm-project/vllm-xpu-kernels`, plus the pinned local tree `~/llama.cpp`).

Companion evidence: `research/2026/qwen38-draft-side-int4-upstream-20260921.md`
(TOPIC A claim + numbers), `docs/plans/2026-09-20-b70-improvement-roadmap/OUTCOMES.md`
W1/W3/L7 (levers), `results/2026/09/b70-improvement-probes-20260920/` (raw),
`research/2026/xmx-dpas-rate-classes-20260915.md` (DPAS rate classes).

## 0. Identity pins used for comparison

| Object | Identity (observed 2026-09-22) |
|---|---|
| Local llama.cpp tree | `~/llama.cpp` detached at `1692f9e50` (2026-08-14, "ggml : recurrent state rollback for ggml_ssm_scan (#26623)"), dirty: `common/arg.cpp`, `common/common.{cpp,h}`, `ggml-backend*`, `ggml-sycl.cpp` (+2 lines), `mmvq.cpp` (+38 lines: device-ID IQ3_XXS/IQ3_S cases), `test-backend-ops.cpp`; untracked `moe-cache.*`, `ggml-sycl.cpp.bak-fix` |
| llama.cpp master HEAD | `f46bc30cb6a7f68a67e34a00061e20a4ad1eff43` (2026-09-22T20:31Z) |
| vLLM main HEAD | `01e15bc94cd29cd1fcbbab484e96acb5b4f6bb14` (2026-09-22T21:26Z) |
| vLLM latest release | `v0.30.0` (2026-09-22T05:20Z) |
| vLLM champion image (host) | digest `f01e24f6…` (champion stack, per `docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/P4-20260920.md`) — in-container commit NOT re-verified this pass |

## 1. TOPIC A — draft-side INT4 for vLLM speculative decoding

### 1.1 Current proposer state on main

Source (full SHA = last commit touching the file, content identical at main HEAD `01e15bc94`):
<https://raw.githubusercontent.com/vllm-project/vllm/c723a831a81cb4ff89ea6d61b2d15109306ed1bc/vllm/v1/spec_decode/llm_base_proposer.py>
(last touched by "[Fast Start] Cache the MTP draft model in a separate daemon group (#57312)", 2026-09-22T03:13Z)

- `_maybe_share_lm_head` **still exists** (def at :1508). For MTP models it is
  **unconditional**: `share_lm_head = True` at :1547 with log "Detected MTP model.
  Sharing target model lm_head weights with the draft model."
- The share replaces the draft's head in-place: `self.model.lm_head = target_language_model.lm_head` (:1556).
- **NEW since our 2026-09-21 note** (NaN fix): :1559–:1573 now *also* re-points each
  MTP layer's `shared_head.head` (a `ParallelLMHead` inside each MTP layer) at the
  target's `lm_head`, because "If the checkpoint omits a copy of the lm_head weights
  at the MTP layer path, shared_head.head stays uninitialised and produces NaN logits."
  Any contribution that quantizes "the draft's head" must handle BOTH sites or the
  sharing is only half-undone.
- `propose()` structure is unchanged: step-0 forward `self.model(**model_kwargs)`
  (:600) then `for token_index in range(self.num_speculative_tokens - 1):` (:689)
  with one forward per iteration (:754). Logits/head application per draft step goes
  through `_sample_draft_tokens` (:478) → greedy: `self.model.compute_logits(hidden_states).argmax(dim=-1)` (:448);
  probabilistic: `self.model.compute_logits(hidden_states)` (:485).
- **Mechanism sanity-check: CONFIRMED on main.** For MTP4: 1 + (4−1) = 4 draft
  forwards, each applying the (shared, F16) head once, plus the target's verify pass
  (5 rows against the target's own F16 head) = 5 head applications per speculative
  step — matching the 5 × 2.543 GB ≈ 12.7 GB/step byte model in the 2026-09-21 note.
  Nothing upstream changes that arithmetic.
- No quantization axis exists anywhere in the proposer (grep for `quant` in the file
  returns nothing relevant).

### 1.2 Draft quantization config axis (canonical way today)

Source: <https://raw.githubusercontent.com/vllm-project/vllm/c723a831a81cb4ff89ea6d61b2d15109306ed1bc/vllm/config/speculative.py>
(same last-touch commit as above)

- `SpeculativeConfig.quantization: me_quant.QuantizationMethods | str | None = None`
  (:416) — "Quantization method that was used to quantize the draft model weights.
  If `None`, we assume the model weights are not quantized. Note that it only takes
  effect when using the draft model-based speculative method."
- It flows into the draft `ModelConfig` at :1300 (`quantization=self.quantization`
  inside the `ModelConfig(... runner="draft" ...)` construction at :1281).
  The draft otherwise **inherits** the target's `dtype` (:1293) and, since
  `_DRAFT_VLLM_CONFIG_OVERRIDES` (:376–:382), can also override `moe_backend`,
  `attention_backend`, `kv_cache_dtype` — per-field draft overrides are an
  established pattern our contribution can extend.
- CLI surface: `quantization` is reachable ONLY through the JSON dict
  `--speculative-config '{"quantization": "gptq", ...}'` (short flags exist only for
  `--spec-method`, `--spec-model`, `--spec-tokens`; verified in
  `vllm/engine/arg_utils.py` at main, same commit family).

### 1.3 Existing issues/PRs in the neighborhood (numbers, dates, state)

| # | Date | State | Relevance |
|---|---|---|---|
| [#40849](https://github.com/vllm-project/vllm/pull/40849) | 2026-04-25 | OPEN PR | "[Spec Decode] Inherit online quantization for draft models" — propagates *online* (load-time fp8) quantization to external drafts for **memory**. Root cause it cites: quantization inheritance only ran in the `self.model is None` (MTP) branch. Memory-motivated; not our bandwidth/throughput claim. |
| [#48816](https://github.com/vllm-project/vllm/pull/48816) | 2026-07-23 | MERGED | "Fix GPTQ quantized Qwen3.5 MTP weight loading with spec decode" — MTP layers unquantized inside a quantized checkpoint crash loading; fix skips quant config for MTP layers. Establishes the "draft unquantized-build contract". |
| [#55527](https://github.com/vllm-project/vllm/pull/55527) | 2026-09-06 | OPEN PR | Regression-test the Qwen3.5-MTP draft unquantized-build contract (#48816 follow-up). |
| [#49553](https://github.com/vllm-project/vllm/pull/49553) | 2026-07-23 | OPEN PR | Per-layer quant config (`block_name_to_quantize`) silently unresolved for draft classes (`hf_to_vllm_mapper` missing on registered MTP classes) → every MTP layer resolves Unquantized. Fixes #49552. |
| [#53122](https://github.com/vllm-project/vllm/pull/53122) | 2026-08-20 | OPEN PR | populate draft model quant mapping + guard DFlash fused KV. |
| [#55262](https://github.com/vllm-project/vllm/pull/55262) | 2026-09-04 | OPEN PR | Support GPTQ-quantized DFlash draft models in context-KV precompute. |
| [#40425](https://github.com/vllm-project/vllm/pull/40425) | 2026-04-21 | OPEN PR | Fix quantized DFlash Qwen3 draft support. |
| [#38494](https://github.com/vllm-project/vllm/pull/38494) | 2026-03-30 | OPEN PR | Step-3.5-Flash MTP acceptance fix by detecting own shared_head weights. |
| [#53214](https://github.com/vllm-project/vllm/pull/53214) | 2026-08-21 | OPEN PR | single-file target+draft checkpoints. |
| [#25618](https://github.com/ggml-org/llama.cpp/issues/25618) | 2026-07-13 | OPEN issue (llama.cpp) | spec-decode greedy divergence on quantized targets — adjacent correctness context for "quantized draft is safe" claims. |

**Gap confirmed:** every item above is about *correctly loading* or *memory*; none
claims the **decode-throughput lever** (draft-only W4A16 while the target's F16 head
stays untouched) and none measures it as a function of host-overhead removal. No
existing issue/PR found on "draft lm_head bandwidth" (searched `share_lm_head`,
"draft lm_head bandwidth" — empty, 2026-09-22).

### 1.4 XPU graph capture state on main

Source: <https://raw.githubusercontent.com/vllm-project/vllm/e4340e41c9a32b2247cc2442b35aa91b081c424f/vllm/platforms/xpu.py>
(last touch "[Feat][XPU] VLLM_BATCH_INVARIANT support for Dense/MoE models (#55881)", 2026-09-22T15:06Z)

- `VLLM_XPU_ENABLE_XPU_GRAPH` still gates capture (:339–:344), still default-off,
  still warned as "experimental and currently only supports single-GPU execution" (:346–:348).
- **Blocking correctness caveat for our claim:** issue
  [#54785](https://github.com/vllm-project/vllm/issues/54785) (OPEN, 2026-09-01,
  by MikkoP88, label `intel-gpu`): "XPU graph capture + MTP num_speculative_tokens=4
  produce non-deterministic, wrong logits at temperature=0 (k<=3 clean, eager clean)".
  Related: [#54796](https://github.com/vllm-project/vllm/issues/54796) (inductor ×
  spec-decode livelock ≥32k), [#54698](https://github.com/vllm-project/vllm/issues/54698)
  (graph replay hang), [#56917](https://github.com/vllm-project/vllm/issues/56917)
  (TP2 capture + MTP crash on Arc B70). Our A/B arms both ran capture-on so the
  *delta* stands, but any upstream write-up must disclose that k=4 + capture itself
  has an open correctness bug we did not author and cannot yet explain.

### 1.5 Recommended contribution shape for TOPIC A (facts → shape, no prose)

- Anchor: main `c723a831a8` (both files above).
- Shape: a `SpeculativeConfig` boolean (e.g. `share_draft_lm_head`, default `True`,
  following the `_DRAFT_VLLM_CONFIG_OVERRIDES` / draft-override pattern in
  `speculative.py` :376–:427) that, when false (or when
  `speculative_config.quantization` is set for the draft), keeps a **private,
  optionally-quantized** draft head instead of aliasing the target's — covering BOTH
  `self.model.lm_head` (:1556) and each MTP layer's `shared_head.head` (:1559–:1573).
- Expected reviewer objections to pre-load: (1) memory regression by un-sharing
  (2.54 GB F16) — answer: quantized private copy is ~0.66 GB; (2) duplicate-weight
  drift — answer: verification semantics unchanged; (3) interacts with #54785;
  (4) "why is this not just #40849?" — answer: #40849 is online-quant inheritance for
  memory, ours is a bandwidth lever measured under capture.
- XPU-specific framing is honest but NOT XPU-exclusive: the 5×head-re-read byte model
  is architecture-independent; XPU graph capture is what made it *measurable* here
  (CUDA graphs would be the analogous enabler — unverified, do not claim).

## 2. TOPIC B — grouped sub-byte MoE GEMM in llama.cpp SYCL

### 2.1 Current MUL_MAT_ID state on master (`f46bc30cb`, observed 2026-09-22)

Source: <https://raw.githubusercontent.com/ggml-org/llama.cpp/f46bc30cb6a7f68a67e34a00061e20a4ad1eff43/ggml/src/ggml-sycl/ggml-sycl.cpp>
(and `dmmv.cpp`, `mmvq.cpp`, `mmq.cpp` at the same SHA)

- `ggml_sycl_mul_mat_id` (:5138): when `ne12 == 1` (decode) it first tries
  `ggml_sycl_mul_mat_id_mmvq_fused` (:5037) — this quantizes activations to q8_1
  on-device and passes `(const int32_t *) ids->data` **directly** (no D2H, no
  `stream->wait()`), dispatching `ggml_sycl_mul_mat_vec_q_id`.
- **IQ3_XXS/IQ3_S are IN the fused decode dispatch on master** — `mmvq.cpp` :2868
  (IQ3_XXS) and :2873 (IQ3_S) under `ggml_sycl_mul_mat_vec_q_id` (:2774), plus a
  SoA `*_reorder` variant (:2903–:2964). This landed in
  **PR [#28476](https://github.com/ggml-org/llama.cpp/pull/28476) "SYCL: Add IQ type
  handling for MoE"** (author cwriter, MERGED 2026-09-09, commit `304665fe7`), which
  exists precisely because "missing IQ quant support in `mul_mat_vec_q_moe` caus[ed]
  a host serialization and therefore horrific performance" (+60.4 % tg512 on
  3× Arc Pro B60, IQ3_XXS UD quant; touched only `mmvq.cpp` + `vecdotq.hpp`).
  **Our pinned tree's uncommitted +38-line mmvq patch is therefore REDUNDANT against
  master** (same fix class, decode path).
- **Multi-token (prefill, `ne12 != 1`) path is UNCHANGED and still host-bound**:
  `std::vector<char> ids_host` + `stream->memcpy(ids_host.data(), ids_dev, …)` +
  `stream->wait()` (:5157–:5162), then host counting sort
  (`mmid_counting_sort_rows`, :5114) and per-expert `ggml_sycl_mul_mat` calls
  (:5207, :5282). A source comment at :6117 still documents the blocking wait.
- **SYCL MMQ is dead code**: `ggml_sycl_supports_mmq()` (:4018) returns `false`
  unconditionally ("// TO-DO: accuracy issues in MMQ"), so `use_mul_mat_q` can never
  be true in `ggml_sycl_mul_mat`'s decision tree (:4798) and every quantized
  multi-token mul_mat falls to the final `else` — dequantize-to-F16 + oneMKL GEMM
  (`ggml_sycl_op_mul_mat_sycl`). This has been false since
  `b864b50ce5e2beefc8c2fd31733e4e1a978b7754` "[SYCL] Align GEMM dispatch (#7566)"
  (2024-05-29; verified still false at our pin :3740 and on master :4018). The
  dormant `mmq.cpp` (3030 lines) covers only Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K–Q6_K —
  **no IQ types**.
- CUDA contrast: `ggml/src/ggml-cuda/mmq.cu` at the same SHA dispatches IQ3_XXS
  (:60) and IQ3_S (:63) — in-kernel dequant GEMM for sub-byte weights exists as
  prior art on CUDA, not on SYCL.

### 2.2 THE COLLISION: PR #29245 already proposes our kernel

**PR [#29245](https://github.com/ggml-org/llama.cpp/pull/29245) "sycl: add grouped
MoE XMX GEMM"** — OPEN, filed 2026-09-21T18:19Z (yesterday), author **cwriter**
(same author as #28476), 0 reviews, 0 comments as of 2026-09-22.

- Approach (from the PR body): "Build a host-side schedule of active expert tiles
  and launch as one grouped kernel instead of one GEMM per expert"; "Dequantizes
  **IQ4_NL or IQ3_S** weights into f16 tiles in local memory"; "Packs routed f32
  activations into the VNNI f16 layout consumed by XMX" via `joint_matrix`.
- Files: `ggml/src/ggml-sycl/common.hpp`, `fused-gemm.cpp/.hpp`, `ggml-sycl.cpp`.
- Numbers (their rig, `unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS`): pp 274.1 →
  368.3 (**+34.4 %**), tg +0.95 %.
- Implication for us: the contribution we planned ("grouped-IQ3 kernel that
  reconstructs weights tile-locally instead of materializing whole experts as FP16")
  is **already open upstream**. Our differentiators that remain: **IQ3_XXS** (their
  list names IQ4_NL/IQ3_S), **216-expert** K216 shape, our isolated-op baseline
  (28.87 ms/op, 1.79 TFLOPS, ~25× dense gap; `results/2026/09/b70-improvement-probes-20260920/perf-k216-prefill-n512.txt`),
  and a B70 end-to-end gate. The clean move is engagement/review/benchmarking of
  #29245 + an IQ3_XXS/K216 extension, NOT a competing PR (llama.cpp AGENTS.md
  explicitly asks contributors to check existing PRs and work with the author).
- Adjacent prior art: **#27402** (MERGED 2026-08-31) "AVX2: Speed up large batch
  size prompt processing of IQ models" — CPU-side `block_iqp_x8` GEMM panel decoding
  8 rows × 256 cols once per batch instead of per-token; the CPU-side precedent for
  tile-local IQ reconstruction. Also #29107 (OPEN 2026-09-18, "sycl: IQ3 code
  reorder"), #28725 (OPEN 2026-09-10, "Add SYCL graph record and replay"), #25089
  (OPEN 2026-06-28, graph compatibility for the MUL_MAT_ID fused path), #24635
  (MERGED 2026-06-15, bound in-flight expert matmuls in mul_mat_id), #25455
  (CLOSED 2026-07-08, MUL_MAT_ID prefill wrong-results history on B70).

### 2.3 oneDNN / IPEX / torch-xpu-ops / DPAS facts

- **oneDNN**: repo is now `uxlfoundation/oneDNN` (redirects from `oneapi-src/oneDNN`
  work for the repo API but NOT for issue search). Latest release
  **v3.13.2, 2026-08-26**. Int4: [#1712](https://github.com/uxlfoundation/oneDNN/issues/1712)
  "Add support for INT4/UINT4" CLOSED 2024-06-11 (body is a generic feature
  request; no GPU-Xe2 delivery note visible);
  [#3641](https://github.com/uxlfoundation/oneDNN/issues/3641) "Please provide more
  4bit matmul docs and example?" OPEN 2025-08-18;
  [#2081](https://github.com/uxlfoundation/oneDNN/issues/2081) (ARM 8/4-bit weight
  decompression) OPEN. **No Battlemage/Xe2-tagged int4 matmul issue found**
  (searched "battlemage OR xe2" — empty, 2026-09-22). Treat "oneDNN Xe2 sub-byte
  GEMM" as UNVERIFIED — do not cite it as available.
- **IPEX is ARCHIVED**: repo README (main HEAD `6d3ba89cf9c40eb7184d6092ab7fa9b5231553c1`,
  2026-03-30, "Add discontinuation notice") — "THIS PROJECT IS ARCHIVED … Patches to
  this project are no longer accepted by Intel." Last XPU release tag
  `v2.8.10+xpu` (2025-08-06); `main`'s `IPEXWeightOnlyQuantizedLinear.from_weight`
  now asserts `device_type in {"cpu"}` — the XPU WOQ path was removed from IPEX.
  **`torch.ops._xpu_C.int4_gemm_w4a16` does NOT exist under that name anywhere
  current**; the successor surface is:
- **intel/torch-xpu-ops** (public, active; main HEAD `9e33153d7636eaedcd8f6c50a50d4b844e15d1e2`,
  2026-09-22): contains `src/ATen/native/xpu/LinearInt4.cpp`,
  `WeightInt4Pack.cpp`, `sycl/LinearInt4.cpp`, `sycl/Dequant_int4.cpp`,
  `test/regressions/test_int4pack.py`. The W4A16 GEMM is
  **`at::native::_weight_int4pack_mm_xpu`**, i.e. the ATen op
  `torch.ops.aten._weight_int4pack_mm` on XPU: A ∈ {BF16, F16, F32} 2-D contiguous,
  B ∈ {int32, uint32, uint8} packed, `qGroupSize ∈ {16, 32, 64, 128, 256}` — g128
  is supported. This repo is cloned into PyTorch via `third_party/xpu.txt` and is
  where "the int4 GEMM in the box" lives now.
  Raw: <https://raw.githubusercontent.com/intel/torch-xpu-ops/9e33153d7636eaedcd8f6c50a50d4b844e15d1e2/src/ATen/native/xpu/LinearInt4.cpp>
- **DPAS geometry** (our own measured authority — `xmx-dpas-rate-classes-20260915.md`):
  Xe2 DPAS selects `OPS_PER_CHAN=4` whenever EITHER integer operand is 8-bit, so
  every 8×8/8×4/8×2/4×8 mix runs one rate class (363 TOPS class vs 170 BF16);
  `OPS_PER_CHAN=8` (pure 4×4) doubles K-instruction throughput — measured
  s4×s4 = 33.91 vs s8×s8 = 16.96 TOPS in the same ESIMD harness = exactly 2.00× —
  but sub-byte weights remain a bandwidth/capacity lever, not a TOPS multiplier.
  Consequence for #29245-style kernels: dequant-to-F16-tiles + `joint_matrix`
  (BF16 class) is not automatically wrong on Xe2; the win is traffic, not rate class.

### 2.4 Recommended contribution shape for TOPIC B

1. **Do not open a competing grouped-GEMM PR.** Engage #29245 first (comment/review
   is an external action — operator authorization required; prepare a draft comment
   locally). Our concrete value-adds: IQ3_XXS coverage, 216-expert K216 measurements,
   the isolated-op gate (`≥2×` vs the 28.87 ms/op baseline), and the
   host-sync/graph-capture interaction notes.
2. If extending independently, the cleanest shape is the one #29245 also chose:
   new `ggml/src/ggml-sycl/fused-gemm.{cpp,hpp}` + a dispatch hook in
   `ggml_sycl_mul_mat_id` behind the existing fused-path pattern (as
   `ggml_sycl_mul_mat_id_mmvq_fused` does at :5154), NOT a new backend op and NOT
   reviving dead `mmq.cpp` wholesale. Re-enabling `ggml_sycl_supports_mmq()` is a
   separate, riskier conversation (16 months dead for "accuracy issues").
3. Bug-fix-class rule from llama.cpp CONTRIBUTING.md: reproducible issue +
   regression test that fails before / passes after, `llama-perplexity` +
   `llama-bench` not negatively affected, full local CI pass before publishing.

## 3. TOPIC C — version pins (observed 2026-09-22)

| Component | Current (2026-09-22) | Notes |
|---|---|---|
| vLLM main HEAD | `01e15bc94cd29cd1fcbbab484e96acb5b4f6bb14` (21:26Z) | latest commit "[Bugfix] Backport Inductor custom-op pattern matching fix (#58189)" |
| vLLM latest release | `v0.30.0` (2026-09-22T05:20Z) | |
| vLLM XPU install path | wheels at `wheels.vllm.ai/nightly/xpu` (+ per-commit), `triton==3.7.2+xpu`, PyTorch from `download.pytorch.org/whl/xpu`, Python 3.12 required | source: `docs/getting_started/installation/gpu.xpu.inc.md` at main; compute-runtime ≥ 26.18 recommended for vllm-xpu-kernels ≥ 0.1.10 |
| vllm-xpu-kernels | latest release **v0.1.15** (2026-09-22); main HEAD `d7c35d281cf50564996fb0b9c41676b64f13f970` (2026-09-18) | public, active |
| llama.cpp master HEAD | `f46bc30cb6a7f68a67e34a00061e20a4ad1eff43` (2026-09-22T20:31Z) | |
| llama.cpp latest release | **v0.4.1** (2026-09-14) — newer than our pin | tags v0.4.1, v0.4.0, v0.3.0, v0.2.0, v0.1.2 |
| oneDNN | **v3.13.2** (2026-08-26), repo `uxlfoundation/oneDNN` | Xe2 sub-byte GEMM: no evidence found; UNVERIFIED |
| IPEX | **ARCHIVED**; last xpu tag `v2.8.10+xpu` (2025-08-06); main = discontinuation notice `6d3ba89c` (2026-03-30) | do not target IPEX for anything new |
| torch-xpu-ops | main `9e33153d7` (2026-09-22); no releases/tags | `_weight_int4pack_mm_xpu` lives here (W4A16, g16–g256) |

## 4. What changed since our pinned versions

| Pin | Was | Upstream now (2026-09-22) | Impact |
|---|---|---|---|
| llama.cpp `1692f9e50` (2026-08-14) | IQ3_XXS/IQ3_S missing from fused SYCL MUL_MAT_ID MMVQ; our local +38-line device-ID patch uncommitted | #28476 (merged 2026-09-09) added IQ decode handling upstream; also new: SoA reorder GEMV, ESIMD DMMV variants, pinned-memory device-context fix (#28895), B70 >19.3 GB alloc fix (#28953), SYCL graph record/replay PR open (#28725) | **Local patch redundant — rebase drops it.** L7/AGENTS §16 wording is now stale vs upstream. Any re-measure must re-base. |
| llama.cpp `1692f9e50` | No grouped MoE GEMM anywhere in ggml-sycl | #29245 OPEN (2026-09-21): grouped XMX GEMM, IQ4_NL/IQ3_S, +34 % pp | **Our planned TOPIC B contribution collided.** Pivot to review/extend. |
| llama.cpp `1692f9e50` | `ggml_sycl_supports_mmq()` == false since 2024-05-29 (#7566) | still false on master | unchanged; prefill dequant+F16+oneMKL remains the only quantized multi-token path |
| vLLM champion digest `f01e24f6…` | proposer semantics as our 2026-09-21 note | main `c723a831a8`: `_maybe_share_lm_head` now ALSO patches per-layer `shared_head.head` (NaN fix); Fast-Start MTP daemon caching (#57312) landed on top | draft-head quantization patch must cover both sharing sites; rebase before any submission |
| IPEX-era assumptions | `torch.ops._xpu_C…` style ops | IPEX archived; ops live in `intel/torch-xpu-ops` as `aten::_weight_int4pack_mm` (`_weight_int4pack_mm_xpu`) | any W4A16 kernel reuse should target the ATen op, not IPEX |

## 5. Contribution-policy constraints (verified 2026-09-22)

- **llama.cpp** `CONTRIBUTING.md` @ `f46bc30cb`: AI-generated code allowed but
  "Undisclosed AI usage may result in your account being permanently banned";
  explicit disclosure required; check for an existing PR and work with its author;
  ~1 h review per 200–400 LOC expected; bug-fix PRs need reproducible issue +
  regression test; new CLI/public API carries a higher bar. Repo-root `AGENTS.md`
  @ `f46bc30cb` adds: "AI-generated code is allowed. What is not allowed is
  submitting code you do not understand"; open an issue to gauge interest BEFORE
  implementing. Note both #28476 and #29245 carry the "AI usage disclosure: YES"
  block — the norm on this repo now.
  Raw: <https://raw.githubusercontent.com/ggml-org/llama.cpp/f46bc30cb6a7f68a67e34a00061e20a4ad1eff43/CONTRIBUTING.md>,
  <https://raw.githubusercontent.com/ggml-org/llama.cpp/f46bc30cb6a7f68a67e34a00061e20a4ad1eff43/AGENTS.md>
- **vLLM** `CONTRIBUTING.md` @ main just links to docs.vllm.ai/en/latest/contributing;
  no CLA text found in the file itself (contributing flow: issue → PR → code-owner
  review). Re-verify the current PR checklist at submission time.
  Raw: <https://raw.githubusercontent.com/vllm-project/vllm/c723a831a81cb4ff89ea6d61b2d15109306ed1bc/CONTRIBUTING.md>

## 6. Unverified / open items

- In-container vLLM commit of champion digest `f01e24f6…` not re-verified this pass.
- #29245's exact IQ coverage (whether IQ3_XXS is included but unlisted) — needs the
  diff read; 0 reviews so far. Its `fused-gemm.cpp` not yet fetched.
- oneDNN Xe2/BMG int4 matmul: absence of evidence only.
- Whether the ATen `_weight_int4pack_mm` XPU path is wired into any vLLM XPU
  quantization method today (GPTQ W4A16 on XPU dispatch target) — not traced.
- #54785 (XPU graph + MTP k=4 wrong logits) root cause — not ours, unresolved;
  gates how strongly we can word the "capture as enabler" framing.

## 7. Ledger

Row to be filed by the orchestrator (typed JSON) — this note is the evidence payload.
Suggested question: "Upstream anchoring 2026-09-22 for draft-side-INT4 (vLLM) and
grouped-IQ3 MoE GEMM (llama.cpp) contributions — current state, collision #29245,
pin drift". Status: PROVISIONAL.
