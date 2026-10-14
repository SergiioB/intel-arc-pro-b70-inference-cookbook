---
title: "B70 Quantization Decision Record"
status: living
type: reference
date: 2026-08-22
---

---
id: b70-ref-quantization-decision-record
title: B70 Quantization Decision Record
document_type: reference
status: canonical
version: "1.3"
effective_date: 2026-08-19
last_reviewed: 2026-09-14
authority: docs/methods/quantization-evaluation.md
scope: Current B70 decisions for model-weight, KV-cache, and draft quantization
---

# B70 Quantization Decision Record

## Decision status

Accepted for current B70 documentation, subject to exact engine, checkpoint, artifact availability, correctness, capacity, and workload evidence. This record consolidates decisions; it does not replace the detailed evidence linked below or create universal hardware claims.

## Context

The B70 serves both llama.cpp SYCL GGUF and vLLM XPU checkpoint paths. Quantization choices affect four independent outcomes:

- engine/kernel compatibility;
- model and task quality;
- VRAM/context capacity; and
- throughput or latency at a named workload.

A single “best quant” across engines and models would hide these differences. The repository therefore uses gate-based, per-path decisions under the [quantization evaluation method](../methods/quantization-evaluation.md).

## Decisions

### 1. llama.cpp KV cache

**Decision:** retain `q8_0` K + `q4_1` V with flash attention as the fleet-standard llama.cpp KV configuration.

**Rationale:** the existing KL-divergence analysis finds that protecting K precision while compressing V provides the current quality/capacity trade-off. K controls attention routing and is more sensitive than V.

**Prohibition:** do not use `q4_0` for K cache. Existing evidence records a catastrophic quality cliff. Model-weight `Q4_0` is a separate format and does not relax this prohibition.

**Evidence:** [KV Cache Quantization — Definitive Reference](../reference/kv-cache-quantization.md) and [B70 Benchmark Protocol §2](../methods/benchmark-protocol.md).

**Limit:** this decision is specific to the recorded llama.cpp KV formats and evidence. Other engines use their own KV implementations and must be evaluated separately.

### 2. llama.cpp model weights

**Decision:** select GGUF quantization per model and task; do not publish a universal Q4/Q5/Q6 winner.

- Prefer a native-source artifact when it meets quality and capacity requirements.
- Treat a local requantization as a separately named lossy artifact.
- Require target-task quality validation before adopting a throughput-first lower-bit candidate.
- Keep context headroom, MTP buffers, and multimodal components in the capacity decision.

The recorded Nemotron model-weight `Q4_0` speed result is a model-specific historical finding from a local requantization, not a general Xe2 rule and not a current artifact-availability claim. See [Recommended Configurations](../operations/recommended-configs.md) and its linked evidence.

### 3. vLLM XPU model weights

**Decision:** treat GPTQ-INT4 through the intended XPU kernel as a demonstrated path only for exact architectures and checkpoints that have loaded and passed bounded correctness tests.

Do not infer equivalent support for AWQ, MXFP4, NVFP4, compressed-tensors, or another GPTQ packing from the common bit width. Verify architecture registration, packing, group size, symmetry, activation order, zero-point handling, ordinary linear dispatch, and routed-expert dispatch.

A compatible repack may be lossless only if represented values are preserved. MLX affine INT4 to symmetric GPTQ and similar transformations are dequantize/requantize operations and require new correctness and task-quality evidence.

**Evidence:** [Quantization Format Strategy](../../research/2026/quantization-format-strategy.md), [Recommended Configurations](../operations/recommended-configs.md), and exact campaign manifests indexed by [Benchmark History](../../archive/legacy-docs-20260915/benchmark-history.md).

**Limit:** some research paths depend on local patches or overlays. A local-only image or artifact is not a public reproduction prerequisite.

### 4. Baseline for new B70 vLLM XPU conversions

**Decision:** start a new compatible conversion from a separately named, symmetric INT4 (W4A16) gptq-packed candidate with mixed-domain calibration, produced by **AutoRound (signSGD) as the default converter** (ADR-0007, operator-accepted 2026-08-22). GPTQModel remains the documented fallback when autograd or host memory blocks AutoRound; the blocker must be recorded in the conversion manifest. Quantize routed expert projections first. Keep attention, routers, shared experts, embeddings, LM head, vision tensors, and MTP/draft tensors at their source precision until each additional quantization scope passes its own quality and workload gates.

This is a conversion baseline, not a claim that group size, tensor scope, or gptq packing is universally best. Choose the group size only after the exact checkpoint layout and XPU kernel dispatch are known. The current starting candidates are symmetric group-64 or group-128; neither may be substituted for the other without matched evidence.

The calibration set must cover the intended use, such as code, reasoning, tool calls, technical prose, and general instruction following. Preserve the sample count, token count, seed, source revision, excluded tensors, and every fallback reported by the quantizer. A quantizer fallback, including RTN, is a property of the saved weights. It is not evidence of a CPU fallback, XPU kernel fallback, or unquantized runtime tensor. AutoRound's advantage of record: it tuned 30,720/30,720 targeted projections with zero RTN fallback in the Ornith head-to-head, where GPTQModel left 10.37–24.76% untuned.

The Ornith MixedCal-v2 artifact (GPTQModel) remains the serving reference and the first complete example of the scope/format baseline; the Ornith AutoRound artifact (`…-AutoRound-W4A16-sym-G128-MTP-BF16`) is the first complete example of the converter baseline, and on 2026-08-22 it passed reference logprob parity against the BF16 source — equal-or-best vs the GPTQ control on mean Δlogprob (0.2006 vs 0.2025), top-1 agreement (91.4% vs 90.5%), and top-20-union KL (0.277 vs 0.319) over 315 teacher-forced positions (`results/2026/08/ornith15-ar-parity-20260822/`). Neither establishes task-quality parity, and neither makes its exact group-128 choice a fleet rule. See the [Ornith artifact record](../../models/ornith/ornith-1.5-35b-a3b/artifacts.md), [ADR-0007](../../decisions/adr-0007-ornith-autoround-w4a16-candidate.md), and the [head-to-head campaign](../../research/2026/ornith15-autoround-vs-gptq-20260822.md).

Before promotion, require all of the following:

- static tensor and index coverage, including forbidden-tensor checks;
- numerical and deterministic comparison against the declared source/reference;
- exact XPU loader and routed-expert kernel dispatch evidence;
- staged target-context capacity checks; and
- matched task-quality and C1/Cn workload evidence for the intended route.

### 5. Follow-up conversion candidates

Evaluate one variable at a time against the baseline artifact. Rank candidates in this order:

1. Improve calibration coverage and reduce fallback matrices while keeping format, tensor scope, and runtime fixed.
2. Quantize MTP draft and LM-head tensors in a separately named artifact or portable overlay, then compare speculation-off and MTP-on quality, acceptance, capacity, and latency.
3. Compare group-64 and group-128 symmetric GPTQ at matched tensor scope and calibration data.
4. Add selected non-expert tensors only when reference and task-quality tests show that the VRAM or speed benefit is worth the loss risk.
5. Reassess non-GPTQ formats only after their exact loader and XPU kernel path are demonstrated for the target model.

### 6. Activation quantization

**Decision:** activation-quantized performance paths are experimental until they pass parity and target-task quality tests. A kernel speedup alone is insufficient.

The repository contains a prior W4A8 research path that was discarded after observed quality divergence. Preserve it as historical evidence; do not promote it into recommendations or public performance claims. See the correction in [Benchmark History](../../archive/legacy-docs-20260915/benchmark-history.md).

### 7. Draft-model and MTP quantization

**Decision:** evaluate draft quantization independently from main-model quantization.

A draft candidate must pass:

- speculation-off main-model control;
- speculation-on coherence and deterministic tests;
- target-task quality comparison;
- acceptance and fallback diagnostics;
- matched C1 and, when intended, Cn workload measurements; and
- capacity/VRAM checks for the deployed context.

Optimize the declared throughput or latency metric, not acceptance rate. A faster short synthetic cell does not establish long-context or concurrent serving quality.

## Addendum 2026-09-14 — quant-format showdown re-confirmation (Qwen3.8-27B, one B70)

Source campaign: [`research/2026/quant-format-showdown-20260913/FINDINGS.md`](../../research/2026/quant-format-showdown-20260913/FINDINGS.md)
+ `results/2026/09/qw38-quant-format-showdown-20260913/` (claims-audited; ledger RL-024/RL-025;
cookbook PR #19). Question asked: does the AutoRound/GPTQ-INT4 route stay "best" after a
same-source GGUF matrix?

**Naming precision:** the baseline is the **AutoRound-contract GPTQ INT4 sym G128
(W4A16)** checkpoint — AutoRound is the *converter*, GPTQ-packed INT4 is the *format*.
**AWQ itself was not measured** (no artifacts; Decision 3 still requires exact
loader/kernel demonstration before any AWQ claim). Nothing tonight changes AWQ's status:
it remains unevaluated on this host with no route to displace the GPTQ-packed path.

**Re-confirmations (with tonight's evidence):**

1. **AutoRound/GPTQ-INT4 stays the C1 decode leader among measured serving routes**
   (Decision 4 unchanged). Single card, 150 W, compile-only, recommended sampling:
   24.41/23.89 tok/s spec-off at p512/p8192 g128; **45.72/37.11 with MTP-4** (+87%/+55%).
   Note: vLLM-vs-GGUF rows are `directional_cross_engine` — the edge is within-engine
   context plus speculation, not a format-only comparison.
2. **llama.cpp Q4_K_M stays the GGUF workhorse for this model** (Decision 2 unchanged,
   now with a same-source matrix): fresh-from-verified-BF16 decode 20.01 tok/s vs mixed
   18.85 / Q5_K_M 16.97 / Q8_0 13.62; long-prompt prefill byte-flat 844–871 across all
   formats (pp512 611–677 is launch-bound — do not mix regimes).
3. **Mixed-precision "rescue" of GDN/attn tensors is a negative on both measured axes**
   (+10.7 % bytes → −5.8 % decode, no measurable quality gain): the Q4_K_M preset's own
   tensor mixture already allocates precision well on hybrid-GDN. Do not redo blanket
   `--tensor-type` protection sweeps for this architecture without new evidence.
4. **Q8_0 has the best per-byte kernel efficiency** (~396 GB/s effective streaming vs
   ~336 for Q4_K_M) but loses on byte count at decode — a capacity/quality-lean option,
   not a speed option.

**New methodology rule (affects any quality gate using perplexity):** on hybrid
linear-attention models (qwen3_5-class GDN), `llama-perplexity` is **not** a valid
weight-precision probe — the full-F16 reference scored worse than Q4_K_M on 7/8 chunks
and PPL grows with chunk length (root cause: recurrent-state accumulation in the eval;
quant kernels exonerated). Upstream: [ggml-org/llama.cpp#28879](https://github.com/ggml-org/llama.cpp/issues/28879).
For format-quality questions on this model class, require per-position logit parity
against an F16 reference; never rank formats by PPL.

## Addendum 2026-09-14b — drafter economics on quantized targets (DFlash2 vs MTP4, closed)

Source: `research/2026/qwen38-dflash2-objective-campaign-20260830.md` §12 (2026-09-14 closure)
+ `results/2026/09/qw38-mtp-depth-20260914/` (RL-026/027/028/029). This records the
quantization × drafter interaction that the DFlash2 closure rests on.

**Decision: for C1 decode on a quantized dense target, the near-free single-layer
native MTP drafter beats every heavier learned drafter measured on this host.
MTP4 stays the spec; DFlash2 is closed at exactly half the throughput with all
known bugs fixed (25.16/25.30 vs 50.78/46.58 tok/s, 230 W, n=3, compiled target
+ eager draft + K-norm fix).**

Why, in quantization terms:

1. **The drafter was trained against full-precision target hidden states; the
   serving target is quantized.** A drafter's acceptance is conditioned on the
   hidden-state distribution of its training target. GPTQ-INT4 (and worse, FP8
   W8A16-decode) targets shift that distribution; measured acceptance lands at
   ~2.0 accepted tokens/step versus ~3.5 for the native MTP head (which is
   quantization-preserving BF16 by construction in our artifact — its tensors
   are excluded from quantization). This is the mechanism behind "drafter
   acceptance collapses on quantized targets" — now demonstrated with the
   kernel-bug confounder removed (selector NaN shown to be an eager-mode
   artifact; 115-window probe, RL-029).
2. **Weight-streaming asymmetry is quantization-independent.** MTP4's drafter is
   one layer sharing the target's embedding/head — near-zero marginal bytes per
   cycle. DFlash2's drafter streams a 3.85 GB BF16 5-layer model every cycle.
   At M=1 on a 608 GB/s card, cycle cost dominates acceptance gains: even at
   acceptance parity, DFlash2 would still need ~2× the accepted length to break
   even. It has ~0.6×.
3. **Where the inversion does NOT apply (fair scoping):** on H200-class hardware
   with BF16 targets (matching the drafter's training distribution) and batch
   serving that absorbs verify cost, DFlash2's published edge (4.8 vs 4.28 on
   Inco's numbers) is plausible and was never disputed. The closing is scoped
   to: single 32 GB card, quantized dense target, C1. Retry trigger stays as
   recorded: a drafter *trained on the quantized target's hidden states*
   (SpecForge-class capture on GPTQ/INT4 serving), not another port of the
   BF16-trained artifact.

Practical rule for future drafter candidates on this host: **a drafter must be
trained against (or proven robust to) the exact quantized representation it
will serve, and its parameter count must be justified by ≥1.5× the native
head's acceptance. Neither holds for any non-native drafter measured to date.**

## Alternatives considered

| Alternative | Disposition | Reason |
|---|---|---|
| One quant rule for all engines/models | Rejected | Hides architecture, kernel, capacity, and quality differences |
| Symmetric K/V KV precision | Not the default | Existing evidence supports asymmetric protection of K and stronger V compression |
| `q4_0` K cache | Prohibited | Existing recorded quality failure |
| Treat every INT4 checkpoint as the same XPU path | Rejected | Packing, symmetry, zero points, group size, architecture, and routed kernels differ |
| Approve conversion on file creation or load alone | Rejected | Does not establish parity, task quality, capacity, or intended dispatch |
| Choose quant solely by maximum observed throughput | Rejected | Violates representative-statistic and correctness/quality gates |

## Required decision record for a new candidate

Any change to these decisions must document:

- exact source and candidate artifact identities;
- transformation lineage;
- engine/kernel compatibility evidence;
- correctness scope and truth verdict;
- parity or task-quality method;
- capacity at the target context/concurrency;
- matched performance cells and comparison class;
- evidence tier and public reproducibility; and
- what current decision is superseded.

## Reconsideration triggers

Review this record when:

- upstream adds or changes an XPU quantized kernel or loader;
- a native artifact replaces a local conversion;
- a new model architecture changes quant sensitivity;
- target context, concurrency, modality, or task quality changes;
- new matched parity or task-quality evidence contradicts the current decision; or
- a previously local-only stack becomes reproducibly public.

Apply the [evidence lifecycle](../governance/evidence-lifecycle.md): preserve the old decision and evidence, add a dated correction or supersession, and update dependent recommendations.
