---
title: "Qwen3.8-27B: evidence-corrected C1 improvement plan after D11"
status: final
type: research
date: 2026-09-24
truth_verdict: PARTIAL
evidence_tier: E2
publication_gate: PROVISIONAL_NOT_FOR_PUBLIC_HEADLINE
scope: "Research decision for the retained Qwen3.8-27B GPTQ vLLM XPU route; no new performance measurement"
---

# Qwen3.8-27B: the plan after D11

**Decision:** retain the current single-card recipe. The next commitment is a **bounded attribution of one graph-captured MTP4 decode step on that exact recipe**, not a retrain, whole-model quantization conversion, speculative kernel rewrite, or TP2 server. Choose an implementation only if the measured critical path supports a credible **at least 5% net C1 improvement** with unchanged output semantics. A separate, much smaller bidirectional peer-capability test can decide whether direct-peer TP2 is even available on the current X570-F; it does not depend on the decode-attribution result.

**Executed 2026-09-24/25 — see [§9 verdict addendum](#9-post-night-verdict-addendum-2026-09-25).** The attribution ran, the capability probe PASSED, and the single-card lever hunt closed at the byte floor with zero promotions.

**No additional speedup was measured in this review.** The proposed QuaRot → 140 tok/s, IQ3 rollout → 1.5–2× Qwen27 prefill, and zero-copy TP2 → 160–190+ tok/s are **withdrawn as commitments**. They respectively assumed an unsupported low-bit runtime and incomplete byte budget, applied a different model/operator's result, and skipped the capability/collective/graph-safety gates.

## 1. Correction record and retained baseline

This revision supersedes the earlier four-frontier narrative in this same file. The original is preserved verbatim as [EXL3-PRE-REPLAN-20260924.txt](../../docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/EXL3-PRE-REPLAN-20260924.txt), SHA256 `77d378e57afcc600ef1b31395bcd6c906f1dc6a5092092c13db607738ca80788`.

The selected route remains:

- **Target:** `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`, GPTQ symmetric INT4 group 128; retain its mixed-precision exclusions.
- **Draft:** stock pretrained MTP weights, MTP4, greedy drafting, separate runtime **draft-only INT4 body and LM-head pair**. The verifier's shared/full-vocabulary head is **not** replaced by the draft quantization.
- **Runtime:** the pinned, proven vLLM XPU graph-enabled recipe and its exact patch generation; do not silently substitute a newer wheel or eager execution.
- **KV and sampling:** retain FP8 KV and the checkpoint's recommended non-thinking sampling for representative C1 tests. Greedy target sampling is a separately labeled diagnostic.
- **Placement:** one card for the target C1 route; independent stacks remain the dual-card default under [ADR-0008](../../decisions/adr-0008-dual-b70-independent-default.md).

Recipe and patch identity: [model overview](../../models/qwen/qwen3.8-27b/README.md), [serving dossier](../../models/qwen/qwen3.8-27b/serving.md), [draft-INT4 provenance](qwen38-draft-int4-patch-provenance-20260922.md). Historical Run 48 and later cache-off publication cells are different measurements; **102.61 tok/s is not a portable baseline** for a new experiment.

## 2. What the outcomes actually establish

The campaign had real wins. They do not all belong to Qwen27 C1, and already-adopted gains cannot be multiplied into a new forecast.

| Result | Scope and status for this plan |
|---|---|
| Draft-only INT4 pair | Retained Qwen27 C1 win. Fresh publication cells measured +39.6% and +48.6% client post-first rate, with actual-token differences disclosed; not another unclaimed future gain. |
| Graph capture | Already part of the retained stack and an enabler of the draft-pair gain. The older eager/compiled comparison is not incremental headroom today. |
| IQ3_XXS fused operator, 3.60× | Real isolated **K216 MoE / llama.cpp SYCL** result. No direct application to the **dense GPTQ / vLLM** Qwen27 route. K216 end-to-end follow-up was canceled when the model was retired. |
| Ubatch 512 → 1024 | Real K216 prefill result; not a Qwen27 vLLM prefill knob or gain. |
| Pure s4×s4 DPAS, 2× work/instruction | Hardware instruction experiment, not a 2× W4A16 serving result. Activations and dispatch matter; do not substitute it for achieved application TOPS. |
| Concurrency scaling | Real aggregate-serving result; not a single-user decode gain and not this task's objective. |

The draft-pair publication cells were C1, requested g128 with all five measured requests completing at length, recommended non-thinking sampling, prefix cache off with zero hit delta, and a 230 W configured cap. At the nominal p512 cell, actual prompts were 516/517 tokens; at nominal p8192, 8190/8194. Results were client post-first **medians of n=5 per arm**, not engine-native decode or sustained rates. Measured card draw was approximately 225–230 W. Tokens/J improved **+31.4% at the short cell and +9.6% at the long cell** over the **whole five-request window including prefill**, not decode-only energy. Publication on LocalMaxxing is **E3 accepted self-report, not independent reproduction**. Tested greedy byte identity and supportive sampled-output evidence remain scoped tests, not proof over all prompts.

Evidence, exact values and qualifiers: [OUTCOMES W1/W1a/W2/W11/W14](../../docs/plans/2026-09-20-b70-improvement-roadmap/OUTCOMES.md), [LEVERS](../../docs/plans/2026-09-20-b70-improvement-roadmap/LEVERS.md), and [independent claims review](../../docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/QWEN27-REPLAN-CLAIMS-20260924.md).

### D11 stays closed, but its mechanism must be described precisely

The actual stitched-result directory is d11-stitched-fastmtp1500-probe-20260924T172339Z (`../../results/2026/09/b70-improvement-probes-20260920/d11-stitched-fastmtp1500-probe-20260924T172339Z/`), **not** the nonexistent `T182834Z` previously cited. The same-day comparator is d11-baseline-decay-probe-20260924T142540Z (`../../results/2026/09/b70-improvement-probes-20260920/d11-baseline-decay-probe-20260924T142540Z/`). The fresh D12 finalist's greedy control, not historical Run 48, supplied the older 77.26 tok/s / alpha 0.5537 comparator.

The recorded same-day comparison is stock alpha **0.5421** versus retrained **0.5023**, with recorded client post-first medians **73.68 versus 71.83 tok/s** (about **−2.5%**). The approximately −7% comparison is to the older D12 control and is **historical/directional**, not a second same-day matched result. Neither comparison promotes the retrained head. See the audit for raw-counter, cell, thermal and matching limitations; the failed adoption decision does not require presenting this as a publication-grade A/B.

Per-position accepted-token counters divided by all draft rounds are **marginal survival rates**: position 1 means that the accepted prefix reached the second drafted token. They are not independently measured conditional token accuracies. Given a common counter window and fixed four-token drafting:

```text
E[accepted drafted tokens] = sum_i P(accepted prefix length >= i+1)
E[emitted tokens per round] ≈ 1 + accepted_tokens / draft_rounds
                           = 1 + 4 * alpha
```

The extra one is the verifier/bonus emission in the idealized round model; request boundaries and actual scheduling must be accounted for when reconciling a real run. Sequential acceptance does not mean the verifier computes only one row at a time: verification can process multiple rows together and then retain the accepted prefix.

The observed retrained-head profile loses early-prefix survival and increases the tail; **the net yield decreases in this experiment**. It does not prove that tail improvements can never compensate shallow losses under any training method. Nor did this review establish a primary source for the specific claim that this native MTP head itself was trained on trillions of tokens alongside the base model. That explanation is not needed for the operational decision:

**D11 is CLOSED / REJECTED. Do not repeat small-corpus consumer retraining or deploy this head. Preserve stock MTP weights.** The policy remains a no-retry rule for this program, not a universal theorem that all post-training of all draft models must fail. No reopening is proposed here.

## 3. Correct the roofline before using it to choose work

The earlier note labeled `21.44 GB / 39.8 ms = 538.7 GB/s` a measured bandwidth and inferred 96.2% utilization of a 560 GB/s practical ceiling. **That was a traffic/timing model, not measured memory-controller traffic on the retained stack.** Artifact size, bytes accessed, cache reuse, repeated reads, and critical-path duration are different quantities. The fixed 560 GB/s application ceiling was also not established by the cited material.

The existing P4 profiler capture is **pre-draft-pair, eager, and from a different image generation**. Its GEMM fraction cannot establish the remaining bottleneck under the retained graph-enabled recipe. Conversely, a whole-model megakernel does not become justified merely because that older trace is unsuitable: it still needs a measured removable fraction.

There is also a yield/timing mismatch in the former forecast. Using the stock diagnostic's 438 accepted tokens over 202 rounds gives an illustrative yield of `1 + 438/202 = 3.1683`. At that yield:

- 39.8 ms per round would mean approximately **79.6 tok/s**, not 102 or 140.
- 25 ms would mean approximately **126.7 tok/s**.
- 140 tok/s would require approximately **22.6 ms** per round.

These are **E0 arithmetic illustrations**, not throughput predictions. Dividing an aggregate yield by a median client rate does not produce a measured GPU step time. Do not combine yield from one run, traffic from another artifact, and a historical best client rate.

The @0xSero EXL3 screenshot's original public artifact was not located in targeted searches. Its claimed isolated LM-head bandwidth cannot establish full-model speed or a matched comparison. Treat the supplied screenshot account as **E1 external self-report without located raw evidence/provenance**, not a validated counterexample or an adopted runtime. Details: [upstream source review](../../docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/QWEN27-REPLAN-UPSTREAM-20260924.md).

## 4. Research verdict on the three proposed frontiers

### 4.1 Rotated GPTQ / QuaRot: compatibility research, not the first conversion

[QuaRot](https://arxiv.org/abs/2404.00456) and [SpinQuant](https://arxiv.org/abs/2405.16406) are legitimate low-bit quantization methods. They do **not** establish that this target can become an unchanged, fast INT3 artifact through a simple offline conversion.

1. **The target body is already INT4.** Rotation at unchanged bit width offers potential quantization-quality improvement, not automatic weight-traffic reduction. Activation quantization is a different quality/compute hypothesis and is not included in this plan.
2. **Distinguish the rotation scheme.** A restricted weight-only residual rotation can fold offline. The full QuaRot activation/KV-quantized scheme also includes online rotations, including transforms around nonlinearities/attention; those are not free and are not automatically required by a weight-only variant. “One-cycle unpack” omits packing, scale loads, conversion, memory layout, and the matmul consuming the weights.
3. **This is not an ordinary dense-attention-only model.** The locally inventoried model has dense FFNs with hybrid GatedDeltaNet/full-attention blocks, hidden width 5120 and intermediate width 17408. The transform must respect recurrent state, gates, norms, RoPE, residual connections, the final verifier head and MTP interfaces. QuaRot's composite Hadamard implementation supports 5120 as 20×256; the inspected implementation does not support 17408 as a full transform, though block-diagonal alternatives exist with weaker mixing. Residual rotation can in principle fold offline into input/output projections while leaving internal GDN channels fixed; GDN does not categorically forbid it. Full-model numerical integration for this exact serving graph remains unvalidated.
4. **INT3 is not supported by the inspected XPU serving path.** The GPTQ loader and XPU WNA16 dispatch/kernel inspected by the quantization specialist do not provide an INT3 packed-weight route. An INT4 kernel cannot consume three-bit packing just because both use bit operations. New packing, loader support, kernel work, and correctness gates would be required; ordinary W4A16 import does not supply them.
5. **The old forecast dropped residual traffic.** Even using its own unverified model, reducing only a 15.20 GB INT4 body by 25% changes total modeled step traffic from 21.44 to **17.64 GB**: an ideal **1.215×** bandwidth-only ratio, before rotation, metadata, padding, unchanged compute and acceptance effects. Compressing *every* byte 4→3 bits would have an ideal 4/3 ratio, but the unquantized verifier head and other exclusions forbid that assumption here. Neither number is a measured gain. **Note 2026-09-29** (register row 5): these ratios describe the *GPTQ* variant. The adopted **EXL3** artifact is mixed precision (safetensors: body 4.0 bpw / head 6 / MTP 4), so its body-only 4→3 ratio is **1.224×** (body = 12.230 GB of the ~16.68 GB round) — keep both figures labeled by representation; any sub-4-bit proposal must name which axes move.

A bounded mathematical option exists: rotate only the **MLP down-projection input** after the nonlinear gate, fold the inverse into its weights, and keep GDN internal state unchanged. That inserts an **online** intermediate-space transform. It can test numerical equivalence before quantization; it neither proves full-model quality nor removes the INT3 runtime blocker. Quantization would start from a preserved high-precision source, not repeatedly quantize the existing GPTQ values.

**Disposition:** park full-model rotation/INT3 conversion. A cheaper first rotation study would test an offline-folded residual rotation at unchanged W4 bytes; the MLP-intermediate online option is a separate hypothesis. Require a one-layer algebra/packing/quality/traffic gate first, and only pursue a speed program if current-stack attribution makes the attainable byte savings worth a new runtime implementation. [Quantization research and source pins](../../docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/QWEN27-REPLAN-QUANT-20260924.md).

### 4.2 P-3b IQ3_XXS rollout: not applicable to this Qwen27 route

The measured 3.60× improvement is the routed **`MUL_MAT_ID` IQ3_XXS** operator, extending the grouped MoE structure of [llama.cpp #29245](https://github.com/ggml-org/llama.cpp/pull/29245). This task's target uses **dense GPTQ projections in vLLM**, not GGUF IQ3_XXS routed experts. There is no direct dispatch to “roll out.”

The reusable idea is **tile-local reconstruction and reuse without whole-weight FP16 materialization**. Applying that idea to Qwen27 would be a distinct dense W4A16 kernel proposal, justified only by tracing the actual verify shapes and finding materialization, rereads, padding, spills, or other avoidable work. Dense FFN does not imply every decode step is M=1: MTP verification can use up to k+1 rows, with boundaries/padding determined by the actual runtime.

The isolated n=1 change from 192.2 to 166.6 microseconds means **13.3% lower operator latency / 1.154× operator throughput**, not a measured +13% full-model decode gain. K216 end-to-end work remains canceled. **Remove this rollout from the Qwen27 prefill plan.**

### 4.3 Direct-peer TP2: test capability, do not invent the transport

The Level Zero [memory specification](https://github.com/oneapi-src/level-zero-spec/blob/143853c129b66d337d3823db7f1ec1abd61c6940/scripts/core/memory.yml) states that importing a device allocation on a **different** device requires peer access to the exporting device. IPC shares an allocation across processes; it does not create GPU peer routing. Same-device IPC does not require a second GPU. Pinned host memory still travels over PCIe and is not device-to-device zero-copy.

Two cards' **local** GDDR bandwidth can indeed sum to more than 1 TB/s under appropriately partitioned work. That is not inter-card bandwidth, nor proof that the single request can exploit the sum after activation collectives, synchronization, draft execution and graph constraints. The proposed 160–190+ tok/s band is unsupported on this host.

There is nevertheless a real uncertainty worth testing. The [local X570-F note](x570f-dual-b70-pcie-topology-20260830.md) records both CPU-attached x8 links and H2D measurements, but the reviewed evidence does **not** include a current-host bidirectional peer-access query plus checksum test. [Intel compute-runtime #935](https://github.com/intel/compute-runtime/issues/935) includes clean peer-copy reports on other AMD hosts, whereas the documented Intel-host failure has a different root-complex/kernel-policy condition. This machine's Ryzen family and enabled P2PDMA configuration do not alone prove Level Zero support, but they invalidate a categorical inference from the other motherboard's failure.

**Disposition:** keep ADR-0008 and the existing host-staged-only serving assumption until a controlled capability test passes. Do not rewrite the operational contract or enable peer collectives based on this literature review. A clean copy still does not prove required atomics, oneCCL correctness, capture safety, latency, or C1 scaling. Relevant open graph/collective issues include [vLLM #58388](https://github.com/vllm-project/vllm/issues/58388) and [#48946](https://github.com/vllm-project/vllm/issues/48946); applicability must be checked against the exact stack, not assumed from another GPU.

## 5. The next-lever map (decode and prefill attacks)

Every figure here is an **E0 model bound** under the fixed-traffic/fixed-bandwidth/no-other-change
assumptions of §3 — a pricing ceiling for decision-making, never a prediction or a promise. The
attribution probe replaces the model with measured shares before any lever is engineered.

### 5.1 Decode (single stream), in dependency order

| # | Lever | Status | Basis and bound |
|---|---|---|---|
| L-A | Retained recipe: MTP4 + greedy drafting + FP8 KV + XPU graphs + draft-only INT4 pair | **Won — already in the stack** | Measured +39.6/+48.6 % cells; no further gain exists on these axes without a changed mechanism. |
| L-B | **Weight-stationary draft fusion** (read draft weights once per step, unroll positions in-kernel) | Attribution-priced | The MTP head is position-shared: the 4 unrolled draft passes re-read the same ≈0.848 GB of quantized draft weights (≈0.636 GB head + 0.212 GB body) each pass — ≈3.39 GB traffic where ≈0.85 GB is unique, ≈2.5 GB redundant ≈ **12 % of the 21.44 GB modeled step**, ideal bound **≈1.13×** if (a) the attribution shows these are GDDR re-streams and not cache-served (draft weights are orders of magnitude above L2), (b) they sit on the critical path, and (c) a fused step is implementable with native attention/state/sampler. This is **not** the RL-021 quarantined constant-token sketch; a real native implementation is the changed mechanism that would justify reopening that quarantine. |
| L-C | **Exact faster verify-head kernel** (same FP16/BF16 math, better schedule) | Attribution-priced, output-preserving by construction | D8 closed head *quantization* (INT4 72.07 %, FP8 93.3–93.4 % top-1), not faster *exact* implementations. The head is ≈2.54 GB ≈ 12 % of the modeled step; a 2× head kernel saves ≈6 % — clears the 5 % net gate **only if** the attribution shows the current kernel is ≥2× off its weight-read floor. Unquantized, full-vocabulary, distribution unchanged. |
| L-D | **#56096 fused QK-norm+RoPE+gate dispatch** (merged upstream 2026-09-16, post-pin) | Static-first | One-line `qwen3_next.py` XPU enablement. Author region numbers on a 9B at bs=8 (−55 % decode region, −23 % prefill region) are external and scale with M; at C1 M=1–5 on a 27B the decode effect is plausibly <5 %, the TTFT/prefill effect more plausible. Zero-GPU dispatch trace first (`supports_dtype` + Triton import under pinned torch); only then a bounded n=3 screen. Not a rerun of the thermally invalid #53989 screen. |
| L-E | Verify-head quantization, probabilistic drafting, KV-precision, depth k≠4, DFlash2, sampler wheel 0.1.15.4, small-corpus retraining | **Closed** | All measured failures or gates; unchanged retries banned (§7). |

### 5.2 Prefill / cold input

| # | Lever | Status | Basis and bound |
|---|---|---|---|
| P-A | #56096 prefill region (−23 % author, 9B, external) | Static-first, same trace as L-D | TTFT-side candidate; effect at 27B C1 unknown until dispatch is traced. |
| P-B | **vLLM prefill scheduler knob** (`max-num-batched-tokens` / chunked-prefill size) on Qwen27 | Verify-then-screen | The measured ubatch-1024 prefill win was llama.cpp/K216; no sweep of this vLLM knob for this model is recorded in the ledger. Verify it is genuinely unswept, then one cheap bounded screen if cold-input/TTFT is the objective. Do not assume the K216 result transfers. |
| P-C | **Rotation-enabled W4A8** (parked, quality-gated) | Parked — the one real prefill *compute* hypothesis | Rate-class law: W4A16 verify GEMMs run in the BF16 DPAS class (≈170 TOPS); INT8 activations enter the INT8 class (≈363 TOPS) — the prefill compute-rate doubling. The recorded W4A8 quality divergence (Q4) was measured **without** rotation; activation-side rotation is precisely the mechanism targeting the outliers that sank it, so the prior failure does not cover this variant. Requires the full one-layer → whole-model quality ladder and a supported loader/kernel; no speed claim exists until quality passes. |
| P-D | IQ3_XXS MoE operator rollout; K216 ubatch | **Not applicable** | Different engine/operator/quant; no dense-GPTQ dispatch. |

### 5.3 Yet to resolve — the open questions that decide the map

1. **Q1 — step composition** (the 60-minute attribution): measured share of target GEMMs, draft passes, verify head, GDN/KV, graph/host residual. Decides whether L-B/L-C are worth engineering at all; honest outcomes include "close kernel hunting."
2. **Q2 — local peer capability** (the ≤5-minute probe): whether this exact X570-F/driver pair supports correct bidirectional peer copies. Decides any TP2 future; does not affect L-A…L-D.
3. **Q3 — #56096 dispatch conditions** under the pinned torch/kernels tree (zero-GPU static): gates L-D and P-A.
4. **Q4 — are draft-pass weight reads cache-served or GDDR-streamed?** Part of Q1; decides L-B's premise.
5. **Q5 — is the verify-head kernel at its weight-read floor?** Part of Q1; decides L-C's premise.
6. **Q6 — MTP draft-drop function at ≥98 K context** (O5, unchanged): capacity interaction, not speed.
7. **Q7 — concurrency-knee × draft-INT4 interaction at high B** (S3): serving-throughput only; deprioritized by the operator.

**Execution order:** Q3 (static, free) → Q1 (60 min, one window) → price L-B/L-C from measured shares → engineering only on a ≥5 % net gate. Q2 runs whenever an owned both-GPU window exists. P-B is a tiny screen only if cold-input latency becomes the objective.

## 6. Revised priority order and stop gates

These are **research decisions**, not commands executed by this review. No new benchmark, model load, conversion, training, kernel build or serving change occurred.

| Priority | Bounded question | Decision it can earn |
|---|---|---|
| **First** | Where is the retained graph-captured MTP4 step's removable time? | At most one exact dense-kernel/runtime candidate with credible ≥5% net C1 benefit. |
| **Second, independent** | Does this exact X570-F/driver support correct peer copies both ways? | Capability-only result; a separate collective/graph correctness proposal if clean. |
| **Parked** | Can a restricted rotation preserve the graph and enable useful, supported lower-bit packing? | One-layer research only; no whole-model conversion or speed target. |
| **Not applicable / closed** | IQ3 MoE rollout for dense GPTQ; D11; unchanged failed flags, heads and wheel bumps | No new run or implementation. |

### First: exact-stack decode attribution, hard budget 60 minutes

**Hypothesis:** a current, named implementation inefficiency—not nominal INT4 bit width or an old eager trace—accounts for enough of the retained MTP4 step to justify one change.

**Before GPU work:** reuse any already-valid exact-stack evidence. Freeze image digest, vLLM/kernel versions, all patch hashes, model/tensor exclusions, graph settings, sampler, MTP depth, scheduler and KV. Check actual dense GPTQ dispatch and trace-tool compatibility statically. **Step 0, zero-GPU falsifier:** identify on the pinned image whether the dense W4A16 small-M dispatch can even be named by the available tracing path; if it cannot, fix that first or the window buys nothing. The recent [vLLM #56096](https://github.com/vllm-project/vllm/pull/56096) XPU QK-normalization/RoPE/gate enablement is a **dispatch question inside this attribution**, not an automatic rerun of the rejected #53989 screen. A merged upstream equivalent is not a changed performance mechanism by itself.

**Owned resource scope:** one B70 chosen and recorded by PCI BDF; other card idle; one profiler/benchmark owner; empty-GPU and post-load reserve gates; fixed declared cap and dynamically verified telemetry; protocol cooldown ≤55°C; exact cleanup of owned processes and restoration of the declared permitted state. Do not change desktop services, caps or runtime policy merely to get a run to start. The 60-minute deadline includes startup, warmup, trace and cleanup; there is no compilation/image-pull extension.

**Minimum observation:** one exact p512/g128 C1 natural-text diagnostic cell on the retained recommended-sampling/cache recipe, same-shape warmup discarded, a brief uninstrumented control before and after an instrumented capture. Control windows must match the captured cell's shape and prompt family per step. Bracket prefix-cache and speculative counters; preserve actual tokens, sampling seeds and prompts. A diagnostic is not a fresh representative benchmark. Graph replay must remain enabled and visible in the trace; eager fallback cannot answer this question.

**Attribute critical-path contributions**, not overlapping kernel-duration sums. The budget must include an explicit **graph-replay residual plus inter-kernel host time** bucket — a capture-era lever would live exactly there and vanish in a kernel-only table:

- target dense GPTQ verification GEMMs at the actual M shapes, including padding/layout — the trace must **name** the W4A16 small-M dispatch for at least one verify layer; a generic `gemm` label repeats the old trace's attribution trap;
- four sequential draft passes and their separate quantized head/body tensors;
- the unquantized **full-vocabulary target head** and verify/bonus work — report its exclusive share separately even if it clears no engineering gate;
- GDN/recurrent-state, attention/KV, norms, sampler and the graph/host residual bucket above.

Preserve weight/activation precision and full-vocabulary normalization. “Keep the full verifier head” does not prohibit a faster, numerically validated unquantized implementation; it prohibits quietly changing the verifier distribution through the already-rejected quantization or vocabulary slicing.

**Fail closed** if replay is not visible, instrumentation changes dispatch, controls/counters disagree materially, thermal/cache/isolation gates fail, or the budget expires. Record "attribution unresolved," not a fabricated utilization conclusion — and **unresolved is not "GEMM-bound, build a kernel."** A fully legitimate outcome is that no removable ≥5% inefficiency exists and kernel hunting closes. Do not use instrumented latency as a public speed result.

For a component with exclusive baseline fraction `f`, credible local speedup `s`, and normalized added cost `o`:

```text
S_net = 1 / ((1 - f) + f / s + o)
```

To justify a 5% end-to-end target requires removing at least **4.76% of step time net**. A credible 2× optimization therefore needs more than **9.52% of the step** before new overhead. A 2× kernel occupying 15% would offer only **1.081×** overall, even ideally. This threshold is a predeclared investment gate, not a prediction that such a kernel exists. If nothing passes, keep the current stack and close kernel hunting rather than force a project.

After a candidate is identified, its independent architect/correctness review and isolated-op gate precede a separately scoped screen: matched p512/g128 and p8192/g128 n=3, then fresh n=5 confirmation and a real workload. Require at least 5% representative C1 improvement at the declared objective without material regression at the other anchor, correct outputs, cache/thermal validity, measured energy and no excess VRAM cost. Use paired prompt families and enough windows to resolve the effect; n=5 is a minimum, not statistical proof. Never promote an acceptance-only change or a profiler timing.

### Second: direct-peer capability only, hard budget five minutes after preflight

In a separately owned empty-both-GPU window, retain both BDFs, driver/runtime identity and link state. Query `zeDeviceCanAccessPeer` **both directions** and `zeDeviceGetP2PProperties` access/atomic flags. If access is absent, stop without forcing a copy. If access is present, perform bounded nontrivial-pattern **bidirectional** device copies (at most 256 MiB per buffer), checking every byte against the expected pattern and retaining a host-staged control. API success alone is insufficient.

**Stop:** false access, API error, reset, timeout, checksum mismatch or contention. No debug force-return flags, kernel whitelist, ACS/IOMMU modification, PCI rescan or power cycle is part of this test. **Pass:** capability only. No automatic `CCL_TOPO_P2P_ACCESS=1`, stock TP2 server, or production switch follows. Collective correctness, required atomic behavior, graph capture and request latency need separate gates. The earlier 1–2 minute estimate is not a license to omit preflight/cleanup; if no validated probe binary is available, record the tooling block rather than begin an unbounded build.

### Parked rotation branch: static gate first

Before allocating full-model conversion time, require an exact transform/packing specification for one real FFN matrix, numerical equivalence against preserved high-precision weights, tensor coverage and exclusion accounting, and a demonstrably executable XPU consumer. Any INT3 route must count scale/zero metadata, cross-word packing/padding and online transform costs. A kernel storing the result padded back to INT4 has not delivered a 25% read reduction. A weight-only quality improvement at identical bytes is not a C1 performance win by itself.

No model conversion is selected until the first attribution and these gates justify its cost. No re-quantization of the lossy champion is a substitute for a high-precision source.

## 7. Closed hypotheses remain closed

- **D11:** no small-corpus retraining, no FastMTP-1500 adoption.
- **D12/D14:** no probabilistic-draft or KV-precision retry on the same premise; the D14 flag changed the KV policy globally and cannot establish a draft-only precision theorem.
- **D8:** no direct verifier-head INT4/FP8 or candidate-vocabulary shortcut. Synthetic top-1 divergence screens rejected these candidates; their thresholds are not a complete task-quality definition.
- **Static depth sweeps / k=3 clamp:** retain MTP4. The tested canary did not reproduce the feared corruption; that is a scoped canary result, not proof of safety for all shapes.
- **DFlash2:** no reopening merely because a previously ported upstream fix merged.
- **Broad kernel-wheel bump:** the tested 0.1.15.4 regression does not become a win by relabeling it “new compiler optimizations.” Revisit only a separately anchored mechanism with correctness and performance evidence.
- **#53989/QK-RoPE:** no win claimed from the thermally invalid screen; no duplicate upstream contribution. Check #56096 dispatch within a justified attribution.
- **Concurrency and other models:** no B32 result or MoE kernel multiplier is presented as Qwen27 C1 improvement.

## 8. Sources, reviews and evidence boundaries

Local operating authority: [benchmark protocol §1.9](../../docs/methods/benchmark-protocol.md), [model status](../../registry/model-status.json), [research ledger](../../registry/research-ledger.json). Raw results are unchanged; this document corrects interpretation and the next decision, not the experiment history.

Specialist records:

- [Lead architecture and bounded probe](../../docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/QWEN27-REPLAN-ARCHITECT-20260924.md).
- [Independent adversarial architecture review](../../docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/QWEN27-REPLAN-REVIEW-20260924.md).
- [Quantization papers, packing and model-interface audit](../../docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/QWEN27-REPLAN-QUANT-20260924.md).
- [Primary upstream/version/topology research](../../docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/QWEN27-REPLAN-UPSTREAM-20260924.md).
- [Independent claims and D11 evidence audit](../../docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/QWEN27-REPLAN-CLAIMS-20260924.md).

Primary papers/code explain mechanisms, not performance on this exact model/driver. External B70 and AMD-host results motivate bounded checks but are not independent reproductions of our cells. Research completed here is CPU/file/web work only: **no new GPU result, no retraining, no kernel build, no service/configuration change, no commit, no publication.**

## 9. Post-night verdict addendum (2026-09-25)

Executed by the 2026-09-25 improvement night (full record: [NIGHT-REPORT-20260925](../../docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/NIGHT-REPORT-20260925.md); ledger RL-061…RL-065; campaign `CAM-2026-QWEN27-IMPROVEMENT-NIGHT-0925`). **Zero promotions; champion recipe unchanged** — and the plan's open questions are now measured verdicts:

| Item (§5/§6 ref) | Verdict | Evidence |
|---|---|---|
| Q1 step composition | **Answered: single-card decode is at its byte budget floor.** 44.8 ms round = host ≈4.8 ms (screen-proven mostly overlapped/irreducible) + draft ≈5.5 ms + head ≈4.2–4.7 ms + body ≈28–30 ms wall-clock (in-graph composition still untraced) + misc ≈2 ms | `p2-qwen27-eager-composition-20260925T085704Z` |
| L-B draft fusion | **CLOSED — physically inapplicable.** The draft bucket sits at the unavoidable 4×-read bytes floor: re-reads DO stream at pin rate (3.39 GB in 5.5 ms ≈ 5.57 ms floor) and cannot be cached on-chip across sequentially dependent positions | RL-062 + NIGHT-REPORT correction block |
| L-C exact head kernel | **CLOSED at floor** (4.2–4.7 ms ≈ bytes floor; s ≥ 1.9 implausible) | RL-062 |
| L-D / #56096 | **NOT APPLICABLE** — gate exists only in `qwen3_next.py` + `is_cuda()`; this route is `qwen3_5.py`. Corrects RL-054's "equivalent merged" note | RL-063 |
| Host-prep metadata cache | **LOSS** — p512 +1.51 % vs +4 % gate (p8192 +3.27 %); output-preserving, iso-power | RL-061 |
| P-B prefill knob | **Non-lever** — TTFT ±0.25 % across 4096/8192/16384 | RL-064 |
| Q2 peer capability | **PASS** — bidirectional ACCESS-flag copies ~11.1–11.9 GB/s checksum-clean, 2.02× host-staged; **ATOMICS absent** ⇒ copy-based synchronization only | RL-059 |
| P-A/P-C rotation, Q5–Q7 | Unchanged (parked/closed as in §5.3; P-C's prefill-compute rationale is independent of the decode floor) | — |

**Next to verify, in order:**

1. **O14 — copy-based collective correctness** on the verified P2P path (oneCCL/L0, no server, checksummed) → if clean, a gated TP2 research campaign against a matched one-card control behind ADR-0008. Honest modeled band: **~1.4–1.6× decode** (collectives are latency-bound at ~128 small round-trips/round), not the retired 160–190 figure.
2. **O13 — unitrace capture of one graph-captured round**: record-closing for the 63–67 % body bucket; expected slack ≤2–4 ms. A formality, not a lever.
3. **O15 — thermal window (operator)**: the ≤55 °C protocol gate is unreachable while a server is loaded (RL-065); chassis/ambient intervention or a formally accepted substitution rule is required before any promotion-grade measurement.

Standing caveat: the night's numbers used matched-arm thermal substitution (±1 °C, counterbalanced) — diagnostic-grade, never public.

## 10. Terminal state (2026-09-26, after the confirm/deny program)

The follow-on program (operator-authorized; ledger RL-070…072; [NIGHT-REPORT2](../../docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/NIGHT-REPORT2-20260926.md)) closed every open item in §9:

| §9 item | Terminal verdict |
|---|---|
| O13 body bucket | **CLOSED** — L0-visible: 87.7 % GEMM, ~5 ms rest, no hidden waste; single-card decode record closed at three independent levels (RL-072) |
| O14 collectives | **CONFIRMED exact** on both transports; 14.3 ms worst-case round → **gated TP2 campaign is the ONE open lever**, pre-declared +20…50 % (RL-071) |
| O15 thermal | **RESOLVED** — root cause (Xwayland card-node fd) fixed by PCI cycle, executed and verified (RL-069/072) |
| flash-next | **MEASURED** (first time): 21.6 tok/s median as placed — capacity route, not speed (RL-070) |
| T3 rotated-W4 | Spec'd, **storage-gated** on the ~55 GB BF16 re-fetch (operator decision) |

**Next steps, in order:** (1) the gated TP2 campaign (F2's follow-on — matched control, graphs-off,
backend xccl, abort gates per ADR-0008); (2) T3 if the storage decision lands; (3) flash-next speed
work only as a separately priced placement program.
