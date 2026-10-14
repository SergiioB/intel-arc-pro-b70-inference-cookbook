---
title: "Upstream contribution — optional low-precision copies of a speculative draft's weights (2026-09-21)"
type: research
date: 2026-09-21
status: provisional
---

# Upstream contribution: draft-side low precision for speculative decoding

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** This is a contribution draft, not a published result.
It exists so the finding can leave this host in a reviewable form instead of living as a runtime
monkey-patch in a local cookbook.

## The claim, with its evidence

Quantizing **only the draft's own weights** — its LM head and its MTP body — to GPTQ-style INT4
g128 symmetric, while the target model stays untouched, raised decode throughput on a
Qwen3.8-27B GPTQ-INT4 G128 + BF16 MTP4 route (vLLM XPU, single card, XPU graph capture on,
recommended non-thinking sampling) by **+37 % to +46 %** across independent runs:

| cell | baseline | draft-side INT4 | Δ | n |
|---|---:|---:|---:|---:|
| C1 p8192/g128, calibrated (8,189 of 8,192 tokens) | 55.61 tok/s post-first | **80.32** | **+44.4 %** | 5 fresh |
| C1 p8192/g128 (first measurement) | 52.99 | **72.68** | **+37.2 %** | 5 fresh |
| C1 p512/g128 | 52.83 | **77.14** | **+46.0 %** | 3 |
| C1 p8192/g128, both arms patched (repeatability control) | 80.61 / 78.27 | — | — | 5+5 |

Acceptance rate (a property of drafting, not of the emitted output) moved between runs in the
patched configuration (0.55–0.69) while the baseline stayed in a narrow 0.53–0.58 band; throughput
did not. TTFT was unchanged at long context and slightly *lower* in the patched arm at p512.

Evidence on this host: `results/2026/09/b70-improvement-probes-20260920/p4-champion-ab-*` with the
narrative in `docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/P4-20260920.md` §11–§17.

## Why it works (the mechanism, in bytes)

A draft is executed once per draft token plus once for the target's verify pass. With MTP4 the
shared LM head is therefore applied **five times per speculative step**. On this artifact the head is
**2.54 GB in F16** (248,320 × 5,120), so a step re-reads ≈12.7 GB of head alone — ≈21 ms at the
card's DRAM class, a large fraction of a ≈34 ms step. Quantizing the draft's copy to INT4 g128
(≈0.66 GB including scales) removes ≈7.6 GB per step. The target keeps its F16 head, so nothing about
verification changes.

> **CORRECTED 2026-09-29** (register row 4): the 12.7 GB / 7.6 GB figures are payload-only model
> arithmetic. The scale-inclusive budget from P4 tensor shapes + conversion logs is **0.87453696 GB
> per draft pass (head 0.6555648 + body 0.21897216, incl. fp16 g128 scales) = 3.49814784 GB per
> 4-pass round** for the quantized pair. Re-derive any per-step traffic claim from these figures.

This is not a context-length effect: on the same cell with XPU graph capture **disabled**, the same
two patches measured ≤1.5 % (host overhead was hiding the traffic). The enabling condition is that
the step be device-bound, which is what capture makes it.

## Why it is safe (and where "safe" stops)

Speculative decoding verifies every drafted token with the target, so the draft's precision affects
what is *proposed*, never what is *accepted*. Verified on this host:

- **Greedy: byte-identical output** between patched and unpatched arms across three prompts × two
  repeats, deterministic within each arm (same sha256).
- **Seeded stochastic sampling: identical** across arms — but on prompts so constrained that the
  sampled hashes equalled the greedy ones, so this is supportive, not a distribution-level result.
  A distribution-level check on open-ended prompts is the honest next step before anyone claims
  "distribution preserving".

Not covered: **TP > 1**. With a tensor-parallel target, verification no longer protects the emitted
sequence on every path, so the implementation fails closed (skips quantization) whenever
`tensor_parallel_size` is unknown or > 1.

## The upstream shape, and what was tried locally

vLLM has no supported way to serve a draft at lower precision than its checkpoint: the draft is
loaded exactly as the checkpoint stores it, and there is no knob to quantize the draft's own head or
body. The change below is deliberately small.

**Tree-level attempt made on this host** (`~/vllm-src/vllm`, base `066ae99e`, uncommitted,
`py_compile` clean):

1. New module `vllm/model_executor/model_loader/draft_int4.py` (~260 lines, English, no fleet
   specifics in the API): a chunked symmetric INT4 g128 quantizer producing the W4A16 NT layout the
   platform op consumes, an availability check for that op, a fail-closed TP gate, a
   `build_draft_int4` for the draft's LM head, a `build_draft_body_int4` that converts the draft's own
   projection matrices in place (replacing each weight with its INT4 copies and its `quant_method`
   with a small W4A16 shim), and `draft_logits_int4` / `draft_logits` helpers.
2. Thirty lines in `vllm/v1/spec_decode/llm_base_proposer.py`: the two build calls at the end of
   `load_model()` right after `_maybe_share_lm_head()` (behind `VLLM_DRAFT_INT4=1`, wrapped so an
   optional optimization can never break loading), plus a `_draft_logits()` method that every
   draft-side logits computation now goes through — all three call sites (`_greedy_sample` ×2,
   `_sample_draft_tokens` ×1) were routed to it, so the quantized head is actually exercised rather
   than merely built.

An independent architecture review caught the first version of this patch storing the quantized head
without any caller and omitting the body entirely; the shape above is the corrected one. That review
also stands as the warning: **`py_compile` clean is not validation.** What is still owed on the tree
patch is real wiring validation — target/shared-weight non-mutation, zero-scale groups, unsupported
configurations, graph-on execution, and a reproduction of the measured speedup from this tree rather
than from the runtime patches.

Revert: `git -C ~/vllm-src/vllm checkout vllm/v1/spec_decode/llm_base_proposer.py &&
rm vllm/model_executor/model_loader/draft_int4.py`.

**What "try" did not include:** building a wheel from this tree and re-measuring. The behaviour above
was measured with the fleet's runtime patches (which monkey-patch the same two objects at load time:
the draft's `compute_logits` and the MTP linears' `quant_method`), so the *mechanism* and the *effect*
are verified; the tree change is the packaging of it, and it needs a wheel build plus one A/B to
confirm equivalence. That is the next step, not a claim.

## The ask

Either shape is fine, and the second is preferred:

1. **An issue** with this evidence and the reference implementation attached, plus a maintainers'
   pointer on where the config surface should live (a `draft_quantization` field on
   `SpeculativeConfig`, or a `--draft-*` CLI flag), so the option is discoverable rather than an
   env var.
2. **A PR** implementing (1): the module above, the config field, the hook in `load_model()`, and
   the TP fail-closed gate. The per-backend requirement is one op — on XPU
   `torch.ops._xpu_C.int4_gemm_w4a16`, already used by the quantized body — and a backend without an
   equivalent op simply skips, which is what the availability check is for.

## Open items before this is more than provisional

1. Distribution-level equivalence under stochastic sampling (open-ended prompts, ≥30 seeds per arm,
   emitted-distribution comparison).
2. Energy and temperature provenance for the A/B cells (the fleet's own power semantics ask for raw
   energy, cadence, and coverage; the current runs record the configured cap only).
3. The patch pair's generation provenance: one of the two runtime patches names a vLLM revision that
   is not the one measured here, which must be reconciled before a PR. The tree patch above is written
   against `066ae99e` and must be re-anchored to whatever generation the PR targets.
4. Why the patched configuration's accept *rate* varies between runs (0.55–0.69) while its throughput
   does not. An independent review's call, recorded here: **document it rather than chase it** — greedy
   output is byte-identical, the 30-draw distributions agree on the modal class, and no further broad
   distribution campaign is owed before proposing the feature. It stays a documented limitation with the
   evidence attached.
5. Tree-patch validation (see above): the corrected patch has to reproduce the measured speedup from
   source, not merely compile.
