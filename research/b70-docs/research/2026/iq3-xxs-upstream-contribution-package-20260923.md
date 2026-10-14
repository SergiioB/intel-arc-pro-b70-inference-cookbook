---
title: "IQ3_XXS upstream contribution package — prepared, operator action required"
type: research
date: 2026-09-23
status: living
---

---
doc_id: RES-IQ3XXS-CONTRIBUTION-PACKAGE
doc_type: research_note
title: IQ3_XXS upstream contribution package (prepared; comment text left to operator)
lifecycle: active
created: 2026-09-23
updated: 2026-09-23
canonical_for:
  - iq3-xxs-contribution-package
  - llama-cpp-upstream-preparation
---

# IQ3_XXS upstream contribution package

**PROVISIONAL — NOT FOR PUBLIC HEADLINE. DRAFT ONLY — nothing has been filed, posted, or
submitted anywhere.** This note assembles everything the operator needs to make the
llama.cpp contribution personally. **The comment/PR text is deliberately NOT drafted here**:
`llama.cpp/AGENTS.md:88–98` forbids AI-written upstream comments, including drafting them
on the user's behalf. What follows is the technical substance only.

## Why this is worth contributing

Two separate, independently useful pieces:

### A. The IQ3_XXS reconstruction oracle + the FP32→FP16 overflow finding (small, immediately mergeable)

1. **`tests/test-iq3-xxs-reconstruct.cpp`** (161 lines) — a CPU-only bit-exact oracle for
   IQ3_XXS reconstruction: 131,072 blocks / **33,554,432 FP16 values, 0 mismatches** against
   `dequantize_row_iq3_xxs` (`ggml/src/ggml-quants.c:2575-2603`), covering pathological
   cases (scale nibbles 0 and 15, all four 7-bit sign-index fields × 128 values, all 256
   grid indices). It embeds its own exact RNE+inf FP16 converter and **documents why**: the
   exported `ggml_fp32_to_fp16()` in `libggml-base` is **sign-corrupting for |x| > 65504**
   under icpx default contraction (measured: `+196512` and `−196512` both → `0x81ff`, where
   g++ and `-ffp-contract=off` give `7c00`/`fc00`). Root cause: contraction folds
   `(x·2^112)·2^-110` and loses overflow detection. Device path (`sycl::half`) verified clean
   on the B70. **This is a real host-side bug worth its own report** — any CPU oracle for a
   device FP16 conversion will otherwise produce false parity failures at overflow
   magnitudes, which is exactly how we found it.

### B. The grouped-IQ3 `MUL_MAT_ID` kernel family (the reference implementation for #29245's IQ3_XXS extension)

`ggml/src/ggml-sycl/{iq3-xxs-reconstruct.hpp, mul-mat-id-iq3.hpp, mul-mat-id-iq3.cpp}` +
an opt-in `GGML_SYCL_IQ3_GROUPED_EXPERIMENTAL` CMake block + a fail-open hook in
`ggml-sycl.cpp` (placed *before* the ids D2H memcpy + `stream->wait()`, so a taken path also
eliminates the host sync). Gate/up AND down shapes, 216 experts, top-6 contiguous ids,
fp16/XMX/SLM + B70-name device checks, fail-open with a reason string on every unsupported
case. **Status: correct but slow** — parity perfect, zero spills (AOT ocloc `bmg`), 6/6
eval rows NMSE-clean, but **1.39× slower than stock** (39.36 vs 28.37 ms gate/up at n=512)
against a ≥2× gate (OUTCOMES L9). It is offered upstream as (i) a correctness reference and
(ii) the IQ3_XXS format semantics that #29245's grouped kernel needs — **not** as a
performance win. Be explicit about that in the write-up; overselling it is how contributions
get rejected.

## What the contribution needs (checklist for the operator)

Per llama.cpp contribution policy:

- [ ] **Explicit AI-usage disclosure** (undisclosed use = ban risk). State that the
      reconstruction oracle and kernel draft were developed with AI assistance and that the
      human is submitting/reviewing them.
- [ ] **A reproducible issue** describing (i) the `ggml_fp32_to_fp16` overflow sign
      corruption with the minimal repro, and (ii) the request for IQ3_XXS support in #29245's
      grouped MoE GEMM. These are arguably two issues — recommend filing them separately:
      the FP16 bug is independent and immediately actionable; the IQ3_XXS request is
      cross-referenced to #29245.
- [ ] **Regression test** — `tests/test-iq3-xxs-reconstruct.cpp` is ready (CPU-only, no
      backend init).
- [ ] **`llama-perplexity` / `llama-bench` non-regression evidence** — not yet produced
      (needs a GPU window on the K216 GGUF; the harness `scripts/tmp/p1-k216-e2e-ab-20260921.sh`
      shows the protocol).
- [ ] **Coordinate with `cwriter` on #29245** — the message must be the operator's own words.
      Technical content it should cover: IQ3_XXS support (`fg_stage_a` / `fg_block_traits` for
      `block_iq3_xxs`, 98-byte blocks per 256 weights = 3.0625 bpw, `db = d*(0.5f+s)*0.5f`
      two-multiply FP32 order, sign via 0x8000 XOR after FP16 rounding, reuse `iq3xxs_grid`
      and `ksigns_iq2xs` verbatim), the exact K216 shapes (gate/up `m=2048,n=512,k=4096`, down
      `m=4096,n=512,k=2048`, 216 experts, top-6), and the two review findings below.
- [ ] **Do NOT post anything from an agent session.** Draft-only is the standing rule.

## Two review findings to raise with #29245 (from `worklogs/P3-review-pr29245-20260922.md`)

1. **Deterministic intra-WG split-K** (`fused-gemm.cpp:156-157` partitions K; `:204-217`
   sums four partials in fixed sub-group order). It IS deterministic (no atomics, no cross-WG
   reduction) but changes accumulation rounding vs a single-pass reduction. Worth asking the
   author whether a non-split variant is acceptable for strict-parity users, or whether the
   rel. squared-L2 ≤ 1e-5 tolerance is the intended contract.
2. **The D2H expert-ID copy + `stream->wait()` survives** (`ggml-sycl.cpp:5145-5152`) even
   though the grouped call happens after. Our hook's placement (before that copy) is the
   pattern that eliminates it — worth suggesting for #29245's path too. Note this is also the
   named prerequisite for SYCL graph capture on `MUL_MAT_ID` (llama.cpp #28725 excludes that
   node "due to a memcpy+wait"), so fixing it unlocks the graph-capture decode lever as well.
3. Minor: the device-capability cache is one function-static bool (`:311-314`), not
   per-device — wrong on heterogeneous multi-device hosts.

## Package contents (paths)

| Artifact | Path | Lines | Status |
|---|---|---:|---|
| Contribution diff (tracked + untracked sources) | `results/2026/09/b70-improvement-probes-20260920/kernel-iq3-e0a/contribution-diff-20260923.txt` | 190 | ready |
| Parity/regression test | `llama.cpp/tests/test-iq3-xxs-reconstruct.cpp` | 161 | green (33.5 M values, 0 mismatches) |
| Reconstruction helper | `llama.cpp/ggml/src/ggml-sycl/iq3-xxs-reconstruct.hpp` | 31 | bit-verified |
| Kernel entry | `llama.cpp/ggml/src/ggml-sycl/mul-mat-id-iq3.hpp` | 11 | ready |
| Grouped kernel | `llama.cpp/ggml/src/ggml-sycl/mul-mat-id-iq3.cpp` | 304 | correct but slow (L9) |
| FP16 overflow repro + disposition | `results/2026/09/b70-improvement-probes-20260920/kernel-iq3-e0a/fp32-to-fp16-overflow-finding.md` | — | ready |
| #29245 review + delta list | `worklogs/P3-review-pr29245-20260922.md` | — | ready |
| Kernel spec (rev. R, the parity contract) | `worklogs/ARCHITECT-kernel-spec-20260922e.md` | — | ready |
| Perf evidence (the honest performance status) | `results/2026/09/b70-improvement-probes-20260920/perf-k216-grouped-iq3-20260923T002622Z.txt` | — | ready |

## Sequencing recommendation

1. **File the `ggml_fp32_to_fp16` overflow report first** — it is small, independent,
   immediately actionable, and unblocks correct CPU-oracle testing for everyone. Two-paragraph
   issue + the repro from the finding file.
2. Then the **#29245 IQ3_XXS request** (cross-link the PR, attach `test-iq3-xxs-reconstruct.cpp`
   and the reconstruction semantics, and the two review findings above).
3. The full kernel (`mul-mat-id-iq3.cpp`) as a follow-up reference attachment or a separate PR
   **only after** the non-regression evidence exists — and framed as a correctness reference,
   not a perf win.

Nothing here is filed, posted, or submitted. The operator writes the words and takes the action.
