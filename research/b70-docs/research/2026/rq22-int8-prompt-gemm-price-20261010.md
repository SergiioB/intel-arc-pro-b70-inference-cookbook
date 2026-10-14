---
title: "RQ-22 price: INT8 prompt GEMM for Strata SYCL — Amdahl on the measured 8K prefill stage table (2026-10-10)"
type: research
question: RQ-22
date: 2026-10-10
status: provisional
---

# RQ-22 price: INT8 prompt GEMM — Amdahl at 2x on the measured 8K prefill GEMM share

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Zero-GPU arithmetic only. Classification
of both tool runs: `theoretical_inferred_bounds_not_measured_gains`. This note does
not change RQ-22's `deferred` program status and adopts nothing (AGENTS.md §12).

## 1. What was priced, and the registered rule

RQ-22 smallest_probe (`registry/research-program.json`, id RQ-22): "read the Strata
prompt path and stage timings to get the GEMM share of an 8K prefill; Amdahl at 2x
local speedup; reject below a 10 % whole-prefill gain." Stop rule: "Amdahl price
below 10 % closes the question without GPU." Success gate: ">= 1.3x 8K prefill"
with the exact-answer battery, an NLL budget and decode unchanged.

**What this session produced: no new stage timings.** A filesystem sweep for files
modified 2026-10-10 (repo + `/mnt` data mounts) found only download-staging state
(`scripts/tmp/stage_b70_speed_20261010.log` — a fatal pre-download refusal, no
measurement) and workflow bookkeeping; no new results root exists under
`results/2026/10/`. I could not enumerate sibling workflow steps' in-flight outputs
(workflow introspection is unreachable from this subagent session). The price
therefore rests on the sealed RQ-14 prefill-stage evidence below.

## 2. Measured evidence (2026-10-09 RQ-14 artifacts, sealed in the repo)

The Strata prompt path has a built-in per-phase GPU profiler:
`STRATA_PREFILL_TIMING=1` (`PfTimer`, external pinned tree
`/mnt/<sata-disk>/strata/repo/strata-0.1.40.3/sycl/src/prefill/prefill.cpp:1955-2020`;
phase marks at lines 3278, 3901-4001; stats struct
`include/strata/prefill/prefill.hpp:30-41`). The expert GEMMs run as oneMKL FP16
GEMMs on dequantized weights (`Gemm::f16` → `dpct::blas::gemm`, `real_half`
operands, f32 compute — `sycl/src/prefill/gemm.dp.cpp:799-808`); the fused int8 MMQ
prompt path is `__HIPCC__`-only, i.e. not built in the SYCL port
(`gemm.dp.cpp:810`), exactly as RQ-22's scope states ("upstream fused int8 prompt
kernels (#136, CUDA) are not built in the port").

Stage table of the 8191-token prefill on the adopted Flash-Next route (one B70,
IQ3_XXS, `--prefill 4096 --kv int8`, expert cache auto, PLE on), retained verbatim
at `results/2026/10/rq14-strata-matched-screen-20261009/raw/prefill-2k/base.log:45`
(sha256-identical to the working copy on the T-FORCE strata tree; headline wall at
`base.log:69`: 889.48 tok/s):

| phase group | ms | share of 9207 ms GPU timeline |
|---|---:|---:|
| **gemm gate/up + gemm down (oneMKL FP16 expert GEMMs)** | **3434** | **37.30 %** |
| dequant (separate pass feeding those GEMMs) | 1224 | 13.3 % |
| qsa attn (prompt attention) | 1607 | 17.5 % |
| host grouping / gather / wait copy / combine | 1052 | 11.4 % |
| hc read, gdn, gdn conv+gates, gdn recurrence, gdn out proj | 1388 | 15.1 % |
| qsa proj, indexer, select, router+shared, embed+steps, ple | 503 | 5.5 % |
| all 19 phases (sum) | 9208 | ≈ timeline |

Replicate: `base230.log:45` (230 W cap) reproduces every phase within 0.2 % — the
split is cap-insensitive. *[Citation corrected 2026-10-10: this sentence originally
ended "…matching the sweep's no-cap-sensitivity finding
(`results/2026/10/rq14-strata-matched-screen-20261009/RESULTS.md:241-249`, external
copy in the same directory)". That pointer is dangling — correction record in §7
below; the cap-insensitivity fact rests on the two log lines themselves.]*

## 3. Arithmetic

+10 % whole-prefill at a 2x local speedup needs GEMM share
s ≥ 2 × (1 − 1/1.1) = **18.18 %**. Clearing the registered ≥ 1.3x gate at 2x needs
s ≥ **46.15 %**. Measured: GEMM-only s = 0.37298; dequant+GEMM s = 0.50592.

| case | wall_fraction | at 2x | whole-prefill gain | vs 10 % stop rule | vs 1.3x gate |
|---|---:|---:|---:|---|---|
| GEMM phases only (primary) | 0.37298 | 1.2292x | **+22.9 %** | pass (2.05x margin) | **fail (cap 1.229x)** |
| dequant+GEMM fused (sensitivity) | 0.50592 | 1.3386x | **+33.9 %** | pass | pass |
| GEMM-only, infinite speedup ceiling | 0.37298 | 1.5948x | +59.5 % | — | reachable only above 2.62x local |
| GEMM absolutes held on the un-instrumented confirm wall (8281 ms) | 0.41468 | 1.2618x | +26.2 % | pass | fail |

Tool runs (input hash retained by the tool):

- `python3 tools/research.py price --input research/references/rq22-int8-prompt-gemm-price.json`
  → predicted_wall_speedup **1.229241** (gain +22.92 %), ceiling 1.594845, input
  sha256 `8579ef2903f16718816014dabbe17a9037e2739bdf262b8aa4216be94fd172bb`.
- `python3 tools/research.py price --input research/references/rq22-int8-prompt-gemm-fused-price.json`
  → predicted_wall_speedup **1.338616** (gain +33.86 %), ceiling 2.023964, input
  sha256 `fd102bbef8046ad441ac3d332b52e7326c1651963d66e97bdb66b33940f762b2`.

Provenance declared in both inputs; no hardware defaults are used anywhere — the
only quantities are the measured fraction and the rate-class speedup. The 2x is the
measured s8-vs-16-bit DPAS rate class (s8 363 TOPS vs bf16 170 TFLOPS on this card
family; ratio 2.135, price uses the conservative 2.0 —
`research/2026/xmx-dpas-rate-classes-20260915.md:33-46`, which itself warns a 2x
*application* figure "is still not implied by anything").

## 4. Verdict

**Not closed by arithmetic — the stop rule does not fire.** The measured GEMM share
(37.30 %) is 2.05x the 18.18 % threshold, so RQ-22 is worth GPU hours under its own
predeclared price rule.

**But the cheap version of the lever cannot clear the question's own success gate.**
A GEMM-swap-only change (dequant still writes FP16, oneMKL call swapped to s8)
caps at 1.229x < 1.3x even at a perfect 2x. Reaching ≥ 1.3x at the 2x rate class
requires the fused s8 datapath that absorbs the dequant phase (13.3 %) — the
upstream-#136-class kernel port whose cost RQ-22's `economic_gate` flags (external
precedent for that class: the fused dequant+XMX kernel measured 0.22x of the
separate pair on the vendor's card, `research/2026/flashnext-strata-source-math-20261009.md:40-44`,
E0). GPU budget is therefore justified **only for the fused design**; a plain s8
oneMKL swap is a priced dead end against the current gate.

## 5. Limits

1. The stage table is n=1 instrumented run plus one cap-replicate, not 5 pairs; the
   instrumented wall is ~11 % slower than the confirm cell (889.48 vs 989.11 tok/s).
   Shares, not absolutes, feed the price; if GEMM absolutes held on the
   un-instrumented wall the share rises to 0.415 and the price only improves. A
   `STRATA_PREFILL_TIMING` re-run belongs in any GPU-phase confirmation.
2. local_speedup=2 is a device rate class, not a measured application speedup on
   these phase shapes (M=4096 expert-grouped GEMMs); the realizable gain may be
   lower — establishing it is exactly the registry's next deciding step, the
   isolated kernel parity probe.
3. s8 activations change outputs: approximate method, needs its own predeclared
   quality objective (AGENTS.md §11; the 1.89x-linears/+0.17 %-NLL precedent is
   external, E0). IQ3_XXS dequant targets non-uniform codebook levels, so an s8
   dequant target is itself an approximation — any parity claim must define its
   reference arm.
4. Decode is excluded by RQ-22 scope; the prefill path runs outside the captured
   decode graphs, but any engine change is still evaluated in the normal serving
   path with capture on (standing rules §1).
5. Dense projections (qsa proj 171 ms, gdn out proj 166 ms, router+shared 77 ms)
   were conservatively excluded from the fraction; including them moves the price
   up, never below the pass margin.

## 6. Next

RQ-22 stays `deferred` (kernel budget after RQ-17/RQ-18 — unchanged by this note).
If a budget opens: the first GPU step is the isolated kernel parity microbench on
real shapes (per the registry `smallest_probe`), and per §4 it must be the **fused**
s8 dequant+GEMM design, not a oneMKL s8 swap, or the ≥ 1.3x gate is unreachable by
construction.

## 7. Correction record (2026-10-10, report-writer pass)

The §2 replicate sentence cited
`results/2026/10/rq14-strata-matched-screen-20261009/RESULTS.md:241-249` ("external
copy in the same directory") for the sweep's no-cap-sensitivity finding. Verified
dangling 2026-10-10 by this writer: `wc -l` on that RESULTS.md returns exactly 130
lines (no cap-sensitivity section exists at any line range), and the directory
holds no external copy (`ls`: ARTIFACTS.sha256, RESULTS.md, ROUTE-CHANGE-DRAFT.md,
SHA256SUMS, prefill-sweep.sh, raw, serve-flashnext-iq3xxs.json). The underlying
measurement is unaffected and was re-verified in place: `base.log:45` and
`base230.log:45` both carry the full phase table (re-read 2026-10-10), and
`sha256sum base.log` = `ef9aa512c82231b76c679175916b9dfb8643491eee6c8a0008d86070a2368d0a`,
equal to the hash recorded in ledger row RL-131. The cap-insensitivity fact now
rests solely on those two log lines. RL-131 never propagated the dangling pointer
(it cites `base230.log:45`). No number in this note changes. Run-1's separate
rounding nit on this sentence ("within 0.2 %" vs the recomputed 0.21 %, workflow
report §3.4) stands as flagged and is left as written. Defect first surfaced by the
2026-10-10 adversarial sweep; the run-1 workflow report §3.4 recorded it as still
unfixed — fixed by this edit.
