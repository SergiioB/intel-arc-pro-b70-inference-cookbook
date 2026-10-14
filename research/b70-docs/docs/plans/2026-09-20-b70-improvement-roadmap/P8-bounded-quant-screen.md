---
type: plan
date: 2026-09-20
phase: P8
title: "Bounded quant screen: AWQ → rotated GPTQ → changed-hypothesis W4A8"
status: RESEARCH
opened: 2026-09-20
updated: 2026-09-20
ledger: RL-TBD-P8
depends_on: []
effort: 3–5 sessions screen; 6–12 more only for a winning branch
---

# P8 — Bounded quantization screen

## Status log

| Date | Status change | Note / evidence |
|---|---|---|
| 2026-09-20 | filed as PLANNED | Roadmap filed from b70-architect draft |
| 2026-09-21 | PLANNED → RESEARCH | **AWQ entry gate PASSED — do not stop cheaply.** AWQ on this stack resolves to an existing kernel, not a kernel project: `vllm/model_executor/layers/quantization/auto_awq.py:304` returns `AutoAWQMarlinLinearMethod` when `current_platform.is_xpu()`, which reaches `kernels/linear/__init__.py:492-495` → `mixed_precision/xpu.py:113` → `torch.ops._xpu_C.int4_gemm_w4a16` (bound in `vllm-xpu-kernels` `csrc/xpu/onednn/onednn_matmul.cpp:199`, `torch_bindings.cpp:37`) — the same op the champion body already dispatches (observed in `results/2026/08/nemotron-gptq-builtin-profile-20260812T171658Z-671822/`). So AWQ is a **packing change on a validated W4A16 path**. Two corrections to the branch's framing: the `docs/reference/quantization-decision-record.md:132-134` "no route" clause is superseded at this revision, and the earlier W4A8 quality failure was a local patch on this same kernel (`scripts/tmp/patch_dense_w4a8.py`), so it does not implicate default AWQ. AWQ itself is genuinely unrun here (`grep -i awq registry/research-ledger.json` → 0 hits). One real gap to close first: the local kernels tree is `v0.1.13-5-g4043e76` while the deployed wheel is `0.1.12.3` (`models/qwen/qwen3.8-27b/serving.md:32`). Probe: `worklogs/P8-awq-loader-probe-20260921.md`. |
| 2026-09-21 | falsifying test executed — verdict holds | `torch.ops._xpu_C.int4_gemm_w4a16` **present** in the deployed local image (vLLM 0.26.1rc1.dev457+gc810e5ee9, vllm-xpu-kernels 0.1.12); it is the only int4/w4a op in the wheel, so no `w4a8` path is shipped. Deployed wheel 0.1.12 vs local tree v0.1.13-5-g4043e76 remains the one item to reconcile before the iso-byte screen. |
| 2026-09-21 | AWQ branch screened — two corrections, branch stays open | Downloaded `cyankiwi/Qwen3.8-27B-AWQ-INT4` and served it on the local image: **loads and serves cleanly** (5,075 MiB free after load; 5/6 on a bounded task suite, with one item ill-posed by my own construction). **Correction 1:** the engine reports `quantization=compressed-tensors` for this checkpoint, so it does **not** exercise the AWQ loader path — the AWQ route stays source-verified + falsification-tested but not end-to-end exercised. **Correction 2:** it is **not iso-byte** — language-only weights **20.10 GB vs the champion's 18.63 GB (1.079×)**, so it cannot answer "quality at iso-bytes". Next step specified: obtain or build an **AWQ-declared, iso-byte** W4A16 artifact, then screen. Nothing here supports a speed claim. Worklog `worklogs/P8-20260921.md`. |
| 2026-09-21 | AWQ branch: the format question is overtaken by the ecosystem | Checked the declared `quantization_config` of **five** community "AWQ" artifacts for this model (`cyankiwi` ×2, `philbert440`, `TheUnderscore`, `ulkaa`, `abihsoro`): **every one declares `quant_method=compressed-tensors`, none declares `awq`**. So for this model family the AWQ-format route is legacy — the artifacts the community publishes under an AWQ name are compressed-tensors W4A16, which this stack already serves (proved in the screen above at 5,075 MiB free after load). **Consequence:** the AWQ loader-kernel investigation is closed as "not the artifact class in use"; what remains for P8 is the **iso-byte quality** comparison on a compressed-tensors W4A16 artifact, and the available ones are ~8 % larger than the champion's bytes. The branch is therefore re-scoped, not abandoned. |
| 2026-09-21 | **AWQ-format branch: non-recommendation recorded** | With all five community "AWQ" artifacts for this model declaring `compressed-tensors` (above) and no iso-byte AWQ artifact available, the branch's honest verdict is: **do not pursue AWQ as a format route on this fleet** — it is not what the ecosystem produces for this model, the loader path it targets is already served by the compressed-tensors class that this stack runs, and no speed gain is assumed by the phase anyway (equal W4A16 arithmetic). If a quality-per-byte question ever needs an AWQ-format answer, the way to get an iso-byte sample is a **local conversion from the BF16 source with a declared group size**, not another download. Until then this branch is closed with the evidence attached and no further spend. |

## Why this decision (the reasoning)

**The thesis: the format question stays open exactly as long as it is cheap to keep
open. Three branches, ordered by cost, each stopping cheap — and the EXL3 analysis is
what defines what "cheap" means here.**

1. **Why the baseline stays until beaten on evidence.** AutoRound-contract GPTQ-INT4
   sym G128 W4A16 is the measured champion (102.61 tok/s with MTP4; dominates FP8 on
   both axes per the rate-class law: INT4 W4A8-class weights at 0.5 bytes vs FP8's
   1-byte in the half-rate compute class). The quant decision record's rule 5:
   reassess non-GPTQ formats only after their exact loader and XPU kernel path are
   demonstrated. Reputation (community praise) is not a gate — that is the direct
   lesson of the EXL3 analysis: a format can be excellent *elsewhere* and still have
   no runtime here and no arithmetic edge on Xe2 (trellis codes reconstruct to
   floating tiles; sub-8-bit adds zero TOPS; the only win is bytes).
2. **Why AWQ first.** It is the cheapest unresolved question in the format space:
   never measured on this host (no artifacts, no demonstrated XPU kernel path). If the
   loader/kernel path exists or a provably lossless repack does, the screen is an
   iso-byte quality comparison — *expected speed ≈ 1.0×* because identical arithmetic
   and packing would be used; the possible win is quality-at-iso-bytes, which could
   then justify smaller groups or fewer BF16 exceptions (a capacity/quality lever).
   If compatibility turns into "write a new backend," stop — that is the EXL3 trap at
   smaller scale.
3. **Why rotated GPTQ is the "EXL3 quality without EXL3 kernels" path.** The part of
   EXL3 that genuinely improves quality-per-bit at low bpw is incoherence processing
   (Hadamard-style rotations à la QuaRot/SpinQuant) — and rotations can, in favorable
   cases, be folded into the weights or run as cheap activation transforms while
   *reusing the existing validated W4A16 kernels*. No new kernel stack. The honest
   caveats are in the spec: transforms that cannot be folded add per-layer online
   cost; norm/residual/recurrent-state semantics must be preserved (hybrid-GDN!); and
   the claim to test is quality-at-iso-bytes, not speed.
4. **Why W4A8 is last and hypothesis-gated.** It is the only route into the 363-TOPS
   DPAS class for the champion's weight format (bound: 363/170 ≈ 2.14× on the
   compute-bound fraction — S ≤ 1.36× at g=0.5, 1.74× at g=0.8, *before* quant/scaling
   overhead). But a W4A8 path already failed here once (quality divergence; discarded,
   preserved as historical evidence). Re-running the same recipe is banned; the branch
   opens only with a genuinely changed hypothesis (calibrated smoothing, rotations,
   selective BF16 exceptions, corrected group scaling) and verification of *actual*
   integer DPAS dispatch. The P0 W4A4 microbench informs the arithmetic ceiling but
   does not make A4 activations quality-viable.
5. **Why the branches stop cheap.** Each branch's first gate is a narrow layer/operator
   probe (loader compatibility, reconstruction parity, activation-outlier behavior) —
   hours, not the multi-hour conversions. The EXL3 deferral conditions (see
   DEFERRED-AND-TRACKED.md) apply by analogy to every exotic format: capacity-blocked
   model + demonstrated low-bpw quality + a maintained backend before any kernel
   program starts.

**Alternatives considered and rejected.**

- *Adopt EXL3.* Deferred with conditions — the full reasoning lives in
  DEFERRED-AND-TRACKED.md §1 (no runtime, BF16-class after reconstruction, ≤1.25×
  bound at 3.5 bpw before overhead, no capacity crisis in the current fleet).
- *MXFP4 revival.* Measured slow (10.4 tok/s) on the hybrid model *because of* Triton
  FLA fallbacks — a runtime fix, not a format case; and 2× worse KLD at the same
  bitwidth stands on its own.
-*Q4_0/Q5/Q6 GGUF sweeps in vLLM.* Wrong engine; the GGUF matrix was closed in the
  2026-09-13/14 showdown (fresh Q4_K_M 20.01 tok/s champion within llama.cpp; PPL
  invalid on hybrid-GDN, #28879).
- *GPTQ-Int8 / W8A16 INT8.* More weight bytes than INT4 without integer-activation
  throughput — no advantage on either axis (rate-class law).

**What would change this decision.** A capacity-blocked desirable model arriving
(escalates the rotated-GPTQ branch and possibly the EXL3 deferral review); upstream
vLLM XPU landing maintained AWQ or incoherence-aware loaders (collapse branches to
validation); the W4A8 changed hypothesis failing quality twice (close the branch,
record the non-recommendation).

## Objective

Investigate quality-per-byte and compute-class opportunities without displacing the
validated GPTQ baseline on speculation. Execution order: (1) AWQ compatibility +
iso-byte screen; (2) rotated/incoherence-processed GPTQ; (3) changed-hypothesis W4A8
only if justified.

## Baseline and E0 targets

Baseline: quality-passed AutoRound-contract GPTQ-INT4 W4A16
(`results/2026/09/qw38-quant-format-showdown-20260913/`,
`research/2026/xmx-dpas-rate-classes-20260915.md`).

- AWQ: no assumed speed gain (≈1.0× with identical arithmetic/packing); possible
  quality-at-iso-bytes.
- Rotated GPTQ: quality-at-iso-bytes or fewer high-precision exceptions; no speed gain
  assumed.
- W4A8: integer-DPAS opportunity only after activation quality passes; bound S ≤
  1/(1−g+g/2.14) ≈ 1.36× (g=0.5) to 1.74× (g=0.8) before overhead.

## Engineering work

### AWQ

- Confirm loader and kernel support before conversion.
- If repacking: prove it lossless (scales/zeros/layout preserved).
- Match effective bytes and mixed-precision exclusions.
- Stop cheaply if no credible compatible route exists.

### Rotated GPTQ

- Identify transforms foldable into weights; account for online transforms that cannot
  be folded.
- Preserve norm, residual, attention, recurrent-state semantics.
- Test whether quality gains permit smaller groups, fewer exceptions, or lower
  bitrate; do not assume a speed gain.

### W4A8

- Document why the earlier quality-divergent path failed.
- Declare a genuinely changed hypothesis (calibrated smoothing, rotations, selective
  BF16 exceptions, corrected scaling).
- Verify actual integer DPAS dispatch.
- Keep the native MTP head and sensitive recurrent operations separate.
- Test activation outliers and long-sequence error accumulation before full conversion.

## Validation and benchmark plan

- Tensor reconstruction and operator tests.
- Per-position F16/reference logits on hybrid-GDN; KL/logit error distributions;
  deterministic replay; task quality.
- Quantized-target acceptance diagnostics (drafter interaction — P9 depends on the
  frozen answer).
- Standard five-cell performance card only after quality passes.
- No PPL-based hybrid ranking. Source artifact preserved unchanged.

## Risks and stop criteria

- Stop each branch after two failed quality gates.
- Do not spend hours converting before the narrow layer/operator probe passes.
- Stop AWQ if compatibility cost becomes a new backend project.
- Reject quality gains requiring online transform overhead exceeding the deployment
  objective.
- Reject W4A8 if quality divergence persists under the changed hypothesis.
- Reject resident weight expansion that destroys the intended capacity advantage.

## Deliverables

- `research/2026/b70-bounded-quant-screen-<YYYYMMDD>.md`
- `results/2026/<MM>/b70-bounded-quant-screen-<UTC>/`
- Ledger entry with separate branch outcomes.

## Cookbook integration

Update the quant decision matrix with measured compatibility, quality, and performance
— not format reputation. Failed branches receive documented non-recommendations. Only
quality-passed, reproducible artifacts become recipes.
