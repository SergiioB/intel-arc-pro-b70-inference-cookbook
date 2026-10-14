---
title: "QWEN27 P2 — W3 line closure + T3 whole-model fake-quant spec"
type: worklog
date: 2026-09-25
status: final
opened: 2026-09-25T20:58Z
closed: 2026-09-25T21:05Z
ledger: RL-TBD (P8 rotated-GPTQ branch evidence; accretion by controlling session)
depends_on: ["QWEN27-T1T2-RUN-20260925.md", "QWEN27-REPLAN-QUANT-20260924.md §3/§5/§7/§8", "exl3-analysis-and-next-inference-levers-20260924.md §5.3/§9", "NIGHT-REPORT-20260925.md (measured step buckets, RL-061…065)"]
evidence: "results/2026/09/qwen27-rotation-t1t2-20260925T1812Z/ (manifest + RESULTS.md + t1/t2 JSONs/CSVs/logs)"
---

# QWEN27 P2 — W3/bytes line closure and T3 spec (2026-09-25)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** CPU/file work only: no GPU, no serving, no
model loads, no conversion, no downloads, no shared-authority edits (ledger/OUTCOMES/
status accretion belongs to the controlling session at close-out). A GPU campaign is
running on GPU 1 and was not touched. Owner: b70-quantization-engineer. Inputs consumed
read-only: the T1/T2 evidence root
(`results/2026/09/qwen27-rotation-t1t2-20260925T1812Z/`), the replan and T1/T2 worklogs,
the plan of record §5.3/§9, and live `df` on all four mounts.

Deliverables of this worklog: (1) formal CLOSURE of the W3/INT3 kernel-escalation line,
now with T2 as quality-axis supporting evidence; (2) the T3 whole-model fake-quant parity
SPEC (next rung, storage-gated); (3) a T1 gate-miscalibration correction note for the
record; (4) the plan-of-record implication. No kernel spec is written — see §1.

## 1. W3/INT3 kernel-escalation line — CLOSED

Verdict: **CLOSED, unchanged blockers, no escalation.** Nothing in T1/T2 moves any W3
gate, and T2 actively argues the quality payoff lives at W4, not W3. A kernel spec is NOT
written: the data do not justify a new-kernel program (would route to
b70-kernel-engineer / b70-upstream-researcher only if a gate below changed upstream).
Closure basis, enumerated:

1. **Loader (hard):** vLLM GPTQ loader `auto_gptq.py`
   `TYPE_MAP = {(4, True): uint4b8, (8, True): uint8b128}` (pinned vLLM
   `066ae99e24ef6aedc73c94b876c4f297d35a45e4`) — bits ∈ {4, 8} only; anything else
   raises at config parse. An INT3 artifact cannot be loaded, independent of quality.
2. **Packing (hard):** `pack_factor = 32 // weight_bits` is non-integral for 3. Any
   hypothetical layout (10 values/int32 with 2 dead bits ≈ 3.2 data bits/value; or
   cross-word 8-per-3-bytes needing multi-byte load+shift) is unimplemented and none is
   a 1-cycle nibble-class unpack. This is a new-packing program, not a conversion.
3. **Kernel (hard):** XPU chooser `_XPUWNA16_SUPPORTED_QUANT_TYPES = (uint4, uint4b8)`;
   `vllm-xpu-kernels` (pinned `4043e76…`, v0.1.13-5) exposes only
   `int4_gemm_w4a16`/`int4_gemm_w4a8` into oneDNN joint dtypes `f16_int4`/`bf16_int4`
   with **no `num_bits` parameter anywhere in the chain** — 4 is baked into the datatype.
   Release notes v0.1.12–0.1.15 add no W2/W3 weight-only GEMM; the one external
   "Xe2 2-bit GEMM (arXiv 2026-01)" mention remains unverified against pinned trees and
   is treated as nonexistent.
4. **Byte ceiling, now on the MEASURED-bucket basis (upgraded from the E0 model):** the
   2026-09-25 night measured the 44.8 ms decode round as host ≈4.8 + draft ≈5.5 +
   head ≈4.2–4.7 + **body ≈28–30 ms** + misc ≈2 (RL-061…065; §9 addendum). A 4→3
   body-only byte scaling (×0.75 data-only; ×0.7576 with G128 scales) saves 6.8–7.5 ms
   ⇒ round 37.3–38.0 ms ⇒ bound **≤1.18–1.20×**; the older E0 all-body reading
   (15.2 GB incl. resident F16 embed) tops at 1.215×. Even the unreachable all-bytes
   bound (1.333×) sits below the 1.37–1.76× multiple the proposal needed. A new-kernel
   program whose best case is ~18–21 % bytes on a stack measured at its byte floor.
5. **Rate-class law (independent closure argument):** sub-8-bit operands add no TOPS on
   Xe2 (`research/2026/xmx-dpas-rate-classes-20260915.md`) — W3 is a bytes-only lever,
   and the bytes bound is item 4.
6. **Acceptance risk unpriced:** a 4→3 target shifts the hidden-state distribution the
   MTP head accepts against; every 0.05 α ≈ 8.3–8.8 % decode (D9 economics), so
   0.05–0.10 α of drift erases most of a ≤1.2× byte bound. No W3 speed arithmetic is
   quotable without a measured-α gate first.
7. **D8/L10 head gates (lock the all-bytes scenario):** verify-LM-head INT4 retired at
   top-1 72.07 % and FP8 screens 93.3–93.4 % vs the ≥99 % offline promotion line — the
   head stays F16, capping any 4→3 program at the body-only bound.
8. **NEW tonight (T2, the quality-axis evidence):** rotation's measured value on this
   architecture is W4 quality at iso-bytes (KL 0.40–0.58×, top-1 +0.11–0.25 pts per
   block; every quantized tensor's GPTQ rel-Fro error dropped). The open quality
   question this branch can answer is "better W4", not "viable W3" — W3 has no loader,
   no kernel, and ≤1.2× best-case bytes. Effort goes to the rung that can load.

## 2. T3 SPEC — whole-model fake-quant parity harness (next rung, W4-quality framing)

Purpose: decide whether a rotated W4-G128 artifact is worth BUILDING as a
quality/capacity lever (iso-bytes vs the champion recipe). Fake-quant only — no packed
artifact is written, no serving, **no speed claim of any kind** (rotation at fixed W4
buys quality headroom, not bytes). This is `logit_or_KL_parity_tested`-class evidence,
below `task_quality_tested` on the correctness ladder; task quality is a later, separate
gate owned with b70-correctness-engineer.

**PRECONDITION (storage-gated, fail-closed): intact BF16 source re-fetch.**
The local "BF16 source" is a symlink stub farm (all 18 shard entries dangling; storage
doc §9 "known broken content") with no intact host copy — T3 needs a canonical re-fetch:

- Fetch all 18 shards of `Qwen/Qwen3.8-27B` (~55 GB planning figure; confirm the exact
  per-shard sizes and LFS oids live from the HF API immediately before the fetch) into a
  NEW explicit directory (e.g. `<mount>/tmp/qwen27-bf16-refetch-<date>/`); never write
  through or into the T-FORCE stub directory or any symlink.
- **Single-mount 2× headroom required before the download:** one mount with ≥110 GiB
  available (2 × 55 GB). Live probe at session time: NVMe SN770 103 GiB, WD 74 GiB,
  T-FORCE 54 GiB, root 20 GiB — **none currently satisfies the gate.** The SN770 could
  clear it by removing the ~95 GiB checksum-verified staging copy noted in storage doc
  §9, but that deletion needs explicit user authorization (not taken, not implied).
  A historical free-space figure is never authorization; re-probe at fetch time.
- Verify every shard sha256 against the HF LFS oid (shard 9 precedent:
  `af3c48cc…3a56d8`, verified 2026-09-25). Record all 18 in the results-root manifest.
- The already-staged shard-9 copy on WD
  (`…/tmp/qwen27-rotation-t1t2-20260925/model-00009-of-00018.safetensors`) may be
  reused for that shard; it is staging, not a source artifact.

**Harness design (CPU-only, single pass, streaming — 27B BF16 ≈ 54 GB exceeds 48 GB
host RAM, so no whole-model materialization):**

- Environment: `d11-venv` (torch 2.9.0+cpu, transformers 5.16.1, safetensors 0.8.0),
  8 CPU threads — exactly the proven T1/T2 stack. Reuse
  `scripts/tmp/rotation_t1t2_common.py` (H construction, fold map, GPTQ port).
- **Rotation: same Hadamard as T1/T2, seed recorded from the results-root manifest —
  torch seed `20260925` (torch.Generator, CPU, fp32): H_5120 = D @ (K_20 ⊗ S_256)/√5120,
  K_20 Paley type-I over GF(19), S_256 Sylvester, D Rademacher.** Orthogonality asserts
  (K@K.T == 20·I, S exact, H fp32 atol 1e-4) re-run before use. Fold map unchanged:
  residual-stream rotation only; RMSNorm gain pre-fold into readers; readers
  `W@H_folded`, writers `H^T@W`; GDN v/qk spaces, conv1d, RMSNormGated, q/k norms,
  RoPE, gates unrotated; FFN 17408 unrotated (offline-fold-unavailable space; the
  block-diag online variant stays the separate Q1bd branch, out of T3 scope).
- **Arms (three activation streams propagated simultaneously, layer by layer):**
  R = reference (BF16, untouched); Q0 = unrotated + GPTQ fake-quant (champion-contract
  control); Q1 = rotated+folded + GPTQ fake-quant, run in rotated space, fp32
  un-rotation at comparison points only (T1/T2 harness convention; deployment would
  fold H into the head — same reader-fold arithmetic class already exercised at block
  scope).
- **Quantizer: verbatim T2 port** — bits=4, sym=True, group_size=128, desc_act=False,
  percdamp 0.01, blocksize 128; calibration Hessians from the arm's own propagated
  stream (standard sequential GPTQ; Q1 calibrates in rotated space, as a real rotated
  conversion would). **Exclusions (champion parity, recorded mixed precision):**
  in_proj_a/in_proj_b and conv1d BF16 (still receive the H fold in Q1); all norms,
  embedding, LM head, MTP tensors BF16. That is the champion's 400-qweight scope.
- **Input provenance:** the D11 capture files carry `token_ids` alongside the hidden
  states — T3 feeds the real serving-distribution TOKENS through the arm embeddings
  (embedding rotated via the writer fold in Q1). Same fixed slices as T1/T2:
  calibration = first 12 sorted files excluding eval (12 seqs, ~5438 rows); evaluation =
  hs_1003 + hs_1004 (821 positions). Layer-indexed hidden states exist ONLY for the
  final layer (post-63) in that capture, so the D11 hs arrays are used solely as a
  final-layer cross-check; any intermediate layer-scope input need would be synthetic
  and labeled (none is required by this design — inputs enter at layer 0 by
  construction). Caveat carried: token distribution originates from the INT4-champion
  serving run; identical across arms, so arm comparisons are internally valid.
- **Metrics:** per probed layer — every 8th: layers 8, 16, 24, 32, 40, 48, 56, and the
  final post-layer-63 residual — per-position cosine and RMS-normalized max-abs of each
  arm vs R (T1 formulas). Final logits (fp32-unrotated final residual @ original BF16
  head): top-1 argmax agreement and KL(softmax) vs R per position (T2/D8 conventions).
- **PRE-DECLARED GATES (declared here, before any T3 run):**
  - (a) per probed layer: mean per-position cosine(Q1 vs R) ≥ 0.999 AND min
    single-position cosine ≥ 0.99;
  - (b) final top-1(Q1 vs R) ≥ 0.98;
  - (c) final mean KL(Q1‖R) ≤ 0.05 nats;
  - (d) iso-bytes continuity (T2 form): top1(Q1) ≥ top1(Q0) − 0.005 AND
    meanKL(Q1) ≤ 1.1 × meanKL(Q0).
  Justification: (a) unquantized rotation measured 0.99999+ (T1); single-block Q1
  cos_min 0.99919–0.99946 (T2) — the 0.999 mean line allows an order of magnitude of
  64-layer accumulation while the 0.99 min catches catastrophic positions. (b)/(c)
  sit between the D8 disaster marker (INT4 head 72 % top-1) and the fp16-matmul noise
  floor (99.58 %), with block-scope KL at 1.1e-4 leaving wide margin; a W4 body that
  cannot hold 98 % top-1 / 0.05 nats through its own BF16 head is not buildable.
  (d) preserves the T2 decision comparison at whole-model scope. Q0 has no formal gate
  (control); its values contextualize (d). Max-abs is REPORTED, not gated (see §3).
- **RAM/storage budget:** peak < ~8 GB RAM (one layer's weights + folded copy + fp32
  Hessians + three activation streams); disk = the 55 GB source + small results files;
  no scratch artifact written (single pass, layers never revisited). Results root:
  `results/2026/09/qwen27-rotation-t3-<UTCstamp>/` with manifest (gates declared
  before measurement), per-layer/per-position CSVs, JSON summaries, run log.
- **Effort estimate:** harness dev 3–5 h (mostly reuse); run ≈ 64 layers × (3-stream
  forward of 14 seqs + 2-arm GPTQ) extrapolated from measured T1/T2 block costs
  (~11.7 s per block-sequence-arm, ~49 s per block-arm GPTQ) ⇒ ~8–12 h wall — one
  overnight CPU window; fetch + sha256 verification 0.5–2 h. Total: one prepared
  session, storage-gated at the start.
- **Decision T3 feeds:** GO/NO-GO on building a rotated W4-G128 artifact as a
  QUALITY/capacity lever (P8 framing; iso-bytes vs champion recipe — never a speed
  claim). PASS ⇒ hand to a separately named artifact build + static-coverage/numerical/
  coherent-load/deterministic-replay ladder, then task quality with
  b70-correctness-engineer. FAIL (two failed quality gates, P8 rule) ⇒ close the
  rotation branch; W3 is already closed independently (§1).

In-flight note: the T2 secondary arm Q1bd (online block-diag 17408) was still running
under the prior session's nohup process at this session's close (observed alive; not
touched — occupancy not created here is not mine to manage). It carries an online
runtime cost, sits outside the zero-runtime branch, has no gate, and does not affect
this closure or the T3 spec.

## 3. T1 gate-miscalibration correction (for the record)

The T1 pre-declared max-abs tolerance (≤5e-2 RMS-normalized) was set BELOW the floor
that ANY BF16 weight-rewrite produces on this model's real activations — the T1/T2
controls measured that floor directly: gain-fold-only (exact in exact arithmetic, no
rotation) 0.1745 (GDN) / 0.2322 (ATT); one extra BF16 state round-trip alone
0.1467 / 0.1565; the declared primary arm sat at 0.2782 / 0.2412 with the deployment-
fair input series ≈ equal (0.2764 / 0.2371), attributing the excess to the re-rounding
floor, not the rotation. The T1 verdict stands as declared (fail-closed discipline —
no retroactive pass; the recorded interpretation is attribution, not a gate change).
**Correction for future T-runs:** absolute max-abs tolerances must be derived from a
measured no-op rewrite control on the same model BEFORE declaration. Derived bound
from tonight's controls: gain-fold-only floor 0.17–0.23 ⇒ set future max-abs gates at
~2× that control (≈0.35–0.47 for this model/class), and keep cosine as the primary
equivalence metric — it was the discriminating statistic (0.99999+ per position on
both block types). Same principle for any logits gate: measure the BF16 rewrite/
round-trip noise floor first, then set the tolerance from it, not from a round number.

## 4. Implication for the plan of record (§9)

Rotation's measured value on this architecture is W4 quality headroom at iso-bytes —
the parked P8 framing, now with block-scope evidence (T2: KL 0.40–0.58×, top-1
+0.11–0.25 pts, every tensor's rel-Fro down) and with the equivalence question
resolved in the cosine sense (T1) — so the plan of record's open rotation item is
answered as "W4-quality lever, not a W3/bytes lever": W3 stays closed on
loader/packing/kernel plus the now-measured ≤1.18–1.21× bucket bound and the unpriced
α risk, T3 (whole-model fake-quant parity, §2) is the only sanctioned next rung and is
storage-gated behind a ~55 GB canonical BF16 re-fetch with 2× single-mount headroom,
and the separate parked P-C (rotation-enabled W4A8) remains its own ladder — tonight's
evidence is weight-only W4A16-scope and does not cover activation-side transforms.

## 5. Outcome record

- Changed files: this worklog only. No GPU, no serving, no downloads, no conversions,
  no shared-authority edits, no commits, nothing deleted (SN770 reclaim explicitly left
  to user authorization).
- Checks: live `df` on all four mounts recorded (§2); T2bd process observed and left
  running; all closure claims traced to the pinned-SHA replan §3, the T1/T2 evidence
  root, and the §9 measured buckets; situation discipline: research intent, CPU-only.
- Unverified: whole-model transfer of the T2 block-scope margin (that IS T3); Q1bd
  final numbers (pending, prior session's process); AutoRound rotated-export runtime
  contract (unchanged, irrelevant until T3 passes); exact HF shard sizes/oids at fetch
  time.
- Next deciding experiment: T3 per §2, strictly after the storage precondition passes
  (single mount ≥110 GiB avail + user-authorized space changes if any).
