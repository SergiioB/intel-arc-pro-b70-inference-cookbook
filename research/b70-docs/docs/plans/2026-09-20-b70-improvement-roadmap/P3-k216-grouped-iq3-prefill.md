---
type: plan
date: 2026-09-20
phase: P3
title: "K216 grouped-IQ3 prefill (device-side grouping + tiled reconstruction)"
status: RESEARCH
opened: 2026-09-20
updated: 2026-09-20
ledger: RL-TBD-P3   # link RL-011 when execution starts
depends_on: [P1 (withdrawn; its prerequisites are delivered and validated — see below)]
effort: 5–10 sessions
---

# P3 — K216 grouped-IQ3 prefill

## Status log

| Date | Status change | Note / evidence |
|---|---|---|
| 2026-09-20 | filed as PLANNED | Roadmap filed from b70-architect draft |
| 2026-09-21 | dependency re-stated after P1's withdrawal | P1 is **WITHDRAWN** (+3.54 % e2e, gate failed), but the three primitives P3 was waiting on are now **validated on the real artifact**: the device-resident-ID IQ3 path exists and is parity-tested at K216 shapes, the expert-shape knowledge is measured (216 experts, top-6, h=4096, ffn=2048 → 3·h·f = 25.2 M weights/expert), and the fused decode op is measured at **174.9 µs vs a 31.7 µs weight-read floor (5.52× off)** with the non-fused prefill path materializing FP16 at ≈100 MB/op. The artifact is back on the host (`/mnt/<nvme-disk>/models/DeepSeek-V4-Flash-0731-REAP-K216-UD-IQ3_XXS/`, 89 GB, 3 shards), so this phase's entry probe (per-expert-M distribution + prefill attribution trace) is **payable now**. Entry criterion (architect, 2026-09-21): withdraw if grouping + FP16 expansion turns out to be < ~30 % of prefill time. |
| 2026-09-21 | PLANNED → RESEARCH — **entry probe PASSED** | Prefill ubatch scaling measured on the re-downloaded artifact (dual card, L5 placement, 230 W, n=3 cold p7000/g1, warmup discarded): `-ub 512` **101.20 pp** → `-ub 1024` **130.78 pp = 1.292×**. The criterion was fixed before the run (proceed ≥1.15×, withdraw ≤1.05×), so the phase proceeds: the per-ubatch fixed cost (host counting-sort grouping + IQ3→FP16 expansion ≈100 MB/op ≈ 5.2× the IQ3 bytes + per-expert launches) — not GEMM math — is the prefill constraint. This independently reproduces the recorded 1.26–1.30× sensitivity at a different context config. Refreshed anchors: **101.2 pp @ub512 / 130.8 pp @ub1024** (p7000/g1, `-c 8448`); the recorded 105.1 pp was a different cell and is not directly comparable. Worklog: `worklogs/P3-20260921.md`; evidence `results/2026/09/b70-improvement-probes-20260920/p3-prefill-ub-20260921T*/`. |
| 2026-09-23 | **E0a isolated-op gate FAILED — kernel closed as LOSS** | Same-session thermally-matched A/B (`perf -b SYCL0`, caps untouched): gate/up n=512 old 32,439 µs vs new **45,647 µs (1.41× slower)** — gate demanded ≤14,400 µs; down 1.32× slower; n=32 2.2–2.3× slower; decode n=1 unchanged (mmvq shield + `tokens>=2` guard verified). Parity ladder **PASSED**: 502,192/502,192 blocks bit-identical FP16+F32; L2 3.5e-7 (gate/up) / 4.5e-7 (down); the ragged n=97 letter-miss (1.42e-5 vs 1e-5) is the STOCK path's own error — the new kernel is 7.5e-8 (~188× more accurate); NMSE 9/9. Failure attribution: latency/occupancy-bound (serial 16-step k-loop, 2 barriers/tile, ~4 WGs/Xe-core) at 1.13 TFLOPS / ~20 GB/s — the spec's 6–9 ms traffic-floor model is falsified. E0b not attempted (premise cannot recover 33 ms; `unattributed slowdown → STOP` fired). Fallback `gemm_batch` recorded not attempted (est. 1.3–1.8×, below gate). Kernel stays behind `GGML_SYCL_IQ3_GROUPED_EXPERIMENTAL` + `GGML_SYCL_IQ3_GROUPED=1`, default off. Evidence `results/2026/09/b70-improvement-probes-20260920/kernel-iq3-e0a/`. |
| 2026-09-23 | **SPEC DONE → kernel IN FLIGHT** | `worklogs/KERNEL-grouped-iq3-spec-20260923.md`: the design was started from what the pinned tree already contains — untracked drafts `ggml/src/ggml-sycl/mul-mat-id-iq3.cpp/.hpp` + `iq3-xxs-reconstruct.hpp` (joint_matrix FP16, tile M16/N64/K256, 128 work-items, 8×16 subgroups, 42,176 B SLM) with a reconstruction verified bit-faithful against `dequantize_block_iq3_xxs` (`ggml-quants.c:2575-2603`) — adopted as **E0a**. Record corrections: **IQ3_XXS is 3.0625 bpw / 98 B per 256-block** (the "4.25 bits" noted earlier is IQ4_XS) ⇒ gate/up traffic 693.6 MB IQ3 vs 3.62 GB FP16; the 28.87 ms is one [4096→2048] op the layer runs **twice**; `test-backend-ops -o` is exact-match and `-p` is the vars() regex. Route decision: FP16-class XMX (INT8 dies on both the ≤1e-5 L2 gate and the rate-class law). Hook placement `ggml-sycl.cpp` after the `ne12==1` mmvq block (~4789) and before the `ids_host` memcpy — removes the prefill host sync as a side effect. E0a predicts **6–9 ms (3.2–4.8×)**; the one bounded revision (TN 64→128 + FP16 activation pre-convert) predicts 3–4.5 ms; honest floor ≈2–2.5 ms. The ≥2× isolated gate (≤14.4 ms) is kept. Implementation + parity ladder + isolated gate dispatched. |
| 2026-09-21 | attribution step complete — premise holds with numbers | Operator cost at the prefill regime (216 experts, top-6, three reps): gate/up **28.87 ms/op (1.79 TFLOPS)** and down **29.91 ms/op (1.72 TFLOPS)** at n=512 (recapture `results/2026/09/b70-improvement-probes-20260920/perf-k216-prefill-n512.txt` as re-run 2026-09-21; supersedes the earlier 28.28/30.02 capture), versus 6.3/6.4 ms (≈0.5 TFLOPS) at n=32. Per layer that is 87.7 ms of routed projections, ×43 layers = **3.77 s per 512-token ubatch against a measured 5.06 s** — so the routed MoE path is essentially the whole prefill budget, with no large unattributed remainder. The decisive number is the **~25× efficiency gap** to a comparable dense GEMM on this card (~45 TFLOPS at the same ubatch, same harness) while weight bytes favour IQ3 — i.e. the gap is IQ3→FP16 materialization plus per-expert structure, exactly the phase's target. **Kill criterion (withdraw if grouping+expansion < ~30 % of prefill) is NOT triggered.** Remaining work is the kernel itself (device-side grouping + tile-local reconstruction) plus parity gates. Evidence: `results/2026/09/b70-improvement-probes-20260920/perf-k216-prefill-n512.txt`. |

## Why this decision (the reasoning)

**The thesis: the prefill wall is a documented host-side orchestration problem with a
scoped ~2× kernel opportunity, and it deliberately reuses P1's primitives rather than
duplicating them.**

1. **The mechanism is known.** Prefill today = host counting-sort grouping + IQ3→FP16
   expansion + oneMKL GEMM. The ubatch sensitivity is already proven: ub512 → 1024
   delivered ~1.26–1.30× *without any new kernel* — evidence that the grouping/expansion
   path, not the GEMM math, is the constraint. The remaining ~2× is scoped as a grouped
   IQ3 kernel (`research/2026/dsv4-k216-prefill-ubatch-20260825.md`).
2. **Why this is not already done (the honesty note).** The ledger's validated ubatch
   improvement is *not* proof a grouped-IQ3 kernel exists — nobody has built it. The E0
   target (~1.5–2.0× isolated prefill) is conditional on the P0 trace confirming that
   grouping/reconstruction/launch dominates. No HTTP TTFT multiplier is assumed: TTFT
   includes tokenize/socket/scheduling, and the repo's naming rules forbid conflating
   cold-input rate with engine-native pp.
3. **Why after P1, not parallel.** The device-side grouping/offsets machinery, the IQ3
   reconstruction-in-kernel arithmetic, and the expert-shape knowledge are the same work
   products. P1 establishes them at M=1 where correctness is easiest to gate; P3 reuses
   them at large M where tiling policy (per-expert token counts, tails, empty experts)
   is the new design surface. Doing prefill first would mean debugging reconstruction
   arithmetic and tiling policy simultaneously.
4. **Why "tile by per-expert M" is a design rule, not a detail.** A p8192 prompt does
   not make per-expert GEMMs large: 216 experts × top-6 routing spreads tokens thinly,
   so expert-M can stay tiny even at long prompts. Inferring "compute-bound" from global
   prompt length is the exact trap the rate-class law warns about; tile selection must
   follow the *measured* per-expert-M distribution (P0 probe).
5. **Why full-model FP16 expansion is banned.** The artifact is ~83 GiB; expanding
   weights resident would violate VRAM safety (AGENTS §5 hard-crash class for >30 GB
   full-offload) and destroy the capacity point of the model. Reconstruction must be
   tiled/on-demand.

**Alternatives considered and rejected.**

- *Bigger ubatch / more oneMKL tuning.* Config-space; the 1024 result already harvested
  most of it; the remainder is the kernel (scoped, not measured).
- *Prefill via the fused decode kernel at large M.* Decode and prefill are different
  algorithms (MMVQ vs grouped GEMM); sharing one kernel is the P0 dispatch-separation
  anti-pattern in reverse.
- *CPU-side grouped prefill (host sorting retained, faster sort).* Leaves the expansion
  copy and launch pattern on the critical path; same non-fix class as P1's host-loop
  alternative.

**What would change this decision.** P0 attribution showing grouping/reconstruction is
<~30% of prefill time (then the 2× is not there — withdraw); upstream oneDNN/oneMKL
landing a grouped IQ3 path (collapse to validation + benchmark); K216 being retired from
the fleet (drop the phase, record why).

## Objective

Replace host grouping and IQ3-to-floating orchestration with device-side grouping and
tiled reconstruction/compute. Reuses P1's format knowledge and device-routing primitives;
treats prefill as a separate algorithm.

## Baseline and E0 target

Historical ~105 pp at established ubatch 1024
(`research/2026/dsv4-k216-prefill-ubatch-20260825.md`).

E0 target: ~1.5–2.0× isolated prefill if the targeted grouping/reconstruction/launch
path dominates. No HTTP TTFT multiplier is assumed.

## Engineering work

1. Measure host counting-sort, reconstruction, GEMM, and synchronization time separately
   (P0 trace).
2. Build device-side token-to-expert grouping and offsets.
3. Reconstruct IQ3 tiles on demand — never whole-model expansion.
4. Feed reconstructed tiles to appropriate floating-point DPAS/GEMM.
5. Select tiles by per-expert token count.
6. Support tail experts, skewed routing, empty experts.
7. Compare persistent work scheduling vs simpler grouped launches.
8. Preserve existing ubatch and placement in the first A/B.
9. Tune ubatch only after the new algorithm is isolated.

## Validation gates

- Grouping equivalence and stable expert weighting.
- Numerical parity across tile boundaries and tails.
- No token loss, duplication, or expert permutation error.
- Model logits and quality across long prefill + decode.
- Workspace and VRAM accounting under the established offload budget.

## Benchmark plan

- llama-bench pp2048 and pp8192 with exact emitted metrics.
- HTTP p2048/g1 and p8192/g1 separately labeled.
- Decode p512/g128 and p8192/g128 regression checks.
- n=3 screening; n=5 finals. Fixed ubatch 1024 initially; separate declared ubatch
  screen only if warranted.

## Risks and stop criteria

- Stop if targetable work cannot support the projected gain.
- Stop after two correctness failures.
- Reject full-model FP16 expansion.
- Reject a prefill improvement that causes unacceptable decode or capacity regression.
- Do not infer "compute-bound" from global prompt length; inspect expert-M.

## Deliverables

- `research/2026/k216-grouped-iq3-prefill-<YYYYMMDD>.md`
- `results/2026/<MM>/k216-grouped-iq3-prefill-<UTC>/`
- Ledger entry linking RL-011 without rewriting its meaning.

## Cookbook integration

Add separate engine-native pp and HTTP cold-input records. Update the K216 recipe's
ubatch guidance only after matched evidence. Never combine those timing definitions in
one ranking field.
