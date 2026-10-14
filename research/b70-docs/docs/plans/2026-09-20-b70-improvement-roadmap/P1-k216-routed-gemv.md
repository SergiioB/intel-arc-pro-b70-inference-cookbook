---
type: plan
date: 2026-09-20
phase: P1
title: "K216 smallest routed-IQ3 GEMV repair (device-resident expert IDs)"
status: WITHDRAWN
opened: 2026-09-20
updated: 2026-09-20
ledger: RL-030
depends_on: [P0]
effort: 4–8 sessions
---

# P1 — K216 smallest routed-IQ3 GEMV repair

## Status log

| Date | Status change | Note / evidence |
|---|---|---|
| 2026-09-20 | filed as PLANNED | Roadmap filed from b70-architect draft |
| 2026-09-20 | PLANNED → RESEARCH (P0) | P0 probe batch opened; operator A/B designed at K216 shapes. Worklog `worklogs/P0-20260920.md`. |
| 2026-09-20 | RESEARCH → BLOCKED | **Projection gate FAILED, target withdrawn; phase blocked on the absent artifact.** The repair already exists uncommitted in `~/llama.cpp` (written 2026-08-26, built, never measured). Shape-matched operator A/B (clean `1692f9e50` vs patched, 3 reps, GPU 1, 216 experts/top-6/h=4096/ffn=2048): device-resident IDs save **32.57 µs** (gate/up) and **32.95 µs** (down) per op at decode, i.e. **2.75 ms/token of 112.74 ms = 2.44 % ≈ 1.025×**, versus the required 62.74 ms for 20 tok/s. The ≥10 % engineering gate is not met by this scope. Batch (n=32) is unchanged (the fused gate is `ne12 == 1`). Blocker: the 83 GiB K216 GGUF is **ABSENT host-wide**, so no matched end-to-end confirmation is possible in this window. Evidence: `research/2026/b70-improvement-probe-baseline-20260920.md` §2; `results/2026/09/b70-improvement-probes-20260920/operator-ab-summary.json`. **Re-scope recommendation:** keep the repair (correct, parity-passing, and it is the prerequisite that unblocks SYCL graph capture for `MUL_MAT_ID` — `ggml-sycl.cpp:5687`), but move the speed hunt to the measured kernel-efficiency gap: the fused decode op runs **5.52× off the 608 GB/s weight-read floor**, and the non-fused path materializes each expert as FP16 (≈100 MB/op ≈ 5.2× the IQ3 bytes) — that is P3/P5 territory, worth up to ~12 ms/token. |
| 2026-09-21 | BLOCKED → WITHDRAWN (projection refuted) | The 83 GiB artifact was **re-downloaded**, so the end-to-end gate ran after all: **ABBA A/B, n=6 per build, L5 recipe verbatim, both cards, 230 W** → clean (byte-export of `1692f9e50`) **9.25** vs patched (device-ID IQ3 cases) **9.577 engine tok/s = +3.54 %**. Parity holds, prompt processing unchanged, distributions nearly disjoint. **Gate result: ≥10 % FAILS; the 20 tok/s aspiration (needs +125 %) is refuted.** Per the stop rule ("a P0 trace showing <10 % removable → withdraw the phase, record evidence, move on") the phase is withdrawn **with the repair kept** — it is correct, it costs nothing, and it is the prerequisite that unblocks SYCL graph capture for `MUL_MAT_ID` (`ggml-sycl.cpp:5687`). The speed object passes to the measured kernel gap (**5.52× off the 608 GB/s weight-read floor**; FP16 materialization ≈100 MB/op): prefill → **P3**, decode GEMV → **P5 secondary**. Refreshed baseline for matched use: **9.25 tok/s clean** at this cell (recorded anchor was 8.871 on the old artifact/disk). Evidence: `results/2026/09/b70-improvement-probes-20260920/p1-k216-e2e-ab-20260921T081155Z/`; worklog `worklogs/P0-20260920.md` addendum. |
| 2026-09-21 | claims audit — attribution qualified | The same audit confirms the **ABBA design is real** (clean/patched/patched/clean, fresh loads, ≤55 °C cooldown between arms), that `engine_tg` is the engine's own `predicted_per_second`, and that no naming rule was violated (client and engine rates are kept in separate fields). But the **patched build carries 283 insertions across 12 files**, so the headline belongs to *tree-vs-tree*, not to the device-ID cases alone — the single-variable attribution rests on the separate operator A/B (+2.44 %). Minimum fix if a single-variable e2e number is ever wanted: rebuild with `mmvq.cpp` only. The ≥10 % gate FAIL and the withdrawal stand regardless. |

## Why this decision (the reasoning)

**The thesis: this is the fleet's largest *structural* loss, and it is a kernel-shaped
problem that no configuration change can reach — but the headline target must be
re-derived from a trace before anyone builds anything.**

1. **The wall is architectural, not tunable.** IQ3_XXS is omitted from the fused SYCL
   `MUL_MAT_ID` MMVQ path, so every routed projection D2H-copies ~24 B of expert IDs and
   `stream->wait()`s (~84 waits/token on GPU-routed layers), then the host launches tiny
   GEMVs. The entire config space around it is already closed by the tried-matrix:
   `-ts 46,54`/`50,50` low-ncmoe, `-sm row` (segfaults), single GPU, GPU-resident
   Unsloth DSpark (256-expert dflash on L3/L5), `ggml_sycl_supports_mmq()` flip, SYCL
   graphs on dual+`MUL_MAT_ID`, `--no-mmap`/`--mlock`, q4_0 K, spiritbuun
   `-ot 'exps=CPU' --moe-cache on`. When every flag is exhausted, the remaining lever is
   the kernel itself.
2. **The honest budget math (why the entry gate is ≥10%, not 2.25×).** 8.87 tok/s =
   112.74 ms/token. 20 tok/s needs 50 ms/token → 62.74 ms saved → ~747 μs across each of
   84 sites. The copied payload is ~2 KB/token — bandwidth is irrelevant; what matters is
   *critical-path time*: exposed sync + launch serialization + tiny-kernel
   inefficiency. If exposed sync is ~50 μs/site, removing all of it buys ~1.04×. So the
   architectural range recorded in AGENTS §16 (~9.6–18 tok/s with fused IQ3 MMVQ and a
   compatible draft) is the realistic envelope, and 20 tok/s is an *aspiration retained
   only if the P0 trace supports it*. The engineering gate is therefore: **≥10%
   matched end-to-end gain with zero intermediate host expert-ID reads** — a target that
   is defensible from the known mechanics alone.
3. **Why the *smallest* repair, not the persistent megakernel.** The old roadmap's P1
   was an on-device router + fused `mul_mat_id_fused_moe_sycl` + (eventually) graph
   capture. The architect demoted this to a sequence for three reasons: (a) a narrow
   IQ3_XXS operator that accepts device-resident IDs **reuses the existing IQ3
   reconstruction arithmetic and scale semantics**, so parity risk is minimal and
   falsification takes one session; (b) a full fused gate+up+down persistent kernel
   changes many variables at once (work scheduling, SLM staging, fusion boundaries) —
   if parity fails you learn nothing about which piece broke; (c) graph capture on
   dual+`MUL_MAT_ID` is on the recorded do-not-retry list and needs a changed, testable
   hypothesis (P0's capture probe) before re-entry.
4. **Why placement stays frozen.** Dual split + `-ncmoe` CPU-expert placement is
   capacity-motivated and validated. Changing placement and the kernel in one A/B
   destroys attribution (one variable per gate). If the trace shows CPU-expert time is
   the dominant floor, the correct response is `WITHDRAWN` for the 20 tok/s aspiration —
   not a placement sweep smuggled into a kernel phase.

**Alternatives considered and rejected.**

- *Fused MoE megakernel first (old-roadmap P1).* Rejected as first move: too many
  variables, RL-021-shaped risk. It remains the *second* increment inside this phase
  after parity (see work breakdown step 8).
- *SYCL command graphs (`GGML_SYCL_GRAPH=1`).* Blocked by recorded dual-card +
  `MUL_MAT_ID` failures; revisit only with the P0 capture evidence.
- *Switching K216 to a different quant format (EXL3-class) to dodge the IQ3 kernel
  gap.* Deferred with conditions — see DEFERRED-AND-TRACKED.md. The artifact is already
  ~3 bpw-class; another ~3 bpw format does not change placement.
- *CPU-side fix (batch the host loop).* Rejected: it leaves the D2H serialization on the
  critical path; any "fix" that substitutes host-side expert loops elsewhere is
  explicitly a non-fix.

**What would change this decision.** A P0 trace showing <10% of token time is removable
via device-resident IDs → withdraw the phase, record evidence, move to P2. Upstream
llama.cpp landing a fused IQ3 `MUL_MAT_ID` in the pinned lineage → collapse this phase
to a validation + benchmark of the upstream fix.

## Objective

Remove host expert-ID synchronization from the existing IQ3 routed decode path without
changing model representation, placement, or sampling.

## Baseline and E0 target

Historical: ~8.87 engine decode tok/s, established partial CPU-expert offload + dual-card
layer placement (`research/2026/dsv4-k216-request-to-output-pipeline-20260825.md`).

- Budget: 112.74 ms/token; 20 tok/s ⇒ save 62.74 ms/token (~747 μs/site over 84 sites,
  including launch serialization and tiny-kernel inefficiency — not merely the D2H copy).
- **Initial engineering target: ≥10% matched end-to-end gain, zero intermediate host
  expert-ID reads in the repaired path.** The 20 tok/s aspiration is retained only if
  tracing supports the necessary savings.

## Engineering work

1. Locate the pinned llama.cpp SYCL `MUL_MAT_ID` dispatch and IQ3_XXS exclusion.
2. Capture routed operator shapes, expert counts, strides, and current host waits (P0).
3. Add a narrow IQ3_XXS implementation accepting device-resident expert IDs.
4. Reuse existing IQ3 reconstruction arithmetic and scale semantics.
5. Repair one projection/operator boundary first — not a full-model persistent kernel.
6. Compare output against the old implementation.
7. Extend to all eligible routed projections.
8. Only after parity: consider gate/up fusion and device-side expert work queues.
9. Keep CPU expert placement and cross-card transfer behavior unchanged.
10. Defer graph capture until the repaired path is stable and the blocked interaction
    has a changed, testable hypothesis.

Code search targets: pinned `ggml-sycl` sources implementing `MUL_MAT_ID`, MMVQ, IQ3_XXS
reconstruction, routed expert launch loops. Record resolved absolute filenames before
editing.

## Validation gates

- Operator parity vs the old path and a high-precision reconstruction reference.
- Top-k ID ordering, expert weighting, repeated IDs if supported, boundary/empty experts.
- Deterministic replay.
- Model-level logit/KL comparison + representative task-quality checks.
- No new host synchronization hidden in allocator, routing, or reduction code.
- No unexpected resident-weight expansion.

## Benchmark plan

- C1 p512/g128 and p8192/g128, exact established sampling; engine timing primary,
  client timing separate.
- p2048/g1 and p8192/g1 regression checks; one fixed natural-text workload.
- n=3 screen, fresh n=5 finalist. Identical placement, context, KV, power cap, model
  bytes.

## Risks and stop criteria

- **Projection gate:** if measured removable time cannot support 20 tok/s, formally
  withdraw that target *before* implementation.
- Stop after two parity failures from the same design; redesign rather than loosen
  tolerance.
- Stop if CPU-expert time becomes the dominant floor and the ≥10% gate is unreachable.
- Stop if the implementation requires unsafe VRAM expansion.
- Reject any "fix" that substitutes host-side expert loops elsewhere.

## Deliverables

- `research/2026/k216-iq3-device-routed-gemv-<YYYYMMDD>.md`
- `results/2026/<MM>/k216-iq3-device-routed-gemv-<UTC>/`
- Ledger entry linking RL-010; source patch, operator tests, trace comparison, matched
  final card, cleanup evidence.

## Cookbook integration

After claims audit: update the K216 llama.cpp SYCL recipe with exact build revision,
patch status, placement, and measured limits. Eligible catalog records: C1 decode and
prefill-regression cells with explicit dual-card/CPU-offload topology — never labeled
single-card or GPU-only.
