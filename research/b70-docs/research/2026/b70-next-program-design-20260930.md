---
title: "Next-program design — the M=4 verify-pass program (VERIM4), gated build spec"
type: research
date: 2026-09-30
status: provisional
author: "substitute b70 silicon architect (CPU/source analysis only — no GPU, no builds, no model loads)"
scope: "Design successor to research/2026/b70-architect-postmortem-20260930.md. Picks ONE primary program for EXL3 Qwen3.8-27B single-card decode and specs it to buildable level. Evidence: RL-090/098/100/101/102/111, C2-V2, pinned upstream tree 15ded2f3."
publication_gate: PROVISIONAL_NOT_FOR_PUBLIC_HEADLINE
---

# Design: VERIM4 — M=4 verify-pass bandwidth recovery

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Private design note. No shared
authorities edited, no commits, no GPU work performed.

## 0. Decision

Primary program: **(a) the M=4 verify-pass rewrite on the K=4 path** — the only
candidate with a ceiling that matters, and the gate under which every other
program (K3 route, residual work) inherits its kernel basis.

| candidate | ceiling (tok/s) | basis | cost | verdict |
|---|---|---|---|---|
| (a) M=4 verify pass → floor | **~76–79** best / ~70–72 expected | RL-098 §6: 34.46→26.5 ms pass; round 47.2→38.1/39.2 | days, kernel work | **PRIMARY** |
| (b) K3 3bpw route | ~90–95 only *after* (a) lands; standalone: cannot reach 63.25 (§2 arithmetic) | RL-100 42.44 measured; needs 1.7× DPAS-K3 kernel speedup | same kernel work as (a) + more | gated behind (a) |
| (c) residual split + cheap flags | ~65–68 aggregate (independent, additive) | ~6.6 ms residual + flag cells | hours | parallel cheap cells (§3) |

Why (a) and not (b) as the primary: K3's loss is the *same* decode pipeline at
worse geometry (Geo<3> does ~4× per-lane decode work; RL-100 §verdict). Fixing
the K4 M=4 decode path is a strict prerequisite to making K3's −17 % bytes
real, and (a) alone already has the bigger unconditional ceiling. Why not (c):
it is real but bounded (~+2–4 tok/s) and runs in the same window cheaply — it
is a Stage-0 sibling, not the program.

## 1. The program

### 1.1 Named floor and priced slack

- Floor: **521 GB/s measured-achievable read** (`sum()` reference, RL-098 §4),
  not the 608 GB/s pin. Target = 95 % of achievable = 495 GB/s → **26.50 ms**
  for the 13.126 GB M=4 pass; stretch = 85 % pin (516.8) → 25.38 ms.
- Current M=4 pass: **34.46 ms = 381 GB/s** (RL-098 §5). Slack 7.95–9.07 ms.
- Decomposition bounds already measured/estimated:
  - had_in+had_out scaffolding: **1.4–2.1 ms/pass** (SKIP_GEMV guard build, RL-102 §(ii)).
  - trellis-decode issue cost: **~4 ms at M=4** — upstream's own
    `EXL3_DEBUG_B_FAKE` whole-step probe: 30.7→26.6 ms (PROGRESS.md "Timing
    probes"; also M=16 ~6 ms, M=32 ~15 ms). *Not yet reproduced per-family on
    the pinned tree.*
  - unexplained "M≥3 DPAS-class" residual: ~2–3.5 ms — what the NT sweep and
    MB4 falsifications left on the table (RL-101/102).

### 1.2 Why the three dead levers could not move it

- **MB=4 padding (P6/RL-101):** bit-identical outputs; +0.41/+4.72 pin-pts vs a
  ≥5 gate. Padding was never the mechanism — the M=4→M=1 regression survives
  it untouched.
- **NT strip sweep (RL-102):** launched threads are pinned ≈1408 by
  `g_target_threads_mb16` (`exl3_ops.sycl:178`) regardless of NT; the sweep
  moved per-thread strip width at constant parallelism — the plateau is
  geometry-insensitive. Also rules out outstanding-bytes limitation: NT8 at
  MB4 doubles per-thread in-flight bytes (1 KB/row) and was flat.
- **Register binding (C2, C2-V2):** the B-register WAR/scheduling theory is
  dead three ways — upstream `EXL3_BV_ALL` no-change, C2 falsified at ISA
  (reused bank r17), V2 parity-pass but −3–4 % (allocator freedom removed).

What remains standing after all falsifications: the per-K-row serial
`load → build_k4 (≈300 SIMD32 instr/row, ~9 SIMD32/32-value group,
PROGRESS.md:271-273) → Bv materialization → dpas` issue stream on the Vector
Engine pipe, on ~1408 threads with no software pipelining
(`load_words`, `exl3_esimd.h:154-169`; loop at `exl3_esimd.h:766-837`).

### 1.3 Stage 0 — the attribution battery (buildable ≤1 day, flag-only)

All arms are `EXL3_FLAGS` variants on the retained tree — no code edits:

```
tree:    /mnt/<nvme-disk>/b70-k3-build-20260929/src  (15ded2f3 + 0001..0007)
daemon:  unix:///run/<exl3-docker>.sock (isolated; never main dockerd)
build:   in-container:  EXL3_FLAGS="<flags>" bash scripts/build_ext.sh
         → exl3xpu/_C.so (~1 min warm); docker commit → tagged variant image
harness: results/2026/09/b70-b6-microprobe-20260929/harness_graphprobe.py
         (graph-replay medians; --entrypoint python3; artifact :ro)
cells:   M=4 (and M=1 control) on down_proj, gate_up_fused, qkv_fused,
         in_proj_qkvz, out_proj, lm_head  (~10 min GPU per image)
```

| arm | flags | isolates | read |
|---|---|---|---|
| control | (none) | incumbent | RL-098 table; re-run fresh |
| t_BFAKE | `-DEXL3_DEBUG_B_FAKE` | loads+dpas+partials, **no state-build, no codebook** (`exl3_esimd.h:813-818`) | `t_full−t_BFAKE` = total decode+build issue |
| t_NODECODE | `-DEXL3_DEBUG_NODECODE` | state-build kept, codebook `dec()` removed (`:42-44`) | `t_BFAKE... t_NODECODE` splits state-build vs mul+dp4a+mad |
| t_AONCE | `-DEXL3_DEBUG_A_ONCE` | activation reload removed (`:793-797`) | Am L1-traffic share at M=4 (upstream: ~0 at M≤32) |
| t_SKIP | `-DEXL3_DEBUG_SKIP_GEMV` | had_in+had_out only | already built (`k3-p6-hadguard`); re-run for in-window |
| t_PF256 | `-DEXL3_DPAS_LARGE_GRF -DEXL3_DPAS_PREFETCH` | the never-tried combination: 256-GRF (`exl3_ops.sycl:113-122` `large=true` for all MB) + double-buffered word loads (`:768-781`) | the memory-side falsifier (§1.5) |
| t_FOLD | `-DEXL3_FOLD` | codebook affine folded out of decode loop, MB≤8 (`:614-617`) | proven +3.5 % upstream; needs Gate-A1 extension |

**Decision rule (pre-declared):**
- If `t_full − t_BFAKE ≥ 20 %` of pass per family → decode-issue owns it → §1.4.
- If `t_BFAKE` lands within ~10 % of the 26.5 ms line (i.e., memory core at
  floor) → the paper slack was floor mythology; program collapses to FOLD +
  any `t_PF256` win → declare M=4 at floor, pivot to K3/bytes.
- If `t_PF256` alone clears ≥5 % on down_proj → memory-latency branch (§1.5)
  regardless of the B_FAKE split.

All debug arms produce **wrong results by construction** — timing probes only,
never serving candidates; record as such in every JSON.

### 1.4 Stage 1a — issue branch: the decode-pipeline restructure

Exact structural change (in `DpasKernel<4,2,MB,NT>`, `exl3_esimd.h:591-861`):

1. **FOLD on** — moves the codebook multiply (`st*0x83DCD12D`, `dp4a`,
   `decode_cb_h:46-60`) out of the per-group inner loop into a one-time
   accumulator epilogue (`:831-842`). Removes ~3 of ~9 SIMD32 ops per 32-value
   group. Already coded and scoped `CB==2 && MB<=8 && !FUSED` — the only work
   is the M24/40 gemm_raw anomaly: it never executes at C1 decode shapes, so
   scope FOLD to `M<=8` production dispatch and keep the flag documented
   (it is already `MB<=8`-gated; verify no production path reaches FOLD at
   MB>8 — `fused_small` is off by default).
2. **Kill the 16→32-lane duplication moves** — `build_k4` (`:669-716`)
   materializes `cw/pw` as two select-copied halves (~2 SIMD32-equivalent
   moves per m per q1 per the upstream ISA read). Replace with a strided
   `replicate_vs_w_hs<8,4,2,0>` read feeding `dec<16>` twice at half-width
   (the K4_HALVES geometry) — **but** K4_HALVES produced wrong outputs at
   M=3..8 (PROGRESS.md:275-277); the repair is to keep its *reads* and fix
   only the Bv write offsets: lane l of half h must land at
   `Bv[32m + 128q1 + 16h + l]`, not the layout K4_HALVES used. Cheapest
   correct variant: keep the 32-lane `st` combine, replace only the two
   select-copies with `select<16,1>` views over a single 16-lane combine.
3. **Two-K-row issue batching** — unroll `r` by 2 (loads already independent;
   `load_words` has no carry dependency across rows) so one issue quantum
   covers 1 KB of trellis words at NT=4, matching the M=1 GEMV's per-iteration
   byte depth — the NT8-flat datum says this alone is not sufficient, so this
   is second-order behind 1+2.
4. Keep `dpas<8,RC>` per tile; do NOT re-open BV_ALL, fused kernel, SLM
   staging (upstream priced SLM decode-sharing at ~4× A-operand traffic —
   recorded rejection, not a candidate), or the vector op-class swap (measured
   250-265 GB/s at M=4, 25-30 % worse — RL-102 guard i).

### 1.5 Stage 1b — memory branch: 256-GRF + prefetch pairing

`EXL3_DPAS_LARGE_GRF` exists and applies `grf_size<256>` to every MB
(`exl3_ops.sycl:113-125`); `EXL3_DPAS_PREFETCH` exists (`:766-781`). Each
rejected *alone* (256-GRF: occupancy halves to ~1024 with nothing to spend the
registers on — 44.4/53.7 vs 35.1 ms; prefetch: spills at 128 GRF — 70.0 ms).
**Together they solve each other's defect** — the untested cell in the grid.
If `t_PF256` clears ≥5 % on one paired down_proj cell, extend to all families;
flat-or-worse closes the memory branch permanently.

### 1.6 Tile/schedule geometry + paper budget (MB=8, NT=4, 1 GRF = 32 B)

Per-thread live state per K-row iteration:

| structure | bytes | 128-GRF (4 KB) | 256-GRF (8 KB) |
|---|---|---|---|
| acc fp32 MB×16×NT | 2048 | 64 | 64 |
| words+prevs (NT×W dwords ×2) | 1024 | 32 | 32 (64 w/ prefetch) |
| Bv fp16 256 | 512 | 16 | 16 (128 w/ BV_ALL) |
| Am fp16 MB×16 | 256 | 8 | 8 |
| dw/dwprev tile selects + indices/addr | ~400–600 | ~15 | ~15 |
| **total** | **~4.3 KB** | **spills on any addition** | **~5.2 KB — headroom** |

- MB=4 variant: acc halves to 1 KB → ~3.3 KB — explains why the MB4 build had
  room and still did not win (the bottleneck is not capacity at MB8 either;
  it is the serial issue chain — pending Stage-0).
- SLM: zero today (work-group local=8, no SLM use). Any producer/consumer
  SLM redesign needs ≥2×8 KB per WG — fits 64 KiB/XVE trivially, but the
  operand-traffic arithmetic already kills it (§1.4.4).
- L2: 13.1 GB/pass streams ≫ L2; only the ~2.6 MB k/v_proj cells cache
  (RL-098 §3 finding 2). No L2 levers at verify-pass scale.
- In-flight bytes: NT4×512 B/row ×1408 thr ≈ 0.72 MB outstanding — NT8 probe
  already pushed this to ~1.4 MB and was flat → outstanding-bytes is not the
  binding constraint. Consistent with issue-bound leaning.

### 1.7 Capture-safety

All Stage-0/1 variants change kernel internals only — same op topology
(had_in → dpas → had_out), same launch counts, no new allocations or host
waits → capture-safe by construction. Gates that must still pass: D3a/D3b
replay hash + spec-decode counter continuity on the serving A/B (the C1-R4
failure mode was a host-wait *addition*, which none of these arms introduces).
FOLD changes epilogue math on the XMX pipe — same kernels, extra dpas on
`accs`; still capture-safe.

### 1.8 Ceiling (honest branches)

| outcome | recoverable | round 47.2 → | tok/s |
|---|---|---|---|
| everything to 95 % achievable | 7.95 ms | 39.2 ms | ~76 |
| stretch to 85 % pin | 9.07 ms | 38.1 ms | ~78–79 |
| **upstream split reproduces** (decode ~4 + FOLD ~1.2 + partial hads) | ~4–5 ms | ~42–43 ms | **~70–72** |
| floor-only (B_FAKE ≈ floor, PF256 flat) | ~1.2 ms (FOLD) | ~46 ms | ~64–65 |

### 1.9 Gate ladder

1. **Isolated-op:** ≥5 % weighted M=4-pass reduction vs in-window control
   (`harness_graphprobe.py`, n=3 screens, fresh n=5 finalist, family order
   fixed, ±5 % session drift recorded).
2. **Numerics:** extended Gate-A1 bit-exact at M∈{3,4,8} weight-level
   (`test_bitexact_xpu.py` pattern — the K4_HALVES lesson: a −4 % win with
   wrong outputs is a loss); exact-answer battery before any speed claim.
3. **Capture contract:** D3a/D3b replay equivalence + zero new allocs/waits.
4. **Serving A/B:** matched n=5 p512/g128 cells on derived image, 230 W as-
   found caps, prefix-cache state declared, spec counters reconciled, byte-
   identical greedy outputs; acceptance Δ ≤ noise. Adoption gate ≥5 % e2e.

### 1.10 Resource budget

GPU0 only (GPU1 belongs to another lane), isolated dockerd, artifact `:ro`.
Builds: 6 flag images × ~1 min warm (first cold build ~15–20 min), 8 CPU
threads. Probe windows: ~10 min/image ×7 ≈ 1.5 h GPU for Stage 0; Stage-1
iteration ~4–6 h GPU across days. Disk: derived `_C.so` deltas ~200 MB each
on the large mount — never into main dockerd. RAM <4 GB.

### 1.11 Most likely failure mode (stated in advance)

**B_FAKE lands within ~10 % of the 26.5 ms achievable line** — i.e., the
memory core is already saturated and the 9.07 ms slack was priced against a
floor no kernel reaches. Upstream's own whole-step number predicts exactly
this (decode ≈4 ms of 30.7). The program's real recoverable pool is then
~4–6 ms issue-side, ceiling ~70 tok/s, and the correct action is to take
FOLD + the state-build cut, declare M=4 at floor, and pivot the kernel budget
to K3 — not to keep hunting the imaginary last 3 ms.

## 2. Candidate (b) — K3 3bpw route, spec-level (gated behind §1)

**What must be true for 3bpw to beat 63.25.** K3 measured 42.44 → round ≈
69.6 ms at yield ~2.95. Non-verify terms ≈ 47.2−34.46 = 12.7 ms ⇒ the K3 M=4
verify pass is ≈ **56.9 ms for ~10.9–11.3 GB (≈195 GB/s = 0.51× of K4's
per-byte rate)**. To beat 63.25 the round must reach ≤47.2 ms ⇒ K3 verify ≤
**34.5 ms ⇒ ≥327 GB/s — a 1.7× DPAS-K3 kernel speedup**. Cause is structural:
`Geo<3>` gives V=32/G=8 (vs K4 V=8/G=32) → ~4× per-lane decode work per
weight stream + `planarize` per row (`exl3_esimd.h:28-35,198-205`).

**Hard truth for the sweep spec:** `EXL3_K3_NT_MR1/2/4/8` tune `nt_of(3,MR)`
— the **vector GEMV** path, which serves only M≤2 (the draft forwards, ~3 ms
of the 69.6 ms round). P2 cannot reach the dominant term. The K3 program that
matters is a `build<0>`/planarize restructure for K=3 in `DpasKernel` — the
same skill package §1.4 exercises on K4 first.

Build coordinates (P2-SWEEP-SPEC): base 15ded2f3 + patches 0001–0006 (0007 is
the MB4 patch); flags `EXL3_FLAGS="-DEXL3_K3_NT_MR2=8"` etc. Order: (1)
control [8,4,4,4] rebuild; (2) **MR1=4** (the serving-relevant draft shape);
(3) MR2=8; (4-5) MR4=8, MR8=8 diagnostics (P3 raw only, never serving). Then
P3 zero-tolerance reconstruction vs `exl3_ops.sycl:545` reference, n=3
screens, P4 matched A/B needs ≥69.6 tok/s target — unreachable via P2 alone;
P2's honest role is regression-screening the draft steps while §1 lands.

## 3. Candidate (c) — parallel cheap cells (same window, independent)

- **P-B probe:** matched MTP1 vs MTP3 serving cells (n=3, 230 W) — first split
  of the ~6.6 ms non-EXL3 residual; Δround/2 ≈ one in-situ draft forward.
- **Draft head 512→256 blocks** via bind-mount over in-image `draft_vocab.json`
  (env override is void — recipe env wins; bind-mount is the mechanism):
  −~0.38 GB/round ≈ +1–1.5 %, gate = acceptance Δ ≤ noise + greedy byte-equal.
- **k=4 speculative depth** serving-flag A/B on pinned image — never measured
  this build generation; upstream think-on showed +8 %, think-off a loss.
  One matched n=5 cell.
- **EXL3_FOLD** is covered inside §1.4.1 (it is a Stage-1a component, not
  optional polish).

None of these beat (a)'s ceiling; all are worth their ~1–2 % each as
independent additives while kernel work runs.

## 4. The blunt question, restated

>2× single-card (>126.5 tok/s) needs round ≤23.3 ms; weight bytes alone are
25.2 ms at the *measured* achievable floor before attention/draft/sampler.
Single-card tops at ~76–79 (VERIM4 full success), ~90–95 (+ K3 + residual).
Past ~95 is the unbuilt custom TP2 path (model bound 127 tok/s @60 µs
barrier) or nothing. VERIM4 is the right program *because* its ceiling is
real and its day-1 cost is six environment variables.

## 5. Do-not-retry (closed; evidence in post-mortem §4)

MB4 padding, NT axis, fused single-kernel, staged/serial reduction, register
binding, LUT/MUL16/HALVES/BV_ALL/DECODE_INLINE, 256-GRF alone, prefetch alone,
vector op-class at M=4, Triton backend, W4A16 class, stock TP2, draft-vocab
env override, flag sweeps beyond measured plateaus.
