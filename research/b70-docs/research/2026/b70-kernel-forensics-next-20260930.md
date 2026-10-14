# B70 EXL3 kernel forensics — what failed, why, and the next gated experiment

> **PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Private working note. CPU/source
> forensics only; no GPU was used for this document. Not a shared authority;
> do not cite from ledger/STATUS/cookbook surfaces.
> Author: substitute kernel lane (Xe2 decode-kernel engineer), 2026-09-30.
> Host: `b70-host` (local). Contract: AGENTS.md §0 research intent
> compiled, `halt=false`.

Scope: the EXL3 4bpw route's M=4 verify-pass in-kernel bandwidth loss and the
four kernel-side efforts that failed or landed inconclusive against it
(RL-098 microprobe → RL-101 MB4 → RL-102 NT sweep → RL-111 C1/C2 → RL-112 C2V2).
K=3 (RL-100) is included where it bears on the next-experiment choice.

---

## 1. Per-effort forensics

Conventions: "measured" = retained evidence; "hypothesis" = inference, marked
as such. All timings are GPU0 @180 W isolated-op graph-replay medians
(XPUGraph kcap=32, ≥100 replays) unless noted; C1/C2 ran GPU1 @230 W.

### 1.1 RL-098 — B6 isolated-op microprobe (validated, REDIRECT verdict)

Evidence: `results/2026/09/b70-b6-microprobe-20260929/RESULTS.md`.

- Intended: test whether the whole-round 57–63 % of pin loss (RL-090) is
  uniform in-kernel slack. Measured: M=4 verify-class pass = 34.46 ms over
  13.12 GB = 381 GB/s = 62.6 % of the 608 GB/s pin floor; per-family spread
  10.6–83.5 % of pin. lm_head K6 / in_proj_qkvz / gate_up are at 90–98 % of
  the *measured-achievable* 521 GB/s read reference — at ceiling. Mid-N
  families sit at 46–75 %; standalone [5120→1024] k/v GEMV is latency-bound
  at 11–13 % (~856 work items; engine never pays it — fused into qkv).
- Structural finding (measured): every M≤128 linear is `had_in →
  GemvKernel|DpasKernel → had_out` (`exl3_ops.sycl:267–316`); M≥3 dispatches
  `DpasKernel<K,CB,8,4>`, M≤2 `GemvKernel` (`g_vec_max_m=2`,
  `exl3_ops.sycl:171`); split-K P is set from `g_target_threads_mb16=1408`
  (`exl3_ops.sycl:177–178`, `:291–295`).
- The M=4→M=1 gap is 5.02 ms/pass (34.46 vs 29.44) on identical weight bytes;
  the to-85 %-pin gap is 9.07 ms — the priced ceiling for the whole kernel
  program is round 44.1→~35 ms = ×1.26 (63.25 → ~79 tok/s), not the
  85–95 tok/s framing.

### 1.2 RL-101 — P6 MB=4 dispatch (failed the ≥5-pin-pt gate on both families)

Evidence: `results/2026/09/b70-b6-mb4-ab-20260929/RESULTS.md`.

- Intended mechanism: MB=8 padding wastes half the DPAS rows at M=4;
  dispatch M∈{3,4} to `DpasKernel<K,CB,4,NT=4>` (RC=4, no zero-row work;
  `exl3_esimd.h:823–830`) and recover ≥5 pin points.
- Measured: numerics **bit-identical** ×4 cells; timing +0.41 pts
  (down_proj) / +4.72 (qkv-class) at M=4 — MISS on both, family-dependent.
- Microarchitectural read (hypothesis, consistent with ISA): wasted DPAS
  lanes were never the cost — dpas is only ~4 of ~336 instructions per
  row-iteration and the XMX pipe is not the wall; halving padded-DPAS work
  and halving the per-row activation load (256 B→128 B at MB4) moved
  nothing on the split-K-heavy family.
- Measured side-product that matters for every future arm: the M=4 DPAS
  class drifts ±5 % across sessions while M=1 vector reproduces within
  1–2 %; MB8 tails are heavy (p90/p50 up to ~1.4). All future M=4 claims
  need in-window paired controls in both orderings (the RL-102 rule).

### 1.3 RL-102 — NT strip-geometry sweep + guards (validated, family-confined)

Evidence: `results/2026/09/b70-b6-sweep-20260929/RESULTS.md`.

- Intended: assign the ≥~6.5 ms residual between strip geometry
  (`EXL3_DPAS4_NT` ∈ {2,4,8}, never swept) and a uniform M=4-intensity
  class term; measure the reduction stage directly.
- Measured: one family breaks the wall — in_proj_z at NT2, +10.5/+9.5 %
  GB/s both orderings (n_strips 96→192; bulk secondary only +4–5 %, hint
  band). down_proj REGRESSES at NT2 (−12.1/−4.6 %), lm_head (P=1) never
  moves ⇒ not split-correlated; it is small-N strip-starvation relief.
  NT8 is flat/regressing — and at MB4/NT8 the named register inventory
  (acc 64 + words/prevs 64 + Bv 16 + Am 8 GRF ≈ 152) exceeds 128 GRF, so
  that arm is spill-confounded, not a clean test.
- Guards (measured): forced-vector M=4 collapses to 250–265 GB/s, 36–40 %
  below the 404 GB/s M=1 vector rate — the M=4 4-row shape degrades
  *whichever* op class runs it (4× FMA work per weight byte for the vector
  path; `GemvKernel::step` exl3_esimd.h:470–495). had_in+had_out alone:
  whole-stage latency-floored — down 14.0 µs for ~1.8 MB moved (~130 GB/s
  effective), total ~1.42 ms/verify pass on measured families, full
  split-K/reduction term measured-bounded 1.4–2.1 ms (P-marginal slice
  ~0.5 ms).
- Reading: the residual is owned by the M=4 verify-shape kernel economics,
  not by op-class choice, strip geometry, or the reduction stage.

### 1.4 RL-111 — Stage C: C1 staged reduction, C2 K-pair schedule (failed)

Evidence: `results/2026/09/b70-resume-breakthrough-20260930T063200Z/C/`
(manifest-index.json, paired-complete.json, strict-D3-failure.log,
c1_r4.sycl, c2_kpair.sycl, C2-BOUNDS-SCHEDULE.md).

- C1 (staged/grouped fp32 reduction, `c1_r4.sycl`): intended to cut the
  had_out price. **Missed its gate twice over**: (a) strict D3 host-wait —
  covered official-loader tracing found inherited event host waits on BOTH
  arms during replay (`strict-D3-failure.log:20-26`); the candidate op was
  never capture-clean at the strict gate. (b) Diagnostic paired timing:
  serial→staged 12.16→12.34 µs one ordering, staged 11.66 vs serial 7.26 µs
  reversed — the ≥15 % improvement gate missed in *both* orders and bulk200
  regresses (9.08/7.51 vs 11.62/11.59). Why it missed (hypothesis grounded
  in the RL-102 guard): the reduction is launch/latency-floored, not
  bandwidth-floored — ~160–7,760 work items at ~130 GB/s effective; grouping
  P→⌈P/4⌉ adds a second launch + an intermediate round-trip to work that
  was already latency-bound, and the prize was only ever ~1.4–2.1 ms/pass.
  Also scope caveat: partials were synthetic down-shaped, not real DPAS
  partials.
- C2 (K-pair specialization, `c2_kpair.sycl`): process two K-rows per
  iteration with intended independent B banks. CPU index simulator PASS
  (25,600 windows, odd-tail rps=61/P=18 exact); native BMG-G31 compile PASS
  (128 GRF, no scratch/SLM, dpas.8x4 width-16 sites). **Cheap falsifier
  FAIL**: the compiler emitted the SAME B register (r17) for both paired
  dpas (`isa/c2/.text._ZTS7C2KPair.asm:239,320`) — the intended independent
  banking was never realized, so it never earned GPU timing.

### 1.5 RL-112 — C2 V2: ESIMD_REGISTER B-bank binding (validated-negative)

Evidence: `results/2026/09/b70-c2-repair-20260930T085200Z/v2/`
(RESULTS-V2.md, GATE-REPORT.md, C2-MILESTONE-CLOSE.json, isa-torch/).

- V1 was rejected pre-GPU: `ESIMD_REGISTER(n)` takes a BYTE offset — the
  authored (32)/(48) would have overlapped two 512 B simd<fp16,256> regions.
- V2 pinned B0/B1 at byte offsets 2048/3072 (GRF 64/96) on the K-pair path.
  Numerical gate PASS (bit-identical to stock `DpasKernel<4,2,4,4>` on all
  7 cells, capture/mutation clean, identical 1440-subgroup grid). Paired
  timing: down_proj 111.85 vs 108.41 µs (+3.17 %), qkv 76.23 vs 73.00 µs
  (+4.43 %) — SLOWER, non-overlapping. The earlier 106-vs-354 µs "incumbent
  gap" was two concurrent timing drivers — contamination, not signal.
- Microarchitectural reason it failed (ISA-level, measured):
  `isa-torch/.text._ZTSN4exl310DpasKernelILi4ELi2ELi4ELi4ELb0EEE.asm` shows
  the stock kernel already rotates B across **three** register regions
  (dpas at lines 241/369/396/458 read r51, r49, r49, r18) with A resident
  at r33. Pinning B to exactly two fixed banks *reduced* the allocator's
  freedom below what the compiler already did — WAR serialization per
  pinned bank (8 uses → 2 banks) plus extra mov pressure (274 vs 187
  static movs). Conclusion the record supports: B-operand banking was
  never the bottleneck; the compiler's allocation was already adequate and
  constraining it cost 3–4 %.

### 1.6 RL-100 — K=3 3bpw chain (validated, adoption NO-GO)

Evidence: `results/2026/09/b70-k3-completion-20260929/RESULTS.md`.
Numerics/capture/quality all pass; performance gates failed decisively:
42.44 vs 63.25 tok/s (0.671×, gate ≥1.10×) and cold-input TTFT 1.63×.
Acceptance counters unchanged (267/801/502 vs 267/801/506) → the loss is
kernel/round speed, not speculation. Reads ~13 % fewer bytes yet ~33 %
slower ⇒ the K3 kernel generation's per-byte efficiency is well below K4's —
the same per-byte-cost signature RL-098..102 keep measuring on the
incumbent.

---

## 2. The single most probable cause of the 37 % in-kernel loss on the M=4 pass

**Hypothesis (falsifiable):** the dominant term is *not* DRAM bandwidth and
not any memory-side quantity — it is the **per-byte instruction-issue cost
of the serial `load_words → build_k4 decode → dpas` inner loop**, which runs
at partial issue occupancy because nothing overlaps row r+1's memory work
with row r's compute. Stated precisely:

> The M=4 pass spends its time issuing ~0.5–0.7 SIMD8-class instructions per
> trellis byte (decode chain + operand-shuffle movs + region writes), and the
> measured 62.6 %-of-pin is the issue-bound steady state of that stream —
> latency exposure per row is the second-order term, DRAM bandwidth is not
> the binding resource.

Evidence for (measured / ISA):

- Stock `DpasKernel<4,2,4,4>` inner loop ≈ 336 instructions per 512 B
  weight tile-group (4 dpas + 3 sends + ~330 ALU/mov for build_k4:
  shl/or/shr/mul/dp4a/mad plus replicate-region movs;
  `isa-torch/.text._ZTSN4exl310DpasKernelILi4ELi2ELi4ELi4ELb0EEE.asm`,
  loop body ~lines 123–459). 13.12 GB/pass ÷ 512 B ≈ 25.6 M row-iterations
  × ~336 ≈ **~8.6 G instructions per M=4 pass**; at the measured 34.46 ms
  that implies ~250 G instr/s sustained ≈ ~3.1 instr/cycle/Xe-core against
  a ~8/cycle SIMD8 theoretical issue — the right order for an issue-bound
  loop with dependency gaps. (Paper model; consistency check, not proof.)
- Every mechanism that was *memory-side or scheduling-side* failed:
  padding (RL-101), strip geometry (RL-102, flat/regress on 3/4 families
  with the NT8 arm spill-confounded), staged reduction (RL-111 C1),
  K-pair/bank binding (RL-111 C2 falsifier, RL-112 −3.4 %).
- The op-class discriminator inverted the expectation: forced-vector M=4
  is *worse* than DPAS M=4 (250–265 vs 336–350 GB/s) because it issues
  4× the FMA work per byte — an instruction-count signature, not a
  bandwidth one.
- had_in+had_out is separately bounded at 1.4–2.1 ms/pass (measured,
  RL-102 guard) — real but only ~25–40 % of the 5.02 ms M4−M1 gap.

Evidence against / residual unknowns (honest list):

- The instruction model alone would predict a flatter per-family spread
  than the measured 46–75 % of pin; family-dependent parallel-efficiency
  and DRAM-access-pattern terms (512 B strided walks at 40 KB stride for
  down_proj) are unresolved second-order contributors.
- Activation re-reads: the DPAS path reads `Am = block_load<fp16,MB*16>`
  per (strip, K-row) — ~22 MB L2-side reads per down_proj call vs ~1.4 MB
  at M=1 (~16× per weight byte; `exl3_esimd.h:796`). The MB4 arm halved
  this and gained ~nothing on down_proj — so it is likely second-order,
  but the arm was not designed to isolate it (only `EXL3_DEBUG_A_ONCE`
  does that cleanly).
- Per-thread weight reads are 512 B contiguous at NT4 vs 1 KB at M=1/NT8 —
  a DRAM burst-efficiency difference that the NT8 arm could not isolate
  (spill-confounded).

**The exact isolated-op measurement that confirms/refutes it** — already
built into the source as diagnostic flags, zero new kernel code:

1. `EXL3_DEBUG_B_FAKE` (`exl3_esimd.h:812–819`): Bv = raw trellis words
   bit-cast to fp16, no `build_k4`/decode — removes ~85 % of the per-tile
   ALU/mov stream, keeps loads + 4 dpas + stores identical. Timing-only
   (wrong numerics by construction). **If M=4 GB/s recovers ≥ +15 % on
   down_proj/gate_up in both pass orderings → decode-issue is confirmed as
   the majority owner**; flat ⇒ not issue-bound → pivot to memory pattern.
2. `EXL3_DEBUG_A_ONCE` (`exl3_esimd.h:792–796`): Am loaded once per thread,
   not per row — isolates the activation-load term. ≥ +8 % both orderings
   = meaningful term; flat ⇒ closed.
3. (Secondary) `EXL3_BV_ALL` at MB4/NT4 under `EXL3_DPAS_LARGE_GRF`
   (`exl3_esimd.h:798–811`, `exl3_ops.sycl:110–124`): per-j B regions —
   the WAR/operand-banking question asked *correctly* (C2V2 asked it
   wrongly by pinning). Requires the 256-GRF launch path; ~136+ GRF named.
   ≥ +8 % both orderings = banking term real; flat ⇒ permanently closed.

Harness to reuse verbatim: `results/2026/09/b70-b6-sweep-20260929/nt_sweep.py`
(sweep-mode cell table + paired reversed orderings + M=1 drift references)
or `results/2026/09/b70-b6-mb4-ab-20260929/mb4_ab.py`; cells
F_down_proj / B2_gate_up / C_qkv at M=4, M=1 reference per container,
graph-replay kcap=32 ×100 medians + bulk200, fresh-container arms with the
C2V2-learned 2 s preheat.

## 3. The ONE next experiment

**Recommendation: the three-arm flag ablation above IS the next experiment.**
Rationale: every kernel effort in §1 burned a GPU window implementing a
mechanism without first proving the mechanism owns the time. The ablation
is the cheapest possible next step (three ~1-minute rebuilds, no new code,
one short GPU window) and each arm's gate maps directly onto a named
kernel decision. Expected value dominates both alternatives:

- vs. the **K3 P2 strip sweep** (the other named lever, RL-100): the sweep
  tunes `nt_of(3,MR)` — the *vector* GEMV axis (M≤2 serving shapes;
  P2-SWEEP-SPEC.txt §changed-parameter). The M=4 verify residual lives in
  the DPAS path, whose NT axis was already swept (RL-102). And the
  adoption arithmetic is unreachable: 42.44 tok/s needs ×1.49 to reach the
  ≥1.10×63.25 gate; strip geometry has moved ≤ ±10 % on exactly one
  family ever. If it is run anyway (it is cheap and the spec is frozen):
  base tree `/mnt/<nvme-disk>/b70-k3-build-20260929/src`
  @`15ded2f3ad` + patches 0001..0006, `EXL3_ALL_CODEBOOKS=1` image via
  Dockerfile.k3 on `unix:///run/<exl3-docker>.sock`; legal candidates =
  default `[8,4,4,4]` control plus `EXL3_FLAGS=-DEXL3_K3_NT_MR1=4`,
  `-DEXL3_K3_NT_MR4=8`, `-DEXL3_K3_NT_MR8=8` one-at-a-time (MR2 stays 4 —
  P1b legality; NT2 illegal for K3 per spec §legality). Gates per spec:
  P3 screen per arm, isolated ≥5 % weighted M=1/2 latency cut, then P4 —
  expected outcome: adoption stays NO-GO regardless of isolated wins.

- vs. any new kernel authoring: premature — the residual owner is
  unassigned. The ablation assigns it.

### Concrete experiment card

- **Source/build**: retained tree
  `/mnt/<nvme-disk>/b70-k3-build-20260929/src` @
  `15ded2f3add148c4db3c900cba7de878238f53bc` + patches 0001..0007;
  Dockerfile.k3, `EXL3_ALL_CODEBOOKS=0`, isolated dockerd
  `unix:///run/<exl3-docker>.sock`. Three derived images:
  `-DEXL3_DEBUG_B_FAKE`, `-DEXL3_DEBUG_A_ONCE`,
  `-DEXL3_BV_ALL -DEXL3_DPAS_LARGE_GRF`. Binary census per image to prove
  the ifdef landed (RL-101/102 census method). All three are
  **diagnostic-only / timing-only arms** — B_FAKE and A_ONCE produce wrong
  results by construction (flag comments say so), BV_ALL is numerics-true
  but the 256-GRF launcher edit is diagnostic scope.
- **Paper resource budget**: B_FAKE/A_ONCE strictly reduce live state
  (no budget risk). BV_ALL at MB4/NT4: Ball 64 GRF + acc 32 + words+prev
  32 + Am 8 ≈ 136 named → forces the existing `grf_size<256>` launch path
  (`exl3_ops.sycl:118–121`); SLM unused; record `.ze_info` grf_count and
  scratch per build — reject any spilling arm before timing (P2 spec rule).
- **Cells**: F_down_proj (P18/rps61), B2_gate_up (P3), C_qkv (P8) at M=4;
  M=1 flag-immune drift reference per container; in-window incumbent
  control in both orderings (RL-102 binding design verbatim).
- **Isolated-op gates (pre-declared)**: per arm, Δ GB/s vs in-window
  incumbent M=4 control, consistent in both pass orderings: ≥ +8 %
  ≈ "fires" (RL-102 wall-break rule); 2–8 % = hint; < 2 % = that term
  closed. Numerics arms B_FAKE/A_ONCE: none (wrong-by-construction);
  BV_ALL: RL-099 amended bound vs in-image control.
- **Consequence map (the deliverable of the window)**:
  - B_FAKE fires → decode-issue confirmed → the next kernel is a
    decode-cost reduction: re-test `EXL3_FOLD` confined to MB≤8 (`exl3_esimd.h:613–618`;
    measured +3.5 % at M≤8 historically, rejection reason was an M=24/40
    anomaly outside the serving shape — needs an RL-099-tolerance cell at
    M=4 to reopen) and, if bigger headroom is wanted, a decode-chain
    redesign (fewer ops/state; the mul+dp4a+affine chain in
    `decode_cb_h`/`build_k4`, exl3_esimd.h:41–80, 668–716).
  - A_ONCE fires → activation-term confirmed → next kernel = shared-A
    staging (one SLM fill per work-group per row-block, local=8 threads
    read identical Am: paper SLM 8×512 B = 4 KB, +1 barrier/row-group) or
    a 2-row A-batch that C2's pairing suggests is weak alone — pick per
    measured delta size.
  - BV_ALL fires → banking/WAR term real → per-j B regions folded into a
    proper MB4 inner loop (not register pinning).
  - All flat → the residual is the raw load/store/issue structure +
    DRAM access pattern → the remaining kernel-class lever is a layout
    repack for contiguous per-thread streaming (offline trellis
    repack = reconstruction-parity-gated transform) or accept the
    ~63 tok/s route and close the micro-tuning program.
- **End-to-end gate (only if a real candidate emerges)**: candidate kernel
  numerics under RL-099 amended bound + capture contract (D3a/D3b class),
  then P4-style matched protocol vs the RL-090 anchors (63.25/61.12 tok/s
  @230 W, p512/p8192 g128, single client, cache-hit delta 0), threshold
  ≥ +5 % decode for promotion discussion.
- **Failure modes to watch**: (a) the ±5 % M=4 class cross-session drift —
  in-window paired controls are mandatory, never compare against RL-098's
  absolute µs; (b) in-context pkg-temp pinning (~60–75 °C while a context
  lives) — fresh-process arms + 2 s preheat (C2V2 lesson: 349 µs cold vs
  ~106 µs warm); (c) concurrent drivers on the card — the o0 354 µs pair
  was contamination; empty-GPU gate + single controlling process per
  AGENTS §4; (d) spill sneaking in on the 256-GRF arm — check `.ze_info`/
  ISA before timing.

## 4. Abandon permanently (closed by measurement)

| Item | Closing measurement |
|---|---|
| MB=8→4 padding as the M=4 mechanism (re-measure needs a NEW split-K/tile/scheduling/kernel) | RL-101: +0.41/+4.72 pin pts vs ≥5 gate, bit-identical — blocked_pattern recorded |
| NT-axis re-sweep on the M=4 DPAS path | RL-102: flat/regress 3/4 families; in_proj_z +10 % is family-confined, ≤0.5 ms priced — blocked_pattern recorded |
| ESIMD_REGISTER explicit B-bank pinning; K-pair schedule as authored | RL-111 cheap falsifier (r17 reuse) + RL-112 V2 −3.4 %: stock compiler already multi-banks B (r18/r49/r51 in ISA) — "new mechanism" required |
| Staged/grouped had_out reduction as a standalone patch | RL-111 C1: strict D3 host-wait FAIL both arms + ≥15 % gate missed both orderings + bulk regresses; the entire stage is measured-bounded 1.4–2.1 ms/pass |
| The 354 µs "incumbent" M=4 reading | C2-MILESTONE-CLOSE.json §contamination: two concurrent drivers; clean incumbent ≈ 107.5–108.1 µs |
| K3-strip-sweep as a decode-speed rescue | RL-100: 0.671× incumbent; needs ×1.49 to gate; axis moves ≤ ±10 % on one family — arithmetic cannot close it |

## 5. Source anchors

- `/tmp/b70-k3-exl3-source-20260929/csrc/exl3_ops.sycl` (== in-image pinned
  @15ded2f3ad; same tree at
  `/mnt/<nvme-disk>/b70-k3-build-20260929/src/csrc/`):
  dispatch M≤2 vec / M≥3 DPAS `:267–280`; `g_vec_max_m=2` `:171`;
  `nt_of` `:182`; `g_target_threads{,_mb16,_mb64}=1024/1408/2048`
  `:176–178`; P/rps selection `:288–295`; `EXL3_DEBUG_SKIP_GEMV` `:299`;
  fused opt-in `:251` ("measured slower under XPU graphs"); 256-GRF
  launch path `:110–124`.
- `/tmp/b70-k3-exl3-source-20260929/csrc/exl3_esimd.h`: `Geo<K>` `:28–33`
  (K4: G8 V8 W32 D1); `decode_cb_h`/`decode_mul1_raw` `:41–94`;
  `load_words` `:154–168`; `HadOutKernel` `:321–336`; `GemvKernel`
  `:443–579`; `DpasKernel` `:591–861` — FOLD `:613–618`, PREFETCH
  (rejected M=16 spills) `:766–784`, A_ONCE `:792–796`, BV_ALL `:798–811`,
  B_FAKE `:812–819`, dpas emit `:823–830`, part store `:845–858`.
- ISA evidence:
  `results/2026/09/b70-c2-repair-20260930T085200Z/v2/isa-torch/.text._ZTSN4exl310DpasKernelILi4ELi2ELi4ELi4ELb0EEE.asm`
  (incumbent MB4: 4 dpas/row, B banked across r18/r49/r51, ~336
  instr/row-iter) and `.ze_info` (grf_count 128, no scratch);
  `results/2026/09/b70-resume-breakthrough-20260930T063200Z/C/isa/c2/.text._ZTS7C2KPair.asm:239,320`
  (C2's shared-B falsifier).
- Campaign evidence dirs: `results/2026/09/b70-b6-microprobe-20260929/`,
  `b70-b6-mb4-ab-20260929/`, `b70-b6-sweep-20260929/`,
  `b70-k3-completion-20260929/`,
  `b70-resume-breakthrough-20260930T063200Z/C/`,
  `b70-c2-repair-20260930T085200Z/`.
