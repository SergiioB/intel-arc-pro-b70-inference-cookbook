---
title: "ARCHITECT kernel spec — grouped-IQ3 prefill MUL_MAT_ID (20260921d rev. R)"
type: worklog
date: 2026-09-22
status: living
---

---
doc_id: WL-ARCHITECT-KERNEL-SPEC-R
doc_type: worklog
title: Grouped-IQ3 prefill MUL_MAT_ID kernel spec (re-emitted, rev. R)
lifecycle: active
created: 2026-09-22
updated: 2026-09-22
canonical_for:
  - grouped-iq3-prefill-kernel-spec
  - k216-routed-mul-mat-id
---

# ARCHITECT kernel spec — `grouped-IQ3 prefill MUL_MAT_ID` (20260921d, rev. R)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Re-emitted 2026-09-22 by `b70-architect`
and transcribed here by the orchestrator (the architect's role forbids file
creation; this is its reply in full). **Re-derived from evidence, not transcribed
from the lost fragment.** Where it corrects the 2026-09-21 fragment it marks
**[CORRECTION n]** — each correction is a review finding against any in-flight
implementation.

> **Supersession note (2026-09-22):** the execution shape changed after this spec
> was written. Upstream llama.cpp **#29245 "sycl: add grouped MoE XMX GEMM"**
> (open, 2026-09-21) is the same object, and **#29107 "IQ3 code reorder"**
> (open, 2026-09-18, +75 % tg on a B70) plus merged **#28476** (IQ3_XXS/IQ3_S
> fused decode IDs, `304665fe7`) and **#23142** (device-side expert mapping +
> counting sort, pp +70 % on Arc B60) also apply. This spec remains **the review
> instrument and the parity oracle** — it defines exactly the semantics and gates
> any implementation (ours or #29245's) must satisfy. See
> `P3-review-pr29245-20260922.md` for the delta analysis.

> **REV. R CORRECTIONS (2026-09-23, kernel-engineer review of #29245).** Six
> numbers in this spec were wrong and are corrected in place below; the review
> file holds the full reasoning. (R1) The FP16 round trip is **21.74 GB per
> routed layer**, not 10.9 GB — the three-op inventory doubles the waste and the
> kernel's upside grows. (R2) The gate weight-row stride is **1568 B (16×98)**,
> not 392 B. (R3) SLM total is **49.25 KiB**, not ~50.4 KiB (still ≪192 KiB).
> (R4) **M=16 padding waste is ≈29.5 % under uniform random top-6 routing**, not
> 11–15 % — the M=16 decision must be re-justified from real K216 expert-count
> histograms, not the uniform assumption. (R5) The sub-group mapping is
> **`s = 4q + r`** (two M halves × four N slices), not `s = 2q + r`. (R6) The two
> barriers shown per K-tile are **insufficient**: a proven post-compute barrier
> before W reuse is required, or faster sub-groups will overwrite W still being
> read by slower ones.

## 0. Fragment audit — corrections first (these are review findings)

**[C1 — LOUD] "≈100 MB/op" is wrong by ~36×.** Per gate-class op the path reads
**693.7 MB** of IQ3_XXS (216 experts × 2048×4096 weights × 98 B/256 w) and
materializes **3.62 GB of FP16**, which oneMKL then reads back — a **~7.2 GB
round trip per op** against a 694 MB useful read. Per routed layer (3 ops):
2.08 GB IQ3 → **10.9 GB FP16 round trip**. At ~608 GB/s the round trip alone is
~13 ms/op of pure waste, which is most of the 28.87 ms. If any worklog or review
note repeats "100 MB", kill it. *(Independently confirmed by the kernel engineer
2026-09-22: 3.375 GiB FP16 materialization per op over all 216 experts.)*

**[C2 — LOUD] The op inventory is three ops per layer, and `I=2048`, not 1024.**
Evidence (`results/2026/09/b70-improvement-probes-20260920/perf-k216-prefill-n512.txt`):

- gate-class: `m=2048, n=512, k=4096` → 28.87 ms, 1.79 TFLOPS, 51.54 GFLOP/run
- down: `m=4096, n=512, k=2048` → 29.91 ms, 1.72 TFLOPS, 51.54 GFLOP/run

Down's `k=2048` fixes `moe_intermediate=2048`, hidden=4096. Since a fused gate+up
op would show `m=4096` at `k=4096`, the 28.87 ms op is gate (or up) **alone**;
the "84 waits/token = 28 GPU-routed layers × 3 projections" arithmetic
(`research/2026/dsv4-k216-request-to-output-pipeline-20260825.md` lines 36, 92)
confirms **three separate `MUL_MAT_ID` calls per routed layer** (gate, up, down),
43 routed layers total. The kernel is one parametric shape family:
`{N_out ∈ {2048, 4096}, K ∈ {4096, 2048}}` — instantiate twice, do not fork the code.

**[C3] Activation parity anchor is FP16 — confirmed in source, do not "improve"
it.** The reference converts **both** operands before oneMKL: src0 IQ3→FP16
(`~/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp:2701-2707`) and
src1 F32→FP16 (`:2715-2721`). So our FP16 A-tile rounding is *identical* to the
reference's, and the ≤1e-5 output gate reduces to accumulation-order difference
only. **Output dtype is F32**: the reference stages routed results in an F32 pool
(`ggml-sycl.cpp:4845`, `sizeof(float)*n_routed_rows*ne0`). Our epilogue stores
FP32 accumulators directly — **no final rounding at all**, which is strictly
better for parity than any FP16 store.

**[C4] "5.52× above weight-read floor" is the *decode* op's number
(AGENTS.md §16). This prefill op is ~25.3× above its own floor** (28.87 ms vs
693.7 MB/608 GB/s = 1.141 ms). Do not cross-cite the two.

**[C5] SLM total is ~50 KiB with staging, not ~42 KiB.** 42 KiB holds only if
reconstruction gathers raw block bytes directly from global. The spec below
stages raw blocks in SLM (the 98 B rows are unaligned; see §5) → ~50.4 KiB. Both
fit 192 KiB; v2 double-buffering needs ~98 KiB — also fits.

**[C6] The 8–13 ms budget is honest but conservative; the ≥2× gate must not be
renegotiated.** Physics floor per op: **1.141 ms weight-bandwidth, 0.281 ms XMX
compute** (51.54 GFLOP at the 183.5 TFLOPS FP16 class peak, 170 measured —
`research/2026/xmx-dpas-rate-classes-20260915.md`). Expected v1
(gather-pessimistic): 7–11 ms; clean v1: 3–6 ms; v2 overlapped: 1.5–2.5 ms. If v1
ships at 10 ms and passes the ≤14.4 ms gate, that is a pass — but the review
should still demand the v1.5 items of §10, because the headroom to ~2 ms is real
arithmetic, not aspiration.

**[C7] M=16 padding waste — REVISED 2026-09-23 (R4).** E[T_e] = 3072/216 =
14.2 rows/expert, but under **uniform random top-6 routing** the padding waste is
**≈29.5 % of DPAS issue**, not the 11–15 % originally estimated. That is still on
a resource only ~25–28 % duty-cycled at the bandwidth floor (§3), so the gate may
still survive — but **the M=16 choice must be re-justified from real K216
expert-count histograms**, not the uniform assumption. Record the histogram as
part of the first kernel probe. Do not silently switch to M=8 either (it halves
the DPAS tile and raises K-tile count); decide from data.

Everything else in the fragment — tile (16,64,256), 128 work-items / 8 sub-groups
of 16, single-buffered v1 with two barriers per K-tile, FP16-first with BF16
separately gated, exact IQ3 semantics preservation, the parity ladder, the ≥2×
then ≥15 % gates — **survives re-derivation and stands**. K=256 in particular is
not arbitrary: `QK_K=256`, so one K-tile = exactly one IQ3 block per weight row
(§5).

## 1. Operator contract and roofline

| Quantity | gate/up-class op | down op | Source |
|---|---:|---:|---|
| Shape (tokens × N_out × K) | 3072 × 2048 × 4096 | 3072 × 4096 × 2048 | evidence file, C2 |
| FLOP | 51.54 G | 51.54 G | measured field |
| IQ3 weight bytes | 693.7 MB | 693.7 MB | 216·N·K·98/256 |
| FP16 materialization (current path) | 3.62 GB w + 3.62 GB r | same | C1; ×3 ops ⇒ **21.74 GB round trip per routed layer** (R1) |
| Measured | 28.87 / 29.91 ms (1.79/1.72 TFLOPS) | | evidence file |
| BW floor (608 GB/s) | 1.141 ms | 1.141 ms | |
| XMX floor (170 TFLOPS class) | 0.303 ms | 0.303 ms | rate-class note |
| Gate | **≤14.4 ms** | ≤14.5 ms | ≥2× vs recapture |

The kernel eliminates the round trip: read IQ3 once (694 MB), reconstruct in
SLM/GRF, feed DPAS directly. Compute is 25 % of bandwidth → **this is a bandwidth
kernel with an ALU tax (reconstruction)**; every design decision follows from that.

## 2. Grid decomposition, persistence, scheduling

- **Work unit** = (expert e, M-tile of ≤16 routed rows, N-tile of 64 output
  columns), K reduced fully inside the unit. Per op: ~216 × ~1.15 × {32 | 64}
  N-tiles ≈ **8k (gate/up) / 13.8k (down) units** — coarse enough for persistence.
- **Persistent grid: 32 work-groups, one per Xe-core**, launched once per op;
  units pulled from a global work-queue index. **An atomic on the queue index is
  permitted (scheduling only). Atomics on accumulators or outputs are FORBIDDEN**
  (§7).
- **Expert-major order within each WG's walk** (all N-tiles of expert e, then
  e+1): maximizes A-tile SLM/L2 reuse and keeps output writes sequential per
  expert. Token-major straddling of expert boundaries is illegal for correctness
  anyway (rows within a unit must share one weight matrix).
- Host-side inputs already exist and stay unchanged: the counting sort
  (`mmid_counting_sort_rows`, `ggml-sycl.cpp:4857`) produces contiguous
  expert-major routed rows and the row-mapping — the kernel consumes its outputs
  (`expert_row_counts/offsets`, contiguous src1) and the same expert-ID list. The
  D2H wait itself is NOT this kernel's scope (that repair is already in-tree,
  withdrawn as a speed lever, +3.54 % — L1/OUTCOMES).

## 3. Sub-group → DPAS mapping and occupancy math

Work-group = 128 work-items = **8 sub-groups of 16** (declare
`[[intel::reqd_sub_group_size(16)]]` — `research/2026/xe2-esimd-launch-bounds-2026-08.md`).
One WG per Xe-core: each sub-group lands on one XVE (Xe2 XVE is 512-bit → one
SIMD16 sub-group per thread slot); each sub-group issues DPAS on its core's
8th-shared XMX array.

Output tile (M=16, N=64) is split **8 = 2 (M-halves) × 4 (N-slices)**:

| sub-group s | rows (M) | cols (N) | DPAS shape per call |
|---|---|---|---|
| s = 4·q + r | rows [8q, 8q+8) | cols [16r, 16r+16) | `dpas<M=8, N=16, K=16>` FP16→FP32 |

*(R5: mapping corrected to `s = 4q + r` — two M halves (q ∈ {0,1}) × four N
slices (r ∈ {0..3}).)*

- K per DPAS instruction on Xe2 FP16: systolic depth 8 × OPS_PER_CHAN 2 = **16**
  → **16 DPAS per K-tile (256) per sub-group, 256 per unit**, 128 per K-tile per WG.
- Per K-tile per core: 262,144 MAC compute (≈91 ns at 128 FP16 MAC/clk/XMX × 8
  XMX @ 2.8 GHz) vs 6272 B weight read (≈330 ns at 19 GB/s/core share of 608) →
  **XMX duty ≈ 28 %** at full bandwidth. This is why v1 may serialize
  recon/compute/loads and still pass the gate.
- **GRF budget per sub-group** (≤8 KiB/thread, 512 B/lane at SIMD16): A-fragment
  held resident per K-tile 4 KiB (its 8 M-rows × 256 K × FP16) + accumulators
  8×16 FP32 = 512 B + one B-fragment 512 B + recon temporaries ≤1 KiB ≈
  **6 KiB of 8 KiB**. Tight → **zero-spill compile gate is mandatory** (inspect
  the compiler report/spill count before any GPU run; this is a CPU-side check).
- Occupancy is 1 thread per XVE (of 8 slots) — latency exposure is deliberate in
  v1 and hidden by double-buffering/producers in v2. **First fallback if v1 misses
  budget: 2 WGs per core** (SLM 2×50.4 = 101 KiB ≤ 192 KiB), not a rewrite.

## 4. SLM layout (v1, byte offsets, all 128-B aligned)

| Offset | Size | Buffer | Notes |
|---|---:|---|---|
| `0x0000` | 8192 | `A[16][256]` FP16 | activation K-tile, row stride 512 B; F32→FP16 RN on load (C3) |
| `0x2000` | 8192 | `RAW[64][128]` | staged raw IQ3 blocks: 64 rows × 98 B payload, padded to 128 B |
| `0x4000` | 32768 | `W[64][256]` FP16 | reconstructed weights, row = output col, stride 512 B |
| `0xC000` | 1024 | `iq3xxs_grid` | exact upstream table, 256 × uint32 (`ggml-common.h:1017`) |
| `0xC400` | 128 | `ksigns_iq2xs` | exact upstream table |
| `0xC480` | ~128 | unit metadata | T_e, ptrs, expert base |
| **total** | **≈50.4 KiB** | | ≤192 KiB; v2 double-buffer ≈98 KiB |

Bank-contract: A and W rows are long-strided (512 B); the gather hotspot is the
**grid table** (4 B gathers at data-dependent indices). Replicate
`iq3xxs_grid` ×2 at `0xC000`/`0xC800`-style padding if profiling shows banking —
1 KiB is cheap. The zero-fill of A's padding rows happens once at unit start;
never read uninitialized SLM.

## 5. IQ3_XXS reconstruction datapath (exact, from the pinned tree)

Source of truth: `dequantize_row_iq3_xxs`,
`~/llama.cpp/ggml/src/ggml-quants.c:2575-2603`; block struct
`ggml-common.h:408-411` (`sizeof = 2 + 96 = 98 B` per 256 weights =
3.0625 bpw). Because **K-tile = QK_K = 256, each weight row contributes exactly
one block per K-tile**: row n's block k sits at `n·(K/256 · 98) + k·98` — 98 B
contiguous, stride 392 B (gate-class) — hence the RAW staging pass (byte-granular
copy, then aligned SLM reads).

Semantics to preserve bit-for-bit:

1. Payload `qs[96]`: first 64 B are **grid indices** (8 per 32-weight group),
   last 32 B (`qs + QK_K/4`) are **scales+signs** (one uint32 per 32-weight group).
2. Per 32-weight group, `aux32`: top 4 bits (`aux32 >> 28`) are the sub-block
   scale s ∈ [0,15]; the four 7-bit fields `(aux32 >> 7l) & 127`, l = 0..3, index
   `ksigns_iq2xs[128]` → 8 sign bits applied via `kmask_iq2xs`.
3. `db = d · (0.5f + s) · 0.5f` — **two FP32 multiplies in exactly this order**
   (`quants.c:2589`); `d` = FP16→FP32 promotion of the block scale (exact).
4. Grid lookups: `qs[2l]`, `qs[2l+1]` each fetch a uint32 = **4 packed int8
   magnitudes** from `iq3xxs_grid[256]` (observed magnitude set
   {4,12,20,28,36,44,62}).
5. Weight value = `db × mag × ±1`.

Kernel reconstruction, provably bit-identical to the reference's FP16 tensor:
compute `db` in FP32 with the same two-multiply order; `mag` is exact in FP32;
one FP32 multiply `db×|mag|`; one FP32→FP16 round-to-nearest-even; **apply the
sign by XOR of 0x8000 after the FP16 conversion** (symmetric rounding makes
`f16(−x) ≡ −f16(x)` exactly, so this equals the reference's `×(−1.f)` in FP32
before conversion).

Per-lane assignment: one lane per 32-weight group → per K-tile/N-tile, 512 groups
over 128 lanes = 4 groups/lane; each lane does 1 `aux32` extract (4 shifts+masks),
4 ksigns lookups (GRF-resident, 8 B/lane), 8 grid gathers (SLM), 32 FP32 muls +
32 FP16 packs. Recon cost ≈ 0.3–1.0 µs per K-tile per core depending on gather
latency/ILP — the dominant v1 uncertainty and the reason the budget band is wide
(C6).

## 6. K-tile loop and barrier structure (v1)

```
for each work unit (expert e, m-tile, n-tile):          // persistent, expert-major
  load unit metadata; zero-fill A pad rows
  for k = 0 .. K/256-1:                                  // 16 (gate/up) or 8 (down)
    // phase L: all 8 sub-groups cooperatively
      async copy RAW <- global blocks (64 x 98 B)         // rows n0..n0+63, block k
      copy A <- src1_contiguous rows (16 x 512 B FP16)    // F32->FP16 RN here
      barrier                                            // #1: RAW/A complete
    // phase R: recon (all 8 sub-groups over 512 groups)
      RAW + tables -> W[64][256] FP16 (per section 5)
      barrier                                            // #2: W complete, RAW free
    // phase D: post-compute barrier (R6: REQUIRED)         // W reused next K-tile only
                                                          // after every sub-group's
                                                          // final dpas read of W
    // phase C: dpas -- each sub-group: load A-frag (its 8 rows) once into GRF,
    //            16 x dpas<8,16,16> with B-frags streamed from W
  epilogue (section 7); next unit
```

Two barriers per K-tile, 33 (17) per unit — matches the fragment. **Barrier
budget is visible, not free**: ~250 units/WG/op × 33 × ~30–100 ns ≈
**0.25–0.8 ms/op**. If v1 lands between 6 and 14 ms, this and gather latency are
the first two suspects (§10).

## 7. Epilogue and determinism rules

- Accumulate FP32 in GRF across all K-tiles in **fixed ascending K order**, per
  sub-group, per output tile. Single DPAS accumulation semantics per instruction
  are hardware-fixed; the instruction sequence is fixed → **run-to-run
  bit-identical**.
- One WG writes each output tile (single-writer), FP32 store to `dst` (C3),
  masked to valid rows. **No atomics anywhere in the numeric path; no split-K**
  (split-K would need cross-WG reduction → order nondeterminism). Cross-WG
  scheduling order may vary freely — it cannot touch numerics because tiles are
  disjoint.
- Store path: FP32 accumulators → `dst_contiguous`-equivalent rows (the scatter
  back to per-slot rows stays in the existing copy kernel — out of scope).

## 8. `n_used=6` grouped scheduling — determinism and cache behavior

Expert-major (chosen, §2): weights are stream-once data — **DRAM traffic is
order-invariant**; what ordering changes is (a) A-tile reuse (per WG, SLM/L2) and
(b) output write locality. Token-major interleaving would fragment A reuse and
buy nothing (weights aren't reused across experts). Decisive cache fact: the full
activation matrix per op is 512×4096×2 B = **4 MiB ≤ 18 MB shared L2** → the 32×
A-re-reads implied by N-tiling are L2 hits, so DRAM traffic stays ≈ weights-only
(694 MB) + outputs (25 MB) + one A pass. At n=1024 A is 8 MiB — still
L2-resident; at n=2048, 16 MiB — borderline, but ubatch is capped at 1024 anyway
(P-2 LOSS).

## 9. Parity and gates (unchanged from fragment, plus two additions)

1. Reconstructed FP16 weight tensor **bit-identical** to the existing route's
   materialized tensor (same table bytes, same op order — §5). Cheapest first
   check: CPU/SYCL unit test of the recon routine vs `dequantize_row_iq3_xxs` →
   FP16, over whole experts. **No GPU needed.**
2. Finite outputs on real ubatches (no NaN/Inf).
3. Relative squared-L2 ≤ **1e-5** vs the existing FP16 route's outputs, same
   inputs. **Do not relax after failures** — a failure here means an
   implementation bug (only accumulation order should differ), not tolerance
   starvation.
4. Backend NMSE ≤ **5e-4** on downstream activations.
5. **NEW — zero-spill compile gate**: sub-group-size attribute set, compiler
   report shows 0 spill instructions (CPU-side, before any GPU time).
6. Isolated-op gate: **≥2× (≤14.4 ms** vs the 28.87 ms recapture, n=512 k=4096;
   ≤14.95 ms for down vs 29.91), then **≥15 % matched end-to-end prefill** (same
   protocol as P-1/P-2: p7000/g1, matched n=3, fresh n=5 finalist).
7. BF16 variant: **separate gate, not parity-neutral** (different rounding of
   `db`, different DPAS accumulate behavior) — do not reuse gate 1/3 results.

## 10. Failure modes and mitigations (watch list for the reviewer)

| # | Failure | Signal | Mitigation |
|---|---|---|---|
| F1 | GRF spill (6 of 8 KiB budget) | compiler report; perf cliff | bound unrolls; drop A-frag GRF residency (fall back to SLM reads, costs SLM BW) |
| F2 | SLM bank conflicts on grid gathers / W reads | VTune SLM counters | replicate grid table; keep 512-B row strides; audit gather pattern |
| F3 | Recon gather latency serializing phase R | per-phase timing (K-tile time ≫ 330 ns floor) | 2 WGs/core (interleaved threads hide latency); batch uint32 grid fetches (one gather yields 4 weights) |
| F4 | Barrier overhead 0.25–0.8 ms/op | unit time vs K-tile count | v1.5: fuse RAW copy+recon; v2: double-buffer (one barrier/step) then producer/consumer (zero) |
| F5 | Unaligned 98 B global reads | silent slow path / faults | RAW staging pass (spec'd); never read block bytes directly from global in v1 |
| F6 | Tail M rows garbage | parity gate 3 fails on small experts | zero-fill pads once; mask stores to valid rows |
| F7 | T_e = 0 / duplicate IDs | hangs or OOB | skip empty experts in work list; assert `0 ≤ id < 216` |
| F8 | Grid/ksigns table drift | gate 1 fails bit-identity | copy tables verbatim from `ggml-common.h`; never regenerate |
| F9 | Accumulation atomics "for convenience" in review | nondeterminism | forbidden by §7 — reject in review on sight |
| F10 | Output dtype FP16 "optimization" | gate 3 margin eaten | output is F32 (C3); rounding removal is free parity margin |

## 11. Explicit non-goals and deferred hooks (do not creep v1)

- **INT8/s4 DPAS variant: REJECTED by arithmetic.** s8×s8 doubles the rate class
  (363 vs 183), but compute is already only ~25–28 % of the bandwidth floor — a
  2× compute multiplier buys ~0 ms and adds activation-quantization error, i.e.
  it drifts toward the killed W4A8 premise. Cite
  `xmx-dpas-rate-classes-20260915.md`.
- v1.5/v2, in order: double-buffer W (kill barrier #2) → producer/consumer
  sub-group split (1–2 producers on `esimd` async 2D block loads, 6–7 DPAS
  consumers — AGENTS.md §17 rule 4) → SiLU-mul fusion into the gate/up epilogue
  (removes one 12.6 MB activation round trip; own parity gate) → **M≈4–16
  fold-in (O4)**: the same kernel at M∈{4,8,16} is the measured ~2× headroom
  object of H3 (`perf-mulmat-mregime.txt`: 8.87 vs ~17.3 TFLOPS at floor) and
  also the decode-side object — evaluate after v1 lands, per OUTCOMES O4.

---

# JOB 2 — RANKED NEXT POINTS (single-stream weighted: gain × confidence ÷ cost)

**The flat statement first: the huge wins are gone.** After D1 (+37–46 % decode),
D3 (1.68× capture), S1/S2 (2.79× aggregate) — all measured — and with P-3
specified above, nothing in the evidence supports another ≥1.5× single-stream
lever on this hardware. What remains is one structural bet (the kernel), a stack
of 5–10 % compounds that do compose, and serving-shape plays that score zero on
single-stream. Also: the highest certainty-per-day item on the board is not
silicon at all — **closing O1's three publication chores converts the
already-measured +44 % from PROVISIONAL to shippable at zero measurement risk.**
Do that in parallel with everything below.

| # | Lever | Route | Expected (single-stream) | Conf. | Cost | 15-min kill probe | Must not duplicate |
|---|---|---|---|---|---|---|---|
| 1 | **Grouped-IQ3 kernel (this spec)** | K216 prefill | **+32…+47 % pp E2E** (routed = 73.5 % of 5.06 s ubatch; op ≥2× → ~133 pp, op 3× → ~149 pp; at ub1024: 130.78 → ~180–206 pp) | high physics / med-eng | 3–5 d | CPU-only: recon unit test vs `dequantize_row_iq3_xxs` — bit-identity in minutes, no GPU | P-2 (ubatch >1024: killed); device-ID speed repair (L1) |
| 2 | **Verify-pass (5th) LM-head INT4** (D8, open) | Qwen champion decode | **+7–9 %** — head is 2.54 GB F16 read once per verify step; W4-g128 → ~0.64 GB ⇒ ~1.9–2.5 ms of the 39.8–46.8 ms step | med-high (same arithmetic that predicted D1) | 1–2 d | Offline: head F16 vs W4 matmul over ~1k cached hidden states; top-1/top-32 logit agreement + ΔNLL — modal flips ⇒ dead | W4A8 (killed for weight/activation quant; this is the LM head — still quality-gated via the O2 30-draw protocol) |
| 3 | **KV-cache dtype INT8 on the champion** (novel) | Qwen vLLM XPU, long-ctx | **+4–8 % at p8192, growing with ctx** (KV@8K ≈ 3–4 GB F16 ⇒ ~1.6–2 GB/step saved ≈ 2.6–3.3 ms of ~43 ms) **+ 2× KV capacity** (moves the 163840 ceiling / ≥98K reserve) | med — hinges on vllm-xpu int8 KV kernels | 2–4 d if kernels exist | `--kv-cache-dtype` launch on p2048 n=1: unsupported/fallback ⇒ dead in 15 min | Q4 (KV ≠ weight path; separate quality gate, same O2 instrument) |
| 4 | **Speculation economics: context-dependent draft depth + ≥98K drop function** (O5+S4, novel packaging) | Qwen champion | **+0–8 %** — tokens/step = 1+4a; with a ∈ 0.55–0.69 run-variable, static depth wastes step time wherever a is low; dynamic depth or a ≥98K drop protects the tail without touching the champion cell | med-low | 1–2 d | Parse existing spec-decode counter snapshots: per-position acceptance vs ctx — flat and high ⇒ dead | D6 (depth *retune* banned — dynamic control is a changed implementation, allowed) |
| 5 | **230 W cap rider on the new prefill kernel** (novel, cheap) | K216, after Job 1 lands | **+0–4 % prefill** — the new kernel runs 28 % XMX duty + ~540 GB/s sustained; power sensitivity of *that* mix is unmeasured (decode was recorded flat, H5 — prefill never was) | low-med | 0.25 d | Registered paired A/B, one ub512 cell at 150 vs 230 W with energy/temp telemetry (existing harness) | H5 (decode-only neutrality); restore `cap_before` after |
| 6 | **Hot-expert GPU placement for K216 decode** (novel) | K216 L5 decode | **≤+5 %, likely less** — CPU-MoE term ≈ 15 layers × 55 MiB/token ÷ ~25 GB/s real DDR4 ≈ 33 ms of the ~108 ms token; top-40/216 experts cover maybe 45–55 % of routed mass, but only ~1.8 GiB free fits ~5 layers' hot sets (need ~5.4 GiB for all 15) ⇒ saves ≲5 ms/token | low | 3–5 d (`-ot` overrides are per-tensor; partial expert-bank placement is new plumbing) | Instrument `ids_host` (`ggml-sycl.cpp:4791`) for one p512 run → coverage curve; top-40 coverage <35 % ⇒ dead | `-ncmoe` sweeps (killed); spiritbuun `exps=CPU` (killed, no cache) |
| 7 | **Cross-card prefill/decode disaggregation** (novel, serving-class) | Qwen champion, B≥2 | **0 % single-stream — stated flatly.** Aggregate/TTFT only: B=4 is 140 tok/s with TTFT ~2 s; a prefill-dedicated card 1 removes decode interruption, worth up to ~1.3–1.8× on TTFT-bound serving. KV handoff is cheap in principle (KV@p512 ≈ 0.2–0.3 GB over ~8 GB/s x8 ≈ 30 ms ≪ 2 s TTFT) | low-med | 5–10 d | From existing `conc-B*.jsonl`: decompose TTFT into queue vs compute — if TTFT is queueing-dominated, dead | ADR-0008 TP2 (this is not tensor-parallel: no per-layer collectives, one-way stream — but it inherits the no-P2P host-staging constraint) |

**Deliberately not ranked:** draft-DFlash transfer to Nemotron's n7 (D7's drafter
measured ~25 vs 51 tok/s with defects root-caused upstream; secondary route, high
cost); concurrency-knee depth beyond B=32 (S2 shows it still rising but the
operator has deprioritized aggregate — revisit only if the serving objective
changes); M≈4–16 as a separate program (folded into Job 1 v2 per O4).

**Recommended sequencing:** O1 chores + Job 1 in parallel (independent); then #2
→ #3 on the same champion harness (they stack: both cut per-step bytes, combined
~+13–16 %); #4 rides the O2/O5 instrumentation already built for the re-measure;
#5 is a rider on Job 1's campaign; #6 and #7 wait for an explicit operator call
on K216 decode and serving-shape priorities respectively.

Evidence anchors:
`results/2026/09/b70-improvement-probes-20260920/perf-k216-prefill-n512.txt`,
`…/perf-mulmat-mregime.txt`, `OUTCOMES.md` + `LEVERS.md`,
`research/2026/xmx-dpas-rate-classes-20260915.md`,
`research/2026/dsv4-k216-request-to-output-pipeline-20260825.md`,
`~/llama.cpp/ggml/src/ggml-quants.c:2575-2603`,
`…/ggml-sycl/ggml-sycl.cpp:2701-2721, 4770-4884`,
`…/ggml-common.h:408-411, 1017`.

Unverified items, labeled: 608 GB/s effective bandwidth and the 170/363 rate
classes are measured anchors from the rate-class note (operator ngen probe,
unverified-by-protocol flag there); the 8–13 ms v1 budget and all Job 2 effect
sizes are the architect's arithmetic, not measurements — each carries its kill
probe above. Nothing here ran on the GPUs; no files were created by the architect.
