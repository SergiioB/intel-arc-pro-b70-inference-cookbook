---
title: "Grouped-IQ3 prefill kernel — implementation-ready design spec (DSV4 K216, B70, llama.cpp SYCL)"
type: plan
date: 2026-09-23
status: spec
phase: P3 (K216 grouped-IQ3 prefill); ledger RL-011
source-of-truth: llama.cpp tree at ~/llama.cpp, pinned 1692f9e50
---

# Grouped-IQ3 prefill kernel — implementation-ready design spec

Author role: B70 hardware architect (design only; nothing here has been compiled or run).
Consumer: the implementing engineer. Every file/line reference below was read on
2026-09-23 in the pinned tree `~/llama.cpp` (commit `1692f9e50` + the
documented local modifications, `git status` verified).

## 0. Baseline, scope, and three record corrections

Measured baseline (trust, do not re-derive; evidence
`~/B70-DOCS/results/2026/09/b70-improvement-probes-20260920/perf-k216-prefill-n512.txt`):

| op (test-backend-ops vars) | n=32 | n=512 | gate |
|---|---:|---:|---|
| gate/up shape `m=2048,n=512,k=4096` | 6.28 ms (0.51 TF) | **28.87 ms (1.79 TF)** | ≤ 14.4 ms (≥2×) |
| down shape `m=4096,n=512,k=2048` | 6.40 ms (0.50 TF) | **30.02 ms (1.72 TF)** | ≤ 15.0 ms (≥2×) |

Per layer: 2× gate/up-shaped + 1× down = 86.6 ms; × 43 GPU-routed layers = 3.72 s of a
5.06 s 512-token ubatch. Dense-GEMM reference on the same card/harness ≈ 45–46 TFLOPS →
the routed op runs ~25× below dense efficiency. Ubatch lever exhausted (512→1024 = 1.292×,
2048 = +0.2 %, 4096 OOM). The kernel is the only remaining prefill lever.

Model geometry (from the P3 attribution, verified against the draft kernel's shape guard):
216 experts, top-6, hidden 4096, expert FFN 2048. A gate/up-shaped routed op is
per-expert W [K=4096 → N=2048] (8.39 M weights); the layer runs it twice (gate, up).
Down is per-expert [K=2048 → N=4096]. At ubatch 512: 3072 routed rows, all 216 experts
selected, mean per-expert M ≈ 14.2 (near-Poisson), expected M-tiles ≈ 280 at TM=16.

**Correction 1 — IQ3_XXS is 3.0625 bpw, not 4.25.** `block_iq3_xxs` = 98 B per 256-value
super-block (`~/llama.cpp/ggml/src/ggml-common.h:404-411`). 4.25 bpw is
IQ4_XS. This matters for every traffic calculation: weight bytes per gate/up op =
216 × 2048 × 4096 × 98/256 B = **693.6 MB** IQ3, vs **3.62 GB** if materialized as FP16
(5.22× expansion).

**Correction 2 — a draft kernel already exists on disk.** Untracked files in the pinned
tree: `~/llama.cpp/ggml/src/ggml-sycl/mul-mat-id-iq3.cpp` (204 lines),
`mul-mat-id-iq3.hpp`, `iq3-xxs-reconstruct.hpp`; CMake option
`GGML_SYCL_IQ3_GROUPED_EXPERIMENTAL` (`ggml-sycl/CMakeLists.txt:35-37`); runtime guard
env `GGML_SYCL_IQ3_GROUPED=1`. It is joint_matrix (FP16 XMX) with tile (M,N,K) =
(16, 64, 256), 128 work-items, 8 sub-groups of 16. Its in-kernel reconstruction was
verified by this review to be **bit-faithful** to the reference dequant (section 2.3).
What is missing: the dispatch hook (no call site exists — `grep iq3_grouped
ggml-sycl.cpp` is empty), all parity/perf gates, and the down shape. This spec adopts
the draft as E0a rather than redesigning it.

**Correction 3 — one op vs "gate/up".** The 28.87 ms figure is ONE [4096→2048] routed
op; the layer executes it twice. "First build ONE unfused gate/up MUL_MAT_ID reused for
both projections" therefore means: one kernel, one shape guard `[4096, 2048, 216]`,
called twice per layer — exactly what the draft guard admits.

## 1. Kernel data-flow design

### 1.1 What the current path does (functions to supersede)

`ggml_sycl_mul_mat_id` (`ggml/src/ggml-sycl/ggml-sycl.cpp:4770-4937`), prefill branch
(ne12 > 1):

1. `memcpy` of the full `ids` tensor D2H + `stream->wait()` (lines 4791-4798) — host sync.
2. Host counting sort `mmid_counting_sort_rows` (line 4857) → per-expert row counts/
   offsets; row mapping H2D (line 4861).
3. `k_copy_src1_to_contiguous` gather kernel (line 4879) — 3072×4096×4 B = 50.3 MB.
4. **Per-expert loop** (line 4888-4915): 216 × `ggml_sycl_mul_mat` on the expert slice.
   Each call hits the FP16 branch of `ggml_sycl_op_mul_mat` (lines 2695-2730):
   whole-expert dequant via `to_fp16_sycl` (= `dequantize_block_iq3_xxs` in
   `ggml/src/ggml-sycl/dequantize.hpp:1421-1450`) into a pool buffer, then oneMKL/oneDNN
   `row_gemm` in FP16 at M ≈ 14.
5. `k_copy_dst_from_contiguous` scatter kernel (line 4929) — 25.2 MB × (read+write).

### 1.2 Traffic accounting, per gate/up op at n=512 (all 216 experts selected)

| stage | current path | E0a (this design) | E0b (one bounded revision) |
|---|---:|---:|---:|
| ids routing | 12 KB D2H + host sync | 0 (stays on device) | 0 |
| activation gather | 50.3 MB wr + ≤50.3 rd | none (indexed in-kernel) | 4 MB one-shot F16 convert |
| weight read (IQ3) | 693.6 MB | ~900 MB (1.3× M-tile amplification) | ~900 MB |
| FP16 materialization | **3.62 GB wr + 3.62 GB rd** | 0 | 0 |
| activation read by GEMM | ~50 MB | 2.35 GB (F32, re-read per 32 N-col-groups) | 0.59 GB (F16, 16 col-groups) |
| output write | 25.2 MB (+25.2 scatter rd + 25.2 wr) | 25.2 MB (routed direct) | 25.2 MB |
| **GDDR total** | **≈ 7.4–7.5 GB** | **≈ 3.3 GB** | **≈ 1.5 GB** |
| **floor @ 608 GB/s** | **12.2 ms** | 5.4 ms | 2.5 ms |
| kernel launches + syncs | ~434 (216 dequant + 216 GEMM + gather + scatter + sort + 1 hard sync) | 3 (group_rows, task build, main) + 0 syncs | 4 |
| measured / predicted | 28.87 ms | 6–9 ms | 3–4.5 ms |

Current-path decomposition sanity: 3.62 GB dequant write ≈ 6 ms (write-bound scalar
unpack) + 3.62 GB GEMM re-read at M≈14 (oneMKL F16 small-M, ~6–12 TF effective → 4–8 ms)
+ 216 × launch/alloc overhead (~20–40 µs each → 4–8 ms) + gather/scatter ≈ 0.5 ms ≈
21–29 ms — consistent with the measured 28.87 ms. **The 25× dense-gap is therefore
~41 % pure materialization traffic + ~30 % small-M GEMM inefficiency + ~25 % launch
structure.** The design deletes all three; what it cannot delete is the IQ3 unpack ALU
and the 98-B-stride uncoalesced weight loads — that is the residual ~1.3–2 ms-class
floor that keeps this out of dense-GEMM territory (section 3.3).

### 1.3 New data flow (block-dequant-in-SLM → XMX, never GDDR FP16)

Per routed op, three device kernels, `ids` never leaves the device:

1. **`group_rows`** (exists, draft lines 26-51): 216 work-groups of 128 WI each scan the
   3072-slot `ids` tensor, compact per-expert row indices via
   `exclusive_scan_over_group` — no atomics, no host round-trip. O(216×3072) id reads,
   trivial.
2. **Task-list build** (draft lines 119-129): single work-item loop over 216 experts
   emitting `(expert, row_offset)` 16-row tasks, `tasks[0] = nt`. ~281 iterations;
   acceptable for E0 (tens of µs); replace with a device scan in E1 if it shows in the
   profile.
3. **Main tiled kernel** (draft lines 130-199). Work-group = one (task, N-col-group)
   pair; grid = `tiles_max × (N/64)`. Per K-super-block of 256 (one IQ3 super-block
   deep — see 2.4):
   - 128 WI cooperatively load activations: 16 routed rows × 256 k × 4 B = 16,384 B
     GDDR → SLM FP16 fragment `a[16][256]` (8,192 B SLM).
   - 128 WI cooperatively unpack weights: 64 output cols × 1 super-block × 98 B =
     **6,272 B GDDR** → `reconstruct()` in registers → FP16 → SLM `b[256][64]` in
     joint_matrix `ext_intel_packed` layout (32,768 B SLM).
   - barrier; 8 sub-groups run 16 K-steps of `joint_matrix_mad` (8×16 × 16×16 packed →
     FP32 acc) entirely from SLM; barrier.
   - Epilogue `joint_matrix_apply` writes FP32 results straight to the routed output
     rows (ragged rows masked via `route[] = -1`).

Per-expert, per-M-tile byte ledger (all 32 col-groups, K=4096 → 16 super-blocks):

| stage | GDDR bytes in | SLM bytes written | GDDR bytes out |
|---|---:|---:|---:|
| weights, per (task,col,k-tile) | 6,272 | 32,768 (FP16 B) | — |
| weights, per task (32 cols × 16 k-tiles) | 3,211,264 (= whole expert IQ3) | — | — |
| activations, per (task,col) — E0a | 262,144 (F32) | 8,192 ×16 (A frag) | — |
| activations, per task — E0a | 8,388,608 | — | — |
| accumulators | — | 4,096 regs/WG | — |
| epilogue, per (task,col) | — | — | ≤ 4,096 (F32, masked) |

Workspace adds: `rows` 216×3072×4 B = 2.65 MB, `counts` 864 B, `tasks` ≈ 2.3 KB,
E0b adds a 4 MB shared FP16 activation buffer. All from the SYCL pool; well inside the
post-load VRAM gates (contract §5).

Why it beats the current path, in one sentence: the FP16 expert never exists in GDDR —
it exists for one K-super-block at a time in 42 KB of SLM — and 434 launches with a
hard host sync collapse into 3 launches with none.

## 2. Exact first-tile spec

### 2.1 Tile and mapping (E0a — keep the draft numbers)

- **Tile (M, N, K) = (16, 64, 256).** TM=16 matches DPAS/joint_matrix M-fragmentation
  (2 × 8); TN=64 gives N/64 = 32 col-groups; TK=256 equals exactly one IQ3_XXS
  super-block (section 2.4) — zero cross-block fragmentation, one 98-B block per
  (output col, k-tile).
- Work-group: **128 work-items = 8 sub-groups of 16**;
  `[[intel::reqd_sub_group_size(16)]]` sits ABOVE `operator()` (draft lines 30, 137 —
  already correct; icpx rejects the post-parameter-list form).
- Sub-group map (draft lines 147-148): `sm = (sgid/4)*8`, `sn = (sgid%4)*16` →
  2 M-fragments × 4 N-fragments cover the 16×64 tile.
- joint_matrix fragments (draft lines 158-160): `a` 8×16 `half` row-major (stride TK);
  `b` 16×16 `half` `ext_intel_packed` (stride 2×TN); `acc` 8×16 `float`. K-loop: 16
  steps of k=16 per super-block (draft lines 184-188).
- Grid: `nd_range<2>({tiles_max, (N/TN)*128}, {1, 128})`; work-groups with
  `task >= tasks[0]` return before any collective (draft lines 139-141 — the early-out
  is work-group-uniform, so barrier legality holds).
- Tail/skew handling: `route[tid] = offset+tid < counts[expert] ? rows[...] : -1`;
  padded A rows load 0; epilogue masks `r < 0`. Empty experts produce zero tasks
  naturally. This satisfies the P3 requirement (tails, skew, empty experts) with no
  separate code path.

### 2.2 SLM and register budgets

- SLM (draft static_assert, line 20-22): A 8,192 B + B 32,768 B + grid 1,024 B +
  ksigns 128 B + route 64 B = **42,176 B** per work-group. Xe2 L1/SLM is 192 KB per
  Xe-core → up to 4 WGs resident per Xe-core by SLM (fewer by GRF; either way the
  8,960-WG grid has no tail problem: ~70 scheduling rounds at 4 WGs/core).
- E0b (TN=128): A 8,192 + B 65,536 + 1,216 ≈ **74.9 KB** → 2 WGs/Xe-core. Still safe.
- Registers per lane (est.): acc 8×16 F32 ÷ 16 lanes = 8 regs; a-fragment 4; b-fragment
  4; unpack temporaries + addressing 20–40 → **≈ 40–60 regs/thread**. No spill expected
  at the standard allocation; if icpx spills, add `-zeopt-large-register-file`
  (`-Xsycl-driver '-zeopt-large-register-file'` / env `ZE_OPT_LARGE_REGISTER_FILE=1`,
  the IGC-mandated flag for GEMM-class kernels) before any tile change — spills at this
  budget would indicate a codegen issue, not pressure.

### 2.3 IQ3_XXS bit layout the kernel must unpack (authoritative)

Sources: reference dequant `~/llama.cpp/ggml/src/ggml-quants.c:2575-2603`
(`dequantize_row_iq3_xxs`); device twin `~/llama.cpp/ggml/src/ggml-sycl/dequantize.hpp:1421-1450`
(`dequantize_block_iq3_xxs<dst_t>` — this is the current path's FP16 producer and the
parity oracle); block struct `ggml-common.h:404-411`; tables `ggml-common.h:1017`
(`GGML_TABLE_BEGIN(uint32_t, iq3xxs_grid, 256)`, and nearby `ksigns_iq2xs`[128],
`kmask_iq2xs`).

Super-block = 256 consecutive k values, 98 B:

```
struct block_iq3_xxs { ggml_half d; uint8_t qs[96]; };   // 98 B, 3.0625 bpw
```

- `d`: FP16 super-block scale.
- `qs[0..63]`: grid-index bytes. Grouped by ib32 ∈ [0,8) (32 values per group): bytes
  `qs[8*ib32 .. 8*ib32+7]`. Within a group, l ∈ [0,4): indices `q3[2l]`, `q3[2l+1]`
  select two `iq3xxs_grid` entries; each entry is a uint32 whose four bytes (LSB first)
  are the multipliers for k = 32·ib32 + 8·l + j and +4 (j ∈ [0,4)). Grid bytes are a
  nonuniform lattice (0x04, 0x0c, 0x14, 0x1c, 0x24, 0x2c, 0x3e, …) — use them verbatim
  as integer multipliers; do NOT reinterpret them as signs or magnitudes.
- `qs[64..95]`: scales+signs, 4 B per ib32, read little-endian as `aux32`
  (`scales_and_signs + 4*ib32`). Alignment note: block starts are only 2-B aligned for
  odd block indices (row stride 1,568 B, block stride 98 B) — read `aux32` as two
  uint16 or four bytes, never as one uint32 load.
  - Scale: `db = (float)d * (0.5f + (aux32 >> 28)) * 0.5f` (F32 math, exact order).
  - Signs: per l, `signs = ksigns_iq2xs[(aux32 >> (7*l)) & 127]` (four 7-bit selectors
    in the low 28 bits); bit tests `signs & kmask_iq2xs[j]`, kmask = {0x01,0x02,0x04,0x08}.
- Value: `y[..] = db * grid_byte * (sign ? -1.f : 1.f)` — multiply order fixed:
  `(db * grid) * ±1`, then one F32→FP16 rounding at store time.

The draft's `ggml_sycl_iq3::reconstruct` (`iq3-xxs-reconstruct.hpp:21-29`) implements
exactly this and was checked against all three sources; its F32 arithmetic order
matches, so **bit-identical FP16 output vs `dequantize_block_iq3_xxs<half>` is
achievable by construction** — the parity gate (4.1) verifies the indexing/plumbing,
not new math.

### 2.4 Super-block/K-tile alignment (the load-bearing convenience)

TK = 256 = one super-block in K. For weight row (output col) n and k-offset k0:
`block = (const block_iq3_xxs *)(weights + expert*w_nb[2] + n*w_nb[1]) + k0/256`.
Row = 16 blocks = 1,568 B; expert = 2048 rows = 3,211,264 B; op = 216 experts =
693.6 MB. One (task, col, k-tile) touches 64 blocks, 6,272 B — contiguous per row,
98-B strided across rows. (If a later revision wants coalescing, a one-shot per-expert
weight repack into a [k-major][n] scratch is possible but NOT E0 — it re-introduces a
materialization pass; measure first.)

## 3. Implementation routes compared, and the honest ceiling

### 3.1 Route (a) — ESIMD/DPAS INT8-class with in-register IQ3→INT8 conversion: REJECTED for E0

Two independent kills, either sufficient:

1. **Parity.** INT8 XMX needs both operands integer (no mixed s8×f16 DPAS), so
   activations must be quantized to INT8/s8 per-row-group. Activation quantization
   injects ~1e-3 relative error — three orders above the ≤ 1e-5 relative-L2 gate vs
   the existing FP16 route. The gate is fixed; the route loses.
2. **Rate-class law** (`research/2026/xmx-dpas-rate-classes-20260915.md`, contract §17):
   sub-8-bit operands add NO TOPS on Xe2 — every ≤8-bit mix ties the INT8 class
   (363 TOPS-class vs 170 for BF16/FP16); only s4×s4 doubles MACs/instruction (2.00×
   measured), and IQ3 has no 3-bit DPAS mode at all. Even granting the full 2×, compute
   drops from ~0.86 ms to ~0.43 ms while the E0b traffic floor is ~2.5 ms — the win
   would be invisible. **The speedup must come from deleting the 7.2 GB materialization
   and the 434-launch structure, and it does.**

An ESIMD `xmx::dpas` rewrite (vs joint_matrix) is legal and can express sub-byte
operands joint_matrix cannot — but we need no sub-byte operand. Keep ESIMD in reserve
only if icpx's `ext_intel_packed` codegen for joint_matrix B-frags proves broken on
this B70 toolchain (symptom: NMSE gate cannot pass with correct indexing).

### 3.2 Route (a′) — joint_matrix FP16 XMX with in-register/SLM IQ3→FP16: CHOSEN (the draft)

- Operands FP16, accumulate/output FP32: same numeric class as the current
  oneMKL path, so the ≤ 1e-5 L2 gate is reachable (differences only from tile-wise
  F32 accumulation order, typically ≤ 1e-6 at these K).
- XMX FP16 class ceiling on B70 ≈ 170–183 TFLOPS (2·M·N·K convention; the Intel-doc
  8192-vs-4096 ops/cycle discrepancy leaves a ±2× hardware-peak ambiguity — flagged in
  the architecture reference; irrelevant at these margins). At 30–50 % achievable for
  this tile shape: 51.54 GFLOP / 51–85 TF ≈ **0.6–1.0 ms of math** — subordinate to
  traffic in both E0a and E0b. The kernel is bandwidth- and unpack-bound, which is
  exactly what the tile design addresses.

### 3.3 Route (b) — oneMKL grouped GEMM (`gemm_batch`) on FP16 fragments: REJECTED as primary

oneMKL consumes GDDR pointers, not SLM fragments — route (b) necessarily keeps an
INT8/FP16 GDDR materialization of every expert (≥ 693.6 MB wr + rd for INT8,
≥ 7.2 GB round trip for FP16) and keeps per-expert M≈14 inside the batched GEMM. It
fixes only the launch count (1 launch, no host sync). Expected 1.3–1.8× — below the
gate. Retained as the diagnostic fallback (section 6) because it isolates "launch
structure" from "materialization" if attribution is ever disputed.

### 3.4 Honest E0 speedup range for the FULL OP

| | floor (traffic) | + unpack/compute overlap | predicted full-op | vs 28.87 ms |
|---|---:|---|---:|---:|
| E0a (draft as-is) | 5.4 ms | SLM barriers, 98-B loads | **6–9 ms** | **3.2–4.8×** |
| E0b (TN=128 + FP16 A) | 2.5 ms | same | **3–4.5 ms** | **6.4–9.6×** |

Recoverable fraction of the 25× dense-gap: E0b lands at ~11–17 TFLOPS-equivalent, i.e.
recovers ~50–70 % of the log-gap. Dense-class (~46 TF / ~1.1 ms) is NOT reachable:
IQ3 unpack ALU, 98-B-strided weight loads, and SLM barrier serialization put a hard
~2–2.5 ms floor on this design, and a dense GEMM pays none of those. Do not promise it.

**Gate verdict: the ≥2× isolated gate (≤ 14.4 ms) is plausible with ~2× margin at E0a
and should NOT move.** The end-to-end ≥15 % gate is not binding if the kernel gate
passes: with gate/up at 7 ms and down still on the old path, a layer drops 86.6 →
44 ms → 512-token ubatch ≈ 3.24 s → ≈ 158 pp vs 101.2 pp baseline (+56 %); with down
supported (E1) at the same quality, ≈ 227 pp (+124 %). These are projections for
planning, PROVISIONAL, not promises.

Staged sequence (refines the prior direction; keep this order):
- **E0a**: wire + gate the existing draft, gate/up shape only (both projections reuse it).
- **E1**: generalize the guard for down (K=2048: 8 super-blocks; guard
  `(K % 256 == 0 && N % 64 == 0)`, drop the hard `[4096,2048,216]` pin in favor of
  declared-shape checks), same kernel.
- **E0b/E2**: the one bounded performance revision — TN 64→128 (halves activation
  re-reads) + one-shot F32→FP16 activation convert (4 MB) into the shared pool buffer.
- **E3 (optional, last)**: fused gate+up with SiLU×mul epilogue (saves the 25 MB
  intermediate round trip and one launch), only after all gates pass.

## 4. Parity / validation harness design

Order is deliberate: cheap and bit-exact first. All runners live in
`scripts/tmp/` until promoted (contract §14).

### 4.1 P0 — bit-identical FP16 reconstruction (gate: 0 mismatched bits)

- Oracle: wrap `dequantize_block_iq3_xxs<sycl::half>` (dequantize.hpp:1421) in a
  trivial SYCL kernel over N blocks → buffer O.
- Candidate: a test wrapper invoking the main kernel's unpack loop (same
  `reconstruct` + same `(n, k-tile) → block` indexing as mul-mat-id-iq3.cpp:168-182)
  → buffer C.
- Compare bitwise (`memcmp` on host). Iterate on a deterministic sample (every 64th
  block of every 8th expert); run the FULL tensor (7.08 M blocks, 1.8 GB FP16 oracle —
  allocate from the pool, one pass) once before any end-to-end run.
- The arithmetic orders were verified identical on paper (section 2.3); failures here
  mean indexing/plumbing bugs (wrong ib32, misaligned aux32 read, swapped grid1/grid2).

### 4.2 P1 — relative squared-L2 ≤ 1e-3 … ≤ 1e-5 vs the existing FP16 route

- Fixed seeded inputs: deterministic F32 activations + an IQ3_XXS tensor built via
  `ggml_quantize_chunk` from a seeded source; run `ggml_mul_mat_id` on the SYCL backend
  with `GGML_SYCL_IQ3_GROUPED=0` (old path) and `=1` (new), dump both F32 outputs.
- Metric: `Σ(a−b)² / Σa²` at test shapes (2048, 512, 4096) and (4096, 512, 2048) plus
  one ragged-routing case (n=97, prime, to stress tails). Gate ≤ 1e-5; investigate
  anything above 1e-6 even if passing.

### 4.3 P2 — backend-ops NMSE ≤ 5e-4

`test-backend-ops` already carries the exact K216 shapes locally
(tests/test-backend-ops.cpp:9348-9357; `test_mul_mat_id::max_nmse_err()` = 5e-4).
Run test mode with the env ON; the harness compares SYCL vs CPU reference. The n=512
cells are the gate; n=1/32 cells double as no-regression checks for the dispatch guard.

### 4.4 P3 — matched end-to-end prefill A/B (gate ≥ 15 %)

- Cell: the P3 anchor — p7000/g1 cold, `-c 8448`, dual card, `-sm layer -ncmoe 15
  -ts 65,35`, `-ub 512`, `-ngl all -fa on`, `-ctk f16 -ctv f16 -t 8` (match
  `worklogs/P3-20260921.md` exactly), unique leading nonces, prefix-cache guard per
  contract §8; n=3 screen → fresh n=5 finalist. Decode regression check p512/g128
  afterward (the guard must make decode a no-op — verify the mmvq fused path still
  takes `ne12 == 1`).
- Files to patch (complete list — nothing else):
  1. `~/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp` — include
     `mul-mat-id-iq3.hpp`; in `ggml_sycl_mul_mat_id` (line 4770), immediately after
     the `ne12 == 1` mmvq_fused block (line 4789) and BEFORE the `ids_host`
     allocation/memcpy (line 4791) — that placement is the entire point: no D2H, no
     `wait()` — insert:
     `if (ggml_sycl_mul_mat_id_iq3_grouped(ctx, dst)) { return; }`
  2. `~/llama.cpp/ggml/src/ggml-sycl/mul-mat-id-iq3.cpp` — the kernel
     (exists; E1 guard generalization, E0b revision here).
  3. `~/llama.cpp/ggml/src/ggml-sycl/CMakeLists.txt` — option exists
     (lines 35-37), no change needed for E0a.
  4. `tests/test-backend-ops.cpp` — cases already present (9348-9357).
  - Do NOT touch `mmvq.cpp`, `dmmv.cpp`, or the `ne12 == 1` decode branch: decode
    keeps `ggml_sycl_mul_mat_id_mmvq_fused` (ggml-sycl.cpp:4669) untouched.

## 5. Build / run recipe

```bash
source /opt/intel/oneapi/setvars.sh --force            # or export PATH=/opt/intel/oneapi/compiler/2026.0/bin:$PATH
cd ~/llama.cpp
cmake -B build-iq3 -S . \
  -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DGGML_SYCL_F16=ON \
  -DGGML_SYCL_IQ3_GROUPED_EXPERIMENTAL=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-iq3 --target ggml test-backend-ops -- -j 8
export ZE_AFFINITY_MASK=0            # single-card kernel work
export GGML_SYCL_IQ3_GROUPED=1
```

`GGML_SYCL_F16=ON` is mandatory (the kernel compiles to a no-op returning false
without it, mul-mat-id-iq3.cpp:56-60). JIT (default, no `-fsycl-targets` AOT) is
correct for iteration; AOT for spir64_gen with the B70 device id only if launch JIT
latency ever shows (it will not at 3 launches).

test-backend-ops filter semantics (verified, tests/test-backend-ops.cpp:1286-1314 and
10486-10489) — correcting the "regex" shorthand:

- `-o` is NOT a regex: comma-separated EXACT matches of either the bare op name
  (`MUL_MAT_ID` — runs all ~100 variations, slow) or the full case string with all
  eight vars (`MUL_MAT_ID(type_a=iq3_xxs,type_b=f32,n_mats=216,n_used=6,b=0,m=2048,n=512,k=4096)`,
  paren-aware comma splitting — shell-quote it). A partial string like
  `MUL_MAT_ID(type_a=iq3_xxs)` matches NOTHING.
- `-p` IS a regex (`std::regex_search` over the case's `vars()` string). The working
  one-field form: `test-backend-ops test -o MUL_MAT_ID -p 'type_a=iq3_xxs'`.
- Perf mode reproduces the baseline harness:
  `GGML_SYCL_IQ3_GROUPED=1 ./build-iq3/bin/test-backend-ops perf -b SYCL -o MUL_MAT_ID -p 'type_a=iq3_xxs'`
  (compare against `results/2026/09/b70-improvement-probes-20260920/perf-k216-prefill-n512.txt`,
  same machine state; warmup + n≥3 per protocol §9).

Iteration cost per revision: cold build 5–10 min; incremental `ggml-sycl` 1–2 min;
filtered test-backend-ops correctness pass seconds, perf pass ~1–2 min; the expensive
step is only the end-to-end A/B (~83 GB model load, ~30 min including cooldowns).
Budget: parity plumbing 1–2 sessions, E0a gating 1 session, E1 + E0b 1–2 sessions,
A/B + close-out 1 session.

## 6. Stop rules, recording, and fallback

Stop rules (prior direction, refined — all measured against the n=512 isolated op):

1. **Reconstruction alone > 10 ms** in a probe that unpacks without GEMM (healthy is
   < 1 ms; the pure IQ3 read is 1.5 ms at 1.3× amplification) → the tile design is
   broken; stop.
2. **Spills persist** after `-zeopt-large-register-file` AND one tile reduction
   (TN 128→64 or TM 16→8) → stop; do not chase further tile shapes.
3. **Full op < 2× (> 14.4 ms) after ONE bounded revision** (E0a → E0b counts as the
   one revision) → gate failed; record and fall back.
4. **Two distinct root-cause parity failures** (P0/P1/P2) → stop (P3 spec's
   two-correctness-failures rule).
5. **Isolated ≥ 2× but e2e < 15 %** → attribution investigation only (where did the
   win go — down op? non-routed layers?); ONE bounded fix attempt, then close. Do not
   tune the kernel against an unattributed e2e number.
6. Any decode regression (p512/g128) or capacity loss from the pool allocations →
   revert the hook, keep the kernel behind the env guard, record.

If the gate fails, record: verdict + evidence paths under
`results/2026/09/b70-improvement-probes-20260920/` (or a fresh dated root), LEVERS.md /
OUTCOMES.md entries, ledger RL-011 status, and the fallback assessment (route (b)
`gemm_batch` diagnostic: 1 launch, no host sync, keeps materialization — expected
1.3–1.8×, below gate, useful only to complete the attribution record).

What NOT to retry (measured dead, do not reopen): ubatch/flag sweeps (exhausted at
1024; 2048 saturated; 4096 OOM); SYCL graph capture over MUL_MAT_ID (upstream disables
graphs for this op — ggml-sycl.cpp:5686-5691); requantizing K216 to INT4/FP8 (a
representation decision with its own quality gates, not a kernel lever); power-cap
changes (measured zero prefill effect on MoE); the P1 device-ID repair as a speed
lever (withdrawn, +3.54 %, gate failed); decode-path/MMVQ changes (out of scope,
decode is a different algorithm per the dispatch-separation rule).

Ongoing labeling: everything above is PROVISIONAL — NOT FOR PUBLIC HEADLINE until the
lane, evidence, and claims audit pass (contract §12).
