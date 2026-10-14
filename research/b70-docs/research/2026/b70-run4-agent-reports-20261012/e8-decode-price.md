---
title: "Run-4 subagent report: e8-decode-price (2026-10-11)"
type: research
date: 2026-10-11
status: provisional
---

# E8DecodePrice — E8RVQ4 kernel decode price (lane closed 2026-10-11 ~23:5x)

**Status: LANE CLOSED by lead decision before any build.** E8RVQ4 is killed on quality mathematics (FormatTheory: Gaussian MSE 2.37× D(4) vs Lloyd-Max scalar 2.43× and uniform scalar 2.95×; the trellis class sits at ~1.1×). Per the closing directive I stopped E8 work immediately: **no kernel was written, nothing was compiled, no ocloc disasm exists, and no GPU1 time was ever requested or used** (no "E8 GPU GO" was ever issued; the final kill made the gate moot). Everything numeric below that is not a cited measurement is an **[UNVERIFIED] estimate** produced before the stop. What is fully grounded: prior-evidence extraction, the exact E8P12 decode rule, the toolchain, and the instruction-budget arithmetic — all reusable by XeTrellisCode (handed over).

## 1. Verified evidence (read this round)

### EXL3 instruction-count anchor (the price baseline)
- `research/2026/b70-kernel-forensics-next-20260930.md:169-177`: stock `DpasKernel<4,2,4,4>` inner loop ≈ **336 instructions per 512 B** weight tile-group = 4 dpas + 3 sends + ~330 ALU/mov for `build_k4` (shl/or/shr/mul/dp4a/mad + replicate-region movs); loop body ≈ asm lines 123–459 of `results/2026/09/b70-c2-repair-20260930T085200Z/v2/isa-torch/.text._ZTSN4exl310DpasKernelILi4ELi2ELi4ELi4ELb0EEE.asm` (775 lines; re-read today, structure confirmed: `(1|M0)` packed byte-state mul/mach/macl/add3 chains interleaved with `(16|M0)`/`(32|M0)` decode/mov, `load.ugm.d32x64t` sends, `dp4a (32|M0)`). "SIMD8-class" in the docs = each asm line counted as one issue.
- Same doc §2: 13.12 GB/pass ÷ 512 B ≈ 25.6 M row-iterations × 336 ≈ **8.6 G issues per M=4 pass** → ~250 G issues/s ≈ 3.1 issues/cycle/Xe-core (32 cores @ ~2.5 GHz) vs ~8/cycle SIMD8 theoretical. Decode/state/mov ≈ 330 of 336.
- RL-119 (`results/2026/09/b70-m4-flag-ablation-20260930T143631Z/RESULTS.md:46-77`): `EXL3_DEBUG_B_FAKE` (loads+dpas+partials, no trellis decode) = **636 logical-byte GB/s** route anchor vs **381 GB/s** real decode-bound; B_FAKE numerics wrong by construction; 636 is not a physical ceiling.
- Speed bars (lead/FormatReview): 1.30× route decode at f≈0.75 ⇒ kernel gain 1.444× ⇒ **≈550.3 GB/s logical at 4.0 bpw** (551.2 at E8's 4.00625); weaker kernel-only bar **495.3 GB/s** (1.30×). 5.125 bpw needs 634–705 GB/s — above the 608 GB/s pin, dead.

### Toolchain (verified on host, no GPU opened)
- `source /opt/intel/oneapi/setvars.sh` → **icpx 2026.1.1.20260724**, **ocloc 26.x at /usr/bin/ocloc** (printed today).
- Build pattern to reuse verbatim: `scripts/tmp/rq25_soa_20261011/build_offline.sh` (icpx `-fsycl` TU → `clang-offload-bundler --unbundle sycl-spir64` → `llvm-spirv --spirv-ext=+SPV_INTEL_subgroups` (only that ext; `+all` breaks ocloc IGC via SPV_KHR_untyped_pointers) → `ocloc compile -spirv_input -device bmg` → `ocloc disasm`); large-GRF: `build_g256.sh` `-Xsycl-target-backend "-device bmg -options \"-cl-intel-256-GRF-per-thread\""` (128-GRF incumbent spills; 256 GRF gave 0 spills and 16-B `load.ugm.d32x4.a64` q-plane loads — `research/2026/b70-run3-agent-reports-20261011/rq25-isa-round3.md` §2).

### Exact E8P12 decode rule (extracted from source; confirmed against the codec)
QuIP# `lib/codebook/latticee8_padded12.py::get_full_grid`, repo https://github.com/Cornell-RelaxML/quip-sharp @ **1d8f873e9a2a8b86b12bb1064c312c5689b77d98** (GPL-3.0; logic reimplementation, cite on use); matched line-by-line against our codec `scripts/tmp/e8rvq_20261011.py:104-124` (`_parity8`, `e8p_decode`) and `:29-33` (`_PERM=(0,2,4,6,1,3,5,7)`, `_INV_PERM=(0,4,1,5,2,6,3,7)`):

- Code `c` uint16: `abs_idx = c >> 8`, sign byte `b = c & 255`.
- `p = parity8(b)` (XOR of its 8 bits: `b^=b>>4; b^=b>>2; b^=b>>1; &1`); `fixed = b ^ p` (flips bit 0 → always even parity).
- Packed abs word `g = E8P_PACKED_ABS[abs_idx]` (int32; nibble `j` = abs-grid coord `_PERM[j]` mapped `v → 2v+8`; the D8 parity is baked into coord 7's nibble, so nibble decode `(n-8)*0.5` already carries its sign).
- Value for output coord `i`, slot `j = _INV_PERM[i]`: `v_i = (1 − 2·bit_j(fixed)) · ((g>>(4j)&15) − 8)·0.5 + (0.25 − 0.5·p)`.
- All codewords are multiples of 0.25 in [−3.75, 3.75] → **exact in fp16 and fp32**; bit-identical decode is achievable on GPU.
- RVQ4 packed contract (codec `pack`/`unpack_decode`, `:404-421`): `codes uint16[2,out,in/8]` (stage 1, stage 2; `codes[s,o,kk]` covers weights `[8kk, 8kk+8)`), `scales fp16[2,out]`; `w8 = s1·E8P(c1) + s2·E8P(c2)`, 4.0 bpw + 32 bits/row. Codec provenance header itself pins the same QuIP# commit.

CUDA reference for in-kernel decode idioms (our reimplementation would have mirrored the logic, not the code): `quiptools/quiptools_e8p_gemv.cu` @ same commit — fp16 trick `((g<<s)&0x00f000f0)|0x50085008` + half2-add of `0xd080d080`/`0xd090d090` encodes `v = (n−8)/2 ± 0.25` with 2 values per op; sign flip `g ^= spread(signs)·14` (nibble `n→16−n`); their pack pre-transposes sign bits across codes (`s & 0x11111111` trick), which our per-code uint16 layout cannot do.

## 2. ISA price — NOT MEASURED (killed before compile); budget arithmetic [UNVERIFIED]

Comparison unit (from the doc, but with a convention hazard — see below): inner-loop **instruction issues per 512 B of packed weights**; EXL3 = **336** (2.63 per 4-B code; B_FAKE floor 4 dpas + 3 sends ≈ 7). Bars: **≤258/512 B** for the 495 GB/s (1.30×) hypothesis, **≤233/512 B** for the 550 GB/s (1.444×) one.

**Convention hazard (unresolved, would have been settled by the ocloc disasm):** the forensics doc states "per-thread weight reads are 512 B at NT4" but its throughput arithmetic (13.12 GB ÷ 512 B = 25.6 M row-iterations; 8.6 G issues → 250 G/s → 3.1/cycle/core) implies 512 B **total per SIMD32 group-body** (16 B/lane), with mostly mixed-width issues (~12 codes/issue average). Under "512 B per lane" the required decode chain budget is 32× looser than under "512 B per group". A compiled E8RVQ4 body compared against the same arithmetic (or a recount of EXL3's own asm widths) is the only way to close this.

Estimated per-code-pair decode chain (lane = code-pair, one SIMD32 issue processes 32 code-pairs; body = 4 code-pair positions × 32 lanes = 128 code-pairs = 512 B), **[UNVERIFIED] hand estimate, no ISA emitted**:

| formulation | per-code-pair issues (unpack+parity+sign+lookup+convert) | + dot (2 stages × 8 mad) | chain total | body = 4 × chain + ~8 (2× d32x4 code-plane loads, epilogue) | vs 258 / 233 bars |
|---|---|---|---|---|---|
| naive (parity fold + bit→nibble spread inline, 2×1 KiB SLM tables) | ~34 | 16 | ~50 | **~208** | under 233 |
| w/ precomputed `TBL_B[b]`=flip-mask·14, `TBL_C[b]`=trick offset (2 more 1 KiB gathers) | ~26 | 16 | ~42 | **~176** | under 233 |
| worst case (shifts un-fused in the fp16 trick: 4 shl+4 lop3+4 add per code) | ~48 | 16 | ~64 | **~264** | over 258 |

Design notes captured for reuse (XeTrellisCode):
- Factor per-row fp16 scales **out** of the K-loop: two per-stage fp32 accumulators per M row; `y = s1·acc1 + s2·acc2` once per row. Loop carries no scale math. (Bit-exact `w` decode = separate check mode; `y` gets a declared relative tolerance, fp32 reduction order differs.)
- The `_INV_PERM` coordinate shuffle can be folded into the activation indexing once (`x̃[8t+j] = x[8t+_PERM[j]]` or an FMA source region) — zero per-code cost.
- Cheapest decode = QuIP# CUDA idioms: 256×32-bit packed-abs table + precomputed 256×32-bit sign-flip mask table + parity-offset table (all ≤2 KiB total, SLM or const); fp16 trick gives 2 values/op; all values exact in fp16.
- Two-stage RVQ doubles every per-code cost (2 lookups, 2 parities, 2 tricks per 8 weights) — this is why the chain is 15–25× the per-byte rate of EXL3's packed trellis state machine at the *same byte budget per issue* under the strict reading, or comfortably under it under the loose one. The estimate spans the verdict; that is exactly why the compile was the next step.
- INT4 W4A16 control: same body structure, ~25–30 issues per 4 B (8 nibbles → trick extract + G128 scale + 8 mad) — would have been the A/B arm.

## 3. Deliverables / handoff state

- No files created under `scripts/tmp/e8rvq_kernel_20261011/` (stopped before the build). No GPU, no guard needed (nothing ran).
- Codec contract (owned by E8Codec, now also closed): `scripts/tmp/e8rvq_20261011.py` — read and matched to QuIP# this round; sha-pinned grid verification inside its header (`E8P_PACKED_ABS` sha[:16] efc2c03c60acd955 etc.).
- Coordination done: decode rule + packed layout sent to **XeTrellisCode** with the build-toolchain recipe (build_offline.sh pattern, `+SPV_INTEL_subgroups`-only llvm-spirv, `-cl-intel-256-GRF-per-thread`, ISA-price convention + bars) so the trellis-code lane reuses the tooling; units correction sent to **FormatReview** (per-weight instruction estimates must be re-expressed per 512 B packed; EXL3 anchor 336/512 B ≈ 0.66 instr/byte-equivalent; SLM gathers are send-pipeline ops).
- QTIP HYB LUT pricing was reassigned to XeTrellisCode before I started it (lead redirect).

## 4. Acceptance items — final disposition

| acceptance | disposition |
|---|---|
| Instruction count table vs EXL3 | **Estimate only** (§2 table, [UNVERIFIED]); no ocloc disasm — killed before compile. EXL3 row is measured/cited. |
| GB/s table per shape and M | Not run (no GPU work ever authorized for E8; format killed). |
| Correctness check numbers | Not run; the declared tolerance class would have been **relative ≤1e-3 on y** (fp32 reduction-order differences over ≤17408 terms) plus a **bitwise** w-decode check (E8P values are exact multiples of 0.25). |
| Exact commands | Toolchain check: `source /opt/intel/oneapi/setvars.sh; icpx --version; ocloc` (ran, output above). Reads cited with file:line throughout. |
| Guard before/after | Not applicable — zero GPU1 contact this lane. |
| Verdict vs 495 GB/s bar | **Undetermined at compile level; format killed on quality** (2.37× D(4) Gaussian MSE kills it before speed matters). Under the doc's strict group-body reading the two-stage decode chain lands ~1.3–1.6× under EXL3's 336/512 B only in the optimized form and ~1.2× over in the naive form; under the per-lane reading it is comfortably under. A ~1 h compile pass would have closed this; the lead chose not to spend it.

Commands/evidence provenance: all numbers above are either cited by file:line or explicitly [UNVERIFIED]. QuIP# source was fetched read-only from raw.githubusercontent.com (commit pinned) into /tmp/quipsharp/ for rule extraction.
