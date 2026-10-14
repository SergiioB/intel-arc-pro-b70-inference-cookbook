# B70 M=1–4 quant-layout shortlist and falsifying benchmark specification

**Status:** E0 static analysis — not a performance result or implementation authorization. **No execution/build authorized. RL-122 EXL3 holds remain in force.** No quality-equivalent format winner is established.

## One-page decision table

Priority ranks engineering confidence conditional on a quality-compatible artifact, not demonstrated end-to-end gain. Costs are hand-estimated SIMD16-equivalent decode/preparation operations per 32 weights, excluding loads, dots, activation quantization and final reduction; metadata extraction is amortized. These are not emitted ISA counts.

| Priority / layout | Stored bpw | Estimated preparation / 32 weights | LUT / scratch requirement | M1 DP4A / M4 DPAS feed | BW assessment and capture condition |
|---|---:|---|---|---|---|
| **1: contiguous signed INT4, FP16 scale/G128** | 4.125 | 4–8; direct packed DPAS can avoid expansion | None / none intended | Excellent after M1 byte expansion / native integer operand | M1/M4 unmeasured; **500–580 GB/s is a hypothesis target**, not a forecast. Repack before capture; fixed scale buffers. |
| **2: Q4_0** | 4.5 | 5–9: nibble extraction + centering | None / none intended | Excellent after expansion / good with group-scale handling | Unmeasured; simplest GGUF decode hypothesis. Static blocks. |
| **2–3: Q4_1** | 5.0 | 4–8 plus offset × sum(A) correction | None / none intended | Good; affine correction outside dot loop | Unmeasured; more bytes than Q4_0. Keep activation sums device-side. |
| **3: IQ4_XS** | 4.25 | 10–20 + two SIMD16 LUT accesses | 16-entry int8 LUT; compiler lowering may expand storage | Good after nonlinear nibble→int8 remap / not direct linear INT4 | Unmeasured; 5.56% fewer bytes than Q4_K can be erased by LUT cost. Immutable table. |
| **3: Q4_K core** | 4.5 | 10–20: packed scale/min decode + correction | None / none intended | Good after metadata unpack | Unmeasured. **Q4_K_XL is a mixed-tensor recipe**, requiring a tensor histogram, not a separate block format. Stable dispatch. |
| **4: MXFP4 E2M1 + E8M0/G32** | 4.25 | 14–24: arithmetic remap + sign | **No LUT necessary** / none intended | Exact int8 remap feeds either path; no native FP4 | Unmeasured. Fixed buffers; handle exceptional scales and signed-zero semantics. |
| **Quality-unranked: TQ2_0** | 2.0625 | 8–16: two-bit extraction and ternary centering | None / none intended | Good after signed-byte reconstruction | Unmeasured; byte advantage does not establish unchanged quality. Static unpack. |
| **Quality-unranked: TQ1_0** | 1.6875 | 20–40: radix-3 unpack estimate | None / none intended | Good after unpack; constant-division lowering decisive | Unmeasured; avoid general divide loops. Static unpack. |
| **Quality-unranked: IQ2/IQ1 variants** | Variant-dependent | Roughly 16–40 + codebook accesses; too variant-dependent for a universal count | Typically kilobyte-class codebooks; register/SLM/cache tradeoff | Dependent gather/sign/scale reconstruction before dots | Unmeasured; artifact and exact variant mandatory. Preallocated LUTs capture-safe; occupancy unproven. |
| **Held reference: EXL3 nominal 4bpw** | ≈4 | Historical ~336 **total SIMD8-class** instructions/512B; incomparable to estimates above | State/codebook arithmetic and operand regions | Expensive preparation ahead of matrix math | Supplied M4 anchor 381 GB/s; B_FAKE 636 logical-byte GB/s. No new EXL3 probe. |

### Decision and arithmetic

**First engineering choice:** contiguous signed INT4 G128 with separate M1 vector-dot and M4 integer-DPAS designs. This is a kernel/packing hypothesis in an established format family, not a new quantization method. If “new” excludes linear INT4, **MXFP4 arithmetic remap is the cleanest new hypothesis**, not an established winner.

MXFP4 needs no LUT: doubled unsigned E2M1 magnitudes are `[0,1,2,3,4,6,8,12]`; apply sign and group scale/2. Shifts/selects can construct the exact int8 values. This saves dependent lookup, not all conversion instructions. DP4A and integer DPAS still require integer activations: changing A16→A8 changes numerics and must not silently replace exact verification.

The 608 GB/s pin ceiling is not the same metric as 636 logical-byte effective GB/s. `1−381/608 = 37.3%`; `1−381/636 = 40.1%`. The respective unchanged-byte isolated ceilings are 1.596× and 1.669×, not end-to-end gains. Cold-DRAM predictions must not exceed pin rate; logical-byte rates need cache/traffic reconciliation.

Bandwidth required merely to match nominal 4bpw at 381 GB/s, equal yield: **392.9 GB/s** INT4-G128; **404.8** IQ4_XS/MXFP4; **428.6** Q4_0/Q4_K; **476.3** Q4_1. Formula: `381 × candidate_bpw / 4`. At illustrative 550 GB/s, G128 weight-only speedup is 1.400×. With affected wall fraction f, whole-step speedup is `1/(1−f+f/1.400)` — 1.167× for f=.5 and 1.296× for f=.8. No f or yield is established here.

EXL3's ~336 instructions comprise approximately 4 DPAS + 3 sends + ~330 ALU/moves (rounded): state reconstruction, codebook decode and operand-region moves. Finite trellis diameter bounds history, **not a minimum instruction count**. No irreducibility claim and no reopening of stopped classes follows.

## Standalone falsifying microbenchmark specification

### Authorization and hypothesis

- **Specification only.** No build, GPU run, conversion, model download, or EXL3 reopening is authorized.
- Proposed changed variable: K-contiguous signed INT4-G128 packing plus regime-specific M1 vector-dot / M4 DPAS kernels, against a same-quant reference implementation.
- Hypothesis: reducing decode/preparation issue cost while retaining compressed weights yields at least 500 GB/s metadata-inclusive effective streaming bandwidth on predeclared dominant matrix families, with a priced whole-decode gain of at least 10% at unchanged outputs.
- A future MXFP4 arm is a separate format/quality experiment, not a same-artifact control.

### Inputs and controls

1. Before authorization, identify an exact repository commit, compiler/runtime versions, immutable weight artifact, tensor histogram, model shapes, and reference dispatch. These are **TBD**, not inferred from route names.
2. Freeze dominant production `(N,K)` families and padding/tail cases; run **M=1,2,3,4** independently. M4 may mean verify rows or independent requests; label which. Never treat four rows as four accepted tokens.
3. Same integer weights, scales and activation representation in candidate/control. Include signed extrema, zero blocks, scale boundaries and odd/tail shapes in correctness cases. If A8 quantization is introduced, preserve an A16 quality/reference arm and explicitly account for the numerics change.
4. Control: incumbent same-format captured kernel. Optional predecoded-weight control measures arithmetic behavior but has different bytes and must not be presented as an inference speedup. **No new EXL3 B_FAKE run.** Retained EXL3 numbers remain contextual only.

### Byte, instruction and resource budget

- Weight traffic: `N*K/2 + 2*N*ceil(K/128)` bytes for the simple padded grouping convention; record padding and alignment separately. Include activation input/output, sums, scales and any temporary traffic in total-byte accounting.
- M rows share one weight read only if the implementation actually does so. Report unique logical weight bytes separately from total logical and measured DRAM bytes.
- Coalesce K; choose block loads/cache hints after shape audit. Accumulate in GRF; no scratch spills intended. No mandatory SLM staging; if added, price bytes, barriers and occupancy.
- Start with explicit subgroup16 where applicable and a 128-GRF target. Any 256-GRF variant requires compiler resource proof and its resident-thread/occupancy consequence; ESIMD does not inherently prevent spilling.
- No persistent-grid assumption, global barriers, cross-card work, host routing, D2H IDs, host wait or dynamic per-step allocation.

### Build and compiler gate — future lead-assigned window only

- Source oneAPI; check storage live; dedicated build directory.
- Record exact commit and complete build command; target BMG with `-fsycl-targets=spir64_gen -Xs "-device bmg"` plus the selected register-allocation flags.
- ISA analyst must verify dot/DPAS emission, operand types, preparation instruction counts, moves, GRF allocation, scratch size, sends and loop geometry **before any GPU run**.
- A compile or resource failure is retained and ends this arm pending review; no silent shape/quality relaxation.

### Correctness and capture gate

- CPU reference for exact integer dots and offset corrections. Bitwise match integer accumulators; prove no overflow for the declared dimensions/scaling boundaries.
- Floating scaling/epilogue: predeclare an error bound against reference and test extrema. No generic error tolerance is supplied here because accumulation and epilogue types are TBD.
- Exact-path changes require retained greedy-output parity. A16→A8 or new weight formats require a separate approved quality objective and cannot be called exact verification merely because the integer kernel is correct.
- Capture and repeatedly replay fixed-pointer, shape-stable buffers. Read source and API traces to verify no allocation, host reads, D2H expert IDs or stream waits inside the step.

### Timing design — future authorized run only

- Empty-GPU/ownership and live VRAM gates; as-found power/clocks recorded and restored. One owned campaign; never displace unrelated work.
- Preallocated weight ring larger than L2 and sufficient to avoid resident-weight-only conclusions. Freeze ring size and cache policy before timing; include a separately labelled cache-resident diagnostic if needed.
- Graph replay only. One discarded same-shape warmup; **three interleaved paired screen repetitions**, then **five fresh confirmation pairs** for a surviving candidate. Predeclare thermal regime, repetitions per timed block, duration and timeout.
- Report median, dispersion and every raw sample; kernel, activation-quantization and complete captured-step timings separately. Effective GB/s = explicitly named logical bytes / time; measured DRAM throughput, if collected, is a separate counter metric.
- Include activation quantization, activation sums, group scaling, affine corrections, output epilogue and launch costs in the complete-step comparison.

### Pass, falsifier and handoff

- Provisional kernel pass: correctness/capture/compiler gates pass; ≥500 GB/s on **predeclared dominant families**, with no material M1/M4 regression hidden by aggregation. Family weights and regression tolerance must be frozen before running.
- **Falsify** the proposed high-bandwidth hypothesis if dominant families fail that target; diagnose rather than claim a format win. A lower-bandwidth useful result may be recorded but does not pass this predeclared hypothesis.
- Reject advancement if measured removable wall-time share prices below 10% whole-decode gain, or quality/capture fails. For M4 verification, include draft cost and accepted-token yield: optimize accepted tokens per complete round, not GEMM rows/s.
- Isolated pass → profiler evidence → correctness specialist → speculation economics for verify → matched end-to-end bench. Only a separate reviewed route decision may adopt a result.

## Evidence and limitations

- [Current research contract](../../AGENTS.md#L160-L165).
- [DPAS rate-class authority](xmx-dpas-rate-classes-20260915.md#L19-L79).
- [EXL3 instruction decomposition and diagnostic design](b70-kernel-forensics-next-20260930.md#L153-L213).
- [Retained format/route findings](b70-format-route-lessons-20260930.md#L53-L95).
- Retained M4 ablation artifact (`../../results/2026/09/b70-m4-flag-ablation-20260930T143631Z/RESULTS.md`) was located but its raw content was not inspected in this analysis; 381/636 anchors are supplied scope inputs, not newly audited measurements.

No source or benchmark execution occurred. Broad source-discovery timed out, so exact quant source/ISA counts remain unverified. No comprehensive-reference read was needed. Unknowns: exact kernel tree/commit, shape census, resource budgets, sustainable per-format BW, target quality, activation policy, affected wall share and verify yield.

## Addendum — census correction (2026-10-09, later; isa-analyst session + lead arbitration)

**TQ1_0 byte count: the table above stands.** The census first claimed 56 B/1.75 bpw (`qs[50]`) and then contradicted itself; the authoritative ggml-common.h @ `50e3e3e480c80b9ee0ea067e475602d540dc62c8` static-asserts `2 + 4 + (256-16)/5 = 54` B = **1.6875 bpw** (upstream comment "// 1.6875 bpw"). The census's genuinely new findings stand independently: **MMQ is hard-disabled in llama.cpp SYCL** (`ggml_sycl_supports_mmq()` returns false, accuracy TO-DO — all tiled quantized matmul goes dequant+GEMM or CPU; MMVQ `__dp4a` remains for M=1-8), TQ1_0/TQ2_0 are rejected to CPU by `ggml_sycl_can_mul_mat()`, Strata dequants to FP16 buffers then runs oneMKL FP16 GEMM, and the retained EXL3 M=4 ablation disassembly shows **zero integer DPAS** (`:s4`/`:s8`) — every DPAS emitted is FP16 (`:hf`). Consequence for the priority-1 design: the M=1-DP4A / M=4-DPAS dual-path INT4 G128 kernel is not just faster-decode, it is the *only* route to the integer XMX class on this stack, and any new integer format must ship with a parity story because the one historical tiled-integer attempt (MMQ) was disabled for accuracy.
