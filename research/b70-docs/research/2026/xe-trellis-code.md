---
title: "XTQ code candidates: Xe2-native trellis weight formats — K2/K3 acceptance"
type: research
date: 2026-10-12
status: provisional
---

# xe-trellis-code — per-code acceptance table (K2 CPU MSE + K3 BMG ISA)

K4 GPU results in the K4 section (2026-10-12, GPU1): xe_ih2x2 = 342-420 GB/s
vs exl3 = 465-506 at 4.0 bpw -> the 550 GB/s gate fails (structural DRAM
ceiling ~500 for this GEMV class on B70).  Scratch:
/mnt/<nvme-disk>/b70-run4-20261011/xtq-code/

## Format common to all candidates

Tile = 16x16 weights. Circular bitstream T*K bits per tile (T=256, K=4 at 4.0 bpw ->
128 B per tile). 16-bit state window, MSB-first, tail-biting. Stream step t emits V
weights. The tile weight->stream permutation pi is chosen at ENCODE time (offline,
free) so that each V=2 pair lands directly in a DPAS B-operand VNNI dword:

  pair t covers (k,n) = (2*(t>>4), t&15) and (2*(t>>4)+1, t&15)
  stream pos 2t <-> w_idx 32*(t>>4)+(t&15); 2t+1 <-> +16

pi is a fixed permutation of 256 positions; trellis MSE is permutation-invariant,
so quality numbers apply unchanged. Decoded pair = packed fp16x2 written straight
into the B operand (no pairing shuffle anywhere).

State extraction for K=4,V=2 (states on byte boundaries, t=4g+u per lane g):
  u=1: w>>16 (1)  u=2: (w>>8)&0xFFFF (2)  u=3: w&0xFFFF (1)
  u=0: ((pw&0xFF)<<8)|(w>>24) (~3) + word rotate (~2, amortised/4 phases)

## Candidates (CODE id in kernel)

| code | eval per 32-lane phase | eval ops/state |
|---|---|---|
| 3 qtip_hyb | mul+add (p=s(s+1)), shr+and (idx), gather fp16x2 512x4B tlut, xor sign | ~5 |
| 4 xe_u2 | mad + bfn (mask&xor fused) | ~3 |
| 5 xe_ih2x2 | 2x (mad + bfn) + packed fp16x2 add | ~7 |
| 6 xe_bs2 | 2x (mad + dp4a) + or-pack | ~6 |
| 7 xe_hyb10 | mad, shr (top10 idx), gather fp16x2 1024x4B tlut | ~3.5 |

exl3 baseline (mul1): mad + dp4a + fp16 affine, V=1, ~85.5 non-mem instrs/512B
(prior disasm bvall, loop body 355 instrs / 4 tiles — flag-ablation 20260930).

## K3 — BMG AOT ISA (compile-only; ocloc 26.35 disasm, simd32, 128 GRF, 0 spills)

Per-512B (one 16x16 tile) issue counts; "decode" = non-send/non-dpas/non-guard
instructions; "loop" = full inner-loop total incl. loads/dpas/guards.

| code | dec kern total (4 tiles) | decode instrs/512B | decode/w | vs exl3 0.334 |
|---|---|---|---|---|
| exl3 mul1 (prior) | 355/loop-iter | ~85.5 | 0.334 | 1.00x |
| xtq_dec c3 qtip_hyb | 300 (233 nonovh) | ~58 | 0.228 | 1.47x |
| xtq_dec c4 xe_u2 | 256 (197 nonovh) | ~49 | 0.192 | 1.74x |
| xtq_dec c5 xe_ih2x2 | 247 (200 nonovh) | ~50 | 0.195 | 1.71x |
| xtq_dec c6 xe_bs2 | 320 (272 nonovh) | ~68 | 0.266 | 1.26x FAIL |
| xtq_dec c7 xe_hyb10 | 299 (229 nonovh) | ~57 | 0.224 | 1.49x |

K3 kill rule (>=1.4x vs exl3 decode issue): c4, c5 pass; c3, c7 marginal pass;
c6 (bs2) fails. All kernels 128 GRF, no spills.

M=4 DPAS loop (per-iter total /4 tiles, incl everything):
| code | instrs/512B | dpas+load overhead |
|---|---|---|
| exl3 mul1 | 88.75 | (3 sends + 4 dpas + loop) |
| xtq_dpas c3 | ~80 | same structure |
| xtq_dpas c4 | ~68 | same |
| xtq_dpas c5 | ~66 | same |
| xtq_dpas c6 | ~84 | same |
| xtq_dpas c7 | ~76 | same |

M=1 GEMV loop totals (4 tiles): c4 311, c5 343, c3 335, c6 343, c7 335 instrs.

## K2 — CPU Viterbi MSE (iid Gaussian, L=16, tail-biting, 256-w seqs, n=2048)

D(2)=0.0625, D(4)=0.0039.  Kill rule: K4 MSE > 1.05x exl3 (0.004832) -> drop.

| code | K=2 MSE | K=2 MSE/D | K=4 MSE | K=4 MSE/D | K2 verdict |
|---|---|---|---|---|---|
| exl3 (mul1) | 0.068276 | 1.0924 | 0.004602 | 1.1780 | ref |
| qtip_1mad | 0.068881 | 1.1021 | 0.004729 | 1.2107 | PASS 1.028x |
| exl3_mcg | 0.074069 | 1.1851 | 0.004726 | 1.2098 | PASS 1.027x |
| exl3_3inst | 0.074508 | 1.1921 | 0.004791 | 1.2265 | PASS 1.041x |
| qtip_hyb | 0.070831 | 1.1333 | 0.006043 | 1.5470 | FAIL 1.31x |
| xe_u2 | 0.081532 | 1.3045 | 0.017418 | 4.4591 | FAIL 3.78x |
| xe_ih2x2 | 0.075495 | 1.2079 | 0.005061 | 1.2957 | FAIL 1.10x |
| xe_bs2 | 0.069380 | 1.1101 | 0.005324 | 1.3629 | FAIL 1.16x |
| xe_hyb10 | 0.070026 | 1.1204 | 0.005769 | 1.4770 | FAIL 1.25x |

K=4 kill line = 1.05 * 0.004602 = 0.004832.  K4 GB/s: not run (K4 gated).

## Verdict: intersection empty under current kill rules

- Every V=2 candidate passes the K3 decode budget (except bs2) and every one
  fails the K2 distortion gate.  Every V=1 code passes K2 and cannot reach the
  decode budget (overlapping 4-bit windows -> ~5-6 ops/state extraction).
- The V=2 structural loss is the PAIR COMMITMENT: one 8-bit symbol fixes two
  weights; per-step choice set = 256 states->pairs.  Nearest-centroid 2-D VQ
  floor at Q=10 is 0.62x exl3, so the loss is commitment, not table size.
- Big-Q HYB tables (same gather cost, trained by trellis-Lloyd):
  Q=11 0.00507 (1.10x), Q=12 0.00497 (1.08x), Q=13 0.00489 (1.063x) at K=4.
  Extrapolated ceiling ~0.0048-0.0049 — the 1.05x line is just out of reach
  even with a 32KB SLM-resident pair table.  Tables saved:
  isa/../hyb_tlut_q{11,12,13}.npy (nvme lane).
- Nearest-to-passing V=2: xe_ih2x2 at 1.099x exl3 and 0.195 i/w decode.
  At a 1.15x K2 bar it (or a slightly-better-trained variant) would clear K3.
K=4 GB/s: not run (K4 gated on Main approval).

## Toolchain notes (compile-only)

- ESIMD requires the spir64_gen path: standalone llvm-spirv rejects ESIMD vector
  types unless given +SPV_INTEL_vector_compute and
  -spirv-allow-unknown-intrinsics=llvm.genx. (exact icpx flags replicated).
- icpx device LINK fails on exl3::GemvKernel/ReconstructKernel
  ("more than one module with an entry point") on both docker 2026.1.0 and host
  2026.1.1 — worked around by driving icpx -c -> clang-offload-bundler ->
  sycl-post-link -> llvm-spirv -> ocloc per kernel. exl3::DpasKernel compiles;
  other baselines taken from prior disasm evidence.
- llvm-spirv CANNOT be used on the icpx -c .o directly either — must go through
  sycl-post-link -split-esimd -lower-esimd first.

## Reproduce

  # K2 sweep (memory-bounded, desktop-b70)
  systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G -p MemorySwapMax=0 \
    --unit=xtq-k2-sweep -- bash -c \
    "cd ~/B70-DOCS/scripts/tmp && python3 xtq_viterbi_ref_20261011.py \
     --codes exl3,qtip_1mad,exl3_mcg,exl3_3inst,qtip_hyb,xe_u2,xe_ih2x2,xe_bs2,xe_hyb10 \
     --ks 4,2 --nseq 2048 --batch 256"
  # log: /mnt/<nvme-disk>/b70-run4-20261011/xtq-code/k2_sweep.log

  # EXL3 bit-exactness (full-tensor torch.equal, real EXL3 artifact)
  python3 scripts/tmp/xtq_verify_exl3_20261011.py

  # K3 AOT ISA (docker, no GPU)
  cd scripts/tmp/xtq_kernel_20261011 && bash build_isa.sh
  # artifacts: /mnt/<nvme-disk>/b70-run4-20261011/xtq-code/isa/
  # counts:   python3 /tmp/xtq_isa_count.py

## K5 integration notes (for HarnessFmt)

- Quantizer API: quantize_linear(W, H_x, code=<CODES key>, K=4, L=16, V=0,
  use_ldlq=True, device='cpu') in scripts/tmp/xtq_quant_20261011.py ->
  {W_hat, trellis int16 [k/16,n/16,16K] EXL3 ref layout (pack self-checked),
  suh, svh, Wscale, lut_rms, code, K, L, V, use_ldlq}. V=0 uses the code's
  native V from CODES meta. get_lut(code,16) -> [65536,V] fp32 codebook.
- Baseline alias for plumbing: code='exl3' (== exl3_mul1, bit-exact). Keep the
  code parametric; a final pick is pending Main's K2-relax/close decision.
  If relaxed to ~1.10x the provisional pick is xe_ih2x2 (no table, ~50
  instr/512B, dpas_direct).
- Sidecar npy tables (qtip_hyb q9, xe_hyb10 q10) must sit next to
  xtq_codes_20261011.py; bigger tables (q11-q13) live in the lane dir and are
  not registered in CODES (rescue experiment, all still >1.05x).

## K4 — GPU1 speed bench (operator amendment 2026-10-12)

Operator decision: continue with xe_ih2x2 despite the 1.099x K2 result
(recorded in RQ-29; end-to-end gates unchanged).

### Method (declared before the run)

- Binary: scripts/tmp/xtq_kernel_20261011/xtq_bench.cpp, JIT build
  (icpx -fsycl-targets=spir64, same path as the deployed exl3xpu library;
  build_jit.sh, strata-sycl-dev image). Source sha in the cell json/log.
- Arms, all in one binary, same launch shape: ih2x2 (xtq code 5),
  exl3 (upstream exl3::GemvKernel<4,2,1,4> / DpasKernel<4,2,4,4>, mul1 CB2),
  int4 (plain W4A16 nibble decode, xtq code 8 — value = nib*0.125-1).
- M=1 -> vector GEMV (packed fp16x2 pair-dot); M=4 -> DPAS (decode once into
  the B operand, fp32 acc). K=4 throughout.
- Shapes (Qwen3.8-27B linears): 5120x17408, 17408x5120, 5120x12288.
- Ring: weight stream replicated R = ceil(3*L2/tr_bytes) in one device buffer;
  iteration i reads copy (i % R). L2 read live via global_mem_cache_size.
- Schedule per cell: 1 warmup iteration per arm, then 3 interleaved rounds
  ih2x2 -> exl3 -> int4, one timed sycl event per iteration.
- Logical GB/s = K*N*0.5 B (4.0 bpw) / kernel event time.
- Random states (speed independent of values); every launch index and buffer
  bound computed and asserted on the CPU before submission.
- Safety: ONEAPI_DEVICE_SELECTOR=level_zero:1 + positive PCI match
  0000:0f:00.0; flock /tmp/b70-gpu1.lock per cell; gpu_guard before/after;
  lane aborts on any new xe engine reset mentioning 0000:0f:00.0.

### Declared output tolerances (fixed before the run)

  rel-L2 = ||y_gpu - y_ref|| / ||y_ref|| vs a numpy/torch reference
  (decode_stream / fast numpy decode + scatter + x@W):
  - ih2x2/int4 M=1 (fp16x2 pair acc over <=32-row splits):         <= 1e-2
  - exl3 M=1 (fp32 acc, fp16 row fold):                            <= 1e-2
  - all M=4 DPAS arms (fp32 accumulator):                          <= 1e-2

### Gate

  >= 550 GB/s logical for xe_ih2x2 at 4.0 bpw (M=1 and M=4).

### Results (GPU1 0000:0f:00.0, 2026-10-12)

Steady-state (median of last 3 timed rounds; warmup=60 interleaved rounds —
the card needs ~100+ back-to-back launches to reach sustained boost clocks;
per-launch host waits must be removed or measurements sit at idle clocks):

| cell           | M | ih2x2 GB/s | exl3 GB/s | int4  | pure-load | ih2x2/exl3 |
|----------------|---|-----------:|----------:|------:|----------:|-----------:|
| 5120x17408     | 1 |    342     |    495    |  357  |    415    |   0.69     |
| 5120x17408     | 4 |    407     |    503    |  412  |    440    |   0.81     |
| 17408x5120     | 1 |    348     |    476    |  360  |    420    |   0.73     |
| 17408x5120     | 4 |    418     |    496    |  419  |    489    |   0.84     |
| 5120x12288     | 1 |    365     |    465    |  365  |    414    |   0.78     |
| 5120x12288     | 4 |    420     |    502    |  427  |    451    |   0.84     |

Correctness (vs the CPU/torch reference, declared tol 1e-2):
ih2x2 rel-L2 = 1.7e-3 (M=1, fp16 pair-acc over <=32-row splits),
2e-6 (M=4 DPAS fp32 acc); int4 = 1.6e-3 / ~0; all PASS.

**GATE: FAIL.**  ih2x2 max = 420 GB/s < 550.  The exl3 baseline itself tops
out at ~500 GB/s (~82% of the B70's 608 GB/s spec); the ~550 bar is above
what this GEMV class sustains on the card at all.  ih2x2 lands at
0.69-0.84x of the production baseline — the 1.7x decode-instruction
saving does not translate to bandwidth because both kernels are
DRAM-latency/bandwidth-bound, not issue-bound.

### Deviations / engineering notes

- **pi layout (new encode-time weight permutation):** stream step t=4g+u ->
  VNNI dword pi=32u+g.  Makes every u-step's 32-lane pair group write
  CONTIGUOUS B-operand dwords and contiguous column accs — the earlier
  stride-4 selects compiled to a0-indirect register addressing (slow).
  Needed for both M=1 (col-fold) and M=4 (Bv write); the quantizer applies
  the same permutation at pack time.
- M=1 ih2x2: fp16x2 pair-domain accumulator (1 packed fma per u-step, 32
  regs/tile), folded to fp32 columns once per split; P chosen so
  rows_per_split <= 32 (rel err ~1.7e-3).  NT=4 tiles/thread (512B/iter).
- M=4 ih2x2: NT=4 upfront tile loads + contiguous Bv build + dpas<8,RC>
  (acc fp32).  Same shape as exl3's production dispatch (MB=8, NT=4).
- Launch config matched production: WG=8, P sized to ~1024-1408 threads,
  rps capped, no 256-GRF props on final config (they cost half the
  residency for no win).
- Async-queue timing: per-launch e.wait() serialization left the card at
  idle clocks (~90-150 GB/s everywhere); launches are now all queued then
  waited once.  Reported cells used warmup=60 + rounds=6 (more than the
  spec's 1+3 because DVFS ramp needs it; timing median-of-3).
- GPU guard clean before/after every cell; NO new xe engine reset on
  0000:0f:00.0 during the whole sweep (the 07:27Z bcs reset predates
  this run, peer lane, under RCA).  All CPU-side bounds checked pre-launch.

### Verdict

K4 measured verdict on the operator amendment: xe_ih2x2 = 342-420 GB/s
logical vs exl3 = 465-506 -> 0.69-0.84x, gate >=550 fails for both.
The V=2/K=4 decode advantage is real (1.7x fewer decode issues, direct
VNNI-pair emission) but the workload is DRAM-bound: decode ops are not
the limiter.  Combined with the unpaid 1.099x K2 quality cost, the
XTQ/K4 arm does not beat the EXL3 baseline on this hardware.

### K4 rerun on GPU0 (0000:0b:00.0, renderD128, 2026-10-12 ~12:50 local)

Operator decision 2026-10-12: GPU1 busy with the SC screen, K4 rerun on
GPU0 (desktop card).  Same binary rebuilt with --pci positive match
(aborts before any launch if the PCI is absent), flock /tmp/b70-gpu0.lock
per cell, gpu_guard (both nodes) before/after, 8 GiB device-alloc cap in
the binary, and an ACTIVE-LOAD check on renderD128: non-desktop holders
are allowed only while their drm-cycles counters are flat (the peer
xtq_esimd_test held the fd open but issued 0 engine cycles — CPU Viterbi;
it exited before the sweep anyway).  Any new xe reset on 0000:0b:00.0
aborts the lane — none occurred.

Steady-state (mean of timed rounds 1-5; round 0 is the DVFS ramp-in):

| cell           | M | ih2x2 GB/s | exl3 GB/s | int4  | pure-load | ih2x2/exl3 |
|----------------|---|-----------:|----------:|------:|----------:|-----------:|
| 5120x17408     | 1 |    315     |    481    |  328  |    393    |   0.66     |
| 5120x17408     | 4 |    392     |    452    |  401  |    410    |   0.87     |
| 17408x5120     | 1 |    315     |    474    |  329  |    392    |   0.66     |
| 17408x5120     | 4 |    404     |    448    |  406  |    460    |   0.90     |
| 5120x12288     | 1 |    379     |    466    |  390  |    420    |   0.81     |
| 5120x12288     | 4 |    415     |    465    |  421  |    447    |   0.89     |

Correctness on GPU0 dumps (declared tol 1e-2), checked with the
STREAMING checker (xtq_check_k4.py: 256-row K-blocks, no full W in
memory, prints getrusage peak): all 3 arms x all 6 cells PASS —
ih2x2 rel-L2 = 1.7e-3 (M=1) / 0.0 (M=4); int4 = 1.7e-3 / 0.0;
exl3 = 3.3e-4 / 0.0.  Peak RSS 0.64-1.51 GiB per cell under
system-slice scopes (sudo systemd-run --scope --uid=sergio,
MemoryMax=12G) — the earlier --user scope had thrashed user@1000.
Every cell json records card=0000:0b:00.0, "Intel(R) Graphics [0xe223]".

**GATE: FAIL on GPU0 as well.**  ih2x2 max = 415 GB/s.  GPU0 runs a few
percent below GPU1 for the same kernels (exl3 ~448-481 vs ~465-506 —
desktop composition traffic plus different clock residency), but the
ordering and the verdict are identical: DRAM-bound, ~550 GB/s is above
the sustained ceiling of this GEMV class on either card.

Artifacts: /mnt/<nvme-disk>/b70-run4-20261011/xtq-code/k4-gpu0/
(cell jsons, guards/, check logs, dmesg snapshots); driver
scripts/tmp/xtq_kernel_20261011/run_k4_gpu0.sh; bench
xtq_bench.cpp now takes --pci (default 0000:0f:00.0).
