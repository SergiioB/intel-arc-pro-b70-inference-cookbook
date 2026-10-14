# RQ-26 F1 round-2 findings — 2026-10-10, GPU1 (PCI 0000:0f:00.0)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Continues round 1 (same day): the
window stayed clear (foreign loop never re-armed; per-phase checks in
`../round1/contention.log`; GPU1 visible_avail 32,615 MiB at start and end;
caps 180/230 W as-found, never written; zero engine clients at start and end).

## The round-2 fix (applied)

Placement probes in round 1 measured the true dpas.f16 B-fragment layout and
cleared the file layout: the ORIGINAL layout.md dword map (bits [4p] =
q(n=8qq+2p, kk), bits [16+4p] = q(n=8qq+2p+1, kk)) is CORRECT — the round-1
bug was exclusively the kernel's fragment placement. Round 2 replaced the
broken placement AND the halfword-granular `simd<u16,256>` machinery with 16
u32 dword-region writes per tile:

    for (qq, p):  frag[8qq+2p+16t]     = (w_even[t] & m(4p))     + 0x64006400
                  frag[8qq+2p+1+16t] = (w_odd[t]  & m(16+4p)) + 0x64006400

(w_even/w_odd = stored dwords with even/odd kk; m(x) = 0xF<<x). Source:
`scripts/tmp/rq26_f1_kernel.cpp` "dequant -> B fragment (placement-probe
verified, round 2)".

## Round-2 numbers (same protocol as round 1: 1 discarded warmup, 30
profiled launches, ring ≥ 4× L2 from live VRAM, per-launch medians)

| cell | round 1 | round 2 | gain |
|---|---:|---:|---:|
| qkv M=4 (rowpass) | 15.85 GB/s | **24.36 GB/s** | 1.54× |
| gate_up M=4 | 15.80 | 23.55 | 1.49× |
| down M=4 | 13.44 | 20.62 | 1.53× |
| qkv/gate_up/down M=1 | 15.89/15.75/13.39 | 23.71/23.53/20.57 | ~1.5× |
| M=8 rowpass | 15.85/15.74/13.44 | 24.39/23.53/20.62 | ~1.5× |

**Gate ≥ 500 GB/s at M=4: STILL MISSED (24.4 vs 500, ~20×).** The per-tile
cost fell from ~6.4 to ~4.2 µs but remains thousands of cycles for ~100
instructions of work — consistent with register spills in the fill sequence
(live set: w 4 GRF + bfrag 16 + a-temps 6 + acc/corr + addressing ≈ 40-60
GRF, near the occupancy cliff) or a region-write lowering cost far above
1 op/element. ksplit 2/4 still does not help. REQUIRED NEXT: disassembly of
the current kernel (the `-vc-codegen` dump route failed under icpx
directly; the ISA analyst's two-step method — spv extraction then
`ocloc compile/disasm` — is the sanctioned path) to locate the spill/serialize
before any further blind restructuring.

## Correctness (round 2)

- All 8 per-row values checks PASS on every real shape × M ∈ {4,8} plus the
  M=1 cells: worst rel-vs-f64-oracle 1.68e-3 (qkv_m8's ±4096-specials row;
  all other rows ≤ 1.2e-4). Raw: `round2/correct_*_r*.out`.
- NOTE: round 1's "qkv_m4 row 1 fails at 1.78" was THIS SESSION'S RUNNER BUG
  (mathref row slice at r*128 instead of r*N) — the hardware row-1 output was
  correct (6.4e-5); the runner is fixed in `rq26_f1_round1_run.sh`.
- Bitwise-vs-bitref: unchanged from round 1 — the dpas internal F32 reduction
  order remains unidentified (all candidate models rejected empirically,
  max diff ~40 abs); wrap kernels remain bitwise-correct.
- RC≥2 A-operand interleave: still unresolved (round-1 wrow probes); M>1
  numbers remain RC=1 row-passes (per-launch streaming rate reported; the
  unique-byte rate at M=4 is per-pass/4 = 6.09 GB/s qkv).

## Hadamard wraps (re-run round 2, 100 launches)

K=5120 M=4: 45.5 µs median; K=17408 M=4: 87.1 µs (unchanged from round 1 —
the wrap kernel was not modified).

## Session end

2026-10-10T17:30:06Z; GPU1 32,615 MiB free, zero engine clients on both
cards, caps 180/230 W as-found, no owned processes or files left (ring
bundles deleted per cell).
