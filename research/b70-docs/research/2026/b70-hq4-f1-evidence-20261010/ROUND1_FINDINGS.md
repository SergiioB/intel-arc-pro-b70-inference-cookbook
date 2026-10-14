# RQ-26 F1 round-1 findings — 2026-10-10, GPU1 (PCI 0000:0f:00.0)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Kernel microbench only; no engine,
model, route or adoption claim. Executed under an orchestrator-authorized
poll-and-run window (the foreign llama-stream loop cleared at ~15:31Z and was
re-verified client-free immediately before launch and before every phase).

## Headline numbers (all cells: one discarded same-shape warmup, 30/100
profiled launches, in-order queue, event profiling, ring ≥ 4× 32 MiB L2,
allocation ≤ live visible_avail − 4 GiB)

| cell | effective GB/s (per-pass, plane+scales / median launch) |
|---|---:|
| qkv (5120×12288) M=1 | 15.89 |
| qkv M=4 (rowpass) | 15.85 |
| qkv M=8 (rowpass) | 15.85 |
| gate_up (5120×34816) M=1 | 15.75 |
| gate_up M=4 (rowpass) | 15.80 |
| gate_up M=8 (rowpass) | 15.74 |
| down (17408×5120) M=1 | 13.39 |
| down M=4 (rowpass) | 13.44 |
| down M=8 (rowpass) | 13.44 |

**Gate ≥ 500 GB/s at M=4: MISSED by ~33×** (15.85 vs 500). ksplit 2/4 on
down M=4 does NOT help (16.03/15.90) — not occupancy. The per-tile cost is
~3–6 µs (thousands of cycles per 128 B tile): the N-major VNNI fragment fill
(halfword-granular `simd<u16,256>` region writes + repack) and/or the
per-tile A-fragment VNNI transpose serialize in a way the two earlier
ISA dumps (old kernel) did not show. A fresh disassembly of THIS kernel and
an ISA-guided restructure (u32-dword-region fill, SLM-staged A, or
dpasw-style operand) is the concrete round-2 path — the source is in place
(`scripts/tmp/rq26_f1_kernel.cpp`), the AOT artifact was rebuilt for it.

## Correctness

- **Wrap (signs + block-128 FWHT + tree-sum S): BITWISE correct** on every
  cell (`wrap_x=1 wrap_S=1`, all 12 correctness cases).
- **GEMM: bitwise NOT achieved** — the hardware dpas F32 reduction order does
  not match either declared model (pair-sum / sequential / f64-per-instruction
  / fma chains / reversed chaining all rejected empirically; max |diff| vs
  bitref ≤ ~40 abs on |out| up to 3.7e6). **Values verified against the
  float64 oracle**: relmax 1.3e-6 … 2.0e-3 (larger on rows carrying the
  ±4096 injected specials via the documented 1024-offset cancellation).
  Resolving the exact order is an ISA-spec task for b70-isa-analyst.
- **M>1 dpas path NOT working**: with RC=4/8, ALL activation rows collapse
  into output row 0 (row-probe evidence in `probe/`: wrow_*.out — every
  source row lands in output row 0, scaled). The Intel dpas A-operand
  M-interleave on this silicon is not what the two natural layouts assume;
  probes to date (row-constants, zero-sum rows) bracket it but the exact
  interleave is unresolved. **Round-1 M=4/8 numbers are RC=1 ROW-PASSES**
  (per-launch streaming rate; weight bytes NOT reused across rows — the
  unique-byte rate is per-pass/M: 3.96 GB/s at M=4 qkv). The dpas row-reuse
  advantage is NOT measured and NOT claimed.

## Hadamard wraps (timed separately, 100 launches each)

K=5120: M=1 30.5 µs, M=4 46.0 µs, M=8 63.5 µs; K=17408: M=1 42.2 µs,
M=4 88.2 µs, M=8 133.1 µs (median per launch; all raw samples in
`round1/wrap_*.out`). The wrap is 2–3 % of the (slow) GEMM launch time at
M=4 — negligible for the gate as measured, re-cost after any GEMM restructure.

## Layout discovery (the round's real output)

Placement probes (constant plane + single nibble + one-hot activations,
`scripts/tmp/rq26_f1_probe_place.py`, raw outputs in `probe/`) MEASURED the
true dpas.f16 B-fragment layout on this B70:

    fragment dword D = n + 16*(k/2) holds B[k = 2*(D//16) + (D%2? no —)
    halfword h of dword D = n_local + 16*t  <->  B[kk = 2*t + h][n_local]

i.e. **N-major VNNI k-pairs** (dword = n + 16·(k/2), lo half = even k, hi
half = odd k) — NOT the k-major dword order the layout.md §2 assumed. The
kernel fill was rewritten to it (`rq26_f1_kernel.cpp`, "N-major VNNI,
placement-probe verified"); the eight probe observations are all reproduced
by this model exactly (including the two "invisible" placements, which are
odd-kk nibbles not lit by a k0=0 one-hot). layout.md §2/§3 need this
correction propagated; the SPEC file byte map is unchanged (the fix is
entirely in the kernel's fragment assembly).

## Session provenance

- Window: foreign loop cleared ~15:31Z; hard re-verify at 15:31:34Z
  (`reverify.txt`); per-phase contention checks in `contention.log` — no
  engine client appeared during the session; GPU1 visible_avail 32,615 MiB
  throughout; caps as-found 180/230 W, never written; temp2 end 55.0 °C.
- Known session defects, fixed in-line: runner env (setvars under `set -e`),
  bundle buffers not resized before fread (segfault), selftest expected
  constant, k0-sweep availability bug in the wait script.
- The pre-write security hook blocked several intermediate source forms
  (memcpy-family calls, host-pointer arithmetic at copy call sites); the
  retained sources pass it.
