# Dual-B70 TP2 decode feasibility — analysis note (shelved, not adopted)

Date: 2026-09-29. Status: ANALYSIS ONLY — no GPU work performed, no adoption
proposal. Operator direction: TP1 speedups are the active track; this note
preserves the TP2 pricing for a future window.

## Premise (MEASURED facts this is built on)

- Decode on the adopted route (Qwen3.8-27B, exl3xpu, 4bpw, GPU0) is
  weight-bandwidth-bound: verify pass = 34.46 ms for 13.12 GB = 381 GB/s
  (62.6% of the 608 GB/s pin; RL-098). Round = 44.1 ms at 63.25 tok/s.
- GPU1 is idle during single-stream decode (independent-stacks default,
  ADR-0008).
- Copy-only P2P verified bidirectional ~11–12 GB/s, checksum-clean, ACCESS
  flag set, ATOMICS absent (RL-059, 2026-09-25). Any TP2 barrier must be a
  flag/sequence-number copy protocol — no remote atomics.
- TP2 is a gated research path, not banned (ADR-0008: collective-correctness
  gate + matched-control campaign before any adoption).

## Model (INFERRED — every number here is a bound, not a measurement)

Weight bytes halve per card under TP2 → verify pass bound ≈ 17.2 ms + barrier
cost. Activations per sync are tiny (hidden 5120 × 2–4 B ≈ 10–20 KB) so the
link is latency-bound, not bandwidth-bound. Estimated barrier structure:
~2 all-reduce/all-gather per transformer block × 64 layers; if a copy-based
barrier costs ~20–50 µs, sync adds ~2.6–6.4 ms vs ~17 ms saved.

- Estimated round ≈ 30–33 ms → ~85–110 tok/s (×1.4–1.75 on incumbent).
- Stacked with a fixed M=4 kernel: ~110–140 tok/s inferred ceiling.
- Stacked with TP2 + sub-4bpw: ~150+ tok/s class — speculative.

## Blockers / open questions (the deciding probes, cheapest first)

1. **exl3xpu TP support** — unknown; likely absent (single-card fork). If
   absent, TP2 means a custom TP runner or another engine. Upstream dig +
   source inspection. (b70-upstream-researcher)
2. **Barrier latency microbench** — small-payload copy-P2P flag/seq-number
   round-trip cost at 32-wg scale; ~minutes of GPU, decides whether the
   +2.6–6.4 ms term is 2 or 6. (b70-profiler / small SYCL probe)
3. **llama.cpp TP2 fallback** — captured recipe exists
   (`research/2026/humble-b70-llm-dual-tp2-20260824.md`) but on a ~26–31 tok/s
   base route; even ×1.8 lands below the 63.25 incumbent. Only viable on a
   fast base. NOT comparable to the incumbent (different checkpoint/engine,
   see serving.md Run 48 note).
4. **Collective-correctness gate** (ADR-0008) and graph-capture compatibility
   of cross-card waits — unverified; capture may not span remote waits.
5. Non-EXL3 9.6 ms term is card-independent — TP2 does not shrink it; its
   attribution (profiler) changes TP2's relative gain.

## Decision

Shelved per operator direction 2026-09-29. Reopen condition: a TP1 campaign
saturates in-engine levers (M=4 rewrite + had_out + sub-4bpw retrial) and
single-stream decode remains bandwidth-bound, OR a serving product wants
one fast stream over two independent stacks. On reopen: run probe (2) first —
it prices the whole idea for ~minutes of GPU.
