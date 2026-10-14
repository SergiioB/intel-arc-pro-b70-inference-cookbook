---
title: "B6 L0-tracer route decision — isolated-op micro-timing first"
type: research
date: 2026-09-29
status: final
decision_by: b70-architect-2 (inherited from quota-dead b70-architect dispatch)
publication_gate: PROVISIONAL_NOT_FOR_PUBLIC_HEADLINE
---

# B6 tracer route — decision (2026-09-29)

**Ordering: (d) → (a, conditional) → (c, floor only) → (b) REJECTED.** B6 = the in-kernel efficiency bucket on the EXL3 route (probe 1/RL-090: ccs0 ~88 % busy, bandwidth 57–63 % of pin floor; the fleet's biggest live lever, ~63→85–95 tok/s at zero quality risk).

1. **(d) Targeted micro-timing — first, and it may finish B6 by itself.** The blocker is engine-specific, not tracer-specific: pti-gpu dies inside oneCCL `coll_init` during *EngineCore init*, and the fork pins `dist_backend="xccl"` (`platforms/xpu.py:115`). An isolated-op harness on the real on-host tensors (`/mnt/<nvme-disk>/b70-exl3-qwen38-4bpw-20260926`) never initializes oneCCL, never starts the engine, needs no image build — and the existing host interposer build becomes usable there for kernel-name splits. Per-family `exl3_linear` timing at M=1/M=4 vs the bytes floor decomposes exactly the bucket probe 1 could not (trellis/GEMV vs GEMM, by N-family).
2. **(a) Newer pti-gpu in a derived image — second, conditional.** No release-note evidence any pti-gpu version fixes a oneCCL interposer crash (nearest: intel/pti-gpu#29, Windows oneprof unload). An unvalidated bet costing an image build — gate behind the falsifier below.
3. **(c) Interposer-free PMU/zeMetrics — exhausted at its ceiling** (engine-granularity on this host; probe 1 already spent it). Keep only as a busy% bracket around (d).
4. **(b) Base refresh ≥v0.29 — REJECTED for B6.** Multi-variable by construction (moves the whole correctness/cache/fix landscape), voids the D3 capture contract (evidence pinned to digest `21412bdd…`, non-transferable), and strands the K=3 series off its anchor into a rebase. A measurement problem is not solved by a platform migration.

**Cheapest falsifying probe (low GPU, no engine, queued behind the K=3 GPU window):** isolated-op script on GPU0 — load the two extreme real families ([5120→1024], [5120→17408]), time `exl3_linear` at M=1 and M=4 with XPU events, divide by weight bytes against the 608 GB/s pin. Uniform ~60 %-of-floor across families ⇒ in-kernel inefficiency confirmed + B6 headroom priced; any op class at floor ⇒ redirect B6. Gate for (a): a tiny SYCL+oneCCL `coll_init` repro under the candidate interposer LD_PRELOAD (30 s, no vLLM, no image) — if it still dies, (a) is dead until oneCCL-side evidence appears.

**K=3 interaction:** (d)'s harness is exactly the P2/P4 isolated-op methodology the K=3 series reuses post-build on the same pinned base; the tracer route must not move the base.

Sources: intel/pti-gpu (repo, issue #29, sdk README).
