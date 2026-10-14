---
type: plan
date: 2026-09-20
phase: P5
title: "Small-M packed INT4 specialization (M=1 / small-M verify / expert-M)"
status: WITHDRAWN
opened: 2026-09-20
updated: 2026-09-20
ledger: RL-TBD-P5
depends_on: [P0, "P2/P4 stable winners as controls"]
effort: 4–8 sessions
---

# P5 — Small-M packed INT4 specialization

## Status log

| Date | Status change | Note / evidence |
|---|---|---|
| 2026-09-20 | filed as PLANNED | Roadmap filed from b70-architect draft |
| 2026-09-21 | PLANNED → RESEARCH, entry evidence delivered by P2's withdrawal | P5's entry attribution arrived from the Nemotron trace rather than from a new probe: on a graph-captured vLLM XPU run the device time is **73 % MoE+GEMM** (MoE cutlass 41.7 %, `gemm_kernel` 31.3 %) while everything else is fragmented small work — the same shape the K216 operator A/B found from the other direction (a fused decode op running 5.52× off its weight-read floor). **Entry probe for this phase is therefore: does the graph-captured stack (XPU graphs on, compiled) show M-regime waste — i.e. a measurable gap between small-M GEMM time and the same work at a batched M — and is it addressable without changing placement?** Kill criterion per the architect: if the entry probe shows no M-regime waste on the graph-captured stack, **withdraw P5** (its strongest measured object is already sub-bar on K216: the decode-GEMV ceiling there is 1.119× e2e). |
| 2026-09-21 | entry evidence measured — **M=1 object refuted by arithmetic** | M-regime sweep on this card (`q4_0` MUL_MAT, m=4096, k=14336, same harness; `results/2026/09/b70-improvement-probes-20260920/perf-mulmat-mregime.txt`): n=1 **2.14 TFLOPS**, n=2 4.49, n=3 6.33, n=4 7.37, n=5 8.11, n=8 8.87, n=512 **46.76 TFLOPS**. The apparent 5–22× small-M gap is **not waste at M=1**: the op reads ≈33 MB of weights once, so at 608 GB/s its floor is ≈54 µs and the measured time is **54.78 µs — at the floor**. Arithmetic intensity, not kernel quality, explains M=1. At n=8 the same arithmetic gives ≈17.3 TFLOPS at the floor against 8.87 measured ⇒ **≈2× of floor gap in the small-M verify regime**; at n=512 the op is compute-bound at 46.76 TFLOPS. Applying the phase's own kill criterion ("withdraw if the entry probe shows no M-regime waste"): **withdraw the decode-M=1 object** (no waste exists to specialize), and keep only a **narrow M≈4–16 object** whose measured headroom is ≈2× of the floor. That is a much smaller prize than the phase's 5–15 % framing assumed, and it must be re-scoped before any kernel work starts. |
| 2026-09-21 | RESEARCH → **WITHDRAWN** (decision, not a failed gate) | Decision recorded so no further spend goes here: the phase's subject was recovering the small-M efficiency gap, and the M-regime sweep shows the M=1 end of it is **arithmetic, not waste** — 54.78 µs measured against a ~54 µs weight-read floor — while the n=8 end is ~2× of floor. That is not enough to justify a specialization program (the phase's own precedent framing was 5–15 %, and the K216 decode-GEMV ceiling was already measured sub-bar at 1.119× e2e). **Residual open question folded into P3:** whether the M≈4–16 regime has addressable waste is decided **after** P3's grouped kernel changes the M-regime behaviour — evaluating it against today's kernel would measure the thing P3 is about to replace. Withdrawal does not imply the measurements were wrong; it means the object is too small and already being moved by another phase. Evidence: `results/2026/09/b70-improvement-probes-20260920/perf-mulmat-mregime.txt`. |

## Why this decision (the reasoning)

**The thesis: the FP8 campaign already proved that shape-specialized small-M kernels pay
on this silicon; the INT4 champion route has never had the same treatment, and the only
reason it is conditional is that nobody has attributed where its milliseconds go.**

1. **The precedent is strong and cheap to believe.** The FP8 ladder went 32.44 → 56.85
   tok/s largely through exactly this class of work: W8A16 reroute (remove act-quant),
   scratchpad pooling, and a custom Xe2 small-M block-FP8 GEMM reused from the
   grouped-GEMM source (M≤40 dispatch). That kernel family — `csrc/xpu/grouped_gemm/`
   Xe2 persistent scheduler with DPAS-16 tiles — exists in the pinned lineage. The
   question is simply whether the INT4 path's M=1 decode, M≤9 k=8 verify rows, and
   large-M prefill are hitting suboptimal tiles/packing through a shared dispatch.
2. **Why conditional 5–15% and not a headline.** Two disciplines apply: (a) no
   attribution exists yet — that is P0's dispatch-separation and register/SLM probes;
   (b) the rate-class correction forbids claiming integer-rate throughput for W4A16:
   **W4A16 is BF16-class compute**. Its dominance over FP8-W8A16 comes from halved
   weight bytes plus implementation, not from 363-TOPS math. So the honest upside here
   is tile/packing/dispatch efficiency — historically worth double digits *when waste
   is shown*, and worth nothing when it isn't.
3. **Why after P2/P4.** The A/B controls for this phase are the stable winners of the
   earlier kernel phases; measuring tile improvements against a moving baseline
   destroys attribution. Also P4's verified-subgraph work and P0's captured operator
   shapes feed the tile-set choice directly ("sweep only a small, shape-derived tile
   set" — not a generic tile fishing expedition).
4. **Why per-expert-M skew matters here specifically.** On A3B/MoE routes, routed
   experts see wildly varying M per step. A tile set tuned on dense-27B shapes will be
   wrong for expert GEMMs. Empty experts must be free. This is the same per-expert-M
   rule as P3, applied at dispatch level.
5. **Why repacking is gated on access analysis.** A lossless weight repack (padding/
   interleaving for subgroups and DPAS tiles, expert boundaries) can fix uncoalesced
   access — but repacking that expands resident weights (INT8/FP16 materialization)
   destroys the capacity advantage that justifies INT4, violating the phase's own
   premise. Repack only what access analysis proves is wasted.

**Alternatives considered and rejected.**

- *Assume the gain and build.* Rejected: RL-021 lesson — no target without attribution.
- *W4A8 as part of this phase.* Rejected: activation quantization is a quality research
  question (prior W4A8 path discarded after divergence), owned by P8 with its own
  hypothesis requirements. This phase changes kernels, not representations.
- *Synthetic-kernel TOPS as the deliverable.* Rejected: cookbook records model-level
  outcomes only; kernel microbenchmarks are attribution evidence.

**What would change this decision.** P0 probes showing M-regimes already hit distinct,
efficient kernels (withdraw — nothing to do); P8 landing a W4A8 path (re-scope this
phase to the INT8-class kernels that W4A8 then enables); upstream `vllm-xpu-kernels`
landing shape-specialized INT4 dispatch (collapse to validation + benchmark).

## Objective

Exploit the distinction between single-token GEMV, small-M speculative verification,
and large-M GEMM in the packed-INT4 kernel path.

## Baseline and E0 target

Controls: current P2/P4 winners and exact captured operator shapes (P0). No standalone
INT4 specialization gain is established today. Conditional target: **5–15% end-to-end**
where attribution shows meaningful tile, packing, or dispatch waste. Kernel speedup
alone is insufficient.

## Engineering work

1. Resolve pinned `vllm-xpu-kernels` grouped-GEMM dispatch, including the
   `csrc/xpu/grouped_gemm/` Xe2 implementation.
2. Verify actual W4A16 arithmetic and packed-weight layout.
3. Sweep only a small, shape-derived tile set.
4. Split M=1, small-M verification, and large-M dispatch if justified.
5. Handle per-expert-M skew and empty experts.
6. Inspect GRF pressure and subgroup requirements (`[[intel::reqd_sub_group_size(16)]]`
   launch-bounds discipline, RL-001).
7. Fuse compatible scale/unpack/epilogue work.
8. Repack weights losslessly only if access analysis supports it.
9. Avoid resident INT8/FP16 expansion that sacrifices capacity.
10. Reuse P0 probe results; do not create a second blind-spot campaign.

## Validation and benchmark plan

- Operator parity over group sizes, tails, scales, and supported shapes; exact
  packed-weight reconstruction tests.
- Standard five-cell card for the affected model.
- C2/C4 p2048/g128 aggregate checks if the new dispatch changes batched execution.
- Matched n=3 screen and n=5 finals.

## Risks and stop criteria

- Stop after two numerical/dispatch failures.
- Stop if the best kernel gain does not survive end-to-end measurement.
- Reject register spilling or weight expansion that violates capacity gates.
- Do not extrapolate a tile win to all experts or contexts.
- Do not claim integer-rate throughput for W4A16.

## Deliverables

- `research/2026/xe2-int4-shape-specialization-<YYYYMMDD>.md`
- `results/2026/<MM>/xe2-int4-shape-specialization-<UTC>/`
- Ledger entry.

## Cookbook integration

Document automatic dispatch and supported shapes rather than exposing unstable tuning
flags as universal recipes. Add audited model-level records, not synthetic kernel TOPS
to inference rankings.
