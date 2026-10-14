---
title: "EXL3 decode-round attribution audit"
type: research
date: 2026-09-29
status: provisional
author: "b70-profiler substitute (audit)"
scope: "CPU-only measurement-class audit of the RL-090/098/102 evidence chain; no GPU, no server, no build, no power changes; no re-measurement."
publication_gate: PROVISIONAL_NOT_FOR_PUBLIC_HEADLINE
---

# EXL3 decode-round attribution audit (2026-09-29)

> PROVISIONAL — NOT FOR PUBLIC HEADLINE. Private audit note (substitute for
> b70-profiler). CPU-only: no GPU, no server, no build, no power changes.
> Every number below is re-derived from retained raw evidence or pinned
> source; nothing was re-measured. No shared authority was edited.
>
> Question: in the RL-090/098/102 evidence chain feeding
> `research/2026/b70-tp2-decode-feasibility-20260929.md` (target note),
> `research/2026/b70-exl3-tp-feasibility-bound-20260929.md` (bound review)
> and `research/2026/b70-exl3-tp-feasibility-upstream-20260929.md` (upstream
> audit), which terms of the "44.1 ms round" budget are measured wall,
> which are isolated/estimated, and do the "138 collectives" (bound) and
> "zero draft collectives" (upstream) claims conflict?

## 1. Measurement-class census — the core of this audit

| value | file:evidence | class | what it actually is |
|---|---|---|---|
| **47.2 ms** | `results/2026/09/b70-probe1-exl3-rate-20260929/cells-230W.jsonl` (p512/g128: `chunk_rate_median`=21.17 → 1/rate) | **measured wall** (rate-inverse statistic, n=5 cell, 230 W, main server) | round period of the representative serving cell |
| **46.0 ms** | `requests-230W.jsonl` p512/g128: `median_chunk_step_ms` 45.88–46.44 across the 5 measured + warmup requests | **measured wall** (median-of-intervals statistic, same cell) | same cell, different statistic — the interval distribution is right-skewed, so median-step < 1/rate |
| **44.10 ms** | `attr-window-tight.json`: `median_round_ms`, single request p~6064/g256 (prompt overran ±96, declared), gen window 3.6514 s, 82 spec drafts / 83 SSE chunks, on the `-prof` server instance, 230 W | **measured wall, diagnostic only (n=1 window)** | median inter-round interval; mean = 3651.4/82 = 44.53 ms. NOT the representative serving round — the same `-prof` server's own p512/g128 request showed 49.69 ms median step, and B16 depth-flatness rules out the deeper prompt as the speed cause; treat 44.10 as a favorable sample |
| **48.7 ms** | cells-230W.jsonl p8192/g128: 1/20.53 | measured wall | second n=5 cell |
| **34.46 ms** | `results/2026/09/b70-b6-microprobe-20260929/RESULTS.md` §5–6 + `microprobe_results_run2.json` | **isolated-op sum — NOT wall** | arithmetic sum of per-family XPUGraph kernel-replay medians for the M=4 verify pass (13.126 GB → 381 GB/s), GPU0 at **180 W as-found cap**, no engine, no attention/GDN/sampler/host, uncontended L2/DRAM |
| **29.44 ms** | same, M=1 pass | isolated-op sum | the M=1 (per draft-class shape) analog |
| **7.14 ms** | bound review §1.2 | **estimate** (isolated-rate assembly) | 3.499 GB draft traffic ÷ ~490 GB/s. Re-derived: lm_head M=1 measured 508 GB/s → 954 MB = 1.88 ms; MTP module ~0.213 GB at M=1 family mix (~430–470 GB/s) → ~0.5 ms; ≈2.4 ms × 3 forwards ≈ 7.1 ms. Internally consistent, but it is isolated M=1 rates (at 180 W) applied to a 230 W wall — never measured in-situ |
| **2.50 ms** | bound review §1.2 | **subtraction residual — not a measurement** | 44.10 − 34.46 − 7.14 across two measurement classes AND two power caps (see §3) |

Closure check: 34.46 + 7.14 + 2.50 = 44.10 — arithmetic closes by
construction; that is not validation.

## 2. Counter → token/yield reconciliation (raw, `cells-*.jsonl` `counter_deltas`)

| cell | drafts | draft_tokens | accepted | per-pos survivals (0/1/2) | emitted (d+a) | usage tokens | yield (d+a)/d |
|---|---:|---:|---:|---|---:|---:|---:|
| 150 W p512/g128 (n=5 +warm bracket) | 267 | 801 | 506 | .787/.652/.457 | 773 | 768 | 2.895 |
| 150 W p8192/g128 | 256 | 768 | 513 | .801/.676/.527 | 769 | 768 | 3.004 |
| 230 W p512/g128 | 260 | 780 | 508 | **.827/.696/.431** | **768** | **768** | **2.954** |
| 230 W p8192/g128 | 263 | 789 | 509 | .829/.684/.422 | 772 | 768 | 2.935 |
| 230 W p2048/g128 (B16, n=3+warm) | 173 | 519 | 346 | .815/.647/.538 | 519 | 512 | 3.000 |
| 230 W p16384/g128 (B16) | 163 | 489 | 349 | .834/.718/.589 | 512 | 512 | 3.141 |
| tight window (p~6064/g256, 1 req) | 82 | 246 | 175 | (not split) | 257 | 256 | 3.134 |

Verified facts:

- `draft_tokens / drafts = exactly 3.0` in every cell → MTP3 proposes all 3
  every round; no partial-draft fallback in this data.
- Emitted-vs-usage edge effect: ≤ +7 tokens (+1.4 %, p2048 B16 cell; the four
  main cells are within +5 / +0.7 %) — tokens accepted in the truncated final
  round count in counters but not in `completion_tokens`.
  `usage.completion_tokens = 128`, `finish_reason=length` on every measured
  request (Gate A).
- Per-position survivals are marginal and non-uniform (0.83/0.70/0.43 at 230 W
  p512) — the §20 α-bookkeeping profile, not a single α.
- Cross-check: chunk-rate × yield ≈ token-rate within ~1 % (21.17 × 2.954 =
  62.5 vs measured 63.25 — ratio-of-medians vs median-of-ratios + edge
  tokens). Counter-rate column in RESULTS.md is a window-aggregate
  (64.37/61.42); its exact bracket wasn't retained in a re-derivable form —
  the bound review's residual 0.5–2 % spread stands open but is inside Gate B
  (±5 %) everywhere.
- **Pairing rule (confirmed on raw):** 63.25 tok/s pairs with 47.2 ms and
  yield 2.954; 70.1 tok/s pairs with 44.10 ms and yield 3.134. The target
  note's premise ("44.1 ms at 63.25") and the yield product 2.79
  (= 63.25 × 0.0441) are cross-window mixes — bound review verdicts 1/5 are
  correct on the raw numbers.

## 3. Why the residual is unidentified (and not fusion headroom)

`rest = 44.10 − 34.46 − 7.14 = 2.50 ms` subtracts an 180 W isolated-op sum
from a 230 W serving wall, minus an isolated-rate estimate. Two bias
directions coexist and neither is quantified:

- **180 W isolated vs 230 W wall:** the route is strongly power-limited
  (token-rate +28.7 % for 150→230 W, RL-090). The isolated kernels at the
  lower cap plausibly overstate their in-situ time ⇒ the true non-EXL3
  remainder at 230 W is likely **larger** than 2.5 ms. (Magnitude unknown —
  not extrapolated here.)
- **Uncontended isolated replays:** the harness streams one op back-to-back
  with a warm, exclusive L2/DRAM; in-situ the EXL3 pass interleaves with
  attention/KV/GDN traffic that evicts lines — in-situ verify could be
  **slower** than the isolated sum, shrinking the residual the other way.

The residual's contents are real work, not slack: 16 full-attention ops, 48
GDN conv/state/norm sequences, unquantized `in_proj_ba`, embedding lookups,
sampler over 248 K vocab ×4 verify positions + 3 draft argmaxes, host/
scheduler/graph-boundary time. RL-090's own PMU evidence says **ccs0 ~88 %
busy during decode with no idle bubble** — the residual is predominantly
on-device kernels, not launch gaps. **No in-situ per-op split exists:**
unitrace dies in this fork's oneCCL `coll_init` (no gloo escape flag),
`/start_profile` is 404, and decode is XPU-graph-captured so torch-level
profiling sees only graph launches (RL-090 §M6).

Therefore: **any persistent-fusion/megakernel headroom figure derived by
subtracting isolated sums from the wall is unfounded.** A fusible-overhead
claim requires the blocked L0-tracer split or engine instrumentation —
neither exists tonight. This applies symmetrically: the residual also cannot
be claimed as "all attention" for TP2-card-independence arguments.

Related boundary: RL-102's had_in+had_out reduction price (~1.42 ms measured
on three families, bounded 1.4–2.1 ms/pass) is a component **inside** the
34.46 ms isolated verify sum — it must never be added to the residual.

## 4. Draft-collective reconciliation — two distinct designs, not a conflict

Shared core (both designs): **128 row-parallel allreduces/round** on the
target trunk (o/out_proj + down_proj, 2 × 64 layers, 10 KiB hidden states).

| | **Design A — "shard everything"** (bound review §2.1 census) | **Design B — "replicated draft stack"** (upstream audit §3/§8) |
|---|---|---|
| extra collectives | +6 draft allreduces (o_proj+down per draft fwd) + 4 embedding allreduces (1 verify + 3 draft) + 1 logits allgather (~2.0 MB) + 3 argmax pair gathers (~10 B) | none — 128 total, zero draft collectives, zero vocab ops |
| total ops | **138 allreduces + 4 small ops** | **128 allreduces** |
| per-card draft+head traffic | halved: 3×(0.107+0.477) = 1.75 GB + verify head inside the 6.56 GB pass | duplicated: verify head 0.954 + 3×(0.213+0.954) = **4.45 GB** |
| engine work | draft-path TP sharding also needed (MTP qkv/o/gate_up/down + vocab-parallel head + gathers) | replicate MTP module (0.21 GB) + 6-bit head (0.95 GB) + embedding (2.54 GB); draft layer is `full_attention` with no GDN state (`qwen3_5_mtp.py:109`, pinned extract verified today) |
| correctness gate | fp-reassociation inside the draft forward (draft numerics differ slightly from TP1) | draft is bit-identical to TP1 by construction |
| determinism caveat | — | proposals must stay identical across ranks — fine under argmax drafting; would need seed-sync if the drafter ever samples |

Cost reconciliation at the only measured barrier (RL-071: 101.7 µs @4 KB,
104.5 @64 KB, 129.2 @512 KB — flat ⇒ ~98 % fixed cost; wire floor
1.74 µs/op at the verified ~11.5 GB/s copy path):

- A vs B per-card trade: B pays ≈ **+2.2 GB/card/round (~+4.4 ms at
  ~490 GB/s class rate)** to remove ~14 ops (~1.4 ms at 102 µs, ~0.5 ms at
  35 µs). On round latency A wins at every plausible barrier; B wins on
  engineering risk and a cleaner correctness story.
- Bound review's S3/S4 tables used design A (draft = 3.52 ms); under B the
  draft term is ~7.1 ms + ~1.0 ms extra verify-head read. S3 reprices to
  ~43–46 ms/round (~64–69 tok/s) — the TP2 verdict direction is unchanged
  either way.
- **Do not mix:** quoting B's "≈128 ops" while keeping A's halved draft
  bytes double-counts the draft saving; quoting A's "138" does not refute
  B's "zero draft collectives". They are different sharding decisions.

Verified in pinned source this session (also in upstream audit):
`vllm_plugin.py:157-162` hard-raises `NotImplementedError` on any sharded
partition (TP2 needs a custom/forked loader regardless of design — bound
blocker 2, resolved statically by upstream audit); `qwen3_5_mtp.py:109`
draft layer type = `full_attention`; `qwen3_5_mtp.py:230-238` MTP `lm_head`
is a `ParallelLMHead` bound to the shared head (no MTP-private head tensor
in the checkpoint — the 3× shared-head re-read = 2.86 GB/round is real).

**Incidental finding:** the pinned plugin already ships an opt-in pruned
draft head — `EXL3_DRAFT_VOCAB` + `_build_draft_head` +
`_patch_mtp_draft_logits` (`vllm_plugin.py:271-319`): draft logits computed
over a 128-block-aligned vocab subset of lm_head; target still verifies
full vocab. RL-090's launch did **not** set it (serve command in RESULTS.md
has no such env). This is (a) a TP1 byte lever on the largest non-main
traffic term (2.86 GB/round shared-head re-read), and (b) a way to shrink
design B's replicated-head cost. Acceptance impact needs a gate — it's a
draft-proposal-subset change.

## 5. Correct attribution statements (what the evidence supports)

- "The representative decode round at 230 W is ~47–48 ms of measured wall;
  a favorable single-request window showed 44.10 ms." ✓
- "The M=4 verify EXL3 pass is ≥34.46 ms-equivalent at 62.6 % of pin in
  isolated replay at 180 W" — ✓ as an isolated figure only; in-situ value
  unknown in either direction.
- "Draft bytes are 3.499 GB/round (2.86 GB = the 6-bit shared head read
  3×)" — ✓ measured census + source-verified.
- "Draft wall time is 7.14 ms" — ✗ estimate; unmeasured.
- "2.5 ms of non-EXL3 remainder" — ✗ subtraction residual across classes
  and caps; unbounded either direction.
- "Round traffic 16.63 GB vs artifact one-read 16.86 GB" — near-coincidence
  of two different derivations (bound §1.2 note stands).
- Effective rate on the correct numerator: 16.63 GB / 44.10 ms = 377 GB/s
  (62 % pin, 72 % of the measured-achievable 521 GB/s).

## 6. Minimal next probe (ordered, small)

1. **P-A (~10 min GPU, isolated, no engine — removes the cross-cap term):**
   re-run the RL-098 graph-replay harness at **230 W** on GPU0 over (a) the
   M=4 verify family pass and (b) the draft package at M=1 (lm_head K6 +
   MTP-module families + fc proxy). This replaces the 7.14 estimate with a
   measurement and makes the residual a same-cap subtraction. Reuse
   `harness_graphprobe.py` — no new code.
2. **P-B (~20 min, serving — first in-situ draft measurement):** matched
   **MTP1 vs MTP3** cells on one server (n=3, p512/g128, 230 W, spec-counter
   brackets). Δround/2 ≈ wall cost of one draft forward in-situ (verify M
   differs 2 vs 4 — near-equal bytes). If the fork refuses k=1, fall back to
   an MTP0 cell and correct with P-A's (M=4 − M=1) isolated delta. Either
   arm bounds the residual end-to-end without a tracer.
3. Explicitly **not** proposed: any decomposition of the residual's interior
   (blocked on L0-tracer/engine instrumentation, RL-090 §M6) and any TP2 GPU
   work (still behind: plugin TP guard → custom runner; flag-slot primitive
   greenfield at 60 µs kill-threshold; capture-of-remote-waits unproven).

## 7. Limits of this audit

- No re-measurement; all figures re-derived from `results/2026/09/
  b70-probe1-exl3-rate-20260929/` (cells/requests/attr-window JSON),
  `b70-b6-microprobe-20260929/` (RESULTS.md §3–6), `b70-b6-sweep-20260929/`
  (RESULTS.md had_out guard), RL-071 ladder, and the pinned extract at
  `/tmp/pinned-extract-20260929/` + clone `/tmp/b70-k3-exl3-source-20260929`.
- Counter brackets are cell-level (incl. warmup), not per-request; edge
  effects absorbed at ≤1.4 %.
- The `-prof` server context is deeper (p~6064); B16 flatness makes depth an
  unlikely cause of the 44.10-vs-47.2 spread, but the cause is not assigned.
- Draft-head sharing verified at source level; whether the draft verify path
  in-engine replays inside the same captured graph is unverified (decode is
  FULL_DECODE_ONLY captured).
