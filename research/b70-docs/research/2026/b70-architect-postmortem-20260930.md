---
title: "EXL3 Qwen3.8-27B single-card decode post-mortem — where the cycles go, what failed, what next"
type: research
date: 2026-09-30
status: provisional
author: "substitute b70 silicon architect (CPU/source analysis only — no GPU, no builds, no model loads)"
scope: "Post-mortem on the upstream 0xSero/exl3xpu stock route (image 77785c51423f == ghcr.io/0xSero/exl3xpu byte-identical; source pinned @15ded2f3 at /tmp/b70-k3-exl3-source-20260929). Synthesizes RL-090/094/096/098-102/106/111 + C2-V2 + upstream PROGRESS.md + kernel source."
publication_gate: PROVISIONAL_NOT_FOR_PUBLIC_HEADLINE
---

# EXL3 Qwen3.8-27B decode post-mortem (one B70)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Private analysis note. No shared
authorities edited, no commits, no GPU work performed.

## 0. The measured state (what the cycle budget actually is)

Production cell: **63.25 tok/s** p512/g128 n=5 @230 W (probe 1, RL-090,
`results/2026/09/b70-probe1-exl3-rate-20260929/`). Diagnostic window: 44.10 ms
round at yield 3.13; representative round **47.2 ms at yield 2.954** — never
mix the two (`b70-exl3-round-attribution-audit-20260929.md` §1-2).

Round anatomy (best current accounting; caveat classes flagged):

| term | value | class |
|---|---|---|
| M=4 verify EXL3 pass (trunk linears + full lm_head) | **34.46 ms / 13.126 GB / 381 GB/s** | isolated-op sum, XPUGraph replay, **180 W** cap (`b70-b6-microprobe-20260929/RESULTS.md` §5-6) — cross-cap vs the 230 W wall, in-situ value unbounded either direction |
| M=1 analog pass | 29.44 ms / 446 GB/s | same |
| Draft package (3× MTP fwd) | ~1.4 GB → ~2.9-3.1 ms at M=1 rates | **corrected estimate** — see below |
| Non-EXL3 remainder (attn/GDN/norm/sampler/host/graph gaps) | **~6.5-6.8 ms** | subtraction residual, contents unmeasured |
| total | 44.1 ms diagnostic / 47.2 ms representative | measured wall |

**Accounting correction of record (2026-09-30):** the pruned draft head
(`EXL3_DRAFT_VOCAB`, 512/1940 vocab blocks = 26.4 % of lm_head) **is engaged on
the production route** — confirmed in the champion's own startup log
(`results/2026/09/b70-probe1-exl3-rate-20260929/server-startup.log:58`) and set
by `model.yaml` (`EXL3_DRAFT_VOCAB: "{model_dir}/draft_vocab.json"`), which
overrides container env. The attribution audit's "2.86 GB/round shared-head
re-read" (`b70-exl3-round-attribution-audit-20260929.md` §4) and the TP2
decision's design-B draft-byte figures are therefore **overstated ~2×**: actual
draft lm_head ≈ 3×252 MB ≈ 0.76 GB; total draft ≈ 1.4 GB/round, ~2.9-3.1 ms.
The non-EXL3 residual is ~6.6 ms, not 2.5 ms — still a cross-class subtraction,
unbounded in detail, but the draft side is now closer to measured.

Floors used throughout: pin 608 GB/s; **measured-achievable read 521 GB/s**
(`sum()` reference, RL-098 §4); r+w copy 489 GB/s; dense GEMM 117.5 TFLOPS
bf16; decode yield 2.954-3.13 tok/round (counter-verified, audit §2).

## 1. Failure taxonomy — every lever tried, what it measured, how much to trust it

Verdicts cite the evidence. "Trust" = whether the negative/positive is
measurement-backed or inference-backed.

### 1.1 TRUSTWORTHY negatives (measured, paired, in-window)

| lever | evidence | measured | why it did not deliver | trust |
|---|---|---|---|---|
| **MB=8→4 tile padding** (P6 patch) | RL-101, `b70-b6-mb4-ab-20260929/RESULTS.md` §3,§6 | M=4: down_proj +0.41 pin pts, qkv +4.72 (gate ≥5); MB4 bit-identical to MB8 | Padding was never the mechanism. The M=4→M=1 regression (8-15 pts/family) survives untouched; round value ≈0.07 ms (~+0.16 %) | **High** — bit-exact, both orderings, in-image A/B; ±5 % cross-session class drift recorded |
| **NT strip geometry sweep** (EXL3_DPAS4_NT 2/4/8) | RL-102, `b70-b6-sweep-20260929/RESULTS.md` §4-6 | flat/regressive on 3/4 families; ONE confined win: in_proj_z NT2 +10.5/+9.5 % (bulk secondary +4.7/+4.1 %, hint band) | Thread count = n_strips×P is pinned ≈1408 by `g_target_threads_mb16` (`exl3_ops.sycl:178,292-296`) regardless of NT — the sweep moved per-thread strip width at constant parallelism; the plateau is geometry-insensitive | **High** as a negative on the NT axis; the in_proj_z break is family-confined and prices ≲0.5 ms/pass |
| **Split-K partials + had_out reduction** (bounded then attempted) | RL-102 guard (ii); RL-111 stage C | t_skip = had_in+had_out measured: down 14.0 / gate_up 7.8 / lm_head 12.7 µs/call → **1.4-2.1 ms/pass total**; C1-R4 staged/serial kernels: no order clears ≥15 % (12.16→12.34; reversed 11.66 vs 7.26), D3 host-wait gate FAILED both arms | The whole reduction stage is latency/launch-floored but *absolutely small*; the authored staged variant didn't beat it and broke the capture contract | **High** — the bound is a direct measurement (SKIP_GEMV build), not arithmetic |
| **Register binding / B-bank serialization** (C2 K-pair V2) | `b70-c2-repair-20260930T085200Z/v2/{RESULTS-V2.md,GATE-REPORT.md}`; earlier C2 falsified at ISA level (`b70-resume-breakthrough-20260930T063200Z/REPORT.md`:54, reused bank r17) | bit-exact parity; **3-4 % SLOWER** (down 111.85 vs 108.41; qkv 76.23 vs 73.00 µs) | Third strike on the "decode→dpas serial chain" theory: upstream EXL3_BV_ALL (per-tile B regs) = no change (PROGRESS.md:294-296); C2's independent-bank intent disproved at ISA; V2's explicit pinning regresses (removes IGC allocator freedom) | **High** — three independent constructions converge: the loss is NOT a B-register WAR/scheduling artifact |
| **Fused single-kernel linear** | upstream PROGRESS.md:77,251 | M=4 45.2 vs 35.1 ms; small layers worse | serialized split-K tail inside one kernel; hads cost only ~1.2 ms under graphs | High (upstream measured, consistent with our 1.4-2.1 ms bound) |
| **EXL3_LUT codebook gather** | PROGRESS.md:424-428 | M=1 107.65 vs 27.24 ms | per-lane gather ~4× below ALU decode rate; SLM table would exceed the weights' bytes | High |
| **EXL3_DPAS_PREFETCH (double-buffered words)** | PROGRESS.md:50,188 | M=4 70.0 ms (spills); M=16 86.4 | 128-GRF budget can't hold words×2 + acc(2 KB) + Bv + Am | High at 128 GRF — **but see §3.4: never tried at 256 GRF** |
| **256-GRF DPAS (no prefetch)** | PROGRESS.md:132 | NT4 44.4 / NT8 53.7 vs 35.1 ms | halves resident threads (~1024) without giving back anything — the added registers weren't used for prefetch | High for the standalone flag |
| **Forced-vector M=4** (op-class swap) | RL-102 guard (i) | 250-265 GB/s vs M=1 vector 404; DPAS@M4 beats it 25-30 % | The M=4 *shape* degrades both op classes — this is the decisive "not the op class" datum | High |
| **MUL16 decode split, K4_HALVES, DECODE_INLINE** | PROGRESS.md:213-214, 276-277, 275 | slower / wrong / no-change | instruction-level decode tweaks exhausted | High |

### 1.2 TRUSTWORTHY negative with an open reopener

**The K=3 / 3bpw route** (RL-100, `b70-k3-completion-20260929/RESULTS.md`):
every correctness/capture gate passed (boundary-flip class confirmed
bit-exact-deterministic, no race, D3a/D3b capture contract PASS, quality 6/6×2
incl. the code_execution discriminator) — and it lost **42.44 vs 63.25 =
0.671×** with acceptance unchanged, so the loss is unambiguously kernel round
speed, not speculation quality. Why: `Geo<3>` (`exl3_esimd.h:28-35`) has
g=gcd(3,32)=1 → V=32 values/period over G=8 lanes with TP=4 tile pairing —
**~4× the per-lane decode work per weight stream vs K=4** (V=8/G=32), plus the
PLANAR de-interleave pass per row (`planarize`, exl3_esimd.h:198-205).
Structural, not incidental. Reopeners: the P2 strip sweep (defaults
[8,4,4,4] never swept — `P2-SWEEP-SPEC.txt`) and whatever the §2 experiment
attributes; a K3 kernel at K4-class per-byte efficiency would carry −13 %
bytes → the route's promise is real but gated behind kernel work that the
K4 evidence says is hard.

### 1.3 PLAUSIBLE-BUT-THIN attributions (be honest)

- **"M=4 intensity class owns the ~6.5-9 ms residual"** (RL-102 verdict): this
  is a *residual label*, not a mechanism. The guards eliminated op class,
  geometry, padding, and reduction — the positive owner is unmeasured. The two
  live hypotheses are (a) front-end issue bound: upstream measured the M=1
  GEMV "co-limited by issue (~22 ms) and DRAM (~23.5 ms)" (PROGRESS.md:53) and
  the K=4 DPAS loop is ~300 SIMD32 instr/K-row (PROGRESS.md:271-273 +
  `build_k4`, `exl3_esimd.h:669-716`) — decode issue at M=4 is likely ≥half the
  kernel time; (b) memory-latency exposure: ~1440 launched threads × 512 B/row
  with a serial load→decode→dpas chain per row and *no* software pipelining
  (`load_words`, exl3_esimd.h:154-169) — but NT8 flat argues against pure
  outstanding-bytes limitation. **Unresolved; see §2.**
- **The 34.46 ms cross-cap term**: measured at 180 W, the wall at 230 W. If
  kernels are issue-bound they scale with clock (PL2 sits ~2600/2800 MHz under
  decode, PROGRESS.md:102,136); the in-situ M=4 pass could be a few % faster
  than the isolated sum in either direction (audit §3).
- **DRAM-side expectations**: the "37 % in-kernel bandwidth loss" is vs the
  608 GB/s *pin*; vs measured-achievable 521 it is ~27 %, and lm_head/gate_up/
  in_proj_qkvz already sit at 90-98 % of achievable at M=1 (RL-098 §5). Some
  of the "loss" was floor mythology; the real structured slack is the M=4
  path + mid-N families.
- **C2 V2's excluded 354 µs sample** was multi-driver contamination + cold
  clock (recorded); per-process harnesses need the 2 s preheat (GATE-REPORT).

## 2. The missing measurement

**What would actually explain the M=4 plateau:** a per-family **three-arm
ablation on the pinned image**, reusing `harness_graphprobe.py` verbatim:

- t_full — incumbent cell (have: RL-098 table);
- t_BFAKE — `EXL3_DEBUG_B_FAKE` build: B operand = raw words, **decode+VNNI
  build removed, loads+dpas+partials intact** (`exl3_esimd.h:812-818`);
- t_skip — `EXL3_DEBUG_SKIP_GEMV` build: had_in+had_out only (**already
  measured**: image `d4ca20a3e451` + guard JSONs).

Per family: `t_full − t_BFAKE` = decode+build issue cost; `t_BFAKE − t_skip` =
the pure load+dpas memory core; `t_skip` = scaffolding. One extra derived image
(~1 min build on the retained P6 tree), ~30 min GPU, production-faithful
(graph-replay) methodology. This single experiment decides whether the ~9 ms
residual is **instruction-issue** (→ §3.3 decode-restructure program, hard,
bounded by every decode-variant rejection) or **memory-system** (→ §3.4 the
never-tried 256-GRF+prefetch combination and thread/occupancy retuning).

Secondary discriminating capture (if reachable): pti-gpu unitrace **in the
isolated harness** — the RL-096 blocker was oneCCL `coll_init` during
EngineCore init; this path has no oneCCL at all, so the interposer may work
here for the first time (untested). Desired counters: per-kernel EU active/
stalled split, XMX-pipe occupancy, L2 miss rate / actual DRAM bytes vs the
13.126 GB paper figure, resident-thread occupancy. If pti still dies, the
ablation stands alone and an IGC `.text` dump of `DpasKernel<4,2,8,4>` +
register counts (the C2 pipeline already produced this machinery) gives a
static instruction/issue cross-check at zero GPU cost.

## 3. Ranked next programs (each with gate + cheapest falsifier)

### 3.0 Gate the rest: the §2 ablation (P0)
Cost ~1 h wall including build. Decides which of 3.3/3.4 is real.
Falsifier: `t_BFAKE` within ±10 % of `t_full` at M=4 → memory-structure owns
it; a ≥25 % drop → decode-issue owns it.

### 3.1 Measure the residual's interior (P1, cheap, possibly the largest easy win)
The ~6.6 ms non-EXL3 remainder has never been split in-situ (no tracer, graph
capture hides per-op). The audit's P-B probe — matched **MTP1 vs MTP3 serving
cells** (n=3, p512/g128, 230 W; Δround/2 ≈ one in-situ draft forward) — plus
torch-level timing of the *uncaptured* per-step ops (upstream's own profile:
target lm_head 1.78 + draft lm_head 3×0.49 + draft linears ~1.3 +
FA2/argmax/reduce ~0.8 ≈ 5.7 ms outside the graph, PROGRESS.md:263-266) prices
draft wall and residual simultaneously. Then attack the largest bucket:
GDN state/norm kernels, the 248320-col sampler path, graph-boundary gaps.
Gate: identify ≥1 ms of killable/fusible time. Falsifier: residual fully
explained by named irreducible ops → close it.

### 3.2 Cheap byte/config levers (~hours, each gated)
- **Draft head 512→256 blocks** via bind-mount over the in-image
  `draft_vocab.json` (env alone is overridden by recipe env — today's
  VOIDED experiment proved the mechanism, `MEASURE-DRAFTVOCAB.json`):
  −~0.38 GB/round ≈ +1-1.5 % decode; gate = matched cells + acceptance delta
  ≤ noise + byte-identical greedy outputs (the draft subset change can shift
  proposals → acceptance gate is load-bearing).
- **k=4 speculative depth, serving-flag A/B** on the pinned image: never
  measured on this build generation; upstream think-on C1 showed +8 %,
  think-off synthetic showed a loss. One matched n=5 cell answers it. Gate:
  ≥5 % with counters reconciled.
- **EXL3_FOLD scoped to MB≤8 decode** (codebook-affine fold, +3.5 % all-linears
  upstream, PROGRESS.md:365-372): opt-in because of an *unexplained* M=24/40
  gemm_raw anomaly — which never executes at C1 decode shapes. Gate: resolve or
  scope the anomaly + amended-tolerance + 6/6 battery. ~+1.2 ms/pass ≈ +2.7 %.

### 3.3 The M=4 kernel program (the real money — conditional on 3.0)
Ceiling arithmetic (RL-098 §6): all-families-to-85 %-pin saves 9.07 ms/verify
→ round 44.1→35.0 → **~79 tok/s** (95 %-of-achievable target ~7.95 ms → ~77).
If 3.0 says issue-bound: restructure decode so it is not on the load→dpas
critical path — e.g. decode-to-SLM staging inside the work-group (upstream's
4×-traffic objection was against a different split; reprice honestly), or a
per-tile decode pipeline decoupled from DPAS issue, or K4-specific decode at
lower instr/value (LUT/MUL16/HALVES all dead — the budget is ~9 SIMD32 per
32-value group; DPAS-side B-build is what differs from the vector path).
If 3.0 says memory-bound: see 3.4.
Gate: ≥5 % weighted M=4-pass improvement isolated, then serving gate.

### 3.4 The never-tried combination (cheap sibling of 3.3)
**256-GRF + `EXL3_DPAS_PREFETCH` together.** Each was rejected *alone*
(256-GRF wastes occupancy with no consumer; prefetch spills at 128 GRF).
The MB8 kernel at 256 GRF has ~8 KB/thread: acc 2 KB + double-buffered
words/prevs 2×1 KB + Bv 0.5 + Am 0.25 + indices ≈ 5-6 KB — fits. Fewer
resident threads (~1024) × deeper in-flight per thread is exactly the trade a
latency-bound kernel wants. Cost: two build-arg flips on the retained tree.
Falsifier: one paired cell (down_proj M=4) — flat or worse closes it forever.

### 3.5 3bpw route revival (conditional, staged)
(P2 strip sweep on the existing k3 image → if ≥5 % isolated, a fresh matched
serving A/B; P4 spec exists: needs ≥69.6 tok/s to pass.) Requires the same
kernel program as 3.3/3.4 to land first — the 3bpw upside (−13 % round bytes)
multiplies whatever the M=4 path becomes. Combined ceiling if both land:
~90-95 tok/s.

### 3.6 Head-side bytes
verify lm_head is K6, 954 MB, at 83.5 % of pin — byte-cut only: head_bits
6→4 saves ~0.32 GB verify + 3×0.13 GB draft ≈ ~1.1 ms ≈ +2.4 %, but it is a
requant of the logits margin source — real quality gate (the 30-defect class
says heads are where margin lives). Low priority; quality-risk-priced.

### 3.7 The blunt question — is there a single-card path to >2×?
**No. Arithmetic:**
- >126.5 tok/s at measured yield 2.954 ⇒ round ≤ 23.3 ms.
- Weight bytes alone (13.126 GB) at the *dream* pin floor = 21.6 ms; at
  measured-achievable 521 GB/s = **25.2 ms > 23.3 ms** — before attention,
  draft, sampler, host. Even at K3 bytes (~11.3 GB): 21.7 ms of linears.
- Add the irreducible-looking ~6-9 ms non-EXL3 + ~3 ms draft: the
  everything-perfect fantasy bound is ~85-105 tok/s. To beat 126.5 needs
  bytes cut AND kernel-perfection AND yield ≫2.95 simultaneously — three
  unproven things multiplied, which §5 of the TP decision forbids counting.
- **Dual-card:** custom TP2 model bound = 127 tok/s @60 µs barrier /
  100 @102 µs measured (decision §3). The only >95 tok/s path, gated behind
  the unbuilt flag-slot copy collective (60 µs kill threshold, capture-of-
  remote-waits unproven). Independent DP2 stacks = 2×63.25 aggregate
  throughput — not C1.

So: single-card realistic program tops at ~79 tok/s (kernel) → ~90-95 (kernel
+ 3bpw + residual fixes). Anything past ~95 is TP2-or-nothing, and TP2 is
priced but unbuilt.

## 4. Closed classes — do not retry (measurement that closed each)

| closed | evidence |
|---|---|
| MB=4/8 padding variants as speed levers | RL-101: +0.41/+4.72 pts vs ≥5 gate |
| NT axis on K=4 DPAS (2/4/8) | RL-102: flat/regressive 3/4 families |
| Fused single-kernel linear | upstream M=4 45.2 vs 35.1; RL-102 bound |
| Staged/serial standalone reduction (C1-R4) | RL-111: no order ≥15 %; D3 host-wait FAIL |
| Explicit register binding / B-bank split (C2, C2-V2) | ISA-disproved; V2 bit-exact but −3-4 % |
| EXL3_LUT, MUL16, K4_HALVES, DECODE_INLINE, BV_ALL, per-layer split-K target | upstream rejected/wrong/no-change (PROGRESS.md) |
| 256-GRF alone, DPAS_PREFETCH alone | spills / occupancy loss measured |
| Vector path at M=4 | 250-265 GB/s — 25-30 % worse than DPAS |
| Triton backend for 3bpw | 10.25 tok/s floor (RL-094); K3-ESIMD is 4.14× over it but 0.671× vs K4 |
| Whole W4A16/GPTQ class on this model | ADR-0010; 30-defect in the weights (RL-092) |
| Stock TP2 | impossible at load — `vllm_plugin.py:157-162`; xccl −45..−71 % class (RL-075) |
| Requant without margin awareness | RL-107 systematic gate failed |
| Draft-vocab env override | recipe env wins — must bind-mount (2026-09-30 VOID record) |
| Fresh flag sweeps on ubatch/target-threads beyond measured plateaus | upstream sweeps: cliff at 1664 threads; 1024/1408/2048 tuned |

## 5. Evidence index

- `results/2026/09/b70-probe1-exl3-rate-20260929/` — the 63.25/61.12 cells,
  PMU attribution (ccs0 ~88 %), draft-head line `server-startup.log:58`.
- `results/2026/09/b70-b6-microprobe-20260929/` — RL-098 per-family M=1/M=4
  table, 521 GB/s achievable recalibration, 34.46/29.44 ms passes.
- `results/2026/09/b70-b6-mb4-ab-20260929/` — RL-101 MB4 A/B (EXACT numerics,
  gate MISS).
- `results/2026/09/b70-b6-sweep-20260929/` — RL-102 NT sweep + guards
  (forced-vector, SKIP_GEMV had_out prices 14.0/7.8/12.7 µs).
- `results/2026/09/b70-k3-completion-20260929/RESULTS.md` — RL-100 chain,
  42.44 vs 63.25 verdict table.
- `results/2026/09/b70-c2-repair-20260930T085200Z/v2/` — C2-V2 parity-PASS /
  −3-4 % timing verdict.
- `results/2026/09/b70-resume-breakthrough-20260930T063200Z/REPORT.md` — C1-R4
  + C2 falsification record.
- `results/2026/09/b70-exl3-draftvocab-20260930T131812Z/` — VOIDED arm A/B;
  proves pruned draft head is production default.
- `/tmp/b70-k3-exl3-source-20260929/` — pinned upstream tree (15ded2f3):
  `csrc/exl3_ops.sycl` (dispatch/ladder/geometry: lines 172-318),
  `csrc/exl3_esimd.h` (GemvKernel 443-578, DpasKernel 591-861, build_k4
  669-716, B_FAKE 812-818), `docs/PROGRESS.md` (per-step kernel profile ~263,
  floors ~83, PL2 ~102/136, debug-probe history).
- `research/2026/b70-exl3-tp-decision-20260929.md` — TP2 pricing + kill rules.
- `research/2026/b70-exl3-round-attribution-audit-20260929.md` — measurement
  classes; §4 draft-bytes superseded by §0 correction above.
- `research/2026/b70-k3-exl3-extension-20260929/` — K3 patch series + P2 spec.
