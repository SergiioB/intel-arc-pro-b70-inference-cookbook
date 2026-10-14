---
title: "B70 Improvement Roadmap — Master Board"
type: plan
status: draft
date: 2026-09-20
---

# B70 Improvement Roadmap — Master Board

> **PROVISIONAL — NOT FOR PUBLIC HEADLINE.** All proposed gains, latency budgets, effort
> estimates, and target throughput figures in this folder are **E0 architectural projections,
> not measured results**. Historical measurements are explicitly identified. Nothing becomes
> a public benchmark headline until the benchmark protocol, correctness gates, and claims
> standard pass. This folder authorizes no model load, interruption, download, conversion,
> commit, push, deployment, or publication — each phase requires its own approved resource
> scope, and public release requires separate operator authorization.
>
> Filed 2026-09-20 from the b70-architect two-session analysis (EXL3 fit + improvement map).
> Engine dossier paths verified at filing (`engines/llamacpp-sycl/`, `vllm-xpu/`,
> `openvino-genai/`, `cascadia/`).

## Why this folder exists in this shape

The operator asked for (a) the **reasoning behind every decision** to live next to the
decision, not in a chat log; (b) one file per phase so an executing agent can load exactly
one point in detail; and (c) a **status convention** (`RESEARCH` / current status /
`COMPLETED`, plus a `DONE-` filename prefix on completion) that the master board mirrors.
Every phase file therefore has three parts: a machine-updatable **status block**, a
**"Why this decision"** section (evidence chain, alternatives considered and rejected,
what would change the call), and the **execution spec** (baseline, target, work breakdown,
gates, benchmark plan, stop rules, deliverables, cookbook integration).

## Status board

| Phase | File | Status | Depends on | Outcome / gate result |
|---|---|---|---|---|
| P0 — Baselines + probe batch | [DONE-P0-baseline-and-probes.md](DONE-P0-baseline-and-probes.md) | COMPLETED | — | 2026-09-21 (RL-030): all 7 probes resolved; device-ID repair measured at **2.44 %** (gate fails); DPAS `OPS_PER_CHAN=8` measured at **2.0× MACs/instruction**; two recorded claims corrected (fused IQ3 exists locally; the host wait is the graph blocker). No speed record implied by this phase. |
| P1 — K216 routed-IQ3 GEMV repair | [P1-k216-routed-gemv.md](P1-k216-routed-gemv.md) | WITHDRAWN | P0 | end-to-end ABBA A/B on the re-downloaded artifact: **clean 9.25 → patched 9.577 engine tok/s = +3.54 %** (n=6 each, parity holds). Gate ≥10 % fails, 20 tok/s refuted. **Repair kept** (correct + unblocks SYCL graph capture for `MUL_MAT_ID`); speed object → P3 (prefill) / P5 (decode GEMV, 5.52× off floor) |
| P2 — Nemotron Mamba2 fusion | [P2-nemotron-mamba2-fusion.md](P2-nemotron-mamba2-fusion.md) | WITHDRAWN | P0 | premise refuted by measurement: the fusable Mamba/SSD family is **8 % of device time** (medians 55–357 µs) against **73 % MoE+GEMM**, and graph capture has already paid down the launch cost the 1.01 ms/token budget assumed ⇒ fusion cannot deliver its gain. Trace reassigned to **P5** |
| P3 — K216 grouped-IQ3 prefill | [P3-k216-grouped-iq3-prefill.md](P3-k216-grouped-iq3-prefill.md) | RESEARCH | P1 (withdrawn; prerequisites validated) | entry PASSED (ub 512→1024 = **1.292×**) and **attribution complete**: the routed op costs 86.6 ms/layer at ub512 (≈3.72 s of a 5.06 s ubatch) at **1.72–1.82 TFLOPS vs ~45 TFLOPS for a comparable dense GEMM — a ~25× efficiency gap** that is expansion + per-expert structure. Kill criterion not triggered ⇒ the kernel work is the remaining object |
| P4 — Native-MTP subgraph (correct) | [P4-qwen38-native-mtp-subgraph.md](P4-qwen38-native-mtp-subgraph.md) | RESEARCH | P0 | draft-side INT4 pair = **+37…46 % decode** (reproduced under a both-arms-patched control) and **output-preserving under greedy** (byte-identical to baseline, deterministic within each arm) ⇒ the acceptance variance is drafting efficiency, not emitted-output change. Remaining before publication: claims-audit fix list (metric naming, patch provenance, telemetry) + sampled-path equivalence statement |
| P5 — Small-M INT4 specialization | [P5-smallm-int4-specialization.md](P5-smallm-int4-specialization.md) | WITHDRAWN | — | **decision**: the small-M "gap" is arithmetic, not waste (M=1 measures 54.78 µs vs a ~54 µs weight-read floor); not enough to justify a program. Residual M≈4–16 question folded into P3 (evaluating it against today's kernel would measure what P3 replaces) |
| P6 — Long-context/mixed-load gates | [P6-long-context-gates.md](P6-long-context-gates.md) | RESEARCH | — | OV ≥196K classified as **second-generation growth**: one long generation passes at any length (40/64/96 tokens), the **second** fails with `CL_OUT_OF_RESOURCES` at 44 MiB free; 131072/163840 full protocols pass. Not a capacity ceiling or a misclassification — upstream-report shape written for #4483 |
| P7 — Scheduler + context-aware speculation | [P7-scheduler-speculation.md](P7-scheduler-speculation.md) | RESEARCH | P6 envelope declared | **the serving lever is measured**: concurrency B=1 → B=4 gives **2.39× aggregate** (48.45 → 115.70 tok/s) on the baseline config and **2.27×** (61.86 → 140.30) on the draft-pair config, saturating ~150 at B=8. Open: where the knee turns, long-context interaction, latency budget |
| P8 — Bounded quant screen | [P8-bounded-quant-screen.md](P8-bounded-quant-screen.md) | RESEARCH | — | **format question closed**: all five community "AWQ" artifacts for this model declare `compressed-tensors`, not `awq` — and that path loads and serves here (5/6 bounded suite). Remaining P8 object: **iso-byte quality** on a compressed-tensors W4A16 artifact (available ones are ~8 % larger than the champion). No speed claim |
| P9 — SpecForge quantized-target drafter | [P9-specforge-quantized-drafter.md](P9-specforge-quantized-drafter.md) | PLANNED | P4/P7 stable, P8 frozen | — |
| P10 — Cookbook completeness | [P10-cookbook-completeness.md](P10-cookbook-completeness.md) | RESEARCH | P0 inventory; closes last | gap plan filed (`worklogs/P10-gap-plan-20260920.md`): OV + Cascadia still at **0 catalog records**, 1 live stale record (60.13), 5 taxonomy violations to fix; no edits made yet |
| Deferred / tracked | [DEFERRED-AND-TRACKED.md](DEFERRED-AND-TRACKED.md) | TRACKED | reopen triggers per item | — |

**Start here for results:** [OUTCOMES.md](OUTCOMES.md) — one page of what won (with numbers),
what the measurements killed, and the single decider for each open question.
**Every lever, one row each:** [LEVERS.md](LEVERS.md) — all tested levers with effect size, cell,
engine, verdict (win / loss / neutral / open / blocked) and evidence path.

## Status vocabulary (the only legal values)

| Status | Meaning | Who sets it |
|---|---|---|
| `PLANNED` | Filed, not started. No resources consumed. | filing (done) |
| `RESEARCH` | Active investigation: probes, traces, manifests, source reading. **Set before any probe runs.** | executing agent |
| `IN_PROGRESS` | Implementation/build/campaign underway after the phase's entry probe passed. | executing agent |
| `COMPLETED` | Gates passed **and** close-out loop finished (evidence → ledger → renderer/charts → dossiers → cookbook candidate). | executing agent |
| `BLOCKED` | External gate (upstream fix, image, resource window). Blocker must be named in the status log. | executing agent |
| `WITHDRAWN` | No-go decided with recorded evidence (projection refuted, or two failed gates). Content is preserved, never deleted. | executing agent |

Completion marking: when a phase reaches `COMPLETED`, the executing agent renames the file
with a `DONE-` prefix (e.g. `DONE-P1-k216-routed-gemv.md`), updates every link in this
README, and fills the "Outcome / gate result" cell with the measured result and evidence
path. `WITHDRAWN` phases keep their filename and record the refuting evidence.

Execution worklogs go in `worklogs/Pn-<YYYYMMDD>.md` (create the folder on first use) so
phase files stay specs, not diaries.

## Session log — 2026-09-21 (first execution session)

Worklogs from this session, in the order the work happened. Each one is self-contained: what was
asked, what was measured, what it refuted.

| Worklog | Phase | Outcome in one line |
|---|---|---|
| `worklogs/P0-20260920.md` | P0 | probe batch opened; K216 operator A/B; later a **K216 end-to-end ABBA A/B** addendum (+3.54 %) |
| `worklogs/P0-inventory-20260920.md`, `P0-publication-inventory-20260920.md` | P0 | route identities; catalog gap inventory (OV + Cascadia at 0 records) |
| `worklogs/P1-…` → see `P1-k216-routed-gemv.md` | P1 | device-ID repair measured end-to-end, **gate fails → WITHDRAWN**, repair kept as the SYCL-graph prerequisite |
| `worklogs/P2-20260921.md` | P2 | Nemotron baseline + dispatch trace: **GEMM 73 % / fusable Mamba family 8 %** ⇒ the phase's 1.01 ms/token budget is not available |
| `worklogs/P3-20260921.md` | P3 | prefill ubatch probe: **1.292×** from doubling the ubatch ⇒ entry gate PASSED, phase proceeds |
| `worklogs/P4-20260920.md` (§1–§13) | P4 | byte budget kills the 0.25 ms/draft target; step is GEMM-bound; the draft-side INT4 pair is **+37 % @p≈7000 (n=5) / +46 % @p512 (n=3)** on the champion stack, gated by host-overhead removal, not context |
| `worklogs/P6-20260921.md` | P6 | OV ≥196K failure classified: **real memory exhaustion** (44 MiB free / 16 MiB largest block at failure), phase premise refuted |
| `worklogs/P8-awq-loader-probe-20260921.md`, `P8-…` | P8 | AWQ entry gate PASSED (routes to an existing XPU op) + falsifying test executed on the deployed wheel |
| `worklogs/P10-gap-plan-20260920.md`, `P10-apply-ready-20260921.md` | P10 | catalog gap plan; three power-claim edits applied to the cookbook working tree (uncommitted) |
| `worklogs/ARCHITECT-next-points-20260921.md` | — | architect's ranked next points after the no-huge-wins finding |
| `worklogs/CLAIMS-AUDIT-20260921.md` | P4/P1 | independent audit: both headlines **provisional**; cell label, metric name, measured-window acceptance and TTFT wording need the fixes it lists |

Artifacts re-downloaded this session (not previously on the host): the 83 GiB K216 GGUF, the
Nemotron GPTQ-INT4 target, the champion vLLM image digest, and a community AWQ W4A16 artifact for
P8's iso-byte screen.

## Disk discipline (learned the hard way — root hit 97 % twice)

The 2026-09-21 reclaim freed ≈140 GB (95 GB of a staging copy the host-state doc already marked
"until explicitly deleted", plus 45 GB of caches and two unused images); root went 8.3 → 32 GB free
and the model disk 46 → 141 GB. Manifest: `results/2026/09/disk-reclaim-20260921/`. Rules for the
next agent:

1. **Nothing growable lives on `/`.** Docker images, HF downloads, conversions and staging belong on
   `/mnt/*`. Check `df` and size ×2 before every download; keep a root floor of ~10 GB; artifacts
   above ~1 GB must never transit root.
2. **Evidence is immutable, caches are not.** Everything under `results/**` stays. Reclaimable: OV /
   compilers / package caches, digest-pinned container images, and re-downloadable model artifacts —
   each only with a logged way to re-derive it. Sole-copy conversion products under `models/` count
   as evidence-grade: do not delete them to make room.
3. **Declare the byte budget in the manifest** of any run that writes more than a few GB, and treat
   the staging teardown as part of §13 cleanup, not an afterthought.

## Protocol for the executing agent

1. **Load context the repo way first**: follow `AGENTS.md` §0 — name the intent
   (`research`/`bench`), run `tools/situation.py compile`, obey the empty-GPU / VRAM /
   power gates before any model load. This folder does not override the contract.
2. Read this README and the **status board**. Pick the lowest-numbered phase that is not
   `COMPLETED`/`WITHDRAWN` and whose dependencies are `COMPLETED` (or not applicable).
3. Read that phase file **in full** — including "Why this decision" and the stop rules.
4. Move it `PLANNED → RESEARCH` (status block + status log + this board) **before**
   running anything. Probe before building: every phase names its smallest falsifying
   probe; run it before multi-hour work.
5. Obey the stop rules exactly. Two failed gates from the same design = stop, not retry.
   A refuted projection = `WITHDRAWN` with evidence, then move to the next phase.
6. Close out per §1.5 below, set `COMPLETED` only after the full loop, apply the `DONE-`
   rename, update this board.
7. Never weaken a gate to finish a phase. Preserve rejected/failed evidence.

## Why the phases are ordered this way (the global rationale)

Priority is

```
P(success) × (expected multiplier − 1) ÷ engineering cost,
```

with separate credit for enabling otherwise-unavailable capacity or correctness. A 1.00×
change earns no speed credit. On top of that formula, three ordering rules decided the
queue:

1. **Falsify before constructing.** P0 exists because the two most expensive lessons in
   this repo — the quarantined MTP prototype (RL-021, which validated a computation that
   omitted the actual MTP math) and the K216 "2.25×" target (which budget math shows the
   84 sync sites alone cannot deliver) — were both *plausible targets that failed
   arithmetic*. Probes are hours; kernels are weeks.
2. **Structural losses before tuning.** The fleet's biggest gaps are host synchronization
   and fragmented execution (K216's 84+ D2H waits, Nemotron's 736 launches/token), which
   no config change can touch — the tried-matrix already closed config space. Scheduler
   and speculation-economics tuning (P7) only makes sense on a correct, stable kernel
   path (P6 first: never tune around a correctness bug).
3. **Representation changes last.** The quant baseline (AutoRound-contract GPTQ-INT4
   W4A16) is the measured champion; it gets displaced only by evidence (P8), and a
   drafter trained against the target (P9) requires that target representation to be
   frozen first — training against a moving target is exactly the mismatch that capped
   the off-the-shelf drafter at ~23–30% acceptance.

## Hardware and numerical invariants (accepted measurements for planning)

- BF16×BF16 DPAS ≈ 170 TFLOPS; any integer mix containing an 8-bit operand ≈ 363 TOPS.
- W4A8 does not exceed W8A8 arithmetic throughput; **W4A16 does not inherit the integer
  DPAS rate class** (its dominance is weight bytes + implementation, not math rate).
- Battlemage has no native FP8 DPAS path; FP8-W8A16 reconstructs into a floating-point
  path (half-rate class).
- Pure W4A4/W2A2 (OPS_PER_CHAN=8) is unmeasured — P0 probe.
- Single-card GDDR6 bandwidth class ≈ 608 GB/s; dual-card default is independent engines
  (no P2P, host-staged TP2 is gated research per ADR-0008).
- Native operand support, compressed storage, actual kernel dispatch, and end-to-end
  speed are four separate facts.

## Guiding principles

### Optimize a measured critical path

Before any multi-hour build, conversion, calibration, or training run:

1. Declare the hypothesis.
2. Declare the metric and minimum useful improvement.
3. Identify the smallest falsifying probe.
4. Set the resource scope and timeout.
5. Pass that probe before committing the larger effort.

One variable changes per comparison gate. Do not combine a new quant, kernel, draft,
power cap, and scheduler configuration into a purported causal A/B.

### Safety and operation

Follow `~/B70-DOCS/AGENTS.md` before every load: identify the host, compile
the intent working set, verify no conflicting process/container, check empty-card VRAM,
resolve DRM/hwmon paths dynamically, restore the 150 W baseline before applying a
declared campaign cap, enforce post-load VRAM gates, one declared engine + one campaign
harness, restore 150 W and retain cleanup evidence at exit. K216 retains its established
partial-offload and dual-card placement budget.

### Benchmark and quality contract

Default final card: C1 p2048/g1 cold input; C1 p8192/g1 cold input; C1 p512/g128 decode;
C1 p8192/g128 decode; one fixed, hashed representative natural-text workload. Screens use
matched n=3; finalists use **fresh n=5**. Discard same-shape warmup. Recommended model
sampling; greedy is diagnostic only. HTTP cold input: unique leading entropy, exact token
counts, zero prefix-cache-hit delta, full requested generation length, exactly one
measurement client for C1. For hybrid-GDN models: **per-position logit parity against
F16/reference — never PPL ranking** (upstream llama.cpp#28879). New kernel paths require
reference/operator comparisons before model-quality evaluation; stateful kernels
additionally require prefill/decode equivalence, sequence isolation, reset, and
speculative rollback tests.

### 1.5 — every phase closes the loop

1. Immutable raw evidence and manifest.
2. Claims/correctness assessment.
3. New research-ledger entry and campaign registration (`RL-TBD-P*` in phase files are
   **planning aliases, not allocated ledger IDs** — allocate on execution start).
4. Evidence-driven renderer.
5. Charts and vision-QA PASS.
6. Engine and model dossiers.
7. Cookbook page/catalog candidate.
8. Authorized public release.

Display-refresh order: **renderer → charts → engine dossiers → cookbook pages →
release.** Result roots use the actual execution date (`results/2026/<MM>/…`) — never
backdate evidence.

## Sources and baseline anchors

| Topic | Source |
|---|---|
| Current routes and champion coordinates | `registry/model-status.json` |
| Blocked / validated / quarantined paths | `registry/research-ledger.json` |
| DPAS rate classes | `research/2026/xmx-dpas-rate-classes-20260915.md` |
| Older implementation targets (superseded where corrected here) | `research/2026/megakernel-implementation-roadmap-20260829.md` |
| K216 decode pipeline | `research/2026/dsv4-k216-request-to-output-pipeline-20260825.md` |
| K216 prefill | `research/2026/dsv4-k216-prefill-ubatch-20260825.md` |
| Invalid MTP prototype (quarantine) | `research/2026/fused_mtp_draft_kernel/QUARANTINE.md` |
| Qwen quant comparison | `results/2026/09/qw38-quant-format-showdown-20260913/` |
| OV MTP/context evidence | `results/2026/09/qw38-ov-mtp-20260914/` |
| Native-MTP/DFlash diagnostics | `results/2026/09/qw38-mtp-depth-20260914/` |
| FP8 speed attack + ceilings | `research/2026/qwen38-fp8-tp2-speed-attack-and-ceilings-20260829.md` |
| Benchmark protocol | `docs/methods/benchmark-protocol.md` |
| Chart standard | `docs/methods/chart-standard.md` |
| Public promotion | `docs/governance/public-benchmark-publication.md` |

Historical baseline anchors (re-associate exact coordinates before matched use; unknown
fields stay unknown until the manifest is recovered):

- Qwen3.8-27B GPTQ-INT4 G128 + BF16 MTP4: 102.61 tok/s, p512/g128, recommended sampling, n=5.
- Nemotron-3.5-Lightning-30B-A3B GPTQ-INT4 G64 + DFlash n7: 186.61 tok/s, p2048/g128, n=5 client post-first, 150 W cap.
- K216: ~8.87 engine decode tok/s; ~105 pp prefill in scoped configurations.
- OV GenAI INT4-GDN8 + MTP5: ~79 tok/s @512; composite curve to 128K.

## First five sessions

1. **Session 1 — Compile the evidence and source baseline** (P0). Recover exact manifests
   for K216, Nemotron, Qwen MTP; resolve engine revisions and source paths; verify the
   RL-021 quarantine; inventory OV/Cascadia catalog gaps. Declare probes, resource
   windows, minimum useful gains, deadlines. No multi-hour build or conversion.
2. **Session 2 — Prove or refute the critical-path budgets** (P0). Capture the smallest
   K216 routing trace, Nemotron state/dispatch trace, native-MTP subgraph trace. Verify
   graph replay separately from compilation. Decide whether K216 can plausibly save
   62.74 ms/token; withdraw unsupported targets immediately.
3. **Session 3 — Smallest K216 routed-IQ3 operator repair only** (P1). Expert IDs
   on-device for one eligible path; placement and quantization preserved; operator parity
   tests. No full-model persistence, graph capture, or new offload settings. If the probe
   refuted useful gain: record no-go, advance to P2.
4. **Session 4 — Validate and screen the repair** (P1). Operator/model correctness,
   verified removal of the intended host waits, matched n=3 p512/g128 and p8192/g128
   screens with prefill regression checks. Stop after the second failed gate.
5. **Session 5 — Close the first decision** (P1 close-out or P2 entry). Fresh n=5
   finalist cells and the full evidence → ledger → charts → dossiers → cookbook-candidate
   loop; or preserve evidence and start Nemotron's smallest state-safe fusion. Either way:
   a reproducible decision, not an unmeasured patch.
