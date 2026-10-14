---
title: "Independent adversarial review — Qwen3.8-27B C1 replan (QuaRot / IQ3 rollout / TP2 claims, and the rewritten main note)"
type: plan
date: 2026-09-24
status: final
scope: "Adversarial pre-commit architecture review. CPU/file work only; no GPU workload, build, server, config change, commit or publication performed."
---

# Independent adversarial review — Qwen3.8-27B C1 replan (2026-09-24)

Reviewer: B70 adversarial architect (independent of the lead design). Reviewed
artifact is the **rewritten** `research/2026/exl3-analysis-and-next-inference-levers-20260924.md`
(mtime 2026-09-24 21:25, `truth_verdict: PARTIAL`), the pre-replan original preserved as
`worklogs/EXL3-PRE-REPLAN-20260924.txt`, plus `OUTCOMES.md`/`LEVERS.md`, the D11 evidence roots,
the P4 trace record, the X570-F topology note, ADR-0008, and the xmx-dpas rate-class note.
No shared authority was edited.

## Verdict

**CONDITIONAL PASS of the rewritten main note; the pre-replan four-frontier claims were REFUTED
and are correctly withdrawn.** The rewrite's three withdrawals (QuaRot → 140 tok/s, IQ3 rollout
→ 1.5–2× Qwen27 prefill, zero-copy TP2 → 160–190+ tok/s) match my independent findings, with the
exceptions and residual corrections listed below. One blocking correction remains in a typed
authority (`registry/model-status.json`'s route summary repeats the un-reconciled 102.61 tok/s
headline without the non-portable-baseline caveat); one wording correction remains in the note.

## Per-claim verdicts on the pre-replan claims

| Claim (user/lead brief) | Verdict | One-line reason | Citation |
|---|---|---|---|
| "21.44 GB / 39.8 ms = 538.7 GB/s = 96.2% ceiling" is measured | **REFUTED** | Byte budget is an artifact-size model (15.20+2.54+3.39+0.30 = 21.43 GB, my recompute) and 39.8 ms is a step time *back-inferred* from an 80.3 tok/s client rate at an assumed yield — never a timing sample of the retained stack; no memory-controller traffic was ever measured | `worklogs/ARCHITECT-next-levers-20260921c.md` §0 (the arithmetic's own source); rewrite §3 |
| Token yield consistent with α = 0.5421 | **REFUTED (the old roofline's pairing)** | D11 counters give yield = 1 + 4·α = 3.1683 (438 accepted / 202 rounds, my recompute matches). At 73.68 tok/s that is a 43.0 ms step = 498.6 GB/s modeled (82% of 608), not 39.8 ms / 538.7 GB/s. 39.8 ms at yield 3.168 implies ≈79.6 tok/s — the rewrite's §3 numbers reproduce exactly. The old roofline silently mixed a 0.55–0.69-α run's step with the D11 yield and with 102.61 tok/s, which at 21.44 GB/step would imply 694 GB/s = 114% of rated peak (physically impossible on this model) | `results/2026/09/b70-improvement-probes-20260920/d11-baseline-decay-probe-20260924T142540Z/`; my recompute |
| Step 39.8 ms does not match a timing sample | **CONFIRMED** | 39.8–46.8 ms is a range derived from tokens/step at 80.3 tok/s (client post-first), not engine step timing; the D11-era cells give 43.0–44.1 ms on the same arithmetic | `ARCHITECT-next-levers-20260921c.md` §0; L15 cells |
| Kernel saturation ("diminishing returns <4% headroom", "memory-bound 96.2%") licensed by analysis alone | **REFUTED** | No non-measurement analysis licenses a saturation declaration: the effective-BW figure is the model divided by a derived step, and the rate-class note's own law says the 608/560 "practical ceiling" is not established for this access pattern. Post-first client rates are also not step rates (prefill amortization excluded, acceptance counted) | rewrite §3 (corrected); `research/2026/xmx-dpas-rate-classes-20260915.md` |
| QuaRot INT3/4: "all rotations free, one-cycle unpack, 39.8→25 ms, +20–30% ⇒ 140 t/s" | **REFUTED** | (a) 39.8→25 ms is 1.59× = **+59%**, internally inconsistent with "+20–30%"; (b) 25 ms at the D11 yield is 126.7 tok/s, and 140 tok/s needs a 22.6 ms step at yield 3.168 — the headline number is unreachable by its own arithmetic; (c) the "25 ms" uses only the 13.1 GB body: a real W4 step also reads the F16 full-vocab head (2.54 GB), draft side (3.39 GB) and state (0.30 GB) → 19.3–19.4 GB ⇒ 34.6 ms at 560 GB/s ⇒ ≤92 tok/s at yield 3.168. INT3 body-only gives ~17.6 GB total ⇒ 1.215× upper bound under unchanged bandwidth — quant's corrected byte model matches my recompute; (d) there is **no native INT3 DPAS or INT3 packed-weight path** in the inspected XPU loader/kernels (auto_gptq TYPE_MAP 4/8-bit only; oneDNN f16_int4 has no 3-bit variant), so INT3 is a new packing+kernel program, not a conversion; (e) pure s4×s4 DPAS 2× work/instruction (W7) is an instruction-rate fact and does **not** apply to W4A16 — every ≤8-bit int mix with an 8-bit operand is the 363-TOPS class and INT4 is a bandwidth lever only; (f) "all rotations free" is false for the GDN hybrid (online transforms around nonlinearities/recurrent state) | rewrite §4.1; `research/2026/xmx-dpas-rate-classes-20260915.md` (s4×s4 2.00× harness note; no INT3 row); OUTCOMES W7/L10 |
| 3.60× IQ3_XXS MUL_MAT_ID prefill rollout applies to dense GPTQ Qwen27 | **REFUTED** | The 3.60× is a **K216 MoE routed `MUL_MAT_ID` IQ3_XXS** isolated-op win on llama.cpp SYCL; Qwen27 is dense GPTQ on vLLM XPU. Different engine, operator class and quant; there is no dispatch to "roll out". Also §16/OUTCOMES W14: the e2e rung was **canceled on K216 retirement**, never measured; the n=1 "13% decode" figure is operator latency, not model decode | OUTCOMES W14 + scope-change note; rewrite §4.2 |
| Level Zero IPC no-copy TP2 >1 TB/s ⇒ 160–190 tok/s on documented no-P2P x8/x8 | **REFUTED as stated** | (a) IPC shares allocations across processes; it does not create peer routing (Level Zero memory spec, cited in rewrite §4.3); (b) >1 TB/s is the *sum of two local VRAM* bandwidths — irrelevant to a single TP2 step whose activation collectives must cross the interconnect; (c) per-card weight halving gives ~10.7 GB/step ⇒ ~17.6 ms at 608 GB/s **plus** 2·43 collective exchanges ≈ 2.6–8.6 ms ⇒ 20–26 ms step ⇒ 122–160 tok/s at yield 3.2 on optimistic assumptions, and measured host-staged collectives over these x8 links are far worse — 160–190 is a top-edge fantasy; (d) **the Cn knee is already measured at 172.48 tok/s aggregate (B=32) on the current single-card stack** (OUTCOMES W10/O8), so a multi-week TP2 transport program has no aggregate-serving justification either | rewrite §4.3; ADR-0008; OUTCOMES W10/O8; my recompute |
| P2P "documented no-P2P" is a permanent physics fact | **WEAK — downgrade the epistemic status, keep the operational block** | Local provenance is inherited, not measured on this board: the X570-F note asserts "no GPU-P2P interconnect exists" by *analogy to the B450-F* ("same conclusion as the B450-F"); the B450-F basis is topology inference ("No ACS on the GPU bridges; consumer AM4 root ports provide no usable P2P") plus one **context-confounded** `zeMemOpenIpcHandle` INVALID_ARGUMENT observed under `ZE_AFFINITY_MASK=0,1` (oneCCL multi-GPU-context path — the level-zero issue-8 thread itself attributes the error primarily to that misconfiguration, which is a different root cause than physical peer unavailability). No bidirectional `zeDeviceCanAccessPeer` + checksum probe has ever run on this host. compute-runtime #935 reports clean peer copies on AMD fam ≥ 0x17 with P2PDMA trust — the X570-F is exactly that class. **Conclusion: "definite no-P2P" is unproven; ADR-0008 default (independent stacks) stays BLOCKED/unchanged on risk grounds, and the note's ≤5-min capability-only probe (bidirectional query + checksum, fail-closed, no forcing) is the correct minimal decider** | `research/2026/b450f-dual-b70-pcie-topology-20260824.md` §3; `research/2026/x570f-dual-b70-pcie-topology-20260830.md` §1; `scripts/tmp/gh_issue_8_response.md`; ADR-0008 |
| Target / draft / head precision distinction | **CONFIRMED in the rewrite** | Draft-side INT4 pair is draft-only (0.636 GB head + 0.212 GB body ×4); the verify pass retains the unquantized 2.54 GB F16 full-vocab head; D8/L10 killed head quantization (72% / 93.4% top-1 vs ≥99% gate) and vocabulary slicing is not output-preserving. Rewrite §1/§5 keeps this boundary correctly | OUTCOMES L10; P4-qwen38-native-mtp-subgraph.md (head sharing: `_maybe_share_lm_head()`, 4 head reads/step) |
| Qwen dense vs MoE operator class | **CONFIRMED in the rewrite** | Qwen3.8-27B is dense GPTQ + hybrid GDN/attention on vLLM; every "routed"/IQ3/MUL_MAT_ID result belongs to retired K216. The rewrite's §2 scope table and §4.2 say so | OUTCOMES scope change 2026-09-23 |
| Closures (D11, D8, D12, D14, k=3, k>4, wheel bump) | **CONFIRMED CLOSED** | D11 measured FAILED (α 0.5421→0.5023, 73.68→71.83 tok/s, gate Δα≥+0.08/+3%); D8/D12/D14/L12 measured failures; k>4 flat. No reopening is licensed; rewrite §6 matches. Note the corrected D11 evidence root is `…T172339Z` (the `T182834Z` citation in OUTCOMES L15 row is a stale path and should be fixed by its owner — out of scope here) | OUTCOMES L10/L12/L13/L15, W13 |

## Recomputed arithmetic (the check, shown)

```text
α(mean of 0.980/0.975/0.124/0.089)      = 0.5420        (recorded 0.5421 OK)
yield = 1 + 4α = 1 + 438/202            = 3.1683
step(yield, tps)  : 73.68 → 43.0 ms ; 71.83 → 44.1 ms ; 80.3 → 39.5 ms
modeled BW 21.44 GB/step                : 43.0 ms → 498.6 GB/s (82.0% of 608)
                                          39.8 ms → 538.7 GB/s — only valid at the 80.3 tok/s pairing (α≈0.55), NOT at α=0.5421
102.61 tok/s @ 21.44 GB/step            : 694 GB/s = 114% of peak  ⇒ the 21.44 GB model cannot describe the 102.61 cell (acceptance/byte accounting differ); 102.61 is warm/Run-48 class and non-portable
byte budget                             : 2.5428 (F16 head) + 15.20 (INT4 body) + 4×0.8477 (draft) + 0.30 = 21.43 GB (model, not measured traffic)
rotation upper bound (body 15.2→17.64 GB total): 21.44/17.64 = 1.215× ceiling, before metadata/online transforms/acceptance
"25 ms ⇒ 140 tok/s"                     : 140×0.025 = yield 3.50 (α 0.625) — not the measured 3.168; at 3.168, 25 ms = 126.7 tok/s
"39.8 → 25 ms"                          : 1.592× = +59.2% step speed, not "20–30%"
TP2 sketch (per-card 10.7 GB + 86 collectives): ~20–26 ms step ⇒ 122–160 tok/s optimistic vs measured Cn B=32 172.5 tok/s aggregate already in hand
```

## Residual corrections required (small, exact)

1. **`registry/model-status.json` route summary** still leads with `best_tok_s: 102.61` without the
   non-portable-baseline caveat the rewrite §1 states. Not my file to edit — flagging for the status
   owner: add "warm/Run-48-class, not a portable baseline" to the route record or the next reader
   will rebuild the same impossible roofline. (Coordinator indicates status sync is in progress.)
2. Rewrite §3's "39.8 ms … approximately 79.6 tok/s" is correct at the D11 yield but the surrounding
   prose should stay explicit that **no step time on the retained stack has ever been measured** —
   both 39.8 and 43.0 ms are derived. (Present in substance; keep the wording in any downstream cut.)
3. Rewrite §4.3's P2P paragraph is correct on capability-vs-transport and correctly refuses a
   categorical no-P2P inference. Recommend adding one clause of provenance honesty: the local
   "no P2P" line descends from B450-F topology inference plus a context-confounded IPC error,
   never from a peer-access query on this board. (Optional wording; the disposition is already right.)
4. The n=1 192.2→166.6 µs P-3b figure must never be quoted as "+13% decode" — the rewrite §4.2
   already fixes this (1.154× operator throughput). Keep that phrasing downstream.

## On the lead's minimal first diagnostic (60-min exact-stack attribution)

**APPROVE with the agreed controls** (before/after uninstrumented controls, one discarded warmup,
graph replay identity + visibility in trace, exclusive longest/critical-path attribution not summed
overlapping durations, fail-fast stop on thermal/cache/dispatch anomalies, ≤60 min hard cap including
startup and cleanup, ≥5% credible net C1 gate via `S_net = 1/((1−f)+f/s+o)`). Specific verdicts:

- **Trace overhead controls — sufficient as specified**, with one addition: the uninstrumented
  bracket must be the *same request shape and prompt family* as the captured cell so the control-to-
  capture ratio is per-step, not per-request.
- **Longest-path attribution — correct method.** The stale `gemm_kernel` 79.7% figure is a summed
  device-event share from an `--enforce-eager`, pre-draft-pair, non-champion digest
  (`p4-mtp-profile-20260920T212400Z`, digest `2c427ef…`); it is inapplicable to the retained
  `f01e24f6…` + overlay + `VLLM_XPU_ENABLE_XPU_GRAPH=1` stack and must not be carried into the new
  attribution. Agreed: it does not decompose target W4A16 verify GEMMs (M ≤ k+1 = 5) from the
  full-vocab F16 head, the four draft INT4 head/body passes, GDN/graph residual, or host gaps.
  Add: report **graph-residual and inter-kernel host time as an explicit named bucket** — under
  graph capture that residual is exactly where a future lever would live, and it disappears in a
  kernel-only exclusive-path table.
- **60-min scope — reasonable and correctly fail-closed.** "Attribution unresolved" is an acceptable
  outcome; do not let the deadline induce a summary guess.
- **Dispatch visibility** ("fail-fast capture not visible") is correctly the primary abort: without
  replay visibility the trace answers nothing about the champion stack. Additionally require the
  trace to *name* the dense W4A16 small-M dispatch (oneDNN/int4 GEMM) for at least one verify layer;
  a generic `gemm` label is not evidence of the path (same trap as the 79.7% figure).
- **≥5% Amdahl gate — correct and appropriately brutal** (2× on 15% of step = 1.081× net). This is
  the mechanism that keeps the parked branches parked.
- One caution: the target head full-vocab read (2.54 GB modeled) is the largest *single identified*
  component by bytes and D8 has closed its quantization; the only surviving hypothesis about it is a
  faster unquantized implementation. The attribution should therefore report the head's exclusive
  share separately even if it clears no gate — it bounds what any future kernel program can claim.

## Blocking-obstacle list (what must be true for the retained plan to work)

1. **That a ≥5%-of-step removable inefficiency exists on the retained stack at all.** Least
   established: the modeled byte budget says the step is dominated by rate-class-floored weight reads
   and one closed-to-quantization F16 head. If attribution finds f < 9.5% everywhere, kernel hunting
   closes and the C1 frontier is acceptance/serving (Cn knee), which is already measured. This is the
   honest possible outcome.
2. **That the attribution can be obtained under graph capture at all** (replay visibility, tool
   compatibility with the pinned digest). Fail-closed to "unresolved" is specified; unresolved must
   not silently become "GEMM-bound, build a kernel."
3. **Any rotation program needs an INT3-capable runtime** (packing + loader + kernel + quality gates)
   — currently non-existent on the inspected XPU path — and a full-model transform story for the GDN
   hybrid (17408 not in the composite-Hadamard library; block-diagonal weaker). The parked branch's
   one-layer static gate is the correct minimum; the 1.215× modeled ceiling is its honest upper bound.
4. **TP2 remains blocked (ADR-0008) regardless of the capability probe**; a clean peer copy would
   earn a proposal, not a deployment, and the aggregate-serving case is already covered by the
   measured 172.5 tok/s Cn knee.

## Recommendation

Adopt the rewritten note's disposition as the plan of record: retain the champion recipe, run only
the bounded 60-min exact-stack attribution as the first decision (with the control and named-dispatch
additions above), keep the ≤5-min P2P capability probe as an independent fail-closed side test,
keep rotation at the one-layer static gate with the 1.215× ceiling, and leave all closures closed.
**Vetoed as promised gains:** QuaRot +20–30%/140 tok/s, IQ3 3.60×-rollout 1.5–2× Qwen27 prefill,
TP2 160–190+ tok/s, and any "kernel saturation / <4% headroom" claim resting on the 21.44/39.8 model.
No public headline from any of this (PROVISIONAL), and no authority edits from this review.

## Falsifier (smallest experiment that settles the strongest remaining doubt)

The 60-minute attribution itself is the falsifier for the central doubt (is there a ≥5% removable
fraction?). If forced to name one cheaper pre-step: **static dispatch identification on the pinned
image** (confirm the retained stack's verify GEMMs are the oneDNN f16_int4/bf16_int4 W4A16 path at
M ≤ 5, and that the F16 head is a separate full-vocab call) — zero GPU, and it either names the
object the trace must attribute or kills the kernel hypothesis before the window opens.

Evidence/working set used: `registry/model-status.json`, `research/2026/exl3-analysis-and-next-inference-levers-20260924.md` (rewritten), `worklogs/EXL3-PRE-REPLAN-20260924.txt`, `worklogs/ARCHITECT-next-levers-20260921c.md`, `OUTCOMES.md`, `LEVERS.md`, `worklogs/P4-20260920.md`, `research/2026/xmx-dpas-rate-classes-20260915.md`, `research/2026/x570f-dual-b70-pcie-topology-20260830.md`, `research/2026/b450f-dual-b70-pcie-topology-20260824.md`, `decisions/adr-0008-dual-b70-independent-default.md`, `results/2026/09/b70-improvement-probes-20260920/d11-*`, `scripts/tmp/gh_issue_8_response.md`.
