---
title: "ARCHITECT decision — next points after the no-huge-wins measurements"
type: plan
date: 2026-09-21
status: final
---

# ARCHITECT decision — next points (2026-09-21)

Operator question: "any huge wins? if not, give next points."
**Answer: no huge wins were found. Confirmed.** Three measurements (device-ID repair
= 1.025×, MTP draft-cost budget refuted 22.3×, both draft-side INT4 patches inert at
C1 short context) overturned the roadmap's three biggest E0 assumptions. What remains
is one cheap unresolved thread with a measured +36.9 % behind it, one scoped ~2×
kernel opportunity whose artifact lands within hours, and a queue of sub-1.15×
certainty work. Decision below.

Inputs: `DONE-P0-baseline-and-probes.md`, `P1-k216-routed-gemv.md`,
`P3-k216-grouped-iq3-prefill.md`, `worklogs/P4-20260920.md` (§7–§9),
`results/2026/08/qwen38-pr-ab-20260818T205619Z/` (three-arm screen),
`results/2026/09/b70-improvement-probes-20260920/`.

## 1. Is anything left with a credible >1.15× end-to-end opportunity, runnable now?

**No, not in hand.** Ceiling arithmetic on every live candidate:

| Candidate | Mechanism | Ceiling (e2e) | Status |
|---|---|---|---|
| Draft-INT4 pair under mixed load | Removes ~12 GB/step of head+MTP reads (head F16 2.543 GB × 5 = 12.7 GB/step; pair → ~4 GB) | **Measured +36.9 %** on champion, agentic screen | Attribution unresolved — see §2. The only thread with a measured >1.15× behind it. |
| K216 grouped-IQ3 prefill (P3) | Kills host counting-sort + IQ3→FP16 expansion (5.2× byte inflation) feeding oneMKL | ~1.5–2.0× **isolated prefill** (E0, scoped) | Artifact lands in hours; entry probe unpaid. Largest scoped opportunity. |
| K216 decode GEMV to floor (P1 remnant) | Fused op 174.9 µs vs 31.7 µs weight-read floor; 84 ops/token | **1.119× max** (84 × 143.2 µs = 12.03 ms of 112.74 ms); realistic ~1.07× at 2× kernel | Below the 1.15× bar alone. Cannot be a phase objective. |
| K216 compatible 216-expert GPU draft | Speculation over the repaired fused path | ~1.1–2.0× (9.6–18 tok/s envelope, AGENTS §16) | Draft does not exist; multi-week; not runnable now. |
| Nemotron Mamba2 fusion (P2) | 736 launches/token → fused SSD block in GRF/SLM | Potentially large, unscoped | Kernel project; artifact landing re-opens the premise, not the win. |
| AWQ (P8) | Packing change onto dispatched `int4_gemm_w4a16` | **~1.00× expected** — same bytes, same rate class as GPTQ champion | Cheap, but the passed entry gate proves it is *cheap*, not *valuable*. |

Refuted/withdrawn (do not revisit): MTP draft-cost reduction (byte budget 22.3× over),
device-ID repair as a speed lever (2.44 %), elementwise fusion (~10 % hard ceiling from
the ~300 k-kernel trace), graph capture as a P4 tool (step is 79.7 % GEMM-device time).

**What would create a win, ranked:** (1) reproduce and protocol-grade the pair at the
regime it was measured in; (2) P3 prefill kernel, gated by its entry probe; (3) a
216-expert-compatible drafter (only after P1/P3 close). Sub-byte DPAS note: the
s4×s4 = 2.00× s8×s8 microbench confirms pure-W4A4 doubles MACs/instruction, does not
contradict the mixed-operand rate-class law (W4A8 ties INT8), and changes no serving
decision — decode is byte-bound and no A4 activation path exists.

## 2. The +36.9 % discrepancy

Structure of the evidence (P4 worklog §7–§9): the +36.9 % came from a three-arm
agentic screen on the **champion image** (`f01e24f6…`, `cache=off`,
`VLLM_XPU_ENABLE_XPU_GRAPH=1`, 230 W): v5 44.01 → v5+pair 60.23 median t2+. The
nulling A/Bs ran at **C1 p512/g128 on the local image** (`2c427ef4…`, torch.compile),
one variable each: head −0.5 %, MTP body +0.7 %.

**Attribution: (a) regime, with (d) unresolved.** The byte model is coherent: at C1,
baseline step reads ≈ 15.2 GB target + 12.7 GB head + 3.4 GB MTP ≈ 31 GB; the pair
removes ~12 GB, worth ≥1.25× *if the step were byte-floor-bound*. It measured ±0.5 %,
so at C1-compiled the step is overhead/latency-bound and byte savings do not convert.
Under concurrent long-context load, batching saturates the engine, the same 12 GB/step
saving converts to throughput — exactly the screen's regime. That is mechanism (a).
But (d) is not excluded: every null ran on a different image *and a different execution
mode* (torch.compile vs XPU graph), so the champion's C1 behavior is unmeasured. (b)
v5-GDN interaction is subsumed: the delta is measured against the v5 arm itself. (c)
methodology is down-weighted because the screen recorded `cache=off`; residual risk is
arm-order/session variance (n = 15 turns/arm).

**Single cheapest discriminator:** rerun the screen **on the champion image, two arms
only** (v5 vs v5+pair), **arm order reversed** (pair first), prefix-cache counters
bracketed, 3 sessions × 5 turns, GPU 1, 230 W, restore 150 W after. Same image as the
original positive, so a null cannot be blamed on (d) — it would convict (c); a
reproduction (pair−v5 ≥ +20 %) convicts (a) and immediately earns the follow-on: a
protocol-grade long-context/batched cell (p8192/g128, one variable = pair, n=3) to
convert the screen into a citable cell.

**Ranking: yes, it outranks all other GPU work** — it is the only live thread with a
measured >1.15×, it costs ~1 h on an idle GPU 1 while the K216 download finishes, and
every other P4 conclusion stays confounded until the image constant is fixed. It does
not block CPU-only work (item 5) or the K216 landing checks.

## 3. P1's remaining object

**Decision: re-scope and close P1; do not transfer its kernel object into it.**

- The repair itself: keep. It is written, parity-passing at K216 shapes, and it is the
  named prerequisite for `MUL_MAT_ID` graph capture (`ggml-sycl.cpp:5687`). Its speed
  value is measured at 1.025× — record that honestly; never cite it as a win. Note
  dual-card capture stays refused (`device_count > 1`) regardless, so the prerequisite
  only matters for single-GPU experiments; no graph work is authorized by this.
- When the 83 GiB artifact lands: one matched e2e confirmation (L5 placement, p512/g128
  and p8192/g128, n=3, repair on/off) to confirm the 2.44 % end-to-end, then the commit
  decision. P1 then goes to `COMPLETED` (repair validated + targets withdrawn), not
  `WITHDRAWN` — the deliverable (a correct prerequisite) exists.
- The 5.52×-off-floor decode GEMV: **ceiling 1.119× end-to-end — below the phase bar.**
  Criterion applied: *an object earns its own phase only if it has a credible ≥1.15×
  end-to-end path on an artifact on this host within the phase budget.* It fails.
  Reassign: decode GEMV (M=1 efficiency, SLM staging, producer/consumer sub-groups) →
  **P5 as a secondary object**; prefill grouped-IQ3 (the ~1.5–2.0× scoped opportunity,
  sharing the same reconstruction primitives) → **P3 as primary**, which already
  declares this dependency structure. K216 landing re-opens both for e2e testing
  within days — which is precisely why P1 should close cleanly now rather than linger.

## 4. Ordered next five work items

1. **Guarded pair reproduction on champion `f01e24f6…` (§2 experiment).** Only
   measured >1.15× thread; ~1 h GPU 1; runs while K216 downloads.
   *Precondition:* docker root lives on the 27 GB root FS — verify headroom / prune
   before the pull (prior incident class).
2. **K216 e2e re-baseline + P1 close-out on landing.** Refresh the 8.87 tok/s / 105 pp
   anchors at the established L5 config (n=3 screens + regression), confirm the 2.44 %
   repair delta e2e, then the commit decision. Needed before any P3 claim is quotable.
3. **P3 entry probe (same session as 2 where possible): per-expert-M distribution +
   prefill attribution** (host counting-sort vs IQ3→FP16 expansion vs oneMKL at
   -ub 1024). Gate per P3's own rule: grouping+expansion+launch < ~30 % of prefill
   time → withdraw the 2× and record. Hours, and it decides the only large scoped win.
4. **P2 entry probe when the Nemotron artifact lands:** baseline refresh against the
   186.61 anchor + the 736-launch dispatch trace. Cheap; converts P2 from projection
   to measured premise (or kills it early). AWQ/P8 A/B is demoted to opportunistic
   idle-window work — expected ~1.00× by the rate-class law.
5. **Publication backlog burn-down (zero GPU, start immediately):** OV GenAI and
   Cascadia catalog records (0 each), retire the stale 60.13 record, fix the five
   taxonomy violations. Converts banked evidence to value and is prerequisite hygiene
   for any headline the above might eventually produce.

**Operator decisions needed (do not spend without the operator):**

- **D1** (only if the pair reproduces): approve concurrent-agentic mixed-load as a
  publication-eligible evidence class — without a protocol amendment P4 can never cite
  the win it measured. The champion already ships the pair env; no deployment decision.
- **D2** (after item 2): commit destination for the 2026-08-26 uncommitted repair in
  `~/llama.cpp` — local-only branch vs upstream PR (§7 patch registry).
- **D3** (before item 1): authorize docker prune / data-root relocation for the
  champion image pull on the constrained root FS.
- **D4** (only if the K216 download finishes mid-screen): GPU scheduling — recommend
  finishing the 1 h screen first, since K216 needs both GPUs.
- **D5** (only if head-dominance is confirmed at protocol grade): authorize extending
  head INT4 to the target verify pass (the 5th head application) behind a logit-quality
  gate — logits-touching change, acceptance risk.

All numbers herein remain PROVISIONAL — NOT FOR PUBLIC HEADLINE until the protocol,
correctness, and claims gates pass.
