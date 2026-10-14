---
title: "Campaign — Qwen3.8-27B improvement night (2026-09-25)"
type: campaign
date: 2026-09-25
status: living
---

# Campaign — Qwen3.8-27B improvement night (2026-09-25)

Registry: `CAM-2026-QWEN27-IMPROVEMENT-NIGHT-0925`. Status: validated_internal (E2),
self-reported, not independently reproduced. **Screens only — no promotion passed its
gate; the champion recipe is unchanged.**

## Question

Convert the open questions of the 2026-09-24 replan (`research/2026/exl3-analysis-and-next-inference-levers-20260924.md`
§5/§6) into measured verdicts on the retained single-card champion
(`SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` `-nogidx`, image f01e24f6, MTP4 greedy
drafting, draft-only INT4 pair, FP8 KV, XPU graphs, recommended non-thinking sampling).

## Results (all PROVISIONAL — thermal substitution, RL-065)

| Phase | Verdict | Numbers | Evidence |
|---|---|---|---|
| P1 host-prep metadata cache | **LOSS** | p512 pooled medians 78.18 vs 77.02 tok/s = +1.51% (gate +4%); p8192 72.92 vs 70.61 = +3.27%; n=6/arm/anchor, ABBA/BAAB; canary + greedy byte-identity preserved; iso-power 226.5–229.3 W | `results/2026/09/b70-improvement-probes-20260920/p1-qwen27-hostprep-screen-20260925T082440Z/` |
| P2 verify-phase composition | **Kernel hunting closed** | Measured buckets: host 4.8 ms/10.7%, draft 5.5 ms/12.3%, head 4.2–4.7 ms/9.4–10.5% (model), body 28–30 ms/63–67% (unresolved under graphs). L-B premise (3.39 GB redundant reads) inconsistent with the measured bucket (≥616 GB/s implied); L-C at its weight-read floor | `results/2026/09/b70-improvement-probes-20260920/p2-qwen27-eager-composition-20260925T085704Z/RESULTS.md` |
| P3 #56096 dispatch | **NOT APPLICABLE** | Gate is `qwen3_next.py`-only + requires `is_cuda()`; artifact resolves `qwen3_5.py`. Screen correctly skipped | `results/2026/09/b70-improvement-probes-20260920/P3-56096-DISPATCH-GATE-20260925.md` |
| P4 prefill knob | **Non-lever** | TTFT medians 4.679/4.689/4.683 s at max-num-batched-tokens 8192/4096/16384 (±0.25%), p8192/g1 exact endpoints, paired prompts n=3/config | `results/2026/09/b70-improvement-probes-20260920/p4-qwen27-prefill-knob-screen-20260925T093105Z/` |

## Limits

- Thermal substitution (RL-065): the protocol ≤55 °C package window is unreachable on this
  host while loaded (floor rose 58→69 °C over the night, heat-soaked chassis, fan at RPM
  floor). Arms started at the achievable floor, matched ±1 °C, counterbalanced, with per-block
  temperature and energy telemetry; a 80 °C heat-safety line applied. Champion attribution
  controls themselves ran 57–72 °C. Nothing here is protocol-1.3-cooled or public-grade.
- Eager-mode composition data is hypothesis-generation only (labelled) and cannot establish
  champion headroom.
- Acceptance-luck dispersion under recommended sampling is ±10 % per sample at these n; medians
  and counterbalanced order were used; the P1 gate was decided on pooled medians per the
  pre-declared rule.

## Sibling roots

- `p1-qwen27-hostprep-screen-20260925T082440Z` (primary) and the aborted attempts
  `p1-qwen27-hostprep-screen-20260925T062604Z` / `…T063556Z` (infrastructure aborts, kept).
- `p2-qwen27-eager-composition-20260925T085704Z`.
- `p4-qwen27-prefill-knob-screen-20260925T093105Z`.
- `P3-56096-DISPATCH-GATE-20260925.md`.
