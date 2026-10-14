---
title: "ADR-0011: Strata IQ3_XXS on one B70 as the Qwen3.8-Flash-Next route"
status: final
type: adr
date: 2026-10-09
---

---
id: b70-ref-adr-0011-flashnext-strata-one-card-route
title: "ADR-0011: Strata IQ3_XXS on one B70 as the Qwen3.8-Flash-Next route"
document_type: reference
status: canonical
version: "1.0"
effective_date: 2026-10-09
last_reviewed: 2026-10-09
authority: docs/methods/benchmark-protocol.md
scope: Serving-route selection for Qwen3.8-Flash-Next on the dual-B70 host
adr:
  number: 11
  decision_status: accepted
  truth_verdict: ESTABLISHED
  evidence_tier: E2
  validation_state: campaign_measured
  publication_state: PROVISIONAL_not_for_public_headline
  correctness_state: exact_answer_battery_3_of_3
  operational_state: primary_route
  supersession_state: current
---

# ADR-0011: Strata IQ3_XXS on one B70 as the Qwen3.8-Flash-Next route

## Context

The serving route for Qwen3.8-Flash-Next was llama.cpp `qwen4exp` AP-Q4_K_XL
dual-split across two B70s. RL-070 (2026-09-26) measured it at 21.6 tok/s
median engine tg (p512/g128) and 185-195 tok/s cold input and recorded the
verdict "capacity route, not a speed route as placed" — binding the placement,
not the model. RL-121 (2026-10-07) assessed Strata and the sybil tier build as
externally ahead on one card and queued the matched screen as RQ-14. RQ-14 ran
2026-10-09: matched cells on one B70 with a fresh n=5 confirmation (RL-125,
provisional; consolidated in
`results/2026/10/rq14-strata-matched-screen-20261009/`).

## Decision (operator-approved 2026-10-09)

1. **Strata v0.1.40.3 SYCL, Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS + native pack +
   mtp-rt, on ONE B70 is the primary serving route** for Qwen3.8-Flash-Next,
   replacing the llama.cpp AP-Q4_K_XL dual-split placement. Config of record:
   `serve-flashnext-iq3xxs.json` in the result root (`--stream-experts
   --expert-cache auto --spec 4 --spec-min-p 0.5 --kv int8 --prefill 4096
   --max-context 10240 --vram-reserve-mib 256`).
2. **Evidence (RL-125):** p512/g128 decode 48.03 tok/s median n=5 (2.22x the
   21.6 anchor), 8K prefill 989.11 tok/s median (5.1-5.3x the 185-195 anchor),
   exact-answer battery 3/3 (703 / Paris / B70-FLASH-OK). Both RQ-14 gates
   clear on one card instead of two. Artifact identity sealed
   (ARTIFACTS.sha256).
3. **Two-card placement is rejected on this engine:** the
   `--expert-cache-device1` path is host-staged (0 expert bytes by DMA) and
   slower on both axes (33.44 tok/s decode, 111.86 tok/s prefill).
4. **Open secondary gates are follow-up work, and a kill gate:** representative
   task quality, long/multi-turn cases and deterministic replay drift were not
   run at adoption (only 3 exact-answer cases). Any quality failure reopens
   this ADR and the route. Run them under the standing correctness rules.
5. **History preserved:** RL-070's dual-split anchors, recipe and calibration
   ("never redo") stay as records; nothing is superseded in the ledger.

## Operational notes

- Discard one same-shape warmup request on a cold engine (the first request
  exits 2 with `--spec needs the device residency table`).
- Battery and answer-bearing tasks need `gen >= 64` (the model thinks first).
- Restore as-found caps 180 W / 230 W at cleanup (the historical 150 W default
  is not the as-found state on this host).
- The 256 MiB VRAM reserve is below INTEL.md's 1,024 MiB recommendation for
  <=32K configs; revisit before longer-context cells.
