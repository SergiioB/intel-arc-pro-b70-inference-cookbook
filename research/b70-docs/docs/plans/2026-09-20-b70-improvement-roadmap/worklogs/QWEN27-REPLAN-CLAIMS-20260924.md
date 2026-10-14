---
title: "QWEN27 replan — D11 and campaign claims audit (relay record)"
type: worklog
date: 2026-09-24
status: final
provenance: "Composed by the coordinating session from the claims auditor's delivered interim findings after its provider quota exhausted; the auditor did not complete its own file. Raw numbers below were verified by the auditor against raw JSON before the failure."
---

# QWEN27 replan — D11 and campaign claims audit (relay record, 2026-09-24)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** This is a relay record, not the auditor's own completed
report. The independent claims-auditor agent verified the decisive items below against raw evidence
and delivered them mid-session, then failed with a provider usage-limit error before writing its
file (`usage_limit_reached`, model `gpt-6-luna` via provider `codex`). Per the substitute-and-record
rule, the coordinator recorded the delivered findings here, marked as relayed. Nothing in this file
is new measurement; raw result roots are untouched.

## 1. D11 — decisive verified findings

**Actual evidence root.** The FastMTP-1500 stitched probe lives at
`results/2026/09/b70-improvement-probes-20260920/d11-stitched-fastmtp1500-probe-20260924T172339Z/`.
The `T182834Z` suffix cited by RL-056, OUTCOMES L15 and LEVERS D11 did not exist on disk (stale
timestamp). The same-day matched baseline is
`results/2026/09/b70-improvement-probes-20260920/d11-baseline-decay-probe-20260924T142540Z/`.
Shared-authority corrections have since been applied by the coordinator (OUTCOMES/LEVERS rows fixed;
RL-056 itself left untouched per append-only rules, corrected by successor RL-058).

**Raw counter deltas (auditor-verified against raw JSON).**

| Arm | Draft rounds | Drafted tokens | Accepted tokens | α = accepted/drafted | Accepted per round |
|---|---:|---:|---:|---:|---:|
| Stock (same-day baseline) | 202 | 808 | 438 | 0.5421 | 2.1683 |
| FastMTP-1500 retrained | 214 | 856 | 430 | 0.5023 | 2.0093 |

Δα = **−0.0398**. Positional accepted counts sum exactly to the aggregate on both arms
(stock 198+197+25+18 = 438; retrained 178+147+60+45 = 430), confirming a common counter window per arm.

**Positional percentages are marginal survival, not conditional accuracy.** The recorded
98.0/97.5/12.4/8.9 % and 83.2/68.7/28.0/21.0 % figures divide each position's accepted count by **all
draft rounds** — i.e. they are the marginal contribution of each position to total acceptance
(P(prefix reaches position i+1)). They are not "accuracy at position i given arrival." Conditional
rates derived on prior-position survivors are: stock ≈ 98.0 / 99.5 / 12.7 / 72.0 % vs retrained
≈ 83.2 / 82.6 / 40.8 / 75.0 %. Do not describe the retrained head as lifting position 3 from 8.9 %
to 21.0 % conditionally; the tail lift exists but is smaller in conditional terms, and the early
collapse dominates the sum.

**Throughput.** Five client post-first samples per arm, sequential same-day servers: medians
**73.68 → 71.83 tok/s = −2.51 %**. The older comparator 77.26 tok/s / α 0.5537 is the **D12 finalist
greedy-draft control** (`d12-finalist-20260923T045006Z`), not "Run 48"; the −7.03 % / −0.0514 deltas
against it are historical cross-session context, not the matched D11 effect.

**Verdict on D11.** CLOSED / REJECTED stands: aggregate acceptance and throughput both fell; the
gate (Δα ≥ +0.08, tps ≥ +3 %) failed decisively. The observed failure supports closing this
*program*; it is not a universal theorem about post-training all pretrained draft heads, and the
specific "trained on trillions of tokens" provenance claim for this native head was not
primary-source verified. Root-cause wording in shared docs has been softened accordingly.

## 2. Completeness limits on the D11 comparison (explicit)

- Same-day but **sequential separate servers**; n=5 client samples per arm; no counterbalancing
  order evidence relayed. This is a diagnostic, not a publication-grade matched A/B.
- Raw **power evidence for the baseline arm was not located** by the auditor; the "measured card
  draw 226.74 W" figure in RL-056 applies to the retrained-arm window. Do not quote iso-power for
  this pair.
- Cache counters, cap history, and thermal alignment between the two arms were **not fully
  reconciled** before the auditor failed. None of these limits changes the adoption decision
  (both gates failed with margin), but they bar promoting any D11 number publicly.

## 3. Wins scoreboard — scope checks (auditor findings, applied to shared docs)

- Draft-side INT4 pair: retained Qwen27 C1 win; cells and qualifiers already claims-audited at
  publication time (RL-036/RL-037). It is **already in the recipe**, not future headroom.
- Graph capture 1.68×: enabler already in the stack; not incremental.
- IQ3_XXS 3.60×: isolated MoE operator on llama.cpp SYCL; no dense GPTQ Qwen27 dispatch; "192→166 µs"
  is 13.3 % lower operator latency / 1.154× operator throughput, never "+13 % decode."
- DPAS s4×s4 2.00×: instruction-throughput experiment, not a serving result.
- Concurrency 2.79×: aggregate Cn, never comparable to C1.
- D12/D14 closures: D12's screen PROMOTE was refuted by its own fresh finalist (RL-048); D14 changed
  KV policy globally, so it cannot isolate a draft-only precision effect — both stay closed.

## 4. Status

This relay record plus the coordinator's corrections (research note rewrite, OUTCOMES/LEVERS D11 rows,
model-status open_gate, RL-058) complete the claims-gate intent of the failed agent run. If the
claims-auditor agent is re-run later with quota available, its independent report supersedes this
relay in authority; until then, treat the D11 numbers above as auditor-verified and everything in §2
as explicitly unresolved.
