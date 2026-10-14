---
title: Claims audit — LocalMaxxing submission payloads (draft-side INT4, 2026-09-22)
type: worklog
date: 2026-09-22
status: living
---

Independent claims audit of the four exact payload files in
`scripts/tmp/lmx-submit-20260922/` against raw evidence
`results/2026/09/b70-improvement-probes-20260920/p4-champion-ab-p8192-n5-…-20260922T213957Z/`
and `…p512-n5-…-20260922T214938Z/`, the claims standard
(`.agents/skills/b70-benchmark-claims-standard/SKILL.md`), and
`docs/methods/benchmark-protocol.md`. Every number below was recomputed from
`client.log` JSON lines, `power.json`, and `metrics-before/after.txt` — not
taken from `comparison.json` or the payloads. No GPU work was run; no payload
file was edited.

## Verdict per payload

| Payload | Verdict | Blocking reasons |
|---|---|---|
| `c1-p8192-g128-draftint4-patched.json` | **PUBLISHABLE-WITH-FIXES** | F2 (throughput-stability claim contradicted by own samples), F4 (tokSPrefill naming), F5 (no evidence link), F6 (missing `-p 8000:8000`), F7 ("Reproduced" ambiguity) |
| `c1-p8192-g128-baseline.json` | **PUBLISHABLE-WITH-FIXES** | **F1 (acceptance band 0.53–0.58 excludes its own reported 0.4873)**, F4, F5, F6, F7 |
| `c1-p512-g128-draftint4-patched.json` | **PUBLISHABLE-WITH-FIXES** | F3 (**+31.5 % tokens/J does not reproduce; +31.4 % does**), F2, F4, F5, F6, F7 |
| `c1-p512-g128-baseline.json` | **PUBLISHABLE-WITH-FIXES** | F4, F5, F6, F7 (cleanest of the four; no number failures) |

No BLOCKER findings. Every headline number (72.55, 48.81, 73.43, 52.61 tok/s;
TTFT medians; prompt-token counts; absolute spec counters; acceptance rates;
configured cap; multipliers 1.49x / 1.40x) reproduces from raw evidence.

## Recomputed vs payload — full number table

Statistic: median of the n=5 post-warmup samples in each `client.log`.
Average card W = `energy1_input` delta / elapsed / 1e6 (xe hwmon6, cap 230 W,
bracketed measured window). J/token and tokens/J use 640 output tokens
(5 requests × 128) over that same window (window includes prefill — disclosed
in all four notes).

### c1-p8192-g128-draftint4-patched.json (arm v5_draftpair, 213957Z)

| Quantity | Recomputed from raw | Payload | Status |
|---|---:|---:|---|
| tokSOut (median post-first) | 72.55 (samples 64.25–91.17) | 72.55 | MATCH |
| ttftMs (median) | 4791.6 | 4791.6 | MATCH |
| promptTokens (server, all 5) | 8194 | 8194 | MATCH |
| tokSPrefill = 8194/4.7916 | 1710.08 | 1710.1 | MATCH |
| out_tokens / token_count_ok | 128×5 / all true | 10/10 length | MATCH |
| specDraftTokens | 936−152 = 784 | 784 | MATCH |
| specAcceptedTokens | 535−90 = 445 | 445 | MATCH |
| specAcceptanceRate | 445/784 = 0.56760 | 0.5676 | MATCH |
| measured avg W | 229.4958 | 229.49 | MATCH value, wrong rounding (229.50) — NIT F8 |
| J/token | 15.2952 | 15.29 | wrong rounding (15.30) — NIT F8 |
| tokens/J | 0.065380 | 0.0654 | MATCH |
| multiplier vs baseline | 1.4864 (+48.64 %) | 1.49x (+48.6 %) | MATCH |
| tokens/J delta | +9.6228 % | +9.6 % | MATCH |

### c1-p8192-g128-baseline.json (arm v5_baseline, 213957Z)

| Quantity | Recomputed | Payload | Status |
|---|---:|---:|---|
| tokSOut | 48.81 (samples 40.44–57.23) | 48.81 | MATCH |
| ttftMs | 4738.1 | 4738.1 | MATCH |
| promptTokens | 8190 | 8190 | MATCH |
| tokSPrefill = 8190/4.7381 | 1728.54 | 1728.5 | MATCH |
| specDraftTokens | 1016−152 = 864 | 864 | MATCH |
| specAcceptedTokens | 514−93 = 421 | 421 | MATCH |
| specAcceptanceRate | 421/864 = 0.48727 | 0.4873 | MATCH |
| measured avg W | 228.7352 | 228.74 | MATCH |
| J/token | 16.7670 | 16.77 | MATCH |
| tokens/J | 0.059641 | 0.0596 | MATCH |
| acceptance band claim | 0.4873 this run vs 0.5676 window / 0.5369 cumulative in the only earlier p8192 n5 baseline (114444Z) | "band across runs 0.53-0.58" | **FAILS — own value outside stated band; F1** |

### c1-p512-g128-draftint4-patched.json (arm v5_draftpair, 214938Z)

| Quantity | Recomputed | Payload | Status |
|---|---:|---:|---|
| tokSOut | 73.43 (samples 73.38–88.67) | 73.43 | MATCH |
| ttftMs | 352.0 | 352.0 | MATCH |
| promptTokens | 517 | 517 | MATCH |
| tokSPrefill = 517/0.352 | 1468.75 | 1468.8 | MATCH |
| specDraftTokens | 972−180 = 792 | 792 | MATCH |
| specAcceptedTokens | 525−83 = 442 | 442 | MATCH |
| specAcceptanceRate | 442/792 = 0.55808 | 0.5581 | MATCH |
| measured avg W | 228.5969 | 228.59 | wrong rounding (228.60) — NIT F8 |
| J/token | 3.8965 | 3.89 | wrong rounding (3.90) — NIT F8 |
| tokens/J | 0.256640 | 0.2566 | MATCH |
| multiplier vs baseline | 1.3957 (+39.57 %) | 1.40x (+39.6 %) | MATCH |
| tokens/J delta | **+31.4459 %** | **+31.5 %** | **MISMATCH — correct 1-dp value +31.4 %; F3** |

### c1-p512-g128-baseline.json (arm v5_baseline, 214938Z)

| Quantity | Recomputed | Payload | Status |
|---|---:|---:|---|
| tokSOut | 52.61 (samples 52.56–55.39) | 52.61 | MATCH |
| ttftMs | 383.2 | 383.2 | MATCH |
| promptTokens | 516 | 516 | MATCH |
| tokSPrefill = 516/0.3832 | 1346.56 | 1346.6 | MATCH |
| specDraftTokens | 980−176 = 804 | 804 | MATCH |
| specAcceptedTokens | 521−84 = 437 | 437 | MATCH |
| specAcceptanceRate | 437/804 = 0.54353 | 0.5435 | MATCH |
| measured avg W | 224.5325 | 224.53 | MATCH |
| J/token | 5.1218 | 5.12 | MATCH |
| tokens/J | 0.195244 | 0.1952 | MATCH |

### Cross-referenced claims (all verified against registered evidence)

| Claim in payloads | Evidence | Status |
|---|---|---|
| "Reproduced +39.6 % (p512)" / "+48.6 % (p8192)" | the two audited runs themselves | MATCH |
| "earlier +37.2/+44.4/+46.0 %" | +37.2 = p8192 n5 093453Z; +44.4 = p8192 n5 114444Z; +46.0 = p512 **n=3 screen** 094524Z | MATCH (label the n=3 — F7) |
| "both-arms-patched control reproduced 80.61/78.27 tok/s" | 121436Z v5_draftpair_A/B medians 80.61 / 78.27 | MATCH |
| greedy byte-identical, 3 prompts × 2 repeats | OUTCOMES W2; evidence 122542Z (DET=1 harness, sha256) | MATCH, scope correctly labeled |
| 30 seeded draws × 6 prompts, identical modal class | OUTCOMES O2; P4 §17 (modal 6/6, overlap 0.56–1.00) | MATCH; no overclaim ("distribution-preserving" absent) |
| capture off inert ≤1.5 %; capture alone 1.68× | LEVERS D3, OUTCOMES W3 (25.40 → 42.76) | MATCH |
| issue #54785 wording | `research/2026/upstream-contribution-anchoring-20260922.md` (OPEN, 2026-09-01, temp=0 wrong logits, k≤3 clean) | MATCH |
| 2.54 GB F16 LM head | P4 §2 (2 543 MB); anchoring note | MATCH |
| prefix-cache hit delta zero | counters 0→0 before/after in all four arms; server.log `enable_prefix_caching: False` | MATCH |
| version 0.27.2rc1.dev77+gac7509e2b (.xpu observed) | server.log banner | MATCH |
| host: Ryzen 7 5700X3D, 60 GB RAM, Ubuntu 26.04.1 | live host (`free -g`, `/etc/os-release`) | MATCH |
| warmup discarded, one per arm | warmup.log, 1 same-shape entry per arm | MATCH |

### Hygiene and schema checks

- Notes lengths: 1893 / 1779 / 1845 / 1758 chars — all ≤ 2000. PASS.
- No hostnames, LAN IPs, absolute local paths, model-dir paths, API keys, or
  bearer tokens in any payload field a public reader sees. `commandSnippet`
  uses `<model_dir>` placeholder and relative patch names. PASS.
- No claim of LocalMaxxing verification anywhere; all four end
  "PROVISIONAL self-report." PASS.
- Local unpublished overlay patches declared; public pullable digest
  distinguished. Observed in-container version recorded. PASS.
- Enabler disclosure (capture REQUIRED; capture-only 1.68×) and issue #54785
  present in all four; claim scoped to capture-on configs. PASS.
- C1 never mixed with Cn; no aggregate-vs-single-stream comparison. PASS.
- Configured cap (230 W) separated from measured draw everywhere; coverage
  limitation ("bracketing the measured window only") stated; J/token marked
  "INCLUDING prefill". PASS.

## Findings

### F1 — MAJOR — baseline p8192 acceptance band excludes its own value

`c1-p8192-g128-baseline.json` notes state: "Acceptance 0.4873 … band across
runs 0.53-0.58". The reported value 0.4873 is outside the stated band, and the
band itself was assembled from the earlier run's two different metrics (0.5369
cumulative, 0.5676 window — 114444Z), not from a like-for-like series. This
matters beyond bookkeeping: this run's baseline arm sits at the low tail of
both acceptance (0.4873 vs 0.5676) and throughput (48.81 vs 52.99/55.61/55.68
in earlier baseline runs), which inflates this run's 1.49x relative to the
earlier +37.2/+44.4 % runs. A platform reviewer who cross-reads the two
records will catch the inconsistency.

Exact replacement (fits: notes go 1779 → ~1880 chars):

> "Acceptance 0.4873 from measured-window counters (864 draft / 421 accepted); run-variable: 0.4873 this run vs 0.53-0.58 in the earlier baseline run — this run's baseline arm is at the low tail of acceptance and throughput (earlier baseline medians 53.0-55.6 tok/s same cell), so the 1.49x here is the high end of the reproduced +37.2/+44.4/+48.6 % range."

### F2 — MAJOR — "while throughput is not [run-variable]" is contradicted by the payloads' own evidence (both patched payloads)

p8192-patched notes: "run-variable 0.55-0.69 across runs while throughput is
not." Two defects: (a) the patched measured-window acceptance in registered
evidence reaches 0.7027 (114444Z), above the stated band top 0.69; (b) patched
throughput IS run-variable: medians 72.55 / 72.68 / 80.32 across runs
(+10.7 %), and this run's own five samples span 64.25–91.17 (±20 %). The
claims standard requires n, median, maximum, and dispersion; dispersion is
absent. p512-patched repeats the same "while throughput is not" clause.

Exact replacements:

- p8192-patched: "Acceptance 0.5676 from measured-window counters (784 draft / 445 accepted); run-variable 0.55-0.70 across patched runs. Throughput samples this run 64.3-91.2 (median 72.55); patched medians 72.6-80.3 across runs."
- p512-patched: "Acceptance 0.5581 from measured-window counters (792 draft / 442 accepted); run-variable across runs (0.55-0.70 observed on this config class). Throughput samples this run 73.4-88.7 (median 73.43)."

### F3 — MINOR — "+31.5 %" tokens/J does not reproduce (p512 patched)

Recomputed from the raw energy delta: +31.4459 % → **+31.4 %** at one decimal;
even the payloads' own rounded tokens/J values give 0.2566/0.1952 = +31.4 %.
The claimed +31.5 % is a mis-rounding (the +9.6 % p8192 figure is correct:
+9.6228 %).

Exact replacement: "tokens/J 0.2566 vs 0.1952 baseline = +31.4 %" (and the
mirrored baseline-payload phrase "= +31.5 % for the patched arm" → "= +31.4 %
for the patched arm").

### F4 — MINOR — tokSPrefill not named as cold input rate in the patched payloads

The gate requires tokSPrefill named as *cold input rate = actual endpoint
prompt tokens / client TTFT*, derivation stated. Both baseline payloads carry
the full naming; both patched payloads say only "tokSPrefill derived =
promptTokens/TTFT". Fix: use the baseline phrasing verbatim in both patched
payloads: "tokSPrefill derived = promptTokens/TTFT (cold input rate = actual
endpoint prompt tokens / client TTFT)". Costs ~55 chars, still within limit
(p8192 patched → ~1948 with F5's shorter pointer; see below).

### F5 — MINOR — no evidence pointer; comparison gate requires numerator and denominator linked to raw evidence

All four payloads cite cross-arm numbers (and patched payloads cite historical
runs) with no link to the evidence. The platform convention established in the
prepared submission worklog (§2.1 notes) is a relative B70-DOCS repo path —
that is acceptable and should be restored, shortened:

> "Evidence: B70-DOCS results/2026/09/b70-improvement-probes-20260920/p4-champion-ab-p8192-n5-*213957Z (comparison.json, client.log, power.json, metrics-before/after)."

(p512 payloads: `…p512-n5-*214938Z…`.) Because notes are tight, pair F4+F5
with one trim: drop "; every emitted token target-verified" from the baseline
correctness line (it is a mechanism property of MTP, not added evidence) and
shorten "cold_model_cold_prefix (prefix caching disabled, zero hit delta)" to
"cold_model_cold_prefix (caching disabled, zero hit delta)". Both patched
payloads then fit ≤ 2000 with F2+F4+F5 applied (recheck after edits).

### F6 — MINOR — commandSnippet lost the port publish flag

The prepared draft had `-p 8000:8000`; all four final snippets dropped it, so
the published recipe's server is unreachable from the host as written.
Re-insert `-p 8000:8000` after `--name b70-q38` in all four.

### F7 — MINOR — "Reproduced …" wording can read as independent reproduction; n=3 screen unlabeled

"Reproduced +39.6 % (p512) and earlier +37.2/+44.4/+46.0 % …" — all these are
same-host, same-stack re-runs; "Reproduced" unqualified invites an
independent-reproduction reading (the E-tier the run does not have), and
+46.0 % comes from an n=3 tuning screen that the standard bars from public
representative use (citing its ratio as history is acceptable only labeled).

Exact replacement (both patched payloads):

> "Same-host re-runs: +39.6 % (p512 cell) / +48.6 % (p8192 cell); earlier +37.2/+44.4 % (n=5) and +46.0 % (n=3 screen); both-arms-patched control 80.61/78.27 tok/s."

### F8 — NIT — truncation instead of rounding on four derived values

229.49 (→229.50), 15.29 (→15.30), 228.59 (→228.60), 3.89 (→3.90). Immaterial
(<0.01 in each), but pick one convention — round half-up, or carry one more
decimal (229.496, 15.295, 228.597, 3.897).

### F9 — NIT — p512 patched TTFT sentence is a double negative

"TTFT median 352.0 ms - NOT unchanged vs baseline (383.2 ms), it improves
here." Honest but confusing. Replace: "TTFT median 352.0 ms (baseline 383.2 ms
— TTFT also improves in this cell)."

### F10 — NIT (guard for accompanying prose, not a payload defect) — keep the catalog row and these payloads distinct

The registered cookbook row `qwen38-27-draft-int4-p512-g128-c1` (106.7 tok/s,
**cache enabled**, calibrated Pi prompt, context 131072) and the p512-patched
payload (73.43 tok/s, **prefix caching disabled**, synthetic entropy-nonce
prompt, 517 tokens, context 16384) are different cells and correctly never
conflated inside the payloads. Any summary text, thread, or README that cites
both must name the cache-state and prompt difference; never present 106.7 and
73.43 as the same configuration. Catalog row verified at
`~/intel-arc-pro-b70-inference-cookbook/data/benchmarks.v1.json`.

### Checks that passed cleanly (no finding)

Comparison classification `matched_with_tokenizer_delta` is **correct** for
both multipliers: same engine/build/image+overlay, same target quant and KV,
same cache state, same configured cap, same sampling and speculation policy,
same statistic (median, n=5) and same run; the arms differ only in the two
draft-INT4 env gates (confirmed by manifests `arm2_env`) plus the disclosed
tokenizer delta (8190 vs 8194 = 0.05 %; 516 vs 517 = 0.19 %), which is exactly
what that classification is for. Absolute spec counters match the platform's
MTP requirement. Correctness wording stays inside its measured scope. No
privacy/secret leakage. Notes limits respected.

## Final verdict

**All four payloads are PUBLISHABLE-WITH-FIXES — no payload number failed to
reproduce from raw evidence except the p512 "+31.5 %" tokens/J delta (correct
value +31.4 %), and the only must-fix honesty defects are the baseline
acceptance band that excludes its own value (F1) and the "throughput is not
run-variable" clause (F2); apply F1–F7, then submit as provisional
self-reported records.**
