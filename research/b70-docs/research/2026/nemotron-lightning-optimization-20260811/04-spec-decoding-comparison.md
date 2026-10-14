---
title: "04 — Speculative Decoding Comparison"
type: research
date: 2026-08-11
status: final
---

# 04 — Speculative Decoding Comparison

## Methods tested

All measurements at 230W cap, Q4_K_M, q8_0/q4_1 KV, FA on, build f785fc9ea.
HTTP API with `timings.predicted_per_second`. Quick screens (n=3-5), not full campaigns.

| Method | Draft model | n-max | Median t/s | Acceptance | Mean draft len |
|--------|------------|-------|-----------|------------|----------------|
| No-spec | — | — | 37 | — | — |
| **MTP4** (best) | Native nextn head | 4 | **~52** | **95-100%** | ~4.0 |
| MTP4 (150W) | Native nextn head | 4 | ~47 | 95-96% | ~3.9 |
| MTP8 | Native nextn head | 8 | ~49 | 90-100% | ~4.2 |
| DFlash7 | 6-layer dense qwen3 (833M) | 7 | ~42 | 75-92% | ~3.5 |

## Why MTP4 wins

**MTP uses the target model's own next-token prediction head** (nextn_predict_layers=1).
The draft tokens ARE what the target would predict — acceptance is near-perfect (95-100%).
Mean draft length ≈ 4 (close to the max of 4), meaning almost all drafts are fully accepted.

The throughput gain is limited not by acceptance but by **verification cost**:
each verification step processes 5 tokens (1 target + 4 drafts) through all 53 layers.
The 5-token batch amortizes launch overhead but doesn't dramatically improve GEMV
efficiency (M=5 vs M=1).

## Why DFlash7 loses

DFlash uses a **separate 6-layer dense qwen3 model** (833M params) as the draft.
While the draft model itself is fast (no Mamba2 layers), its predictions don't
match the target model's distribution as well as MTP's native head:

- **Acceptance 75-92%** (vs MTP4's 95-100%)
- Lower acceptance = more rejected drafts = wasted verification compute
- The net throughput (42 t/s) is LOWER than MTP4 (52 t/s)

DFlash would win if it could draft many more tokens with high acceptance (e.g., 15
tokens at 90% acceptance). But the current single-block DFlash draft model can't
achieve that on this model.

## Why MTP8 doesn't help

Higher n-max (8 vs 4) increases the draft length but **decreases per-position acceptance**
(fundamental MTP property: each additional draft position is an autoregressive guess
off the previous guess, so errors compound). The net effect: more drafts proposed but
proportionally more rejected, giving similar or slightly lower throughput.

This matches the vLLM MTP findings from Run 24 (§15.8): N=4 is the throughput optimum.

## MTP acceptance per-position (from server log)

```
draft acceptance = 0.95604 (87 accepted / 91 generated), mean len = 3.90
```

Mean draft length 3.90 means ~3.9 tokens accepted per verification step (out of 4 proposed).
This is excellent — the native MTP head is well-trained for this model.

## Production recommendation

**MTP4 at 230W** is the optimal serving configuration:
- `--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.75`
- Expected: ~50-54 t/s decode, 95-100% acceptance
- Power scaling gives +10% (150W → 230W), modest but measurable
