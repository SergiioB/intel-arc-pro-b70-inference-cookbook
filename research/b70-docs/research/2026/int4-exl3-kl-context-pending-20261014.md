---
title: "Qwen3.8-27B INT4 versus EXL3: matched KL comparison pending"
type: research
date: 2026-10-14
status: provisional
---

# Matched INT4 KL is pending

The user requested INT4 versus EXL3 plain 4.0bpw and SC5 5.0bpw, and explicitly prefers the tested 196K SC5 inference setting despite lower VRAM reserve.

| Artifact | Wikitext KL | Held-out chat KL | Scope |
|---|---:|---:|---|
| EXL3 plain 4.00bpw @113cf7ab | 0.01391 | 0.00876 | RL-179, single-run full-64-layer reference-weight screen |
| EXL3 SC_5.00bpw_H6 @dfe2f4fd | 0.00582 | 0.00141 | Same screen and reference |
| GPTQ/AutoRound W4A16 | unavailable | unavailable | No matched full-depth record; removed or absent artifact |
| OpenVINO INT4 | unavailable | unavailable | Local IR exists; matched reference-weight adapter needed |
| GGUF UD-Q4_K_M | unavailable | unavailable | Local artifact exists; matched reference-weight adapter needed |

Evaluation: wikitext-2-raw-v1 test slice, 1,664 scored positions; turboderp held-out chat trace, 1,531 response positions. Mean KL(P_BF16 || P_quant), reference-decoded weights, native heads. These are not served-engine KL figures.

Partial-layer scalar INT4 results exist, but cannot be compared numerically to the full-depth EXL3 rows. No INT4 value is inferred from them. W4A16's historical arithmetic defect is a separate correctness finding, not a KL measurement.

## Context and operator preference

- SC5: 196,438 occupied tokens passed exact retrieval and multi-turn checks at memory utilization 0.94, with 1,069 MiB free VRAM. The operator selects the 196,608-token configured inference profile. Capacity observation, single-request scope, lower reserve.
- SC5 alternate: 131,072 configured; 130,872 occupied checked at utilization 0.90, with 2,327 MiB free.
- Plain EXL3: separate 262,144-token route with retained 261,920-token prompt completion; do not transfer this result to SC5.
- Engine cache budget and configured context are not completed-request evidence. See capacity report (`../../results/2026/10/ctx-capacity-sc5-20261014/REPORT.md`).

## Pending measurement

The current launcher GGUF UD-Q4_K_M is the selected available candidate, not a substitute for removed GPTQ weights. Its reference-weight adapter is implemented and CPU-verified: all 866 tensors are accounted for, with the extra MTP layer explicitly outside the 64-layer target; dequantization matched the retained libggml C++ implementation bit-for-bit on real blocks across all used quantization types. Frozen evaluation states were regenerated deterministically and passed their manifest checks. See `scripts/tmp/gguf_q4km_arm_20261014/PLAN.md` and its decoder/map/coverage reports.

The matched full-depth BF16/plain-EXL3/GGUF screen produced no quality result: harness setup failures were retained, then an unrelated inference container occupied GPU1. It was not stopped or replaced. Run status remains **pending**, with exact failures and ownership scope in `scripts/tmp/gguf_q4km_arm_20261014/RUN-ATTEMPTS-20261014.md`. OpenVINO IR requires its own validated adapter. Do not reopen removed quality-failed W4A16 exports merely to fill the table.

Flash-Next EXL3 inference and its full-depth KL are separate ongoing work; no Flash-Next KL value is included here.
