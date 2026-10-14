---
status: living
title: "Serving"
type: model-dossier
date: 2026-09-15
---

---
schema_version: 1
model_id: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B
model_slug: nemotron-3.5-lightning-30b-a3b
document: serving
as_of: 2026-08-19
status: validated_internal
truth_verdict: CONFIRMED
evidence_tier: E2
engine: vllm-xpu
primary_evidence:
  - ../../../results/2026/08/nemotron-dflash-bf16-n7-n5-20260813T082203Z-1407994/n5-summary.json
  - ../../../results/2026/08/nemotron-dflash-bf16-n7-n5-20260813T082203Z-1407994/correctness-summary.json
  - ../../../results/2026/08/nemotron-dflash-ctx-capacity-20260813T103627Z-1501089/results.json
---

# Serving

## Mamba2/SSD fusion — measured and withdrawn (2026-09-21)

PROVISIONAL. The improvement roadmap's P2 proposed fusing the Mamba/SSD block chain, on the thesis
that ~736 dispatches/token of fragmented Mamba work was the loss class fusion removes (budgeted at
1.01 ms/token of a 5.36 ms token, ≈19 %). A dispatch attribution was run instead of a build: with the
route's XPU graph capture on, the trace (**2,876 kernels / 3,338 device events**) puts
**MoE cutlass gemm 41.7 % + `gemm_kernel` 31.3 % = 73 % of device time**, while the **entire fusable
Mamba/SSD family is 8 %** (medians 55–357 µs — large calls, not tiny fragmented launches). So the
fragmentation budget is already paid down by capture and fusion cannot deliver the declared gain.
**Phase withdrawn with the evidence attached** (`docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/P2-20260921.md`;
evidence `results/2026/09/b70-improvement-probes-20260920/p2-nemotron-entry-20260921T102137Z/`; ledger
RL-032). The 73 %-GEMM attribution is reassigned to the small-M/grouped-GEMM question (P5/P3).

Restricted re-measure from the same probe, on this route **and its deviations, not the recorded
baseline**: the recorded image tag `vllm-openai-xpu:1da0a954-det0123` is absent from this host and is
a local build tag with no registry source, so the local `vllm/vllm-openai-xpu@sha256:2c427ef4…` image
was used; GPU 1 and a 230 W cap instead of GPU 0 at 150 W. Result: **252.30 tok/s post-first /
178.55 tok/s out-throughput** at p2048/g128 (median of 3, 1,761 server tokens, TTFT 0.21 s, 5,220 MiB
free after load). The recorded 186.61 anchor sits near this run's *out-throughput*, not its
post-first; no cross-comparison to the recorded card is claimed.

## Current recipe class

The tested route uses vLLM XPU, GPTQ-INT4 G64 symmetric target weights, a BF16 DFlash draft, `method=dflash`, 7 speculative tokens, XPU graphs, async scheduling, float16 runtime dtype, prefix caching disabled, C1, and a configured 150 W cap. The isolated speed card used 16K max model length and an 8,192-token scheduler budget.

### Drafter structure (measured 2026-09-23 from source + checkpoint — corrects a Qwen-derived assumption)

`DFlashDraftModel` -> `DFlashQwen3ForCausalLM` (a 6-layer full-attention BF16 body: hidden 2688, vocab 131072) with **`has_lm_head: false`** — the checkpoint carries **no `lm_head` tensor at all**. At runtime `LLMBaseProposer._maybe_share_lm_head` binds the **TARGET's** head object into the draft (`llm_base_proposer.py`, MTP branch; `process_eagle_weight` returns early for non-EAGLE so `has_own_lm_head` is never set). So **DFlash shares the target's LM head — the opposite of the Qwen3.8 MTP case**, where the patch header records the draft holding its own head. Any drafter-quantization patch here must therefore build a **draft-private INT4 copy** and reroute `DFlashQwen3ForCausalLM.compute_logits` (the proposer's sole logits call site) without mutating the shared weight.

DFlash is **parallel drafting**: per n7 step there is ONE draft forward (8 query tokens) and ONE head application — not 7 passes, and not Qwen MTP4's 4 passes. Draft-side reads per step: body 962.0 MB (6 x `qkv_proj` [4608,2688] + `o_proj` [2688,4096] + `gate_up_proj` [12288,2688] + `down_proj` [2688,6144], plus `fc` [2688,16128] = 86.70 MB) + head 704.64 MB = **1.67 GB BF16/step**. A W4A16 g128 copy of body+fc (962.0 -> ~248 MB) and head (704.6 -> ~182 MB) would save ~1.24 GB/step = ~2.08 ms at the effective DRAM class. Against the 2026-09-21 restricted re-measure baseline (252.30 tok/s post-first p2048, 18.4-19.6 ms/step) that is **10.6-11.3 % of step time** — a predicted **+7-11 % at acceptance parity**, i.e. a screen-only effect below a +15 % promote gate. Harness and patches are ready (`scripts/tmp/p9-nemotron-draftint4-ab-20260923.sh` + `patch9_nemotron_draft_lmhead_int4.py` + `patch9_nemotron_draft_body_int4.py`, env-gated `B70_DRAFT_LMHEAD_INT4`/`B70_DRAFT_MTP_INT4`, fail-closed anchors) and un-run.

## Canonical n=5 speed card

| Cell | Metric | Median | Range | Acceptance |
|---|---|---:|---:|---:|
| p2048/g1 | cold input | 6,455.6 tok/s | 6,262.9–7,353.3 | — |
| p8192/g1 | cold input | 7,160.1 tok/s | 7,117.3–7,226.3 | — |
| p512/g128 | client post-first | 194.61 tok/s | 140.20–220.01 | 45.1% |
| p2048/g128 | client post-first | 186.61 tok/s | 174.60–201.83 | 56.5% |
| p8192/g128 | client post-first | 157.92 tok/s | 143.50–170.25 | 53.0% |

Window acceptance was 1,830/3,521 = 52.0%. The matched-except-speculation p8192/g128 comparison was 157.92 versus 87.25 tok/s, or 1.81× **on that cell only**.

Cold-input values are prompt tokens divided by client TTFT, not isolated engine prefill. A later 16K scheduler-budget n=3 spike did not survive n=5; retain the 8,192 budget. A separate capacity run completed 119,904 tokens but did not test 128K and does not replace this speed card.
