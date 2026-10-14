---
status: living
title: "Qwen3.8-27B"
type: model-dossier
date: 2026-09-15
---

---
schema_version: 1
model_id: Qwen/Qwen3.8-27B
model_slug: qwen3.8-27b
document: overview
as_of: 2026-08-19
status: validated_internal
truth_verdict: PARTIAL
evidence_tier: E2
publication_gate: PROVISIONAL_NOT_FOR_PUBLIC_HEADLINE
engines: [vllm-xpu, llama.cpp-sycl]
primary_evidence:
  - ../../../research/2026/qwen38-27b-day0-adoption-20260814.md
  - ../../../results/2026/08/vllm-qwen38-4mode-230w-20260815T170840Z-581426/summary.json
  - ../../../archive/legacy-docs-20260915/benchmark-history.md
---

# Qwen3.8-27B

**Route state — 2026-09-27 (ADR-0010): the primary serving route is now EXL3** (exl3xpu engine, turboderp 4bpw artifact + native MTP3 + fp8 KV) — the only arm reaching native 262,144 context on one card, clean on the exact-answer battery, and the most cap-sensitive arm (+33–42 % sustained at 230 W). The prior champion `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` was removed: the W4A16/GPTQ-family symmetric-int4 path is **closed for this model** — three independent exports (GPTQModel, AutoRound→compressed-tensors, AutoRound→auto_gptq with a code-heavy local corpus) all reproduce the "30" code_execution defect, are ~4.9× slower than the removed champion, and cannot reach 256K on one card. OpenVINO int4-ov is the alternate route (clean, ~28.5 tok/s @150 W roofline). Full evidence: `results/2026/09/autoround-vs-exl3-20260926/RESULTS.md`, ledger RL-084…RL-088, engine dossier `engines/exl3xpu/README.md`.

The GPTQ-era context below is retained for provenance — it describes the removed route. The retained graph-captured MTP4 recipe uses the validated **draft-only INT4 body/head pair**. The base artifact retains the stock MTP weights; the runtime overlay uses separate quantized draft tensors and leaves the **target verifier's full-vocabulary LM head unquantized**. This is not a target-model INT4 conversion or a retrained draft. See the [patch provenance](../../../research/2026/qwen38-draft-int4-patch-provenance-20260922.md). DFlash2 is a separate research overlay and is not the serving recommendation.

**Research decision — 2026-09-24:** D11 FastMTP retraining is closed and rejected; retain stock MTP weights, MTP4, greedy drafting, FP8 KV, graph capture, and the draft-side INT4 pair. The [revised Qwen27 plan](../../../research/2026/exl3-analysis-and-next-inference-levers-20260924.md) selects a bounded current-stack attribution probe before any new kernel investment. Rotated INT3 is not a ready serving path, the K216 IQ3_XXS operator win does not dispatch on this dense GPTQ target, and IPC does not remove the documented no-P2P transport constraint. This research decision does not authorize a benchmark, conversion, or service change.

**Route state — 2026-09-25:** the attribution decider executed ([night report](../../../docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/NIGHT-REPORT-20260925.md), RL-061…RL-065): zero promotions, recipe unchanged, and **single-card decode measured at its byte-budget floor** — single-card kernel/host hunting is closed. The live research lines are the dual-card copy-collectives path (P2P capability verified at ~11–12 GB/s, gated behind collective correctness and ADR-0008; honest band ~1.4–1.6×), a record-closing unitrace capture, and the operator thermal-window decision. No recipe, benchmark, or service change is authorized by this note.

**Framing correction (2026-10-07):** the 2026-09-24/25 statements above bound the
**removed GPTQ/W4A16 route** and must not be read as closing the EXL3 route's kernel
question. On EXL3 the M=4 verify-pass program is open and ranked first
(`../../../research/PROGRAM.md` RQ-01…RQ-04; evidence RL-098…RL-120), and the
"byte floor" wording was denominator-confounded for the retired route too
(`../../../research/2026/b70-adversarial-closure-review-20260930.md`).

Performance results must retain their sampling labels. Greedy diagnostic, Qwen-recommended non-thinking, and Qwen-recommended thinking are separate regimes; their results are not interchangeable.

Dual-card default is one independent engine per GPU, not vLLM TP2. An external TP2 recipe for a different AutoRound checkpoint is captured in [`research/2026/humble-b70-llm-dual-tp2-20260824.md`](../../../research/2026/humble-b70-llm-dual-tp2-20260824.md) and is not this host’s serving route ([ADR-0008](../../../decisions/adr-0008-dual-b70-independent-default.md)).

See [adoption](adoption.md), [artifacts](artifacts.md), [serving](serving.md), [quality](quality.md), and [timeline](timeline.md).
