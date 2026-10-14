---
doc_id: DOC-CURRENT-RECOMMENDED-CONFIGS
title: "Current B70 Recommended Configurations"
type: operation
status: living
date: 2026-08-18
updated: 2026-10-09
last_verified: 2026-10-09
canonical_for:
  - current-research-candidates
---

# Current B70 Recommended Configurations

This view separates **currently available serving routes**, **provisional research candidates**, and **historical or unavailable artifacts**. It is a navigation view, not a numeric authority. The route authority is `registry/model-status.json` (rendered as [models/STATUS.md](../../models/STATUS.md)); if this page and that file disagree, the registry wins. Format and engine choice per workload regime: [B70 format and lever guide](../reference/b70-format-and-lever-guide.md).

Live artifact state is volatile. Recheck [`docs/operations/current-host-state.md`](../operations/current-host-state.md), `df -h`, the exact artifact path, and the empty-GPU gates immediately before launch.

## Current serving routes

| Model and route | Engine | Speculation / KV | Recorded rate (cell) | Evidence and open gates |
|---|---|---|---|---|
| **Qwen3.8-27B** — turboderp EXL3 4.00 bpw @113cf7ab | exl3xpu (vLLM plugin), one card | native MTP3; fp8 KV (only dtype); 262,144 native context | 63.25 tok/s usage-token rate, p512/g128 n=5 @230 W | [ADR-0010](../../decisions/adr-0010-exl3-primary-route-qwen38.md); dossier [qwen3.8-27b](../../models/qwen/qwen3.8-27b/README.md). Prefix cache measured 5.4× on this route; declare cache state per cell. |
| **Qwen3.8-Flash-Next** — GSQ-RCO IQ3_XXS GGUF + native pack | Strata v0.1.40.3 SYCL, one card, `--stream-experts --expert-cache auto` | MTP `--spec 4 --spec-min-p 0.5`; int8 KV | 48.03 tok/s engine decode-wall p512/g128 n=5; 8K prefill 989.11 tok/s | [ADR-0011](../../decisions/adr-0011-flashnext-strata-one-card-route.md), RL-125. Representative-task, long/multi-turn and replay-drift quality gates are still open (RQ-17). |
| **Nemotron 3.5 Lightning 30B-A3B** — GPTQ-INT4 symmetric G64 + BF16 DFlash draft | vLLM XPU | DFlash depth 7; engine-native KV; 120,000 configured | 186.61 tok/s client post-first, p2048/g128 n=5 @150 W (2026-08-13 harness; re-measure owed, RQ-11) | [model dossier](../../models/nemotron/nemotron-3.5-lightning-30b-a3b/README.md), [ADR-0002](../../decisions/adr-0002-nemotron-dflash-serving-default.md). 7,160 prompt tok/s at p8192/g1 is cold input rate, not isolated prefill. |
| **Muse Glimmer 30B** — Q4_K_XL + mmproj | llama.cpp SYCL (`GGML_SYCL_F16=ON`) | DFlash n2; fleet KV q8_0 K / q4_1 V | 26.8 tok/s p512/g128 @128K n=5 | [model dossier](../../models/muse/muse-glimmer-30b/README.md). Local artifacts were deleted earlier; verify presence before launch. Open: projector and reasoning budget. |

## Provisional research candidates

| Model and route | State | Required next gates |
|---|---|---|
| **Qwen3.8-27B-FP8** — native block-FP8 W8A16 | Diagnostic route, 48.08 tok/s p1024/g128 greedy | Small-M kernel and DFlash2 selector/conv; below the EXL3 route at every recorded cell. See [serving notes](../../models/qwen/qwen3.8-27b/serving.md). |
| **Muse Glimmer vLLM compressed-tensors overlay** | Experimental exact-artifact support | Requires the local overlay/reacquired artifact, exact text dispatch, vision/XPU correctness, and public reproducibility. It is not a generic compressed-tensors support claim. See the [model dossier](../../models/muse/muse-glimmer-30b/README.md). |
| **Qwen3.6-35B-A3B preserved-MTP GPTQ** | Historical/research route | Recheck artifact availability, image/patch generation, exact 128K boundary behavior, mixed-load limitations, and current task need before relaunch. See the [model dossier](../../models/qwen/qwen3.6-35b-a3b/README.md). |
| **Dual-B70 TP2** | Not a route on this host | Stock TP2 measured −45…−71 % (RL-075); EXL3 TP2 blocked at load (RL-103). The only open TP2 path is RQ-06 (custom copy-only collective). Default stays independent stacks, [ADR-0008](../../decisions/adr-0008-dual-b70-independent-default.md). |

## Historical or unavailable routes

These findings remain scientifically useful but are not copy-paste launch recommendations.

| Route | Availability / disposition |
|---|---|
| Qwen3.8-27B GPTQ-INT4 G128 + BF16 MTP4 (former champion, 102.61 tok/s historical) | Artifact **removed** 2026-09-27: deterministic in-head arithmetic defect ("30" instead of "14") in the weights (RL-087, RL-092). W4A16/GPTQ family closed for this model (ADR-0010). |
| Ornith 1.5 35B-A3B MixedCal-v2 GPTQ-INT4 + MTP1 | Closed, quality-rejected 2026-08-24 (model-status). 96.43 tok/s @150 W remains a historical speed record only. |
| Qwen3.8-Flash-Next llama.cpp AP-Q4_K_XL dual-card split | Superseded 2026-10-09 by the one-card Strata route (ADR-0011); 21.6 tok/s historical anchor (RL-070). |
| Nemotron llama.cpp locally requantized model-weight Q4_0 | Historical artifact deleted. Its model-specific speed result remains evidence; it does not generalize to other models and is unrelated to prohibited KV-cache `q4_0` K. |
| ThinkingCap Qwen3.6-27B GGUF profiles | Weight files recorded absent as of 2026-08-18. Launcher entries are not evidence that the artifacts exist. |
| Qwen3.6-35B-A3B GGUF profiles | Recorded GGUF files absent as of 2026-08-18. Do not feed the active vLLM profile path to llama-server. |

## Cross-route rules

1. **Do not use one universal quantization rule.** Follow the [quantization decision record](../reference/quantization-decision-record.md) and exact model evidence.
2. **llama.cpp KV default:** `q8_0` K + `q4_1` V with flash attention. KV-cache `q4_0` K is prohibited. This does not apply to vLLM KV configuration.
3. **Configured cap is not measured draw.** Report card draw only from synchronized energy-counter evidence.
4. **Cold input rate is not llama-bench prefill.** For vLLM, name actual prompt tokens/client TTFT and retain cache-counter controls.
5. **C1 and Cn are separate.** Never place aggregate concurrency in a C1 decode field or derive wall aggregate by summing per-request rates.
6. **Sampling belongs to the result identity.** Qwen3.8 serving uses the model-card preset selected by thinking level; greedy results remain diagnostic.
7. **Availability, correctness, capacity, speed, and publication approval are independent.** A local directory or successful server load proves none of the others.
8. **Dual-card default is independent stacks**, one engine per GPU. vLLM TP2 is not a current recommended route (host: X570-F dual x8/x8 since 2026-08-27); see [ADR-0008](../../decisions/adr-0008-dual-b70-independent-default.md). External humble-b70-llm TP2 numbers are not this host.

## Launch authority

Use the current launcher/control-plane guide at [`docs/operations/launcher-guide.md`](../operations/launcher-guide.md). Do not copy launch commands from archived configuration tables without verifying the exact image digest, artifact, patches, cache policy, context, scheduler, and VRAM reserve.
