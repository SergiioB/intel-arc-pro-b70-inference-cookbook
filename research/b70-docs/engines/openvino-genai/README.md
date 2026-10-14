---
title: "OpenVINO GenAI on B70 — engine dossier"
type: engine-dossier
status: living
date: 2026-09-15
updated: 2026-10-09
engine: openvino-genai
---

# OpenVINO GenAI — living dossier

Newest engine on the host (first serious campaign 2026-09-11..15). Alternate
Qwen3.8-27B route behind EXL3 (model-status): INT4 clean p512 median 28.5 tok/s
post-first @150 W (RL-086). The 2026-09-15 "79.5 tok/s champion" row below came
from the grafted-MTP5 ctx sweep and is not comparable with the EXL3 usage-token
rate; RL-086 also found earlier OV speed rows depressed by accumulated chat
history. Thinking-mode handicap on long tasks (4/10 host tasks, RL-088).

## Current state (as of 2026-09-15)

- Runtime: openvino `2026.5.0-22967`, genai `2026.5.0.0-3412`
  (venv `/mnt/<models-disk>/.ov-genai-venv`, deleted 2026-10-13), device `GPU.1` via `VLMPipeline`.
- INT8 OV (Hub int8): base 18.7 tok/s → MTP4 61.6 tok/s (3.29×), byte-identical
  outputs, external ~61 tok/s claim REPRODUCED — `research/2026/ov-cascadia-qwen38-bakeoff-20260911.md`
  + `results/2026/09/qw38-ov-mtp-20260914/SUMMARY.md`.
- INT4-GDN8 (our fixed INT4, GDN float) + Hub int4-ov MTP5 graft: 3.43× over
  no-MTP (76.8 vs 22.4 @512); ctx sweep medians 79.5 / 73.0 / 45.8 / 47.7 /
  29.1 tok/s at 512/2K/8K/32K/64K — `ctx-sweep-ov-int4-gdn8-mtp5.json`,
  graft validation `mtp-graft-validate.json`.

## Required pipeline setup (the recorded blockers)

1. Export is `Qwen3_5ForConditionalGeneration` (VL-style; language graph takes
   `inputs_embeds` only) → `LLMPipeline` fails; use `og.VLMPipeline(dir, 'GPU.1')`.
2. Nightly Minja cannot compile the chat template → render ChatML manually,
   `apply_chat_template=False`.
3. MTP speculative decode requires `GenerationConfig.num_assistant_tokens > 0`
   AND `SchedulerConfig.enable_prefix_caching = false`.
4. Do not override `cache_size` with weights resident — budget check OOMs at load.

## Verified capabilities / limits

| Capability | State | Evidence |
|---|---|---|
| INT8 + MTP4 | 61.6 tok/s @512, parity identical | 2026-09-14 reproduction |
| INT4-GDN8 + grafted MTP5 | 79.5 tok/s @512 client post-first (single-card best, n=3, 2026-09-15) — **≤64K speed arm only** | 2026-09-15 sweep |
| **One-config full range (serving default)** | INT4-GDN8, no-MTP, u4 KV, cache_size=0: 512→131K full protocol (decode 22.4→15.9, cold prefill 1326→493) | `ctx-sweep-ov-int4-gdn8-u4-full.json` |
| u8 KV at ≥98K (MTP) / ≥64K (no-MTP) | crash `CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST (-14)`; KV arithmetic (16 attn × 4 heads × 256 dim ≈ 3 GiB u8 @98K) does NOT explain it vs ~8 GiB free — no clean budget error; upstream issue https://github.com/openvinotoolkit/openvino.genai/issues/4483 | mtp5 sweep + `research/2026/ov-genai-mtp-longctx-wall-upstream-draft-20260915.md` |
| u4 full-protocol ceiling | **131 072**; 196K/262K fail `CL_OUT_OF_RESOURCES (-5)` at `clFinish` (~33.4 GB in use, 814 MiB free) | `ctx-sweep-ov-int4-gdn8-u4-max.json` |
| u4 capacity-only beyond 131K | 40-token probes reach 262 144 (12.8 @196K / 11.3 tok/s @262K, 3.8 GiB free) — capacity proof, not a serving cell | `ov-u4-maxctx.json` |
| MTP at long ctx | 98K: MTP 4.77 vs no-MTP 14.96 tok/s (3.14× SLOWER; 3.4× faster @512) — adaptive draft-disable requested upstream | `ov-ceiling-confirm.json` |
| INT4 vs INT8 (mechanism) | same DPAS rate class (363 TOPS) → prefill parity expected and observed; INT4 wins decode only via weight bandwidth (measured +20%: 22.4 vs 18.7 no-MTP — well under 2× because KV/activations/scales don't shrink; load-time expansion to 8/16-bit would erase it — inspect kernels if prefill regresses) | `research/2026/xmx-dpas-rate-classes-20260915.md` |

## Open next steps (from SUMMARY.md)

- INT4-GDN8 draft tuning (assistant depth 5–6, schedule, batched tokens).
- Cross-GPU draft (`og.draft_model(dir,'GPU.0')` + main on `GPU.1`) — untested.

## Related

Conversion record: `research/2026/qwen38-ov-int4-gdn8-convert-20260912.md`.
Campaign evidence root: `results/2026/09/qw38-ov-mtp-20260914/`.
