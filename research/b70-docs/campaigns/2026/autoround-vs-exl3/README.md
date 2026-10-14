---
id: CAM-2026-AUTOROUND-VS-EXL3
status: done
title: "AutoRound vs EXL3 on Qwen3.8-27B, one B70 — complete re-measure"
date: 2026-09-26
round2_20260927: "task-quality + cache round: arm C 9/10, EXL3 8/10, OV 4/10 on host-executed bugfix/codegen battery (W4A16 defect = narrow in-head-arithmetic only); cache gains EXL3 5.4x/armC 4.3x/OV 2.9x on 10K shared prefix; RL-088. Seagate HDD died mid-round (arm C artifact on it, evidence preserved). final_20260927: "COMPLETE. VERDICT: W4A16/GPTQ-family int4 is defective on Qwen3.8-27B — bf16 base + EXL3 + OV int4 + Q4_K_M answer 14; ALL three W4A16 exports (GPTQModel, AR compressed-tensors, AR auto_gptq+our corpus) answer 30. Arm C also brought no speed (15/14.4 vs champion 73.4). KV: W4A16 cannot reach 262144 (max ~211K fp8); EXL3 fp8 fits 262144 and proved 131,098-tok completion. Caps: EXL3 +38-42% 150->230W vs arext +9%, OV +6%. Full: results/2026/09/autoround-vs-exl3-20260926/RESULTS.md; RL-084..087"
operator_order: "DO NOT PUSH anywhere until cleared; remove GPTQ checkpoint; get AutoRound; completely re-measure EXL3 vs AutoRound: speed, correctness, quality outputs, KV fitting, memory bandwidth, B70 optimization"
---

# AutoRound vs EXL3 — complete re-measure (2026-09-26)

## Arms

| arm | artifact | engine | MTP | notes |
|---|---|---|---|---|
| A (EXL3) | turboderp/Qwen3.8-27B-exl3 @113cf7ab (4.00bpw, 15.73 GiB) | exl3xpu image @21412bdd, vLLM 0.26.1 | 3 (config) + no-spec arm | RL-082 first run; this campaign re-measures on the full matrix |
| B (AutoRound ext) | dbirks/Qwen3.8-27B-W4A16-AutoRound (compressed-tensors, 18.15 GiB, downloading) | pinned f01e24f6 champion image | none (converter strips mtp) | external calibration, unknown corpus |
| C (AutoRound local) | own conversion, code-heavy corpus, mtp.* BF16 preserved, converter = only variable vs removed GPTQ | f01e24f6 | 4-capable | overnight conversion; the "was it our GPTQ conversion" decider |
| GPTQ | REMOVED per operator order after manifest | — | — | provenance: gptq-removal-manifest.txt; HF copy remains |

## Matrix (per arm, GPU0, serial)

1. **Speed**: C1 p512/g128 + p8192/g128, client post-first + TTFT, n=5 fresh + 1 warmup,
   recommended sampling (temp .7/.8/20/pp 1.5, ignore_eos, unique entropy, cache-hit
   delta 0), all arms WITH MTP (contract below).
2. **Correctness**: 6-case exact-answer battery (greedy, thinking off) + deterministic
   replay (3 prompts x2, sha compare).
3. **Quality outputs**: battery + 3 generative probes (JSON contract, code fix task,
   factual paragraph) retained verbatim, scored by rubric; not PPL (retired, RL-025).
4. **KV fitting**: max_model_len / KV-pool tokens at standard gmu on this card
   (startup-free recorded), one 131,072-total-token completion where pool allows.

## Fairness contract (operator order 2026-09-26)

- **All arms WITH MTP** at each route's native/production depth: EXL3 MTP3,
  AutoRound MTP4 (dbirks export ships the 15 `mtp.*` BF16 tensors — verified in
  `model_extra_tensors.safetensors`), OV MTP5 (`openvino_mtp_model.xml`).
  Depth differences are part of the route; labeled, not hidden.
- **Same KV quant across vLLM arms for the primary cells: fp8 KV, ctx 102400** —
  the recorded champion config. OV runs its native u4-KV export (its only
  long-context mode; u8 KV ≥98K crashes, upstream #4483) — labeled deviation.
- **KV/context probe per arm (secondary):** fp8 vs bf16 KV pool size at
  max-model-len 262144 from engine startup logs (GPU-free arithmetic), plus one
  131,072-total-token completion on each arm that fits — answers "does this route
  NEED quantized KV for ≥256K" directly.
5. **Memory bandwidth**: decode byte-floor = streamed weight+KV bytes per token /
   measured step time vs card peak; report effective GB/s and % of peak (RL-072 method).
6. **B70 optimization**: 150 vs 230 W paired A/B (n=3/arm) on the winning config;
   graphs on; eager-vs-graph single check.

## Publication state

ALL pushes frozen (operator order). Cookbook GitHub got b8c3ab0 BEFORE the order
(screen-labeled docs only); XeCores NOT deployed (qa:pin failed on stale pin, unfixed
by design now). Nothing else leaves this host until cleared.

## Evidence

- `gptq-removal-manifest.txt` — hashes of the removed artifact
- per-arm subdirs as they complete
