---
title: "Campaign: Qwen3.8-27B OpenVINO GenAI vs Cascadia — 2026-09-11/12"
type: research
date: 2026-09-11
status: final
---

# Campaign: Qwen3.8-27B OpenVINO GenAI vs Cascadia — 2026-09-11/12

Private evidence. Public claim boundary is the cookbook pages
`docs/architecture/openvino-and-cascadia.md` and
`docs/qwen38-27/OPENVINO-CASCADIA.md`.

## Host

- desktop-b70, 2× B70, 60 GiB RAM, 150 W both cards
- Python `/mnt/<models-disk>/.ov-genai-venv`: openvino 2026.5.0 nightly, genai 2026.5.0
- Cascadia rebuilt 2026-09-12 against OpenVINO GenAI **2026.3.1** (`libopenvino.so.2631`)
  at `~/cascadia/cascadia-ov2631/cascadia`

## What actually works (2026-09-12)

Official **INT4** Hub IR is experimental and emits collapsed logits (CPU=GPU).
Official **INT8** Hub IR `OpenVINO/Qwen3.8-27B-int8-ov` (language bin 25 640 570 530 B)
emits real text.

Matched protocol: greedy n=128, 1 warmup + 3 reps, GPU.0, 150 W.

| Engine | Artifact | wall tok/s | Quality |
|---|---|---|---|
| OpenVINO GenAI Python 2026.5 `VLMPipeline` | INT8-OV whole IR | **13.09** (9.78 s / 128) | PASS — English, describes the HF ai2d image |
| Cascadia 0.2.3 `qwen35` 1-stage (ov 2026.3.1) | surgery of that INT8 IR | **4.56** (28.0 s / 128) | PASS — English; includes thinking tokens; `hello world` smoke |
| Cascadia `ov-genai` (stock 0.2.3, OV 2026.2) | INT4 or INT8 VLM | SIGSEGV | Tokenizer/VLM CB |
| Cascadia `ov-genai` (rebuild OV 2026.3.1) | INT8 VLM | SIGSEGV | still dies on `runner load` |
| llama.cpp SYCL Q4_K_M | GGUF, same night | **18.58 ± 0.31** tg128 | PASS (different artifact) |

JSON: `/mnt/<models-disk>/ov-vs-cascadia/results/ovpy-int8-latency-f16.json`,
`cascadia-qwen35-int8.json`.

## INT4 night (do not headline)

Every token id 0 (`!` / empty HTTP). Same string on CPU (251 s) and GPU.
`ignored_scope: null` INT4_ASYM ratio 1.0 on GatedDeltaNet.

## Fixes that were real (not the INT4 collapse)

1. Truncated Hub sidecars (tokenizer 0 B, embeddings 8 MB, merger XML).
2. `VLMPipeline.generate(prompt, cfg)` binds cfg as images — use `generation_config=`.
3. Minja `enable_thinking is undefined`.
4. Cascadia stock binary SONAME 2620 vs IR needing 2026.3+/2026.4+.
5. `cascadia shard --num-stages 1` on INT8; worker `--engine qwen35`.

## Open gates

- Cascadia `ov-genai` without SIGSEGV on this VLM (qwen35 is the working Cascadia lane)
- Official INT4-OV with GDN kept out of INT4
- Dual-card `qwen35` stages
- FastDraft companion IR
