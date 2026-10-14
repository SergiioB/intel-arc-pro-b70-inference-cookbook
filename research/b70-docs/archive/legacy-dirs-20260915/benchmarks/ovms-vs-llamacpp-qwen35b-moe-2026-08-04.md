# OVMS/OpenVINO vs llama.cpp SYCL — Qwen3.6 on B70 (MoE + Dense)

**Date:** 2026-08-04
**Host:** B70 desktop (b70-host), Intel Arc Pro B70 32GB, 150W cap
**Engine A:** OpenVINO Model Server 2026.2.1-gpu (Docker), INT4 IR models
**Engine B:** llama.cpp SYCL — b10222 (build-sycl-0801) at bench time; **master 0804 (build-sycl-0804, PROMOTED) added below**
**KV:** llama.cpp used q8_0 K + q4_1 V, FA on, -ngl 99 -ncmoe 0 (fleet standard)

## Full measured matrix (MoE + Dense)

| Model family | Metric | llama.cpp SYCL b10222 | llama.cpp master 0804 | OVMS/OpenVINO | Winner |
|--------------|--------|----------------------|----------------------|---------------|--------|
| **MoE 35B-A3B** | Decode (t/s) | 70.1 | **72.6** | 58.4 | llama.cpp +24% |
| | Prefill pp4096 | 1691 | **2128** | ~1474 | llama.cpp |
| | TTFT | ~69 ms | — | 195 ms | llama.cpp |
| **Dense 27B** | Decode (t/s, base) | 20.8 | 21.3 | **29.8** | OVMS +40% |
| | Decode (t/s, MTP-4) | 24-29 | 24-29 | 29.8 | ~tie |
| | Prefill pp4096 | 795 | **936** | ~900 | llama.cpp 0804 |
| Both | Sustained temp | 63-70°C | — | **58-62°C** | OVMS cooler |
| Both | VRAM after load | heavier | — | **lighter** | OVMS |

## Head-to-head (measured, MoE)

| Metric | llama.cpp SYCL b10222 | OVMS/OpenVINO 2026.2.1 | Winner |
|--------|----------------------|------------------------|--------|
| **Decode (tg128 / sustained)** | **70.1 t/s** | 58.4 t/s | llama.cpp +20% |
| **Prefill pp512** | 1061 t/s | — | — |
| **Prefill pp4096** | 1691 t/s | — | — |
| **Prefill ~1.6K tokens** | ~1600 t/s (interp) | 1474 t/s | ~tie |
| **TTFT (streaming)** | ~69 ms (lmx) | 195 ms | llama.cpp |
| **Temp under sustained load** | 63-70°C | **58-62°C** | OVMS cooler |
| **VRAM after load** | ~24.6 GB | 20.8 GB | OVMS lighter |

## Method notes

- **llama.cpp:** `llama-bench -p 512,4096,8192 -n 128 -r 3 -fa 1 -ctk q8_0 -ctv q4_1 -b 8192 -ub 4096 -mmp 0 -t 8 -dev SYCL0`. Steady-state, warmup-free (llama-bench does internal reps). Raw: `results/ab-llama-bench-20260803-1648.txt` equivalent method.
- **OVMS:** OpenAI `/v3/chat/completions`, `chat_template_kwargs.enable_thinking=false` (else output fills with reasoning tokens and content is empty). Decode measured via `completion_tokens / wall` over 3 reps × 4 diverse prompts (mean 58.4). Prefill via a 1592-token prompt, `prompt_tokens / wall` (steady-state rep = 1474). TTFT via streamed first chunk.
- OVMS measured via HTTP wall-clock; llama.cpp via llama-bench engine rate. Both reflect steady-state generation on identical hardware/model family. Not bit-identical quantization (INT4 vs Q4_K_XL) — see caveats.

## Verdict — MoE vs Dense differ (updated for master 0804)

| Workload | Best engine | Why |
|----------|------------|-----|
| **MoE 35B-A3B** (single-stream) | **llama.cpp SYCL** | decode +24% (72.6 vs 58 t/s on 0804), faster TTFT, prefill 2128 vs ~1474 |
| **Dense 27B** (single-stream) | **OVMS/OpenVINO** | decode +40% over llama.cpp base (30 vs 21 t/s); ~tie with llama.cpp+MTP |
| **Dense 27B prefill** | **llama.cpp 0804** | 936 t/s @ pp4096 (#25874 XMX FA) beats OVMS ~900 |
| High-concurrency / vision | OVMS | continuous batching, native VLM, prefix cache |

**The model type flips the answer.** llama.cpp wins MoE decode; OVMS wins dense decode
(INT4 dense reads fewer bytes/token on a bandwidth-bound model). The master-0804 build
(#25874 quantized-KV XMX FA) closed the dense prefill gap — llama.cpp now also wins
dense prefill. Keep llama.cpp as the default; use OVMS specifically for dense decode or
multi-client serving.

## vLLM and SGLang XPU — EMPIRICALLY DEAD ENDS for this stack (verified 2026-08-04)

**vLLM XPU 0.21.0 (intel/vllm):** Cannot load our models. Both GGUF variants fail with
`ValueError: GGUF model with architecture qwen35moe / qwen35 is not supported yet.`
The vLLM registry has Qwen3_5Moe but no GGUF loader for the `qwen35`/`qwen35moe` GGUF
model_type, and no Qwen3.6-35B-A3B validated entry on XPU. Switching would require
downloading HF safetensors/GPTQ checkpoints (BF16 35B = ~70 GB, won't fit 32 GB VRAM;
quantized HF ckpts for Qwen3.6 aren't in the XPU validated list).

**SGLang XPU:** No XPU docker image exists (`lmsyscommunity/sglang` has zero xpu tags).
Requires python3.12 source build + safetensors model — same GGUF/format blocker as vLLM.
No evidence it beats llama.cpp on single-stream.

**Conclusion:** Neither vLLM nor SGLang is usable with the current GGUF + Qwen3.6 stack on
this B70. OVMS (OpenVINO IR) is the only viable cross-engine alternative, and it wins on
dense. Do not attempt vLLM/SGLang again unless the model format changes (GGUF → safetensors).

## Recommendation

Keep **llama.cpp SYCL master 0804 (build-sycl-0804)** as default production (MoE + MTP dense) — promoted 2026-08-04. Add **OVMS** as a
secondary profile for dense-27B or vision/high-concurrency workloads (it's already
installed with both INT4 models on /mnt/<models-disk>). Skip vLLM/SGLang.

## Files / artifacts

- OVMS benchmark scripts: `/mnt/<models-disk>/ovms_bench.py`, `ovms_decode.py`, `ovms_final.py`, `ovms_dense.py`, `ovms_launch.sh`
- OVMS MoE model (18.6 GB): `/mnt/<models-disk>/OpenVINO-Qwen3.6-35B-A3B-int4-ov`
- OVMS dense model (15 GB): `/mnt/<models-disk>/OpenVINO-Qwen3.6-27B-int4-ov`
- vLLM test script (failed): `/mnt/<models-disk>/vllm_test.sh`
- llama.cpp raw: `results/` (llama-bench)
