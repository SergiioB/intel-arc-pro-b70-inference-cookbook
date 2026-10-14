---
title: "Qwen3.6 27B Reasoning-Control Candidate Discovery"
type: doc
date: 2026-07-19
status: final
---

# Qwen3.6 27B Reasoning-Control Candidate Discovery

**Evidence cut:** 2026-07-19 00:24 CEST  
**Target stack:** llama.cpp SYCL b9853 on Intel Arc Pro B70 32 GB  
**Production control:** Qwen3.6 27B MTP Q5_K_M vision at 128K, unchanged  
**Approved candidate workspace:** `/mnt/<other-disk>/inference-models` (ext4)  
**Selection:** `bottlecapai/ThinkingCap-Qwen3.6-27B-GGUF`, official Q4_K_M plus matching projector

## Decision summary

ThinkingCap Q4_K_M is the single candidate selected for download and staged evaluation. It directly targets the observed failure—runaway thinking that consumes the entire completion—and has a publisher-owned llama.cpp GGUF, embedded MTP support, a matching vision projector, Apache-2.0 lineage, exact LFS checksums, and enough local fit headroom.

The selection is for testing, not promotion. Publisher claims are hypotheses until reproduced on the B70 workload.

No ready GGUF with a model-native hard thinking budget was found in the current public repositories searched. Qwen3.6's inherited chat template supports a per-request thinking on/off switch, and llama.cpp b9853 supports both `chat_template_kwargs.enable_thinking` and request-level `reasoning_budget_tokens`. Those engine/template controls will be tested independently from ThinkingCap's shorter-thinking finetune.

## Verified primary candidate

Source: [official ThinkingCap GGUF repository](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.6-27B-GGUF), commit `0dd71bbcc88a4134cf2abc389480211d77e5a586`, last modified 2026-07-16 10:10 UTC.

| Artifact | Bytes | Published SHA-256 | Role |
|---|---:|---|---|
| `ThinkingCap-Qwen3.6-27B-Q4_K_M.gguf` | 16,810,713,056 | `b0651e28555bde7d2459ce99f091319b1a547143463e8d49f2aa7f572675fe67` | Selected Q4 model |
| `mmproj-ThinkingCap-Qwen3.6-27B-f16.gguf` | 931,145,888 | `81a714ac5e8e15687371fc95a180953a29b732962f6616f791063ff127559412` | Required matching vision projector |
| **Total** | **17,741,858,944** | — | One model candidate plus its projector |

The publisher's [main model card](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.6-27B) reports a 45.8% macro reduction in thinking tokens across its out-of-domain suite, aggregate capability 81.5 for base versus 80.7 for ThinkingCap, and truncation reduction from 2.9% to 0.4%. Individual results vary: MMLU-Pro thinking falls 53.7%, LiveCodeBench 41.1%, LongBench v2 39.1%, RealWorldQA 48.5%, system-prompt adherence 40.0%, and Claw-Eval 25.2%. These are publisher measurements with five seeds and reported confidence intervals, not local results.

The GGUF card's separate 100-example validation claims Q4_K_M+MTP MMLU-Pro accuracy 0.85, RealWorldQA 0.78, and average accepted draft length 3.74 at n=4. Its hardware reports 89.2 token/s, which is not portable to the B70 and will not be used as a projection.

### Fit calculation

At selection time the approved workspace had 49,400,463,360 bytes free. After both artifacts it should retain 31,658,604,416 bytes (29.49 GiB), exceeding the required 10 GiB safety margin by 18.86 GiB. Downloads use a `.part` file followed by checksum verification and atomic rename, so no second full copy is required.

The Q4 weights are about 3.02 GB smaller than the production Q5 file. At a bounded 32K/64K test context they should fit the B70 with more VRAM headroom than production, but actual VRAM/RAM is a required staged measurement.

## Other verified candidates

### bytkim no-thinking PI tune

Source: [bytkim Qwen3.6 27B MTP PI-tune GGUF](https://huggingface.co/bytkim/Qwen3.6-27B-MTP-pi-tune-GGUF), commit `22943ba3319569b743d4cb3832e3aa336785d8b8`.

- Q4_K_M is 16,836,927,680 bytes, SHA-256 `dbc7ff308e07345024466de9f57a0a9f32e64e50f536059ebaf2c03463e92e72`.
- The card describes a no-thinking PI-style agent/coding SFT and publishes Terminal-Bench 2.0 pass@1 28.09, but the Hub metadata marks that metric unverified and the card says finalized coding success benchmarks are forthcoming.
- The repo has no mmproj file despite multimodal tags; a compatible projector is not proven by the artifact set.
- It remains a useful later direct-response candidate, but it does not test controlled reasoning and is not downloaded in this phase.

### natfii VLM Cascade

Source: [natfii Qwen3.6 27B VLM Cascade](https://huggingface.co/natfii/Qwen3.6-27B-VLM-Cascade), commit `fefc6ae485d771aee507c6686ba7532de2e4163d`.

- The card specifies thinking off by default, opt-in thinking, and hard budgets through vLLM/SGLang serving features.
- The public repo contains two BF16 safetensor shards totaling 55,563,008,560 bytes and no GGUF, public convert, alternate branch, or GGUF beta.
- Its sibling deployment repo is NVFP4 for GB10/DGX Spark, not a llama.cpp B70 artifact.
- It exceeds the workspace by itself, is unsupported by the selected runtime path, and is explicitly rejected for this phase. No conversion or Docker experiment will be attempted.

## Search for newer toggle/budget GGUFs

Current Hugging Face search and author/repository inspection found newer-dated ThinkingCap quant uploads from Bartowski and jpetrina. They are alternate quantizations of the same BottleCap checkpoint, not new reasoning-control finetunes:

- [Bartowski ThinkingCap GGUF](https://huggingface.co/bartowski/bottlecapai_ThinkingCap-Qwen3.6-27B-GGUF), modified 2026-07-17, offers imatrix quants and a matching projector. Its Q4_K_M is 17,772,536,512 bytes with SHA-256 `c926c5670e2369896067dd7fc7c9484234725eb8863876bed2dbfe136898175a`.
- [jpetrina pure IQ4_XS](https://huggingface.co/jpetrina/ThinkingCap-Qwen3.6-27B-IQ4_XS-pure-GGUF), modified 2026-07-17, is 14,534,381,856 bytes with SHA-256 `d92d1f37a802a78330e49e35ebe077d52499f03be970a54656fd3810884f0202`.

Neither adds a genuine hard budget or new per-request mode. The official publisher Q4 is preferred because its exact quant appears in the publisher's validation table, is smaller than Bartowski Q4_K_M, and minimizes provenance hops.

## Reasoning-control capability to test

The [official Qwen3.6 card](https://huggingface.co/Qwen/Qwen3.6-27B) states that thinking is on by default and that non-thinking mode is requested with `chat_template_kwargs: {"enable_thinking": false}`. It explicitly says the old `/think` and `/nothink` soft switch is unsupported.

Local llama.cpp source confirms the b9853-era API paths:

- `chat_template_kwargs.enable_thinking` overrides template mode per request;
- `reasoning_budget_tokens` overrides the server's default reasoning budget per request;
- the current production launch sets `--reasoning-budget -1`, meaning unrestricted;
- budget 0 means immediate reasoning end, and positive N sets a hard token budget when start/end reasoning tags are detected.

Therefore the A/B matrix must separate three mechanisms:

1. **ThinkingCap default thinking:** tests whether the finetune naturally terminates earlier.
2. **Template non-thinking:** tests direct response with `enable_thinking=false` when the embedded template supports it.
3. **Hard-budget thinking:** tests `reasoning_budget_tokens` at bounded values while leaving thinking enabled.

ThinkingCap's source repo does not publish a standalone tokenizer chat template, so template support remains a load-stage fact to extract from the downloaded GGUF rather than an assumption.

## Promotion gates

The candidate cannot be recommended unless it passes all of these against the exact production control:

1. Checksum, GGUF metadata, template, MTP-head, and projector compatibility.
2. Isolated load under a bounded cgroup with no host OOM, reboot, xe fault, or production overlap.
3. Non-empty final content and normal stop for representative short text at bounded output caps.
4. Default-thinking, non-thinking, and hard-budget mode behavior when supported.
5. Four-prompt text performance and answer-quality comparison.
6. Deterministic function/tool-call syntax and a bounded local agent task.
7. Vision semantics and exact `B70 INFERENCE` transcription.
8. 4K prefill and bounded 32K retrieval with correct `M00909` / `cobalt-909`.
9. TTFT, server prompt/decode rates, MTP acceptance, VRAM/RAM, power, temperature, and journal/kernel stability.
10. Exact restoration of the production profile, process count, health, power, hashes, and endpoint after every isolated test window.

No production profile, catalog, model, projector, or service definition will be replaced during evaluation.
