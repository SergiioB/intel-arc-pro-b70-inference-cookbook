---
title: "Exact Live Qwen3.6 27B MTP Vision Baseline"
type: doc
date: 2026-07-18
status: final
---

# Exact Live Qwen3.6 27B MTP Vision Baseline

**Run:** 2026-07-18 23:17–23:27 CEST  
**Target:** production profile `qwen27-mtp-q5-256k-vision-180w`  
**Status:** performance and 32K retrieval baseline valid; short-text and vision user-visible response gate failed; quality extension stopped on conservative RAM guard  
**Raw evidence:** `/mnt/<other-disk>/inference-models/baselines/qwen27-live-exact-20260718T231705+0200/`  
**Protocol:** `~/inference/docs/research/protocol-live-exact-baseline-20260718.md`

## Executive result

The exact live production engine sustains **26.11 ± 0.45 token/s** server-reported decode across four diverse 384-token text requests at 180 W. Local streaming decode is 26.04 ± 0.45 token/s, showing negligible loopback/streaming loss. Mean short-request TTFT is **724.8 ± 2.9 ms**, MTP draft acceptance is **88.66% ± 4.48 percentage points**, and measured board power is **178.76 ± 0.10 W** over the full requests.

Long-context prefill is the confirmed performance bottleneck. Prompt processing falls from **620.10 token/s at 4,112 server tokens** to **305.27 token/s at 32,768 tokens**, a 50.77% rate reduction. The 32K prompt has 7.97 times as many tokens but requires 16.19 times as much prompt-evaluation time (6.63 seconds versus 107.34 seconds). TTFT correspondingly increases from 6.83 seconds to 107.39 seconds. A separate 32K quality run reproduces the prompt rate at 307.26 token/s and correctly retrieves `M00909` and `cobalt-909`.

The highest-impact usability finding is independent of engine speed: with the production default `--reasoning auto --reasoning-budget -1`, ordinary text and vision requests exhaust output caps of 384, 768, and 1,024 tokens entirely in `reasoning_content`, return `finish_reason=length`, and provide **empty user-visible `content`**. The earlier guidance that “512+” tokens is sufficient is disproved for these prompts. The 32K retrieval request does stop normally at 460 tokens and returns correct content, so this is prompt-dependent unbounded reasoning behavior rather than a universal formatter failure.

Vision recognition is partial. The model correctly identifies the circular cyan/black B70 microchip logo, yellow lightning bolt, and main colors inside reasoning, but repeatedly transcribes `INFERENCE` as `INFERENC` and never reaches final content through a 1,024-token cap. The production vision path therefore passes basic semantic recognition but fails the transcription and user-visible completion gate on this fixture.

The final 1,024-token extension stopped further work when host `MemAvailable` fell to 20,825,484 KiB (19.86 GiB), just below the deliberately conservative 20 GiB guard. This was not an OOM: swap remained unused apart from 5.7 MiB, the service remained healthy on the same PID with zero restarts, and no xe/OOM kernel fault appeared. No further generation was attempted.

## Fixed production configuration

| Field | Measured condition |
|---|---|
| llama.cpp | build 9853, commit `7af4279f4`, SYCL/Level Zero |
| Model | `Qwen3.6-27B-MTP-Q5_K_M.gguf`, SHA-256 `ffb9a0…c0fe4` |
| Vision projector | `mmproj-Qwen3.6-27B-F16.gguf`, SHA-256 `eacf61…bde45` |
| Context / slots | 131,072 / one |
| KV | K=`q8_0`, V=`q4_1`, flash attention on |
| Offload | 99 GPU layers, `-ncmoe 0`, no mmap |
| MTP | embedded draft head, n-max 4, p-min 0.75 |
| Batch | `-b 8192`, `-ub 4096` |
| Power | fixed 180 W cap |
| Reasoning | `auto`, unlimited budget (`-1`) |
| Service | PID 3025 throughout, `Restart=no`, zero restarts |
| Request path | authenticated streaming `/v1/chat/completions`, loopback, `cache_prompt=false` |

No service, profile, model, KV type, context, batch, power, environment, mount, or recovery setting changed. The warmup was discarded. All measured test requests report `cache_n=0`.

## Four-prompt text baseline

| Test | Prompt / output tokens | TTFT (s) | Engine TG | Stream TG | Wall output | MTP acceptance | Card W | Finish / content |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Law | 49 / 384 | 0.728 | 26.37 | 26.30 | 25.12 | 89.45% | 178.61 | length / empty |
| Systems | 52 / 384 | 0.725 | 25.54 | 25.48 | 24.37 | 82.17% | 178.78 | length / empty |
| Design | 57 / 384 | 0.725 | 26.56 | 26.50 | 25.30 | 92.24% | 178.82 | length / empty |
| Comparison | 46 / 384 | 0.721 | 25.96 | 25.89 | 24.76 | 90.79% | 178.81 | length / empty |
| **Mean ± sample SD** | — | **0.725 ± 0.003** | **26.11 ± 0.45** | **26.04 ± 0.45** | **24.88 ± 0.41** | **88.66% ± 4.48 pp** | **178.76 ± 0.10** | 0/4 final content |

Server engine TG and locally observed streaming TG differ by only 0.07 token/s on average. Wall output throughput is lower because it includes roughly 0.52 seconds of prompt evaluation plus first-token overhead.

Short-prompt `prompt_per_second` ranges from 87.84 to 108.47 token/s and is not a sustained-prefill measure; fixed overhead dominates prompts of only 46–57 tokens.

## Prefill and context scaling

| Gate | Raw tokenizer target | Server prompt tokens | Prompt ms | Prompt token/s | TTFT | Output cap | Engine TG | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 4K | 4,102 | 4,112 | 6,631 | 620.10 | 6.834 s | 32 | 31.96 | Timing valid; output truncated in reasoning |
| 32K | 32,758 | 32,768 | 107,341 | 305.27 | 107.390 s | 32 | 24.30 | Timing valid; output truncated in reasoning |
| 32K quality repeat | approximately 32,779 | 32,789 | 106,716 | 307.26 | 107.562 s | 768 | 23.13 | Correct retrieval; stopped normally at 460 tokens |

The progress telemetry shows decreasing cumulative prefill rate within the 32K request: 590 token/s at 8K, 460 at 16K, 373 at 24K, 337 at about 28.7K, and 308 at 32.8K. This is direct evidence of context-length-dependent prefill cost, not a projection.

The 32K quality response is exact:

```text
Identifier: M00909
Checksum phrase: cobalt-909
```

No 64K or 128K request was attempted in this phase. Extrapolating the 32K rate to those lengths would be a projection and is intentionally omitted.

## Reasoning and response-completion gate

| Workload | Output cap | Completion tokens | Finish | Content chars | Reasoning chars | Gate |
|---|---:|---:|---|---:|---:|---|
| Four diverse text prompts | 384 each | 384 each | length | 0 each | 1,834–2,095 | **Fail** |
| Short systems quality | 768 | 768 | length | 0 | 3,599 | **Fail** |
| Short systems quality | 1,024 | 1,024 | length | 0 | 4,730 | **Fail** |
| Vision logo | 256 | 256 | length | 0 | 1,136 | **Fail** |
| Vision quality | 768 | 768 | length | 0 | 2,785 | **Fail** |
| Vision quality | 1,024 | 1,024 | length | 0 | 3,694 | **Fail** |
| 32K ledger quality | 768 cap | 460 | stop | 46 | 1,497 | **Pass** |

The reasoning traces explicitly draft an answer and repeatedly state that they are ready to answer, but continue self-review until the cap. Raising the cap from 768 to 1,024 merely adds more internal review for the short text and vision prompts. This makes “increase max tokens” an insufficient production recommendation by itself.

This test does not determine whether request-level reasoning budgets, reasoning-off mode, a prompt-template adjustment, or another sampler/server behavior is the correct fix. Those are hypotheses for a separate guarded A/B.

## Vision evidence

Fixture SHA-256: the existing 256×256 `b70-launcher.png` was recorded in run metadata. Expected visible features are a cyan circular dial, central cyan-outlined microchip, `B70`, `INFERENCE`, a yellow lightning bolt, and a dark background.

Measured vision behavior:

- First image request: 95 prompt tokens, 1.003 seconds prompt evaluation, 1.981-second TTFT, 26.21 engine token/s, 85.47% MTP acceptance.
- Warm image quality requests: 104 prompt tokens, 0.935–0.949 seconds prompt evaluation, 1.146–1.830-second TTFT, 25.34–27.30 engine token/s, 84.06–90.66% MTP acceptance.
- Reasoning correctly identifies the shape, colors, `B70`, microchip, and lightning bolt.
- It repeatedly resolves the lower word incorrectly as `INFERENC` rather than `INFERENCE`.
- All three vision requests finish by length with empty final content.
- The server journal records repeated `non-consecutive token position` warnings during vision processing. No xe/kernel error accompanies them, but they warrant focused vision-path investigation.

The earlier profile claim of approximately 29 engine token/s and 93–95% vision MTP acceptance was not reproduced. Current measured vision TG spans 25.34–27.30 and acceptance spans 84.06–90.66% across different output lengths. This is a measured contradiction, though not a controlled A/B against the historical run.

## Historical reconciliation

| Evidence | Older result | Current result | Interpretation |
|---|---:|---:|---|
| Jul-16 Qwen27 MTP TG at 165 W | 25.07 | 26.11 at 180 W | Broadly consistent; +4.15% is not attributable to one factor because power and API workload differ |
| Jul-16 Qwen27 PP4K | 613 | 620.10 | Broadly consistent (+1.16%) |
| Historical vision profile label | 29 TG | 25.34–27.30 | Not reproduced |
| Historical vision MTP acceptance | 93–95% | 84.06–90.66% | Not reproduced |
| Root guide: “512+” avoids empty content | implied pass | empty through 1,024 on text and vision | Disproved for tested prompts |
| Root guide: zero context speed penalty | claimed through large contexts | PP rate falls 50.77% from 4K to 32K | Disproved for exact live API prefill |

The current and Jul-16 TG/PP figures support the same general dense-model performance class. Old `lmx`, repeated-prompt, parallel-throughput, or different-KV results remain unlike comparisons.

## Resource and safety evidence

| Point | GPU temp | Visible VRAM free | Host available RAM | Status |
|---|---:|---:|---:|---|
| Preflight | 50 °C | 4,121 MiB | 25.47 GiB | Pass |
| Core suite end | 72 °C | 3,540 MiB | 21.11 GiB | Pass |
| Final 1,024 extension | 72 °C | 3,540 MiB | 19.86 GiB | **RAM guard stop** |

At the final check:

- production was active/running on PID 3025 with zero restarts;
- exactly one llama-server existed and the slot was idle;
- profile, 180 W cap, endpoint health, and context were unchanged;
- swap remained effectively unused;
- `/mnt/<models-disk>` still had 26.70 GB decimal free, root 15.97 GB, and the experimental workspace 49.40 GB;
- no xe reset/fault/error or OOM-kill appeared in the kernel log;
- the approved workspace consumed only a few MiB and no download occurred;
- `/mnt/<other-disk>` was not accessed for writes or modified.

The fall in available RAM is largely accompanied by about 18.9 GB of filesystem cache and is reclaimable in principle, but no cache-drop or other host mutation was performed. The guard is intentionally based on `MemAvailable`, so the correct action was to stop.

## Evidence integrity

Primary artifacts:

| Artifact | SHA-256 |
|---|---|
| Core `summary.json` | `6c2d953465be8cf0ff703e41d1a781a66e87e43041049225489a5b550e3c3efd` |
| Core `metadata.json` | `6e6339093f721f235d8c9e8a21db42c2122840381d67f936309ca9c627ec2a5d` |
| 768-token quality follow-ups | `f34ff0045084d88db932f17f5b5c4f77be474fa3df07c09b99f96b53e2cf0808` |
| Runner | `e7a008ac030de9be4b8b5924d4cb0e0e9ccdc318bc2488f1762cc839b86d2872` |
| Protocol, including execution addendum | `4e12ed40bf2fa136bc53127e2a85b9f24f7a3bd2274a73053c00181278130c67` |
| Raw `SHA256SUMS` | `b62f57210014c309b55fb980b8f95af7d6ed2b612503e4ba737b8ffbdf3aff44` |

`SHA256SUMS` in the raw run directory covers every prompt, per-test JSON, and SSE stream present at finalization except itself. The 1,024-token extension deliberately has per-test evidence but no combined summary because its post-vision guard stopped the wrapper before combined-file creation.

## Conclusions and next gate

Confirmed facts, ranked by impact:

1. **Response completion, not decode speed, is the immediate user-facing bottleneck.** Short text and vision return no final content through 1,024 tokens under unlimited automatic reasoning.
2. **Long-context prefill is the dominant engine bottleneck.** API PP rate halves between 4K and 32K, producing 107-second TTFT at 32K.
3. **MTP and streaming are working.** Text engine TG is stable near 26.1, streaming overhead is negligible, and mean acceptance is 88.7%.
4. **32K retrieval quality can succeed.** The deterministic final record is returned exactly in a normal-stop response.
5. **Vision is semantically functional but not production-clean.** It recognizes the design, mis-transcribes the small word, emits token-position warnings, and fails to reach final content.

The next lowest-risk falsifiable experiment should be a request-level reasoning-control A/B after `MemAvailable` returns above the 20 GiB floor. It should test whether the live API honors bounded reasoning budgets without changing the service, and promote a client policy only if it produces non-empty correct content, normal stops, no tool/JSON regression, lower wall latency, and unchanged engine stability. No profile or server change should be attempted until that request-level path is exhausted.
