# B70 Evidence-Led Improvement Plan - 2026-07-05

Purpose: stop broad trial-and-error sweeps and focus on changes that have a credible path to beating the current single-card B70 stack.

Current local baseline remains llama.cpp SYCL/Level-Zero with `-fa on -ctk q5_0 -ctv q4_1 -ngl 99 -ncmoe 0 -t 8 --no-mmap`.

## Current Local Position

The July 5 local sweep shows the current stack is already competitive for single-card GGUF inference:

| Local candidate | Context | Best useful cap | Result |
|---|---:|---:|---|
| `ornith35-q5-256k` | 256K | 150W | ~72.3 tg/s, ~26.3 GiB VRAM |
| `qwen35-q4-256k` | 256K | 150W | ~65.2 tg/s, ~24.3 GiB VRAM |
| `qwen35-q5-256k` | 256K | 150W | ~63.5 tg/s, ~27.8 GiB VRAM |
| `gemma4-26b-q4-256k` | 256K | 150W | ~55.6 tg/s, ~19.6 GiB VRAM |
| `qwen27-mtp-200k` | 200K | 200W burst | ~28.3 tg/s, ~27.8 GiB VRAM |

Power conclusion: MoE models do not benefit from caps above 150W. Dense Qwen 27B is the only installed model that materially scales from 150W to 200W.

## What Public Evidence Says

| Evidence | Reported result | What it implies for this host |
|---|---:|---|
| Intel LLM Scaler release notes | B70 support landed in `0.14.0-b8.2.1`; `0.14.0-b8.3.1` added FP8 KV; `0.14.0-b8.3.2` fixed Qwen3.5/3.6-27B accuracy. | Use `0.14.0-b8.3.2` for current vLLM testing, but do not expect stock vLLM to automatically beat llama.cpp. Source: https://github.com/intel/llm-scaler |
| Level1Techs B70 user | Single B70 vLLM with `Qwen/Qwen3.6-35B-A3B` BF16 + online `sym_int4` + FP8 KV got ~20.2 tok/s single stream, but scaled to 868 tok/s aggregate at 64 streams. | vLLM's single-stream path is weak; its value is concurrency/throughput. Source: https://forum.level1techs.com/t/thoughts-from-existing-b70-users/248928?page=2 |
| Same Level1Techs thread | A patched pipeline using `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4`, TurboQuant, and XPU graph claimed ~60 tok/s single stream. | The only credible vLLM path toward local llama.cpp speed requires graph capture and custom/patched kernels, not stock eager mode. |
| Our local vLLM scout | `Intel/Qwen3.6-27B-int4-AutoRound` works only with `--dtype float16`; stock BF16 crashed in XPU GDN attention. 150W result was ~18.3 output tok/s. | Stock Intel vLLM is not currently a better single-stream replacement for llama.cpp on this single B70. |
| Puget Systems | On 4x B70, FP16 vLLM Qwen3.6-27B and Qwen3.6-35B-A3B reached only 13.1 and 16.3 tok/s single-user, but higher aggregate throughput at concurrency. | FP16 vLLM multi-GPU is about capacity and concurrency, not single-user speed. Source: https://www.pugetsystems.com/labs/articles/intel-arc-pro-b70-multi-gpu-ai-inference-performance/ |
| SGLang XPU docs | XPU is optimized for Arc Pro B-Series; install from source/Python 3.12; launch with `--device xpu --attention-backend intel_xpu`; XPU graph decode and prefill are opt-in. Speculative decoding is not implemented on XPU. | SGLang is worth testing specifically for XPU graph behavior, not as a generic replacement. Source: https://docs.sglang.io/docs/hardware-platforms/xpu |
| vLLM XPU TurboQuant issue | TurboQuant XPU kernels compile and are correctness-tested on Arc Pro B70; goal is 3-5x KV compression. | Useful for longer context or vLLM capacity; not proven as a speed win on this single card. Source: https://github.com/vllm-project/vllm-xpu-kernels/issues/271 |
| Hal9000 B70 bugfix notes | vLLM TP=1 is slower because XPU often needs eager mode, lacks CUDA-graph-style wins, and GPTQ fused Marlin-equivalent kernels are missing; llama.cpp has MoE/SYCL fused paths. | Explains why our vLLM numbers are worse and why llama.cpp remains the production lane. Source: https://github.com/Hal9000AIML/arc-pro-b70-ubuntu-gpu-speedup-bugfixes |
| Recent Reddit/Level1Techs Windows Vulkan reports | Qwen3.6-35B-A3B Q4 on Windows Vulkan reported ~98-100 tg/s on B70, while Linux Vulkan was much lower in one comparison. | Highest-upside single-card lead, but requires Windows validation and exact flag parity before believing it. Sources: https://www.reddit.com/r/LocalLLM/comments/1ukjz87/intel_arc_pro_b70_llamacpp_vulkan_benchmarks_with/ and https://forum.level1techs.com/t/thoughts-from-existing-b70-users/248928?page=2 |
| Reddit single B70 SYCL report | Qwen3.6-35B-A3B Q4_K_XL at ~75 tok/s on single B70 with llama.cpp SYCL/oneAPI 2026. | Our 65.2 tg/s Q4 and 72.3 tg/s Ornith Q5 are plausible/high-end, but there may be a small remaining SYCL flag/build delta. Source: https://www.reddit.com/r/IntelArc/comments/1u3l4zx/qwen3635ba3b_at_75_tokens_per_second_on_a_single/ |

## Ranked Improvement Paths

### 1. Keep llama.cpp SYCL as production default

Rationale: it is already the fastest proven stack on this host for single-card 256K GGUF MoE, uses the best local KV cache quality/speed balance, and avoids Python/vLLM scheduler and XPU eager-mode overhead.

Immediate changes:

- Keep MoE power cap at 150W.
- Keep q5_0/q4_1 KV as default for llama.cpp.
- Make `ornith35-q5-256k` the fastest installed profile and `qwen35-q5-256k` the best quality/default profile.
- Stop spending long sweeps on MoE 165/180/200W unless a build or kernel changes.

Validation:

- After any llama.cpp update, rerun only `qwen35-q5-256k`, `qwen35-q4-256k`, and `ornith35-q5-256k` at 150W first.
- Only expand power sweeps for dense Qwen 27B.

### 2. Test Windows Vulkan as a separate high-upside lane

Rationale: multiple recent public reports claim Windows Vulkan is substantially faster for Qwen3.6-35B-A3B on B70, including ~98-100 tg/s. This is the largest plausible single-card speed upside.

Risks:

- Reported configs often use default context, short llama-bench tests, Q4 models, and not q5_0/q4_1 KV.
- Windows Vulkan may not support the same KV/cache/long-context behavior or may not match Linux operational constraints.

Minimal validation test:

- Same model: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`.
- Same target context: 256K where possible.
- Same prompt set as local harness.
- Measure `pp`, `tg`, TTFT, VRAM, and actual GPU power/temperature if available.
- Compare against local Linux SYCL Q4: ~65.2 tg/s at 150W.

Decision gate:

- Adopt Windows Vulkan only if it beats Linux SYCL by at least 20% on a real server request at useful context, not just llama-bench `tg128`.

### 3. Treat vLLM as a concurrency/graph-research lane, not the default

Rationale: stock Intel LLM Scaler vLLM works, but our local single-stream result is slower than llama.cpp. Public reports that approach llama.cpp speed require XPU graph and patched/custom kernels.

Promising vLLM directions:

- Try `Qwen/Qwen3.6-35B-A3B` BF16 with online `--quantization sym_int4` + `--kv-cache-dtype fp8`, because a public single-B70 report got stable high concurrency from it.
- Revisit `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` only if we can reproduce the graph/TurboQuant patch stack. Stock image failed locally in IPEX Marlin/MoE fusion.
- Track TurboQuant XPU for KV capacity, but do not assume it improves decode speed.

Do not do:

- Do not run full vLLM wattage sweeps until a vLLM config reaches at least llama.cpp-near single-stream speed or proves a concurrency use case.
- Do not compare vLLM FP8 KV directly to llama.cpp q5_0/q4_1 KV as equivalent quality.

Decision gate:

- vLLM becomes useful locally if it either reaches >50 tok/s single-stream on Qwen35-class models or produces a measured aggregate throughput use case we actually need.

### 4. Test SGLang only for XPU graph behavior

Rationale: SGLang's XPU documentation exposes opt-in decode/prefill graph features that directly target CPU/GPU launch overhead, which public B70 users identify as the bottleneck.

Minimal test:

- Build/source install SGLang XPU in a disposable environment.
- Start with a small verified XPU model to validate the stack.
- Then test Qwen 35B only if quant support/model loading is realistic.
- Compare three modes: no graph, decode graph, decode graph + torch.compile.

Decision gate:

- Continue only if SGLang can load a relevant Qwen/Gemma-class model and produce stable output. If it only supports small BF16 models on B70 today, keep it as a research note.

### 5. Kernel/custom work should target known bottlenecks

Do not write random kernels. The evidence points to specific missing pieces:

- XPU graph/capture to reduce CPU-to-GPU launch overhead.
- Fused MoE decode kernels equivalent to llama.cpp's SYCL MoE paths.
- Fused/fast GPTQ or AutoRound INT4 kernels on XPU.
- KV compression such as TurboQuant when context capacity, not speed, is the blocker.

## Immediate Next Experiments

1. Update current docs so they stop recommending stale q8 KV defaults and removed models.
2. Run a narrow vLLM evidence test only for `Qwen/Qwen3.6-35B-A3B` + online `sym_int4` + FP8 KV at 32K first, because that is the public comparable report.
3. If vLLM still lands near 20 tok/s single-stream, stop vLLM single-stream work and document it as concurrency-only until XPU graph patches are reproducible.
4. Build a Windows Vulkan validation checklist/script. This is the largest plausible single-card speed improvement.
5. Evaluate SGLang XPU only after the vLLM evidence test, because SGLang setup cost is higher and quant/model support is less proven for our target models.

## Current Recommendation

For the B70 desktop today:

- Production: llama.cpp SYCL/Level-Zero.
- Default quality model: Qwen3.6-35B-A3B Q5 at 256K, q5_0/q4_1 KV, 150W.
- Fastest installed model: Ornith 35B Q5 at 256K, q5_0/q4_1 KV, 150W.
- Dense/reasoning/vision: Qwen3.6-27B MTP at 165-180W sustained or 200W burst, with 200K as the validated high-context profile.
- Research lane with highest upside: Windows Vulkan Qwen35 Q4 validation.
- Research lane with highest engineering value: vLLM/SGLang XPU graph and fused INT4/MoE kernels.
