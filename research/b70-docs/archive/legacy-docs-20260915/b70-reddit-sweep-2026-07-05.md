# B70 Reddit/Forum Sweep - 2026-07-05

Scope: recent Reddit/forum claims about Intel Arc Pro B70 inference, with emphasis on Qwen3.6 27B/35B, llama.cpp, vLLM/LLM-scaler, Vulkan/SYCL, long-context prefill, and agent workloads.

## High-Signal Claims

| Source | Claim | Confidence | Relevance |
|---|---|---:|---|
| Reddit `Intel Arc Pro B70 + llama.cpp (Vulkan) benchmarks with Qwen3.6` | OP reports Qwen3.6-35B-A3B around 98.3 tok/s generation on B70, but commenters point out the prompt was only 18 tokens, making prompt-processing numbers invalid. OP later says best Ubuntu 26.04 vLLM result was 95 for Qwen3.6-35B-A3B. | Medium for decode, low for pp | Explains the 95-100 tok/s claim, but not an apples-to-apples server/256K/q5_0-q4_1 result. Source: https://www.reddit.com/r/LocalLLM/comments/1ukjz87/intel_arc_pro_b70_llamacpp_vulkan_benchmarks_with/ |
| Same Reddit thread comments | Micro Center video reportedly measured single B70 Qwen3.6-27B at 1346.9 pp and Qwen3.6-35B-A3B at 4175.6 pp. Another user claims almost 10k pp on qwen3.6-35b-a3b-int4. | Low until video/config verified | Indicates pp should be far above OP's tiny-prompt result, but config/engine/context unknown. |
| Same Reddit thread comments | CachyLlama/disk caching solved repeated system-prompt cost by caching system prompts after first processing pass. | Medium | Directly relevant to agent coding harnesses with repeated 10K+ system prompts. |
| Reddit `Qwen3.6-35B-A3B at 75 tokens per second on a single Arc Pro B70` | User reports Linux Vulkan ~44 tok/s, then rebuilt llama.cpp SYCL with Intel oneAPI 2026 and got sustained 75 tok/s on Q4_K_XL. Notes `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`, explicit Level-Zero device selection, and current Mesa/toolchain issues. | High | Matches our current production lane. Our local Qwen35 Q4 at ~65 tg/s is close but below this report; Ornith Q5 is ~72 tg/s. Source: https://www.reddit.com/r/IntelArc/comments/1u3l4zx/qwen3635ba3b_at_75_tokens_per_second_on_a_single/ |
| Reddit `ARC B70, Qwen3.6-27B-MTP-GGUF 24-28T/s, Qwen3.6-35B-A3B-GGUF 60-70T/s finally` | User reports self-built SYCL image improved stability/speed; Qwen27 MTP 24-28 t/s and Qwen35 60-70 t/s for Hermes/Claude Code. | High | Almost exactly matches our local llama.cpp numbers. Confirms our stack is in the credible range. Source: https://www.reddit.com/r/LocalLLM/comments/1u35s38/arc_b70_qwen3627bmtpgguf_2428ts_qwen3635ba3bgguf/ |
| Reddit `llama.cpp SYCL - 63 t/s on Qwen 3.6-35B-A3B` comments | User claims Intel scaler LLM gets ~125 t/s at 40K context and 95 t/s at 132K context on this model. | Medium-low | Potentially important vLLM/LLM-scaler lead, but exact model/quant/config not posted in snippet. Source: https://www.reddit.com/r/LocalLLM/comments/1tuf6l1/intel_arc_pro_b70_llamacpp_sycl_63_ts_on_qwen/ |
| Reddit `Intel Arc Pro B70 32GB performance on Qwen3.5-27B@Q4` | vLLM Qwen3.5-27B AutoRound single GPU: pp2048 ~1700 t/s, tg512 ~13.4 t/s; concurrency 32 aggregate tg ~130.9 t/s. Pipeline parallel improves high-concurrency aggregate but hurts single query. | High | Confirms vLLM is throughput/concurrency-oriented and weak for single-stream dense decode. Source: https://www.reddit.com/r/LocalLLaMA/comments/1siar7y/intel_arc_pro_b70_32gb_performance_on_qwen3527bq4/ |
| Reddit `Intel Arc Pro B70 (32GB) for Local LLMs...` comments | Linux Vulkan benchmark shows Qwen35 Q4 `pp512 1314.71`, `tg128 78.72`; commenter says Windows Vulkan can exceed 100 t/s in llama-bench but crashes during Claude Code every ~30 minutes or first prompt. Another says Linux Mesa Vulkan is stable but ~75% of Windows performance. | High for stability warning | Windows Vulkan speed is not useful if real agent sessions crash. Linux Vulkan may be a stable fallback but not clearly better than SYCL. Source: https://www.reddit.com/r/IntelArc/comments/1tr9397/intel_arc_pro_b70_32gb_for_local_llms_llamacpp/ |
| Reddit `Intel Arc B70 with HP z640 workstation` | PCIe 3 workstation, Qwen35 Q4 at 130K context, q8 KV, pp on real 78,720-token prompt ~282.6 t/s and decode ~11.8 t/s; later llama-bench with `-b 4096 -ub 4096` got pp4096 ~611.6 and tg512 ~23.7. | Medium | Shows long-context/older-build/power-platform can collapse performance; not a target, but validates need for long-context prefill testing. Source: https://www.reddit.com/r/LocalLLaMA/comments/1spgkn5/intel_arc_b70_with_hp_z640_workstation_pcie_3/ |
| r/IntelArc new listing | User describes OVMS/OpenVINO Model Server with pre-exported OpenVINO Qwen3 INT4 on Arc; claims results much faster than GGUF with SYCL/Vulkan for their use. | Low until opened/benchmarked | New lane: OpenVINO/OVMS may be worth scouting for Intel-native Windows/server inference, especially smaller Qwen models. Source: https://www.reddit.com/r/IntelArc/new/ |
| Reddit `INTEL ARC PRO B70 ORNITH-1.0 with ovms results` | Single B70, OVMS, `Ornith-1.0-35B-AEON-Ultimate-Uncensored-BF16-int4-ov`, 65K context: `pp2048 1920.52 ± 21.98`, `tg32 88.84 ± 0.51`, peak tg ~91.7. User says faster than GGUF SYCL/Vulkan. A requested Galaga one-shot prompt failed to produce a final HTML file. | Medium-high for speed, low-medium for quality/tool usefulness | This is the strongest new lead because it beats our local Ornith GGUF Q5 ~72 tg/s. It must be validated for quality, chat templates, tools, and long-context behavior. Source: https://www.reddit.com/r/LocalLLM/comments/1unc8de/intel_arc_pro_b70_ornith10_with_ovms_results/ |
| Hugging Face `Wondernutts/Ornith-1.0-35B-AEON-Ultimate-Uncensored-BF16-int4-ov` | Model card claims OpenVINO INT4, stock OpenVINO 2026.2 runtime, no patches, coherent through 32K context, and 106 tok/s on one B70. | Medium | Confirms this is a purpose-built OpenVINO model, not a GGUF backend tweak. Source: https://huggingface.co/Wondernutts/Ornith-1.0-35B-AEON-Ultimate-Uncensored-BF16-int4-ov |
| OpenVINO Model Server docs/releases | OVMS is a C++ server optimized for Intel architectures with OpenAI-like generative APIs; 2026.2.1 includes GPU image and `cache_interval_multiplier` for Qwen3.6-35B-A3B long prompts. Docs explicitly list Qwen3.6-35B-A3B on B70 dGPU and command flags for `--reasoning_parser qwen3 --tool_parser qwen3coder --target_device GPU`. | High | OVMS is now a first-class evidence-backed lane for B70. Source: https://github.com/openvinotoolkit/model_server/releases and https://docs.openvino.ai/nightly/model-server/ovms_demos_continuous_batching_agent.html |
| Reddit `July 2026; where are Intel's GPU speeds today at?` | A user says SGLang is their production choice because vLLM has a tracked concurrency bug where overlapping prefill/decode from different requests can make one stream produce garbage tokens. They report SGLang as rock solid, ~140W load, Linux kernel 7.0+ required for GPU P2P, and ~25 tok/s decode / 3700 tok/s prefill at 8-bit quant, but did not identify the model in visible comments. Another user says Intel's best performance is via LLM Scaler on supported models; another says native OVMS/OpenVINO is competitive while llama.cpp is weaker. | Medium | Adds two actionable cautions: do not use vLLM for concurrent serving until overlap bug is reproduced/fixed; SGLang deserves a stability test for production serving; OVMS remains a top native lane. Source: https://www.reddit.com/r/LocalLLaMA/comments/1ukq7ez/july_2026_where_are_intels_gpu_speeds_today_at/ |
| Reddit `Possible? Arc B50 - multi-GPU Battlematrix for local LLM` | Users say B50 multi-GPU is possible and works similarly to B60/B70 but slower/less VRAM. Hardware notes: motherboard with enough PCIe slots, ReBAR recommended but not always hard-required, SR-IOV useful for Proxmox virtualization but not needed for local AI, VRAM scaling is software/runtime-level rather than hardware pooling. A dual B50 user reports Ubuntu 26.04, W680 ACE, dual x8 PCIe 5.0, 96GB DDR5, ReBAR enabled, llama.cpp on Qwen3.6 35B MoE, guessed 25-30 t/s, handles 128K-256K with slowdown/spill. | Medium | Confirms for our B70: do not think of Battlematrix as free combined VRAM; multi-GPU success depends on runtime, topology, and workload. Also reinforces Ubuntu 26.04 + llama.cpp/SYCL as a practical Intel Arc path. Source: https://www.reddit.com/r/IntelArc/comments/1uiqky6/possible_arc_b50_multigpu_battlematrix_for_local/ |

## Patterns

1. Qwen35 MoE is the B70 sweet spot.
   Multiple users independently report Qwen3.6-35B-A3B in the 60-75 tok/s range on Linux SYCL and sometimes 78-100+ tok/s in Vulkan/Windows or benchmark-only paths.

2. Prompt processing claims are noisy unless prompt length is large.
   Tiny prompts produce misleading pp. The useful comments explicitly call for thousands of input tokens. Agent harnesses need 8K, 32K, 80K, 128K, and 256K prefill tests.

3. vLLM/LLM-scaler is not automatically a single-stream win.
   Public Qwen27 vLLM data shows weak single-stream decode but good aggregate throughput at high concurrency. The 95-130 tok/s Qwen35 LLM-scaler claims need exact reproduction before they should drive production changes.

4. Windows Vulkan may be fast but unstable.
   It is high-upside for llama-bench numbers, but at least one user reports crashes in Claude Code real work. Linux Mesa Vulkan appears more stable but slower.

5. OpenVINO/OVMS is no longer a low-priority curiosity.
   The latest Ornith INT4-OV claim is directly competitive with our fastest local model and comes from a model card plus official OVMS support path. It deserves a narrow reproduction test before more vLLM/SGLang work.

6. vLLM has a concurrency caveat.
   Recent Reddit reports say vLLM can be fine for single streams but corrupt output when prefill/decode overlap under concurrency. This aligns with our decision not to adopt vLLM just because it loads; it must pass concurrent serving and quality tests.

7. SGLang's role is stability/serving, not raw speed until proven.
   A recent user says SGLang is production-stable where vLLM had concurrency issues. That makes SGLang worth testing after OVMS, especially for multi-client serving.

8. Caching matters for agent workflows.
   Repeated large system prompts can dominate perceived latency. CachyLlama/disk prompt cache deserves a practical test because it can improve workflow latency without changing GPU kernels.

6. Our local stack is not obviously behind the field.
   Local Qwen35 Q4 ~65 tg/s and Ornith Q5 ~72 tg/s are aligned with credible Linux SYCL reports. The gaps worth chasing are long-context prefill, caching, and specific vLLM/OpenVINO claims.

## Actionable Next Steps

### A. Build a long-context prefill benchmark first

Purpose: answer the real complaint from agent users: large system prompts and context growth.

Test matrix:

- Models: Qwen35 Q5, Qwen35 Q4, Ornith Q5, Qwen27 MTP.
- Context/prompt lengths: 8K, 32K, 80K, 128K, 200K/256K where model fits.
- Metrics: pp t/s, TTFT, tg t/s for a 256-token continuation, VRAM, power, temperature.
- Keep llama.cpp q5_0/q4_1 KV as default.

Decision:

- If pp collapses after 32K like Reddit users report, prioritize vLLM/LLM-scaler or caching lanes.
- If pp stays healthy, current llama.cpp stack remains production.

### B. Reproduce the OVMS Ornith/Qwen35 claim

This is now the highest-priority non-llama.cpp test.

Candidate models:

- `Wondernutts/Ornith-1.0-35B-AEON-Ultimate-Uncensored-BF16-int4-ov`
- `OpenVINO/Qwen3.6-35B-A3B-int4-ov`

Runtime:

- `openvino/model_server:2026.2.1-gpu` or OVMS 2026.2.1 bare-metal package.
- Target device: GPU.
- Context targets: 32K first, then 65K, then 128K if memory permits.
- Use `cache_interval_multiplier=64` for long prompts per OVMS release guidance.

Metrics:

- pp2048/tg32 to compare against Reddit.
- Our normal 256-token server prompts for user-experience comparison.
- Long-context prefill at 8K/32K/65K/128K.
- Tool/template sanity: system prompt support, reasoning parser, tool parser, OpenAI-compatible endpoint behavior.
- Quality gate: reproduce the Galaga prompt or a smaller deterministic coding task; the Reddit author reports that Galaga failed despite high speed.

Decision:

- If OVMS reaches >85 tg/s and passes quality/tool-template checks, add it as a serious production candidate for Ornith/Qwen35.
- If it is fast but fails agent/tool outputs, keep it as chat-only or benchmark-only.

### C. Reproduce the LLM-scaler Qwen35 claim narrowly

Do not sweep all vLLM. Try only the claim:

- `Qwen/Qwen3.6-35B-A3B` or the exact INT4/AutoRound/GPTQ model that fits.
- 40K and 132K context equivalents.
- FP8 KV if supported.
- Measure single-stream and concurrency 4/8/16.

Decision:

- Continue vLLM only if it beats llama.cpp on either long-context prefill or aggregate throughput.
- Stop if it stays near the local Qwen27 AutoRound result (~18 output tok/s single stream).

### D. Validate caching for agent workloads

Test CachyLlama or equivalent prompt/KV/system-prompt cache:

- Use an OpenCode/Pi/Hermes-like 10K system prompt.
- Run identical session twice.
- Measure first-run prefill, second-run load/cache hit, TTFT, and correctness.

Decision:

- If cache hit reduces repeated prompt cost materially, this is a workflow win independent of engine speed.

### E. Treat Windows Vulkan as a stability-gated experiment

Only worth adopting if:

- It runs real agent workloads for at least 1-2 hours without crash.
- It beats Linux SYCL server generation by >20% with matched prompts/context.
- It can support the needed context/KV behavior.

Otherwise it remains a benchmark curiosity.

## Updated Priority Order

1. Long-context prefill benchmark for current llama.cpp stack.
2. OVMS Ornith/Qwen35 INT4-OV reproduction.
3. Prompt/system cache validation for real agent prompts.
4. Narrow LLM-scaler Qwen35 reproduction at 40K/132K.
5. Windows Vulkan stability/speed validation.
6. SGLang XPU graph scouting.
