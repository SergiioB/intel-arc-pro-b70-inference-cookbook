# B70 Optimization Sweep Plan

Last updated: 2026-07-05

## Objective

Find the best practical inference stack for the Intel Arc Pro B70 by comparing
validated llama.cpp SYCL results against vLLM XPU / Intel LLM Scaler and SGLang
XPU where the model format, runtime, and hardware support make the test valid.

Approach update: do not exhaustively test every backend in the hope of finding a
win. Use public evidence and local failure data to choose narrow candidates with
a plausible improvement mechanism, then validate only those candidates.

This document is the canonical work plan for the next sweep. Historical raw
benchmark files remain useful as evidence, but this file owns current decisions,
candidate rejection rules, and the document-reduction plan.

## Current Ground Truth

Host:
- Intel Arc Pro B70, 32 GB VRAM, Ubuntu 26.04.
- Active production backend: llama.cpp SYCL, `build-sycl-latest-l0`.
- Local production models are GGUF under `/mnt/<models-disk>/gguf`.
- Docker is installed. Intel LLM Scaler image
  `intel/llm-scaler-vllm:0.14.0-b8.3.2` is present and has been smoke-tested.
- vLLM Qwen 27B AutoRound requires `--dtype float16`; the default BF16 path
  crashed in XPU GDN attention. At 150W it measured roughly 18.3 output tok/s,
  which is below llama.cpp Qwen 27B MTP.
- SGLang is not installed. Its XPU path is still a research candidate, mainly
  for opt-in XPU graph behavior.

Installed model files:

| Model file | Role | Sweep status |
|---|---|---|
| `Qwen3.6-35B-A3B-UD-Q5_K_M.gguf` | best-quality MoE | llama.cpp candidate |
| `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` | faster Qwen MoE | llama.cpp candidate |
| `Qwen3.6-27B-MTP-Q5_K_M.gguf` | dense MTP and vision | llama.cpp candidate; 256K rejected |
| `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` | vision MoE | llama.cpp candidate |
| `ornith-1.0-35b-Q5_K_M.gguf` | MoE, Q5 installed quant | llama.cpp measured; fastest installed local model |
| `dflash-draft-3.6-q8_0.gguf` | draft model | research candidate only |
| `mmproj-*` | vision projectors | paired with Qwen/Gemma |

Removed or absent models must not be included as live candidates unless restored:
Agents-A1, Ornith Q4, Ornith 9B, non-MTP Qwen 27B.

## Immediate Fixes Already Applied

- Corrected Qwen 27B production contexts away from unsafe 256K profile values.
- Mapped MoE profiles to 150W defaults instead of overdriving them at 230W.
- Added per-profile `powerWatts` so labels and launched wattage can agree.
- Added `-t 8` insertion in both interactive and systemd launch paths.
- Made systemd and interactive launch environments closer.
- Removed `GGML_SYCL_DISABLE_OPT` from active launcher defaults pending an A/B
  confirmation, because local B70 docs recorded a large decode regression.
- Replaced stale catalog default base URL with `<lab-host>`.
- Replaced `rocm-smi` status output with Intel hwmon/debugfs telemetry.

## Candidate Engine Matrix

| Engine | Current viability | Reason |
|---|---|---|
| llama.cpp SYCL | production baseline | Installed, supports GGUF, MTP, mmproj, q5_0/q4_1 KV, long context; local single-card winner so far |
| Windows llama.cpp Vulkan | highest-upside validation lane | Recent B70 users report ~98-100 tg/s Qwen35 Q4; must be validated with matching context and server requests |
| Intel LLM Scaler vLLM | concurrency / graph research lane | Stock eager-mode single-stream is slower locally; public wins require XPU graph, TurboQuant, or patched kernels |
| vanilla vLLM XPU | secondary research lane | Official XPU path exists, but requires Python 3.12, `VLLM_TARGET_DEVICE=xpu`, `triton-xpu`, and non-GGUF weights |
| SGLang XPU | XPU graph research lane | XPU path exists with opt-in decode/prefill graph; speculative decoding is not implemented on XPU |
| custom Triton/Xe kernels | bottleneck-driven only | Target XPU graph, fused MoE, GPTQ/AutoRound INT4, or KV compression after a profiler identifies the bottleneck |

Primary public sources:
- Intel LLM Scaler: https://github.com/intel/llm-scaler
- vLLM XPU install: https://docs.vllm.ai/en/latest/getting_started/installation/gpu/
- SGLang XPU: https://docs.sglang.io/docs/hardware-platforms/xpu
- Triton XPU backend: https://github.com/intel/intel-xpu-backend-for-triton
- AutoRound: https://github.com/intel/auto-round

Version constraints to honor before engine setup:
- Intel LLM Scaler: start with `intel/llm-scaler-vllm:0.14.0-b8.3.2`
  or the newest exact tag documented by Intel, because current Intel notes mention
  Qwen3.5/3.6-27B accuracy fixes after the original B70 image.
- vLLM XPU: Python 3.12, `requirements/xpu.txt`, replace plain `triton` with
  `triton-xpu==3.7.1` for the torch version in vLLM's XPU requirements, and build
  with `VLLM_TARGET_DEVICE=xpu`.
- SGLang XPU: Python 3.12 source install, XPU PyTorch wheels, `xgrammar --no-deps`,
  launch with `--device xpu --attention-backend intel_xpu`, and test page sizes
  32/64/128.
- AutoRound: usable for future quant production, but speed claims require quality
  validation against the current Q5_K_M/Q4_K_XL llama.cpp baselines.

## Rejection Rules

A candidate is rejected before benchmarking if any of these are true:

- The model file is missing.
- The engine cannot load the model format without conversion.
- The requested context exceeds the validated VRAM budget.
- The engine lacks the feature under test, such as MTP or vision.
- The setup would overwrite the known-good llama.cpp stack.
- The run cannot measure at least TTFT, decode rate, power, and failure mode.
- The candidate has no public evidence, official feature support, or local
  bottleneck hypothesis explaining how it could improve on llama.cpp SYCL.

For the user's requested 256K sweep, Qwen 27B MTP is a deliberate rejection at
256K on llama.cpp because local VRAM budgeting marks it unsafe. It must be tested
at 128K submitted context and 200K validated ceiling instead.

## Measurement Standard

For every valid run:

- Power caps: 150W, 165W, 180W, 200W.
- Context: 256K for models that fit; rejected/alternate context recorded for
  models that do not.
- One warmup request discarded.
- Four diverse prompts, rounds 2-4 averaged.
- Capture:
  - engine and version
  - model path / HF id / quant
  - context
  - KV cache type
  - power cap
  - TTFT
  - prompt processing rate (`pp`)
  - generation/decode rate (`tg`)
  - wall tokens/sec
  - average GPU card power from energy counters
  - peak temperature
  - VRAM usage
  - crash or quality anomaly

## Execution Phases

1. Stabilize local truth.
   Fix launcher/profile drift, regenerate the catalog, and mark stale documents
   as historical rather than canonical.

2. Keep llama.cpp baseline current.
   Use the installed GGUF models and `scripts/b70-engine-sweep.py` to collect
   comparable server-timing metrics. MoE models only need 150W unless the build
   or kernel path changes; dense Qwen 27B still merits 150W-200W sweeps.

3. Validate the high-upside Windows Vulkan claim.
   This is the largest plausible single-card speed delta. Test Qwen35 Q4 with
   matched prompts/context before investing more effort in Python engines.

4. Run only evidence-backed vLLM tests.
   Use Intel LLM Scaler in Docker. First test `Qwen/Qwen3.6-35B-A3B` with online
   `sym_int4` and FP8 KV at 32K because that matches a public single-B70 report.
   Continue only if it proves a concurrency use case or reaches >50 tok/s.

5. Evaluate SGLang XPU only for graph behavior.
   Build/install only after vLLM evidence is exhausted. Compare no graph, decode
   graph, and torch.compile paths on a small model first.

6. Run full sweeps only for finalists.
   Run only one server at a time. Restore 150W and stop servers after each run.
   Full 150W-200W power sweeps are reserved for configs that pass smoke tests and
   have a credible chance to beat a current winner.

7. Compare and decide.
   Pick winners by use case: best quality, fastest decode, lowest TTFT, best
   vision, best long-context, and best efficiency.

## Document Reduction Plan

Canonical documents after cleanup:

| Keep | Purpose |
|---|---|
| `docs/AGENTS.md` | operational single source of truth |
| `docs/recommended-configs.md` | current production launch table |
| `docs/b70-optimization-sweep-plan.md` | active research and sweep plan |
| `docs/benchmark-guide.md` | summarized benchmark narrative |
| `research/*` | evidence archive and analysis |

Historical documents to mark as archived or fold into the above:

- `raw-results-master-v2.md`
- `mtp-power-sweep-raw-results.md`
- `verification-results-2026-07-01.md`
- older single-purpose result docs once their data is linked from the benchmark guide

Rule: raw data stays, but only one document should tell an operator what to run
today.

## Open Risks

- vLLM is installed via Docker, but stock eager-mode single-stream performance is
  lower than llama.cpp on the local 27B test.
- SGLang testing requires a source/Python 3.12 environment or custom container.
- Ubuntu 26.04 is newer than some public validated PyTorch XPU OS notes, so XPU
  Python stacks may expose driver/runtime edge cases.
- Intel XPU performance claims are mostly multi-user vLLM claims, not direct
  single-stream GGUF comparisons.
- Windows Vulkan reports are high-upside but not yet apples-to-apples with local
  q5_0/q4_1 256K server results.
- Custom kernels are not actionable until a profiler identifies the bottleneck in
  a working engine path.
