# OpenVINO GenAI and Cascadia

What these two Intel-side stacks are, how they differ, and what they
offer relative to the cookbook's production engines (vLLM XPU and
llama.cpp SYCL). This page is architecture and claim-boundary, not a
speed leaderboard.

Qwen3.8-27B measurements are on
[docs/qwen38-27/OPENVINO-CASCADIA.md](../qwen38-27/OPENVINO-CASCADIA.md).
Do not copy them into catalog rankings. Hub
`OpenVINO/Qwen3.8-27B-int4-ov` still collapses; the working mixed INT4 is
[`SergiioB/Qwen3.8-27B-int4-gdn8-ov`](https://huggingface.co/SergiioB/Qwen3.8-27B-int4-gdn8-ov).

## OpenVINO GenAI

OpenVINO is Intel's inference toolkit. Models are exported to **OpenVINO
IR** (`*.xml` + `*.bin`), not GGUF. **OpenVINO GenAI** is the generative
API on top of that IR:

- `LLMPipeline` — text graphs that take `input_ids`
- `VLMPipeline` — Qwen3.5/3.8-style exports (`openvino_language_model.xml`
  plus tokenizer IRs). Text-only still goes through VLM layout because
  the language graph takes `inputs_embeds`, mRoPE `position_ids` of shape
  `[4, B, T]`, and `beam_idx`.
- `ContinuousBatchingPipeline` — paged attention. On Qwen3.8 this is
  also the path that owns the **optimized GatedDeltaNet kernel**. The
  stateful ReadValue/Assign path uses the reference GDN kernel.

Serving wrappers: **OVMS** (OpenVINO Model Server) for HTTP, or a local
Python/C++ GenAI process.

What it offers vs llama.cpp / vLLM on Battlemage:

- Official Intel IR + NNCF INT4, including prebuilt
  `OpenVINO/<model>-int4-ov` repos
- Plugin JIT cache (`CACHE_DIR` / `--ov-cache-dir`) so cold compile is
  paid once
- FastDraft speculative decode and prompt-lookup on `LLMPipeline`
- Paged-attention continuous batching for concurrent short prompts
- OVMS already measured on this host for Qwen3.6-35B-A3B INT4 (decode
  behind llama.cpp SYCL; cooler; less VRAM)

What it does not replace:

- vLLM XPU MTP on GPTQ-INT4 remains the dense-27B speed champion in this
  cookbook
- llama.cpp SYCL remains the simple GGUF production path

## Cascadia

[Cascadia](https://github.com/labscommunity/cascadia) is Community Labs'
Rust runtime for Intel silicon. One static binary per node, OpenAI-compatible
HTTP, optional pipeline-parallel shards across machines.

It does **not** replace OpenVINO. Most engines **call** OpenVINO:

| Engine | Input | Role |
|---|---|---|
| `ov-genai` | Whole-model IR | Thin FFI over `LLMPipeline` / `VLMPipeline`. FastDraft, prompt-lookup, `--cb` |
| `ov-runtime` | `cascadia shard` tree | Multi-stage stateful KV, pipeline parallel |
| `qwen35` (alias `qwen36-moe`) | IR-surgery shards | Qwen3.5/3.6 MoE and **dense Qwen3.8**. Greedy, batch=1. Prefix cache |
| `ov-dist-spec` | v5 shards | Distributed speculative decode |
| `gemma4` / `sparse-moe` | Family-specific trees | Not Qwen3.8 |

What it offers:

- One binary, no Python on workers (export/shard still needs Python)
- Pipeline parallel across Intel boxes (`--rank` / `--total` / `--next`)
- `cascadia doctor` catches silent CPU fallback (OpenVINO not seeing the GPU)
- Qwen3.8-specific `qwen35` surgery: split the official INT4 IR on layer
  boundaries without re-quantizing
- FastDraft / prompt-lookup / `--cb` on the `ov-genai` engine

What it does not offer on one B70 for Qwen3.8-27B:

- A working `ov-genai` load of this VLM IR (stock 0.2.3 = OpenVINO 2026.2
  SIGSEGV; 2026.3.1 rebuild still dies in `Tokenizer::setup_tokenizer`)
- Tok/s matching Python `VLMPipeline` (15.79 vs 4.95 at 230 W P512/G128).
  `qwen35` is a host-side greedy loop around a surgery IR, not fused GenAI
- A paged-attention GDN decode that fits 32 GB with a ~21 GB mixed-INT4 graph

## How they differ

- **Artifact:** OpenVINO GenAI and Cascadia consume IR. llama.cpp consumes
  GGUF. vLLM XPU consumes GPTQ/FP8 checkpoints. Do not mix tok/s across
  those artifacts.
- **Process:** GenAI Python is a library in-process. Cascadia is a Rust
  server wrapping the C++ GenAI API (or the qwen35 surgery runtime).
- **Batching:** `--cb` on Cascadia `ov-genai` is GenAI continuous batching.
  `--cb` costs 8–15% at concurrency 1 and wins at ≥4 short concurrent
  requests. Stock 0.2.3 `qwen35` ignored `--ov-*`. A local rebuild
  forwards `LATENCY` + `f16`; that did **not** change tok/s on dense 27B.
- **Speculation:** Cascadia FastDraft is an IR companion, not Qwen MTP.
  Qwen3.8 MTP heads in the Intel IR are unused on both GenAI 2026.5
  stateful and Cascadia 0.2.3 `qwen35`.
- **Version skew:** Python GenAI 2026.5 nightly can load the Intel IR.
  Cascadia 0.2.3's bundled 2026.2 cannot (crash). `convert_tokenizer`
  is required for Intel-published tokenizer IRs on 2026.2/2026.3; it does
  not prevent the 2026.2 VLM SIGSEGV.

## Flag map (tested 2026-09-11)

Cascadia `ov-genai` / `worker` (did not complete load on this IR):

- `--ov-inference-precision f16` — set on Xe2; default can fall to f32
- `--ov-cache-dir` — JIT blob cache; on by default under `~/.cache/cascadia/ov-cache`
- `--ov-performance-mode LATENCY | THROUGHPUT | CUMULATIVE_THROUGHPUT`
- `--ov-kv-precision u8 | f16`
- `--draft-model` / `--spec-k` / `--prompt-lookup` — mutually exclusive; no Qwen3.8 FastDraft IR on disk
- `--cb` — do not enable for single-user decode

Cascadia `qwen35` (loaded, greedy, batch=1):

- Stock 0.2.3: `--ov-*` ignored
- Local rebuild: `--ov-performance-mode LATENCY --ov-inference-precision f16`
  apply; P512/G128 still 4.95 tok/s at ~102 W
- `--prefix-cache-gb` applies

OpenVINO GenAI Python: splat constructor kwargs (`performance_mode`,
`cache_dir`, `scheduler_config`). `VLMPipeline.generate(prompt, cfg)`
treats `cfg` as **images**; pass `generation_config=`. Qwen3.8 chat
jinja uses `enable_thinking is undefined`, which Minja rejects — disable
`apply_chat_template` or wrap `<|im_start|>` yourself.

## When to use which stack on B70

The Reddit post [arc_pro_b70_bench_test_to_compare_ovms_vs_vllm](https://www.reddit.com/r/LocalLLM/comments/1uzqc9d/arc_pro_b70_bench_test_to_compare_ovms_vs_vllm) and the follow-up [b70_ovms_vs_vllm_vs_vllmmtp](https://www.reddit.com/r/LocalLLM/comments/1ub1lse/b70_ovms_vs_vllm_vs_vllmmtp) demonstrated high OpenVINO GenAI throughput (37 t/s on 27B, 101 t/s on 35B). 

**The catch:** those numbers were generated on **Qwen3.6** (standard attention), measuring **Continuous Batching (multi-user concurrency)** where the GPU is saturated by serving 10+ users simultaneously. 

For **Qwen3.8-27B** (GatedDeltaNet) at **single-user latency** (greedy, batch=1):

- **Use vLLM XPU + MTP:** (69 t/s) — Absolute fastest single-user decode, provided you have enough VRAM for the dense model and apply the FP8 KV cache patch.
- **Use llama.cpp SYCL:** (18.5 t/s) — Lowest VRAM footprint, highest quantized compatibility, runs on bare metal with zero dependencies.
- **Use OpenVINO GenAI / Cascadia:** (15.8 t/s) — Use only for model architectures without GDN (like Qwen3.6) where OpenVINO has fused INT4 kernels, or when you are building a multi-user API endpoint relying on `--cb` (Paged Attention). Avoid for Qwen3.8 until the Xe2 GDN kernel is fixed.
