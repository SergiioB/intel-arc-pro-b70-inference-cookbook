# Campaign Log — the 19-run investigation

The full narrative of how we got from "vLLM is 7× slower than llama.cpp"
(Run 14) to "vLLM MTP beats llama.cpp on MoE" (Run 18) to a complete measured
map (Run 19). Each entry is a real benchmark run.

## TL;DR arc

| Run | Stack | Decode | Prefill | Lesson |
|-----|-------|-------:|--------:|--------|
| 13–14 | vLLM 0.17 MXFP4 (7 patches) | 10.4 t/s | 1,738 | Old image; vLLM was 7× slower than llama.cpp here |
| 15 | Concurrency head-to-head | — | — | "150 t/s" claim = multi-user aggregate (C16=153), not single-stream |
| 16 | vLLM 0.21 Triton GPTQ MoE | 57.9 | ~5.3K | Triton path leaves ~40% on the table |
| 17 | **Native XpuFusedMoe int4** (dtype fix) | **72.6** | **9,094** | Prefill beats Reddit 7,975; decode = bandwidth ceiling |
| 18 | **MTP speculative unlocked** (4 patches) | **123** | 7,261 | First vLLM XPU result to beat llama.cpp MoE parity |
| **19** | **Full engine + power sweep** | **126** | **6,217** | vLLM MTP 1.8×/4.2× over llama.cpp; MoE=150W/Dense=180W sweet spots |

## The two breakthroughs

### Run 17: the int4 dtype bug

The native int4 MoE path was crashing with `ptr_A.size(1) must match ptr_B.size(1)`.
Root cause: the C++ int4 detector is `is_B_int4 = (B_dtype == at::kChar)` where
`at::kChar` is `torch.int8`. GPTQ packs weights as `uint8`. So the kernel saw
"not int4", treated B as BF16 `[E, K, N]`, and the shape check failed.

Fix: `implement_zp` → store `torch.int8` (not `uint8`). One-bit dtype bug →
+25% over Triton, prefill beats the Reddit headline.

### Run 18: MTP "impossible" was an overcautious assert

The hybrid GDN model (linear attention + full attention) has an XPU GDN kernel
with `assert attn_metadata.spec_sequence_masks is None`. Run 14 (ngram) hit
this and we wrote it up as "XPU GDN incompatible with speculative decoding."

That was **wrong**. The kernel already receives `num_spec_decodes`,
`spec_query_start_loc`, `spec_token_indx`, `spec_state_indices_tensor` — the
boolean `spec_sequence_masks` is metadata-only and is never passed to the SYCL
kernel. The assert was a guardrail, not a capability limit.

Combined with three load-path patches (BF16 draft, kwarg strip), MTP runs →
**123 t/s single-stream**.

## Methodology (why the numbers are trustworthy)

- **vLLM decode**: streaming `/v1/chat/completions` with
  `stream_options.include_usage`. Decode = `completion_tokens / (total - ttft)`.
  This avoids the `reasoning_content` undercount trap (reasoning models emit
  reasoning tokens separately from `content`).
- **llama.cpp decode**: `/completion` endpoint, `timings.predicted_per_second`
  (the engine rate, not wall-clock; see [BENCHMARK-FORMAT.md](BENCHMARK-FORMAT.md)).
- **Best steady-state rep**: drop JIT warmup (first rep), report best of
  remaining. 2 reps/cell.
- **Thermal discipline**: cooldown to ≤52°C between runs. No two inference
  processes concurrent (VRAM contention = invalid data).
- **Correctness** (Run 18): greedy `temp=0` replays byte-identical across runs
  (a corrupting spec path would diverge); factual probes (17×23, Canberra)
  correct.

## Glossary

- **C1 / Concurrent-1**: one in-flight request (single-stream). Decode = that
  stream's tokens/s.
- **C16 wall-agg**: 16 concurrent clients, wall-clock aggregate tokens/s =
  (sum completion tokens across all 16 users) / (wall time first-send →
  last-finish). Multi-user throughput, *not* per-user decode.
- **tg32 / pp2048**: genai-bench / llama-bench style — generate 32 tokens, or
  prefill a ~2048-token prompt.

## Open questions

- **Dolboyob77 exact stack/URL**: the Reddit user who claimed Concurrent-1
  145 t/s. Their exact measurement/stack remains unrecovered (PullPush miss).
  Our best single-stream is 126 t/s (85% of 145); the gap is likely the
  single-layer MTP ceiling (their `num_spec=2` may have hit).
- **KL/acceptance audit** of the MTP path vs eager — the correctness gate
  before production use.
- **Dense FP8 XPU kernel** — see [DENSE-FP8-GAP.md](qwen36-27/DENSE-FP8-GAP.md).

## Full data

This repo carries the patches, harnesses, and the headline results.
