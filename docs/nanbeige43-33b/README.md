# Nanbeige4.3-33B — route chooser

Two routes are measured on this hardware, on one card and on two. Pick the first when you
want long context, the second when you want the fastest decode and can live inside ~45K
tokens, and the two-card rows when one card's capacity or latency binds.

| Route | Artifact | Status |
|---|---|---|
| [B70 llama.cpp SYCL, one card](NANBEIGE43-33B-B70.md#7-serve-it) | `bartowski/Nanbeige_Nanbeige4.3-33B-GGUF` rev `9beb6842`, `Q4_K_M`, 20,788,825,216 B | **measured, provisional**: 23.14 tok/s decode at empty context, 6.67 tok/s at 32K, 1175 tok/s pp2048, 143,360-token KV ceiling with `q4_1/q4_1` |
| [vLLM-XPU W4A16, one card](NANBEIGE43-33B-B70.md#8-second-route-vllm-xpu--w4a16-faster-less-context) | `jeonghyeon00/Nanbeige4.3-33B-W4A16-AutoRound` rev `d71b1250`, compressed-tensors W4A16 g128, 20.65 GB | **measured, provisional**: 28.87 tok/s decode (28.96 in generation 1), 27.7 tok/s at 6.2K, 46,848-token context at fp8 KV, exact-answer 6/8 strict / 7/8 meaning |
| [vLLM-XPU W4A16, two cards, TP2](NANBEIGE43-33B-B70.md#10-two-b70-cards--three-topologies-pick-by-what-binds) | same W4A16 pack, `--tensor-parallel-size 2` + affinity patch + oneCCL thresholds | **measured, provisional**: 41.01 tok/s at a 238-token prompt, 23.83 at 98K, 307,072-token KV ceiling, 390 W total |
| [Two independent vLLM servers](NANBEIGE43-33B-B70.md#10-two-b70-cards--three-topologies-pick-by-what-binds) | one W4A16 container per card, `ZE_AFFINITY_MASK=0` and `1` | **measured, provisional**: 55.22 tok/s aggregate while both cards serve, 0.135 tokens/J |
| [llama.cpp, two cards, `-sm layer`](NANBEIGE43-33B-B70.md#10-two-b70-cards--three-topologies-pick-by-what-binds) | same Q4_K_M GGUF | **measured, provisional**: capacity route, 262,144-token context served a 131,029-token prompt at 3.87 tok/s decode; decode is 3–4 % below one card |
| [vLLM TP2 + ngram speculation](NANBEIGE43-33B-B70.md#18-speed-levers-what-works-what-does-not-exist) | same W4A16 pack, `--speculative-config` ngram | **measured, provisional**: 30.31 tok/s at a 98K prompt (+27 % over TP2), 53.84 on copy-heavy output (+41 %), battery unchanged |

## Quality

Deterministic suite v2 (40 items: math, executed coding, JSON, instruction following,
4K-32K retrieval, tool calls, multilingual), same engine and flags for both models:
**Nanbeige4.3-33B 40/40, Qwen3.8-27B 40/40**. The suite saturates, so it is a floor, not a
ranking. On the 8-case exact battery the two differ in failure mode, not in count
(see [section 19](NANBEIGE43-33B-B70.md#19-quality-suite-v2--parity-on-40-mechanical-items)).

## Claim boundary

- Single stream per engine (C1) unless the row says aggregate. One artifact per route, one
  build (llama.cpp `1692f9e50`, `GGML_SYCL_F16=ON`) or one image
  (`vllm/vllm-openai-xpu:v0.31.0-95ac8150`), as-found caps 180 W (display card) and 230 W.
- Generation 1 (2026-10-25) measured one card on an earlier host; generation 2 (2026-10-28)
  re-measured both artifacts on a dual-card host. The vLLM cell reproduced within 0.3 %.
- No concurrency above one stream per engine, no KL/perplexity, no task suite beyond the
  8-case exact battery, no measured context above a 131K-token decode.
- Numbers come from one operator on one host class. Treat them as a starting point, then
  measure your own.

## Do not mix

- These cells with Qwen3.8-27B EXL3 numbers. That route runs 63.25 tok/s on one
  card with a 14 GB footprint; a 33B dense model cannot match its decode rate, and
  the comparison at the same prompt/output is not measured. The matched llama.cpp
  comparison (section 11) is the only cross-model table that shares engine, flags and host.
- GGUF numbers with vLLM W4A16 numbers: they are different engines, artifacts and
  context ceilings. Compare them only in the route table above, which names both.
- TP2 single-stream rates with the two-independent-server aggregate: one is latency, the
  other is throughput across two requests.
- **This model has no MTP heads.** Do not advertise MTP for it, and do not treat the ngram
  gains above as MTP gains; they come from draft-free verification against the prompt.
- `llama-bench` rates with client-observed rates.

## Related authorities

- Measurement rules: `docs/BENCHMARK-FORMAT.md`
- Image and patch compatibility: `docs/IMAGE-AND-PATCH-MATRIX.md`
- Operational status: `docs/RELIABILITY-REPORT.md`
