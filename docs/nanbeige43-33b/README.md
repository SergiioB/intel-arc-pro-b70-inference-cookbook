# Nanbeige4.3-33B — route chooser

Two routes are measured on this hardware. Pick the first when you want long context,
the second when you want the fastest decode and can live inside ~45K tokens.

| Route | Artifact | Status |
|---|---|---|
| [B70 llama.cpp SYCL, one card](NANBEIGE43-33B-B70.md#7-serve-it) | `bartowski/Nanbeige_Nanbeige4.3-33B-GGUF` rev `9beb6842`, `Q4_K_M`, 20,788,825,216 B | **measured, provisional**: 23.14 tok/s decode at empty context, 6.67 tok/s at 32K, 1175 tok/s pp2048, 143,360-token KV ceiling with `q4_1/q4_1` |
| [vLLM-XPU W4A16, one card](NANBEIGE43-33B-B70.md#8-second-route-vllm-xpu--w4a16-faster-less-context) | `jeonghyeon00/Nanbeige4.3-33B-W4A16-AutoRound` rev `d71b1250`, compressed-tensors W4A16 g128, 20.65 GB | **measured, provisional**: 28.96 tok/s decode, 23.47 tok/s on a ~1.2K prompt, 46,848-token context at fp8 KV, exact-answer 5/5 |

## Claim boundary

- One card, single stream (C1), one artifact per route, one build (llama.cpp `1692f9e50`,
  `GGML_SYCL_F16=ON`) or one image (`vllm/vllm-openai-xpu:v0.31.0-95ac8150`), 230 W
  configured cap. No concurrency, no measured context above 32K decode, no
  benchmark-suite quality result, no two-card placement.
- Numbers come from one host on 2026-10-25 and are not reproduced by a second
  operator. Treat them as a starting point, then measure your own.
- This family is **not** in `data/benchmarks.v1.json`; do not cite it in rankings
  until it is, with evidence pinned to a commit.

## Do not mix

- These cells with Qwen3.8-27B EXL3 numbers. That route runs 63.25 tok/s on one
  card with a 14 GB footprint; a 33B dense model cannot match its decode rate, and
  the comparison at the same prompt/output is not measured.
- GGUF numbers with vLLM W4A16 numbers: they are different engines, artifacts and
  context ceilings. Compare them only in the route table above, which names both.
- `llama-bench` rates with client-observed rates.

## Related authorities

- Measurement rules: `docs/BENCHMARK-FORMAT.md`
- Image and patch compatibility: `docs/IMAGE-AND-PATCH-MATRIX.md`
- Operational status: `docs/RELIABILITY-REPORT.md`
