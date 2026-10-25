# Nanbeige4.3-33B — route chooser

One route is measured on this hardware: dense 33B, Q4_K_M GGUF, llama.cpp SYCL on a
single Arc Pro B70. Pick it when you want a locally-served agent model around 20
tok/s with a fast prefill, and accept that decode falls steeply with context.

| Route | Artifact | Status |
|---|---|---|
| [B70 llama.cpp SYCL, one card](NANBEIGE43-33B-B70.md) | `bartowski/Nanbeige_Nanbeige4.3-33B-GGUF` rev `9beb6842`, `Q4_K_M`, 20,788,825,216 B | **measured, provisional**: 23.14 tok/s decode at empty context, 6.67 tok/s at 32K, 1,175 tok/s pp2048 |

## Claim boundary

- One card, single stream (C1), one artifact, one build (llama.cpp `1692f9e50`,
  `GGML_SYCL_F16=ON`), 230 W configured cap. No concurrency, no context above 32K,
  no benchmark-suite quality result, no two-card placement.
- Numbers come from one host on 2026-10-25 and are not reproduced by a second
  operator. Treat them as a starting point, then measure your own.
- This family is **not** in `data/benchmarks.v1.json`; do not cite it in rankings
  until it is, with evidence pinned to a commit.

## Do not mix

- These cells with Qwen3.8-27B EXL3 numbers. That route runs 63.25 tok/s on one
  card with a 14 GB footprint; a 33B dense model cannot match its decode rate, and
  the comparison at the same prompt/output is not measured.
- GGUF numbers with vLLM W4A16 numbers. No instruct-weight W4A16 package exists for
  this model at the time of writing.
- `llama-bench` rates with client-observed rates.

## Related authorities

- Measurement rules: `docs/BENCHMARK-FORMAT.md`
- Image and patch compatibility: `docs/IMAGE-AND-PATCH-MATRIX.md`
- Operational status: `docs/RELIABILITY-REPORT.md`
