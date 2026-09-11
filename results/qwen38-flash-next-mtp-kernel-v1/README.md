# Qwen3.8-Flash-Next + MTP + fused multi-token MoE kernel — dual Arc Pro B70

Reference figures for the cookbook records in this repository (rendered by
`scripts/render-benchmark-catalog.py` from `data/benchmarks.v1.json`).

- `01-speed-progression.svg` — decode P512/G128 C1 medians across build stages
- `02-decode-vs-context.svg` — decode vs context (c8 / c16 / c131072), pre-wiring vs wired
- `03-kernel-ab.svg` — fused multi-token MoE kernel A/B, round cost, dispatch proof
- `04-acceptance-thermals.svg` — MTP acceptance by content/warmth; card asymmetry
- `05-prefill-f16.svg` — prefill (cold input) comparison, F16 vs FP32

All numbers are self-measured engine-native C1 llama.cpp timings (single client,
195 W cap, t=1.0 / top_p 0.95 / top_k 20, ignore_eos, exact token counts, zero
prompt-cache reuse per sample). PROVISIONAL: not independently reproduced.
