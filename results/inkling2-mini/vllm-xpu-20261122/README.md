# Evidence: Inkling 2 Mini on vLLM XPU (`vllm-inkling2-xpu` + PR `#69186` + Opt-in CUTLASS Xe2)

- **Date:** 2026-11-22
- **Hardware:** 1x Intel Arc Pro B70 (32 GB, non-display card, `xe` driver, as-found 230 W power cap)
- **Model:** `thinkingmachines/Inkling-2-Mini` @ `ed40e7da625f594fc9c4990f5f66be24873c22d7` (native NVFP4 L0–19 / MXFP8 L20–23 experts, BF16 dense/attention/`lm_head`, BF16 KV cache)
- **Engine:** `vllm-0.34.0rc1.dev1133+g647a76866.xpu` (`647a76866b4559a74e27d1c6b55fbffe85c07ce4`) + `mgaruccio/vllm-inkling2-xpu` @ `2e9d28b04ff23f5d9996a221ffaa0ae3a440d8f9` + [`patches/inkling2-mini-vllm-xpu/`](../../../patches/inkling2-mini-vllm-xpu/APPLY.md), PyTorch `2.15.0+xpu`, Triton `3.9.0+xpu`
- **Observer:** client-side streaming OpenAI `/v1/chat/completions` observer capturing both `delta.reasoning` and `delta.content`, hardware energy counter (`energy1_input`), package temperature (`temp2_input`), live VRAM (`vram_mm`), and Prometheus `vllm:prefix_cache_hits_total` (verified `0.0` delta on every cell).

## Files

| File | Contents |
|---|---|
| `summary.json` | Aggregated medians, ranges, hardware/software provenance, and scenario pass counts for the stock no-spec + PR `#69186` baseline |
| `pr69186-ab.jsonl` | Same-session greedy A/B (`n=3`, 256 tokens out, depths `0` and `8k`): unpatched base vs PR `#69186` |
| `prefill-anchors.jsonl` | Stock no-spec + `#69186` cold prefill anchors (`p2048/g1` through `p196608/g1`, `n=3`, unique entropy-first prefixes) |
| `cells.jsonl` | Stock no-spec + `#69186` full-context window ladder (`0` to `max-197824`; `n=3` seeds 4242–4244, model-card sampling `temp=1.0`, `top_p=1.0`, `effort=0.9`, `2048` out) |
| `effort02-cells.jsonl` | Stock no-spec `reasoning_effort=0.2` cells at `0` and `8k` (`n=3`, 6/6 non-empty final answers) |
| `scenarios.jsonl` | Stock no-spec 9 deterministic chat/code/JSON/multi-turn/document-retrieval tasks (1.4K–135.3K tokens; 8/9 pass) |
| `stageC3-stock-dspark-cells.jsonl` | Stock plugin + `#69186` + DSpark `k=7` full-context ladder (`0` to `max-104320`, `n=3`, model-card sampling) |
| `bt3-cutlass-dspark-cells.jsonl` | Opt-in CUTLASS Xe2 (`INK2_CUTLASS_MOE=1`) + `#69186` + DSpark `k=7` full-context ladder (`0` to `max-104320`, `n=3`, model-card sampling) |
| `bt3-cutlass-dspark-prefill-anchors.jsonl` | Opt-in CUTLASS Xe2 + DSpark `k=7` cold prefill anchors (`p2048/g1`, `p8192/g1`, `p32768/g1`, `p65536/g1`, `n=3`) |
| `bt3-cutlass-dspark-greedy.jsonl` | Opt-in CUTLASS Xe2 + DSpark `k=7` greedy diagnostics (`0` and `8k`, `n=3`) |
| `bt3-cutlass-dspark-effort02.jsonl` | Opt-in CUTLASS Xe2 + DSpark `k=7` `reasoning_effort=0.2` cells (`0` `n=3`, `8k` `n=2`, 5/5 non-empty final answers) |
| `bt3-cutlass-dspark-scenarios.jsonl` | Opt-in CUTLASS Xe2 + DSpark `k=7` 8 deterministic quality & retrieval scenarios (**8/8 PASS**, 33.8K retrieval in `11.14 s`) |
| `capacity-profiles.json` | Separate scheduler capacity probes (`mbt=1024` 163K DSpark / 160K real prompt, `mbt=1024` 256K no-spec real prompt, `mbt=512` 308.7K KV pool) with live VRAM and clean `dmesg` receipts |
