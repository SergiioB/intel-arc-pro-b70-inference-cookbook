# Evidence: Inkling 2 Mini on vLLM XPU (`vllm-inkling2-xpu` + PR #69186)

- **Date:** 2026-11-22
- **Hardware:** 1x Intel Arc Pro B70 (32 GB, non-display card, `xe` driver, as-found 230 W power cap)
- **Model:** `thinkingmachines/Inkling-2-Mini` @ `ed40e7da625f594fc9c4990f5f66be24873c22d7` (native NVFP4 L0–19 / MXFP8 L20–23 experts, BF16 dense/attention/`lm_head`, BF16 KV cache)
- **Engine:** `vllm-0.34.0rc1.dev1133+g647a76866.xpu` (`647a76866b4559a74e27d1c6b55fbffe85c07ce4`) + `mgaruccio/vllm-inkling2-xpu` @ `2e9d28b04ff23f5d9996a221ffaa0ae3a440d8f9` + [`patches/inkling2-mini-vllm-xpu/pr69186-logits-processor.patch`](../../../patches/inkling2-mini-vllm-xpu/pr69186-logits-processor.patch), PyTorch `2.15.0+xpu`, Triton `3.9.0+xpu`
- **Observer:** client-side streaming OpenAI `/v1/chat/completions` observer capturing both `delta.reasoning` and `delta.content`, hardware energy counter (`energy1_input`), package temperature (`temp2_input`), live VRAM (`vram_mm`), and Prometheus `vllm:prefix_cache_hits_total` (verified `0.0` delta on every cell).

## Files

| File | Contents |
|---|---|
| `summary.json` | Aggregated medians, ranges, hardware/software provenance, and scenario pass counts |
| `pr69186-ab.jsonl` | Same-session greedy A/B (`n=3`, 256 tokens out, depths `0` and `8k`): unpatched base vs PR #69186 |
| `prefill-anchors.jsonl` | Cold prefill anchors (`p2048/g1` through `p196608/g1`, `n=3`, unique entropy-first prefixes) |
| `cells.jsonl` | Full-context window ladder (`0`, `8k`, `32k`, `64k`, `max-87936`, `128k`, `max-197824`; `n=3` seeds 4242–4244, model-card sampling `temperature=1.0`, `top_p=1.0`, `reasoning_effort=0.9`, `max_tokens=2048`) |
| `effort02-cells.jsonl` | Model-card `reasoning_effort=0.2` cells at `0` and `8k` (`n=3`) confirming 6/6 non-empty final answers |
| `scenarios.jsonl` | 9 deterministic chat/code/JSON/multi-turn/document-retrieval tasks (1.4K, 8.5K, 33.8K, and 135.3K tokens; 8/9 pass) |
