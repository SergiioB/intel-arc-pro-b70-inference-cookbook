# Patch Stack: Inkling 2 Mini on vLLM XPU

- **Target vLLM base:** `vllm-0.34.0rc1.dev1133+g647a76866.xpu` (upstream commit `647a76866b4559a74e27d1c6b55fbffe85c07ce4`, before upstream [vLLM PR #69186](https://github.com/vllm-project/vllm/pull/69186) merged).
- **Plugin:** [`mgaruccio/vllm-inkling2-xpu`](https://github.com/mgaruccio/vllm-inkling2-xpu) pinned at `2e9d28b04ff23f5d9996a221ffaa0ae3a440d8f9` (installed `--no-deps` into the same environment; does not patch vLLM itself).
- **Ordered patch list (1 patch):**
  1. `pr69186-logits-processor.patch` — enables `torch.mm(..., out_dtype=torch.float32)` on XPU for FP32 `lm_head` projection instead of recasting the full BF16 `lm_head` weight (~1.72 GiB) to FP32 on every decode step.

## Apply

If the environment was installed with `uv --link-mode hardlink`, break the hardlink on `logits_processor.py` before patching so the shared package cache is not modified:

```bash
LP=$(python3 -c "import inspect, vllm.model_executor.layers.logits_processor as m; print(inspect.getfile(m))")
cp "$LP" "${LP}.orig" && rm -f "$LP" && cp "${LP}.orig" "$LP"
patch -p1 -d "$(dirname "$(dirname "$(dirname "$(dirname "$LP")")")")" < patches/inkling2-mini-vllm-xpu/pr69186-logits-processor.patch
```

Skip this patch once your vLLM XPU build includes merged PR #69186.
