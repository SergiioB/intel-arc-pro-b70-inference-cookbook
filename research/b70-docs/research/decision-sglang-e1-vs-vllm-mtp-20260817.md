# Decision 2026-08-17 — stop SGLang E=1 for dense decode

## Verdict
Stay on **vLLM XPU + GPTQ-INT4 + MTP4 + XPUwNa16**. Do not spend more time on SGLang 1-expert dense GEMM for Qwen3.8-27B decode.

## Why
1. Measured E=1 kernel (this session): M=1 is ~160 GB/s, 25% of 608. 64-layer GEMM-only ceiling **16.6 t/s** before QKV/GDN.
2. Production already: **83.7 t/s decode / 1774 t/s 8K prefill** (Run 40, MTP4, 95% accept).
3. Run 39 architect note: MTP4 decode is at the **608 GB/s ceiling**. 95 t/s needs a multi-layer MTP checkpoint or the MoE track — not a new GEMM.
4. SGLang XPU MTP is ~1.3×. vLLM MTP is ~3.3× (24.4 → 83.7). A faster GEMM on a weak MTP stack loses E2E.
5. Already rejected on this model: MTP8, async, 16K batch, W4A8 (corrupt).

## What the E=1 work did prove
- `moe_grouped_mm_nt_xe20_w4a16` with `num_experts=1` runs on B70.
- Occupancy wants M=16–32 (51 TFLOP/s), not M=1 (0.6 TFLOP/s).
- Useful later only as a **vLLM prefill** swap if oneDNN is slower at M≥256. Not a decode lever.

## Next real levers (ranked)
1. Restore pruned `vllm/vllm-openai-xpu@sha256:2c427ef…` (production dense launcher).
2. Multi-layer MTP checkpoint if one appears — only path to ~95 decode.
3. Keep MoE 35B vLLM MTP as the speed champion (~200 t/s).
4. Dual-B70 only for capacity, not single-stream decode.
