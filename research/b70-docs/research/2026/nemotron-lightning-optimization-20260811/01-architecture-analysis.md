---
title: "01 — Nemotron-H MoE Architecture Analysis"
type: research
date: 2026-08-11
status: final
---

# 01 — Nemotron-H MoE Architecture Analysis

## Model identity

- **GGUF:** `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_K_M.gguf` (23.68 GiB, 32.91B params)
- **Architecture:** `nemotron_h_moe` (llama.cpp internal name: `nemotron_h_moe`)
- **Type label:** `128x2.5B` (128 experts)
- **Context:** 1,048,576 (1M) tokens

## Layer structure (53 blocks)

The model is a **hybrid Mamba2-Transformer MoE**. Each block is one of three types:

### Mamba2/SSM layers (46 of 53)

These are the dominant layer type and the performance bottleneck.

| Parameter | Value | Notes |
|-----------|------:|-------|
| `ssm_d_inner` | 4096 | Inner projection dimension |
| `ssm_d_state` | 128 | SSM state size per head |
| `ssm_dt_rank` | 64 | Time-step rank (= n_head for Mamba2 SSD) |
| `ssm_n_group` | 8 | Number of SSM groups |
| `ssm_d_conv` | 4 | Causal convolution kernel width |
| head_dim | 64 | d_inner / dt_rank = 4096 / 64 |
| n_head | 64 | = dt_rank |

**Per-layer ops (ggml graph):**
1. Input projection (GEMV): 2688 → 10304 (2×d_inner + 2×n_group×d_state + n_head)
2. Split into z, xBC, dt
3. Conv state ring buffer update (concat + transpose + cpy)
4. Depthwise causal convolution (d_conv=4)
5. Add conv bias + SiLU
6. Split into x, B, C
7. dt bias add + softplus
8. SSM selective scan (state update + C·state reduction)
9. D×x skip connection
10. z gating (SwiGLU)
11. Grouped RMS norm
12. Output projection (GEMV): 4096 → 2688

Total: ~16-18 kernel dispatches per layer.

### Attention + MoE layers (7 of 53, at indices 5,12,19,26,33,42,52)

| Parameter | Value |
|-----------|------:|
| attention.head_count | 32 |
| attention.head_count_kv | 2 (per attention layer) |
| attention.key_length | 128 |
| attention.value_length | 128 |
| expert_count | 128 |
| expert_used_count | 6 (routed) |
| expert_shared_count | 1 (always active) |
| expert_feed_forward_length | 1856 |
| expert_shared_feed_forward_length | 3712 |
| expert_weights_scale | 2.5 |
| expert_gating | sigmoid |

### MTP head (block 52, `nextn_predict_layers = 1`)

The "Lightning" feature — native multi-token prediction for speculative decoding.
- 1 extra block: NoPE attention + MoE sub-layer
- Activated via `--spec-type draft-mtp`
- When inactive, tensors are skipped (reported as "unused tensor blk.52.nextn.*")

## Active parameters per token

| Component | Params | % of total | Bytes (Q4) |
|-----------|-------:|----------:|-----------:|
| Mamba weights (46 layers) | 1.78B | 62% | 0.98 GB |
| MoE active experts (7 layers) | 0.72B | 25% | 0.40 GB |
| LM head | 0.35B | 12% | 0.19 GB |
| **Total active** | **2.86B** | **100%** | **1.57 GB** |

Plus F32 SSM state: 46 × 128 × 64 × 64 × 4 = 193 MB (read + write per token).

## DFlash companion model

- **Source:** `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash` (HuggingFace, 08/11/2026)
- **Converted:** `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-DFlash.gguf` (1.18 GiB)
- **Architecture:** `DFlashDraftModel` (qwen3-based dense, 6 layers, 833M params)
- **No Mamba2 layers** — pure dense transformer draft
- **Quantization:** MLP weights NVFP4 (W4A16), attention BF16
- **Config:** `target_layer_ids = [1, 5, 19, 29, 41, 51]`, `mask_token_id = 990`
- **Usage:** `--spec-type draft-dflash --spec-draft-n-max 7 -fa on --jinja`
