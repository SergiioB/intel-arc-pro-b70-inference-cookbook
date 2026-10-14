---
title: "VRAM Budget Model"
type: research
date: 2026-07-02
status: final
---

# VRAM Budget Model

## Constants
- Total VRAM: 32,656 MiB = 31.9 GB usable
- bf16 KV cache per 128K context (measured):
  - Qwen 27B: 24.5 GB (dense, more KV per token)
  - Qwen 35B MoE: 9.4 GB (MoE, fewer KV per token due to smaller active params)
  - Gemma 4 26B MoE: 11.3 GB
  - Ornith 35B MoE: 9.4 GB
  - Agents-A1 MoE: 9.4 GB
  - Ornith 9B: 7.5 GB (dense but small)

## Model Weights (from disk)
| Model | Quant | Size |
|-------|-------|------|
| Qwen 27B MTP | Q5_K_M | 19 GB |
| Qwen 35B | Q5_K_M | 25 GB |
| Qwen 35B | Q4_K_XL | 21 GB |
| Gemma 4 26B | Q4_K_M | 16 GB |
| Ornith 35B | Q4_K_M | 20 GB |
| Agents-A1 | Q4_K_M | 20 GB |
| Ornith 9B | Q6_K | 7 GB |

## KV Cache Multipliers (vs bf16)
| Config | Multiplier |
|--------|-----------|
| q8_0-q8_0 | 0.531 |
| q6_0-q5_0 | 0.375 |
| q5_0-q5_0 | 0.344 |
| q5_0-q4_1 | 0.328 |
| q4_0-q4_0 | 0.281 |

## Full VRAM Budget Table

### With q8_0-q8_0 KV (current mandatory)

| Model | Quant | 64K | 128K | 200K | 256K | 512K |
|-------|-------|-----|------|------|------|------|
| Qwen 27B MTP | Q5_K_M | 25.5✓ | 32.0▶ | 39.3✗ | 45.0✗ | 71.0✗ |
| Qwen 35B | Q5_K_M | 27.5✓ | 30.0✓ | 32.8✗ | 35.0✗ | 45.0✗ |
| Qwen 35B | Q4_K_XL | 23.5✓ | 26.0✓ | 28.8✓ | 31.0✓ | 41.0✗ |
| Gemma 4 26B | Q4_K_M | 19.0✓ | 22.0✓ | 25.4✓ | 28.0✓ | 40.0✗ |
| Ornith 35B | Q4_K_M | 22.5✓ | 25.0✓ | 27.8✓ | 30.0✓ | 40.0✗ |
| Agents-A1 | Q4_K_M | 22.5✓ | 25.0✓ | 27.8✓ | 30.0✓ | 40.0✗ |
| Ornith 9B | Q6_K | 9.0✓ | 11.0✓ | 13.2✓ | 15.0✓ | 23.0✓ |

### With q5_0-q4_1 KV (recommended)

| Model | Quant | 64K | 128K | 200K | 256K | 512K |
|-------|-------|-----|------|------|------|------|
| Qwen 27B MTP | Q5_K_M | 23.0✓ | 27.0✓ | 31.5✓ | 35.1✗ | 51.1✗ |
| Qwen 35B | Q5_K_M | 26.5✓ | 28.1✓ | 29.9✓ | 31.2✓ | 37.5✗ |
| Qwen 35B | Q4_K_XL | 22.5✓ | 24.1✓ | 25.9✓ | 27.2✓ | 33.5✗ |
| Gemma 4 26B | Q4_K_M | 17.9✓ | 19.8✓ | 21.9✓ | 23.5✓ | 31.0✓ |
| Ornith 35B | Q4_K_M | 21.5✓ | 23.1✓ | 24.9✓ | 26.2✓ | 32.5✓ |
| Agents-A1 | Q4_K_M | 21.5✓ | 23.1✓ | 24.9✓ | 26.2✓ | 32.5✓ |
| Ornith 9B | Q6_K | 8.2✓ | 9.5✓ | 10.9✓ | 12.0✓ | 17.0✓ |

✓ = fits (≥0.5GB free)  ▶ = borderline  ✗ = OOM

## Key Insights

1. **q5_0-q4_1 KV doubles usable context** for Qwen 35B Q5_K_M (128K→256K)
   and gives Qwen 27B MTP 72K headroom (128K→200K).

2. **Ornith 9B is the context king**: 512K uses only 17GB. Could push 1024K+.

3. **Gemma 4 26B reaches 512K** with q5_0-q4_1 KV — from 16GB weights.

4. **Qwen 35B Q5_K_M at 256K** is the flagship: best quality model at double the
   previous max context, with negligible quality loss from KV quantization.
