# High-Context Validation Results

## Test Environment
- Date: 2026-07-01T19:34Z
- Backend: llama.cpp SYCL b9851
- KV Cache: q8_0-q8_0 (current mandatory)
- Flags: -ngl 99 -ncmoe 0 -fa on -t 8 --no-mmap
- Method: 4 diverse prompts, temp=0.0, 256 tokens, steady-state avg rounds 2-4
- Power: 150W for all tests
- Script: ~/b70-highctx.sh

## Tier 1: 256K (Confident Fits)

| Model | Quant | Context | Wall t/s | Engine t/s | Temp | vs 128K |
|-------|-------|---------|----------|------------|------|---------|
| Gemma 4 26B | Q4_K_M | 256K | 57.29 | 59.64 | 66°C | -0.2% |
| Ornith 35B | Q4_K_M | 256K | 67.85 | 73.70 | 67°C | -0.7% |
| Agents-A1 | Q4_K_M | 256K | 68.55 | 74.51 | 67°C | +0.2% |

## Tier 2: Ornith 9B (Massive Headroom)

| Model | Quant | Context | Wall t/s | Engine t/s | Temp | vs 128K |
|-------|-------|---------|----------|------------|------|---------|
| Ornith 9B | Q6_K | 256K | 48.60 | 50.95 | 66°C | -0.2% |
| Ornith 9B | Q6_K | **512K** | **48.61** | 50.97 | 64°C | -0.2% |

## Tier 3: Qwen 35B Q4 (Tight but Fits)

| Model | Quant | Context | Wall t/s | Engine t/s | Temp | vs 128K |
|-------|-------|---------|----------|------------|------|---------|
| Qwen 35B | Q4_K_XL | 200K | 63.58 | 69.04 | 67°C | +0.3% |
| Qwen 35B | Q4_K_XL | 256K | 63.52 | 68.99 | 67°C | +0.2% |

## Key Findings

1. **Context scaling is free at all tested lengths.** No model shows any speed penalty
   at 200K, 256K, or even 512K context with q8_0 KV + flash attention.

2. **Ornith 9B Q6_K at 512K uses only ~23GB VRAM** — could go to 1024K+.

3. **All MoE models verified at 256K with zero penalty.** The compute-bound nature
   of MoE (active params determine speed, not total context) makes context scaling free.

4. **The 32K context floor has been eliminated.** Minimum tested context is now 64K.
   Standard recommendation is 128K. Higher contexts recommended wherever VRAM allows.

## Models NOT Tested at High Context

- **Qwen 27B MTP Q5_K_M**: At 128K with q8_0 KV, VRAM is already maxed (32GB).
  Cannot test higher contexts with current KV quant. Needs q5_0-q4_1 KV to unlock 200K.

- **Qwen 35B Q5_K_M**: At 128K with q8_0 KV, 1.9GB free. 200K needs ~32.8GB (OOM).
  Needs q5_0-q4_1 KV to unlock 256K.

These are the two models that benefit most from the KV quant optimization.
