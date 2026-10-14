# Optimization Roadmap

Prioritized optimizations to test on B70, ordered by expected impact.

## Phase 1: KV Cache Quantization (HIGHEST IMPACT)

### 1a. q5_0-q4_1 KV Validation ✅ COMPLETE (2026-07-01)
**Expected**: Doubles context for Qwen 35B Q5_K_M (128K→256K) and Qwen 27B MTP (128K→200K)
**Quality trade**: -1.96% tail precision (Anbeeld data)
**Result**: Validated on stock llama.cpp b9851. Works perfectly. 3.3% faster than q8_0.
**Test plan**: Run on 2 models first (Qwen 35B Q5 at 128K, Qwen 27B MTP at 128K),
  verify quality with 4 diverse prompts, compare outputs to q8_0 baseline.

### 1b. q6_0-q5_0 KV Validation  
**Expected**: Better quality than q5_0-q4_1 (93.29% vs 92.65%) at 14% more VRAM
**Test plan**: If q5_0-q4_1 shows visible quality issues, test this as fallback.

### 1c. q8_0-q5_1 KV Validation
**Expected**: Near-identical to q8_0-q8_0 (94.21% vs 94.61%) at 15% less VRAM
**Test plan**: Conservative option. Test on Qwen 27B MTP where every GB matters.

## Phase 2: Model-Specific Optimizations

### 2a. Qwen 35B Q5_K_M @ 256K with q5_0-q4_1
**Impact**: Flagship quality model at 2x context
**Prerequisite**: Phase 1a must pass

### 2b. Qwen 27B MTP Q5_K_M @ 200K with q5_0-q4_1  
**Impact**: Dense model with MTP at higher context
**Prerequisite**: Phase 1a must pass

### 2c. Ornith 9B Q6_K @ 1024K
**Impact**: Maximum context for lightweight model (~17GB at 1024K with q5_0-q4_1)
**Prerequisite**: Phase 1a

## Phase 3: Quant-vs-Context Tradeoff Testing

### 3a. Impact of KV quant on vision quality
**Question**: Does q5_0-q4_1 KV affect vision description quality?
**Test**: Compare vision outputs at q8_0 vs q5_0-q4_1

### 3b. Impact of KV quant on tool calling
**Question**: At what KV quant do tool calls break on B70 models?
**Test**: Structured output tests at each KV quant tier

## Phase 4: Documentation Updates

After each phase, update:
- AGENTS.md §4 (Critical Flags) — change mandatory KV quant
- benchmark-guide.md — update context ceiling tables
- vram-budget-model.md — confirm predictions
- All model tables with verified values

## Risk Assessment

| Optimization | Risk | Mitigation |
|-------------|------|------------|
| q5_0-q4_1 KV | Medium: untested on stock llama.cpp b9851 | Validate on 2 models first |
| q5_0-q4_1 KV | Low: quality regression | Compare outputs to q8_0 baseline |
| 256K context | Low: VRAM OOM | Conservative estimation; test incrementally |
