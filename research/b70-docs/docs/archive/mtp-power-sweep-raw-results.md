---
title: "⚠️ WARNING: VALUES IN THIS DOCUMENT ARE INFLATED ~4-5%"
type: doc
date: 2026-07-02
status: final
---

# ⚠️ WARNING: VALUES IN THIS DOCUMENT ARE INFLATED ~4-5%

The power-sweep script used a single prompt repeated for all rounds,
violating the diverse-prompts methodology (AGENTS.md §8.5). LLM caching
inflates rounds 2-4. Corrected values verified on 2026-07-01:

| Power | Old (inflated) | Correct (wall) |
|-------|----------------|----------------|
| 165W | 25.61 t/s | **24.48 t/s** |
| 180W | 27.16 t/s | **26.01 t/s** |
| 230W | 30.75 t/s | **28.95 t/s** |

See ~/B70-DOCS/verification-results-2026-07-01.md for full corrected data.

---

# B70 MTP Speculative Decoding — Raw Results (Jul 1, 2026)

## Environment
- Backend: llama.cpp SYCL b9851
- Flags: -ngl 99 -ncmoe 0 -fa on -ctk q8_0 -ctv q8_0 -t 8 --no-mmap
- Method: 4 diverse prompts, temp=0.0, 256 tokens, steady-state = avg rounds 2-4
- Speed: wall-clock (tokens/time_total) for MTP comparison tests

## MTP Comparison: 27B Dense @ 180W

### BASE (spec=none, max=0)
ROUND 1: 256 tok / 13.999586s = 18.29 t/s
ROUND 2: 256 tok / 13.364189s = 19.16 t/s
ROUND 3: 256 tok / 13.207143s = 19.38 t/s
MAX_TEMP=71
Steady-state: 19.27 t/s

### MTP-2 (spec=draft-mtp, max=2)
ROUND 1: 256 tok / 10.590320s = 24.17 t/s
ROUND 2: 256 tok / 9.916930s = 25.81 t/s
ROUND 3: 256 tok / 9.788287s = 26.15 t/s
draft acceptance = 0.93793 (136/145), mean len = 2.60
MAX_TEMP=71
Steady-state: 25.98 t/s (+34.8%)

### MTP-4 (spec=draft-mtp, max=4)
ROUND 1: 256 tok / 10.197070s = 25.11 t/s
ROUND 2: 256 tok / 9.483785s = 26.99 t/s
ROUND 3: 256 tok / 9.372080s = 27.32 t/s
draft acceptance = 0.93373 (155/166), mean len = 3.21
MAX_TEMP=71
Steady-state: 27.16 t/s (+40.9%)

## MTP Comparison: 35B MoE @ 130W

### BASE (spec=none, max=0)
ROUND 1: 256 tok / 5.101164s = 50.18 t/s
ROUND 2: 256 tok / 4.057138s = 63.10 t/s
ROUND 3: 256 tok / 3.991860s = 64.13 t/s
MAX_TEMP=64
Steady-state: 63.62 t/s

### MTP-2 (spec=draft-mtp, max=2)
ROUND 1: 256 tok / 5.469444s = 46.81 t/s
ROUND 2: 256 tok / 4.435090s = 57.72 t/s
ROUND 3: 256 tok / 4.427305s = 57.82 t/s
draft acceptance: 0.926/0.946/0.947, mean len ~2.50
MAX_TEMP=64
Steady-state: 57.77 t/s (-9.2%)

### MTP-4 (spec=draft-mtp, max=4)
ROUND 1: 256 tok / 5.061391s = 50.58 t/s
ROUND 2: 256 tok / 3.964910s = 64.57 t/s
ROUND 3: 256 tok / 3.972373s = 64.45 t/s
draft acceptance: 0.933/0.951/0.956, mean len ~3.38
MAX_TEMP=64
Steady-state: 64.51 t/s (+1.4%)

## 27B Power Scaling (MTP-4, steady-state avg rounds 2-4)

### BASE (no MTP)
150W: 16.53 t/s, 68C
230W: 21.07 t/s, 78C

### MTP-4
150W:  23.11 t/s, 69C, eff=0.154 t/s/W
165W:  25.61 t/s, 69C, eff=0.155 t/s/W (PEAK EFFICIENCY)
180W:  27.16 t/s, 71C, eff=0.151 t/s/W
195W:  28.59 t/s, 73C, eff=0.147 t/s/W
230W:  30.75 t/s, 77C, eff=0.134 t/s/W

## 35B Q5_K_M Context Fit (25GB model, 32GB VRAM)

64K ctx @ 150W: 61.96 t/s, 67C (verified Jul 1)
128K ctx @ 150W: 61.90 t/s, 66C (verified Jul 1)
128K ctx @ 150W: 61.9 t/s, 66C

## Key Insights

1. MTP helps dense models (memory-bandwidth bound, ~20 t/s) with +35-46% gain.
   Does NOT help MoE (compute-bound, 60+ t/s). Crossover is 20-60 t/s.

2. Power sweet spot for 27B MTP-4: 165W (peak efficiency 0.155 t/s/W).
   230W adds only 5 t/s for +65W and +8C.

3. Context scaling 64K to 128K is free (zero speed penalty) with q8 KV + FA.

4. 35B Q5_K_M (25GB) fits at 128K context in 32GB VRAM.
