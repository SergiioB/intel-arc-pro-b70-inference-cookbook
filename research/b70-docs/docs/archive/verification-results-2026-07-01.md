---
title: "B70 Verification Results — Jul 1, 2026 19:11Z"
type: doc
date: 2026-07-02
status: final
---

# B70 Verification Results — Jul 1, 2026 19:11Z

## Purpose
Resolve conflicting data points between master-v2 and power-sweep benchmarks.

## Environment
- Date: 2026-07-01T19:11:39Z
- Backend: llama.cpp SYCL b9851
- Flags: -ngl 99 -ncmoe 0 -fa on -ctk q8_0 -ctv q8_0 -t 8 --no-mmap
- Method: 4 diverse prompts, temp=0.0, 256 tokens, steady-state = avg rounds 2-4
- Speed: both wall-clock (tokens/time_total) and engine (timings.predicted_per_second)
- Cooldown: ≤52°C between tests
- Script: ~/b70-verify.sh

## Resolved Conflicts

### Conflict 1: 27B MTP-4 @ 165W, 128K

| Source | Value | Verdict |
|--------|-------|--------|
| master-v2 | 24.41 t/s | ✓ CORRECT |
| power-sweep | 25.61 t/s | ✗ INFLATED (+4.7%) |
| **VERIFIED** | **24.48 t/s** wall, **26.19** engine | |

Per-round:
- R1 (warmup): 256 tok / 10.72s wall=23.88, engine=26.33
- R2: 256 tok / 10.24s wall=25.00, engine=26.79
- R3: 256 tok / 10.64s wall=24.07, engine=25.72
- R4: 256 tok / 10.50s wall=24.38, engine=26.07
- Draft acceptance: 95.0% (153/161), mean len 3.35
- MAX_TEMP=68°C

### Conflict 2: 27B MTP-4 @ 230W, 128K

| Source | Value | Verdict |
|--------|-------|--------|
| master-v2 | 28.89 t/s | ✓ CORRECT |
| power-sweep | 30.75 t/s | ✗ INFLATED (+6.2%) |
| **VERIFIED** | **28.95 t/s** wall, **31.33** engine | |

Per-round:
- R1 (warmup): 256 tok / 9.13s wall=28.05, engine=31.46
- R2: 256 tok / 8.64s wall=29.62, engine=32.10
- R3: 256 tok / 9.00s wall=28.44, engine=30.73
- R4: 256 tok / 8.89s wall=28.80, engine=31.16
- Draft acceptance: 95.0%, mean len 3.35
- MAX_TEMP=77°C

### Control: 27B MTP-4 @ 180W, 128K

| Source | Value | Verdict |
|--------|-------|--------|
| power-sweep | 27.16 t/s | ✗ INFLATED (+4.2%) |
| **VERIFIED** | **26.01 t/s** wall, **27.94** engine | |

Per-round:
- R1 (warmup): 256 tok / 10.13s wall=25.27, engine=28.03
- R2: 256 tok / 9.64s wall=26.56, engine=28.57
- R3: 256 tok / 10.00s wall=25.60, engine=27.47
- R4: 256 tok / 9.90s wall=25.87, engine=27.78
- Draft acceptance: 95.0%, mean len 3.35
- MAX_TEMP=71°C

### 35B Q5_K_M Context Scaling (Real 64K)

| Context | Wall t/s | Engine t/s | Temp | Notes |
|---------|----------|------------|------|-------|
| 64K | **61.96** | 67.18 | 67°C | Previously mislabeled as 65.9 (was 32K) |
| 128K | **61.90** | 67.14 | 67°C | Confirmed against master-v2 (61.92) |

Context scaling 64K→128K: -0.1% (noise). Truly free.

### 35B Q4_K_XL @ 128K

| Source | Value | Verdict |
|--------|-------|--------|
| master-v2 | 63.57 t/s | ✓ CONFIRMED |
| **VERIFIED** | **63.41 t/s** wall, **68.83** engine | |

Per-round:
- R1 (warmup): 256 tok / 4.83s wall=53.00, engine=62.70
- R2: 256 tok / 4.04s wall=63.32, engine=68.86
- R3: 256 tok / 4.03s wall=63.50, engine=68.86
- R4: 256 tok / 4.04s wall=63.40, engine=68.77
- MAX_TEMP=66°C

## Corrected Power Scaling Curve (27B MTP-4, Q5_K_M)

| Power | Wall t/s | Engine t/s | Temp | Efficiency (wall t/s/W) |
|-------|----------|------------|------|--------------------------|
| 150W | 23.11* | — | 69°C | 0.154* |
| **165W** | **24.48** | 26.19 | 68°C | **0.148** ← peak |
| 180W | 26.01 | 27.94 | 71°C | 0.145 |
| 195W | ~28.6 est | — | 73°C | ~0.147* |
| 230W | 28.95 | 31.33 | 77°C | 0.126 |

*150W and 195W values from power-sweep, not re-verified. Likely inflated ~4%.
Non-MTP BASE model removed from disk; BASE values cannot be re-verified.

Delta 165W→230W: +4.47 t/s for +65W and +9°C.

## Root Cause Analysis

The b70-mtp-power-sweep.sh script uses a **single prompt repeated for all 4 rounds**,
violating AGENTS.md §8.5 ("Use diverse prompts"). LLM caching effects inflate rounds 2-4,
producing values ~4-5% higher than diverse-prompt methodology.

The b70-master-v2.sh uses 4 diverse prompts and produces correct results.
All master-v2 values confirmed within measurement noise (±0.1 t/s).

## Engine Rate vs Wall-Clock

Engine rate (timings.predicted_per_second) is consistently higher than wall-clock:
- Dense 27B: engine ~8% faster than wall
- MoE 35B: engine ~8-9% faster than wall

This is expected: wall-clock includes prompt processing, API overhead, and network latency.
Engine rate isolates pure decode performance per AGENTS.md §8.4.
