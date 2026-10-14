# Power Scaling — Corrected Values

## Issue

The b70-mtp-power-sweep.sh script used a single prompt repeated for all 4 rounds,
violating the diverse-prompts rule (AGENTS.md §8.5). LLM caching inflated rounds 2-4
by ~4-5%. Independent verification with diverse prompts corrected all values.

## Corrected 27B MTP-4 Power Scaling (Q5_K_M, 128K, q8_0 KV)

| Power | Wall t/s | Engine t/s | Temp | Efficiency (wall t/s/W) | Notes |
|-------|----------|------------|------|--------------------------|-------|
| 150W | 23.11* | — | 69°C | 0.154* | Not re-verified; likely inflated |
| **165W** | **24.48** | 26.19 | 68°C | **0.148** | Peak efficiency ← verified |
| 180W | 26.01 | 27.94 | 71°C | 0.145 | ← verified |
| 195W | ~28.6* | — | 73°C | ~0.147* | Not re-verified; likely inflated |
| 230W | 28.95 | 31.33 | 77°C | 0.126 | ← verified |

*Values from power-sweep, not independently verified. Non-MTP BASE model removed
from disk, cannot re-verify BASE values.

## Corrected MTP Gain

- MTP-4 vs BASE at 180W: +35% (26.01 / 19.27)
  (Previously claimed +41% using inflated values)

- MTP on 35B MoE: no benefit (compute-bound at 60+ t/s)
  MTP-4 at 130W: 64.51 vs BASE 63.62 = +1.4% (within noise)

## Verification Methodology
- Script: ~/b70-verify.sh (saved as ~/b70-verified-bench.sh)
- Date: 2026-07-01T19:11Z
- 4 diverse prompts, temp=0.0, 256 tokens
- Cooldown ≤52°C between tests
- Both wall-clock and engine rate captured

## Impact on Documentation
All values in AGENTS.md §7, benchmark-guide.md model table, and power scaling
section have been updated to corrected values. Original inflated values are
preserved in mtp-power-sweep-raw-results.md with a warning header.
