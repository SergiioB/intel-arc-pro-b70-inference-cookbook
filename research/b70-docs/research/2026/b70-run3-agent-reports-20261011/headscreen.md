---
title: "Run-3 subagent report: headscreen (2026-10-11)"
type: research
date: 2026-10-11
status: provisional
---

# LM-head real-state screen (unregistered) — smoke deliverable (2026-10-11)

Rewritten `scripts/tmp/headscreen_20261011/head_arms.py`; 64-position smoke passed
(`arms_smoke.json`, `smoke.log`). Label fixed: this is an unregistered screen, not
"RQ-35" (README + `build_final_states.py` docstring updated).

## Memory fix (why the first attempt died)

First attempt materialized a `[7808 x 248320]` fp32 reference-logits matrix
(7.75 GiB) and was killed by systemd-oomd. Now: BF16 head held once (2.37 GiB,
zero-copy safetensors mmap — load RSS stays 227 MiB), each arm quantized AND
dequantized row-block-wise (never stored), positions chunked to <= 256
(`[256 x 248320]` fp32 = 254 MB), only per-position scalars accumulated.

## Smoke numbers (64 real positions, 2 threads, `MemoryMax=3G`)

Command (`systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 -p CPUQuota=200% env OMP_NUM_THREADS=2 /usr/bin/time -v python3 ... --limit 64 --pos-chunk 64 --row-chunk 4096 --tag smoke`): exit 0, wall **1:52.7**.

Real top-1 agreement / mean KL / p99 KL (vs BF16 head, fp32 matmul):

| arm | GiB/round | top-1 agree | mean KL | p99 KL | gate |
|-----|----------:|------------:|--------:|-------:|------|
| a int8 row  | 1.185 | 1.0000 | 1.219e-4 | 4.79e-4 | no-promote |
| b int8 g128 | 1.203 | 1.0000 | 5.96e-5  | 2.20e-4 | no-promote |
| c fp8 row   | 1.185 | 0.9688 | 1.018e-3 | 2.56e-3 | killed |
| d fp8 g128  | 1.203 | 0.9688 | 9.39e-4  | 3.89e-3 | killed |
| e int4 g128 | 0.611 | 0.9688 | 1.819e-2 | 6.56e-2 | killed |
| f fp16 noop | 2.368 | 1.0000 | 1.241e-8 | 2.43e-7 | candidate (noise floor) |

Random-state arms (norm-matched Gaussians): a 0.9844 (KL 2.54e-4), d 0.9688
(1.31e-3), e 0.6719 (3.06e-2) — reproduces the RL-041 INT4 gap.

**Sanity next-token accuracy (BF16 head): top-1 = 33/63 = 52.4 %, top-5 = 73.0 %**
(chance 4.0e-6). Real peakedness: mean top-1 prob 0.496, mean entropy 2.48 nats
(random: 0.0104 / 10.41 nats).

**Peak RSS** (`/usr/bin/time -v` Max RSS) = **3,411,164 KiB = 3.33 GiB**; CPU 192 %;
65.2 M minor faults (head mmap pages reclaimed/re-faulted under the 3 G cap; no OOM).
Per-chunk (64 positions): ref 3.0 s; six real arms 65.4 s (11.7/13.2/10.7/12.3/13.3/4.0 s);
random ref+3 arms 40.3 s.

## Full run (after `CPU GO`) — exact command

```
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -p CPUQuota=600% \
  env OMP_NUM_THREADS=6 /usr/bin/time -v python3 \
  ~/B70-DOCS/scripts/tmp/headscreen_20261011/head_arms.py \
  ~/B70-DOCS/scripts/tmp/headscreen_20261011
```
No `--limit`; defaults `pos_chunk=256`, `row_chunk=8192`; writes `arms.json`.
Projected wall ~30-45 min at 6 threads with the head resident [INFERENCE]; the raw
2-thread thrashing slope (1.76 s/position) over 7808 positions is ~3.6 h.

## Files

- `scripts/tmp/headscreen_20261011/head_arms.py` (rewritten)
- `scripts/tmp/headscreen_20261011/arms_smoke.json`, `smoke.log`
- `results/2026/10/headscreen-real-states-20261011/README.md` (label + smoke section; gate unchanged)
