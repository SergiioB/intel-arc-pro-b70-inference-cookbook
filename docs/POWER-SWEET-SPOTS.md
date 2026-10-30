# Power Sweet Spots — don't waste watts

The B70's power cap is a per-workload setting, not a "max it out" setting.
MoE and Dense want **opposite** configurations.

## The data (single-stream, p2k prompt / g128 gen)

### MoE 35B (vLLM MTP)

| Power cap | Decode (best) | Prefill | Temp peak |
|-----------|--------------:|--------:|----------:|
| **150W**  | **125.7 t/s** | 7,308   | ~58°C     |
| 230W      | 115.4 t/s     | 7,345   | ~58°C     |

**Δ 150→230W: -8.2% (slower!).** MoE self-limits power draw to ~140W regardless
of the cap (the active experts are bandwidth-bound, not frequency-bound). Raising
the cap past ~150W adds heat noise without adding speed.

**→ Run MoE at 150W.** Cooler, same speed, 80W less heat.

### Dense 27B (llama.cpp Q4_K_M)

| Power cap | short/g32 | p2k/g128 | Temp peak |
|-----------|----------:|---------:|----------:|
| 150W      | 22 t/s    | 18 t/s   | 71°C      |
| 230W      | **26 t/s**| **23 t/s** | 79°C    |

**Δ 150→230W: +18–30% (scales!).** Dense reads all ~19 GB/token, so it benefits
from the frequency headroom that higher power buys — but the thermal cost is
real (dense runs 13–21°C hotter than MoE).

**→ Run Dense at 180W sustained** (efficiency sweet spot: 0.155 t/s/W).
**230W only for short bursts** — sustained 230W hits 79°C and risks throttling.

## How to set it

The power cap is in **microwatts** at `/sys/class/hwmon/hwmon4/power1_cap`
(hwmon4 is the B70 on this host; verify with `cat /sys/class/hwmon/hwmon*/name`
looking for `arc` or the Intel driver).

```bash
# Read current cap (microwatts)
cat /sys/class/hwmon/hwmon4/power1_cap

# Set to 150W (MoE sweet spot)
echo 150000000 | sudo tee /sys/class/hwmon/hwmon4/power1_cap

# Set to 180W (Dense sustained sweet spot)
echo 180000000 | sudo tee /sys/class/hwmon/hwmon4/power1_cap

# Set to 230W (Dense burst / stock)
echo 230000000 | sudo tee /sys/class/hwmon/hwmon4/power1_cap

# Monitor temperature (millidegrees C — divide by 1000)
watch -n1 'echo $(($(cat /sys/class/hwmon/hwmon4/temp2_input)/1000))°C'
```

Common values: `150000000`=150W, `165000000`=165W, `180000000`=180W, `230000000`=230W.

## Why the difference?

Both MoE and Dense are **bandwidth-bound** at the B70's 608 GB/s ceiling. The
difference is bytes per token:

- **MoE 35B-A3B** reads only the ~3 GB of active experts per token (router
  picks top-8 of 256). Bandwidth-limited → frequency doesn't help → power
  self-limits.
- **Dense 27B** reads all ~19 GB of weights per token. The memory subsystem is
  the bottleneck, but higher frequency slightly raises effective bandwidth →
  power scales.

This is *not* a universal MoE property — spec decoding helps MoE elsewhere
(Cohere, Gemma 4). It's specific to this card + the Qwen3.6-35B-A3B checkpoint
(which lacks MTP layers in most public quants — the "MTP-preserved" GPTQ in
this repo is the exception).

## Methodology

- Power set via hwmon `power1_cap`, verified by reading it back.
- Temperature from hwmon `temp2_input` (the B70 GPU sensor — `temp1_input` does
  not exist on this card; some older scripts had this wrong).
- Cooldown to ≤52°C between runs (cooldown loop in our harnesses).
- No two inference processes concurrent (VRAM contention = invalid data).
- Decode = best steady-state rep (drops JIT warmup). 2 reps/cell.

Full sweep data: `benchmarks/results/` (see campaign log for the Run 19 reference).

## Field data: cap SKU spread and editability (2026-10-30)

Two independent ASRock B70 systems, same silicon:

| | 180 W SKU (this rig) | 275 W SKU (community report) |
|---|---|---|
| default `power1_cap` | 180 W | 275 W |
| sustained under LLM load | 179-180 W, clocks settle, no thermal events | 275 W (spikes ~300), 2.3-2.4 GHz, 58-60 C |
| idle | ~46 W | ~50 W |
| `power1_cap` sysfs write | accepted (200 W written, read back; restored to 180) | accepted (250/275 W, read back) |
| `power1_cap_min/max` | not exposed | not reported |

Findings:
- The cap is a **default, not a lock**: hwmon writes succeed on both SKUs
  (root). No min/max bounds are exposed to constrain the write.
- The SKU difference is the out-of-box cap - partner defaults vary widely
  on identical silicon (180 vs 275 W is a ~50% sustained performance axis
  set at the factory).
- Writes are **runtime-only**: reset on reboot / driver reload. Persist via
  a udev rule (`ATTR{power1_cap}="..."` with a `TEST==` match) if desired.
- Enforcement under load tracks the cap on this rig: the Run 19 power sweep
  in this repo changes `power1_cap` and measures the load response, and
  throughput follows the cap monotonically.
- Buy guidance for dense builds: confirm which cap SKU a listing is - the
  model name alone does not predict sustained clocks. Cooling on both SKUs
  runs far under stress at stock cap (67 C peak, this rig).
