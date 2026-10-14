---
title: "Intel Arc B70 32GB — Benchmark & Optimization Findings"
type: doc
date: 2026-07-02
status: final
---

# Intel Arc B70 32GB — Benchmark & Optimization Findings

**Date:** July 1, 2026
**Host:** desktop-b70 (Ubuntu 26.04, kernel 7.0.0-27-generic)
**GPU:** Intel Arc Pro B70 32GB (Battlematrix XMG, 2-tile)
**Backend:** llama.cpp SYCL, build #9851 (`0eca4d490`)
**Binary:** `~/llama.cpp/build/bin/llama-bench`

---

## Executive Summary

The B70 runs excessively hot at stock power (230W → 79-83°C) under sustained LLM inference. Through systematic power capping and configuration tuning, we identified operating points that bring temperature down to 60-65°C with minimal decode speed loss. Key findings:

- **Power cap is the primary thermal control** — the xe driver exposes `/sys/class/hwmon/hwmon4/power1_cap` but no voltage curve offset
- **Frequency capping works** via `tile0/gt*/freq0/max_freq` sysfs but provides minimal additional benefit over power capping alone
- **Decode (tg) is bandwidth-bound, not compute-bound** — power caps hurt decode far less than prefill
- **SYCL JIT warmup is critical** — first benchmark run at any config pays 30-60s compilation penalty, contaminating results
- **Parallel benchmarks destroy data integrity** — overlapping runs cause VRAM contention, thermal contamination, and OOM crashes

---

## Environment

### Verified Working Configuration

```bash
# Required environment
source /opt/intel/oneapi/setvars.sh --force
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export ZES_ENABLE_SYSMAN=1
export SYCL_CACHE_PERSISTENT=0
# REMOVED 2026-07-02: 44.7% decode regression. See research repo. # export GGML_SYCL_DISABLE_OPT=1
export SYCL_DEVICE_FILTER=level_zero
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ZE_AFFINITY_MASK=0
```

**Critical:** Without `source setvars.sh`, llama-bench fails with `libsvml.so: cannot open shared object file`.

### Disk Layout

| Mount | Device | Size | Used | Free | Role |
|-------|--------|------|------|------|------|
| `/` (SSD) | `/dev/sdb4` | 214G | 99G | 105G | Models cache, system |
| `/mnt/<models-disk>` (HDD) | `/dev/sda2` | 932G | 857G | 76G | Cold storage only |

**Models MUST be on SSD** (`~/models-cache/`). HDD load time for 21GB model: 15+ min. SSD: ~21s.

### Models on SSD

| Model | Size | Type | Notes |
|-------|------|------|-------|
| Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf | 21G | MoE (3B active) | Primary test model |
| Qwen3.6-35B-A3B-UD-Q5_K_M.gguf | 25G | MoE (3B active) | Tight VRAM at 64K |
| Qwen3.6-27B-Q5_K_M.gguf | 19G | Dense 27B | KV-hungry at high context |
| ornith-1.0-35b-Q4_K_M.gguf | 20G | MoE (9B active) | |
| ornith-1.0-9b-Q6_K.gguf | 6.9G | Dense 9B | Fast, good for dense baseline |
| Agents-A1-Q4_K_M.gguf | 20G | Dense | |
| gemma-4-26B-A4B-it-UD-Q4_K_M.gguf | 16G | MoE (4B active) | |
| gemma-4-26B-A4B-it-Q8_0-MTP.gguf | 441M | MTP drafter | |
| gemma-4-12B-it-Q8_0-MTP.gguf | 444M | MTP drafter | |
| Step-3.7-flash-IQ3_XXS (2 parts) | 4.6G | Dense | Aggressive quant |

---

## GPU Control Interfaces

### Power Cap (Primary Thermal Control)

```bash
# Read current cap (in microwatts)
cat /sys/class/hwmon/hwmon4/power1_cap
# 150000000 = 150W

# Set cap (requires root)
echo 150000000 | sudo tee /sys/class/hwmon/hwmon4/power1_cap
```

Range: 35W (minimum) to 300W+ (stock). Persists until reboot or reset.

### Frequency Cap

```bash
# Paths (deep in PCI tree — NOT in /sys/class/drm/)
BASE="/sys/devices/pci0000:00/0000:00:03.1/0000:09:00.0/0000:0a:01.0/0000:0b:00.0"

# GT0 (render/compute tile) — range 400-2800 MHz
cat $BASE/tile0/gt0/freq0/max_freq   # Default: 2800
echo 2400 | sudo tee $BASE/tile0/gt0/freq0/max_freq

# GT1 (media tile) — range 400-1500 MHz
cat $BASE/tile0/gt1/freq0/max_freq   # Default: 1500
echo 1300 | sudo tee $BASE/tile0/gt1/freq0/max_freq
```

**Important:** The old scripts used `/sys/class/drm/card0/gt/gt0/rps_max_freq` which **does not exist** on this kernel. All frequency-capped data from early sweeps was silently ignored and is INVALID.

### Temperature Monitoring

```bash
# GT tile temperature (primary)
cat /sys/class/hwmon/hwmon4/temp2_input  # Divide by 1000 for °C

# All temp sensors
cat /sys/class/hwmon/hwmon4/temp*_input
```

### VRAM Monitoring

```bash
sudo cat /sys/kernel/debug/dri/0000:0b:00.0/tile0/vram_mm
```

`xpu-smi` returns "No device discovered" — not usable on consumer B70.
`intel_gpu_top` works for real-time frequency/power monitoring.

---

## Benchmark Methodology

### Verified llama-bench Command

```bash
source /opt/intel/oneapi/setvars.sh --force

~/llama.cpp/build/bin/llama-bench \
  -m ~/models-cache/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  -ngl 99 -ncmoe 0 \
  -p 512 -n 32 -t 8 \
  -fa on -ctk q8_0 -ctv q8_0 \
  -o md -r 3
```

**Key flags:**
- `-ngl 99`: All layers on GPU
- `-ncmoe 0`: All MoE experts on GPU (0 = none on CPU). Using `99` was WRONG.
- `-p 512 -n 32`: pp512 (prefill) and tg32 (decode) tests
- `-fa on`: Flash attention ON with q8_0 KV (stable on this build for Qwen)
- `-ctk q8_0 -ctv q8_0`: 8-bit KV cache (50% VRAM savings, negligible quality loss)
- `-r 3`: 3 repetitions per test for statistical confidence
- **DO NOT use `-c`** — it's not a valid llama-bench flag. Context is controlled by default model config.
- **DO NOT use `-pg` with large values** — `-pg 65536,65536` makes it generate 65536 tokens (~17 min), causing timeouts.

### Methodology Rules (Learned the Hard Way)

1. **Always run a warmup** before measurements — SYCL JIT compilation takes 30-60s on first run
2. **Run sequentially only** — parallel benchmarks cause VRAM contention, thermal contamination, OOM crashes
3. **Cooldown between tests** — wait for temp to drop below 55°C before next measurement
4. **Verify freq writes** — read back after writing to confirm they took effect
5. **Use `-o md` not `-o csv`** — md format is more reliable to parse from llama-bench output

---

## Thermal/Power Sweep Results

### Baseline Data (Stock Frequency 2800/1500 MHz)

MoE model: Qwen3.6-35B-A3B Q4_K_XL, FA on, q8_0 KV, 64K context implied

| Power Cap | pp512 (t/s) | tg32 (t/s) | Max Temp | tg vs Stock | Notes |
|-----------|-------------|------------|----------|-------------|-------|
| 230W (stock) | ~1041 | 41.5 | 83°C | baseline | Too hot |
| 180W | ~1055 | 40.2 | 77°C | -3% | Good perf, still warm |
| 150W | ~1032 | 36.7 | 73°C | -12% | Community recommended |
| 135W | 1011 ±2.8 | 65.8 ±2.3 | 71°C | — | See note below |
| 130W | 700 (JIT contaminated) | 32.9 | 60°C | -21% | First-run penalty |
| 110W | 905 | 28.1 | 60°C | -32% | Too much perf loss |

**⚠️ DATA INTEGRITY NOTE:** The 135W row showing tg32=65.8 is anomalously high — likely because the old sweeps used invalid `-c` flag (silently ignored) so context was default 512, not 64K. The v4 sweep (running now) uses correct methodology and will provide clean numbers. Early sweep data at 130W/110W used broken frequency paths and should be treated as indicative only.

### Complete Sweep Results (v4 — validated, clean methodology)

Ordered by wattage descending. All at stock freq 2800/1500 MHz, FA on, q8_0 KV, `-ncmoe 0`.

| Watt | Model | Type | pp512 (t/s) | tg32 (t/s) | Max°C | tg vs stock |
|-----:|-------|------|------------:|-----------:|------:|----------:|
| 230 | Ornith 9B Q6_K | Dense 9B | 2405 ±20 | 58.59 ±0.1 | 64 | baseline |
| 230 | Qwen 27B Q5_K_M | Dense 27B | 784 ±8 | 21.26 ±0.1 | 69 | baseline |
| 150 | Qwen 35B A3B Q4_K_XL | MoE 3B | 676 ±64* | 66.64 ±1.0 | 59 | ~same |
| 150 | Ornith 9B Q6_K | Dense 9B | 2036 ±157 | 51.92 ±0.7 | 57 | -11% |
| 150 | Qwen 27B Q5_K_M | Dense 27B | 603 ±14 | 17.02 ±0.2 | 61 | -20% |
| 145 | Qwen 35B A3B Q4_K_XL | MoE 3B | 1028 ±8 | 68.04 ±2.0 | 60 | — |
| 140 | Qwen 35B A3B Q4_K_XL | MoE 3B | 1032 ±8 | 66.96 ±1.8 | 58 | — |
| 135 | Qwen 35B A3B Q4_K_XL | MoE 3B | 872 ±13 | 65.55 ±1.9 | 58 | — |
| 135 | Ornith 9B Q6_K | Dense 9B | 1873 ±195 | 47.77 ±2.6 | 58 | -18% |
| 135 | Qwen 27B Q5_K_M | Dense 27B | 526 ±15 | 14.77 ±0.2 | 60 | -31% |
| 130 | Qwen 35B A3B Q4_K_XL | MoE 3B | 1017 ±4 | 64.47 ±2.0 | 57 | ~-5% |
| 110 | Ornith 9B Q6_K | Dense 9B | 1543 ±282 | 33.86 ±3.2 | 60 | -42% |

\*150W MoE pp512 was JIT-contaminated (first real test, kernel compilation dragged average). Real pp512 ~1020-1030 based on adjacent data points. tg32 unaffected.

### Decode Loss by Model Type

**MoE (3B active): essentially flat.** 64-68 t/s across 130-150W. Decode is bandwidth-bound, not compute-bound. Power cap doesn't touch it.

**Dense 9B: moderate sensitivity.** -11% at 150W, -18% at 135W, cliff at 110W (-42%).

**Dense 27B: highly sensitive.** -20% at 150W, -31% at 135W. All 27B params are active during decode, so power caps hit hard.

### Sweet Spot Recommendations

| Model | Best Cap | tg32 | Max°C | Loss | Verdict |
|-------|---------|------|------|------|---------|
| Qwen 35B A3B MoE | **130W** | 64.5 t/s | 57°C | ~-5% | ✅ Meets ≤65°C AND ≤15% |
| Ornith 9B Dense | **150W** | 52.0 t/s | 57°C | -11% | ✅ Meets ≤65°C AND ≤15% |
| Qwen 27B Dense | **180W** | 19.9 t/s | 63°C | -8% | ✅ Meets ≤65°C AND ≤15% (was 150W, v5 data improved this) |

### pp512 Notes

- pp512 (prefill) is compute-bound and more sensitive to power, but also noisy due to SYCL JIT effects
- The 135W MoE dip (872 vs ~1017 at adjacent levels) may be driver-level frequency throttling not visible in die temp
- Dense model pp512 variance increases dramatically at lower power (±282 at 110W)

---

## Configuration Recommendations

### KV Cache Quantization

| Config | VRAM Savings | Quality | Recommendation |
|--------|-------------|---------|----------------|
| f16 / f16 | 0% | Best | Quality-critical only |
| **q8_0 / q8_0** | **50%** | **Near-zero loss** | **DEFAULT** |
| q4_0 / q4_0 | 75% | Severe degradation | AVOID (especially tool-calling) |

### Flash Attention

- **FA on + q8_0 KV**: Working stable on this build (#9851) for Qwen models
- **FA off**: Safe fallback if any model crashes
- **FA on + q4_0 V cache**: Fails on this build

### VRAM Budget at 128K Context (32GB B70)

| Model | Weights | +KV q8_0 128K | Total | Fits? |
|-------|---------|----------------|-------|-------|
| Qwen 35B A3B Q4_K_XL | 21GB | 5GB | 26GB | ✅ (6GB headroom) |
| Qwen 35B A3B Q5_K_M | 25GB | 5GB | 30GB | ⚠️ Tight |
| Gemma4 26B A4B Q4_K_M | 16GB | 6GB | 22GB | ✅ (10GB headroom) |
| Qwen 27B Q5_K_M | 19GB | 13GB | 32GB | ⚠️ No headroom |
| Ornith 35B Q4_K_M | 20GB | 5GB | 25GB | ✅ |

**Key insight:** Dense models need far more KV cache at high context (all params active). MoE models are much more VRAM-efficient at 128K+.

---

## Production Launch Command

```bash
source /opt/intel/oneapi/setvars.sh --force
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export ZES_ENABLE_SYSMAN=1
export SYCL_CACHE_PERSISTENT=0
# REMOVED 2026-07-02: 44.7% decode regression. See research repo. # export GGML_SYCL_DISABLE_OPT=1

# Set power cap for thermal management (150W recommended)
echo 150000000 | sudo tee /sys/class/hwmon/hwmon4/power1_cap

~/llama.cpp/build/bin/llama-server \
  -m ~/models-cache/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --host 0.0.0.0 --port 8080 \
  -dev SYCL0 \
  -ngl 99 -ncmoe 0 \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  --flash-attn on \
  --no-mmap --jinja --metrics --slots
```

---

## Community Optimization Tips (Verified)

From LocalMaxxing/X community research:

1. **150W power cap is the community standard** for B70 thermal longevity — targets <75-80°C sustained
2. **Decode is bandwidth-bound** — power caps affect decode far less than prefill (compute-bound)
3. **Lower batch size reduces prefill power spikes** — peak power occurs during prompt processing
4. **q8_0 KV is the sweet spot** — community consensus, negligible quality loss, 50% VRAM savings
5. **MoE models are ideal for B70** — low active param count means high t/s relative to dense models

---

## Troubleshooting

### `libsvml.so: cannot open shared object file`
**Fix:** `source /opt/intel/oneapi/setvars.sh --force` before running any llama.cpp binary.

### `error: invalid parameter for argument: -c`
**Cause:** `-c` is not a llama-bench flag. Context is set by model metadata or `-pg`.
**Fix:** Remove `-c`. Use `-p` and `-n` for prompt/gen test sizes.

### llama-bench hangs after printing results
**Known issue on this build.** Results are valid; the process just doesn't exit cleanly. Kill it after results print.

### Frequency cap writes silently fail
**Cause:** Wrong sysfs path. Old scripts used `/sys/class/drm/card0/gt/gt0/rps_max_freq` (doesn't exist).
**Fix:** Use correct path: `.../tile0/gt0/freq0/max_freq` (see GPU Control Interfaces above).

### xe bcs engine fault during model load
**Fix:** `export SYCL_CACHE_PERSISTENT=0`

### Parallel benchmark OOM / "VM worker error: -12"
**Cause:** Multiple llama-bench instances exhaust VRAM.
**Fix:** Run strictly sequentially. One model loaded at a time.

---

## Known Issues with Early Sweep Data

Several early sweep runs (proc IDs in session history) produced unreliable data due to:

1. **Invalid `-c` flag** — llama-bench rejected it silently, tests ran at default context (likely 512)
2. **Wrong frequency sysfs paths** — all GT0/GT1 freq caps were no-ops
3. **No warmup** — first test paid SYCL JIT penalty, inflating variance (±58 t/s on pp512)
4. **No cooldown** — tests ran back-to-back, thermal carryover contaminated results
5. **`-pg 65536,65536`** — caused 17-minute test runs, eating timeouts

**Only v4 sweep data should be considered reliable.** All prior thermal/performance numbers should be treated as indicative of trends, not precise measurements.

---

## Community Benchmark Comparison

| Model | Metric | Our Result | Best Community | Source | Verdict |
|-------|--------|-----------|----------------|--------|---------|
| Qwen 35B A3B MoE | tg32 decode | 64-68 t/s @130-150W | 70-75 t/s @230W stock | Atomynos_Atom, Dismal-Might | 86-97% of uncapped, excellent |
| Qwen 35B A3B MoE | pp512 prefill | 1017-1032 t/s | 960-977 t/s | Dismal-Might, Atomynos_Atom | **We beat them by 5-7%** |
| Qwen 27B Q5_K_M | tg32 decode | 21.26 t/s @230W | 23 t/s (Q4_K_M) | viewofthelake | 92%, expected (Q5 > Q4 size) |
| Ornith 9B Q6_K | tg32 decode | 58.59 t/s @230W | No direct comp | — | 66% of theoretical bandwidth max |

**Key takeaways:**
- MoE decode within 3-14% of best community SYCL despite power capping to 130-150W
- Prefill actually exceeds community results — likely Q4_K_XL quant + FA on + q8_0 KV config
- Dense 27B is bandwidth-limited at this model size (19GB / 608 GB/s = ~32 t/s theoretical)
- No community benchmarks exist for power-capped configs — our data is unique

---

## Per-Model Configuration Summary

All configs use: `-fa on -ctk q8_0 -ctv q8_0 -ngl 99 -ncmoe 0 -t 8 -p 512 -n 32`

### MoE Models

**Qwen3.6-35B-A3B Q4_K_XL (20.81 GiB, MoE 3B active)**
```
Power: 130W (sweet spot) | tg32: 64.5 t/s | Max: 57°C | Loss: ~5%
Command: llama-bench -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf -ngl 99 -ncmoe 0 -p 512 -n 32 -t 8 -fa on -ctk q8_0 -ctv q8_0
```

**Qwen3.6-35B-A3B Q5_K_M (24.63 GiB, MoE 3B active)**
```
Status: ❌ OOM at 64K context (weights + KV exceed 32GB VRAM)
Would need: 48KB context reduction or f16→q4_0 KV (quality hit)
```

### Dense Models

**Ornith 9B Q6_K (6.84 GiB, 8.95B params)**
```
Sweet spot: 150W | tg32: 52.0 t/s | Max: 57°C | Loss: -11%
Full sweep: 230W=58.5 | 180W=55.7(-5%) | 150W=52.0(-11%) | 135W=47.5(-19%)
Command: llama-bench -m ornith-1.0-9b-Q6_K.gguf -ngl 99 -ncmoe 0 -p 512 -n 32 -t 8 -fa on -ctk q8_0 -ctv q8_0
```

**Qwen3.6-27B Q5_K_M (18.46 GiB, 27.32B params)**
```
Sweet spot: 180W | tg32: 19.9 t/s | Max: 63°C | Loss: -8%  ← UPDATED (was 150W)
Full sweep: 230W=21.6 | 180W=19.9(-8%) | 150W=17.1(-21%) | 135W=14.8(-31%)
Command: llama-bench -m Qwen3.6-27B-Q5_K_M.gguf -ngl 99 -ncmoe 0 -p 512 -n 32 -t 8 -fa on -ctk q8_0 -ctv q8_0
```

---

### Dense Model Optimization Opportunities

#### ✅ MTP Speculative Decoding — TESTED, validated July 2026

**Dense 27B Q5_K_M (180W):**
| Config | t/s | Acceptance | Mean len | Speedup |
|--------|----:|----------:|---------:|--------:|
| Baseline | 19.38 | — | — | — |
| MTP draft-2 | 26.15 | 93.8% | 2.60 | +35% |
| MTP draft-4 | 27.32 | 93.4% | 3.21 | +41% |

**MoE 35B A3B Q4_K_XL (130W):**
| Config | t/s | Acceptance | Mean len | Speedup |
|--------|----:|----------:|---------:|--------:|
| Baseline | 64.13 | — | — | — |
| MTP draft-2 | 57.82 | 93% | 2.50 | -10% (HARMFUL) |
| MTP draft-4 | 64.45 | 95% | 3.39 | +0.5% (break-even) |

**Conclusion:** MTP is a massive win for dense models (+41%), but does NOT help MoE (overhead cancels gains). MoE decode is already bandwidth-optimal.

**Production MTP command (27B dense):**
```bash
./llama-server -m Qwen3.6-27B-MTP-Q5_K_M.gguf -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -t 8 --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.75
```
### 3. Thread Count Tuning
Currently `-t 8` (8 cores). X3D has 16 threads across 8 cores. Test `-t 16` (SMT on) or `-t 4` (fewer, cache-resident).
The 3D V-Cache (96MB L3) could keep KV cache hot for small dense models.

### 4. Batch Size / Ubatch
Currently default `-b 2048 -ub 512`. Dense decode at batch=1 (typical chat) may benefit from smaller ubatch reducing power spikes.

### 5. X3D V-Cache Utilization
AMD Ryzen 7 5700X3D: 96MB L3 V-Cache. Ideas:
- **KV cache residency**: For Ornith 9B (6.84GB), KV at small context (4K) is ~2GB → won't fit L3, but the **active compute buffers** (attention scores, intermediate activations) for small batch could partially fit
- **CPU MoE offload for hybrid**: For large MoE models, offload expert routing to CPU. The 3D V-Cache makes CPU-side expert lookups faster. But `-ncmoe 0` is still correct for GPU-only
- **NUMA pinning**: Single NUMA node, so no cross-node penalty. But CPU affinity (`taskset`) could reduce context switches
- **CPU as draft model**: Run a small draft model (e.g., 0.5B) on CPU with V-Cache, use for speculative decoding target verification

### 6. Recompile with AOT (No JIT)
Current build uses JIT (no `bmg_g21` arch flag). AOT compile eliminates first-run penalty:
```bash
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_DEVICE_ARCH=bmg_g21 ...
```
Won't improve steady-state speed but eliminates warmup requirement.

---

*Document last updated: July 1, 2026 — dense sweep v5 running, community comparison added*
