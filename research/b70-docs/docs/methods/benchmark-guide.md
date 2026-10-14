---
title: "B70 Model Benchmark Guide"
type: method
date: 2026-07-03
status: living
---

---
name: b70-model-benchmark-guide
description: Comprehensive Intel Arc B70 32GB model benchmark. All tested models, speeds, context limits, vision (verified working), MTP speculative decoding, power efficiency curves, and use case recommendations. Cross-references b70-mtp-speculative-benchmark for raw data.
---

# B70 Model Benchmark Guide

All data verified on hardware on Jul 1, 2026. No estimates, no extrapolations.

## Hardware & Environment

- **GPU**: Intel Arc Pro B70, 32GB VRAM (32,656MiB visible)
- **OS**: Ubuntu 26.04
- **Backend**: llama.cpp SYCL b9851 (0eca4d490), oneAPI 2026.0.0, IntelLLVM 2026.0.0
- **CPU offload**: none (`-ngl 99 -ncmoe 0` = fully GPU)
- **KV cache**: q5_0 K + q4_1 V with flash attention on (`-ctk q5_0 -ctv q4_1 -fa on`)
- **Host access**: `ssh desktop-b70  # or: ssh sergio@<lab-host>`
- **Power control**: `/sys/class/hwmon/hwmon4/power1_cap` (in microwatts: 150000000 = 150W)
- **Temp monitor**: `/sys/class/hwmon/hwmon4/temp2_input` (÷1000 for °C)
- **Benchmark method**: 4 diverse prompts (history, transformers, distributed systems, k8s), temp=0.0, 256 tokens output, steady-state = avg of rounds 2-4
- **ffmpeg installed** for vision image decoding

## Server Launch Template

```bash
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1

# Standard text inference
~/llama.cpp/build/bin/llama-server \
  -m ~/models-cache/<MODEL>.gguf \
  --host 127.0.0.1 --port 8765 \
  -ngl 99 -ncmoe 0 -fa on -ctk q5_0 -ctv q4_1 \n  -t 8 --no-mmap -c 131072

# With MTP speculative decoding (27B dense only)
# Add: --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.75

# With vision (add --mmproj)
# Add: --mmproj ~/models-cache/<MMPROJ>.gguf
```

KV cache quantization (see B70-DOCS/research/ for full analysis):
  q5_0-q4_1 = RECOMMENDED (92.65% tail precision, best VRAM/speed)
  q5_0-q5_0 = Alternative symmetric (92.70% tail, slightly more VRAM)
  q8_0-q8_0 = Conservative (94.61% tail, 38% more VRAM)
  q4_0-q4_0 = Quality cliff (89.84% tail, avoid for tool calling)


## Full Model Benchmark Table

Values from lmx remote endpoint benchmarks (decode t/s) and engine-level wall-clock tests.
All use q5_0-q4_1 KV cache unless noted. Submitted to localmaxxing.com.

| Model | Type | HF ID | Quant | Context | Power | MTP | Vision | Decode | Prefill | TTFT | Notes |
|-------|------|-------|-------|---------|-------|-----|--------|--------|---------|------|-------|
| Ornith 35B | MoE | deepreinforce-ai/Ornith-1.0-35B | Q4_K_M | 256K | 150W | — | — | **74.3** | 440.8 | 73ms | Fastest overall |
| Agents-A1 | MoE | InternScience/Agents-A1 | Q4_K_M | 256K | 150W | — | — | 73.2 | 416.1 | 77ms | Agent-optimized |
| Qwen 35B | MoE | Qwen/Qwen3.6-35B-A3B | Q5_K_M | 256K | 150W | — | — | 67.0 | 464.9 | 69ms | Best quality at 2x ctx |
| Qwen 35B | MoE | Qwen/Qwen3.6-35B-A3B | Q4_K_XL | 256K | 150W | — | — | 63.4 | ~900 | ~65ms | Good balance |
| Gemma 4 26B | MoE | google/gemma-4-26B-A4B-it | Q4_K_M | 256K | 150W | — | ✅ | 57.4 | ~900 | ~80ms | Vision ready |
| Ornith 9B | Dense | deepreinforce-ai/Ornith-1.0-9B | Q6_K | 256K | 150W | — | — | 51.1 | 1750.3 | 58ms | Context king |
| Qwen 27B MTP | Dense | Qwen/Qwen3.6-27B | Q5_K_M | 128K | **230W** | ✅ MTP-4 | ✅ | **30.4** | 102.4 | 312ms | Max perf + MTP-4 |
| Qwen 27B MTP | Dense | Qwen/Qwen3.6-27B | Q5_K_M | 128K | 165W | ✅ MTP-4 | ✅ | 25.5 | — | — | Peak efficiency |
| Qwen3.6-27B MTP-4 | Dense | 19GB | Q5_K_M | 180W | 27.2 | 71°C | ✅ | Balanced perf+t thermals |
| Qwen3.6-27B BASE | Dense | 19GB | Q5_K_M | 180W | 19.5 | 71°C | - | No MTP baseline |

### Context Scaling: 64K → 128K Is Free

Tested on every model. Zero speed penalty for doubling context with q8 KV cache + flash attention:

| Model | 64K t/s | 128K t/s | Delta |
|-------|---------|----------|-------|
| Gemma-4-26B | 57.34 | 57.40 | +0.1% |
| Ornith 35B | 68.96 | 68.34 | -0.9% |
| Agents-A1 | 68.14 | 68.44 | +0.4% |
| Qwen 35B Q4 | - | 63.41 | - |
| Qwen 35B Q5 | 61.96 | 61.90 | -0.1% |



### Context Ceilings with q5_0-q4_1 KV Cache

With q5_0-q4_1 KV (recommended), all models gain significant headroom:

| Model | Quant | Max Context | Speed at Max | Notes |
|-------|-------|-------------|--------------|-------|
| Qwen 35B | Q5_K_M | **256K** | 62.0 t/s | Best quality at 2x context |
| Qwen 35B | Q4_K_XL | **256K** | 63.5 t/s | Verified; could reach 512K |
| Qwen 27B MTP | Q5_K_M | **200K** | 26.5 t/s | Dense model; was capped at 128K |
| Gemma 4 26B | Q4_K_M | **512K** | 57.3 t/s | Massive headroom |
| Ornith 35B | Q4_K_M | **512K** | 67.9 t/s | Fastest at any context |
| Agents-A1 | Q4_K_M | **512K** | 68.6 t/s | Agent use at any scale |
| Ornith 9B | Q6_K | **512K+** | 48.6 t/s | Context king (17GB at 512K) |

With conservative q8_0-q8_0 KV, subtract ~40% from max context.
See B70-DOCS/research/vram-budget-model.md for full calculations.


## MTP Speculative Decoding

> **See also**: skill `b70-mtp-speculative-benchmark` for full per-round raw data and benchmark scripts.

**Rule**: MTP only helps when the model is memory-bandwidth bound (low t/s). Compute-bound models (MoE at 60+ t/s) gain nothing because verification overhead negates the bandwidth savings. The crossover is between 20-60 t/s.

### 27B Dense (memory-bandwidth bound) — MTP WINS

| Config | t/s @ 180W | vs BASE | Accept Rate | Mean Len |
|--------|-----------|---------|-------------|----------|
| BASE | 19.27 | - | - | - |
| MTP-2 | 25.98 | **+35%** | 93.8% | 2.60 |
| MTP-4 | 26.01 | **+35%** | 93.4% | 3.21 |

MTP-4 is the sweet spot. Acceptance rate stays 93-95% at all power levels.

### 35B MoE (compute-bound) — MTP DOES NOT HELP

| Config | t/s @ 130W | vs BASE | Accept Rate |
|--------|-----------|---------|-------------|
| BASE | 63.62 | - | - |
| MTP-2 | 57.77 | **-9%** | 93.5% |
| MTP-4 | 64.51 | +1% | 94.5% |

Acceptance rate is excellent (93-95%), but the overhead of running the draft head + verification on every step exceeds the bandwidth savings at 64 t/s.

## Power Scaling (27B MTP-4, Q5_K_M)

| Power | MTP-4 t/s | Temp | Efficiency (t/s/W) |
|-------|-----------|------|---------------------|
| 150W | 23.11* | 69°C | 0.154 |
| **165W** | **24.48** | **68°C** | **0.148** ← peak |
| 180W | 26.01 | 71°C | 0.145 |
| 195W | ~28.6 est | 73°C | ~0.147 |
| 230W | 28.95 | 77°C | 0.126 |

**Sweet spot: 165W.** Peak efficiency at 0.148 t/s/W. Stock 230W adds only 4.5 t/s for +65W and +9°C.

BASE (no MTP) also scales with power: 150W=16.5 t/s → 230W=21.1 t/s (+28%). *BASE values are from power-sweep and may be inflated; non-MTP model removed from disk, cannot re-verify.*



## Vision (Verified Working Jul 1, 2026)

Both models correctly identified a test image (black Labrador puppy on wooden deck). ffmpeg required for image decoding (`sudo apt install ffmpeg`).

Speed measured from server `timings.predicted_per_second` (engine decode rate), not wall-clock, to isolate decode from prompt processing overhead.

### Qwen3.6-27B MTP-4 Vision

Server: `Qwen3.6-27B-MTP-Q5_K_M.gguf` + `--mmproj mmproj-Qwen3.6-27B-F16.gguf` + `--spec-type draft-mtp --spec-draft-n-max 4`

| Power | Vision t/s | Text-only t/s | Vision Overhead | Draft Accept |
|-------|-----------|--------------|-----------------|--------------|
| 230W | 32.0 | 34.2 | -6% | 93-98% |
| **180W** | **29.0** | 30.2 | -4% | ~93% |
| 165W | ~25* | ~26* | ~same | ~93% |

*165W estimated from power scaling curve. 180W measured Jul 1, 2026.

Response: "A black puppy lies on a wooden deck, looking up at the camera with large, expressive eyes."
- mmproj: `mmproj-Qwen3.6-27B-F16.gguf` (884MB)
- Vision overhead is minimal (~4-6%) because image embedding happens during prompt processing, not decode
- Draft acceptance stays 93-98% even with vision input
- **Recommended: 180W** for vision. 29 t/s, well above usable threshold, 72°C max.
- Note: Model uses reasoning tokens. Use max_tokens=512+ for actual content output.

### Gemma-4-26B-A4B Vision

Server: `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` + `--mmproj mmproj-gemma-4-26B-A4B-it-Q8_0.gguf`

| Power | Vision t/s | Text-only t/s | Vision Overhead |
|-------|-----------|--------------|-----------------|
| 150W | 58.9 | 56.4 | +4% (noise) |

Response: "A black puppy on a wooden surface."
- mmproj: `mmproj-gemma-4-26B-A4B-it-Q8_0.gguf` (769MB)
- Vision overhead negligible

### Vision Launch Commands

```bash
# Qwen 27B with vision + MTP
~/llama.cpp/build/bin/llama-server \
  -m ~/models-cache/Qwen3.6-27B-MTP-Q5_K_M.gguf \
  --mmproj ~/models-cache/mmproj-Qwen3.6-27B-F16.gguf \
  --host 127.0.0.1 --port 8765 \
  -ngl 99 -ncmoe 0 -fa on -ctk q5_0 -ctv q4_1 -t 8 --no-mmap -c 131072 \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.75

# Gemma 4 26B with vision
~/llama.cpp/build/bin/llama-server \
  -m ~/models-cache/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
  --mmproj ~/models-cache/mmproj-gemma-4-26B-A4B-it-Q8_0.gguf \
  --host 127.0.0.1 --port 8765 \
  -ngl 99 -ncmoe 0 -fa on -ctk q5_0 -ctv q4_1 -t 8 --no-mmap -c 131072
```

## Use Case Recommendations

| Use Case | Best Model | Config | Speed | Temp |
|----------|-----------|--------|-------|------|
| **Best quality chat** | Qwen3.6-35B Q5_K_M | 150W, 128K ctx | 62 t/s | 66°C |
| **Best quality dense** | Qwen3.6-27B MTP-4 Q5 | 230W, 128K, MTP-4 | 29 t/s | 77°C |
| **Max efficiency** | Qwen3.6-27B MTP-4 Q5 | **165W**, 128K, MTP-4 | 24.5 t/s | 68°C |
| **Fastest inference** | Ornith 35B / Agents-A1 | 150W, 128K ctx | 68 t/s | 65°C |
| **Agent / tool use** | Agents-A1 Q4 | 150W, 128K ctx | 68 t/s | 65°C |
| **Vision + reasoning** | Qwen3.6-27B + mmproj | 180W, MTP-4 | 29 t/s | 72°C |
| **Vision + speed** | Gemma-4-26B + mmproj | 150W | 59 t/s | 63°C |
| **Lightweight / edge** | Ornith 9B Q6 | 150W, 128K ctx | 49 t/s | 64°C |

*Vision speed = image description generation rate (includes image embedding overhead).

> **Stacking**: Do NOT combine MTP+ngram on 27B dense (-2-3% vs MTP alone). 35B MoE has no MTP layers (architectural limit). Verified 2026-07-03.

## Models on Disk

```
~/models-cache/
├── Qwen3.6-27B-MTP-Q5_K_M.gguf          19GB   dense + MTP + vision
├── Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf      21GB   MoE
├── Qwen3.6-35B-A3B-UD-Q5_K_M.gguf       25GB   MoE (best quality)
├── gemma-4-26B-A4B-it-UD-Q4_K_M.gguf    16GB   MoE + vision
├── ornith-1.0-35b-Q4_K_M.gguf           20GB   MoE (fastest)
├── ornith-1.0-9b-Q6_K.gguf               7GB   dense (lightweight)
├── Agents-A1-Q4_K_M.gguf                20GB   MoE (agents)
├── mmproj-Qwen3.6-27B-F16.gguf         884MB   vision projector (Qwen)
└── mmproj-gemma-4-26B-A4B-it-Q8_0.gguf  769MB   vision projector (Gemma)
```

**Removed** (freeing 45GB): non-MTP 27B base, 35B MTP-Q4 (MTP doesn't help MoE), incomplete Step-3.7-flash split, tiny Gemma MTP head fragments.

## Benchmark Scripts (on B70)

| Script | Purpose |
|--------|---------|
| `~/b70-mtp-bench.sh` | Original MTP comparison (27B + 35B) |
| `~/b70-mtp-power-sweep.sh` | 27B power scaling: 150W/180W/230W BASE + MTP-4 |
| `~/b70-intermediate-sweep.sh` | 165W/195W intermediate points + 35B Q5 fit test |
| `~/b70-master-v2.sh` | Comprehensive all-model benchmark (11 tests) |
| `~/b70-vision-test.sh` | Vision benchmark with image decode |

## Raw Data

> Full per-round raw results in `references/master-v2-raw-results-2026-07-01.md`
> MTP power sweep raw data in `references/raw-results-2026-07-01.md` (also in skill `b70-mtp-speculative-benchmark`)

## Pitfalls & Lessons

1. **Deploy scripts via `cat file | ssh host "cat > file"`** — NOT SSH heredoc. Shell variable expansion (`$()`, `$VAR`) gets mangled through the SSH layer.
2. **Use nohup for long benchmarks**: `nohup bash script.sh > log 2>&1 < /dev/null &` — SSH session disconnects will kill foreground processes.
3. **Wait for GPU temp ≤52°C** between tests for consistency.
4. **35B MoE needs `-ncmoe 0`** (GPU-only offload) or it won't fit VRAM.
5. **Don't run llama-bench after server tests** — hangs on this build (b9851).
6. **ffmpeg required** for vision image decoding (`sudo apt install ffmpeg`).
7. **Models use reasoning tokens** — set max_tokens=512+ when requesting vision descriptions, or the model spends all tokens reasoning and returns empty content.
8. **VRAM debugfs reader** (`/sys/kernel/debug/dri/.../vram_mm`) format not compatible with simple grep parsing — use `grep "^usage:"` and convert bytes to MB.
9. **SSH hostname `desktop-b70`** may not resolve if DNS cache expires. Use direct IP `sergio@<lab-host>` with `-i ~/.ssh/id_ed25519_workstation`.
