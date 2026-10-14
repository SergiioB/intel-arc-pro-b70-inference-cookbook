---
title: "Intel Arc Pro B70 — SYCL Inference Quick Guide"
type: doc
date: 2026-07-02
status: final
---

# Intel Arc Pro B70 — SYCL Inference Quick Guide

This guide is for running and testing GGUF models on the desktop Intel Arc Pro B70 using `llama.cpp` SYCL.

## Known-good baseline

```text
OS: Ubuntu 26.04 LTS
Kernel: 7.0.0-27-generic
GPU: Intel Arc Pro B70 / Battlemage G31 / 32GB
Driver: xe
Backend: llama.cpp SYCL / Level Zero
Working llama.cpp: build-sycl-latest-l0, version b9853 / 7af4279f4
```

## Critical fix

Always disable persistent SYCL cache:

```bash
export SYCL_CACHE_PERSISTENT=0
```

Without this, `llama.cpp` SYCL may segfault during model load because of a SYCL persistent-cache bug with dynamically loaded `libggml-sycl.so`.

## Required environment

Before running any test:

```bash
source /opt/intel/oneapi/setvars.sh --force
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export ZES_ENABLE_SYSMAN=1
export SYCL_CACHE_PERSISTENT=0
unset ZEX_NUMBER_OF_CCS SYCL_UR_USE_LEVEL_ZERO_V2
```

Do **not** set `ZEX_NUMBER_OF_CCS=0:4` or `0:8` on this stack. It made `sycl-ls` return `No platforms found` during testing.

## Verify GPU visibility

```bash
source /opt/intel/oneapi/setvars.sh --force
sycl-ls
```

Expected B70 line:

```text
[level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero V2, Intel(R) Graphics [0xe223]
```

Check BAR mapping:

```bash
lspci -vv -s 0b:00.0 | grep -E "Region 0|Region 2"
```

Expected:

```text
Region 2 ... [size=32G]
```

## Binaries

SYCL build:

```bash
~/llama.cpp/build-sycl-latest-l0/bin/llama-server
~/llama.cpp/build-sycl-latest-l0/bin/llama-bench
~/llama.cpp/build-sycl-latest-l0/bin/llama-cli
```

Vulkan fallback exists but is not the preferred backend:

```bash
~/llama.cpp/build-vulkan/bin/llama-server
```

## Models

Fast SSD cache:

```bash
~/models-cache/
```

Cold/storage model directory:

```bash
/mnt/<models-disk>/B70-Models/
```

Prefer copying any model you are benchmarking repeatedly to `~/models-cache/` first:

```bash
mkdir -p ~/models-cache
cp /mnt/<models-disk>/B70-Models/MODEL.gguf ~/models-cache/
```

## Quick benchmark for a new model

Start with a small synthetic benchmark:

```bash
source /opt/intel/oneapi/setvars.sh --force
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export ZES_ENABLE_SYSMAN=1
export SYCL_CACHE_PERSISTENT=0
unset ZEX_NUMBER_OF_CCS SYCL_UR_USE_LEVEL_ZERO_V2

~/llama.cpp/build-sycl-latest-l0/bin/llama-bench \
  -m /path/to/model.gguf \
  -ngl 99 \
  -p 128 \
  -n 32 \
  -r 1 \
  -o md
```

If this fails, do not start a server yet.

## High-context benchmark template

Use this to test context pressure:

```bash
~/llama.cpp/build-sycl-latest-l0/bin/llama-bench \
  -m /path/to/model.gguf \
  -ngl 99 \
  -p 512,2048,4096 \
  -n 128 \
  -r 1 \
  -c 32768 \
  -o md
```

For 64K/128K testing, use the API server so we can measure TTFT/TTLT with real prompts and watch memory/power.

## Start a server for a new model

```bash
source /opt/intel/oneapi/setvars.sh --force
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export ZES_ENABLE_SYSMAN=1
export SYCL_CACHE_PERSISTENT=0
unset ZEX_NUMBER_OF_CCS SYCL_UR_USE_LEVEL_ZERO_V2

~/llama.cpp/build-sycl-latest-l0/bin/llama-server \
  -m /path/to/model.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --api-key test-key \
  -dev SYCL0 \
  -ngl 99 \
  -c 16384 \
  --parallel 1 \
  --no-mmap \
  --flash-attn off \
  --cache-type-k q8_0 \
  --reasoning off \
  --metrics \
  --slots \
  --jinja
```

Increase context only after a smaller context is stable:

```text
16384 -> 32768 -> 65536 -> 131072
```

## API test

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer test-key' \
  -d '{
    "model": "b70-test",
    "messages": [{"role": "user", "content": "Answer with exactly this word: OK"}],
    "max_tokens": 16,
    "temperature": 0,
    "stream": false
  }'
```

## Metrics

Health:

```bash
curl -s http://127.0.0.1:8080/health
```

Throughput metrics:

```bash
curl -s http://127.0.0.1:8080/metrics | grep -E 'prompt_tokens_seconds|predicted_tokens_seconds|tokens_predicted_total|prompt_tokens_total'
```

## Temperature

```bash
for f in /sys/class/drm/card*/device/hwmon/hwmon*/temp*_input; do
  [ -e "$f" ] && echo "$f=$(cat "$f")"
done
```

Values are millidegrees Celsius. `59000` means 59°C.

## Power

Try Intel tools first:

```bash
xpu-smi discovery
xpu-smi stats -d 0
```

If `xpu-smi` is not reliable, inspect hwmon:

```bash
for f in /sys/class/drm/card*/device/hwmon/hwmon*/power*_average /sys/class/drm/card*/device/hwmon/hwmon*/energy*_input; do
  [ -e "$f" ] && echo "$f=$(cat "$f")"
done
```

Power readings may not be exposed consistently on this B70 stack. If direct readings fail, use wall-power or package-level telemetry as fallback.

## Current known-good production command

```bash
/srv/inference/bin/launch-llama-hermes-sycl.sh
```

Important flags in that launcher:

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export ZES_ENABLE_SYSMAN=1
export SYCL_CACHE_PERSISTENT=0
unset ZEX_NUMBER_OF_CCS SYCL_UR_USE_LEVEL_ZERO_V2
-dev SYCL0
-ngl 99
-c 16384
--parallel 1
--no-mmap
--flash-attn off
--cache-type-k q8_0
--reasoning off
```

## Baseline results already verified

```text
Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
SYCL, -ngl 99
llama-bench pp128: 386.89 t/s
llama-bench tg32:   68.45 t/s
API 256-token test: ~60.54 tok/s wall-clock, 66.70 tok/s reported generation
```

```text
ornith-1.0-9b-Q6_K.gguf
SYCL, -ngl 99
llama-bench pp128: 1306.32 t/s
llama-bench tg32:    60.73 t/s
```

## Failure signatures

Persistent cache bug:

```text
SIGSEGV during model load, even with -ngl 0
```

Fix:

```bash
export SYCL_CACHE_PERSISTENT=0
```

Bad env on this stack:

```bash
ZEX_NUMBER_OF_CCS=0:4
ZEX_NUMBER_OF_CCS=0:8
SYCL_UR_USE_LEVEL_ZERO_V2=0
```

Symptom:

```text
sycl-ls: No platforms found
```

## Recommended benchmark workflow

For each new model:

1. Copy to SSD cache if it will be reused.
2. Run `llama-bench -p 128 -n 32`.
3. Run `llama-bench -p 512,2048 -n 128`.
4. Start server at 16K context.
5. Run API TTFT/TTLT test.
6. Increase context to 32K, 64K, 128K as VRAM allows.
7. Record:
   - pp t/s
   - tg t/s
   - TTFT
   - TTLT
   - generated tokens
   - prompt tokens
   - peak temp
   - average/peak wattage if exposed
8. Only then tune power limits.
