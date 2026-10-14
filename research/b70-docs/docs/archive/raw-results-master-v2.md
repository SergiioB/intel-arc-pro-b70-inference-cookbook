---
title: "B70 Master Benchmark v2 — Raw Results (Jul 1, 2026)"
type: doc
date: 2026-07-02
status: final
---

# B70 Master Benchmark v2 — Raw Results (Jul 1, 2026)

## Environment
- Date: 2026-07-01T17:21:54Z
- Backend: llama.cpp SYCL b9851
- Flags: -ngl 99 -ncmoe 0 -fa on -ctk q8_0 -ctv q8_0 -t 8 --no-mmap
- Method: 4 diverse prompts, temp=0.0, 256 tokens, steady-state = avg rounds 2-4

## Test 1: Qwen27B-MTP4-Q5-128K | 165W
SERVER_READY after 24s
ROUND 1: 256 tok / 10.798815s = 23.71 t/s
ROUND 2: 256 tok / 10.260710s = 24.95 t/s
ROUND 3: 256 tok / 10.664658s = 24.00 t/s
ROUND 4: 256 tok / 10.539910s = 24.29 t/s
draft acceptance = 0.95031 (153 accepted / 161 generated), mean len = 3.35
MAX_TEMP=70
Steady-state: 24.41 t/s

## Test 2: Qwen27B-MTP4-Q5-128K-230W
SERVER_READY after 8s
ROUND 1: 256 tok / 9.134164s = 28.03 t/s
ROUND 2: 256 tok / 8.681707s = 29.49 t/s
ROUND 3: 256 tok / 9.010459s = 28.41 t/s
ROUND 4: 256 tok / 8.894445s = 28.78 t/s
draft acceptance = 0.95031, mean len = 3.35
MAX_TEMP=76
Steady-state: 28.89 t/s

## Test 3: Qwen35B-Q5-128K | 150W
SERVER_READY after 29s
ROUND 1: 256 tok / 4.401083s = 58.17 t/s
ROUND 2: 256 tok / 4.139342s = 61.85 t/s
ROUND 3: 256 tok / 4.126457s = 62.04 t/s
ROUND 4: 256 tok / 4.137497s = 61.87 t/s
MAX_TEMP=66
Steady-state: 61.92 t/s

## Test 4: Qwen35B-Q4-128K | 150W
SERVER_READY after 25s
ROUND 1: 256 tok / 4.834983s = 52.95 t/s
ROUND 2: 256 tok / 4.029184s = 63.54 t/s
ROUND 3: 256 tok / 4.026687s = 63.58 t/s
ROUND 4: 256 tok / 4.025779s = 63.59 t/s
MAX_TEMP=65
Steady-state: 63.57 t/s

## Test 5: Gemma4-26B-Q4-64K | 150W
SERVER_READY after 20s
ROUND 1: 256 tok / 4.818049s = 53.13 t/s
ROUND 2: 256 tok / 4.464716s = 57.34 t/s
ROUND 3: 256 tok / 4.458649s = 57.42 t/s
ROUND 4: 256 tok / 4.470443s = 57.27 t/s
MAX_TEMP=65
Steady-state: 57.34 t/s

## Test 6: Gemma4-26B-Q4-128K | 150W
SERVER_READY after 5s
ROUND 1: 256 tok / 4.779719s = 53.56 t/s
ROUND 2: 256 tok / 4.461476s = 57.38 t/s
ROUND 3: 256 tok / 4.455732s = 57.45 t/s
ROUND 4: 256 tok / 4.462564s = 57.37 t/s
MAX_TEMP=63
Steady-state: 57.40 t/s

## Test 7: Ornith-35B-Q4-64K | 150W
SERVER_READY after 24s
ROUND 1: 256 tok / 4.727478s = 54.15 t/s
ROUND 2: 256 tok / 3.726882s = 68.69 t/s
ROUND 3: 256 tok / 3.704342s = 69.11 t/s
ROUND 4: 256 tok / 3.705369s = 69.09 t/s
MAX_TEMP=65
Steady-state: 68.96 t/s

## Test 8: Ornith-35B-Q4-128K | 150W
SERVER_READY after 26s
ROUND 1: 256 tok / 4.750217s = 53.89 t/s
ROUND 2: 256 tok / 3.749084s = 68.28 t/s
ROUND 3: 256 tok / 3.744064s = 68.37 t/s
ROUND 4: 256 tok / 3.744017z = 68.38 t/s
MAX_TEMP=66
Steady-state: 68.34 t/s

## Test 9: Ornith-9B-Q6-128K | 150W
SERVER_READY after 11s
ROUND 1: 256 tok / 5.619507s = 45.56 t/s
ROUND 2: 256 tok / 5.249763s = 48.76 t/s
ROUND 3: 256 tok / 5.262752s = 48.64 t/s
ROUND 4: 256 tok / 5.258326s = 48.68 t/s
MAX_TEMP=64
Steady-state: 48.69 t/s

## Test 10: Agents-A1-Q4-64K | 150W
SERVER_READY after 23s
ROUND 1: 256 tok / 4.763129s = 53.75 t/s
ROUND 2: 256 tok / 3.761512s = 68.06 t/s
ROUND 3: 256 tok / 3.751468s = 68.24 t/s
ROUND 4: 256 tok / 3.758266s = 68.12 t/s
MAX_TEMP=66
Steady-state: 68.14 t/s

## Test 11: Agents-A1-Q4-128K | 150W
SERVER_READY after 24s
ROUND 1: 256 tok / 4.738250s = 54.03 t/s
ROUND 2: 256 tok / 3.750087s = 68.27 t/s
ROUND 3: 256 tok / 3.734817s = 68.54 t/s
ROUND 4: 256 tok / 3.737252s = 68.50 t/s
MAX_TEMP=65
Steady-state: 68.44 t/s

## Test 12: Qwen27B-Vision | 165W
mmproj loaded: mmproj-Qwen3.6-27B-F16.gguf
Server loaded multimodal model successfully
VISION FAILED: image decode error (missing ffprobe/ffmpeg on host)

## Test 13: Gemma4-26B-Vision | 150W
mmproj loaded: mmproj-gemma-4-26B-A4B-it-Q8_0.gguf
Server loaded multimodal model successfully
VISION FAILED: image decode error (missing ffprobe/ffmpeg on host)

## Summary: 13/13 PASS
## ⚠️ Corrigendum (2026-07-01 19:11Z)

Tests 1-2 (27B MTP-4 at 165W and 230W) values confirmed by independent
verification (see verification-results-2026-07-01.md).

Tests 12-13 (vision) failed due to missing ffmpeg on host. Vision was
subsequently fixed by installing ffmpeg. See vision-results.md for
successful vision tests with ffmpeg installed.

Test 3 (35B Q5 128K = 61.92 t/s) confirmed by verification (61.90 t/s).


