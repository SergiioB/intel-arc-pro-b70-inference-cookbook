---
title: "B70 Vision Benchmark — Raw Results (Jul 1, 2026)"
type: doc
date: 2026-07-02
status: final
---

# B70 Vision Benchmark — Raw Results (Jul 1, 2026)

## Environment
- Backend: llama.cpp SYCL b9851
- ffmpeg installed on host (required for image decoding)
- Test image: 35KB JPEG, 400x600, black Labrador puppy on wooden deck
- Method: OpenAI-compatible multipart request, image base64-encoded
- Speed: engine decode rate from `timings.predicted_per_second` (NOT wall-clock)

## Qwen3.6-27B MTP-4 Vision — 230W

Server: MTP-4 (--spec-type draft-mtp --spec-draft-n-max 4)

### Vision rounds (after warmup)
V1: tokens=415 engine=32.07 t/s
V2: tokens=437 engine=31.78 t/s
V3: tokens=415 engine=31.93 t/s

### Text-only comparison
TEXT: engine=34.23 t/s

### Draft acceptance
R1: 0.92568 (274/296), mean 3.34
R2: 0.94604 (263/278), mean 3.39
R3: 0.97727 (172/176), mean 3.87

Content: "A black puppy lies on a wooden deck, looking up at the camera with large, expressive eyes."

## Qwen3.6-27B MTP-4 Vision — 180W (RECOMMENDED)

### Vision rounds (after warmup)
V1: tokens=425 engine=29.03 t/s
V2: tokens=425 engine=28.97 t/s
V3: tokens=425 engine=28.87 t/s

### Text-only comparison
TEXT: engine=30.22 t/s

Vision overhead at 180W: 29.0 vs 30.2 = only 4%.

## Gemma-4-26B-A4B Vision — 150W

### Vision (after warmup)
V1: tokens=203 engine=58.9 t/s
Content: "A black puppy on a wooden surface."

### Text-only comparison
TEXT: engine=56.36 t/s

## Summary

| Model | Power | Vision t/s | Text t/s | Overhead |
|-------|-------|-----------|----------|----------|
| Qwen 27B MTP-4 | 230W | 32.0 | 34.2 | -6% |
| Qwen 27B MTP-4 | **180W** | **29.0** | 30.2 | -4% |
| Gemma 4 26B | 150W | 58.9 | 56.4 | +4% (noise) |

## Key Notes
- Speed from `timings.predicted_per_second` (engine decode rate), NOT wall-clock.
- Both models correctly identified image content.
- Both use reasoning tokens. max_tokens=512+ required for content output.
- Image decode requires ffmpeg/ffprobe on host.
