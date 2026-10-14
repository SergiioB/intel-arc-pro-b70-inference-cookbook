---
title: "MiniMax H3 B70 workflows"
type: research
date: 2026-08-27
status: final
---

# MiniMax H3 B70 workflows

UI graphs live in Comfy at `user/default/workflows/`:

- `b70_h3_fl2va_i2v_face_turbo8.json` — FL2VA first-frame I2V (broken on this XPU packer)
- `b70_h3_ref2va_face.json` — Ref2VA identity (`<Picture 1>`) (XPU index crash on sample)
- `b70_h3_realesrgan_2k.json` — post-gen RealESRGAN → 2560×1440 (not official H3-2K)

Face stills:

- `input/sergio-face-reference.jpg` — portrait for Ref2VA
- `input/sergio-face-i2v-1344x768.jpg` — padded 16:9 for FL2VA first-frame (no stretch)
- `input/face-reference/portraits/` — same stills

## What people actually run

Official Comfy templates (`video_minimax_h3_t2v/i2v/r2v.json`):

| Mode | UNet | Sampler | Steps | Scheduler | Notes |
|------|------|---------|------:|-----------|-------|
| T2V / I2V / FLF | **FL2VA** pruned | `res_multistep` | 20 | `simple` | first/last optional on the same node |
| R2V | **Ref2VA** (different weights) | `res_multistep` | 20 | **`beta` or `normal`** | cite `<Picture 1>`; `ref_image_size` `match` then `max` if ID is weak |

Shared: 24 fps, 768 short edge, area cap 768×1344, length snaps to 17k+5 (124 ≈ 5 s). Shift 12 / 3. CLIP type `minimax`. SageAttention is CUDA-only.

NVIDIA INT8 ConvRot is the Comfy-Org default. On this XPU host that path is not available, so both UNets are **pruned FP8 scaled**.

## B70 measured recipe (T2V)

User-confirmed cinema clip `h3_cinema_turbo8_00001`:

- 1344×768, 124 frames, people LoRA 1.0, LightX2V turbo-8 **0.75**, `er_sde`, **8** steps, simple, shift 12/3, CLIP **cpu**
- ~45–53 s/it, ~11–15 min wall, card avg ~130–150 W at 180 W cap

Do **not** stack turbo-8 on Ref2VA. Those adapters are FL2V-named.

Prompt styles that work locally: one continuous action beat, bright practicals, official `[Shot N]` lists. Cartoon / papercraft / live-action are all T2V text styles — drop the people LoRA for non-photoreal.

## Host rules

CLIP on CPU. No `--highvram`. `--reserve-vram 2.5`. One H3 job. TE ~15 GB + UNet ~20 GB cannot both sit on 32 GB XPU.
