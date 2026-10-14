---
title: "MiniMax H3 dual-B70 ComfyUI (2026-08-25)"
type: research
date: 2026-08-25
status: final
---

# MiniMax H3 dual-B70 ComfyUI (2026-08-25)

Status: `active` — self-reported with raw campaign JSON.  
Label: **PROVISIONAL — NOT FOR PUBLIC HEADLINE** (n=1 per cell; no independent reproduction).

H3 remains ComfyUI + PyTorch XPU only. llama.cpp / vLLM cannot run this model.

Videos: `/mnt/<models-disk>/comfyui/output/h3-runs/run10-dual/`  
Raw: `/tmp/h3-dl/campaign10/`

## What dual-card actually is on this host

This B450-F has two Arc Pro B70s, **no GPU P2P**, GPU 0 Gen3 **x8**, GPU 1 Gen3 **x4**. ADR-0008 default is independent stacks, not tensor-parallel.

ComfyUI has **no H3 tensor-parallel node**. Official MiniMax multi-GPU is **SGLang, 4 GPUs, Ulysses** — not Comfy, no published speed. Two full H3 stacks (CLIP 15 GB + UNet 20 GB each) do not fit in 30 GB host RAM.

The dual-card product that works here is a **split**:

| Piece | Device | Why |
|---|---|---|
| Qwen3-VL-32B CLIP (~15 GB NVFP4/AWQ) | `xpu:1` (or CPU) | Encode only |
| FL2VA UNet (~20 GB FP8) + VAEs | `xpu:0` | Sample + decode |

Stock `CLIPLoader` only offers `cpu` or `default` (primary XPU, then CPU offload). Custom node `CLIPLoaderXPU` (`custom_nodes/h3_clip_xpu.py`) pins load **and** offload to `xpu:N` so the encoder stays resident on GPU 1.

Launch must **not** set `ZE_AFFINITY_MASK` or `--oneapi-device-selector level_zero:0`. Use `h3-ops/launch_comfy_dual.sh`. Caps 180 W both cards during the campaign; restore 150 W on both.

## Matched fox A/B (1344×768, 124f / 5.167 s, turbo-8, `er_sde` 8, seed 8148002)

Same prompt, same seed, cap 180 W. Timing source: Comfy `/history` wall from `/prompt` to `execution_success`. Power: `energy1_input` µJ / elapsed s on each `xe` hwmon. Sample rate from Comfy tqdm.

| Cell | CLIP | wall_s | sample | GPU0 avg W | GPU1 avg W | GPU0 pkg max | GPU1 pkg max | GPU1 VRAM during CLIP |
|---|---|---:|---|---:|---:|---:|---:|---|
| `dual_fox_cpu_clip` | CPU | **802.6** (13.4 min) | 6:44 @ **50.5 s/it**, UNet **full load** ~20 GB on GPU 0 | **123.1** | **44.9** idle | 70 °C | 54 °C | ~0.5 GB |
| `dual_fox_xpu1_clip` | `xpu:1` | **673.3** (11.2 min) | 8 steps @ **50.3 s/it**, UNet full load on GPU 0 | **137.5** | **48.4** | 70 °C | 65 °C | **~19 GB** resident |

Files: `h3_fox_cpuclip_00001_.mp4`, `h3_fox_xpu1clip_00001_.mp4`.

**Comparison (this cell only):** `matched_exact` prompt/seed/res/steps. GPU1 CLIP was **1.19× wall** vs CPU CLIP (802.6 / 673.3). Sample s/it did **not** change (~50.5 vs ~50.3). The wall win is encode + less host-RAM fighting, not faster denoise. GPU1 draw stayed ~45–48 W average over the whole job because encode is a short slice of an 11–13 min wall.

Older run08 fox (CPU CLIP, UNet **partial** 16.7 GB + 3.3 GB offload): wall **1013 s**. Dual-visible launch with UNet full-load on GPU 0 is already faster than that even with CPU CLIP.

## What did not work

- **Tensor-parallel:** not implemented in Comfy H3; no P2P on this board.
- **Two independent Comfy H3 processes:** would be two videos at once, not one video in half the time; RAM cannot hold two stacks.
- **Official MiniMax 2K regen / 4K:** API / hosted Context-IR. Local OSS canvas is 768 short edge (max 1344×768). Native duration snaps `17k+5` at 24 fps: 124f = 5.167 s, 243f = 10.125 s, **362f = 15.083 s ceiling**. Longer than 15 s is concatenated acts, not one native clip.
- **RealESRGAN x4 of 124 frames:** Comfy loaded RRDBNet then the watchdog killed pid 71751 (`swap_used 14.3 GiB`, `RssAnon ~23.7 GB`). Not a 2K MiniMax regen.
- **Local 2K that did ship:** ffmpeg lanczos **2560×1440** of the CPU-CLIP fox (`h3_fox_2k_lanczos_00001_.mp4`). Spatial upscale only. Not generative 2K.

## Custom B70 pieces (this campaign)

| Artifact | Role |
|---|---|
| `custom_nodes/h3_clip_xpu.py` | `CLIPLoaderXPU` — pin MiniMax CLIP to `xpu:1` |
| `h3-ops/launch_comfy_dual.sh` | Both XPUs visible; 180 W both cards; no affinity mask |
| `h3-ops/launch_comfy.sh` | Single-card (GPU 0); `level_zero:0` |
| `h3-ops/meter.py` | Dual `xe` hwmon energy/temp + debugfs VRAM |
| `h3-ops/queue_metered.py` | Sequential jobs, unload between, dual-card JSON |
| `h3-ops/watch_comfy_mem.sh` | Trip on MemAvailable / swap used only — not VmRSS |
| `h3-ops/stop_comfy.sh` | Kill Comfy; restore **150 W on both** `xe` hwmon nodes |
| CLIP `device=cpu` default | Fits 32 GB card; GPU1 pin is the dual-card variant |
| `--reserve-vram 2.5`, no `--highvram`, `--cache-ram 8 8` | Required. `--highvram` OOMs job 2 |

SageAttention is CUDA-only in this Comfy build. NVFP4 CLIP kernels are **emulated** on XPU (`emulated ops: nvfp4`).

## Longer video

Native one-shot max is **~15.1 s** (362 frames). A ~30 s piece is three 10.125 s clips concatenated. Shipped: `h3r9_story_30s.mp4` = bakery + paper post + heist A (729 frames, 30.4 s, 1152×640). Heist B/C never finished (watchdog during VAE on the 10 s attempt). Photoreal kitchen is still a fail (extra cooks + duck through face). Animation keepers are bakery / paper / fox / heist A, not a locked 9/10 cinema bar on every style.

## Run 11 — one heist story (2026-08-25)

Three 5.167 s acts, CLIP on `xpu:1`, UNet full-load on `xpu:0`, 180 W both cards. Prompts: `output/h3-runs/run11-story/PROMPTS.md` and `h3-ops/jobs_heist_story.json`.

| Act | wall_s | GPU0 W | GPU1 W |
|---|---:|---:|---:|
| rooftop | 665.9 | 139.0 | 48.3 |
| bakery | 669.5 | 138.3 | 48.4 |
| dawn (CLIP resident) | 448.2 | 176.4 | 45.6 |

Stitch 15.55 s at 1344×768. 4K file is **ffmpeg lanczos 3840×2160**, not generative 4K. Do not unload models between dual-card acts — `/free` after act 1 filled swap and the watchdog killed act 2 (`H3_SKIP_UNLOAD=1`).

## Run 12 — Moon Lantern (2026-08-25)

Five 5.167 s acts, CLIP `xpu:1`, UNet full-load `xpu:0`, 180 W both cards, `H3_SKIP_UNLOAD=1`. Prompts/timings: `output/h3-runs/run12-lantern/PROMPTS.md`. Jobs: `h3-ops/jobs_moon_lantern.json`. Raw: `/tmp/h3-dl/campaign12/`. n=1. **PROVISIONAL**.

| Act | sample | wall_s | GPU0 W | GPU1 W |
|---|---|---:|---:|---:|
| 1 square (cold load) | 49.96 s/it | 649.49 | 139.8 | 48.5 |
| 2 dock | 50.47 s/it | 444.16 | 177.3 | 45.7 |
| 3 bridge | 50.51 s/it | 444.16 | 177.4 | 45.8 |
| 4 pole | 50.54 s/it | 448.16 | 176.3 | 45.8 |
| 5 dawn | 50.49 s/it | 444.15 | 177.2 | 45.8 |

Stitch 25.87 s / 620f at 1344×768. 2K/4K are ffmpeg lanczos, not generative. Native generation **40.47 min** (20:04:16–20:44:44 CEST). Share table: `output/h3-runs/run12-lantern/SHARE-TABLE.md`.

## Run 13 — OmniXPU ESIMD (2026-08-25, after architect)

Core-only `omni_xpu_kernel 0.1.0b8.dev0` (`OMNI_XPU_DEVICE=bmg`, no CUTE). ComfyUI-OmniXPU patched attention+FP8 GEMM+norm+RoPE. Matched fox vs run10 GPU1-CLIP SDPA.

| Cell | sample | wall_s | GPU0 W | GPU1 W |
|---|---:|---:|---:|---:|
| run10 SDPA `dual_fox_xpu1_clip` | 50.3 s/it | 673.3 | 137.5 | 48.4 |
| run13 ESIMD+FP8 | **43.72 s/it** | **624.0** | 133.1 | 48.6 |

**1.15× s/it** this cell (`matched_exact`). Wall 1.08× (load still in the first job). Dispatch logged: ESIMD heads=56 seq=37909 bf16; FP8 GEMM `[37909,5376]×[21504,5376]`. CUTE not built. Do not source `setvars.sh` in the Comfy launcher; do not prepend oneAPI `compiler/lib` (kills XPU enum). Runtime libs: `h3-ops/omni-runtime-libs/`. n=1 **PROVISIONAL**.

## Quality vs 9/10 goal

Not complete. Cartoon/paper clips are keepers (~8.5–9/10 limbs/motion). Photoreal kitchen fails. Dual-card did not raise sample quality; same seed, same UNet. Run 11 heist is one coherent plot; fox hat drifts in act 3 (blue star-cap vs thief hat). Run 12 lantern is a longer coherent cartoon; act 1 raccoon neckerchief missing, act 4 extra market stalls. Not a locked 10/10.
