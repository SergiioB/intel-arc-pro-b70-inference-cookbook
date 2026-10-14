---
title: "MiniMax H3 on B70 via ComfyUI XPU (2026-08-14)"
type: research
date: 2026-08-14
status: final
---

# MiniMax H3 on B70 via ComfyUI XPU (2026-08-14)

H3 is an omni-modal *generation* model. llama.cpp SYCL here only knows MiniMax M2/M3. vLLM XPU has no H3 implementation. Native ComfyUI ≥0.30.0 + PyTorch XPU is the path.

Videos: `/mnt/<models-disk>/comfyui/output/h3-runs/`.

**Speed recipe that won (SFW cinema):** people LoRA 1.0 + LightX2V turbo-8 @ 0.75 + `er_sde` / 8 steps / shift 12+3 / 1344×768 / 124 frames. Sample ~46–53 s/it, ~11–15 min wall at 180 W. Turbo-4 at 864×480 is ~4.5 s/it once weights are resident.

**VRAM / RAM recipe:** CLIP `device=cpu`, no `--highvram`, `--reserve-vram 2.5`, `--use-pytorch-cross-attention`, `--cache-ram 8 8`. `--highvram` OOMs the second job. Launch only via `/mnt/<models-disk>/comfyui/h3-ops/launch_comfy.sh` (22G `MemoryMax`, OOMScoreAdjust 700, RAM watchdog). Always `h3-ops/stop_comfy.sh` when the queue ends. Idle Comfy after a 124f H3 job holds ~21 GB RSS + ~7 GB swap and systemd-oomd then kills XRDP/syncthing — that looks like a crash. The 10:00 reboot was kernel OOM of Comfy (~25.7 GB) then `oom-watchdog.service`.

**Watchdog must use RssAnon, not VmRSS.** The 15 GB CLIP is file-backed; a 10 s 1344×768 job hit VmRSS 22.6 GB on step 0 and the first watchdog killed a still-healthy process (`h3-comfy` peak 18G + 5.5G swap). Trip on anonymous RAM / MemAvailable / swap used. 10 s VAE decode then hit **RssAnon 17.6 GB** — raise the anon trip to 20 GB / unit `MemoryMax=24G` or it kills Comfy after a successful sample (bakery was already written). That stop is what dropped XRDP at 15:12; the host did not reboot.

**Run 09 bakery (measured):** 1152×640, 243f / 10.125 s, turbo-8, 180 W. Sample **90.4 s/it**, wall **1087 s**, card **142.7 W**, pkg max **70 °C**. Story holds: knead → balloon dough → strawberry cake, one fox, four legs, one tail. Keeper. Remaining: paper post + 3-act heist.

**Length grid (24 fps, 17k+5):** 124f = 5.167 s, 243f = 10.125 s, 362f = 15.083 s (native ceiling). A 30 s story is three 10 s acts concatenated. Official 2K regen is API-only; local is 768-short-edge. Dual-B70 CLIP-on-GPU1 vs CPU-CLIP fox A/B (2026-08-25): `research/2026/minimax-h3-dual-b70-20260825.md`.

**Official prompt IR (MiniMax `h3-prompt-writing` / `references/base-en.txt`):**
`integrated_multimodal_description` → `overall_soundscape` → `non_diegetic_music`. No timestamp on Shot 1; later shots get a rising cut time (`00:03.500`). Cut only when subject/space/view/time changes. Soundscape 1–4 sentences; score 1–3 sentences, no mood-words. FL2VA I2V and Ref2VA stay broken on this XPU — T2V only.

**Run 08 quality (not 9/10):**
- Kitchen: extra background cooks + hanging duck through the hero face. Fail.
- Street: one woman, two arms, face holds. ~7.5/10.
- Fox + paper: clean style, readable motion. ~8.5/10 keepers.

**Do not use on this XPU:** FL2VA `first_frame` I2V (packer shape mismatch) and Ref2VA (XPU index assert on step 0). Stay on FL2VA T2V.

## Run 08 — SFW styles (2026-08-14)

Four FL2VA T2V clips at 1344×768 / 124f / turbo-8 0.75 / `er_sde` 8 / simple / shift 12+3. Cap 180 W; live draw 109–125 W. Sample ~52 s/it. Cap restored to 150 W after the queue.

| Clip | Style | people | wall_s | card_avg_W | pkg_max_C | Notes |
|------|-------|-------:|-------:|-----------:|----------:|-------|
| `h3r8_ultra_kitchen` | photoreal kitchen fight | 1.0 | 807 | 124.6 | 68 | Face + two arms hold; five-finger block; ducks/steam readable |
| `h3r8_cartoon_fox` | 2D cel / watercolor | 0 | 1013 | 109.2 | 64 | Four legs, one tail; leap + kite look-up; no photoreal bleed |
| `h3r8_paper_boat` | stop-motion papercraft | 0 | 961 | 112.7 | 63 | Folded boat, cork lighthouse, paper sun/cloud; tactile shadows |
| `h3r8_ultra_street` | photoreal wet night | 1.0 | 902 | 116.6 | 63 | One woman, two arms; face holds in the push-in; wet asphalt |

Raw: `/tmp/h3-dl/campaign8/summary.json`. Videos: `/mnt/<models-disk>/comfyui/output/h3-runs/run08-styles/`.
