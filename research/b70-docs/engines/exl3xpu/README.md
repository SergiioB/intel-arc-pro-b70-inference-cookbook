---
title: "EXL3-XPU on B70 — engine dossier"
type: engine-dossier
status: living
date: 2026-09-27
updated: 2026-10-13
engine: exl3xpu
---

# EXL3-XPU — living dossier

vLLM-XPU fork carrying turboderp's ExLlamaV3 trellis quantization on Intel XPU,
with `qwen3_5_mtp` speculative decoding. **Primary serving route for
Qwen3.8-27B on one B70 since 2026-09-27** (route change: ADR-0010), replacing
the removed quality-defective GPTQ artifact.

Keep engine taxonomy separate: this is NOT mainline vLLM — it is a
separately-patched engine (`vllm_patches.py` rewrites GDN metadata build,
fp8-KV prefill attention, and mamba page alignment inside a vLLM API server).

> **Current figure (2026-09-29 probe 1):** 63.25 tok/s usage-token rate,
> p512/g128 n=5 @230 W; 61.12 at p8192; MTP3 accepted 2.93–3.16 tokens per
> round. The chunk-rate rows below (17.8 / 24.6) count SSE chunks, not tokens.
> The M=4 kernel probe program on this image is operator-stopped (RL-122).

## Our mixed-K image (SC5 quality tier)

| Tag | Digest | Built from | Status |
|---|---|---|---|
| `20261014` | `sha256:933eda54bb51a2e8a6669a70f5a2d7696fabd12a5f8b7c469df02fdfecc79d3a` | Same filesystem layers as 20261013; labels point to standalone `SergiioB/exl3xpu-mixedk` | Launcher 0.6.3 pin. Actual SC5 serving retry passed 6/6 canaries and 3/3 8K prompts; initial load hit the host's xe BCS CAT fault. No new performance claim. |
| `20261013` | `sha256:5e2d1fab3a8ba559b4d3af7fb1fa5eeff324c4f4274847e3079ec2d57c698ba4` | Public branch `b70` @ `244f57e` + label-only layer (`results/2026/10/exl3xpu-mixedk-public-image-check-20261013/Dockerfile.labels`) | Prior pin. 46/46 kernels, 6/6 canaries, 3/3 8K prompts, 70.32 tok/s greedy p512/g128 (RL-204). |
| `20261012c` | `sha256:7bb0c5cfed8a35da8f4db0034338d8dbe5f53731253d96ea8f1db2b5bddd4a40` | Local tree before the branch commit; Intel base-image labels | Kept: b70-launcher 0.6.0 pins it. Same checks and speed (RL-182, RL-185) |

| Field | Value |
|---|---|
| Repository | `ghcr.io/sergiioB/exl3xpu-mixedk`, public, anonymous pull verified 2026-10-13 for both tags |
| Package page | https://github.com/users/SergiioB/packages/container/package/exl3xpu-mixedk (linked to `SergiioB/exllamav3`) |
| Base | `ghcr.io/0xsero/exl3xpu:0.1.0` (MIT) |
| Source | https://github.com/SergiioB/exl3xpu-mixedk; original patches came from fork branch `b70` @ `244f57e`, with history preserved by subtree split. Local standalone checkout `/data/nvme/engines/exl3xpu-mixedk` |
| Adds | EXL3 mul1 decode for K = 2..7 (GEMV, DPAS, reconstruct, int8 prefill), per-member K in fused QKV, odd-K and tiny-N K6 at M <= 4 routed to GEMV (RL-185) |
| Used by | b70-launcher recipe `qwen38-27b-sc5` (SC_5.00bpw_H6, 0.89x plain speed, RL-186); also serves plain 4.00bpw without regression |
| Local copies | Isolated EXL3 daemon (`/data/nvme/docker/exl3-docker`) holds the published tags. |

Launcher 0.6.3 pins `20261014`. Dense Qwen3.8-27B serving checks do not establish
Flash-Next EXL3 support; that native XPU port and its KL evaluation are separate work.

## Current state (as of 2026-09-27)

- Image: `ghcr.io/0xsero/exl3xpu@sha256:21412bdd7535e9c653eeb3d099dce3bc79a83d440def2cd9556111c79c870fa8`
  (vLLM `0.26.1.dev0+g568afb3a1` inside, ~32 GB image).
- Requires the ISOLATED docker daemon — `DOCKER_HOST=unix:///run/<exl3-docker>.sock`,
  overlay2 data-root `/data/nvme/docker/exl3-docker` (containerd snapshotter off),
  bridge `<exl3-bridge>` (172.31.77.0/24). b70-launcher starts it on demand
  (`b70launcher/exl3_stack.py`, setting `exl3_data_root=/data/nvme/docker/exl3`).
  Rebuild recipe in `results/2026/09/autoround-vs-exl3-20260926/RESULTS.md`
  (round-2 incidents section).
- Artifact: `/data/nvme/models/b70-exl3-qwen38-4bpw-20260926`
  (turboderp `113cf7ab`, ~15.7 GiB, 4.00bpw EXL3 trellis; native 262144 + fp8
  KV shipped in model.yaml).
- Launch: `serve.py models/qwen3.8-27b-exl3-4.00bpw --gpu 0
  --model-path /mnt/<disk> --set vllm.gpu_memory_utilization=0.90
  --set vllm.max_model_len=65536`; port 8100; startup ≈ 6-7 min (compile+graphs).

> **CORRECTIONS 2026-09-29** (register rows 1/5/11): the "24.6 tok/s" figure for this route is a **chunk rate** (SSE chunks/elapsed), not token/s — treat all speed rows as chunk-rate until probe 1 re-derives the true token rate. Effective-bandwidth figures are unit-inconsistent (274 GB/s @61.5 ms, not 255). The artifact is **mixed precision** (safetensors: bits=4.0, head_bits=6, mtp_bits=4) — any sub-4-bit proposal must name which axes move; body-only 4→3 bpw is a 1.224× ideal traffic ratio. See `research/2026/b70-breakthrough-architect-20260928.md` §D.

## Verified capabilities / limits (CAM-2026-AUTOROUND-VS-EXL3, RL-084..088)

| Capability | State | Evidence |
|---|---|---|
| Exact-answer battery (greedy, thinking off) | **6/6 incl. code_execution 14/14** — clean where the whole W4A16 family answers 30 | `exl3.jsonl` |
| Host-executed task battery (bugfix+codegen, cache ON) | 8/10 (lfu_cache real miss; json_flatten spec-ambiguous, all arms fail identically) | `task2.jsonl` |
| Decode @150 W sustained | ~17.8 p512 / ~16.6 p8192 chunk-rate (n=3 paired) — power-limited | `exl3-cappair.jsonl` |
| Decode @230 W sustained | ~24.6 p512 / ~23.6 p8192 chunk-rate (**+33–42 %** — most cap-sensitive arm measured; first-sample burst equals sustained 230 W rate) | `exl3-cappair.jsonl` |
| Context capacity | **full native 262,144 on ONE B70 — proven by real completion**: 261,920-token prompt → correct needle answer (pool 287,040 tok @fp8 KV gmu 0.94; cold 258,480-tok prompt = 389.2 s ≈664 tok/s prefill; cached re-runs 13–28 s) — evidence `ctx_max_probe.jsonl` | `kv-probes.jsonl`, `exl3-kv-fp8-262144-g94.log` |
| KV dtype | **fp8 only** — bf16 KV rejected (`Unsupported data type of kv cache: bfloat16` in reshape_and_cache_flash during capture warmup) | `exl3-kv-bf16-*.log` |
| Prefix caching | ON by default; 10,546-token shared prefix 8.11 s → ~1.5 s (**5.4×**, best of all arms) | `cache2.jsonl` |
| MTP | native MTP3 (`speculative_config qwen3_5_mtp n=3`) — chunk-rates count accepted steps, undercount tokens (harness caveat) | engine args |
| Bandwidth floor | ~240 GB/s effective @150 W → ~355 GB/s @230 W; power-limited ESIMD decode, not byte-floor-bound | RESULTS.md byte-floor section |

## Hard rules learned (campaign traps)

- Isolated daemon mandatory — image is NOT in the main dockerd (rootfs too
  small); `--data-root` alone insufficient for image pulls with
  containerd-snapshotter (writes to `/var/lib/containerd` on the rootfs —
  force `features.containerd-snapshotter=false` or use an external containerd).
- `serve.py` takes `models/<id>` config positional + `--model-path` + `--set
  k=v` overrides; bare flags are rejected.
- gmu 0.965 OOMs on a dirty GPU0 — use 0.90 (or 0.94 when clean).
- `--set vllm.max_model_len` needed to exceed the yaml default for probes.
- EXL3 step-time numbers carry the MTP chunk-undercount caveat (client chunks
  ≠ tokens); cross-check with `usage.completion_tokens / wall`.

## Related

Campaign: `campaigns/2026/autoround-vs-exl3/` + evidence
`results/2026/09/autoround-vs-exl3-20260926/` (RESULTS.md is the full table).
Route decision: `decisions/adr-0010-exl3-primary-route-qwen38.md`.
Prior local validation: `results/2026/09/exl3-first-local-20260926/`.
