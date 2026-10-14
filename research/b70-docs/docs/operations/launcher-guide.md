---
title: "B70 Launcher Guide"
type: operation
date: 2026-09-14
status: living
---

---
doc_id: DOC-OPS-LAUNCHER-GUIDE
doc_type: operational_reference
title: B70 Launcher Guide
lifecycle: active
created: 2026-08-19
updated: 2026-08-22
last_verified: 2026-08-22
canonical_for:
  - launcher-operation
supersedes:
  - DOC-LEGACY-LAUNCHER-GUIDE-CURRENT-GUIDANCE
---

# B70 Launcher Guide

The launcher is a control plane, not an evidence authority. Model availability, recommendations, and measured performance come from the model dossiers, registries, and benchmark campaigns.

## Control paths

- `~/inference/profiles.json`: profile definitions.
- `~/inference/active-profile`: selected profile pointer.
- `~/inference/run-profile`: llama.cpp service driver.
- `~/inference/llama-profile.service`: user service for llama.cpp profiles.
- vLLM profiles may use separate user units and Docker launch scripts.

The active-profile file does not prove that its artifact exists, that a server is healthy, or that the profile is currently recommended.

Dual-card default is one independent engine per GPU (`ZE_AFFINITY_MASK`). There is no launcher profile for vLLM `--tensor-parallel-size 2`. TP2 is a gated research path ([ADR-0008](../../decisions/adr-0008-dual-b70-independent-default.md)) and occupies both cards, so it is incompatible with `b70-gpu1 off` and with a second engine.

## Safe profile change

1. Verify whether the agent is running directly on `b70-host`; do not self-SSH.
2. Discover live artifact/storage state.
3. Stop all inference user units that can respawn a container or server.
4. Verify no inference process/container and empty-card VRAM.
5. Apply the declared power cap and settle.
6. Start exactly one engine.
7. Poll a real health endpoint and log the first polling exception.
8. Verify post-load VRAM against the workload gate.
9. Retain exact engine/model identity when the launch is part of a campaign.

## User systemd environment

Agent shells may need:

```bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export DBUS_SESSION_BUS_ADDRESS=unix:path=${XDG_RUNTIME_DIR}/bus
```

before `systemctl --user` commands.

## Readiness polling rule

Never use `except Exception: pass` around an HTTP health check. A previous launcher bug failed every poll with a silent `NameError` and made zero socket connections while the server was healthy. Log the first exception and test readiness logic against a tiny mock HTTP server when modifying the launcher.

## Current profile discovery

```bash
tr -d '\n' < ~/inference/active-profile
python3 -m json.tool ~/inference/profiles.json >/dev/null
```

Then verify the profile’s artifact paths. Historical profiles may remain in the catalog even when their artifacts or local-only images are absent.

## Catalog state (2026-08-22)

- The eight llama.cpp GGUF profiles whose artifacts were deleted in the
  2026-08-13/08-22 storage cleanups were removed from `profiles.json`
  (backup: `profiles.json.bak-20260822`); `defaultProfile` now matches the
  recorded active pointer (`nemotron-vllm-dflash7-120k`).
- Remaining units: `idle`, three Qwen3.6-35 vLLM MTP profiles, two dense-27
  vLLM profiles, the Nemotron DFlash profile, and
  **`ornith15-vllm-mtp1-16k`** — the AutoRound W4A16 artifact (ADR-0007),
  served on the `f01e24f6…` champion digest with the MTP nightly/boundary
  patches, MTP1, 16K, U=0.85, gptq, fp16, model-card sampling
  (temp 1.0 / top_k 20 / top_p 0.95 from the artifact `generation_config.json`).
- `run-profile-vllm` now honors optional per-profile `image` and `patchFiles`
  fields; profiles without them keep the validated Qwen3.6 defaults
  (Pi digest `2c427ef…` + the MTP patch pair) so their evidence linkage is
  unchanged.
- The Ornith unit was live-validated 2026-08-22 through
  `vllm-mtp-profile.service`: healthy in ~350 s, WNA16 MoE + Flash Attention
  kernels, coherent completion on the `active` alias; state restored after
  (150 W, pointer back to the Nemotron profile).

## vLLM image and patch discipline

- Pin a pullable image by immutable digest.
- Record observed in-container vLLM and `vllm-xpu-kernels` versions.
- Apply only patches anchored to that source generation.
- Never present a local derived image as publicly pullable.
- Keep C1-only and mixed/concurrent overlays distinct.
- Confirm whether each patch is upstream, still required, not applicable, or rejected.
- Set `TORCHINDUCTOR_COMPILE_THREADS=1` on research launches: parallel inductor
  compile workers each open a Level Zero context and can OOM engine init
  (`UR_RESULT_ERROR_OUT_OF_RESOURCES` during `_compile_to_module`); single-threaded
  compile is the verified fix (2026-09-14, cold start ~3–4 min).
- Overlay containers that patch the vllm tree (port installs, spec-decode overlays)
  must run with `--workdir /`: otherwise the draft-model registry subprocess resolves
  the image's `/workspace/vllm` checkout instead of the patched site-packages and
  fails with a misleading `ModuleNotFoundError` (2026-09-14, RL-027).

## Launcher 0.4.8 behaviors (2026-10-07)

- API auth: every GET requires the session token — `X-Launcher-Token` header or the
  `b70_token` HttpOnly cookie set by the one-time `?token=` bootstrap redirect. Token
  lives in `<state>/token` (0600). POSTs still require the header specifically.
- Remote recipe overlays are field-allowlisted (`REMOTE_RECIPE_FIELDS`): image,
  llama_bin, docker_sock, fixed_flags, serve_config and path fields can NOT arrive
  over the update channel. Engine/image fixes ship via launcher releases instead.
- Telemetry split: the UI polls `/api/power` (watts/VRAM, ~68B) and `/api/servers`
  (server state + accounting); `/api/metrics` remains as a debug endpoint.
  `docker stats` only runs while a server is live; VRAM reads are cached.
- New endpoints: `/api/power`, `/api/servers`. Full list in the launcher tarball's `api.md` (ships in
  the tarball). `uninstall.sh` ships too (`--purge` removes state).
- WebKit fallback chain: GTK3+WebKit2 4.1 → GTK3+WebKit2 4.0 → GTK4+WebKit 6.0 →
  browser. `StartupWMClass=b70-launcher` matches `GLib.set_prgname`.

## Launcher 0.4.7 behaviors (2026-10-06)

- Recipe updates: `version.json` carries `recipes_manifest_url`; per-recipe `recipe_ver`
  stamps are compared at startup. Updates overlay into
  `~/.local/state/b70-launcher/recipes-remote.json` — installed `recipes.json` is never
  modified, and `settings-override.json` `recipe_overrides` always re-win after a merge.
  Override the channel for testing with `B70_UPDATE_URL=http://127.0.0.1:<port>/version.json`.
- Custom artifacts: `model_id=__custom__` + `custom_path` in `/api/launch` or the UI
  Browse… control. Path must sit under a configured scan root; format is sniffed
  (gguf/openvino/vllm/exl3) and must match the chosen engine.
- Native llama.cpp: `llama-server` is re-adopted after launcher restart by port probe +
  PID discovery (`running (adopted)`); Stop kills by PID. oneAPI `libsvml`/`libdnnl`
  paths are auto-added to `LD_LIBRARY_PATH`; new builds need `--flash-attn on`.
- `servers-state.json` only persists adoptable entries (running/starting/adopted);
  dry-run and stopped entries are never written.
- Shipped defaults are user-local: `~/models`, `~/Downloads`, `~/.local/share/b70-exl3`.
  Machine paths (this host's /mnt scans, pinned llama binaries) live in state overrides.

## Benchmarking

Launcher smoke numbers are not citable. Use [Benchmark Protocol](../methods/benchmark-protocol.md) and the `benchmark-model` workflow. For vLLM, cold input rate and client post-first output rate retain their endpoint/request-side names; for llama.cpp, use `llama-bench` where supported.

## Cleanup

Follow [Incident and Cleanup Procedure](incident-and-cleanup.md). Restore 150 W and verify the final process/container/VRAM state.
