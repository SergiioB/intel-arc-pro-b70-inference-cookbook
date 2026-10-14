# B70 Launcher Guide

How the inference launcher system works — profiles, power tiers, and the toolchain.

Last updated: 2026-08-03.

---

## 1. Architecture overview

```
User
 │
 ├─ launch <profile> [watts]     ← Interactive TUI / CLI launcher
 │   └─ reads profiles.json
 │   └─ spawns llama-server directly (no systemd)
 │
 └─ switch-profile <profile>     ← Service orchestrator (production path)
     ├─ stops llama-profile.service
     ├─ applies power cap
     ├─ writes active-profile
     └─ starts llama-profile.service
         └─ run-profile (ExecStart)
             ├─ reads active-profile + profiles.json
             ├─ applies power cap (redundant safety)
             ├─ sources oneAPI env
             └─ exec llama-server
```

**Two paths, one config:** Both `launch` (manual) and `switch-profile` (service)
read the same `~/inference/profiles.json`. The service path is
production-recommended (systemd supervision, readiness polling, auto-restart).

---

## 2. profiles.json structure

```json
{
  "defaultProfile": "qwen35-q5-256k",
  "runtime": {
    "binary": "~/llama.cpp/build-sycl-0801/bin/llama-server",
    "fallbackBinaries": [...],
    "buildId": "a7a6d0d26"
  },
  "profiles": {
    "profile-name": {
      "label": "Human-readable name",
      "model": "/mnt/<models-disk>/gguf/MODEL.gguf",
      "context": 262144,
      "power": "eco",              // references powerProfiles
      "powerWatts": 150,           // explicit override
      "serverArgs": ["--parallel","1","--n-gpu-layers","99",...],
      "sampling": {"temperature":0.7, ...},
      "thinking": {"reasoning":"off"},
      "mmproj": "/path/to/mmproj",  // optional, for vision
      "binaryOverride": "/path/to/binary",  // optional, for forks
      "tokensPerSecond": 70.0,     // expected/verified speed
      "evidence": "verified ..."
    }
  },
  "powerProfiles": {
    "eco": {"watts": 150, ...},
    "efficient": {"watts": 165, ...},
    ...
  }
}
```

---

## 3. Available profiles (2026-08-03)

| Profile | Model | Context | Power | KV | t/s | Notes |
|---------|-------|---------|-------|-----|-----|-------|
| `qwen35-q5-256k` | Qwen 35B Q5_K_M | 256K | 150W | q8_0-q4_1 | 70 | **Production default** |
| `qwen35-q4-256k` | Qwen 35B Q4_K_XL | 256K | 150W | q8_0-q4_1 | 72.6 | Fastest MoE (Run 9) |
| `ornith35-q5-256k` | Ornith 35B Q5_K_M | 256K | 150W | q8_0-q4_1 | — | ⚠️ Model not on disk |
| `thinkingcap27-mtp-q4-200k-165w` | ThinkingCap 27B Q4 | 200K | 165W | q8_0-q4_1 | 29.75 | MTP-4 text/agent — ⚠️ 200K+MTP borderline, verify VRAM |
| `thinkingcap27-mtp-q4-128k-vision-180w` | ThinkingCap 27B Q4 | 128K | 180W | q8_0-q4_1 | 29.75 | MTP-4 + vision (measured @200W/128K) |
| `thinkingcap27-mtp-q5-200k-165w` | ThinkingCap 27B Q5_K_M | **128K** | 165W | q8_0-q4_1 | 26.3 | Fleet ACTIVE quant — ctx reduced from 200K (MTP overflow, Run 12) |
| `thinkingcap27-mtp-q6-128k-165w` | ThinkingCap 27B Q6_K | 128K | 165W | q8_0-q4_1 | 27.85 | **Production dense** (measured @200W) |
| `laguna-s2-iq2-16k` | Laguna S-2.1 IQ2_XXS | 16K | 150W | q8_0-q4_1 | 15 | Poolside fork, partial offload |
| `idle` | — | — | 0W | — | — | Stops inference |

**Active profile:** `cat ~/inference/active-profile`

---

## 4. Power tier reference

| Tier | Watts | When to use |
|------|-------|-------------|
| eco | 150 | **Daily default.** All MoE models. Cool (63-67°C). |
| efficient | 165 | Dense 27B MTP-4 — peak efficiency (0.148 t/s/W). |
| balanced | 180 | Sustained dense inference. Max recommended for daily dense use. |
| performance | 200 | High-power sweep testing only. |
| burst | 230 | Stock cap. Short bursts only. +9°C, +65W over 165W. |

**Power is set BEFORE the server starts** (in all paths). The sequence:
1. Stop existing server/service
2. Apply power cap (`echo <watts>000000 | sudo tee power1_cap`)
3. Wait 3s for settle
4. Start server

**No cooldown gate currently exists** — be careful switching from a hot 230W run
to another high-power load. Let the GPU cool to ≤55°C first.

---

## 5. Common commands

```bash
# Switch to a profile (production path)
~/inference/switch-profile qwen35-q5-256k

# Launch interactively (manual path)
~/inference/launch

# Launch directly
~/inference/launch qwen35-q4-256k 150

# List profiles
~/inference/launch list

# Stop inference
~/inference/launch idle
# or
~/inference/switch-profile idle

# Check status
~/inference/inference-status

# Service management
systemctl --user status llama-profile.service
systemctl --user restart llama-profile.service
journalctl --user -u llama-profile.service -f
```

---

## 6. Benchmarking

**For prefill + decode:** use `llama-bench` directly (not the HTTP API).
See `docs/benchmark-protocol.md §1.5` and `benchmarks/` for examples.

```bash
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
~/llama.cpp/build-sycl-0801/bin/llama-bench \
  -m ~/models-cache/MODEL.gguf -ngl 99 -ncmoe 0 \
  -fa 1 -ctk q8_0 -ctv q4_1 \
  -p 512,4096,8192 -n 128 -r 5 -d 0 \
  -b 8192 -ub 4096 -t 8 -dev SYCL0
```

**MTP VRAM ceiling (Run 12, 2026-08-05):** dense 27B + MTP spec buffers fits ONLY at
128K ctx (Q4@256K / Q5@200K overflow, verified). Dense MTP-4 decode @200W/128K (engine
rate, 4 prompts × 3 reps): Q4 29.75 / Q5 26.3 / Q6 27.85 t/s. Always check
`visible_avail` ≥ 1 GiB after load.

**For MTP/speculative decoding** (llama-bench doesn't support spec):
use `llama-server` + HTTP API, but verify `prompt_n ≥ 1024` for prefill.

**Reference A/B** (b9853 vs b10222, Qwen 35B Q4 MoE): see
`benchmarks/b9853-vs-b10222-qwen35b-q4-moe-2026-08-03.md`.


## 7. Known limitations (deferred fixes)

These are known issues not yet fixed (tracked for a future refactor):

- **No VRAM pre-check** before loading — a model too large for VRAM causes a hard crash.
  Workaround: check `docs/lvm-and-storage.md` for the 30 GB rule.
- **No cooldown gate** — switching profiles doesn't wait for temp to drop.
- **Dual arg-builders** — `launch` and `run-profile` each build server args independently.
  They're in sync as of 2026-08-03 but must be updated together.
- **`launch` ignores `binaryOverride`** — only `run-profile` honors it (needed for Laguna).
- **`concurrent-profile` doesn't stop existing servers** — clear port 8765 manually first.

## 8. Historical local vLLM launcher integration (2026-08-06, Run 18-22)

The host launcher supports a Docker-based vLLM MTP path behind the same port-8765 API. This integration still references a local derived vLLM 0.21 image that was never published. It works only on this host while that image exists; it is not the public reproduction path.

The current public recipe pins `vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97` and applies `patch_mtp_nightly.py`, then `patch_mtp_boundary.py`. See the cookbook `docs/IMAGE-AND-PATCH-MATRIX.md`.

### Historical profiles
`profiles.json` includes two vLLM 0.21 profiles:

| Profile | Context | Historical observation | Use case |
|---|---:|---:|---|
| `vllm-mtp-gptq-16k` | 16K | 133 t/s | Local MTP1 research |
| `vllm-mtp-gptq-128k` | 128K configured | about 92 t/s near full context | Local long-context research |

These profiles use `vllmArgs`, `speculativeConfig`, and `prefixCaching`. They do not represent the current public MTP4 two-patch stack. Do not copy their image tag or patch list into a public recipe.

### Architecture
```
launch vllm-mtp-gptq-16k
  → start_profile() detects engine=="vllm"
  → start_vllm_profile(): systemctl start vllm-mtp-profile.service --no-block
    → run-profile-vllm (unit driver):
       reads profile → applies power cap → stops llama-profile if running
       → docker run local-only intel/vllm:0.21.0-xpu-int4moe
         with the historical 4 logical fixes
         + historical profile flags → host 8765 → container 8000
  → poll http://127.0.0.1:8765/v1/models until 200 → "✓ vLLM up after Ns"
```

`--no-block` matters: `systemctl restart` without it blocks until the unit
reaches active, which takes 4-5 min (vLLM graph capture). With `--no-block`
the launcher returns immediately and polls the endpoint itself.

### vLLM startup detection — root cause postmortem (fixed 2026-08-07)

**Symptom:** launcher printed `Loading model... 0..600s` then `✗ vLLM did not
come up in 600s` even though the server came up fine (~245s) and `curl`/
`python3 -c "urllib..."` from another shell got 200 OK instantly.

**Root cause (found via strace):** the wait loop called
`urllib.request.urlopen(...)` but `urllib` was **never imported at module
level** — it was only imported *inside* `get_loaded_model()` (local import).
Every poll raised `NameError: name 'urllib' is not defined` instantly,
silently swallowed by `except Exception: pass`. The loop made **zero network
syscalls** (confirmed: `strace -e trace=connect` showed no connects after the
pre-loop status checks) while printing progress forever. `get_loaded_model()`
worked because its `import urllib.request` was local to the function.

**Why strace caught it:** the 5s loop cadence with no socket activity at all
(nothing to hit the server) — a hung/timeout loop would show connect attempts;
a working loop would show 200s. Neither was visible.

**Fix (2 parts):**
1. `import urllib.request` at module level in `launch`.
2. The loop's `except Exception: pass` now prints the first failure once
   (`(waiting — <Type>: <msg>)`) — silent swallow is what hid this for days.

**Rule:** never `except Exception: pass` around an HTTP call in a poll loop
without logging the first occurrence. Re-check with the mock:
`python3 /tmp/mock8765.py` (200 on any path) + `launch vllm-mtp-gptq-16k`
→ expect `✓ vLLM up after 0s`.

### Tracking
`inference-status` shows:
- `vllm-mtp-profile.service: active|inactive|failed`
- `vllm_endpoint: <model id>` (from port 8765 `/v1/models`)
- `endpoint_model` falls back to the vLLM endpoint when the active profile
  is a vLLM profile.

### Switching between engines
`launch <profile>` stops the other engine's unit first (VRAM contention rule).
`launch idle` stops both. Both units have `Restart=on-failure` and auto-start
on boot if enabled — **only one should be enabled at a time** (or use
`launch` to manage, which stops the other).

### Notes
- The local launcher still depends on `intel/vllm:0.21.0-xpu-int4moe`. That image was built locally and cannot be pulled from Docker Hub.
- The public nightly uses a different source generation and exactly two compatible patches: `patch_mtp_nightly.py`, then `patch_mtp_boundary.py`.
- Do not substitute `intel/vllm:0.21.0-xpu` for the local derived image.
- Migrate `run-profile-vllm` and `profiles.json` before presenting the host launcher as a current reproducible vLLM path.
