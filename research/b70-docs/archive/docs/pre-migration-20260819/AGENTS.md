# B70 INFERENCE HOST — AGENTS.md
# Professional guidance for ML researchers and autonomous agents
# Intel Arc Pro B70 32GB | Ubuntu 26.04 | llama.cpp SYCL b10255+ (master 071327508)
# Last updated: 2026-08-18

## 1. HOST ACCESS

⚠️ CHECK WHERE YOU ARE RUNNING FIRST (verified 2026-08-19). Agents may execute
DIRECTLY ON the B70 desktop (hostname `b70-host`, working dir
~/B70-DOCS) — in that case DO NOT SSH anywhere; run docker/hwmon/df locally
(SSH to <lab-host> from the B70 itself fails with "No route to host").
The SSH paths below apply only when controlling the B70 from another board
(e.g. Rock 5B). `lmx` is installed on the B70 itself (/usr/local/bin/lmx,
x86_64) with live auth — no need to route through Rock 5B.

SSH from Rock 5B to B70 desktop:
  ssh desktop-b70  # alias (may need ~/.ssh/config)
  ssh sergio@<lab-host>  # direct IP (always works)

SSH alias (may not resolve, use IP as fallback):
  ssh desktop-b70

Tailscale: <tailnet-ip> (desktop-b70-ts)


Pi-Bridge Architecture:
  ┌──────────────────────┐         ┌─────────────────────────────────┐
  │ Rock 5B (aarch64)    │  HTTP   │ B70 Desktop (x86_64)            │
  │ This board           │────────▶│ <lab-host>:8765              │
  │                      │         │                                 │
  │ lmx CLI (arm64)      │         │ llama-server SYCL b10255+       │
  │ → benchmark client   │         │ -ctk q8_0 -ctv q4_1 -fa on     │
  │ → submits to         │         │ -ngl 99 -ncmoe 0 -t 8          │
  │   localmaxxing.com   │         │ --no-mmap --host 0.0.0.0       │
  └──────────────────────┘         └─────────────────────────────────┘
  Start B70 server via SSH pipe from this board. Stop with pkill.
  Full model catalog and launch templates in §5 and B70-DOCS/recommended-configs.md.
User: sergio. Passwordless sudo enabled (sudo -n works).

IMPORTANT: Long-running SSH sessions drop. Always use nohup:
  nohup bash script.sh > /tmp/output.log 2>&1 < /dev/null &

Deploy scripts by piping, not scp (scp may get blocked):
  cat script.sh | ssh sergio@<lab-host> "cat > ~/script.sh && chmod +x ~/script.sh"


## 2. HARDWARE

GPU: Intel Arc Pro B70
  - 32GB GDDR6 (32,656 MiB visible)
  - 608 GB/s bandwidth, 256-bit bus
  - Xe2 architecture, 32 Xe-Cores, 256 XMX engines
  - 367 TOPS INT8
  - 1x 8-pin power, TDP 230W (stock cap 230W)

CPU: AMD Ryzen 7 5700X3D, 8 cores / 16 threads (use -t 8)
RAM: 30GB system, 15GB swap
Disk: root plus `/mnt/<models-disk>` and `/mnt/<models-disk>`; sizes and free space changed
after the 2026-08-05 LVM rebuild. Use live `df -h`/`findmnt` before downloads.
  Primary models:   ~/models-cache/   → /mnt/<models-disk>/gguf/
  Extra quants LVM: ~/models2-cache/  → /mnt/<models-disk>/


## 3. SOFTWARE STACK

Backend: llama.cpp SYCL
  Version: b10255+ (071327508) — production build (PROMOTED 2026-08-04)
  Built with: IntelLLVM 2026.0.0, oneAPI 2026.0.0
  Binary: ~/llama.cpp/build-sycl-0804/bin/llama-server
  (Superseded b10222 build-sycl-0801. Adds #25874 oneDNN SDPA for QUANTIZED KV
   = XMX flash attention on q8_0/q4_1 cache: +18-29% prefill at 4K-8K, +140% at
   32K; plus #25880 multi-turn corruption fix. See benchmark-history Run 9.)
  Prior: build-sycl-0801 (b10222), build-sycl-latest-l0 (b9853).

oneAPI activation (required before every session):
  source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1

ffmpeg installed (required for vision image decoding).

No huggingface-cli. Download models with curl:
  curl -L -o ~/models-cache/NAME.gguf 'https://huggingface.co/REPO/resolve/main/FILE.gguf'


## 4. llama.cpp CRITICAL FLAGS AND REQUIREMENTS

Mandatory llama.cpp server flags:
  -ngl 99          All layers on GPU
  -ncmoe 0         All MoE experts on GPU (NOT CPU)
  -fa on           Flash attention (required for quantized KV cache)
  -ctk q8_0        K cache quantization (fleet standard; near-lossless)
  -ctv q4_1        V cache quantization (asymmetric: K gets more bits than V)
  -t 8             CPU threads
  --no-mmap        Keep model in VRAM, don't page to disk

KV cache rules (fact-checked 2026-08-03 against llama.cpp #23470):
  q8_0 K + q4_1 V + FA on = RECOMMENDED (KLD ~0.003, ~50% VRAM savings, near-lossless)
  q5_0 K + q4_1 V + FA on = Acceptable trade-off (KLD ~0.006-0.008, community-validated)
  q8_0 K + q8_0 V + FA on = Conservative option (KLD ~0.003, same K quality, more VRAM)
  q8_0 K + q8_0 V + FA off = FAILS (context creation error)
  q4_0 K + any V = CATASTROPHIC (KLD ~5.5, only ~11.6% top-p match — model destroyed)

Asymmetric K/V rationale: Key precision dominates attention quality (K determines
which tokens to attend to). Value precision matters less (V is averaged).
q8_0 K + q4_1 V compresses V aggressively while protecting K.
NEVER use q4_0 for K-cache — the quality cliff is catastrophic (KLD ~5.5), not
gradual. q5_0/q5_1 K are acceptable trade-offs per #23470 community consensus.

Source: llama.cpp discussion #23470 (measured KLD values), sergiiob.dev
KL-divergence analysis, Anbeeld long-context benchmarks. See kv-cache-quantization.md.

Context: always specify `-c`, but match it to the serving target and VRAM gate.
128K is the production target when the model fits safely; it is not a required
routine benchmark workload. A context-capacity claim needs staged real requests,
not a successful 128K server launch. See §9.3 and benchmark-protocol §12.
Power cap: see section 6.

MTP speculative decoding (dense models only):
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.75
  Test rather than infer benefit from dense/MoE labels. Current llama.cpp dense
  models benefit; current vLLM MoE MTP checkpoints can also benefit. Choose the
  draft depth from matched workload throughput, latency, and capacity evidence.

Vision (multimodal):
  --mmproj ~/models-cache/mmproj-MODEL-F16.gguf
  Requires ffmpeg on host for image decoding.
  Use max_tokens=512+ (models use reasoning tokens, 128 returns empty content).


## 5. MODELS ON DISK

~/models-cache/ → /mnt/<models-disk>/gguf/   (primary; verify current free space with `df -h`)
~/models2-cache/ → /mnt/<models-disk>/      (secondary LVM; verify current size/free with `df -h`)

  Text Models (live check 2026-08-18 — do not treat the 2026-08-05 snapshot as present):
  ───────────────────────────────────────────────────────────────────────────────
  Model                                  Size   Location   Type        MTP   Max Ctx
  ───────────────────────────────────────────────────────────────────────────────
  Qwen3.8-27B-Q4_K_M.gguf                 16 GB models/gguf Dense 27B   —    128K (llama.cpp; MTP stripped)
  Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 20 GB /qw38-gptq-out/ Dense 27B ✅ 128K vLLM U=0.88 (Run 40)
  Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4  —    models/     MoE 35B     ✅    vLLM profiles
  Qwen3.6-27B-MTP-Preserved-GPTQ-Int4     19 GB models2/    Dense 27B   ✅    vLLM (volume remounted 2026-08-18)
  Nemotron GPTQ-INT4-G64-sym              —    models/     MoE 30B-A3B ✅    vLLM DFlash path
  Nemotron DFlash-BF16                    1.6G  models2/    spec draft  —    companion
  Qwen3.8-27B-DSpark-vLLM                 2.6G  models2/    leftover    —    NOT a served cell; do not relaunch
  ───────────────────────────────────────────────────────────────────────────────
  ABSENT on both B70 and Rock 5B (2026-08-18): every ThinkingCap GGUF, both
  Qwen3.6-35B UD GGUFs, Laguna IQ2, Ornith Q5. `llama-profile` GGUF catalog
  therefore cannot start. `active-profile` is `nemotron-vllm-dflash7-120k`
  (vLLM engine) — do not feed that path to llama-server.

  MTP quants source (models2, re-downloaded 2026-08-05 after T-Force wipe):
  bytkim/Qwen3.6-27B-MTP-pi-tune-GGUF → Qwen3.6-27B-MTP-pi-tune-Q6_K.gguf /
  Q5_K_M.gguf (renamed to ThinkingCap-...-MTP.gguf on disk). Ornith:
  ornith-ai/Ornith-1.0-35B-GGUF.

  Vision Projectors (2 files):
  mmproj-ThinkingCap-Qwen3.6-27B-f16.gguf    889 MB  → for ThinkingCap 27B
  mmproj-deepreinforce-ai_Ornith-1.0-35B-f16  859 MB  → for Ornith 35B (on models2)

  CURRENT DENSE SERVE PATH: Qwen3.8-27B GPTQ-INT4 + BF16 MTP4 on vLLM XPU
  (`scripts/tmp/qwen38-pi-serve.sh`, mixed-split v5, U=0.88, 131K). Historical
  ACTIVE PRODUCTION GGUF (ThinkingCap Q6_K-MTP @ 128K) is not on disk.
  Dense + MTP VRAM ceiling = 128K ctx (verified 2026-08-05: Q4@256K/Q5@200K + MTP overflow).

  Launcher profiles (9 total): see ~/inference/profiles.json
  Radxa catalog: ~/.cache/workstation-inference/workstation-catalog.json (7 models)
  DEFAULT_MODEL (pi-telegram-bridge): lc-thinkingcap27-mtp-q5-200k-165w

  MTP = Multi-Token Prediction (speculative decoding). ThinkingCap-Qwen3.6-27B
  and the preserved-MTP vLLM checkpoints have draft heads. Benefit depends on
  engine, artifact, workload, and draft depth; screen it at matched anchors.

Removed/not currently installed (per old §5): Qwen3.6-27B-MTP-Q5_K_M, Gemma 4 26B,
Ornith 9B, Ornith 35B Q4, ornith-1.0-35b-Q5_K_M.gguf (deleted 2026-08-14 to free
/mnt/<models-disk>; launcher profile ornith35-q5-256k still points at the missing file). Also LOST in the 2026-08-05 T-Force wipe (not restored):
Qwen3.6-35B-A3B-MXFP4-native (vLLM ckpt — rebuildable via scripts/tmp/b70-mxfp4-merge.py
from community MXFP4), Qwen3.6-35B-A3B-bf16 (67 GB — do not re-download unless needed).

⚠️ STORAGE LAYOUT: historical inline snapshots conflicted after the 2026-08-05
T-Force/LVM rebuild. Do not use remembered sizes for download or model-fit
decisions. Resolve live state on the B70 with:
  df -h / /mnt/<models-disk> /mnt/<models-disk>
  findmnt -o SOURCE,FSTYPE,SIZE,AVAIL,TARGET / /mnt/<models-disk> /mnt/<models-disk>

### 5.1 HOW TO CHOOSE A MODEL ARTIFACT AND QUANT

Treat **architecture support**, **checkpoint format**, **quality**, and
**performance** as separate questions. A repository name such as `AWQ`, a
successful conversion, or an accepted self-report does not answer all four.

1. **Start with the engine and architecture gate.** Verify the exact vLLM XPU
   image registry before downloading or converting. The pinned XPU image has a
   recorded NemotronH/MTP registry observation, but Nemotron GPTQ has not yet
   passed a B70 vLLM load/coherence smoke test. Muse-Glimmer remains blocked on
   vLLM PR #51655: conversion cannot add an absent model implementation.
2. **Identify the source format from metadata and tensors, not its repo title.**
   Read `config.json` and inspect safetensor names/shapes before selecting a
   tool. Preserve the source checkpoint unchanged during any experiment.
3. **Prefer a native artifact when it meets the engine gate.** Native GGUF is
   cleaner than a local GGUF requantization; a published GPTQ is preferable to
   re-encoding another 4-bit format. A conversion is an experimental new
   artifact, not proof of parity.
4. **Budget storage before acting.** Run live `df -h / /mnt/<models-disk> /mnt/<models-disk>`
   and `findmnt` immediately before downloading, converting, or building. The
   destination must hold every concurrently retained source, output, and
   temporary/staging file plus an explicit reserve. Do not choose a volume from
   a remembered free-space value; see §15.6.
5. **Validate in increasing cost order.** Static metadata/index coverage →
   tensor/dequantization error → deterministic coherence/parity → isolated
   engine load → target workload benchmark. Keep correctness, capacity, and
   speed claims separate.

**GGUF model weights: measured Nemotron-specific result.** On the recorded
llama.cpp SYCL path, a *locally requantized* Nemotron 3.5 Lightning Q4_0 reached
68.0 t/s no-spec at tg128/150 W and 87.4 t/s median native-MTP4 at 230 W
(`n=5`), versus Q4_K_M's 38.2 t/s / about 50 t/s at those different recorded
campaigns. The simpler Q4_0 dequantization was materially faster for this model;
re-benchmark each model/build/quant pair rather than applying it as a universal
Xe2 rule. The Q4_0 artifact was made from Q4_K_M with `--allow-requantize`, so
it has an extra quality-loss step versus a direct-from-BF16 Q4_0.

- **Throughput-first, measured Nemotron case:** consider model-weight Q4_0 after
  task-quality validation; use the logged result only at its declared cells.
- **Quality-first:** prefer native Q4_K_M/Q5_K_M or a validated native artifact.
- **Never confuse model-weight Q4_0 with KV-cache `q4_0 K`.** The latter remains
  catastrophic (KLD ~5.5) and is prohibited by §4/§15.4.

**How to identify a Hugging Face quant source:**
- `quantization_config.quant_method: gptq`, with `qweight`, `qzeros`, `scales`,
  and usually `g_idx`, identifies GPTQ.
- `quant_method: awq` and the checkpoint's actual packing identify AWQ; do not
  infer it from a repository title.
- `quant_method: compressed-tensors` with `format: pack-quantized` and integer
  weight metadata identifies compressed-tensors W4A16. Inspect group size,
  symmetry, activation ordering, tensor names, and shapes: those properties
  control whether any repack is applicable.
- `modelopt` / NVFP4 is NVIDIA-oriented and not a B70 XPU conversion target.
  bitsandbytes/NF4 is also not a drop-in vLLM XPU conversion target here.

**vLLM XPU INT4 and conversion rules:**
- GPTQ-INT4 through `XPUwNa16LinearKernel` is a **documented proven B70 XPU
  INT4 path**, for an exact checkpoint/architecture that has actually loaded.
  Compressed-tensors W4A16 is now also proven for the exact Muse-Glimmer
  group-32 asymmetric artifact through the same XPU kernel family, including
  text and vision smokes; this does not generalize to AWQ or other
  compressed-tensors packing/group/act-order combinations.
- A compatible **asymmetric** compressed-tensors `pack-quantized` W4A16 layout
  may be repacked into GPTQ layouts. This is only a candidate lossless layout
  transformation after metadata, tensor, index, and model validation; it is not
  a generic compressed-tensors-to-GPTQ guarantee. The scratch Muse converter
  currently targets the complete asymmetric group-32 source, not RedHat's
  symmetric group-128/static-actorder artifact.
- **MLX affine INT4 → GPTQ is a dequantize/requantize operation, not a
  transpose.** MLX stores `q * scale + bias`. An asymmetric GPTQ candidate can
  recover an approximate integer zero point through `round(-bias / scale)`, but
  Nemotron's pinned XPU fused-MoE path cannot consume routed-expert zero points.
  The working experiment therefore reconstructs each affine group and genuinely
  requantizes it to symmetric signed INT4 with endpoint-aware scale
  `max(max/7, -min/8)`. Preserve this as a separately named artifact and require
  all-matrix numerical validation, isolated load, deterministic coherence, and
  task-quality/reference comparison; successful conversion is not parity.
- **Asymmetric GPTQ needs end-to-end backend support, not just qzeros in the
  checkpoint.** On the pinned XPU image, ordinary `XPUwNa16LinearKernel` can
  carry zero points after a narrow config/loader patch, but the XPU fused-MoE
  WNA16 backend is symmetric and discards routed-expert qzeros. Therefore an
  all-asymmetric Nemotron checkpoint must fail closed: either requantize every
  dense and expert matrix to a separately validated symmetric INT4 artifact, or
  add zero-point support to both ordinary and routed/grouped XPU kernels. Never
  relabel asymmetric weights as `sym=true`.

**Muse-Glimmer has three independent gates:** (1) a Muse-capable XPU image,
(2) successful exact-quant text-only dispatch, and (3) vision/XPU correctness.
All three have experimental smoke evidence using a thin PR #51655 Python overlay
on the pinned XPU image and the native asymmetric group-32 compressed-tensors
artifact. The overlay is local-only and requires `PYTHONPATH=/workspace/vllm`;
chat responses currently expose reasoning/channel text in `content`, while raw
text completions are clean. This is not yet a public pullable production recipe
or performance recommendation. A future GPTQ artifact can at most change gate 2;
it cannot substitute for architecture or vision validation. Current status:
`research/b70-vllm-muse-int4/PLAN.md`.
Local artifacts DELETED 2026-08-14 to reclaim /mnt/<models-disk> (`muse-glimmer`
GGUFs + `muse-glimmer-awq-int4` checkpoint, ~41G): resuming Muse work requires
re-downloading them; the thin overlay dirs on /mnt/<models-disk> and the muse docker
tags were kept.

**NemotronH GPTQ XPU caveat:** the validated local symmetric group-64 artifact
loads with ordinary and routed XPU WNA16 kernels, but compiled mode on the pinned
image fails in a generated non-gated-MoE expert-index assertion. `--enforce-eager`
is the current safe bypass and retains native XPU kernels. Do not clamp expert
IDs or call the compiled path fixed. The eager artifact is deterministic/coherent
smoke tested but still needs task-quality/reference parity before adoption.

**Source repositories:** for GGUF, choose verified native quant repos such as
bartowski, ggml-org, or lmstudio-community, then verify revision/file hashes and
live storage. For every other format, inspect the actual metadata first. See
`research/nemotron-lightning-optimization-20260811/08-q4_0-breakthrough.md`,
`research/quantization-format-strategy.md`, and `docs/recommended-configs.md`.


## 6. POWER AND THERMAL CONTROL

Power cap (microwatts; resolve the B70 hwmon path first):
  Read:   cat "$B70_HWMON/power1_cap"
  Set:    echo 180000000 | sudo tee "$B70_HWMON/power1_cap"
  Values: 150000000=150W  165000000=165W  180000000=180W  230000000=230W(stock)

Temperature (millidegrees C):
  cat "$B70_HWMON/$B70_TEMP_INPUT"  (divide by 1000; verify the sensor label)

Idle temp: ~49C. Under load at 150W: 63-70C. At 230W: 76-78C.

Power profiles:
  150W = daily/cool, best for MoE (they barely benefit from more power)
  165W = efficiency sweet spot for dense 27B MTP-4 (0.155 t/s/W)
  180W = recommended max for sustained dense inference
  230W = stock, use for short bursts only (+4.5 t/s over 165W, +9C, +65W)

No undervolt available. Live card draw IS measurable from the GPU hwmon `energy1_input` cumulative-energy counter (microjoules):
  actual_average_W = (energy_end_uJ - energy_start_uJ) / elapsed_seconds / 1,000,000
Use a monotonic timer and align the two reads to the exact measured workload. Record the raw start/end energy, elapsed time, resolved hwmon path, and configured `power1_cap` separately. For peak/phase reporting, sample 0.5–1.0s energy deltas; label the maximum as `peak interval-average W`, not instantaneous peak. Never substitute the cap/TDP or a remembered typical value for actual draw.

Do not hard-code `hwmon4` in new public benchmark evidence. Discover `/sys/class/hwmon/hwmon*/name`, verify the B70 PCI mapping, and record all `temp*_label`/`temp*_input` pairs. Sample real temperatures throughout the same measured interval and save the timestamped monitor log. Report start/end and per-sensor maximum (GPU package, VRAM, memory controller when labels verify them), with at least 0.1°C retained in raw evidence. If a sensor cannot be verified, say `not measured`—never estimate. Full requirements: `docs/benchmark-protocol.md` §6.

VRAM check (NOT system RAM):
  sudo cat /sys/kernel/debug/dri/0000:0b:00.0/tile0/vram_mm
  Look for: visible_avail (free MiB), usage (bytes used), visible_size (total)


## 7. BENCHMARK RESULTS — POINTS TO LIVING DOCS

Benchmark numbers go stale every build/model change. Instead of maintaining
inline tables that drift, this section ROUTES you to the authoritative source.

**For current numbers, always check these (in priority order):**

1. **`docs/benchmark-history.md`** — chronological run index. Every campaign links its raw evidence and records build/model/config, actual workload coordinates, cache/concurrency state, power cap versus measured draw, timing source, statistics, and gate status. Routine publishable model cards use benchmark-protocol Lane 1; full pp/tg curves are Lane 3. Append new runs; never rewrite old evidence. `results/` remains the authoritative measurement trail.

2. **`docs/recommended-configs.md`** — per-model optimal configs table with
   current quant/context/power/KV. Updated when boundaries change.

3. **`benchmarks/`** — human-readable A/B reports with commands + analysis
   (e.g., `b9853-vs-b10222-qwen35b-q4-moe-2026-08-03.md`).

4. **`results/`** — raw llama-bench / script output (txt). The evidence trail.

**Current headline numbers** (master 0804, q8_0-q4_1 KV, 150W — see benchmark-history Run 9 + Run 11):

  MoE 35B Q4:  72.6 t/s decode | 2128 t/s prefill pp4096 | 1871 t/s pp32768 | 1211 t/s pp131072
  Dense 27B Q4: 21.3 t/s base  | 936 t/s prefill pp4096 (~24-29 t/s w/MTP-4) | 546 t/s pp131072
  Long-ctx delta (Run 11): #25874 wins grow with context — +52% @64K, +80-90% @128K vs b10222.
  Prior (b10222, Run 5-6): MoE 70.2 t/s | 1659 t/s pp4096; Dense 27B Q4 18.4 t/s | 700 t/s

**Key findings (verified, see §14 Claims Register + §15 Lessons Learned):**
  master 0804 (#25874 quantized-KV XMX FA) = +26-29% prefill @4K-8K, +140% @32K vs b10222.
  MTP-4 gain on 27B dense: +35-50% (93-94% acceptance). Power scaling helps dense, not MoE.
  Zero context-length decode penalty 64K-256K (while KV fits VRAM).
  b10222 was +6-13% over b9853 (oneDNN XMX FA #25222).
  q8_0 K + q4_1 V = fleet standard (KL ~0.003). q4_0 K = catastrophic (KL ~5.5).
## 8. INFERENCE TESTING LIMITS AND DISCIPLINE

These are hard rules. Violating them produces invalid data or crashes.

8.1 EMPTY GPU + ONE ENGINE INSTANCE ONLY
Never run two inference engines, two benchmark campaigns, or an unrelated GPU workload simultaneously. GPU contention causes OOM, corrupted metrics, and xe driver VM worker errors. A controlled concurrency campaign may send Cn requests to one declared server.

Before starting any model, all four gates MUST pass:
  pgrep -af 'llama-server|llama-bench|vllm|vllm-xpu'  # must return nothing
  docker ps --format '{{.ID}} {{.Image}} {{.Names}}'   # no inference container
  cat "$B70_HWMON/power1_cap"                           # restore 150000000 first
  sudo cat /sys/kernel/debug/dri/0000:0b:00.0/tile0/vram_mm | grep -E 'visible_avail|usage|visible_size'
An empty B70 should show approximately 31,000+ MiB `visible_avail` (32,327 MiB observed 2026-08-08). If a process/container exists, VRAM is below 31,000 MiB, or any check fails, do not start. Diagnose and clear the load first. Then set the campaign cap, wait 3 seconds, and launch exactly one engine.

Post-load VRAM reserve depends on workload:
  <500 MiB = abort
  500–1,023 MiB = isolated C1 capacity observation only
  1,024–2,047 MiB = staged C1 research allowed
  ≥2,048 MiB = preferred C1 performance work
  ≥3,072 MiB = target for mixed/concurrent serving
The 500 MiB floor is operator-approved for research, not production-safe headroom. A fit does not prove scheduler, speculative-buffer, or mixed-token correctness.

8.2 WARMUP IS MANDATORY
First request after server start includes SYCL JIT compilation (30-60s overhead).
Always run 1 warmup round and discard it. Steady-state = rounds 2+.

8.3 COOLDOWN BETWEEN COMPARABLE TESTS
Wait for the verified GPU package sensor to drop below the declared threshold,
normally 52-55°C. Discover the B70 hwmon node and sensor label; do not hard-code
`hwmon4` in new scripts or evidence.

8.4 USE THE CORRECT TIMING SOURCE
For llama.cpp C1 engine decode, retain response `timings` token count and predicted duration/rate; report client-observed streaming separately. For vLLM serving latency, use `vllm bench serve` client-observed TTFT/TPOT/ITL/E2EL; those are end-to-end request-side metrics by definition. vLLM `/metrics` latency histograms are cumulative and bucket-granular, so use them only for corroboration. Prefix-cache and spec-decode counters are authoritative cumulative counters when diffed across the measured window. For Cn aggregate throughput, use successful generated tokens divided by the campaign interval—never sum per-request rates. For llama-bench use raw pp/tg repetition data. See benchmark-protocol §1.4 and §9.

8.5 RETAIN SAMPLES; LABEL THE STATISTIC
For llama-bench use controlled synthetic lengths. For HTTP campaigns rotate
diverse prompt families and unique entropy-first prefixes while preserving target
length. Use n=3 only for provisional tuning screens; extend publishable points to
n=5. Report median, maximum observed, sample count, and range. A short best
repetition is peak, not sustained throughput.

8.6 CONTROL POWER AND ORDER EFFECTS
Keep one cap for ordinary comparisons. A power A/B may alternate caps on one
warm server after settle and cooldown only when order alternates across rounds,
prompts remain matched and cold, and measured energy is retained. Do not infer a
power gain from separate unpaired runs.

8.7 NO COMPETING WORKLOADS
A second engine, benchmark script, model load, or unrelated GPU workload contaminates the campaign. Controlled concurrent requests from one harness to one declared server are allowed. Always verify the process list before starting.

8.8 RESTORE STATE ON EXIT
Always restore power to 150W and kill servers when done:
  trap 'echo 150000000 | sudo tee "$B70_HWMON/power1_cap"; pkill -f "llama-server|vllm"' EXIT

8.9 FOREGROUND / LIVE-PROGRESS CAMPAIGNS
Run each campaign under a visible controlling process. Print a milestone before and after every warmup, prompt shape, scheduler budget, and measured repetition. A remote run may use a supervised process with live logs, but never an unobservable detached script. If progress stops past the declared per-cell timeout, abort, restore 150W, stop the exact campaign server/container, and re-run the empty-GPU gates in §8.1.


## 9. PROFESSIONAL TIPS FOR ML RESEARCHERS

9.1 DENSE vs MoE PERFORMANCE MODEL (updated 2026-08-09)
For the measured llama.cpp GGUF paths, dense and MoE decode are bandwidth-bound.
The difference is how many bytes per token must be read:
- Dense 27B reads all ~19 GB/token (~19-30 t/s depending on power)
- MoE 35B-A3B reads only ~3 GB/token (active experts, ~60-70 t/s)
MoE is faster simply because it reads fewer bytes — NOT because it's "compute-bound."
Power and MTP behavior are engine, checkpoint, phase, and workload specific.
Dense llama.cpp decode and dense vLLM prefill have measured power scaling. The
paired vLLM MoE+MTP prefill A/B in Run 24 was flat from 150W to 230W, while its
decode draw was lower. Do not turn that result into a universal MoE rule.
The current vLLM Qwen3.6-35B-A3B preserved-MTP checkpoint benefits strongly from
MTP4 at short C1 decode; the older GGUF checkpoint had no MTP layers. Screen the
actual artifact instead of routing from the architecture label.

KEY INSIGHT — MoE uses FAR LESS KV CACHE than dense (measured 2026-08-03):
KV cache size scales with the ATTENTION layers, NOT the total parameter count.
MoE 35B-A3B has tiny attention (3B active params → small Q/K/V projections),
while dense 27B has full-size attention on all 27B params. Measured with q8_0-q4_1:
  Dense 27B: ~6960 MiB KV per 128K context
  MoE 35B-A3B: ~1847 MiB KV per 128K context (3.8x LESS than dense!)
This means: MoE can reach much higher context at the same VRAM, even though the
MODEL WEIGHTS are larger. A 25GB MoE fits 256K easily; a 19GB dense barely fits 200K.
Common mistake: assuming KV scales with total params — it scales with ATTENTION size.
Fixed compute overhead (SYCL JIT buffers, etc): ~1500-2000 MiB regardless of model.

9.2 QUANTIZATION IMPACT
Q5_K_M is the recommended quant for 32GB VRAM. Near-lossless vs FP16.
Q4_K_M/Q4_K_XL: acceptable for MoE (masking hides degradation). Slight quality
loss on dense.
Never go below Q4 unless pushing 70B+ models.
KV cache: q8_0 K + q4_1 V = fleet standard (KLD ~0.003, ~50% VRAM savings vs FP16).
See docs/kv-cache-quantization.md for the full KL-divergence analysis.

9.3 VRAM BUDGETING (q8_0 K + q4_1 V — MEASURED, boundary sweep 2026-08-03)
VRAM total: 32656 MiB. These are MEASURED values from boundary-sweep Run 6
(not estimates). "VRAM free" = MiB left after model + KV + compute buffers load.

  Config                         Wt      @128K       @200K       @256K       @512K
  ──────────────────────────────────────────────────────────────────────────────────
  27B Q4_K_M +MTP               16GB    8.1GB ✅    4.2GB ✅    0.9GB ⚠️    OVER
  27B Q5_K_M +MTP               19GB    3.3GB ✅    2.7GB ✅    OVER        OVER
  27B Q6_K +MTP                 21GB    0.7GB ⚠️    OVER        OVER        OVER
  35B MoE Q4_K_XL               21GB    8.0GB ✅    ~7GB ✅     6.1GB ✅    1.6GB ✅
  35B MoE Q5_K_M                25GB    4.0GB ✅    ~3GB ✅     2.2GB ✅    OVER ❌

KEY FINDINGS (measured, not estimated):
- Q5_K_M @ 200K WORKS (2.7GB free) — better than linear model predicted (was 0.4GB)
- Q6_K @ 128K is the absolute ceiling (0.7GB free — zero headroom for vision)
- MoE Q4 @ 512K is the context champion (1.6GB free, 1659 t/s prefill, 70 t/s decode)
- MoE Q5 @ 512K = OOM crash (confirmed). Q5 MoE maxes at 256K.
- Dense decode is ~16-18 t/s across ALL quants (bandwidth-bound; quality differs, not speed)
- See benchmarks/boundary-sweep-2026-08-03.md for full llama-bench numbers.

9.4 SPEED MEASUREMENT METHODOLOGY (updated 2026-08-08)
Pick the tool by engine and metric:
- llama.cpp synthetic pp/tg: use `llama-bench`. Lane 1 uses pp2048/pp8192 and
  representative tg128 at declared prompt states; add the full pp/tg curve only
  for a Lane 3 claim.
- llama.cpp MTP/spec: use `llama-server` HTTP. Retain response `timings`, verify `prompt_n`, and apply the cache guards below.
- vLLM serving latency: use `vllm bench serve` for client-observed TTFT/TPOT/ITL/E2EL percentiles. Keep C1 and Cn separate.
- vLLM prefill tok/s: use controlled HTTP with exact rendered tokens, entropy-first unique prefixes, and zero cache-hit delta. `/metrics` latency histograms corroborate only; cache/MTP counters are authoritative.

Canonical llama.cpp broad-characterization command:
  llama-bench -m MODEL.gguf -ngl 99 -ncmoe 0 -fa 1 -ctk q8_0 -ctv q4_1 \
    -p 512,2048,4096,8192 -n 128 -r 5 -d 0 -b 8192 -ub 4096 -t 8 -dev SYCL0
For the default Lane 1 model card, use pp2048/pp8192 plus tg128 at declared short
and long prompt states. Add the full command below only for a curve claim.
  pp4096 = prefill throughput at 4096-token synthetic prompt
  tg128  = decode throughput generating 128 tokens; representative only when statistic/n are stated
For llama.cpp HTTP MTP, retain `timings.predicted_per_second` and verify
`timings.prompt_n >= 1024`; `prompt_n = 4` means prompt truncation.
See docs/benchmark-protocol.md §1.5 and §9.

⚠️ ⚠️ PREFILL + PREFIX CACHING = SILENT 5× LIE (verified 2026-08-07, Run 24).
Any HTTP-API prefill measurement on an engine with prefix caching (vLLM
`--enable-prefix-caching`) can be contaminated: if two calls share a prompt
PREFIX, the second reuses cached KV blocks and reports prefill that is almost
free. We measured p8k jumping 8,640 → 42,733 t/s once cached. A constant-filler
harness + "steady-state = reps 2-3" averaging HIDES this (rep 1, the only honest
one, gets discarded). This produced a false "+16-22% prefill from 230W" claim.
MANDATORY when measuring prefill via HTTP API:
  1. Prepend a UNIQUE random run-id (≥64 chars) per call so no prefix repeats, while preserving the declared total prompt length.
  2. Snapshot V1 token-level cache counters before and after: `vllm:prefix_cache_hits` and `vllm:prefix_cache_queries`. Valid uncached prefill requires a zero hits delta; the absolute process-wide value may already be nonzero.
  3. Sanity check equal-length repetitions. If rep2 > 1.5× rep1, invalidate the run; investigate any unexplained material acceleration.
  4. Use `scripts/tmp/b70-prefill-honest.py` (does all of the above).
  Do NOT use the old `b70-mtp-prefill-sweep.py` / `b70-mtp-grid-bench.py`
  (constant filler, contaminated). Full writeup: benchmark-protocol.md §1.5 Error 3.

9.5 llama-bench HANGS ON THIS BUILD
llama-bench prints results but doesn't exit cleanly. Monitor for tg128 line,
then kill the process. Results are valid if they printed.

9.6 SYCL JIT CACHING
SYCL_CACHE_PERSISTENT=0 is required for stability but means JIT kernels don't
persist across processes. Each model needs its own warmup run (discarded).
Within a sweep of the same model, subsequent runs are clean (driver-level cache).

9.7 ENVIRONMENT VARIABLES (set before launching server)
  # Do NOT disable SYCL optimization. The flag is GGML_SYCL_ENABLE_OPT (default 1).
  # Older builds used GGML_SYCL_DISABLE_OPT=1 (inverse logic). Setting optimization
  # off causes a large decode regression (internally ~44.7% on 9B, though this
  # figure is not corroborated by upstream llama.cpp — treat as approximate).
  # NEVER set GGML_SYCL_ENABLE_OPT=0 unless specifically debugging.
  export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0
  export SYCL_CACHE_PERSISTENT=0
  export SYCL_DEVICE_FILTER=level_zero
  export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
  export ONEAPI_DEVICE_SELECTOR=level_zero:0
  export ZE_AFFINITY_MASK=0

9.8 SERVER ARGUMENTS THAT DON'T EXIST IN LLAMA-BENCH
-c (--ctx-size), --parallel, --host, --port are llama-server only.
--n-prompt, --n-gen, --pg, --repetitions are llama-bench only.
Always run "llama-bench --help" on new builds to verify flags.

9.9 KNOWN WORKING SPEC TYPES (build 9853)
none, draft-simple, draft-eagle3, draft-mtp, draft-dflash,
ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache

9.10 VISION REQUEST FORMAT
OpenAI-compatible multipart:
  {"messages":[{"role":"user","content":[
    {"type":"text","text":"Describe this"},
    {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,<B64>"}}
  ]}]}
Response: content field has the answer, reasoning_content has the thinking.
Both Qwen 27B and Gemma 4 26B correctly describe images. Vision overhead is
4-6% on decode speed (image embedding is prompt-phase, not decode-phase).

9.11 REAL-WORLD PI / TELEGRAM SERVING CAMPAIGNS
Synthetic cold-prefix throughput is only the document/RAG-ingestion case. A
Lane 1 model card uses one cold turn and one representative warm follow-up.
Before a broad production scheduler decision, measure:
- Cold: new conversation; large-document ingestion.
- Warm shared state: system/tool prefix; multi-turn session; follow-up over a resident document.
- Warm append: new RAG/tool payload appended to an existing session.
- Mixed: long prefill while short chat requests arrive.

Record actual endpoint prompt/completion tokens, TTFT, TTFC (first visible content), TPOT/ITL, end-to-end p50/p95, cache query/hit deltas, MTP acceptance, failures/queueing, C1 vs aggregate Cn, and synchronized power/temperature. Calibrate rendered tokens with `len(encoded["input_ids"])`. Run a same-length shape warmup before each distinct prompt length. Long-context progression and VRAM abort gates are normative in benchmark-protocol §12.


## 10. BENCHMARK SCRIPTS — usage matrix and creation policy

### 10.1 Which script to use (decision matrix)

| You want to... | USE THIS SCRIPT | Why |
|----------------|-----------------|-----|
| **Benchmark any model** (default) | `$benchmark-model` | Selects the smallest valid lane, reuses evidence, screens tuning candidates, and routes to the engine-correct tool. |
| **Measure llama.cpp engine prefill + decode** | `b70-ab-llama-bench.sh` | Controlled synthetic pp/tg, immune to HTTP cache and quoting bugs. |
| **Compare two builds** (A/B test SYCL versions) | `b70-ab-llama-bench.sh` | Runs both binaries on same model, side-by-side. |
| **Find VRAM/context limits** (does Q5 fit at 256K?) | `b70-boundary-sweep.sh` | Loads model at target ctx, checks VRAM free, aborts on OOM risk, then benches. |
| **Broad legacy llama.cpp MTP sweep** | `b70-all-models-128k-256k.sh` | Lane 3 HTTP campaign. Do not use for a routine model card. |
| **Sweep all models quickly** | `b70-master-v2.sh` | 11-test all-model sweep. |
| **Validate KV cache configs** | `b70-kvquant.sh` | Tests q8_0-q4_1 vs q8_0-q8_0 vs q4_0. |
| **Test high context (200K-512K)** | `b70-highctx.sh` | Progressive context ceiling finder. |
| **Benchmark vision (image input)** | `b70-vision-test.sh` | Multimodal with ffmpeg image decode. |
| **Quick smoke test (NOT citable)** | `~/inference/benchmark.py` | Wall-clock t/s, no warmup, no long prefill. Development only. |

**Rules:**
- **Never** use `benchmark.py` for numbers that go in `benchmark-history.md`, portfolio posts, or localmaxxing submissions.
- Use `llama-bench` for prefill whenever it supports the tested configuration. For MTP/speculative paths, controlled HTTP prefill is allowed only with exact prompt-token verification, unique entropy-first prefixes, zero cache-hit delta, and retained timing fields.
- **Always** stop the production server first: `systemctl --user stop llama-profile.service`.
- **Always** save raw output to `results/` and human-readable reports to `benchmarks/`.

### 10.2 Script creation policy (MANDATORY for all new scripts)

Every benchmark script MUST follow these rules:

1. **Header comment block (required).** Every `.sh` file must start with:
   ```bash
   #!/usr/bin/env bash
   # B70 <Script Name> — <one-line purpose>
   #
   # WHEN TO USE: <what scenario triggers using this script>
   # WHAT IT DOES: <step-by-step summary>
   # OUTPUT: <where results go — results/ or benchmarks/>
   # SAFETY: <what it stops/restores — production server, power cap, etc.>
   #
   # Example:
   #   bash scripts/b70-boundary-sweep.sh
   ```

2. **Decision matrix mapping (required).** After creating a script, add a row to
   the §10.1 table above ("You want to..." → script name). A script not in the
   table does not exist for practical purposes.

3. **Scratch-first workflow.** New scripts are created in `scripts/tmp/` first
   (scratch space). Only promote to `scripts/` and add to §10.1 when:
   - The script has been run successfully at least once
   - The header comment block is complete
   - It produces results in the correct output location
   - It follows all validity, campaign-matrix, manifest, and publication requirements in `benchmark-protocol.md`
   
   ```bash
   # Create in scratch:
   scripts/tmp/b70-experiment-test.sh    ← scratch, not mapped in §10.1
   
   # Promote when validated:
   mv scripts/tmp/b70-experiment-test.sh scripts/b70-experiment.sh
   # THEN add to §10.1 decision matrix
   ```

4. **Trap + restore (required).** Every script must end with:
   ```bash
   trap 'echo 150000000 | sudo tee .../power1_cap > /dev/null 2>&1; \
         pkill -f "llama-server|llama-bench|vllm" 2>/dev/null; \
         systemctl --user start llama-profile.service' EXIT
   ```

5. **Isolation + cooldown (required).** One engine and one campaign harness only; controlled Cn requests to that server are allowed. Cooldown ≤55°C between comparable tests.

6. **Evidence output (required).** Save every raw repetition plus a manifest containing exact prompt/output tokens, statistic, timing source, cache state/counter delta, C1/Cn scope, configured cap, measured draw when available, temperatures, failures/exclusions, and gate status.

### 10.3 Canonical script inventory

| Script | Purpose | Output | Status |
|--------|---------|--------|--------|
| `b70-ab-llama-bench.sh` | A/B prefill+decode via llama-bench | `results/`, `benchmarks/` | ✅ Active |
| `b70-boundary-sweep.sh` | VRAM/context limit finder | `results/` | ✅ Active |
| `b70-all-models-128k-256k.sh` | All-model decode sweep (MTP via HTTP) | `results/` | ✅ Active |
| `b70-prefill-check.sh` | Prefill via HTTP API (verify prompt_n!) | `results/` | ⚠️ Use with caution |
| `b70-verified-bench.sh` | Canonical decode (4 prompts, warmup) | `results/` | ✅ Active |
| `b70-highctx.sh` | High-context validation (200K-512K) | `results/` | ✅ Active |
| `b70-kvquant.sh` | KV cache quantization validation | `results/` | ✅ Active |
| `b70-master-v2.sh` | All-model benchmark (11 tests) | `results/` | ✅ Active |
| `b70-vision-test.sh` | Vision benchmark with image decode | `results/` | ✅ Active |
| `b70-monitor.sh` | GPU temperature/power monitor | stdout | ✅ Active |
| `~/inference/benchmark.py` | Quick smoke test (wall-clock) | `~/inference/benchmarks/` | ⚠️ Dev only |

Results saved to:
  `~/B70-DOCS/benchmarks/`  ← Human-readable A/B reports (with commands + analysis)
  `~/B70-DOCS/results/`     ← Raw llama-bench / script output (txt, json)
  `~/B70-DOCS/docs/benchmark-history.md` ← Chronological run log (all builds)

Documentation (restructured 2026-08-03):
  ~/B70-DOCS/AGENTS.md              ← This file (master reference)
  ~/B70-DOCS/DOCUMENT-MAP.md        ← Doc index
  ~/B70-DOCS/docs/                  ← All reference documentation
    benchmark-protocol.md           ← How to benchmark (methodology, KL-div, discipline)
    benchmark-history.md            ← Dated run log (each run: date/build/results)
    kv-cache-quantization.md        ← KV cache definitive reference
    recommended-configs.md          ← Per-model launch table
    launcher-guide.md               ← How the launcher works
    lvm-and-storage.md              ← Disk layout, LVM, VRAM rules
    speculative-decoding-on-b70.md  ← MTP/ngram analysis
  ~/B70-DOCS/research/              ← Deep technical analysis
    b70-xe2-architecture-prefill-analysis-20260807.md  ← Xe2/XMX/DPAS hardware
      facts + prefill bottleneck analysis + optimization candidates.
      READ THIS before re-doing any B70 architecture web search — it has the
      verified facts (32 Xe-cores, 256 XMX, DPAS geometry, INT4 peak estimate,
      vllm-xpu-kernels kernel mapping) with citations.
  ~/B70-DOCS/docs/archive/          ← Stale/superseded docs (history)
  ~/inference/                      ← Launcher scripts and profiles.json

## B70 BENCHMARK CLAIMS SKILL
  `.agents/skills/b70-benchmark-claims-standard/SKILL.md` is mandatory for any benchmark, comparison, public number, portfolio/social update, or leaderboard submission. It defines claim-sized lanes, peak/median/sustained vocabulary, C1/Cn separation, evidence tiers, and the publication gate.

## DEFAULT MODEL BENCHMARK SKILL
  `.agentic/skills/benchmark-model/SKILL.md` is the entry point for “benchmark this model” or “get real numbers.” It defaults to the five-cell Lane 1 model card, uses two-anchor n=3 screens for MTP or config selection, confirms finalists with a fresh n=5 set, reuses matching evidence, and expands only when the requested claim needs more dimensions.
  `.agents/skills/benchmark-model/SKILL.md` is the discovery router for
  `$benchmark-model`; it points to the canonical `.agentic` workflow.

## B70 vLLM PI CAMPAIGN SKILL
  `.agents/skills/b70-vllm-pi-campaign/SKILL.md` is the Lane 3 specialist for exact-token Pi workload tests, vLLM XPU scheduler sweeps, MTP long-context gates, the exact-128K MTP4 boundary patch, prefix-cache behavior, or mixed long-prefill plus short-chat campaigns. It is not the default new-model benchmark.

## B70 SPECIALIST SUBAGENTS (~/.zcode/agents/)
  b70-bench-engineer           ← runs publication-grade pp/tg and C1/Cn campaigns
  b70-vllm-server-specialist   ← launches/tunes vLLM XPU + MTP servers (NEW 2026-08-07)
  b70-architect                ← Xe2/XMX/DPAS hardware analysis (NEW 2026-08-07)
  b70-sycl-builder             ← llama.cpp SYCL builds, A/B testing
  b70-kv-cache-analyst         ← KV cache quantization configs
  b70-claims-auditor           ← fact-checks benchmark claims vs external sources
  b70-portfolio-writer         ← portfolio posts from benchmark results
  b70-localmaxxing-submitter   ← formats/submits to localmaxxing.com
  Use these instead of re-deriving B70 context from scratch.

## 11. COMMON PITFALLS

- SSH heredoc mangles $() and $VAR. Use pipe method instead.
- scp may get blocked by security scans. Use: cat file | ssh host "cat > file"
- sudo tee for power cap hangs if sudo needs password. Use sudo -n to test first.
- Don't trust wall-clock for decode speed. Use engine rate from timings JSON.
- 128 max_tokens produces empty content on reasoning models. Use 512+.
- VRAM debugfs format: grep "^usage:" then divide bytes by 1048576.
  (visible_avail line has a "MiB" suffix — strip non-digits.)
- desktop-b70 hostname may not resolve. Use <lab-host> directly.
- xpu-smi does NOT work on this stack. The B70 uses the **`xe` driver** (not
  i915), which exposes no clock-control sysfs — so `xpu-smi discovery` says
  "No device discovered" and `intel_gpu_frequency` says "No Intel card found."
  Both tools target i915/data-center stacks. The ONLY tunable is the hwmon4
  power cap (no GT min/max, no RPS, no clock locking). Use debugfs for VRAM,
  hwmon for power/temp. See §15.8.
- llama-bench hangs after printing results. Kill after tg32 line appears.
- ⚠️ NEVER load a single GGUF larger than ~30 GB into VRAM. The B70 has 32656 MiB
  visible VRAM; a 34 GB file (e.g. Laguna-S-2.1 IQ2_XXS) OVERFLOWS and causes a
  HARD SYSTEM CRASH (full reboot), not a cleanable OOM. Verified 2026-08-03.
  If you must run a >30 GB model, offload tail layers to CPU: `-ngl <N` where N
  leaves enough headroom, and do NOT use --no-mmap for the offloaded portion.
- ⚠️ When benchmarking, STOP the production llama-profile.service first
  (systemctl --user stop llama-profile.service), and confirm VRAM is freed
  (visible_avail should read ~31000+ MiB). Running two models = contention/OOM.
  Restart it after: systemctl --user start llama-profile.service.
- The production server is a USER systemd unit (llama-profile.service), driven by
  ~/inference/run-profile. It has Restart=on-failure and auto-starts
  on boot. Current profile: ThinkingCap-Qwen3.6-27B Q4_K_M, q8_0/q4_1 KV, 128K, MTP.


## 12. LMX BENCHMARKING (localmaxxing.com)

The official benchmark submission tool. Installed BOTH on Rock 5B (arm64) and
on the B70 itself (`/usr/local/bin/lmx`, x86_64, 2026-08-07) with live stored
auth (`lmx auth whoami` works; token in ~/.config/localmaxxing/config.json —
NEVER print it).

⚠️ CLI RENAMED (verified 2026-08-19): the old `lmx benchmark …` group is now
**`lmx speed-test …`**. There are also `lmx eval …` quality suites and
`lmx kvcache …`. Remote mode supports `--concurrency <n>` (submitted as
engineFlags.concurrency) and `--num-parallel <n>`.

### Quick start (current CLI, run on the B70 against a local server)
```bash
# Generate hardware.json (do once; repo copy: ~/B70-DOCS/b70-hardware.json)
lmx hardware template \
  --gpu-name "Intel Arc Pro B70" --gpu-count 1 --vram-gb 32 \
  --cpu "AMD Ryzen 7 5700X3D (8 threads)" --ram-gb 30 \
  --os Linux --power-watts 230 --hw-class DISCRETE_GPU \
  --out b70-hardware.json

# Real eval against a live server (exact HF artifact we serve):
lmx speed-test run vllm \
  --mode remote --base-url http://127.0.0.1:8000 \
  --hf-id "SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16" \
  --served-model qwen38 --quantization GPTQ-Int4 \
  --hardware b70-hardware.json \
  --max-tokens 256 --warmup 1 --iterations 3 \
  --concurrency 1 --out c1.json

# Validate + submit (stored auth; --api-key only if not logged in)
lmx speed-test validate-local c1.json
lmx speed-test runs submit c1.json
```

### KV Cache context sweeps
```bash
lmx kvcache run vllm \
  --mode remote --base-url http://127.0.0.1:8000 \
  --hf-id "SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16" \
  --levels 32768,65536,131072
```

### Agent skill
```bash
lmx skill install --dir .claude/skills
```

### Current lmx benchmark results

Submitted numbers are historical (Jul 2026, b9853, q5_0-q4_1 KV). For current
numbers see `docs/benchmark-history.md` (Run 5-6) and `docs/recommended-configs.md`.
Upload script: `~/inference/benchmarks/upload_b70_localmaxxing_jul16.py`
(see §13.4 for the submission schema and workflow).


## 13. PORTFOLIO & LOCALMAXXING WORKFLOW

The repeatable cycle is: benchmark → save raw evidence and manifest → mark provisional → protocol validation → claims audit → public portfolio/leaderboard/social update.

### 13.1 Benchmark → Results
Select Lane 0-3 from `docs/benchmark-protocol.md` §1.9 and run only the cells
needed by the claim. Index the run in `docs/benchmark-history.md`. Every number
needs actual workload coordinates, timing source, statistic/n, cache and
concurrency state, and power provenance. Keep engine, request-side, and aggregate
metrics separate.

### 13.2 Benchmark Claim Publication Gate
No result may become a README, portfolio, social, recommendation, cross-hardware, or leaderboard headline until:
1. Raw artifacts and all samples are saved.
2. The selected lane is complete and the claim stays inside measured cells.
3. Peak, representative, and sustained labels are correct.
4. C1 and Cn are separated.
5. Cache mode and zero cache-hit delta are documented for uncached HTTP prefill.
6. Configured cap and measured draw are not conflated.
7. Failures, exclusions, contamination, throttling, and OOM risk are recorded.
8. Every delta/multiplier passes an apples-to-apples check.
9. The `b70-claims-auditor` approves the exact wording and evidence links.

Until then label it **PROVISIONAL — NOT FOR PUBLIC HEADLINE**. LocalMaxxing `APPROVED` means published/accepted self-report, not independently verified.

### 13.3 Portfolio Post (sergiiob.dev, Astro)
**Repo:** github.com/SergiioB/portfolio → `src/content/posts/<kebab-slug>.md`
**Frontmatter (zod-typed):**
```yaml
---
title: ""
description: ""
situation: ""
issue: ""
solution: ""
usedIn: ""
impact: ""
pubDate: 2026-08-04
category: ["b70", "local-ai", "infrastructure"]
amazonUrl: https://go.sergiiob.dev/arc-pro
tags: ["local-ai", "llama.cpp", "sycl", "intel-arc", "arc-pro-b70", "benchmark"]
draft: false
---
```
**Body structure:** Situation → VRAM Math → Results (tables) → Config (bash) → Runtime Flags Handbook → Methodology.
**Naming:** lowercase kebab-case, no spaces/underscores.

**Categories (as of 2026-08-04):** `b70`, `infrastructure`, `automation`, `ai`, `cloud`,
`local-ai`, `kotlin`, `snippets`, `career`. All B70/Intel-Arc posts MUST carry the
`b70` category. The archive has a featured "Intel Arc B70" hub card (with a GPU
diagram picture) linking to `/b70/`, plus category filter chips that filter posts
client-side. Category pages are at `/<slug>/` (e.g. `/b70/`), NOT `/category/<slug>/`.
All post dates are 2026 (a 2024 batch was corrected on 2026-08-04).

**Choose tables from the claim. Do not pad every post with a full matrix.**

A Lane 1 post needs one compact model-card table covering the two cold-input
anchors, two decode anchors, and the real-workload control. Add the table types
below only when the post makes the matching broader claim:

1. **Input/prefill curve:** use multiple prompt sizes for a prompt-scaling claim.
```markdown
| Config | pp512 median (t/s) | pp2048 median (t/s) | pp4096 median (t/s) | pp8192 median (t/s) | n |
|--------|-------------------:|--------------------:|--------------------:|--------------------:|--:|
| Model A Q4 | 963 | 1410 | 1525 | 1375 | 5 |
```
For vLLM, label request-side `input tokens / TTFT` as cold input rate, never
llama-bench pp or isolated prefill.

2. **Generation-length curve:** use g32/g128/g256/g512 only when output-length
   behavior is part of the claim.
```markdown
| Config | tg32 peak (t/s) | tg128 median (t/s) | tg256 median (t/s) | tg512 median (t/s) | Δ at tg128 | n |
|--------|----------------:|-------------------:|-------------------:|-------------------:|-----------:|--:|
| Old build | 66.0 | 61.8 | 60.9 | 59.7 | — | 5 |
| New build | 74.2 | 69.5 | 68.3 | 66.8 | **+12.5%** | 5 |
```
Use tg128 representative performance as the default delta. Label tg32 as short/peak, and do not call tg512 sustained unless the predeclared sustained definition is met. Always state prompt/context coordinates and C1/Cn scope.

3. **Cross-hardware comparison:** optional. Include it only when the post's claim
   is cross-hardware and each row has a source plus comparison class.
```markdown
| Hardware | Model | Decode | Prefill | Context | Source |
|----------|-------|--------|---------|---------|--------|
| Arc Pro B70 32GB | Qwen 35B Q4 MoE | 70 t/s | 1659 t/s | 512K | this post |
| RTX 7800 XT 16GB | Qwen 35B Q4 MoE | ~45 t/s | ~900 t/s | 128K | [link] |
| RX 7800 XT 16GB | Qwen 35B Q4 MoE | ~42 t/s | ~850 t/s | 128K | [my bench] |
```
Never fill a comparison with estimates to complete the table. Mark unmatched
rows directional and keep them out of multiplier headlines.

**Table rules:**
- Units always in headers (t/s, MiB, ms, W, °C)
- Every performance row/table states prompt/output coordinates, statistic, `n`, C1/Cn, cache state, timing source, configured cap, and raw evidence/run ID
- A peak appears only in a column labeled peak/maximum observed; representative tables default to median
- Δ column uses **bold** for the percentage and names the exact matched cell
- Cross-hardware sources must be linked and classified `matched`, `matched-except-quant`, `directional`, or `not directly comparable`
- VRAM tables show "free after load" (not "used")
- Note build version, exact checkpoint/quant, KV config, and known correctness limits under every table

**RECIPE POST MAINTENANCE RULE (MANDATORY — every publication-approved campaign):**
The loop: benchmark → save raw evidence/manifest → append provisional Run N → protocol validation → claims audit → update the full-recipe post → `astro build` → commit + push.

The full-recipe post (`intel-arc-pro-b70-full-recipe-llama-cpp-sycl.md`, repo
github.com/SergiioB/portfolio, local clone `~/portfolio`) is the **living
source of truth** for B70 numbers. Every publication-approved campaign that changes headline numbers MUST update ALL of these spots in it:

1. Frontmatter `description` + `impact` (headline numbers)
2. "Current production build:" line (build id + commit)
3. The improvement section — append the new build-generation A/B table (e.g.
   "b10222 → master 0804"); NEVER edit older-generation tables (same rule as
   benchmark-history.md — old A/B data stays historical)
4. Cross-hardware table — B70 rows → current numbers
5. Model config tables (Dense + MoE) — decode/prefill columns
6. Production server launch section — binary path (`build-sycl-XXXX/bin/llama-server`)
   + "Expected:" line
7. "What this replaces" list
8. Top update-note pointing to the newest post (older posts point forward:
   qwen35 → full-recipe → upgrade post)

Hard rules:
- **Only measured numbers. Never extrapolate.** Quants/models not re-run on the new
  build get a "b10222-era, not re-run" footnote — never silently carried forward.
- Historical tables are never rewritten.
- `astro build` must pass before push (post must render; repo has prettier/lint-staged
  hooks on commit).
- Commit + push after every update (`git config` identity: SergiioB /
  <owner-email> — set locally per clone).
- Sensitive scan before push: grep posts for `bhk_`, `192.168.`, credentials — nothing
  goes into a public post.

### 13.4 LocalMaxxing Submission
**Schema (flat, working):**
```json
{
  "hfId": "Qwen/Qwen3.6-35B-A3B",
  "hardware": {"hwClass":"DISCRETE_GPU","gpuName":"Intel Arc Pro B70","gpuCount":1,"vramGb":32.0,"os":"Linux"},
  "engineName": "llama.cpp",
  "engineVersion": "b10222 SYCL",
  "quantization": "Q5_K_M",
  "backend": "xpu",
  "tokSOut": 67.7,
  "tokSPrefill": 1300.0,
  "contextLength": 262144,
  "batchSize": 1,
  "notes": "Single-stream SYCL. q8_0 K + q4_1 V KV, FA on, -ngl 99 -ncmoe 0, 150W.",
  "engineFlags": {
    "commandSnippet": "llama-server -m MODEL.gguf -ngl 99 -ncmoe 0 -fa on -ctk q8_0 -ctv q4_1 -c 262144 -b 8192 -ub 4096 -t 8 --no-mmap -dev SYCL0",
    "kvCacheDtype": "q8_0/q4_1",
    "flashAttn": true,
    "numParallel": 1
  }
}
```
The flat platform payload does not replace the internal evidence manifest. Preserve separate decode and prefill observation IDs and their exact prompt/output coordinates, statistic, cache state, and C1/Cn scope. Never place aggregate Cn throughput in a C1 decode field. Describe accepted records as approved/published self-reported submissions unless independently reproduced.

**Upload script:** `~/inference/benchmarks/upload_b70_localmaxxing_jul16.py`
(env: `LOCALMAXXING_API_KEY=bhk_... python3 upload_b70_localmaxxing_jul16.py`)
**Leaderboard weights:** VRAM 30% / Model size 30% / Run count 25% / TPS 15%.

### 13.5 Social Posts
Draft X/LinkedIn threads in `~/B70-DOCS/social-posts-b70-benchmarks.md`.
Pattern: 3-tweet thread + standalone + LinkedIn deep-dive + short variant.


## 14. CLAIMS VERIFIED REGISTER (2026-08-03 audit)

All technical claims fact-checked against external sources. Future work builds on these.

| # | Claim | Verdict | Source |
|---|-------|---------|--------|
| 1 | q8_0/q4_1 KLD ~0.0044 | REFUTED → **~0.003** measured | llama.cpp #23470 |
| 2 | q4_0 K "KL > 0.03" cliff | REFUTED → **KLD ~5.5** (catastrophic) | #23470 |
| 3 | MoE "compute-bound" | REFUTED → both **bandwidth-bound**; diff = active-param count | Mann, Puget |
| 4 | MTP +45-51%, 95.35% accept | PARTIAL → gain real; **acceptance 93-94%** | repo data |
| 5 | MTP never helps MoE | REFUTED (universal) → locally true (no MTP layers + near-ceiling) | Cohere, Gemma-4 |
| 6 | MTP+ngram stacking -2-3% | **CONFIRMED** (B70-specific) | repo data |
| 7 | Power: MoE 64@130W vs 68@150W | PARTIAL → direction right; MoE **self-limits ~140W** | repo sweep |
| 8 | Zero ctx penalty 64K-256K | **CONFIRMED** (add VRAM-wall caveat for dense 27B) | FA theory |
| 9 | IQ2_XXS 34GB crash | **CONFIRMED** (launcher -ngl 99; partial offload would work) | verified 2026-08-03 |
| 10 | SYCL PRs improve prefill | **CONFIRMED** (#25222 up to 4.26×; #25025 has f16 caveat #26413) | llama.cpp PRs |
| 11 | -ub 4096 optimal | **CONFIRMED** (empirical) | repo + community |
| 12 | DISABLE_OPT 44.7% regression | UNCERTAIN → var is **ENABLE_OPT**; 44.7% uncorroborated upstream | SYCL.md |


## 15. LESSONS LEARNED (append new findings here)

Hard-won knowledge from benchmarking on the B70. Each entry is a non-obvious
finding that would otherwise be re-discovered the hard way. **When you learn
something new, add it here.**

### 15.1 Measurement lessons
- **llama.cpp prefill: use `llama-bench` when supported.** Short prompts (34 tokens)
  report ~140 t/s; the same model at pp4096 reports ~1600 t/s. The HTTP API also
  has a shell-quoting bug that silently truncates prompts to 4 tokens (`prompt_n=4`).
  Always verify `prompt_n ≥ 1024` if using the API. See §9.4, benchmark-protocol §1.5.
- **`benchmark.py` measures wall-clock t/s** (`tokens/wall_time`), not engine rate.
  It also lacks warmup discard. Use it for quick smoke tests only, never for citable
  numbers. Use `llama-bench` or the bash scripts reading `timings.predicted_per_second`.
- **`llama-bench` hangs** after printing results on SYCL builds. Watch for the `tg128`
  line, then kill. Results are valid if printed.

### 15.2 Architecture lessons
- **KV cache scales with ATTENTION size, not total params.** MoE 35B-A3B uses only
  ~1847 MiB KV per 128K (tiny 3B-active attention) vs dense 27B's ~6960 MiB (full
  27B attention) — a 3.8× difference. This means a 25 GB MoE reaches 512K context
  while a 16 GB dense can't pass 256K. Never assume KV scales with model size.
  See §9.3 for the full budget table.
- **Measured llama.cpp dense and MoE decode are bandwidth-bound** at 608 GB/s.
  Prefill can have a different bottleneck; dense vLLM prefill was compute-bound.
- **Power behavior is phase and stack specific.** Run 24's paired vLLM MoE+MTP
  prefill A/B was flat from 150W to 230W. Dense workloads have measured gains.
  Do not route power from the dense/MoE label alone.

### 15.3 Stability lessons
- **Never load a single GGUF >30 GB** with `-ngl 99`. The B70 has 32656 MiB VRAM;
  a 34 GB model overflows and causes a **hard system crash** (full reboot, not a
  cleanable OOM). Use `-ngl <N` for partial CPU offload. Verified with Laguna 2026-08-03.
- **Production server is `llama-profile.service`** (user systemd unit). It has
  `Restart=on-failure` and auto-starts on boot. Stop it before benchmarks:
  `systemctl --user stop llama-profile.service`. Restart after.
- **`temp2_input` is the GPU sensor, not `temp1_input`.** The latter doesn't exist
  on hwmon4. Several launcher scripts had this wrong (fixed 2026-08-03).
- **VRAM overflow degrades SILENTLY, then crashes the system (verified 2026-08-05).**
  With MTP spec buffers, dense 27B fits only at 128K ctx: Q4@256K+MTP and Q5@200K+MTP
  both overflow (0 MiB free) — the §9.3 boundary table was measured WITHOUT MTP buffers.
  Symptoms of overflow: garbage timings (decode 0.07-20 t/s, prefill 4-18 t/s) with no
  error logged, then a system-wide OOM shutdown. Always check `visible_avail` after
  load; treat <1 GiB free as unsafe, and never run two llama processes concurrently
  (VRAM contention → same crash class).

### 15.4 Quantization lessons
- **q4_0 K is catastrophic** (KLD ~5.5, only 11.6% top-p match). Not a gradual
  degradation — the model output is destroyed. Never use q4_0 for K-cache.
- **q8_0 K + q4_1 V is the fleet standard** (KLD ~0.003, ~50% VRAM savings).
- **"Never below q8 K" is NOT universal** — q5_0/q5_1 K are community-accepted
  trade-offs (KLD ~0.006-0.008). See docs/kv-cache-quantization.md.

### 15.5 Build lessons
- **b10222 is +6-13% faster than b9853** on both prefill and decode (measured A/B).
  The gains come from oneDNN XMX FA (#25222, up to 4.26× prefill), fused top-k MoE
  (#25217), RMS_NORM fusion (#26015), and contiguous elementwise (#25946).
  See benchmarks/b9853-vs-b10222-qwen35b-q4-moe-2026-08-03.md.
- **Don't rebuild for CUDA/Metal/OpenCL commits** — they don't affect SYCL. Only
  rebuild when a `sycl:` commit touches kernels/attention/MoE, or when adopting a
  new model architecture (DeepSeek-V4, Qwen3-Next). See b70-sycl-builder agent.

### 15.6 Model download lessons (verified 2026-08-03)
- **FASTEST METHOD: resolve the HF CDN URL first, then `aria2c -x16` on it.**
  HuggingFace redirects `/resolve/main/FILE` to a signed Xet CDN URL
  (`us.aws.cdn.hf.co/...`). Pointing `aria2c` directly at the HF URL gets HTTP 403
  (signed URL expires under multi-connection). But the **resolved** CDN URL
  supports range requests and works perfectly with 16 connections at **~86 MB/s**.
  ```bash
  # CORRECT — 3x faster than wget, 4x faster than curl:
  CDN=$(curl -sI -L -o /dev/null -w '%{url_effective}' \
    'https://huggingface.co/REPO/resolve/main/FILE.gguf')
  aria2c -x16 -s16 -k1M --file-allocation=none -o FILE.gguf "$CDN"
  ```
- **FALLBACK: `wget -c`** (~50 MB/s) if aria2c is unavailable.
  `wget -c --tries=10 -O MODEL.gguf 'https://huggingface.co/REPO/resolve/main/FILE.gguf'`
- **`curl` alone is slowest** (~20-30 MB/s) and `curl -C -` can silently die on
  HF redirects without logging an error. Avoid for large files.
- **`pip`/`pip3` are NOT installed** on this host. `python3 -m pip` also fails.
  To use `huggingface_hub`: `sudo apt install python3-pip && pip3 install huggingface_hub`.
- **Choose the destination from live capacity, not this document.** Before every
  download, conversion, or image build, run `df -h / /mnt/<models-disk> /mnt/<models-disk>`
  plus `findmnt`. For a conversion reserve room for its retained source, full
  output, staging/temporary files, and a safety margin on every affected volume
  (including `/` for container builds). Do not begin a source+output conversion
  based on a historical free-space snapshot.
- **Two simultaneous downloads split bandwidth** (~40 MB/s each vs ~86 single).
  Download one at a time for maximum speed unless a time-critical transfer has
  enough independently verified capacity.
- **Always verify** before benchmarking — check GGUF magic bytes:
  `python3 -c "import struct; f=open('M.gguf','rb'); print(f.read(4), struct.unpack('<I',f.read(4))[0])"`

### 15.7 Launcher lessons (verified 2026-08-07)
- **`except Exception: pass` around an HTTP call in a poll loop is a bug
  factory.** The launcher's vLLM wait-loop called `urllib.request.urlopen` with
  `urllib` never imported at module level (only locally inside
  `get_loaded_model()`), so every poll raised a silent `NameError` and the loop
  printed "Loading model..." forever while making **zero network syscalls** —
  server came up at 245s, launcher still polling at 600s. Diagnosis: `strace
  -e trace=connect` showed no connects after the pre-loop checks, proving the
  loop never touched the socket. Fix: module-level import + print the first
  swallowed exception once. See docs/launcher-guide.md §8 postmortem.
- **Verify engine-up detection with a mock server, not the real one.** Start a
  tiny HTTP server on the launcher port that answers 200 on any path, run
  `launch <vllm-profile>`, expect `✓ vLLM up after 0s` in seconds. Turns a
  5-minute test cycle into a 5-second one.

### 15.8 MTP speculative-decoding lessons (vLLM XPU, verified 2026-08-07)
- **MTP spec acceptance falls with draft position — it's fundamental.** On the
  single-layer MTP head of Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4, per-position
  acceptance measured from `vllm:spec_decode_*` counters: pos0 92.6% → pos1
  84.2% → pos2 72.7% → pos3 68.9% (80.1% overall at N=4). Each draft token is an
  autoregressive guess off the previous guess, so errors compound down the
  chain. Engine logs this: *"num_speculative_tokens > 1 will run multiple times
  of forward on same MTP layer, which may result in lower acceptance rate."*
- **N=4 is the throughput optimum, NOT N=1.** Measured spec-N curve: N=1 =
  97.1% accept / 141.8 t/s; N=2 = 92.1% / 174.5; **N=4 = 80.1% / 198.2**. So
  chasing "98-99% acceptance" by lowering N **costs 30% throughput**. Don't do
  it for serving. Combining high-N + high-acceptance requires a multi-layer
  draft model (DeepSeek-V3-style 3-layer MTP), which this checkpoint lacks
  (`mtp_num_hidden_layers: 1`).
- **Power cap does NOT help MoE+MTP prefill (verified, paired A/B).** Alternated
  150W/230W back-to-back on the same warm server, 3 rounds: p2k/p4k/p8k all
  within ±0.2%. The MoE grouped-GEMM is bandwidth-gated, not power-gated. Raising
  the cap to 230W is pure waste heat here. (Dense llama.cpp decode DOES scale
  +18-30% from 150→230W per Run 19 — different workload.) Live draw: prefill
  ~171W, decode ~113W, idle ~47W.
- **GT clock locking is IMPOSSIBLE on the `xe` driver.** `intel_gpu_frequency`
  and `xpu-smi` are installed but both fail: the B70 uses `xe` (not i915), which
  exposes NO clock-control sysfs/debugfs — only read-only PMU counters at
  `/sys/devices/xe_0000_0b_00.0/events/gt-*-frequency`. There is no gt_min /
  gt_max / RPS knob. The only power lever is the hwmon4 cap. Don't waste time
  trying to "lock clocks during prefill."
- **vLLM prefix caching SILENTLY inflates prefill benchmarks 5×.** Harnesses
  using a CONSTANT filler string (old `b70-mtp-prefill-sweep.py`,
  `b70-mtp-grid-bench.py`) make reps 2-3 hit the cache and report prefill that
  rises from 8,640 → 42,733 t/s — a 5× lie that the "steady-state = reps 2-3"
  averaging hid. ALWAYS prepend a unique random prefix per call. Use
  `scripts/tmp/b70-prefill-honest.py`. Read `vllm:prefix_cache_hits_total` to
  confirm zero cache hits before trusting a prefill number.

### 15.9 Kernel wheel rebuild lessons (vLLM XPU, verified 2026-08-08)
- **The stock w4a16 tile config (128×256×32) is already optimal for our MoE —
  do NOT re-tune it.** Built a wheel with `w4a16_policy_n_128` (128×128×32) +
  env-gated dispatch, clean A/B in the same session: p2k-p8k prefill FLAT
  (±1% noise). The +17% at p500 is launch-overhead at tiny M, irrelevant. The
  prefill bottleneck is L2/DRAM weight streaming (int4), which is
  tile-independent — NOT N-column parallelism. Run 25. Sources:
  `research/b70-wheel-n128/`.
- **Rebuilding vllm-xpu-kernels on this host (the recipe that works):**
  1. Clone to `/mnt/<models-disk>/vxk-build` (persistent — `/tmp` is tmpfs and a
     reboot wipes builds).
  2. Build inside a container: `vllm/vllm-openai-xpu:nightly` + mount host
     `/opt/intel` (oneAPI 2026.0 icpx) read-only. Source setvars.sh BEFORE the
     venv. Set `SYCL_HOME=/opt/intel/oneapi/compiler/2026.0`.
  3. `grouped_gemm_xe2.cpp` single-TU OOMs at >18 GB — build with `-j1` + a 24g
     container memory limit for that file; the rest at `-j3`.
  4. Full build = 1232 objects, ~4-5 hrs at -j3 (oneDNN+cutlass deps,
     grouped-GEMM, 462 attention TUs, gdn, mhc, mqa). Ninja resumes on reboot
     if the source lives on a persistent volume.
  5. Install wheel into a derived image (`pip install --force-reinstall
     --no-deps`), verify the new policy symbols via
     `nm -C libgrouped_gemm_xe_2.so | grep policy_n_128`.


### 15.10 Real-world Pi + exact-128K vLLM lessons (verified 2026-08-08)
- **MTP4 exact 128K works only with the partial-final-group metadata patch on this pinned stack.** The unpatched p130944/g128 request emitted 124 tokens, then the XPU GDN kernel rejected a four-token final group because MTP4 normally requires five query tokens (`1 target + 4 drafts`). `scripts/tmp/patch_mtp_boundary.py` reclassifies only that truncated final group as stateful non-spec prefill. The patched request completed all 131,072 tokens: TTFT 48.601 s, client post-first 96.87 tok/s, 72.31% MTP acceptance, zero cache hits, 2,619 MiB free, logged KV capacity 165,961. Raw: `results/vllm-pi-mtp4-128k-boundary-fix-20260808T214938Z-77748/`.
- **MTP4 correctness does not make it the fastest 128K setting.** The matched single MTP2 boundary observation reached 103.63 client post-first tok/s with 86.96% acceptance versus MTP4's 96.87 tok/s and 72.31%. Treat this as a single-observation production-routing signal, not a median A/B.
- **MTP4 serial speed and mixed-load throughput are different surfaces.** C1 p9445/g128 reached about 159 tok/s post-first. Unpatched MTP4 still crashes when a long prefill and speculative decode share the XPU `causal_conv1d` invocation. Qwen3.8 hybrid-GDN now has a Python overlay that splits that mixed step (see §15.14); that does not rewrite this older no-spec mixed-load number. No-spec completed p65408/g128 plus 20 short requests at 74.46 aggregate output tok/s, but that wall-clock denominator includes the 64K prefill. Historical all-short no-spec C16 reached 694 aggregate tok/s; do not compare the two as the same workload.
- **The public cookbook must pin the real image and matching patch generation.** Current reproducible stack: `vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97`, then `patch_mtp_nightly.py`, then `patch_mtp_boundary.py`. `intel/vllm:0.21.0-xpu-int4moe` was a local derived research image and was never published to Docker Hub. Never tell a reader to pull it or substitute `intel/vllm:0.21.0-xpu`; the historical patches target different source. Public matrix: `intel-arc-pro-b70-inference-cookbook/docs/IMAGE-AND-PATCH-MATRIX.md`.
- **vLLM V1 defaults prefix caching to on; omission is not disablement.** The first Run 27 cache-off attempt omitted `--enable-prefix-caching`, so unique cold prompts showed zero hits even though the feature remained active. The resident request then exposed 119,680 reused tokens and the harness rejected the cell. Use `--no-enable-prefix-caching`, verify `enable_prefix_caching: False` in the server log, and require zero hits on a resident-prefix request.
- **Resident-prefix reuse changes the production choice more than speculative depth.** In matched p130944/g128 cold + resident-120K tests (`n=5` per cell), cache reuse cut resident TTFC by 31.46–66.32× and E2E by 14.57–16.23×. MTP2 + cache on had the best resident E2E median (2.504 s); no-spec + cache on had the fastest TTFC (0.554 s). Cold exact-128K completed fastest with no-spec (43.793 s). Route by workload instead of assuming MTP4 is always best. Raw: `results/vllm-pi-128k-cache-spec-matched-20260808T224033Z-101198/`.

## 16. vLLM LANE 3 FULL CHARACTERIZATION MATRIX

Do not run this matrix for an unqualified “benchmark this model” request. Start
with `.agentic/skills/benchmark-model/SKILL.md`. Use this procedure when the
claim needs a full cross-mode input curve, generation-length curve, and context
boundary on a different model. The normative public table and evidence contract is
`~/intel-arc-pro-b70-inference-cookbook/docs/BENCHMARK-FORMAT.md`.

### 16.1 Intake: start a new result generation

Before launch, write a manifest with:

1. Pullable image repository plus immutable digest; observed in-container vLLM
   and `vllm-xpu-kernels` versions. Do not infer versions from a tag.
2. Model repository, exact revision, local path, quantization, sizes and hashes;
   verify whether MTP tensors really exist rather than trusting config alone.
3. Tokenizer/chat-template identity; context, scheduler budget,
   `gpu-memory-utilization`, max sequences, cache mode, and configured cap.
4. Ordered patch names and SHA-256 hashes. A source-anchor or patch-generation
   mismatch is a hard stop.
5. UTC run ID, protocol version, host/kernel/driver identity, harness paths, and
   evidence status.

A changed image, package, model revision, quantization, tokenizer, chat template,
prompt file, patch, scheduler, context, or memory setting starts a new generation.
The 2026-08-09 tested baseline is image
`vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97`,
vLLM `0.26.1rc1.dev457+gc810e5ee9.xpu`, `vllm-xpu-kernels 0.1.12`,
preserved-MTP Qwen3.6-35B-A3B GPTQ-INT4, context 131072, scheduler 8192,
`gpu-memory-utilization=0.85`, and configured 165 W. PyPI kernel package 0.1.12.2
is newer but untested.

### 16.2 Calibrate exact rendered prompts

Run tokenizer-only calibration inside the tested image, without a GPU model load.
Count `len(encoded["input_ids"])`, never `len(tokenizer(prompt))` or
`len(BatchEncoding)`. Save exact rendered messages and hashes.

Use two prompt classes:

- Compact fixed benchmark system prompt for every standard p512 through p8192 cell.
- Full Pi system prompt for the p9445 historical control and full-context cells.

Generate six entropy-first exact prompts per coordinate: one warmup plus five
measured. Entropy comes before all shared text and must fit inside the target token
count. Prompt hashes must match across no-spec/MTP modes.

### 16.3 Run this exact matrix only for the Lane 3 claim

- Input phase: p512/g1, p2048/g1, p4096/g1, p6144/g1, p8192/g1,
  and full p131071/g1.
- Decode phase: p512 and p8192 at g32/g128/g256/g512.
- Historical control: p9445/g128.
- Full-context decode: p130944/g128 and p130560/g512.
- Modes: no-spec and every model-supported MTP depth. The current baseline uses
  no-spec, MTP1, MTP2, and MTP4. Mark unsupported modes; do not invent rows.

For each mode and coordinate, run one full-output same-shape warmup and discard it,
then five measured C1 requests. Decode cells MUST use `ignore_eos=true` so every
request returns the requested length. A g1 input cell does not need it. Retain an
early-EOS cell as excluded evidence and rerun the complete affected cell with
forced exact output; never mix partial and exact outputs in one median.

Keep cache/resident-prefix and concurrency campaigns separate from this C1
cold-prefix matrix.

### 16.4 Cache, MTP, isolation, and capacity gates

Run the cold matrix with prefix caching enabled, entropy-first unique prompts, and
zero `vllm:prefix_cache_hits` delta in every measured cell. Diff both
`vllm:prefix_cache_hits` and `vllm:prefix_cache_queries`; queries must agree with
actual prompt tokens. The absolute counter may be nonzero. For MTP modes, retain
proposed/accepted counters and acceptance. For no-spec, require zero proposed MTP
tokens.

Apply §8.1 before every model load: empty process/container state and about
31,000+ MiB empty-card `visible_avail`. After load:

- below 500 MiB: abort;
- 500–1,023 MiB: isolated C1 capacity observation only;
- 1,024–2,047 MiB: staged C1 research;
- at least 2,048 MiB: preferred C1 performance;
- at least 3,072 MiB: mixed/concurrent target.

Logged KV capacity must exceed prompt plus requested output before a full-context
cell starts. A server launch at 131072 does not prove an exact-128K completion.

### 16.5 C1 timing names

C1 means one active measured request and no queued request.

- TTFT: client monotonic request send to first SSE output token.
- Cold input rate: actual endpoint prompt tokens divided by client TTFT. It
  includes scheduling, uncached prompt processing, and first-token work. It is
  not isolated prefill and NEVER llama-bench `pp`.
- Client post-first rate:
  `(completion_tokens - 1) / (request_end - first_generated)`. It is request-side,
  not engine-native vLLM decode and NEVER llama-bench `tg`.
- Client post-first TPOT is the inverse interval formula; E2E is client request
  send to completion.

Do not place any of these beside llama-bench pp/tg under an unlabeled shared
metric. Speed does not establish output, token, logit/KL, or task-quality parity.

### 16.6 Evidence layout and compiler

Save:

```text
results/vllm-pi-prefill-decode-matrix-<UTC>-<PID>/
  manifest.txt
  package-versions.txt
  prompts/
  no-spec/
  mtp1/
  mtp2/
  mtp4/
  summary.json
  tables.md
```

Each mode keeps its manifest, complete server log, metrics snapshots, monitor log,
VRAM snapshot, harness logs, raw SSE/request records, per-cell results, failures,
exclusions, and cleanup state. Preserve invalid and superseded cells.

Compile and render only after all supported modes complete:

```bash
python3 scripts/tmp/b70-compile-prefill-decode-matrix.py "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.json"
python3 scripts/tmp/b70-render-prefill-decode-tables.py \
  "$RUN_ROOT/summary.json" --output "$RUN_ROOT/tables.md"
python3 .agentic/skills/b70-benchmark-visuals/scripts/render-prefill-decode-svg.py \
  "$RUN_ROOT/summary.json" \
  --dashboard "$COOKBOOK/docs/assets/b70-prefill-decode-dashboard.svg" \
  --method "$COOKBOOK/docs/assets/b70-benchmark-method.svg"
```

`summary.json` is the canonical numeric source. The compiler must fail closed on
wrong tokens, prompt mismatch, cache contamination, server errors, incomplete
modes, and missing replacement evidence.

### 16.7 Publication and cookbook checklist

Do not copy numbers outward until all items pass:

1. Complete matrix, exact outputs, `n=5`, compiler pass, raw evidence, exclusions,
   package/model/patch provenance, and configured cap versus measured draw.
2. Six separate public views: cold input, p512 decode, p8192 decode, p9445
   control, full-context decode, and exclusions. State units, C1, cache, timing
   source, cap, software stack, statistic, and E-tier next to the tables.
3. Claims audit confirms input rate is not llama-bench pp, post-first rate is not
   engine-native vLLM decode, and speed is not correctness proof.
4. Append `docs/benchmark-history.md` with exact raw path, software, method,
   exclusions, canonical summary tables, and status. Never rewrite old runs.
5. Create or update the cookbook: README current stack and concise tables;
   `FULL-SETUP-COMMANDS.md` clone/device/image/model/version/hash/launch/matrix
   commands; `BENCHMARK-FORMAT.md` contract; newest result first in
   `REAL-WORLD-PI-BENCHMARKS.md`; compact public summary JSON; every portfolio
   surface that repeats a changed number.
6. Use `.agentic/skills/b70-benchmark-visuals/SKILL.md` to regenerate the
   dashboard and method SVG from the canonical summary. Verify both in a browser
   and copy them byte-identically to cookbook and portfolio asset folders.

Keep a self-run with raw evidence at E2 provisional until independent comparable
or matched reproduction reaches E4/E5. No commit, push, deployment, social post,
or leaderboard submission is part of benchmark completion unless separately
authorized.

## 17. DENSE INT4 vLLM XPU LESSONS (verified 2026-08-09, Run 29)
- **Dense 27B GPTQ-INT4 linear DOES run on the pinned nightly via XPUwNa16LinearKernel** — and it is fast: 73.17 t/s C1 decode at p512 with 85.34% MTP acceptance (BF16 MTP head), vs ~24-29 t/s GGUF Q4/Q6-MTP on llama.cpp. Both MTP patches apply unchanged to the dense `Qwen3_5ForConditionalGeneration` (same shared `qwen3_5_mtp.py` / `gdn_attn.py`).
- **fp8 KV is REQUIRED for dense 128K, not optional.** Dense 27B attention needs 9.5 GiB fp16 KV at 128K; at U=0.85-0.90 the fp16 KV available is only 4.77 GiB (est. max 58,240). `--kv-cache-dtype fp8` halves it to ~4.75 GiB → 156,745-token capacity at U=0.90 with 529 MiB free. Without fp8, dense 27B + MTP4 cannot reach 128K on this card.
- **Dense long-context prefill collapses to 4% of XMX peak.** p4096 = 1,284 t/s (~10% of 734 TOPS INT4 peak, 71 TFLOPS); p130944 = 547 t/s (4.1%). Dense prefill is compute-bound, NOT memory-bound — weight streaming is 3-48 GB/s vs the 608 GB/s ceiling (~1% at p4096). The 128K collapse is the full-attention O(N²) term.
- **Dense 27B context ceiling on 32 GB with MTP4 + fp8 KV is 128K, safely.** 200K loads (KV capacity 207,684) but leaves 3 MiB free after load — the §8.1 abort zone, do not run (verified: the launcher's own 500 MiB floor gate aborted it). 256K is VRAM-infeasible (9.5 GiB KV needed > 7.82 GiB available; est. max 209,664). Quantizing the MTP head would reclaim only ~0.6 GB (~2-3K tokens) — not worth it.
- **Cache hits eliminate tokens; they do not speed up prefill.** Resident 32K follow-up: 29,952/32,797 tokens cached (91.3% hit) → TTFT 10.2× faster (35.30 → 3.472 s); new-token prefill rate was ~819-925 t/s in both cold and cached cells. The naive `prompt_tokens/TTFT` on a cached cell (9,447 t/s) is the silent-5×-lie class again — always report new-token rate.
- **Realistic Pi short-turn decode is 44-56 t/s, not 73.** MTP acceptance drops to 44-60% on real prompts (vs 85% synthetic filler). The 73 t/s synthetic number is peak, not the serving expectation.
- **MTP4 wins every dense decode cell; MTP2 is the close second (Run 31, 4-mode Lane 1 card at 230 W).** p512/g128: 69.3 (MTP4) vs 63.6 (MTP2) vs 50.5 (MTP1) vs 32.9 (no-spec) tok/s; p8192/g128: 64.1 vs 60.7 vs 46.9 vs 31.5. Full-context p130944/g128: 47.6 vs 42.7 vs 36.8 vs 23.1. MTP gains shrink at g512 (57.8-64.1 vs 57.1-60.0) — draft-run overhead catches up on long outputs. Prefill is flat across MTP depth (1,654-1,816 t/s; worst cell MTP4 p8192 within ~5% of the best mode).
- **MTP acceptance per mode (p512/g128 totals-diff): MTP1 95.1%, MTP2 94.8%, MTP4 83.8%** — the §15.8 per-position decay holds on dense; higher acceptance does NOT mean higher throughput (MTP4 has the lowest acceptance and the best speed).
- **Dense power draw at 230 W cap: no-spec/MTP1 campaign mean 195-197 W (max 275-278 W), MTP2/MTP4 146 W (max 246-266 W)** — but coverage differs (no-spec/MTP1 windows included heavy full-context prefill cells; MTP2 was a partial 223 s window; MTP4 a dedicated mixed observation). Mode-vs-mode draw deltas are directional only, never paired (§8.6). Max spikes above the cap on short prefill bursts are recorded as measured.
- **MATCHED power A/B (2026-08-10): MTP depth is NOT a power lever on dense.** Same mixed workload (1× p2048/g1 prefill + 2× g128 decode), fresh entropy prompts, per-mode server, 230 W cap, monitor windowing only the active requests, cooldown ≤55°C between modes: no-spec 149.9 W, MTP4 151.0 W, MTP1 156.1 W, MTP2 153.3 W mean (all within a 6 W band; max 0.5s 238-252 W, pkg 70-74°C). The Run 31 campaign-monitor gap (195 vs 146 W) was a window-coverage artifact (full-context prefill cells inside the no-spec/mtp1 windows), not a mode effect. Evidence: `results/dense27-matched-power-20260809T214116Z/`.
- **W4A8 kernel track is DISCARDED (2026-08-10).** The Run 30 W4A8 prefill numbers (2,395-2,576 t/s, +41-43%) are NOT a production path: per-token int8 activation quantization diverges from W4A16 at word 15 (32% content similarity at greedy sampling) and stacks quantization error on long generations. Do not use W4A8 for any publication, recommendation, or submission; keep `patch_dense_w4a8.py` env-gated off. The honest dense prefill ceiling at 230 W is the W4A16 number (~1,654-1,816 t/s) plus whatever the scheduler-budget lever adds.
- **GPU clock: the 230 W cap IS the ceiling; there is no clock cap to un-cap.** Verified 2026-08-10: writing 300,000,000 µW to `power1_cap` is REJECTED (readback stays 230,000,000 — the stock max); the `xe` driver exposes no gt_min/gt_max/RPS (xpu-smi and intel_gpu_frequency both fail, §15.8). Frequency IS measurable via perf PMU: `sudo perf stat -e 'xe_0000_0b_00.0/gt-actual-frequency/' -e 'xe_0000_0b_00.0/gt-requested-frequency/' -a sleep 3` (idle ≈ actual 1,050 MHz / requested 633 MHz). Record frequency + temperatures per run instead of assuming a clock lever exists.

### 15.11 NemotronH native MTP, grouped-topk XPU bug, N-gram, and converter safety (verified 2026-08-12)
- **The official NVIDIA BF16 Nemotron-3.5-Lightning checkpoint carries a real MTP head** (270 tensors: `mtp.layers.0` attention + `mtp.layers.1` MoE with 128 routed + shared experts + e_score_correction_bias, all in `model-00014-of-00014.safetensors`, 2.67 GB). The latest XPU nightly ships the full native machinery: `nemotron_h_mtp.py` (`NemotronHMTP` / `NemotronHMultiTokenPredictor`), registry `NemotronHMTPModel`, `hf_config_override` auto-maps `nemotron_h` + `num_nextn_predict_layers>0` → draft, V1 `MTPSpeculator` (`{"method":"mtp"}`). Embeddings/lm_head are shared from the target. Convert the head with the same GPTQ INT4 G64 symmetric contract as the backbone (`scripts/tmp/b70-nemotron-mtp-head-gptq.py`; output `nemotron-lightning-gptq-sym64-mtp-20260812/`). The MTP-enabled artifact serves fine WITHOUT `--speculative-config`.
- **Native MTP crashes at engine-init warmup on this stack**: the DRAFT's MoE falls back to the torch routing path, which hits the **unconditional `@torch.compile` on `grouped_topk`** (`fused_moe/router/grouped_topk_router.py`), and the compiled graph gathers `scores[topk_ids]` with out-of-range ids → device assert `index out of bounds: 0 <= tmp0 < 128` → EngineCore death. The BASE never executes it (routing fused in the native XPU WNA16 MoE kernel). Retained evidence: `/mnt/<models-disk>/inductor-cache/`, run dirs `results/nemotron-gptq-mtp2-screen-20260812T15*`. Keeping quant_config on the draft (`patch_mtp_quant_draft.py`, `patch_mtp_draft_quant.py`) did NOT fix it — the grouped_topk compile is independent of draft quantization. This is the same `tmp0 < 128` bug family as the compiled-mode base blocker (§15.10 era, `nemotronh-xpu-index-overlay`).
- **N-gram speculation is NON-deterministic at temperature 0 on this stack** (two identical `temperature=0, seed=` raw completions diverged; quality varied across runs). Not a production candidate without upstream fixes.
- **Converter safety**: a merge-style converter that symlinks source files then writes an output index through the symlink will CORRUPT the source artifact. Exclude every file the converter overwrites from symlinking and refuse to write through a symlink. Verify source integrity (index ↔ files, tensor counts) after any merge. See `b70-nemotron-mtp-head-gptq.py`.
- **`systemctl --user` from agent shells fails unless the login session env is set**: export `XDG_RUNTIME_DIR=/run/user/$(id -u)` and `DBUS_SESSION_BUS_ADDRESS=unix:path=${XDG_RUNTIME_DIR}/bus` before any `systemctl --user stop/start` (e.g., to stop `vllm-dense-profile.service` before a campaign). Otherwise `docker rm -f b70dense` alone lets the unit respawn the container.
- **XPU graph capture unblocks Nemotron decode to ~92 t/s (provisional, 2026-08-12).** Root cause of the weak 21.8 t/s eager decode: ~950 kernel launches/token, 32.2 ms/token CPU enqueue time vs 9.45 ms actual GPU (profile in `results/nemotron-gptq-builtin-profile-20260812T171658Z-671822/`). With `VLLM_XPU_ENABLE_XPU_GRAPH=1` + compiled mode + the native graph-safe grouped-topk router (`scripts/tmp/patch_xpu_grouped_topk_native_v2.py`: repeated argmax + masks, `torch.compiler.disable` graph break) + SSU B8/W4 config + `--async-scheduling`: PIECEWISE+FULL decode graphs capture and decode reaches **92.75 / 91.12 t/s** (p512/p8192 g128, n=3, C1, cache off, 150 W) — 4.26x/4.19x over eager, above llama.cpp's 78 reference. Evidence: `results/nemotron-gptq-graph-sampler-eager-screen-20260812T2030Z/`.
- **Deterministic replay FAILS for the compiled/graph path (upstream XPU bug).** Identical temperature-0/seed requests diverge at contested near-tie tokens (both continuations coherent). Isolation: full-eager + native router IS deterministic (router exonerated); sampler-eager and no-async don't help. Root: XPU torch.compile/triton kernel FP race (same class as the PowerPC argmax note in vLLM's `topk_topp_sampler.py`). Not patchable at the app layer; needs upstream vLLM/XPU deterministic reductions. The 92 t/s number is PROVISIONAL until the operator accepts the caveat or the upstream fix lands.

### 15.12 Nemotron DFlash on vLLM XPU (verified 2026-08-13, Run 36)
- **Official `method=dflash` works on this XPU stack.** Isolated n=5, C1, cache off, 150 W: representative decode **186.61 t/s** at p2048/g128 (174.60–201.83, 56.5% accept); p8192/g128 **157.92** (1.81× vs matched no-spec 87.25 on that cell); p8192/g1 cold input **7160**. Window acceptance 1830/3521 = 52.0%. Evidence: `results/nemotron-dflash-bf16-n7-n5-20260813T082203Z-1407994/`.
- **Native MTP is still 0%.** DFlash sidesteps the MTP head. Do not reopen the retracted V1-bypass issue.
- **The official DFlash checkpoint is NVFP4.** Convert MLP/`fc` E2M1→BF16 (`scripts/tmp/b70-nvfp4-dflash-to-bf16.py`) and **rename leftover `hf_quant_config.json`** or vLLM treats the draft as NVFP4 and fails.
- **n=3 is not a public median.** The first DFlash screen (214/185) ran after `vllm-dense-profile` respawned `b70dense`. Isolated n=5 p8192 decode is 157.9, not 185. Pin `Restart=no` for the whole load+matrix; `docker rm` is not enough.
- **Do not headline p512 194.6.** Family range 140–220. Representative scalar is p2048/g128 **186.6**.
- **“10k prefill” is not this campaign.** 10,371→10,349 is a no-spec n=3 TTFT-derived cold input rate on p8192/**g128**. Isolated DFlash Lane 1 input is **7160** at p8192/**g1**. See `research/nemotron-cold-input-vs-10k.md`.
- **`--max-num-batched-tokens=16384` is not a confirmed DFlash input lever (Run 37).** An isolated n=3 screen hit 9874 at p8192/g1 (0 cache hits). Fresh n=5 on the same stack/prompts was **6096** (−15% vs 7160) and p2048/g128 decode fell to 167 vs 186. Keep the published recipe at batched=8192. Do not headline 9874.
- **DFlash context capacity is 120000, not 128K (Run 38).** Isolated load at `--max-model-len 120000` left 5328 MiB free and logged **295,000** KV tokens. Staged C1 completions finished 32K / 64K / 96K / **p119872+g32 = 119904**. Speed card stays Run 36 (16K, 186.61 / 7160). Do not headline the g32 diagnostic rates. 128K was not requested after the operator chose 120000 as the limit.
- **HF canonical account is `SergiioB` (two i's):** `SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym` + `SergiioB/Nemotron-3.5-Lightning-30B-A3B-DFlash-BF16`.
- Next-model memory: `research/next-model-adaptation-playbook.md`. PR split: `research/vllm-xpu-pr-readiness-20260813.md`.

### 15.13 Qwen3.8-27B day-0 adoption (verified 2026-08-14, Run 39)
- **Qwen3.8 is structurally Qwen3.6** (`qwen3_5_text`, zero tensor-family deltas, same 1-layer MTP head) — every 3.6 recipe transferred day-0. Check the new generation's flagship config first; the family string answers whether day-0 is possible before any download.
- **GPTQ self-quantization works ON the B70** (gptqmodel 7.3.2, torch 2.13.0+xpu native, in a derived container from the pinned image; install gptqmodel `--no-deps` + real deps list, never let pip touch torch). Budget RAM+swap ≥ BF16 size (30 GB RAM + 24 GB models2-SSD swapfile + kill-watchdog at <4 GB headroom) AND source+swap+output+reserve per mount simultaneously — run1 OOM-rebooted the host, run2 died ENOSPC because the swapfile ate the output space.
- **MTP exclusion**: `QuantizeConfig(dynamic={"-:.*mtp.*": {}}, offload_to_disk=False)` keeps the draft head BF16, matching the 3.6 MTP-Preserved load path; the saved config's dynamic exclusion even satisfies the upstream BF16-draft condition without the env patch.
- **Hybrid-GDN vLLM serving on this stack**: XPU graphs ON + `--max-num-seqs 64` (Mamba cache caps decode seqs at 72 at U=0.90; default 256 blocks graph capture); async-scheduling and 16K batched budget both REGRESS dense qwen3_5; a week-newer nightly was also slower — the pinned `2c427ef` remains champion.
- **Spec-decode SSE counting**: MTP emits multiple tokens per stream chunk — count real tokens via `stream_options.include_usage`, never chunks, or you halve the rate.
- **Depth optimum is MTP4 for this head too** (MTP1 47.5 / MTP2 64.3 / MTP4 81.26 / MTP8 71.0 @p512) — single-layer draft acceptance decay beats depth. 95 t/s class needs a multi-layer MTP checkpoint.
- **W4A8 stays dead for sym-G128-with-g_idx artifacts** (fast path needs no-g_idx): no gain AND corrupted output day-0, matching the 3.6 quality kill.
- **Results (PROVISIONAL)**: llama.cpp Q4_K_M tg128 22.30 ±0.01 (230W n=5, submitted); vLLM MTP4 **81.26 t/s** decode / **1,731 t/s** cold input p8192 (n=5 median, 230W) = +17% over 3.6's 69.3. Full record: `research/qwen38-27b-day0-adoption-20260814.md`, Run 39.

### 15.14 Qwen3.8 champion stack: prefix-on agentic, 32K→128K ladder, Cn concurrency (verified 2026-08-19, Runs 43–45)
- **Champion stack for Qwen3.8-27B**: image `f01e24f6` (vLLM `0.27.2rc1.dev77+gac7509e2b`, kernels 0.1.12.3) + `patch_mtp_nightly.py` + `patch_mtp_boundary.py` + optional `patch_gdn_mixed_split_v5.py` + draft-INT4 S+M1 (`B70_DRAFT_LMHEAD_INT4=1,B70_DRAFT_MTP_INT4=1`). MTP4, fp8 KV, U=0.88, max-len 131072.
- **Draft-INT4 S+M1 vs BF16 draft (matched n=5, cache off)**: 112.65 vs 81.20 @p512/g128 (+38.7%); 103.63 vs 77.52 @p8192/g128. Quality temp-0 15-task A/B: 12/15 both, zero C-only regressions — speed keep, NOT token/KL parity. LMX rows: 83.7 (BF16) and 112.65 (overlay) both preserved.
- **Prefix-on agentic (Run 43–44)**: S+M1 wins short (58.86 vs 43.81), 8K (66.99 vs 48.04), 16K (65.92 vs 54.40), 32K (61.02 vs 46.34), 64K (56.20 vs 44.83); 96K thin (47.85 vs 40.66, n=2); **~128K agentic t1 n=1 S+M1 LOST (37.48 vs 43.82)** — never headline S+M1 through 128K. Keep synthetic cache-off 62.52 @p130944/g128 n=5 and this agentic single-obs in separate cells.
- **Concurrency NEEDS v5 and drops MTP acceptance (Run 45)**: unpatched `gdn_attention` dies on mixed spec+prefill at C≥2; with v5 + S+M1 + prefix ON: 0 crashes at every cell incl. mixed 1×p32768+4×p512. MTP acceptance falls from ~94% (C1) to 46–56% (Cn) — do NOT "fix" acceptance; throughput still scales.
- **Two aggregate definitions, never merged**: controlled long-prefill wall-aggregate (C32 160.5, per-stream ~28) vs LocalMaxxing short-prompt harness aggregate (C5 203.8, C32 224.2 — records `cmt00hzaf…`, `cmt00hz…`, `cmt00i05…`, `cmt00i0e…`). Cn aggregate never sits beside C1 rows.
- **Cn at 128K max-len runs at sub-1 GiB reserve (631 MiB after load)** — research-class: measured point, not a serving-capacity guarantee. Dedicated concurrent serving should lower `--max-model-len` and keep ≥3 GiB free.
- **lmx CLI renamed (2026-08-19)**: `lmx benchmark …` → `lmx speed-test …`; remote mode has `--concurrency`; auth is stored on the B70 (`lmx auth whoami`); submit works without printing any key.
- **PROMPT CONTENT DRIVES MTP ACCEPTANCE — never compare decode across prompt families (Run 46, 2026-08-19).** Same image/patches/server: calibrated real-world Pi prompts p512/g128 = **100.2 t/s at 89–96% acceptance** (cache ON; cache-off record 112.65 @ 94.4%) vs degenerate `INDEX` filler = **71.5 t/s at ~44%**. Filler-based harnesses understate ANY MTP stack ~30% here; INDEX-filler decode numbers are a lower bound, not a config regression. Prefill is unaffected (p8192/g1 = 1694 t/s both ways). Secondary effect: prefix caching ON costs ~5–11% decode at zero hits. Isolation evidence: `results/qwen38-pi-isolation-20260819T120424Z/`.
- **Realistic C5 (5 users, ~8K coding sessions, g512, prefix on): per-user 27.7 tok/s, Σ-streams 138.3, turn-completion wall 54** (60/60 OK, Run 46). THREE different metrics — always name which one. **Never compute a Cn decode aggregate from max(e2e)−max(ttft) windows**: staggered scheduling inflates it (reads 749 @C16 / 1786 @C32 on the same waves). Σ streams (n × per-stream median) and the wall aggregate are the honest pair. C32 Σ-streams = 903.
- **ALWAYS serve and benchmark Qwen3.8 at the MODEL CARD's recommended sampling, selected by THINKING LEVEL (2026-08-19 rule).** Levels map to presets: **off = temp 0.7 / top_p 0.80 / top_k 20 / presence_penalty 1.5 + `chat_template_kwargs {"enable_thinking": false, "preserve_thinking": true}`**; **low/medium/xhigh = temp 1.0 / top_p 0.95 / top_k 20 / presence_penalty 0 + `chat_template_kwargs {"enable_thinking": true, "preserve_thinking": true}` + `reasoning_effort: "<level>"`** (repetition_penalty 1.0; checkpoint `generation_config.json` ships the thinking preset → pass `--generation-config auto`). Never send `reasoning_effort` without `enable_thinking` plumbing — it crashes the template. Greedy (temp 0) is diagnostic-only. Every public number states its level+preset. Desktop launcher does the mapping: `~/Desktop/b70-launcher.sh [off|low|medium|xhigh]`. Harness constants: `scripts/tmp/qwen38-sampling-real.py` (`NONTHINK`/`THINK`).
- **Agents may execute directly ON the B70** — then NO SSH (self-SSH to <lab-host> fails "No route to host"); run docker/hwmon locally. See §1 note.

### 15.14 Qwen3.8 XPU GDN mixed spec+non-spec split (verified 2026-08-18)
- **Root cause is a host `TORCH_CHECK`, not a missing SYCL kernel.** Fused `torch.ops._xpu_C.gdn_attention` refuses `num_spec_decodes>0 && num_prefills+num_decodes>0`. Two exclusive launchers already exist; Xe2 non-spec intermediates are chunk-padded and spec are not; the fused wrapper consumes one 5-tensor group. CUDA already splits this in `QwenGDNLinearAttention._forward_core`. Tracker: `vllm-xpu-kernels#510`. Writeup: `research/vllm-xpu-mtp-agentic-crash-pr-20260816.md`.
- **Working overlay:** `scripts/tmp/vllm-xpu-causalconv-patch/patch_gdn_mixed_split_v5.py`. Homogeneous C1 stays one fused call. Mixed: compact `index_select` + two fused calls + `index_copy_` **both** `core_attn_out` and `z`. Idle optional side is `None` (`Tensor?`). v1–v4 failed the C++ size-sum / `narrow` contract (global indices into a compact buffer, or unused-side dummy tensors that still counted as tokens).
- **Patch every `_xpu_ops.py`.** `PYTHONPATH` empty is not proof of the import. First live probe patched `/workspace/vllm` only; EngineCore imported `/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py` and still died on the exclusive string. v5 `resolve_paths()` patches all copies.
- **Sequential agentic is not a mixed-step proof.** Autoresearch T1 then T2 never overlapped a live spec-decode with a new prefill. Concurrent probe is the proof: `prefill=1 spec=1 n_ns=1187 n_sp=5`, both requests completed, health 200, 48 log hits. Raw: `results/qwen38-gdn-mixed-split-v5-20260818/`.
- **This is infra, not a speed keep.** C1 n=3 with the overlay was 83.77 / 81.1 / agentic 38.76 / 1734.7 vs last keep 83.49 / 80.94 / 38.57 / 1720.1. Coding/agentic ~38 t/s is MTP acceptance, not kernel death. Do not headline the n=3 as a replacement for Run 40 n=5 **83.7**. No token/task-quality parity of split vs isolated groups; no 64K–128K gather-cost study.
- **Production pi-agent no longer needs `NOSPEC=1` for this crash class.** `scripts/tmp/qwen38-pi-serve.sh` now launches MTP4 + the v5 overlay. Keep `GPU_UTIL=0.88` at 131K (U=0.90 is the §8.1 abort zone). Autoresearch config-space is exhausted; last keep remains `8924799`.

### 15.15 `/mnt/<models-disk>` empty after boot is a partial-VG trap (verified 2026-08-18)
- `vg_data` is missing PV `IJ2FDM-…`. `lv_data` is partial/unavailable. systemd therefore skipped **all** VG auto-activation, including the intact `lv_models` on `/dev/sdb3`. `fstab` `nofail` then left `/mnt/<models-disk>` as an empty directory. That is not a second wipe.
- Activate only the intact LV: `sudo lvchange -ay vg_data/lv_models && sudo mount /mnt/<models-disk>`. Persisted as `vg-data-lv-models.service`. Do **not** activate `lv_data`.
- After remount (143.1G, 53.1G free): Qwen3.6-27B/35B GPTQ, Qwen3.8 nogidx GPTQ, Qwen3.8 BF16 shards, Nemotron DFlash-BF16, leftover DSpark 2.6G. **Still absent on B70 and Rock 5B:** ThinkingCap Q4/Q5/Q6 GGUF, Qwen3.6 UD GGUF, Laguna, Ornith Q5. `llama-profile` GGUF catalog cannot start until those files exist again. `active-profile=nemotron-vllm-dflash7-120k` is a vLLM path — do not feed it to llama-server.
