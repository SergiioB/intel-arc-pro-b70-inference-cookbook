---
title: "Strata v0.1.40.3 source and expert-transfer math — RQ-14 screen pre-commit (2026-10-09)"
type: research
date: 2026-10-09
status: provisional
---

# Strata v0.1.40.3 source and expert-transfer math — RQ-14 screen pre-commit

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Sections 1-6 are source capture and
arithmetic only (no GPU work in that phase). The host is online; live state and
gate results are in section 7. External evidence class E0 (their numbers, their
hosts); all local rates below are bounds or assumptions, not measurements.
Follows RL-121.

## 1. Sources captured (research/references/strata-intel-20261008/)

| source | revision | sha256 (prefix) | fetched |
|---|---|---|---|
| release v0.1.40.3 page | d5ea7133741e67743c0e886bb426c0ce8d69cf6c | 390855ea | 2026-10-09T14:19Z |
| docs/INTEL.md (pinned raw) | same | 020c512a | 2026-10-08T21:24Z |

v0.1.40.3 corrects Intel setup to `./setup.sh --backend sycl`. B70 build and
numbers unchanged versus v0.1.40.2 (RL-121). No new B70 speed evidence.

## 2. SYCL/quant dispatch (pinned INTEL.md)

1. Engine: Strata's own engine ported to SYCL (`sycl/`, 50 CUDA kernels), built
   in Docker `strata-sycl-dev` (oneAPI 2026.1, oneMKL, Level Zero). B70 AOT
   build `bmg-g31` (`build-sycl-aot/strata`), no first-start compile.
2. Setup: `./setup.sh --backend sycl` wraps upstream `setup.py` via
   `sycl/setup_intel.py` (Intel steps swapped in; model, pack, tokenizer, MTP
   draft layer and KV questions are upstream's). Arc found via sysfs 8086 under
   `xe`/`i915`, offered through the AMD path (local build, no images).
3. Quants: same GGUFs plus a native pack. IQ3_S is the Flash-Next quant with a
   B70 `--stream-experts` row (24,576 experts x 1.97 MB = 46 GB; 11.5k
   resident in a 22 GiB cache; the rest in one pinned host mirror over PCIe).
   IQ3_XXS has only the A750 non-streaming row (39.97 GiB RAM arena) — no B70
   streaming evidence.
4. GEMM path: oneMKL FP16 GEMMs on XMX. Per expert the dequant (42 us gate/up)
   costs twice the GEMM (20 us); the fused dequant+XMX kernel is 0.22x of the
   pair on their card but the dequant kernels remain the top speed reserve
   (~200 GB/s effective vs a 600 GB/s card). llama.cpp PR 29864 int8 DPAS
   kernels target K-quant/Q8_0, not IQ experts; not ported.
5. Sampling FP64: native on B70; the native-pack double path is sampler-only,
   tail now accumulates in float (`strata::samp_acc_t`;
   `-DSTRATA_SYCL_SAMPLER_FP64=1` keeps double).
6. Runtime flags in their B70 Flash-Next row: `--spec 4 --spec-min-p 0.5 --mtp`,
   INT8 KV, 32K context, `--stream-experts --vram-reserve-mib 1024 --ple-io ram`,
   `--prefill 4096` (auto-written for >=24 GB cards; `--prefill 8192
   --prefill-borrow` keeps all cache slots for decode at ~1 s refill per prompt).
   `SYCL_CACHE_PERSISTENT=0` is mandatory on Xe2 (persistent JIT cache
   segfaults at first compile). `STRATA_VERIFY_NO_HOST=1` is required whenever
   any expert is outside VRAM (without it the per-layer host/GPU handshake is
   not visible across the xe bus and logits go NaN).

## 3. Allocation and correctness safeguards

1. **Guarded allocator:** every allocation >=32 MiB goes through
   `strata::malloc_device_guarded`: tags every 64 KiB, reads tags back in a
   second kernel, re-allocates behind a growing spacer until clean (warning per
   retry, <0.5 s at startup). `STRATA_ARENA_ALIAS_CHECK=0` skips it — do not.
   This guards a reproduced xe bug: the 22 GiB arena came back with two 2 MiB
   pages aliased (write at +2 MiB visible at +1022 MiB) -> cached experts held
   other experts' bytes -> NaN at first routed use. Standalone probes
   (`sycl/probe/arena_rw.cpp`, `alias_scan.cpp`) do not reproduce it; it needs
   the engine's allocation history.
2. **Slot verification:** `STRATA_VERIFY_ALL_SLOTS=1` reads every filled cache
   slot back and compares with the GGUF (`STRATA_VERIFY_FIND=1`,
   `STRATA_VERIFY_DUMP=dir`). Run at screen start.
3. **Oversubscription livelock:** a GPU allocation past VRAM does not fail on
   xe — buffers evict to host RAM, the kernel runs out of memory, the machine
   livelocks until the hardware watchdog resets (reproduced by a 262K-context
   request, 2026-09-29). Rule: compute KV first, leave 1.5 GB free; never ask
   for more context than the table allows.

## 4. Expert-transfer pricing (research/references/strata-*-price.json)

Inputs and provenance in the two JSON files; `tools/research.py price` output
classification `theoretical_inferred_bounds_not_measured_gains`. Local class
bandwidth 11.5 GB/s (assumption, retained direct-peer copy probe 11-12 GB/s;
local H2D unmeasured; their B70 VM measured 13.3 GB/s H2D, E0).

| cell | bytes | bound at 11.5 GB/s | at 13.3 (their VM) |
|---|---:|---:|---:|
| prefill 4K-chunk miss stream (13.5k x 1.97 MB) | 26.6 GB | 2,313 ms/chunk | 2,000 ms/chunk |
| decode round, sensitivity 135 misses | 266 MB | 23.1 ms/round | 20.0 ms/round |

- **Prefill gate has floor margin.** The 8K matched cell streams two such
  chunks: <=4.63 s of PCIe -> <=1,766 tok/s bound vs the 3x gate (555-585
  tok/s): >=3.0x margin even before compute overlap. Their 8K row (953-1,117
  tok/s) sits inside this bound. PCIe is not what decides the prefill gate.
- **Decode gate is content/MTP-sensitive, not transfer-floor-blocked.** The
  per-round miss count and MoE-layer count are unrecorded in the captures; the
  screen must collect cache-miss counters per cell. At the sensitivity point,
  23 ms/round is a first-order share of their observed 65-89 ms/round.

Storage/RAM/PCIe before acquisition (live 2026-10-07 numbers, re-check at
host):

- Storage: model ~70 GB (RL-121) -> 2x rule needs >=160 GB free. T-FORCE
  (SATA) 281 GB free passes; SN770 NVMe 74 GB free fails. Model must live on
  T-FORCE; the page-cache trick is weaker on SATA.
- RAM: the pinned host mirror is (24,576 - 11.5k) x 1.97 MB ~= 26 GB pinned,
  plus the `--ple-io ram` PLE table (size unrecorded for IQ3_S) and engine
  headroom. Live RAM gate must include both; do not lock the container
  (`memlock=-1` caused swapping on the A750 box).
- VRAM fit (their measured working config): 22 GiB cache + INT8 KV at ~20
  KiB/token (0.65 GB at 32K) + 1,024 MiB reserve + workspaces ~= 24-25 GB of
  32 GB. Headroom for the screen exists; re-derive before any larger context.

## 5. Gate arithmetic vs our baseline (RL-070)

Baseline: 21.6 tok/s median decode (p512/g128, engine tg), 185-195 tok/s cold
input, two B70s. Success gate: >=1.5x decode (>=32.4 tok/s) AND >=3x prefill
(>=555-585 tok/s) on matched cells, one card, battery passing.

Their B70 Flash-Next row (E0, greedy 256-token runs, MTP spec 4, story/code
classes, medians of 3-5 pairs): decode 30.5-31.4 (story, 1.41-1.45x our
baseline) and 40.8 (code, 1.89x). **The decode gate is at risk: their prose
class sits below 1.5x, only their code class clears it.** Their metric class
(engine greedy + MTP, their prompt content) is not ours (engine tg, no spec);
matched cells decide. Prefill external class (953-1,117 tok/s at 8K) clears
3x with margin.

## 6. Decision (source/math stage)

1. **Pinned for the screen:** Strata v0.1.40.3 at d5ea7133741e67743c0e886bb426c0ce8d69cf6c,
   artifact **IQ3_XXS** (pack `iq3xxs` + `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS`
   native GGUFs), `--stream-experts` on one B70. This resolves the plan's
   "IQ3_S or IQ3_XXS" choice; the earlier source-only IQ3_S preference (the
   only quant with a captured B70 streaming row) is void.
2. **Arithmetic did not reject the route.** The prefill gate had >=3.0x floor
   margin; the decode gate was falsifiable on content class, not blocked by
   PCIe arithmetic. Changed variable is engine + quant class + MTP (declared;
   route adoption may change engine per the benchmark protocol).
3. **Falsifying probe (run 2026-10-09):** Lane-1 matched cells on one B70 —
   p512/g128 decode + 8K prefill + exact-answer battery at our sampling and
   metric definitions. Gates and stop rules as registered; never lowered after
   results. Outcome in section 7: both gates clear on the fresh-pairs medians,
   battery 3/3.
4. **Live gates (done 2026-10-09, section 7).** Open items before any new
   acquisition or longer-context cells: root disk 8.3 GB free (97%); restore
   as-found caps 180 W / 230 W at cleanup; local H2D PCIe still unmeasured
   (the 11.5 GB/s class assumption stands until probed); the 256 MiB VRAM
   reserve is below INTEL.md's 1,024 MiB recommendation for <=32K configs.
5. **Carry-over safety items for our own harnesses** (RL-121 item 1, refined):
   the guarded-allocator pattern (>=32 MiB tag-and-readback) and
   `SYCL_CACHE_PERSISTENT=0` apply to any future work on this card family.

## 7. Live state and matched-screen outcome (2026-10-09, host online)

**Supersedes section 6 item 1's artifact choice.** The deployed and screened
artifact is **IQ3_XXS** (pack `/work/pack/iq3xxs` + native
`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` GGUFs), not IQ3_S. The source-only
preference for IQ3_S (the only quant with a captured B70 streaming row) is
overridden by the actual B70 IQ3_XXS `--stream-experts` deployment. Section 6
item 4's "before any download" framing is likewise moot: the artifacts are on
the T-FORCE volume.

Deployment (as found): Strata v0.1.40.3 tree at
`/mnt/<sata-disk>/strata`, dedicated dockerd
(`/run/b70-strata-docker.sock`, data-root on T-FORCE), engine reports 0.1.41;
serve config `serve-flashnext-iq3xxs.json`: `level_zero:0`,
`--stream-experts --expert-cache auto` (14,544 resident slots = 23.60 GiB
profile fill; 10,032 experts mirrored in 16.37 GiB pinned host RAM), MTP
`--spec 4 --spec-min-p 0.5`, `--kv int8`, `--prefill 4096`,
`--max-context 10240`, `--vram-reserve-mib 256`, port 8095,
`STRATA_VERIFY_NO_HOST=1`, `STRATA_VERIFY_DEVICE_PLAN=1`.

Live gates (as found ~2026-10-09T14:30Z):

- GPUs: 0b:00.0 (card0) and 0f:00.0 (card1), both Battlemage G31
  [8086:e223]; renderD128/129. Card0 VRAM usage 33,961,795,584 of
  34,242,297,856 B (visible_avail 267 MiB) = the engine's declared loaded
  steady state (its log: "267 MiB of VRAM free with everything loaded") with
  the 256 MiB reserve; card1 32,429 MiB free. Note: the ops doc's
  "below 500 MiB free = do not issue a request" table is written for
  llama.cpp-class full-offload loads; this streaming engine fills VRAM minus
  its declared reserve by design. 256 MiB reserve is below INTEL.md's
  recommended 1,024 MiB for <=32K configs — keep it in mind for longer cells.
- RAM: 60 GB total, 36 GB available, engine RSS ~29 GB (mirror + PLE in RAM).
- Storage: root 8.3 GB free (97% — tight), T-FORCE 128 GB, SN770 43 GB,
  sda3 WD 38 GB.
- Power caps as found: card0 180 W, card1 230 W (restore these at cleanup;
  the 01:56Z cleanup log's "restored 150W" is stale history).
- Temps ~52-54 C both cards at discovery. Isolation: no llama/vllm/sglang
  processes; display stack (gnome-shell/Xwayland/mutter) holds card0 and
  renderD128; the engine holds renderD128+renderD129; ODS/litellm containers
  are non-GPU services.

**Matched screen + fresh-pairs confirmation (raw, 2026-10-09 early hours,
`/mnt/<sata-disk>/strata/results/`):** arm-a = one card
(`level_zero:0`), arm-b = two cards (`--expert-cache-device1 auto`); confirm =
arm-a config, n=5 fresh decode/prefill pairs (pair 1 cold-failed with exit 2
and was replaced by dec-x/pre-x) + battery at gen=64, greedy.

| cell | arm-a (1 card) | arm-b (2 cards) | confirm n=5 median (1 card) | RL-070 anchor | ratio |
|---|---:|---:|---:|---:|---:|
| p512/g128 decode tok/s | 48.12 | 33.44 | 48.03 (47.77-48.12) | 21.6 | **2.22x** |
| 8K prefill tok/s | 1008.97 | n/a (p8192 log 43K, not extracted here) | 989.11 (984.95-990.22) | 185-195 cold input | **5.1-5.3x** |

Gate verdict: **both declared gates clear on the confirm medians** — decode
2.22x >= 1.5x; prefill 5.1x >= 3x — one card instead of two. Battery (exact
answers, greedy, confirm gen=64): **3/3 correct** — 37*19=703, capital of
France=Paris, exact echo B70-FLASH-OK (detokenized from retained output ids).
The two-card arm-b is slower (CPU expert pool on the second cache path: 0 by
DMA, host-staged) — one card is the right placement on this engine.

Caveats carried forward: the baseline is llama.cpp dual-split engine tg
without speculation; the Strata arm runs MTP spec 4 (engine + quant class +
speculation = the declared changed variable). Metric boundary is the engine's
own decode/prefill wall accounting. Evidence is raw and unsealed outside the
repo results tree (`strata/results/arm-a`, `arm-b`, `confirm`, `prefill-2k`);
registering it requires consolidation into a sealed result root (AGENTS.md
§12) before any route decision. Per §12 only a separate reviewed route-change
record can make this an adopted route.

**Update (2026-10-09, later):** the operator approved the route change.
ADR-0011 (`decisions/adr-0011-flashnext-strata-one-card-route.md`) makes
Strata IQ3_XXS one-card the adopted Flash-Next route;
`registry/model-status.json` is updated and RQ-14 is closed as passed. RL-125
stays `provisional` / `not_adopted` in its observation scope.

Operational findings from the runs:

- **Cold first request fails.** On an engine start with a cold cache the first
  request streams every expert host-staged (0 by DMA, resident 0) and exits 2
  with `--spec needs the device residency table` (dec-1/pre-1 in the confirm
  run). Discard one same-shape warmup request before any measured pair —
  matches the one-discarded-warmup rule.
- **Battery needs gen >= 64.** The thinking model spends early output tokens in
  the think block; at gen=24 the answers were cut (the arm-a battery rows are
  speed-only). The confirm battery at gen=64 emitted answers.
- **Engine naming:** the v0.1.40.3 checkout logs "engine 0.1.41". Cite both.
- The serve config also sets `STRATA_PREFILL_RING=512`. Battery prompts are
  token-id files (`strata/prompts/batt-*.txt`); answers were verified by
  detokenizing retained output-id lines with the pack tokenizer
  (`pack/iq3xxs/tokenizer/`).

Prefill tuning sweep (`strata/results/prefill-2k/`, 8K cell, one launch per
config): the confirmed `--prefill 4096` + `--kv int8` recipe is the best class
(clean 1,008.05 tok/s; res256-mc10k — the serve config's reserve/context —
1,022.69; q4_0 KV 1,012.26 tested but inadmissible as a default, fleet KV
policy is q8_0 K / q4_1 V). Worse: `--prefill-borrow` 8192 (607.87, refill
cost), auto/16384 chunking (747-751, 4 chunks), XMX prompt attention (753.77 /
719.22 — a regression), k8v4 KV (881.36). Base at 180 W vs 230 W (889.48 /
889.77) shows prefill is not cap-sensitive. No adoption beyond the confirmed
recipe; the sweep is tuning evidence only.
