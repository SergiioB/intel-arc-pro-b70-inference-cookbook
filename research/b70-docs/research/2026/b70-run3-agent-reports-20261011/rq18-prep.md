---
title: "Run-3 subagent report: rq18-prep (2026-10-11)"
type: research
date: 2026-10-11
status: provisional
---

# RQ-18 preparation — two-card Strata `--layer-split` screen (Flash-Next IQ3_XXS)

Prepared 2026-10-11, resuming the files left by the killed 17:40Z incarnation.
**No GPU work, no docker daemon start, no registry edits.** The sealed RQ-17 battery
(`b70-rq17-att3`, GPU0) was left untouched; all reads below are read-only.

## 0. What this pass changed

The previous incarnation had already written the adapter, the campaign files and the
plan JSON. This pass added and verified:

1. the RQ-17 **host-memory guard** to the adapter (semantics reused verbatim);
2. a new `dryrun` subcommand (no-GPU, no-docker argument assembly);
3. **honest residency pricing** (per-stage dense/context/MTP overhead, not just the
   4 GiB reserve) and the corrected numbers recorded in the plan JSON;
4. re-ran the no-GPU checks: `py_compile`, `selftest`, `dryrun` — all pass.

### Files
| path | what |
|---|---|
| `scripts/tmp/rq18_layersplit_20261011.py` (76 KB, 1499 lines) | the RQ-18 screen adapter (`preflight`/`screen`/`cleanup`/`selftest`/`dryrun`) |
| `campaigns/2026/rq18-layer-split/canaries.json` | 6 precommitted exact-answer canaries, gen 512 greedy |
| `campaigns/2026/rq18-layer-split/plan-screen-20261011.json` | plan JSON, RQ-17 schema |
| `/mnt/<nvme-disk>/b70-rq18-20261011/<run>/` | big raw (42 G free; root disk has ~8 GiB) |

Plan JSON validated: parses; its top-level key set is **exactly** RQ-17's
(`results/2026/10/RQ-17-20261011T114846278867Z-71a59de867/plan.json`, keys compared in
Python, `keys equal: True`); `question_id RQ-18`, `stage screen`, `resource gpu`,
`comparison = {metric decode_tok_s_engine_side, unit tok/s, direction higher,
min_relative_gain 0.15}`; `hypothesis` and `changed_variable` verbatim from
`registry/research-program.json` RQ-18.

## 1. How `--layer-split` selects devices under `ONEAPI_DEVICE_SELECTOR`

All citations are `sycl/src/program/generate.cpp` in the pinned tree
`/mnt/<sata-disk>/strata/repo/strata-0.1.40.3`.

- **L1980** `n_dev = dpct::device_count()` — the visible device set is whatever the
  selector exposes. `--layer-split K|auto` requires `--serve` (die at **L2024**) and is
  rejected with one visible device: **L2011-2013** prints
  `--layer-split auto: one GPU visible, so no split` and clears the option.
- **With `auto` and no `--split-device`, the later stages are the next visible devices**:
  **L2003** `for (int d = 1; d < n_dev && ...; ++d) split_devs.push_back(d)`.
- **L2019** validation: each split device index must satisfy `0 < idx < n_dev`.
- **L2054** a split across GPUs needs `--expert-profile` (the config of record has it).
- **L2140** prints the placement: `layer split across %zu GPUs: CUDA0, then CUDA%d (split …)`.

So `ONEAPI_DEVICE_SELECTOR=level_zero:0,1` → `device_count()==2` → `split_devs=[1]` →
stage 0 = card0 (`0000:0b:00.0`, renderD128), stage 1 = card1 (`0000:0f:00.0`, renderD129).
`[INFERENCE]` `dpct::device_count()` counts the selector-filtered root devices; the
empirical proof is the L2140 line, which the adapter parses
(`RE_SPLIT_STAGES`, adapter L599). PID/hwmon indices are never remembered — both are
resolved live by PCI at preflight (adapter `pci_of_drm`/`hwmon_by_pci`).

## 2. The startup lines that prove 100 % expert residency

Real source shapes (spot-checked in the vendored tree; the adapter's regexes match them):

| claim | source | line |
|---|---|---|
| auto placement + held pairs | `generate.cpp:3625` — `layer split auto: K=<k> - predicted <x> ms per decode window; the caches hold <n> of <m> profiled pairs (~<p>% of the routed mass), best of <N> placements` | `RE_SPLIT_AUTO` |
| per-stage cache contents | `generate.cpp:4817` — `layer split: CUDA<d> runs layers <a>-<b>, expert cache <s> slots (<g> GiB), <f> of its <m> profiled pairs; slot 0 verified`; `:4823` adds CUDA0's range | `RE_SPLIT_STAGE` |
| **split residency %** | `generate.cpp:6403` — `strata serve: layer split: <x>% of the experts resident, the prompt path's streamed ring <r> slots` | `RE_SERVE_RESIDENT` |
| verify graph all-resident | `sycl/src/core/verify.cpp:709` — `strata verify: window up to <T> tokens, <M> MiB of device buffers (100% VRAM resident: zero-doorbell graph)` | `RE_VERIFY_RESIDENT` |
| per-request hits | `generate.cpp:11421` — `strata serve: decode expert cache hit rate: <x>% (<h> hits / <l> lookups)` | `RE_CACHE_HIT` |

Caveat worth knowing: **L6403 is conditional**. Its `host_res` is the *device residency
table* (built at `generate.cpp:5649`, one entry per (layer, expert), same on every
device), not the host mirror; the line only prints when `multi_gpu && !host_res.empty()`
and the derived ring is > 0, i.e. when **≥ 75 %** of pairs are resident (ring then
default to 96, `generate.cpp:6394-6403`). A 100 %-resident arm B always prints it; a
poorly-resident split would print nothing, so absence is not "pass" — the adapter's
`arm_residency_verdict` (adapter L706) requires, for arm B, **all** of: every stage
`filled == profiled` **and** every verify window all-resident **and** the serve line
`pct >= 100`. Arm A is not required to be fully resident (it holds ~57 %).

## 3. Per-card VRAM pricing (numbers)

Measured live, read-only, with the sealed battery running (the battery's GPU0 usage is
already inside card0's numbers):

```
/sys/kernel/debug/dri/0/tile0/vram_mm   visible_avail: 30748MiB   visible_size: 32656MiB
/sys/kernel/debug/dri/1/tile0/vram_mm   visible_avail: 32615MiB   visible_size: 32656MiB
```
card0 is 1908 MiB below card1's 32656 MiB ceiling → the desktop/compositor. This matches
the task's "GPU0 ~30683 MiB, GPU1 ~32615 MiB idle" envelope (my preflight re-reads it).

Inputs:
- **Experts**: 24,576 IQ3_XXS pairs = **42,912,972,800 B = 39.966 GiB**, 48 layers,
  per-layer 637.5–1137.5 MiB (`pack/iq3xxs/native_experts.txt` header + rows; adapter
  `read_layer_expert_bytes`, which fails closed if the rows do not sum to the declared total).
- **Reserve**: 4 GiB per card (AGENTS §4 living-free rule; `CARD_RESERVE_MIB`).
- **Dense** per stage: `dense.bin` = 1,538,035,200 B = **1466 MiB** (RQ-14 identity table,
  `results/2026/10/rq14-strata-matched-screen-20261009/RESULTS.md` §2). The config of
  record has **no `--trim-stage-weights`**, so every stage loads the *whole* model's dense
  weights, not just its own layers (`generate.cpp:589-592`) — hence a full copy per card.
- **Context/kernels** ≈ 195 MiB (derived: RQ-14 raw `arm-*/…` — “CUDA1 context ready,
  31.66 GiB free” ⇒ 32615 − 32420 MiB).
- **MTP drafter** 993 MiB on the last stage only (RQ-14 raw: `mtp … 993 MiB of VRAM`).

| stage | budget = visible_avail − 4096 − dense − 195 (− drafter) | expert capacity |
|---|---|---|
| card0 (layers `[0,K)`) | 30748 − 4096 − 1466 − 195 | **24.40 GiB** |
| card1 (layers `[K,48)`, last stage) | 32615 − 4096 − 1466 − 195 − 993 | **25.26 GiB** |
| total | | **49.66 GiB** vs **39.97 GiB** of experts → **9.7 GiB slack** |

Because the split is contiguous layer ranges, feasibility is a window on K
(card1 binds at low K, card0 at high K):

| K | card0 experts | card1 experts | fits |
|---:|---:|---:|:--:|
| 20 | 14.33 GiB | 25.63 GiB | no (card1) |
| 22 | 15.94 GiB | 24.02 GiB | yes |
| 24 | 17.72 GiB | 22.24 GiB | yes |
| 26 | 19.29 GiB | 20.68 GiB | yes |
| 28 | 20.63 GiB | 19.34 GiB | yes |
| 30 | 22.68 GiB | 17.29 GiB | yes |
| 32 | 24.63 GiB | 15.33 GiB | no (card0) |

**Verdict: full residency fits — feasible K window 21..31** (adapter
`residency_pricing()` now applies this overhead model too, `first_fit_K` = 21 on the
preflight's live numbers; with the optimistic experts-only budget it would have printed
18, so this pass tightened the gate). `--layer-split auto` picks the K minimising its
predicted decode window (`generate.cpp:3625`); it maximises held routed mass, so with
everything fitting it should land inside 21..31. **Residency is not the blocker.**

## 4. Exact two-card argv / env

Built and printed by `python3 scripts/tmp/rq18_layersplit_20261011.py dryrun`
(assembled from `serve_launch_argv`, adapter L958; no docker, no GPU):

**docker argv (identical for both arms except the config mount file and container name):**
```
docker -H unix:///run/docker.sock run -d --name rq18-layersplit-<a|b>-<ts>-<hex> \
  --device=/dev/dri --group-add=<renderD128 gid> --network=host \
  -v /mnt/<sata-disk>/strata:/work -v <B70_RUN_DIR>/configs:/rq18cfg:ro \
  -w /work/repo/strata-0.1.40.3 -e PYTHONPATH=/work/pylibs \
  --entrypoint python3 strata-sycl-dev:latest \
  -m serve.server --engine strata --config /rq18cfg/arm-<a|b>.json --port 8095 --host 127.0.0.1
```
(`/dev/dri` is required: this is the GPU engine, not a compile. It only runs after the
lead's GPU GO.)

**Engine args — arm A = the config of record verbatim**, file changed only for the log path:
```
--pack /work/pack/iq3xxs
--native  /work/models-iq3xxs/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
--ple-gguf /work/models-iq3xxs/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
--expert-profile data/expert-profile.bin --expert-cache auto --stream-experts
--mtp /work/mtp-rt --spec 4 --spec-min-p 0.5 --vram-reserve-mib 256
--ple-io ram --max-context 10240 --kv int8 --prefill 4096
```
**arm B = exactly the above + `--layer-split auto`** appended, and nothing else
(the adapter *proves* this: `arm_a_vs_record_keys == ["log"]`,
`arm_b_vs_arm_a_keys == ["args","env","gpu","gpu_order","layer_split","log"]`).

**config env:**
- arm A: `ONEAPI_DEVICE_SELECTOR=level_zero:0`, `STRATA_VERIFY_NO_HOST=1`,
  `STRATA_VERIFY_DEVICE_PLAN=1`, `STRATA_STAGER_THREADS=12`, `STRATA_PREFILL_RING=512`
- arm B: identical **except** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1`

Config keys arm B adds: `"gpu": [0,1]`, `"gpu_order": "as_given"`,
`"layer_split": "auto"` (the split-device default places card1 as the second stage).

**Cells (per visit, greedy, one request in flight):** one discarded same-shape warmup
(p512/g128), then `dec` = p512/g128, `pre` = p8192/gen 8, then the 6 `batt-*` canaries
gen 512 (content-only scored; `reasoning_content` kept as evidence). 3 interleaved
**ABBA** pairs = 6 fresh engine visits (`PAIR_ORDER = [(A,B),(B,A),(A,B)]`). Guard
before and after every cell, on BOTH render nodes. Caps, temps and per-card
`visible_avail` recorded at every visit and per cell; as-found caps restored at cleanup.

## 5. BLOCKER — the byte-identical gate is structurally unsatisfiable here

This is the one finding that should change the program before the GPU window is spent.

Facts:
- The RQ-18 gate/stop rule (registry `success_gate`, `stop_rule`) requires the greedy
  battery outputs to be **byte-identical between arm A and arm B**, and rejects on *any*
  output difference.
- Arm B is all-GPU: every one of the 24,576 pairs is resident, so every routed expert is
  computed by a GPU kernel.
- Arm A holds only **14,024 of 24,576** pairs (`pre-filled 14024 of 14024 slots`, RQ-14
  raw) → **10,552 misses** (`10552 of 10552 experts missing from VRAM mirrored in pinned
  host memory`, RQ-14 raw). Misses are split between the GPU (reading the pinned mirror
  over PCIe) and the **CPU expert pool** by `--pcie-frac` (`generate.cpp:4686-4689`).
- On **this host** the probed share is **0.35–0.38**, not the 0.55 default: RQ-14 raw
  `PCIe probe: 13.8 GB/s host->device (best of …) -> pcie_frac 0.38 (default 0.55)`.
  So ~62–65 % of arm A's ~10,552 misses (≈ 6.5–6.8k pairs) are **CPU-computed**.
- The engine's own harness states the two paths are not bit-equal:
  `native_expert_parity.cpp:1-6` — “(b) the CPU path (ggml-cpu vec_dot with its quantized
  activations), (c) the GPU path (`native_expert_grouped`, q8_1 activations) … differ from
  (a) by their activation rounding only (a few 1e-3 relative)”; the test only bounds
  `rel(got_c, got_g)` (`:291-296`), never asserts equality.
- The repo's own measurement of the consequence: CPU-path vs GPU-path flips the top-1
  token at **2–5 % of positions** over 512–1024 tokens
  (`bench/results/2026-09-27-cache-parity/README.md`).
- The vendor's cross-GPU note is exactly this case: “Across GPUs the text is not
  byte-identical to one GPU: experts the CPU computed on one card run on a GPU in the
  split, and they round differently” (`bench/results/2026-09-29-layer-split/README.md`).
  (A split on the *same* GPU is 10/10 byte-identical there — the hand-off itself is exact;
  the divergence is the CPU-vs-GPU expert arithmetic.)

`[INFERENCE]` Arm A and arm B therefore diverge somewhere in a 512-token greedy
generation (order tens of flips), and RQ-18's stop rule fires **regardless of the decode
speed**. The +15 % decode gate may be measurable and may even pass, but the screen as
predeclared **cannot return `success`**. Note the sub-case: a pass would require arm A to
route every generated token through residents, impossible with 43 % of pairs non-resident.

Options for the lead (a program decision, not taken here):
1. keep the screen but **predeclare the byte-identity clause as expected-to-fail**, and
   report the speed deltas plus the divergence count/positions as the finding; or
2. replace “byte-identical across arms” with the repo's own equivalence method —
   teacher-forced `--dump-logits`, median KL and same-top-1 fraction, against the
   cache-parity bound (median KL 3e-4…4.5e-3, same top-1 95–98 %) — the correct bar for
   two placements differing only in expert arithmetic; or
3. restrict “byte-identical” to **within-arm** reproducibility (arm A vs arm A across the
   3 pairs), which the adapter already computes separately (`a_stable`) and which should
   pass, and demote the cross-arm clause.

Also note the plan premise “the miss term (about a third of a ~56 ms round)” is an
estimate from the lever audit (`research/references/rq18-layer-split-price.json`), not a
B70 counter; the deciding probe is still arm A's per-cell expert-miss counters.

## 6. Host-memory guard (RQ-17 semantics) — done

`host_memory()` (adapter L395-409) and the constants `MEM_AVAILABLE_FLOOR_MIB = 8192`,
`MEM_PSI_FULL_AVG10_MAX = 5.0` (L138-139) are copied from
`scripts/tmp/rq17_battery_20261010.py:418-431` / `:121-122`. `run_gpu_guard`
(adapter L412) now records `host_memory` on **every** guard call (preflight, every
visit start, before **and** after every cell) and appends `-1` to
`foreign_inference_pids` when `MemAvailable < 8192 MiB` **or** memory-PSI `full avg10
> 5.0`, so a pressured cell becomes `Contaminated` (exit 3, evidence retained) exactly as
in RQ-17. Live right now, for reference: `MemAvailable 52189 MiB`, memory PSI
`some avg10=0.54 avg60=1.87 avg300=1.24` (`/proc/meminfo`, `/proc/pressure/memory`).

## 7. Verification performed this pass (no GPU, no docker daemon)

```
$ python3 -m py_compile scripts/tmp/rq18_layersplit_20261011.py      -> exit 0  ("PY_COMPILE OK")
$ python3 scripts/tmp/rq18_layersplit_20261011.py selftest           -> exit 0  ("selftest OK (no GPU touched)")
    problems: [], arm A diff = ["log"], arm B diff = ["args","env","gpu","gpu_order","layer_split","log"],
    total_expert_gib 39.966, n_layers 48, prompt counts p512=512 p8192=8192 batt-arith=22,
    first_fit_K 21, feasible true, parse_ok true
$ python3 scripts/tmp/rq18_layersplit_20261011.py dryrun             -> exit 0  (both arms' argv/env, §4)
$ python3 -c '… json.load …'                                         -> plan.json + canaries.json parse; keys == RQ-17 schema
```
The selftest additionally exercises both engine-log accounting shapes (the RQ-17/RL-152
serve per-request line and the RQ-14 one-shot `decode N tokens in X ms`), the residency
parser (`split_auto`, per-stage pairs, verify windows, serve %, cache hit), the config
diff, and the prompt inventory from the real pack tokenizer.

## 8. What the lead must run after `CPU GO`

The remaining work needs the **GPU, the docker daemon and `--device=/dev/dri`** (plus, for
the screen, the engine's ~17 GiB pinned host mirror, RQ-14 raw), i.e. more than the
current caps and contradictory with the running sealed battery. Exactly:

```bash
# 0) sanity, no GPU, ~0.3 s CPU, <200 MiB RSS:
python3 scripts/tmp/rq18_layersplit_20261011.py selftest
python3 scripts/tmp/rq18_layersplit_20261011.py dryrun

# 1) preflight (zero-GPU; needs sudo for debugfs + the docker socket for `docker ps`):
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 -p CPUQuota=200% \
  env B70_RUN_DIR=<run-root> OMP_NUM_THREADS=2 \
  python3 scripts/tmp/rq18_layersplit_20261011.py preflight

# 2) screen (GPU GO only; the adapter itself is light, but the engine CONTAINER
#    pins the host mirror, so it must NOT be under a 3G cap — run it uncapped):
env B70_RUN_DIR=<run-root> \
  python3 scripts/tmp/rq18_layersplit_20261011.py screen
# 3) cleanup:
env B70_RUN_DIR=<run-root> python3 scripts/tmp/rq18_layersplit_20261011.py cleanup
```

Blocking conditions to clear first: the sealed `b70-rq17-att3` battery finishing on GPU0,
and a decision on §5 (byte-identity clause) — otherwise the screen will run to completion
and then reject RQ-18 on output divergence alone.

`[UNVERIFIED]`: nothing in this document is unverified except the two items explicitly
marked `[INFERENCE]` (dpct::device_count selector semantics; the predicted divergence),
and the plan's own “≈ a third of a 56 ms round” estimate, which is quoted as the program's
assumption and not as a B70 measurement.
