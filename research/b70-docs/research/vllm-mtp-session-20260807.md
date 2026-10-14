# vLLM MTP Speculative Decoding — Full Session Log 2026-08-07

Host: B70 32GB · Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4 · intel/vllm:0.21.0-xpu-int4moe
Scope: MTP concurrency unlock, MTP2/4 spec-token sweep, prompt×output grid,
power/thermal telemetry. This document is the authoritative session record.

---

## 1. Session Goal

The Reddit reference (Dolboyob77) claimed `Concurrent-1` = 145.5 t/s decode /
7,975 t/s prefill on this MoE. The campaign had unlocked MTP (single-stream
~123-133 t/s) but MTP + concurrency (C2+) crashed the EngineCore. This session
(1) investigated the MTP concurrency blocker, (2) discovered MTP2/MTP4 are
supported and faster, (3) mapped the full prompt×output surface with prefill +
decode + power, and (4) identified prefill as the next optimization target.

---

## 2. MTP Concurrency Investigation (A19)

**Problem (Run 23):** MTP + C2+ crashed with:
```
RuntimeError: causal_conv1d does not support spec-decode and non-spec
(prefill + decode) tokens in the same invocation
```

**Root cause:** the XPU GDN kernel `gdn::causal_conv1d` (compiled into
`libgdn_attn_kernels_xe_2.so`) cannot mix spec + non-spec tokens in one call.
With C1 every batch is homogeneous; with C2+ vLLM batches a mix.

**Fix attempted:** `scripts/tmp/vllm-xpu-causalconv-patch/patch_gdn_mixed_split.py`
— split the fused `torch.ops._xpu_C.gdn_attention` call into two (non-spec +
spec) at the `_gdn_attention_core_xpu_impl` wrapper.

**Three layered compiled-kernel blockers discovered:**
1. `gdn::causal_conv1d` can't mix spec+non-spec (kernel).
2. `gdn_attention` requires token_indx sets to partition the FULL batch
   (kernel self-check).
3. Even with buffer compaction (index_select → per-group fused call →
   index_copy scatter), the compiled fusions around the op index state against
   the FULL batch → `index out of bounds` in a compiled inductor kernel.

**Result (verified live, clean guardrailed sweep @150W):**
```
   C | wall-agg t/s | ok
   1 |         88.5 | 1/1
   2 |         27.7 | 2/2   ← UNLOCKED (A19 originally recorded "C2 died")
   4 |         0.0  | 0/4   ← compiled-kernel OOB
   8 |         0.0  | 0/8   ← compiled-kernel OOB
```
The split patch **successfully unlocks MTP + C2** (real improvement), but C4+
still dies. Full fix needs vllm-xpu-kernels source + SYCL rebuild (genuine
kernel capability gap, not a removable assert).

**Sandbox test:** `test_gdn_mixed_split.py` (15/15 checks, then 15/15 with the
num_actual_tokens fix) validates control flow + arg composition without a GPU.

**Files:** `scripts/tmp/vllm-xpu-causalconv-patch/` (patch + test + mock);
`scripts/tmp/launch-mtp-mixedsplit-guardrail.sh`; `scripts/tmp/b70-mtp-mixedsplit-sweep.py`.
Full detail: `research/vllm-021-campaign-20260806.md` §A19.

---

## 3. MTP2 / MTP4 Unlock (A20) — the big finding

**Correction to A15/A18:** the MTP draft head is **recurrent** — it selects the
layer by `spec_step_idx % self.num_mtp_layers` (qwen3_5_mtp.py:146). With
`mtp_num_hidden_layers: 1`, `% 1 == 0` always, so the SAME single layer is
reused to emit N draft tokens per step. The A15 "num_speculative_tokens=2
clamped to 1" note was WRONG — that launch merely hardcoded `:1`. **No code
path clamps `num_speculative_tokens`** to layer count.

vLLM warns: "Enabling num_speculative_tokens > 1 will run multiple times of
forward on same MTP layer, which may result in lower acceptance rate" — but
acceptance holds on short/medium generates.

**Launch:** `launch-mtp-bench.sh N` (native INT4 v4 + BF16 MTP draft +
`--speculative-config {"method":"mtp","num_speculative_tokens":N}`).
**Load time grows with N:** MTP2 ~240s, MTP4 ~180s; MTP-N>1 needs
`MTP_LOAD_MAX` >900s (first MTP2 attempt timed-out at 900s while still
compiling).

**Single-stream decode (230W-equivalent harness, 165W actual):**

| N | tg32 | pp2048 | tg128 |
|---|-----:|-------:|------:|
| 1 | 125.8 | 112.1 | 114.4 |
| 2 | 142.4 | 133.4 | 133.0 |
| 4 | **172.1** | 124.4 | **149.4** |

**MTP4 tg32 = 172.1 t/s — EXCEEDS the Reddit 145.5 target (+18%).**

---

## 4. Full Grid Sweep with Power Telemetry (A21)

**Method:** `launch-mtp-bench.sh N` with `B70_NO_PC=1` (prefix caching OFF,
verified `enable_prefix_caching': False`) + `B70_GRID=1` grid harness
(`b70-mtp-grid-bench.py`: 6 prompt lengths × 3 output lengths × 3 reps, unique
suffix per rep to defeat caching) + `monitor-power.sh` (card/pkg watts via
hwmon energy-delta, pkg/vram/mctrl temps). **Power cap 165W explicit.** Single-stream.

**Decode t/s (steady-state reps 2-3):**

| prompt/out | MTP1 | MTP2 | MTP4 |
|-----------|-----:|-----:|-----:|
| short/32 | 137.4 | 152.3 | **204.6** |
| short/128 | 130.9 | 145.6 | **175.7** |
| p500/64 | 126.3 | 156.4 | **175.8** |
| p1k/64 | 127.7 | 151.8 | **179.8** |
| p2k/32 | 121.1 | 133.7 | **159.4** |
| p2k/128 | 128.5 | 143.9 | **165.9** |
| p4k/128 | 123.5 | 143.1 | **171.6** |
| p8k/64 | 118.5 | 137.1 | **165.9** |
| p8k/128 | 119.7 | 133.4 | **159.6** |

**Prefill t/s (cold, no prefix cache):**

| prompt | MTP1 | MTP2 | MTP4 |
|--------|-----:|-----:|-----:|
| short | 1435 | 1373 | 1146 |
| p500 | 5607 | 5381 | 4893 |
| p1k | 7235 | 7078 | 6479 |
| p2k | 8290 | 7962 | 7409 |
| p4k | 8172 | 7795 | 7601 |
| p8k | 7454 | 7337 | 7179 |

**Power/thermal (165W cap, energy-delta):**

| N | card avg | card peak | pkg avg | temp pkg avg/peak |
|---|---------:|----------:|--------:|------------------|
| 1 | 153.4W | 181.8W | 96.7W | 66/73°C |
| 2 | 157.5W | 188.5W | 98.6W | 67/72°C |
| 4 | 159.8W | 190.5W | 99.4W | 66/72°C |

Card avg 154-160W at 165W cap (MoE self-limits below cap); 190W peaks are
prefill bursts. pkg wattage is CPU package, not GPU.

**Downsides by N:**
- **MTP4:** peak decode short/medium ctx (+49% short), but **prefill -11%**
  (4 draft forwards during prefill) and decode edge **narrows at long ctx**
  (short 204.6 → p8k 159.6, -22%).
- **MTP2:** balanced — decode +19%, prefill only -3-4%, shallowest long-ctx dip.
- **MTP1:** least MTP overhead → best cold prefill, lowest decode.

**Frequency:** NOT readable on Xe/Level-Zero (no i915, no sysfs freq, xpu-smi
broken per AGENTS.md §11). Only watts + temps measurable.

---

## 5. Prefill Optimization — Results (A22, 2026-08-07)

**Goal:** raise cold prefill throughput. Levers tested: `max-num-batched-tokens`
(8192→16384), power (165W→230W), MTP spec count (1/4).

**Finding 1 — `max-num-batched-tokens` past 8192 does NOTHING.** A16b lifted
2048→8192 for +21-28%. Raising to 16384 at both 165W and 230W showed zero
change (p4k 7589→7599@165W; 9005→8972@230W — noise). Prefill is not
chunk-bound at 8K.

**Finding 2 — POWER is the prefill lever (opposite of decode).** Decode is
MoE-bandwidth-bound (self-limits ~140W). Prefill is compute-bound and scales
with power. MTP1 cold prefill:

| prompt | MTP1@165W | MTP1@230W | MTP4@165W | MTP4@230W |
|--------|----------|-----------|-----------|-----------|
| p2k | 7358 | **8537** (+16%) | 7409 | 8103 |
| p4k | 7589 | **9005** (+19%) | 7601 | 8715 |
| p8k | 7204 | **8824** (+22%) | 7179 | 8640 |

**MTP1@230W prefill ~9.0K t/s (p4k) — matches the A13 no-spec record (9,094)
and EXCEEDS the Reddit 7,975 target (+13%).** Power draw peaks at ~237W
(card_avg ~180W), temp peak 70°C (safe).

**Finding 3 — MTP1 is the best prefill config** (least draft overhead). MTP4
prefill is always slower (p4k 8715 vs MTP1 9005 @230W). Tradeoff: MTP4 wins
decode, MTP1 wins prefill.

**Finding 4 — no-spec prefill headroom is nearly exhausted.** A13 no-spec
230W record was 9,094. MTP1@230W hits 9,005 — the MTP draft prefill overhead
is now <1%. No meaningful headroom left without a kernel/memory-bandwidth
improvement.

**Bottom line:** to maximize prefill, use **MTP1 @ 230W** (~9.0K t/s). The
prefill is at the compute/power ceiling; the only remaining lever would be a
faster MoE/attention kernel or higher memory bandwidth. For a balanced
serving config, MTP1@230W is the prefill-optimal, MTP2@165W the
decode/efficiency-optimal.

**Harness:** `scripts/tmp/b70-mtp-prefill-sweep.py` (cold prefill, unique
prompts), `launch-mtp-bench.sh` env `B70_PREFILL=1 MTP_POWER_MW B70_MAX_NUM_TOKENS`.
Raw: `results/vllm-mtp-{1,4}-prefill-sweep.out`.

---

## 6. Dynamic Power/Clock Manager (A23, 2026-08-07)

**Concept:** prefill is compute-bound (scales to 230W, +16-22%); decode is
bandwidth-bound (self-limits to ~140W regardless of cap). So boost the power
cap to 230W during prefill bursts, drop to 165W during decode. This captures
the full prefill speedup for a tiny fraction of the duty cycle.

**Implementation:** `scripts/tmp/b70-dynamic-power.sh` — a reactive manager
that samples card watts (hwmon energy-delta) every 0.5s. When 3-sample power
exceeds BOOST_W (170W), it raises the cap to PREFILL_W (230W); when power
stays ≤ RELAX_W (155W) for HOLD samples, it drops to DECODE_W (165W).

**Verified live (MTP1 prefill sweep):**

| metric | static 165W | static 230W | dynamic |
|--------|------------|-------------|---------|
| p2k prefill | 7358 | 8537 | **8530** |
| p4k prefill | 7589 | 9005 | **8989** |
| p8k prefill | 7204 | 8824 | **8836** |
| time at 230W | 0% | 100% | **5%** |
| time at 165W | 100% | 0% | **95%** |

**Result:** dynamic delivers **identical 230W prefill performance** while the
card sits at 230W only 5% of the time (during prefill bursts). 95% of time is
at 165W (decode + idle). Prefill power peaked at 237W; decode settled to
~140W (self-limited). This is the classic "boost for the compute burst, relax
for the bandwidth-bound phase" pattern.

**Note on clocks:** the dynamic manager raises the power cap, which is what
lets the GPU clock up during prefill. Direct GPU frequency is not readable on
this Xe/Level-Zero stack (no i915/sysfs, xpu-smi broken — AGENTS.md §11), so
the power cap is the effective clock-boost control.

**Files:** `scripts/tmp/b70-dynamic-power.sh` (env: B70_PREFILL_W, B70_DECODE_W,
B70_BOOST_W, B70_RELAX_W, B70_HOLD); log `/tmp/dyn-power.log`.

---

## 8. Consolidated Final Results (2026-08-07)

**Model:** Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4 · **Engine:** vLLM
0.21.0-xpu-int4moe, native INT4 MoE v4 + BF16 MTP draft · single-stream ·
prefix-caching OFF (honest cold prefill) · Intel Arc Pro B70 32GB (608 GB/s).
Decode grid @165W, prefill sweep @230W.

### Decode (t/s) — grid @165W, steady-state

| Prompt → Gen | MTP1 | MTP2 | MTP4 |
|-------------|-----:|-----:|-----:|
| short → 32 | 137.4 | 152.3 | **204.6** |
| short → 64 | 134.8 | 155.2 | **190.6** |
| short → 128 | 130.9 | 145.6 | **175.7** |
| p500 → 32 | 122.5 | 156.9 | **154.8** |
| p500 → 64 | 126.3 | 156.4 | **175.8** |
| p500 → 128 | 127.9 | 152.8 | **176.6** |
| p1k → 32 | 127.2 | 150.6 | **167.3** |
| p1k → 64 | 127.7 | 151.8 | **179.8** |
| p1k → 128 | 125.8 | 150.8 | **175.6** |
| p2k → 32 | 121.1 | 133.7 | **159.4** |
| p2k → 64 | 125.9 | 139.1 | **175.8** |
| p2k → 128 | 128.5 | 143.9 | **165.9** |
| p4k → 32 | 116.5 | 134.5 | **157.7** |
| p4k → 64 | 121.8 | 141.2 | **173.1** |
| p4k → 128 | 123.5 | 143.1 | **171.6** |
| p8k → 32 | 115.2 | 129.1 | **150.8** |
| p8k → 64 | 118.5 | 137.1 | **165.9** |
| p8k → 128 | 119.7 | 133.4 | **159.6** |

### Prefill (t/s) — cold, no prefix cache

| Prompt | MTP1@165W | MTP1@230W | MTP4@165W | MTP4@230W |
|--------|----------:|----------:|----------:|----------:|
| p500 | 5607 | 5289 | 4894 | 4801 |
| p1k | 7235 | 7235 | 6479 | 6548 |
| p2k | 8414 | **8530** | 7653 | 8103 |
| p4k | 8277 | **8989** | 7738 | 8715 |
| p8k | 7518 | **8836** | 7246 | 8640 |

### Power / Thermal (measured @165W cap, energy-delta)

| Config | Decode card avg | Decode card peak | Prefill peak | Temp pkg avg/peak |
|--------|----------------:|-----------------:|-------------:|------------------:|
| MTP1 | 153.4W | 181.8W | ~237W | 66/73°C |
| MTP2 | 157.5W | 188.5W | — | 67/72°C |
| MTP4 | 159.8W | 190.5W | ~237W | 66/72°C |

### Headlines
- **MTP4 short/32 decode 204.6 t/s** (+49% vs MTP1, +41% vs community 145)
- **MTP1@230W prefill p4k 9,005 t/s** (+13% vs community 7,975)
- **Dynamic power: identical 230W prefill at 5% duty** (boost prefill, relax decode)
- **Concurrency max: C64 = 1,967.7 t/s aggregate / 1,139.8 gen t/s** (no-MTP native, diverse 512-token prompts). MTP+concurrency blocked by GDN kernel, so max-token uses no-MTP.

### Concurrency max (no-MTP native int4, diverse 512-token prompts, 165W)
| Concurrency | Aggregate t/s | Gen t/s | Median TPOT |
|------------:|--------------:|--------:|------------:|
| C16 | 1,107.8 | 641.7 | 25ms |
| C32 | 1,523.8 | 882.7 | 36ms |
| C48 | 1,578.2 | 914.2 | 52ms |
| **C64** | **1,967.7** | **1,139.8** | 56ms |

### Config roles
| N | Quant | Power | Best role |
|---|-------|-------|-----------|
| MTP1 | GPTQ-Int4 (WNA16, int8 store) | 150-165W decode / 230W prefill | Max prefill, least MTP overhead |
| MTP2 | GPTQ-Int4 | 165W | Balanced decode+prefill |
| MTP4 | GPTQ-Int4 | 165W decode / 230W prefill | Peak short-ctx decode |

### Reddit user comparison (Dolboyob77, custom image + kernel)

His numbers: pp4096 = 9,268 ± 39.69 t/s (TTFR 407ms); tg32 = 174.54 ± 13.05
(peak 180.17). Our MTP4 (stock int4moe image + 2 patches, no custom kernel):

| Metric | Us (MTP4) | Him | Δ |
|--------|-----------:|-----|--:|
| tg32 decode | **204.6 t/s** | 174.54 | +17% |
| pp4096 prefill | ~8.5-8.7K t/s | 9,268 | ~-6% |

**Read:** MTP4 beats his decode; his custom kernel/image edges us on prefill.
His TTFR 407ms covers a true 4096 prompt; ours 646ms covers p4k-5623 (not a
fair e2e comparison). Prefill gap is the target for a custom kernel (his
advantage).

---

## 7. Files Created / Used

| File | Purpose |
|------|---------|
| `scripts/tmp/launch-mtp-bench.sh` | Parametrized MTP-N launch + bench (env: B70_NO_PC, B70_GRID, MTP_POWER_MW, MTP_LOAD_MAX) |
| `scripts/tmp/b70-mtp-grid-bench.py` | prompt×output grid with honest cold prefill + decode |
| `scripts/tmp/monitor-power.sh` | card/pkg watts (energy-delta) + temps, 1s samples |
| `scripts/tmp/b70-mtp-mixedsplit-sweep.py` | C1/C2/C4/C8 concurrency sweep |
| `scripts/tmp/launch-mtp-mixedsplit-guardrail.sh` | guardrailed MTP concurrency test |
| `scripts/tmp/vllm-xpu-causalconv-patch/` | GDN mixed-split patch + sandbox test + mock |
| `results/vllm-mtp-{1,2,4}-bench.{json,out}` | single bench (A20) |
| `results/vllm-mtp-{1,2,4}-grid-bench.out` | full grid (A21) |
| `results/vllm-mtp-{1,2,4}-power.log` | watt/temp telemetry (A21) |

## 7. Machine State / Restore

Machine left in **full research mode** (production NOT restored, per user
request). Power cap 150W, idle ~53°C. To restore production:
`systemctl --user start llama-profile.service` (active-profile currently
`thinkingcap27-mtp-q5-200k-165w`).
---

## 9. LocalMaxxing Submission — the correct endpoint (2026-08-07)

**The bug:** `lmx benchmark submit` POSTs to `/api/benchmarks` (an EVALS-only
path) → returns HTML, reports false `status: submitted`, nothing lands. The
correct speed-test endpoint is **`POST /api/speed-tests`** (schema
`SpeedTestSubmit`, returns **201** with the run + `status: "APPROVED"`).

**Both submissions succeeded (HTTP 201, APPROVED):**

| Run | tokSOut | batchSize | ID | URL |
|-----|--------:|----------:|----|-----|
| MTP4 single-stream | 204.6 | 1 | `cmsiwwpzf00a4qm01z18izmad` | https://www.localmaxxing.com/en/models/Qwen/Qwen3.6-35B-A3B?run=cmsiwwpzf00a4qm01z18izmad |
| Concurrency max | 1139.8 (gen) | 64 | `cmsiwwqmt00a9qm010iekvi3u` | https://www.localmaxxing.com/en/models/Qwen/Qwen3.6-35B-A3B?run=cmsiwwqmt00a9qm010iekvi3u |

**Script:** `submit_speedtest_correct.py` (urllib POST to /api/speed-tests,
SpeedTestSubmit schema, Bearer key). Verified: leaderboard 201, model page 200.

**Author feedback (GitHub issue for localmaxxing-cli):** `lmx benchmark submit`
hits the wrong endpoint (`/api/benchmarks` not `/api/speed-tests`), reports
`status: submitted` without validating the HTML error body, and returns no
submission ID. dry-run also 404s. Fix: route submit to `/api/speed-tests`.

**Filed on github.com/LottoLottoLotto/localmaxxing-cli:**
- **#12** — submit hits /api/benchmarks (evals-only) instead of /api/speed-tests; false success
- **#13** — no way to list own submissions; submit returns no run ID
- **#14** — --concurrency silently dropped; dry-run 404s; --served-model degrades resolution
