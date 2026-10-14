---
title: "B70 Benchmark Protocol"
type: method
date: 2026-08-08
status: living
---

# B70 Benchmark Protocol

**How we benchmark and publish claims on the B70.** Follow these rules or the numbers are invalid or non-publishable. This document is the single methodology and claim-publication authority; it supersedes prior guides.

Protocol version: 3.0
Effective date: 2026-08-09

**Applies to both engines.** Sections 1–8 define shared validity and claim rules.
Section 9 adds vLLM XPU rules. Use `llama-bench` for llama.cpp engine tests and
the vLLM endpoint plus `vllm bench serve` for vLLM. See
`archive/legacy-docs-20260915/benchmark-standards-audit-2026-08-08.md`.

---

## 1. Hard validity rules

### 1.1 One engine instance and one campaign harness
GPU contention causes OOM, corrupted metrics, and xe driver VM worker errors. Never run two inference engines, two benchmark campaigns, or an unrelated GPU workload simultaneously. A controlled concurrency campaign may send Cn requests to one server when concurrency is the declared independent variable.

Before starting any server, all checks must pass:
```bash
pgrep -af 'llama-server|llama-bench|vllm|vllm-xpu'  # no inference process
docker ps --format '{{.ID}} {{.Image}} {{.Names}}'  # no inference container
```
If the production service is running, stop it first:
```bash
systemctl --user stop llama-profile.service
```
Verify empty-card VRAM is about 31,000 MiB or more with the B70 debugfs node.
Record each card's observed `power1_cap` first (the 150 W default is retired
2026-09-14 — caps vary per card and across reboots) and restore that value at
cleanup. Restart production when done:
```bash
systemctl --user start llama-profile.service
```

### 1.2 Warmup is mandatory (discard round 1)
The first request after server start includes SYCL JIT compilation (30–60s overhead).
Always run 1 warmup request and **discard** it. Steady-state = rounds 2+.
Within a sweep of the same model, subsequent runs are clean (driver-level cache).

### 1.3 Cooldown between comparable tests (≤55 °C)
Wait for GPU temp to drop before the next test. Without cooldown, residual heat
can bias a matched comparison. Discover the B70 hwmon node and its labeled
package sensor first; do not copy a historical `hwmon4` path into a new script.
Example after resolving `B70_HWMON` and `B70_TEMP_INPUT`:
```bash
while [ "$(($(cat "$B70_HWMON/$B70_TEMP_INPUT")/1000))" -gt 55 ]; do sleep 2; done
```

### 1.4 Use the correct timing source for each metric

| Metric | Required source |
|---|---|
| `llama-bench` prefill/decode | Raw emitted `ppN`/`tgN` repetition data |
| HTTP C1 engine decode | Server token count and predicted duration/rate; retain the raw fields |
| HTTP prompt processing | Server prompt-token count and prompt timing after cache validation |
| TTFT, TPOT/ITL, request latency | Client-side monotonic timestamps |
| vLLM serving latency (TTFT/TPOT/ITL/E2EL) | `vllm bench serve` percentile output (p50/p99); never client wall-clock of a broken decode definition (§9.1) |
| vLLM server-native cache/spec counters | `/metrics` discrete counters `vllm:prefix_cache_*`, `vllm:spec_decode_*` (authoritative); latency histograms corroborate only (§9.3) |
| Cn aggregate generation | Successfully generated output tokens divided by the monotonic campaign interval |
| Sustained throughput | Full predeclared measured window after declared warmup |
| Actual average power | Energy-counter delta divided by the same monotonic interval, or a documented sampler |

Engine rate isolates decode from prompt and network time. Report end-to-end UX separately. Never sum per-request `predicted_per_second` values to produce aggregate throughput, and never mix engine-reported and client-observed values in one unlabeled column.

### 1.5 ⚠️ Prefill: use the engine-correct method
**This is the most common measurement error.** There are three ways to get prefill wrong:

**Error 1 — short prompts.** With short prompts (34–43 tokens), prefill is dominated
by fixed kernel-launch overhead, not memory bandwidth. A 34-token prompt reports
~140 t/s; the same model on a 4096-token prompt reports ~1600 t/s.

**Error 2 — HTTP-API quoting bug.** Passing a long prompt through `curl -d` with
`jq -Rs . <<<"$PROMPT"` can silently truncate the prompt to 4 tokens, producing
fake ~75 t/s prefill. This is extremely hard to detect.

**Error 3 — ⚠️ prefix-cache contamination (THE SILENT ONE, verified 2026-08-07).**
vLLM (with `--enable-prefix-caching`) and any engine with prefix caching reuses
KV blocks across requests that share a prompt prefix. A benchmark harness that
uses a **constant filler string** + a unique suffix per rep (e.g. `"FILLER... "
+ f"rep {i}"`) makes reps 2-3 (and any later run) **hit the cache on the shared
prefix** and report prefill that is almost free — we measured p8k jumping from
**8,640 → 42,733 t/s** (a **5× lie**) once cached. The "steady-state = reps 2-3"
convention makes this WORSE, because rep 1 is the only honest one and it gets
discarded. This bug produced a false "+16-22% prefill from 230W" claim that
survived for two days until a paired A/B killed it (Run 24).
**Detection (always do BOTH):**
1. Snapshot the cumulative cache-hit counter immediately before and after the measured phase:
   - vLLM: `curl -s localhost:8001/metrics | grep prefix_cache_hits_total`.
   - Valid uncached prefill requires `after - before = 0`. The process-wide counter may already be nonzero from earlier work.
   - llama-server: there is no counter; rely on unique prefixes and the repetition sanity check.
2. Compare repetitions at the same actual prompt length. Any unexplained material acceleration requires investigation; rep 2 >1.5× rep 1 automatically invalidates the run but smaller contamination is still possible.
**Fix:** prepend a **unique random prefix per call** (≥64 chars) so no two calls
ever share a prefix. Use `scripts/tmp/b70-prefill-honest.py` (it does this). Do
NOT use the old `b70-mtp-prefill-sweep.py` or `b70-mtp-grid-bench.py` — they use
a constant filler and are contaminated. When `llama-bench` is usable (no spec),
it is immune (synthetic prompts, prefix caching off internally).

**llama.cpp:** use `llama-bench` when it supports the tested configuration. It
generates synthetic prompts at exact lengths and is immune to these HTTP-specific
errors. For MTP/speculative paths unsupported by `llama-bench`, controlled HTTP
is permitted only with exact endpoint token counts, entropy at the start of each
prompt, preserved target length, and zero cache-hit counter delta.

**vLLM:** use controlled endpoint requests. `llama-bench` cannot test a running
vLLM server. Call `actual input tokens / client TTFT` a **cold input rate**, not
isolated prefill. Claim isolated prefill tok/s only when a server-native phase
timer excludes queueing and first-token work.

```bash
# CORRECT — llama-bench with synthetic prompts
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
llama-bench \
  -m MODEL.gguf -ngl 99 -ncmoe 0 \
  -fa 1 -ctk q8_0 -ctv q4_1 \
  -p 512,2048,4096,8192 -n 128 -r 5 -d 0 \
  -b 8192 -ub 4096 -t 8 -dev SYCL0
```

**Reference numbers** (Qwen3.6-35B Q4_K_XL, b10222, 150W):
- pp512 = 1041 t/s, pp4096 = 1621 t/s, pp8192 = 1551 t/s, tg128 = 69.5 t/s

**If you must use the HTTP API** (e.g., for MTP/speculative decoding which
llama-bench doesn't support): verify `prompt_n` in the response JSON is ≥1024
before trusting the prefill number. If `prompt_n` is 4, your prompt was truncated.
**AND** defeat prefix caching per Error 3 (unique random prefix per call + check
the cache-hit counter).
**vLLM stack:** this section is llama.cpp-specific. For vLLM, latency uses
`vllm bench serve` (§9.1). Controlled HTTP with zero cache-hit delta yields cold
input rate; isolated prefill needs a server-native phase timer (§9.4). See
`archive/legacy-docs-20260915/benchmark-standards-audit-2026-08-08.md`. V1 counter names are
`vllm:prefix_cache_hits` / `vllm:prefix_cache_queries` (token-level), not
`prefix_cache_hits_total`.
**vLLM cache-off validation:** V1 in the pinned v0.26.1rc1 image defaults prefix caching to on. Omitting `--enable-prefix-caching` does not disable the feature. Use `--no-enable-prefix-caching`, retain startup evidence showing `enable_prefix_caching: False`, and run a resident/repeated-prefix request that still records zero hits. A unique cold prompt with zero hits cannot prove cache-off because it has nothing reusable.


### 1.6 Retain all samples; distinguish controlled synthetic and diverse HTTP prompts

For `llama-bench`, fixed synthetic lengths provide control and repeatability. Retain every repetition value.

For HTTP campaigns, rotate multiple prompt families (essay, technical, design, comparison) and prepend a unique random run ID so no full prefix repeats. The entropy must fit inside—not extend beyond—the declared target prompt length.

After one same-shape warmup, use three measured repetitions for a provisional
screen and five for a publishable point. A screen may choose which candidates
deserve five samples; it cannot support a public representative claim. Report
the median, maximum observed, sample count, and range. Do not discard slow valid
samples merely to improve “steady-state.”

### 1.7 Control power and order effects
For ordinary model comparisons, set one cap, wait 3 seconds, then start the
server. Keep it fixed for that campaign. A power-scaling experiment may change
the cap on one warm server only when it uses a paired alternating A/B design:
settle the cap, cool to the declared threshold, alternate which cap runs first,
use matched fresh prompts, and record measured energy for every interval.
```bash
echo 150000000 | sudo tee "$B70_HWMON/power1_cap"  # 150W
```
Values: 150000000=150W, 165000000=165W, 180000000=180W, 230000000=230W(stock).

### 1.8 Restore state on exit
Restore each card's **pre-campaign observed** cap, not a remembered default:
```bash
PRE_CAMPAIGN_CAP=$(cat "$B70_HWMON/power1_cap")   # record BEFORE the campaign
trap 'echo "$PRE_CAMPAIGN_CAP" | sudo tee "$B70_HWMON/power1_cap"; \
      pkill -f "llama-server|vllm"; \
      systemctl --user start llama-profile.service' EXIT
```

### 1.9 Tiered, claim-sized campaigns

Do not run a Cartesian product of every context, output length, power cap, cache
mode, and MTP depth for each new model. Pick the smallest lane that answers the
question. Reuse valid cells when the image/binary, model, tokenizer/template,
quantization, scheduler, KV type, cache policy, context, and power match.

| Lane | Purpose | Minimum measured work | Evidence use |
|---|---|---|---|
| 0: smoke | Load, correctness shape, telemetry, cleanup | One warmup plus one observation | Diagnostic only |
| 1: model card | Honest routine comparison | Cold `p512/g128`, `p8192/g128`, `p2048/g1`, `p8192/g1`, plus one real workload; `n=5` | Narrow public claims for named cells |
| 2: tuning | Choose MTP, scheduler, KV, power, or kernel | Baseline plus candidates at two anchor cells; `n=3` screen, then fresh `n=5` confirmation for finalists | Selection evidence; publish only fresh confirmation sets |
| 3: characterization | Broad curve, sustained, capacity, concurrency, or general winner claim | Only the extra dimensions named by the claim | Broad publication after the full gate |

For four modes, the old 17-coordinate vLLM matrix required 408 warmup and
measured requests before retries. A two-anchor screen plus one winner's Lane 1
card normally needs about 62–76 requests, including smoke and a possible close
finalist. Model loads and long-context cells
also fall sharply. Record the planned request count and estimated wall time in
the manifest before launch.

For llama.cpp, translate `pN/gN` to supported `ppN`/`tgN` coordinates and state
the prompt state for decode. Lane 1 uses `tg128` as representative decode and
two prefill anchors, normally `pp2048` and `pp8192`. Add `tg32`, `tg256`, and
`tg512` only for a generation-length curve or sustained-behavior question.

For MTP tuning, screen no-spec and supported MTP depths at `p512/g128` and
`p8192/g128`. Drop a candidate only when it is slower at both anchors beyond
the observed run-to-run noise and offers no latency, capacity, quality, or
correctness advantage. Confirm the winner and any close finalist with a fresh
five-sample set; do not fold the selection samples into the published median.
Acceptance is a diagnostic, not the objective; rank by the declared user metric. The
[MTP spec-token investigation](https://sergiiob.dev/posts/intel-arc-b70-mtp-spec-token-curve/)
is the reference example.

Full context ladders belong to capacity claims. Power sweeps belong to power or
efficiency claims. Concurrency sweeps belong to serving-capacity claims. A
missing point is `not measured`, never inferred.

#### vLLM lane 1 coordinates

`llama-bench` cannot measure a vLLM server. Keep endpoint metrics distinct:

- Cold input anchors: exact `p2048/g1` and `p8192/g1`.
- Decode anchors: exact `p512/g128` and `p8192/g128`.
- Real-workload control: use a fixed prompt from the target application, such as
  the matched Pi `p9445/g128` control, or record the actual workload shape.
- Add p512/p4096/p6144 input cells, g32/g256/g512 decode cells, and actual
  context-boundary requests only when the claim covers those curves.
- Use a compact fixed system message for the standard p512-p8192 grid. A full Pi
  system/tool prompt can already exceed p512; never claim an impossible token
  coordinate. Keep real-Pi controls in separate rows.
- Force exact decode lengths with the endpoint's `ignore_eos=true` option.
  Natural early-EOS behavior is a separate user-output study, not a valid `g512`
  throughput sample.
- Run one full-output same-shape warmup, then five measured requests with
  distinct entropy-first prefixes for every publishable cell.

Label vLLM `actual endpoint input tokens / client TTFT` as cold input rate; it
includes scheduling and first-token work. Label decode as `client post-first tok/s` and
retain TPOT. Neither field is a llama-bench engine-native result.

#### Pre-commitment probe gate (added 2026-09-14)

Before any work expected to exceed 30 minutes — benchmark campaign, long compilation,
conversion chain, calibration, or container-image pull — the campaign plan must declare:

1. the changed hypothesis and the changed variable class (one axis, not a grid);
2. the objective metric with explicit pass/fail thresholds (thresholds live in the
   campaign manifest, not in this rule);
3. the smallest falsifying probe, time-boxed, in fixed order: static compatibility
   check, then one warmed observation, then a matched short screen (`n=3` class);
4. resource scope (exact GPUs, power caps, container/process identities) and the
   cleanup that returns them;
5. a commitment deadline after which expansion stops for the night.

Rules of engagement:

- Commit to the full matrix only after the probe passes its declared threshold. A
  probe failure earns exactly one time-boxed diagnosis; a retry is justified only by
  a changed mechanism, never by re-running the same configuration.
- Rank candidates by the declared objective (throughput, latency, capacity, quality),
  never by a proxy such as acceptance rate, cache-hit rate, or utilization.
- Confirm finalists on fresh samples; never fold screen samples into a published set.
- Retain failed probes with their stop reasons — a documented no-go is a result.
- A completed negative result does not reopen because its publication status is
  `provisional`. Reopening requires naming the changed mechanism that invalidates the
  original failure (new artifact, fixed implementation, corrected harness — not
  "the PR merged upstream").

Reference applications: the 2026-09-13 compile-vs-eager A/B screen (validation ladder
in `research/2026/quant-format-showdown-20260913/ARCHITECT-VERDICT.md`) and the 2026-09-14
MTP-depth screen (`results/2026/09/qw38-mtp-depth-20260914/`).

### 1.10 C1 and concurrency reporting

**C1** means one active measured request with no queued request. Report prompt/output coordinates, decode, TTFT/latency where relevant, statistic, and sample count.

**Cn** means a target of `n` concurrent in-flight requests to one server. Report requested and achieved concurrency, aggregate generated tokens/s, per-request p50/p95 TTFT/TPOT/end-to-end latency, completed/failed/timed-out requests, scheduler settings, workload distribution, and campaign duration.

Never compare C1 engine rate and Cn aggregate rate under the same generic `tok/s` label. Batch size, server parallel slots, and request concurrency are separate fields.

### 1.11 Peak, representative, and sustained

- **Maximum observed / peak:** highest valid post-warmup sample or named cell; always label it `maximum observed` or `up to`.
- **Representative:** median of all valid point samples by default; report `n` and dispersion. A mean must say mean.
- **Sustained:** full-window throughput over a predeclared duration, request count, or long output length, including ordinary scheduler and thermal variation after warmup.

A short tg32/tg128 request or best repetition is not sustained. If a public headline uses a peak, show the representative value beside it.

---

## 2. KV cache: q8_0 K + q4_1 V (the fleet standard)

**This is the proven-optimal configuration.** See `docs/reference/kv-cache-quantization.md`
for the full KL-divergence analysis. Summary (fact-checked against llama.cpp #23470):

| K cache | V cache | KL-divergence (mean) | Verdict |
|---------|---------|---------------------|---------|
| **q8_0** | **q4_1** | **~0.003** | **✅ Recommended — near-lossless, ~50% VRAM savings** |
| q5_0 | q4_1 | ~0.006–0.008 | Acceptable trade-off (community-validated) |
| q8_0 | q8_0 | ~0.003 | Conservative — same K quality, more VRAM |
| q4_0 | *any* | **~5.5** | **❌ CATASTROPHIC — model output destroyed** |

**Rule: never use q4_0 for K-cache.** K determines which tokens to attend to
(routing); quantizing it to 4 bits causes KLD ~5.5 (only ~11.6% top-p match).
q5_0/q5_1 K are acceptable trade-offs (KLD ~0.006-0.008). q8_0 K is safest.
Compress V aggressively (q4_1), protect K (q8_0 or q5_0).

All profiles in `profiles.json` use `--cache-type-k q8_0 --cache-type-v q4_1`.

---

## 3. Mandatory server flags

```
-ngl 99          All layers on GPU
-ncmoe 0         All MoE experts on GPU (NOT CPU)
-fa on           Flash attention (required for quantized KV cache)
-ctk q8_0        K cache quantization (best KL-divergence match)
-ctv q4_1        V cache quantization (asymmetric: V compressed more than K)
-t 8             CPU threads
--no-mmap        Keep model in VRAM, don't page to disk
-b 8192          Batch size
-ub 4096         Micro-batch size (do NOT reduce — hurts prefill)
```

---

## 4. Power profiles

| Profile | Watts | Use case |
|---------|-------|----------|
| eco | 150 | **Daily default.** Use as the baseline for every power comparison. |
| efficient | 165 | Peak efficiency for dense 27B MTP-4 (0.148 t/s/W). |
| balanced | 180 | Sustained dense inference ceiling. |
| performance | 200 | High-power sweep tier. |
| burst | 230 | Stock cap — **short bursts only**. +9°C over 165W. |

Measured llama.cpp dense and MoE decode paths are bandwidth-bound at 608 GB/s;
their byte traffic differs. Do not generalize that decode model to prefill or to
another engine. Run 24 found vLLM MoE+MTP cold input flat from 150W to 230W in a
paired A/B. Run 30 found dense vLLM prefill scaled with power. Treat cap choice
as a matched workload result, not an architecture rule.

---

## 5. Environment variables (set before launching server)

```bash
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0
export SYCL_CACHE_PERSISTENT=0
export SYCL_DEVICE_FILTER=level_zero
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export ZE_AFFINITY_MASK=0
```

⚠️ **Do NOT disable SYCL optimization:** Setting `GGML_SYCL_ENABLE_OPT=0`
(historically `GGML_SYCL_DISABLE_OPT=1` in older builds) causes a large decode
regression in B70 measurements and was removed from launcher defaults on 2026-07-02.
The exact magnitude (internally measured at ~44.7% on a 9B model) is an internal
finding not corroborated by upstream llama.cpp; treat as approximate. Default is
`GGML_SYCL_ENABLE_OPT=1` (optimization on). Never set to 0 unless debugging.

---

## 6. Mandatory campaign manifest

Every run indexed in `archive/legacy-docs-20260915/benchmark-history.md` must link to raw artifacts and record enough provenance to reconstruct every public number.

### Identity and provenance

- Date/time and protocol version.
- Host/GPU/CPU/RAM/OS/kernel and driver/runtime versions.
- Engine name, version/build, commit, binary identity or pullable container repository plus immutable digest, observed in-container engine/kernel versions, and ordered local patches with hashes. A local-only image tag is not reproducible provenance.
- Model repository/name, exact revision or file, quantization, artifact path, size, and hashes when available.
- Harness path/version or git commit and raw-artifact paths/run ID.

### Exact configuration

- Complete command and relevant environment variables.
- Configured context, batch/microbatch, threads, GPU/MoE offload, KV K/V types, flash attention.
- Speculative-decoding method, draft length/layers, graph/eager mode.
- Prefix-cache state, scheduler limits, server parallel slots, requested and achieved concurrency.

### Workload and statistics

- Actual prompt tokens, requested output tokens, actual output tokens, prompt class/hash.
- Warmup count, measured repetitions, raw sample values, timing source and formula.
- Median, maximum observed, dispersion; sustained duration/request count when claimed.
- Separate C1 and Cn tables; TTFT/TPOT/latency percentiles and failure counts for HTTP concurrency.

### Power, thermals, and memory

- `power_cap_W` separately from measured average/peak draw and sampling method.
- Starting/peak/ending temperature and cooldown threshold.
- VRAM free after load and any throttling/OOM warning.

#### Real power measurement — no estimates

The configured cap is not actual draw. Measure board/card energy from the GPU hwmon `energy1_input` counter (microjoules) over the exact benchmark interval:

```text
actual_average_W = (energy_end_uJ - energy_start_uJ) / elapsed_seconds / 1,000,000
```

Requirements:

1. Discover and record the hwmon device instead of assuming `hwmon4`: inspect `/sys/class/hwmon/hwmon*/name`, verify the node maps to the B70 PCI device, and record the resolved path in the manifest.
2. Read `power1_cap` before the run and convert microwatts to watts; label it `configured_cap_W`.
3. Read `energy1_input` immediately before and after the measured workload using a monotonic timer. Record both raw counter values and elapsed nanoseconds/seconds.
4. Compute average card draw from the energy delta. If the counter wraps or decreases, invalidate the measurement unless the documented counter range permits a correct wrap calculation.
5. For peak and phase behavior, sample energy deltas at a fixed interval (normally 0.5–1.0 s). `peak_measured_W` is the maximum interval average—not an instantaneous electrical peak—and must be labeled accordingly.
6. Start monitoring before the first measured request and stop after the last measured request. Do not include server startup, graph capture, cooldown, or idle time in workload-average draw unless the metric is explicitly end-to-end campaign energy.
7. For separate prefill/decode power claims, align sample timestamps to request-phase timestamps or run isolated phase campaigns. Do not infer phase power from an unsynchronized monitor log.
8. Do not use configured cap, TDP, PSU rating, or a remembered typical value as actual wattage.

`energy2_input` may represent another package/domain. Record its hwmon label and report it separately; never silently add it to `energy1_input` or call it GPU board power without verifying the sensor meaning.

#### Real temperature measurement — no estimates

Temperature sensors are millidegrees Celsius. Discover and record all available `temp*_label`/`temp*_input` pairs on the verified B70 hwmon device. On the current host, `temp2_input` is the primary GPU/package sensor, while `temp3_input` and `temp4_input` may expose VRAM and memory-controller domains; verify labels rather than assuming names.

For every measured run:

1. Sample temperatures at the same fixed cadence as power, using monotonic timestamps.
2. Record `temperature_start_C` immediately before the measured workload.
3. Record the maximum observed temperature for each verified sensor during the measured interval.
4. Record `temperature_end_C` immediately after the final measured request.
5. Preserve the raw timestamped monitor log. Round only in presentation; retain at least 0.1 °C precision when the sensor exposes it.
6. Report the sensor explicitly, for example `GPU package peak: 67.4 °C`, not generic `temperature: 67 °C`.
7. If a sensor path disappears, returns invalid data, or its label cannot be verified, report `not measured`; never substitute an estimated temperature.
8. “No throttling” requires clock/performance evidence or stable interval throughput, not temperature alone.

Every public power/thermal statement must identify the measured interval, sensor/domain, sample cadence, statistic, and raw monitor artifact.

### Validity and publication

- Cache counter before/after/delta for uncached HTTP prefill.
- Prompt token-count verification and service/process isolation checks.
- Exclusions and reasons; contaminated and failed samples remain visible.
- Evidence tier and gate status: `provisional`, `validated_internal`, or `publication_approved`.

---

## 7. Known pitfalls

- **✅ USE `llama-bench` for prefill.** It's the correct, reliable tool (synthetic
  prompts, immune to quoting bugs). The HTTP API (`/v1/chat/completions`) is prone
  to prompt-truncation bugs that produce fake ~75 t/s prefill. See §1.5.
- **llama-bench may hang** after printing results. Monitor for the `tg128` line,
  then kill the process. Results are valid if they printed.
- **llama-bench does NOT support speculative decoding.** For MTP tests, use
  `llama-server` + HTTP API (but verify `prompt_n` ≥1024 for prefill).
- **`benchmark.py` measures wall-clock t/s** (`to/wall`), not engine rate. It also
  lacks warmup discard and long-prompt prefill. Do NOT use it for citable results.
- **128 max_tokens** may produce empty visible content on reasoning models. That does not prohibit a controlled `tg128` engine benchmark when endpoint usage counts reasoning tokens; distinguish visible content from actual completion tokens.
- A single `tg128` does not establish output-length-independent or sustained decode.
- C1 engine rate is not Cn aggregate throughput; summed per-request t/s is invalid.
- A configured power cap is not measured consumption.
- The cumulative cache-counter absolute value is irrelevant; validate its measured delta.
- The best repetition is a peak, not the default representative result.
- Timed-out, failed, partial, throttled, or contaminated samples cannot be silently omitted.
- **xpu-smi does NOT work** on this stack. Use debugfs for VRAM, hwmon for power/temp.
- **Never load a single GGUF larger than ~30 GB.** The B70 has 32656 MiB visible
  VRAM; a 34 GB model overflows and causes a **hard system crash** (full reboot).
  Use `-ngl <N` to offload tail layers to CPU if needed.

---

## 8. Claim publication gate

A campaign may enter a portfolio headline, README, social post, recommendation table, cross-hardware comparison, or leaderboard payload only when:

1. Raw artifacts and every measured sample are saved.
2. Workload coordinates, timing source, statistic, and sample count are explicit.
3. The required matrix is complete, or the claim is narrowed to the measured cells.
4. C1 and Cn results are separated.
5. Peak, representative, and sustained labels are accurate.
6. Uncached HTTP prefill records prefix-cache mode and zero measured cache-hit delta.
7. Configured power cap and actual draw are not conflated.
8. Failures, exclusions, contamination, throttling, and OOM risk are accounted for.
9. Every multiplier passes an apples-to-apples review covering metric, prompt/output, concurrency, cache, statistic, build/model/quant, speculation, and power.
10. A claims review approves the exact proposed wording and evidence links.
11. Every public recipe names a pullable image or binary, immutable revision, compatible patch generation and order, and exact model checkpoint. Historical local-only images must be labeled unavailable and must not appear in the current quick start.
12. Generated tables and SVGs use the same compiled numeric source. Regenerate
    through `.agentic/skills/b70-benchmark-visuals/SKILL.md`, inspect each SVG in
    a browser, and keep cookbook and portfolio copies byte-identical.

Until these gates pass, label the campaign:

> **PROVISIONAL — NOT FOR PUBLIC HEADLINE**

Independent reproduction is preferred but not required if publication clearly labels the result self-reported/evidence-backed. Platform acceptance or `APPROVED` status is not independent reproduction.

### Evidence tiers

- `E0`: estimate or extrapolation.
- `E1`: self-reported without raw artifacts.
- `E2`: self-reported with raw evidence and reproducible command.
- `E3`: externally published/approved but not independently rerun.
- `E4`: independently reproduced under a comparable protocol.
- `E5`: independently reproduced under a matched protocol.

### Correction policy

Preserve historical evidence. Mark defective results `superseded` or `invalid`; never silently erase them. Date the correction, identify the old claim and methodology defect, state what remains valid and what is withdrawn, link replacement evidence, and update every public surface that repeated the number.


---

## 9. vLLM XPU serving benchmark methodology

**Authority:** `archive/legacy-docs-20260915/benchmark-standards-audit-2026-08-08.md`. Benchmark the B70
vLLM XPU stack with vLLM-native tools, not `llama-bench`, which cannot reach a
running vLLM server. The current 2026-08-09 matrix observed vLLM
`0.26.1rc1.dev457+gc810e5ee9.xpu` and `vllm-xpu-kernels 0.1.12`; older 0.21
campaigns remain historical and require different patches.

### 9.1 Primary latency tool — `vllm bench serve`

Industry-standard serving benchmark. Talks to any OpenAI-compatible endpoint and
emits TTFT, TPOT, ITL, E2EL with selectable percentiles (p50/p90/p99) plus goodput
(SLO-constrained throughput). It is client-observed — that is correct by
construction: TTFT/TPOT/ITL are end-to-end UX metrics defined at the request side
by GenAI-Perf, MLPerf, and vLLM alike.

C1 single-stream:
```bash
vllm bench serve --backend openai --base-url http://<lab-host>:8001 \
  --model Qwen3.6-35B-A3B --dataset-name random \
  --random-input-len 9445 --random-output-len 128 --num-prompts 100 \
  --request-rate inf --max-concurrency 1 --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99 \
  --save-result --save-detailed --result-dir results/<run-id>
```
Cn concurrency: loop `--max-concurrency` over 1/2/4/8/16. Add
`--goodput ttft:MS tpot:MS e2el:MS` (ms) for SLO-bound throughput. `--ignore-eos`
forces a controlled `tgN`.

### 9.2 Metric glossary (GenAI-Perf / MLPerf / vLLM aligned)

| Metric | Definition | Formula |
|---|---|---|
| TTFT | request sent → first output token | `first_token_time − send_time` |
| TPOT | request-level average time per output token | `(E2EL − TTFT)/(output − 1)` |
| ITL | per-token inter-arrival distribution | individual `t_j − t_{j−1}` |
| E2EL | request sent → last token | end-to-end request latency |

- **TPOT ≠ ITL.** TPOT is one average per request; ITL is the distribution and
  exposes scheduling stalls. Report both.
- **Never use inclusive TPOT** (`E2EL/output`) — MLPerf forbids it. State which
  formula produced every number.
- Output token throughput is aggregate system capacity, not `1/ITL`; keep C1 and
  Cn in separate columns (§1.10).

### 9.3 Server-native metrics — Prometheus `/metrics`

Standard latency evidence comes from `vllm bench serve` (§9.1). `/metrics` plays
two distinct roles:
- **Discrete counters (authoritative):** `vllm:prefix_cache_queries` /
  `vllm:prefix_cache_hits` (token-level) and `vllm:spec_decode_num_drafts` /
  `num_draft_tokens` / `num_accepted_tokens` / `num_accepted_tokens_per_pos`. These
  are exact cumulative totals; diff across the measured window.
- **Latency histograms (corroborating/monitoring only):**
  `vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds`,
  `vllm:request_time_per_output_token_seconds`, `vllm:e2e_request_latency_seconds`.
  These are cumulative and bucket-granular; `histogram_quantile()` only
  approximates a percentile over a window, so they corroborate the `vllm bench
  serve` numbers — they are NOT a clean per-request decode rate.

Inspect the live endpoint first (V0/V1 names differ):
```bash
curl -s http://<lab-host>:8001/metrics | grep '^vllm:' | \
  grep -E 'time_to_first_token|inter_token_latency|time_per_output_token|prefix_cache|spec_decode'
```
Valid uncached prefill requires a `prefix_cache_hits` delta of 0 (replaces the old
`prefix_cache_hits_total` snapshot in §1.5 Error 3). Mean accepted tokens per
speculation round = `1 + accepted_tokens / num_drafts`.

### 9.4 vLLM cold input and prefill throughput

`vllm bench serve` reports TTFT, not isolated prefill tok/s. A controlled HTTP
runner with exact rendered tokens, a unique entropy-first prefix, and zero
`prefix_cache_hits` delta may report `actual input tokens / client TTFT` as cold
input rate. It still includes scheduling and first-token work. Use the label
`prefill tok/s` only when a server-native phase duration isolates prompt
processing; record that timer and formula.

### 9.5 Where the custom wrapper stays

`vllm bench serve` cannot capture B70 hwmon energy/temperature, debugfs VRAM
guards, synchronized counter snapshots, or exact Pi multi-turn replay. The custom
wrapper runs the latency core and layers on host telemetry and safety (audit §8).

---

## 10. Real-world assistant (Pi) workload matrix

Synthetic cold-prefix input represents document/RAG ingestion only. Real Pi
assistant use is dominated by warm multi-turn prefixes. Lane 1 tests one cold
turn and one representative warm follow-up. Test all seven states below only for
a broad Pi-serving or scheduler claim. Do not pick a scheduler budget from cold
input alone: a larger budget may improve document TTFT while worsening
short-request p95 latency.

### 10.1 Workload states

1. Cold new conversation.
2. Warm shared system/tool prefix.
3. Warm multi-turn session prefix.
4. New RAG/tool payload appended to a warm session.
5. Cold large-document ingestion.
6. Warm follow-ups over a resident document.
7. Mixed load: long prefill while short chat requests arrive.

### 10.2 User-facing metrics to capture

- Actual endpoint prompt and completion tokens (not requested).
- TTFT (request → first token) and TTFC (request → first visible content token).
- TPOT/ITL and end-to-end response latency.
- Server-native `/metrics` where available (corroboration; counters authoritative).
- Prefix-cache query/hit delta per phase.
- MTP acceptance counters (overall + per draft position).
- C1 vs aggregate Cn separation.
- Real energy and named-sensor temperature telemetry.
- Failures, queueing, p50/p95 under mixed load.

### 10.3 Cold vs warm prefix semantics

- **Cold:** zero shared prefix with resident KV. Unique random entropy-first
  prefix per call; `prefix_cache_hits` delta must be 0. Represents first document
  load or a brand-new session.
- **Warm:** reuse a shared system/tool/session prefix across requests (resident
  KV). This is the normal multi-turn steady state. Warm TTFT is the number that
  matters for perceived Pi latency. Always report cold and warm separately.

---

## 11. Exact rendered-token calibration and shape warmup

### 11.1 Token count from the tokenizer, never from field count

`len(tokenizer(prompt))` and `len(BatchEncoding)` return the number of encoding
fields, not the token count — this produced a wrong (huge) prompt in the last
campaign. Always use:
```python
len(encoded["input_ids"])
```
Calibrate every prompt to its actual rendered token count and record it. Re-verify
after any tokenizer or chat-template change.

### 11.2 Shape-specific warmup

A generic warmup does not clear JIT/shape contamination at a new prompt length.
For each distinct prompt length, run one same-length warmup and discard it, then
measure. The pp512 and p≈4757 first samples showed shape-specific contamination
when this was skipped.

---

## 12. Staged long-context capacity guards (claim-driven)

A configured 128K context is not evidence of 128K support. A real request must
complete with prompt + output inside the logged KV capacity. Current logged KV
capacity (Qwen3.6-35B-A3B GPTQ-Int4 + MTP): MTP1 ≈ 157,286; MTP2 ≈ 140,434;
MTP4 ≈ 115,651 tokens. MTP4 cannot complete a full 128K sequence today.

Do not run this ladder for a routine model card. Run it when the model is new to
the host, its KV geometry or memory policy changed, or the proposed claim names
a context capacity. Reuse a prior capacity result only when all memory-relevant
coordinates match.

### 12.1 Safe progression

16K control → 32K → 64K → 96K → 128K, only if logged KV capacity and VRAM reserve pass at each step.

### 12.2 VRAM reserve after load (debugfs `visible_avail`)

- < 500 MiB free: abort.
- 500–1,023 MiB: isolated C1 capacity observation only. One request at a time; no mixed traffic, concurrency, or production claim.
- 1,024–2,047 MiB: staged C1 research is allowed. Record the reserve and stop on any allocator, preemption, or driver anomaly.
- ≥ 2,048 MiB: preferred for C1 performance campaigns.
- ≥ 3,072 MiB: target for mixed or concurrent serving.

The 500 MiB floor is an operator-approved research boundary, not a production-safety claim. A model fitting in VRAM does not prove that its scheduler, speculative buffers, or mixed-token kernels are correct.

### 12.3 Boundary completion cells

A context claim needs a request that fits prompt + output inside the limit:
- p131040/g32, p130944/g128, p130560/g512 for a true 128K test.
- p122880 leaves safer generation/follow-up room for practical document use.

### 12.4 Likely production split

MTP4 for fastest 16K/32K assistant use; MTP2 or MTP1 for 128K document sessions (they have higher KV capacity headroom).
