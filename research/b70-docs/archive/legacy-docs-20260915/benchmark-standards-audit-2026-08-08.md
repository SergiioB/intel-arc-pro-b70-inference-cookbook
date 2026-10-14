# B70 Benchmark Standards & Tooling Audit (2026-08-08)

**Scope:** Resolve the central methodology question — is the custom HTTP harness
the best industry-standard approach, can `llama-bench` be used, and should
LocalMaxxing be trusted — before any new run.

**Status:** Read-only audit, no GPU used. Proposes methodology only; exact public
wording still requires the claims-auditor gate (`benchmark-protocol.md` §8).

---

## 1. Verdict (the central question, answered)

The custom HTTP harness was **honest but used a broken decode definition**: it
counted all output tokens while timing from after the first token — that is
neither clean TPOT nor clean ITL — and retained no server fields for
corroboration. Note: client-observed is NOT the flaw; TTFT/TPOT/ITL are
inherently end-to-end UX metrics measured at the request side by GenAI-Perf,
MLPerf, and vLLM alike. The fix is `vllm bench serve`, which is client-observed
but applies the standard TPOT/ITL definitions by construction. For the llama.cpp
stack, `llama-bench` remains the correct tool.

| Question | Answer |
|---|---|
| Is the custom HTTP harness the best approach for vLLM latency? | **No.** Use `vllm bench serve` as the primary standard tool; keep the custom wrapper only for B70 telemetry + safety guards the official tool cannot capture. |
| Can `llama-bench` be used? | **For the llama.cpp stack, yes** (synthetic pp/tg). **For the vLLM stack, no** — it launches its own GGUF model and cannot benchmark a running vLLM server. |
| Should LocalMaxxing be trusted? | **As a self-report leaderboard, yes; as independent verification, no.** Platform APPROVED = published self-report, not reproduction. Keep it as a normalized reporting surface; never let its flat payload replace the internal raw evidence manifest. |

---

## 2. The B70 runs two engines — pick the tool per engine

The cross-engine comparison (`docs/cross-engine-comparison-2026-08-06.md`)
documents two distinct stacks on the same GPU. They need different tools.

| Stack | Engine | Format | Spec | Primary benchmark tool |
|---|---|---|---|---|
| **Production (Pi/Telegram)** | llama.cpp SYCL b10255+ | GGUF (Q4_K_XL, Q6_K) | MTP-4 on dense 27B | `llama-bench` (synthetic) + HTTP for MTP paths |
| **Research** | vLLM 0.21 XPU | GPTQ-Int4 MoE | MTP on MoE 35B-A3B | `vllm bench serve` + Prometheus `/metrics` |

**Decision:** the current campaign targets the **vLLM research stack**, so the
primary tool is `vllm bench serve`, not `llama-bench`.

---

## 3. Metric vocabulary — GenAI-Perf / MLPerf / vLLM alignment

All three industry references converge on the same four latency metrics. The
definitions below are the canonical ones; cite the source when publishing.

| Metric | Definition | Formula | What it exposes |
|---|---|---|---|
| **TTFT** | Request sent → first output token | `first_token_time − request_send_time` | Queueing + prefill + first-token compute (perceived responsiveness) |
| **TPOT** (vLLM / MLPerf) | Request-level average time per output token | `(E2EL − TTFT) / (output_tokens − 1)` | Decode cadence averaged over one request |
| **ITL** (GenAI-Perf / vLLM) | Distribution of per-token inter-arrival gaps | individual `t_j − t_{j−1}` | Stalls, scheduling jitter, per-token variance |
| **E2EL** | Request sent → last token | end-to-end request latency | Total user-perceived latency |

**Two traps this resolves (both hit the handoff):**

1. **TPOT ≠ ITL.** TPOT is one averaged number per request; ITL is the full
   per-token distribution. vLLM's `vllm bench serve --save-detailed` reports
   both, plus p50/p90/p99 percentiles. The handoff's decode harness "counted
   all output tokens while timing from after the first token" — that is neither
   clean TPOT nor clean ITL. `vllm bench serve` fixes it by construction.
2. **Never use "inclusive TPOT."** Some frameworks spread TTFT across all
   output tokens (`E2EL / output_tokens`). MLPerf explicitly forbids this for
   its TPOT definition. Always state which formula a number came from.

**Aggregate vs per-user:** Output token throughput
(`total_output_tokens / benchmark_duration`) is a system-capacity metric across
concurrent requests; it is NOT `1/ITL` and must not be compared to a single-user
decode rate. Keep C1 (single active request) and Cn (concurrent) in separate
columns, as `benchmark-protocol.md` §1.10 already requires.

---

## 4. vLLM official benchmark tooling — `vllm bench serve`

This is the industry-standard serving benchmark and the **primary tool for the
vLLM stack**. It talks to any OpenAI-compatible endpoint (so it works against
our vLLM XPU server regardless of backend/MTP patches) and emits TTFT/TPOT/ITL/
E2EL with selectable percentiles.

The older `python benchmarks/benchmark_serving.py` is deprecated; use the CLI.

### 4.1 C1 single-stream latency (the default decode/TTFT point)

```bash
vllm bench serve \
  --backend openai \
  --base-url http://<lab-host>:8001 \
  --model Qwen3.6-35B-A3B \
  --dataset-name random \
  --random-input-len 9445 \
  --random-output-len 128 \
  --num-prompts 100 \
  --request-rate inf \
  --max-concurrency 1 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,95,99 \
  --save-result --save-detailed \
  --result-dir results/vllm-bench-c1
```

`--max-concurrency 1 --request-rate inf` = exactly one in-flight request = C1.
`--ignore-eos` forces the requested output length (controlled `tgN`).
`--save-detailed` retains per-request ITL distributions.

### 4.2 Cn concurrency scaling (the multi-user curve)

```bash
for c in 1 2 4 8 16; do
  vllm bench serve \
    --backend openai --base-url http://<lab-host>:8001 \
    --model Qwen3.6-35B-A3B \
    --dataset-name random \
    --random-input-len 2048 --random-output-len 512 \
    --num-prompts 200 --request-rate inf \
    --max-concurrency "$c" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --save-result --result-dir "results/vllm-bench-c${c}"
done
```

### 4.3 Goodput — SLO-constrained throughput

```bash
--goodput ttft:500 tpot:50 e2el:2000   # milliseconds
```

Reports the request rate that *also* satisfies the latency SLOs. This is the
right number for "what throughput can we promise real Pi users," not raw peak.

### 4.4 Output it produces

```
Successful requests:        100
Benchmark duration (s):      12.4
Request throughput (req/s):    8.06
Input token throughput:    9,445 × 100 / 12.4
Output token throughput:     ...   ← aggregate generation capacity
Mean TTFT (ms) / P99 TTFT
Mean TPOT (ms) / P99 TPOT
Mean ITL (ms) / P99 ITL
Mean E2EL (ms) / P99 E2EL
```

JSON saved to `--result-dir` with full per-request detail when `--save-detailed`.

### 4.5 What `vllm bench serve` CANNOT do (→ custom wrapper)

- Capture B70 hwmon energy (`energy1_input`) and temperature for real power/thermal.
- Enforce VRAM-free safety guards (debugfs `visible_avail`) before a 128K run.
- Snapshot vLLM prefix-cache + spec-decode counters synchronized to each request.
- Replay an exact multi-turn Pi/Telegram conversation (warm shared prefix →
  new RAG payload → follow-ups). It uses synthetic random or ShareGPT datasets.
- Abort on capacity boundary (KV capacity vs claimed context).

---

## 5. vLLM server-native metrics — Prometheus `/metrics`

Two roles, not one. The **discrete counters** (prefix-cache, spec-decode) are
authoritative — they are exact cumulative totals you diff across a window. The
**latency histograms** are cumulative and bucket-granular: `histogram_quantile()`
only approximates a percentile over a time window, so they corroborate trends and
monitor the server but are NOT a clean per-request decode rate. **Standard
latency evidence comes from `vllm bench serve` (§4); `/metrics` corroborates it
and owns the cache/MTP counters.** Always inspect the live endpoint first
because names shift between V0/V1:

```bash
curl -s http://<lab-host>:8001/metrics | grep '^vllm:' \
  | grep -E 'time_to_first_token|time_per_output_token|inter_token_latency|prefix_cache|spec_decode'
```

| Metric (V1) | Meaning | Use for |
| `vllm:time_to_first_token_seconds` (histogram) | TTFT incl. queueing/prefill | Corroborating/monitoring trend (bucket-approximated percentile) |
| `vllm:inter_token_latency_seconds` (histogram) | Per-token inter-arrival | Corroborating/monitoring; NOT a clean per-request rate |
| `vllm:request_time_per_output_token_seconds` | Per-request TPOT | Corroborating/monitoring |
| `vllm:e2e_request_latency_seconds` | End-to-end request latency | Corroborating/monitoring |
| `vllm:prefix_cache_queries` / `vllm:prefix_cache_hits` (counters) | Token-level cache query/hit | **Zero-delta guard for uncached prefill** (replaces the old `prefix_cache_hits_total` snapshot) |
| `vllm:spec_decode_num_drafts` / `num_draft_tokens` / `num_accepted_tokens` (counters) | Spec-decoding draft/accept totals | Overall MTP acceptance rate |
| `vllm:spec_decode_num_accepted_tokens_per_pos` (labeled by draft pos) | Acceptance per draft position | The per-position decline curve (pos0 92%→pos3 69%) |

**Critical for honest prefill:** valid uncached prefill requires
`prefix_cache_hits` delta = 0 across the measured window. This is the same rule
as protocol §1.5 Error 3, but now using the V1 token-level counter names.

**Mean accepted tokens per speculation round** (with bonus token):
`1 + accepted_tokens / num_drafts`.

---

## 6. `llama-bench` — scope and limits (llama.cpp stack only)

`llama-bench` is a llama.cpp executable. It **launches its own GGUF model** and
measures synthetic pp/tg under the SYCL engine. It is the correct tool for the
production llama.cpp stack and is already the protocol standard (§1.5, §9.4).

| Can do | Cannot do |
|---|---|
| Synthetic pp512/2048/4096/8192 prefill | Benchmark a running vLLM server |
| Synthetic tg32/128/256/512 decode | MTP/speculative decoding (no spec support) |
| A/B two llama.cpp SYCL builds side by side | Any HTTP endpoint |
| Immune to prefix-cache and quoting bugs | Server-native TTFT/TPOT/ITL percentiles |

For the current vLLM campaign, `llama-bench` is **not applicable**. For the
llama.cpp MTP paths it cannot reach, controlled HTTP prefill (with the protocol
§1.5 guards) remains valid.

llama-server (the production server) exposes `timings` in the response JSON
(`prompt_n`, `prompt_per_second`, `predicted_per_second`) — that is the
engine-native rate for the llama.cpp HTTP path.

---

## 7. LocalMaxxing — what it is and what it isn't

| Claim | True? |
|---|---|
| Useful normalized reporting surface (comparable numbers across hardware) | ✅ Yes |
| Submissions are self-reported by the operator | ✅ Yes |
| Platform "APPROVED" = independently reproduced / verified | ❌ **No** — it means accepted/published self-report |
| Its flat payload (`tokSOut`, `tokSPrefill`, `contextLength`, …) can replace the internal raw manifest | ❌ **No** — preserve separate decode/prefill observation IDs with exact coordinates, cache state, C1/Cn scope |

**Decision:** keep using LocalMaxxing as a reporting surface. Never present an
APPROVED record as independently verified. The internal manifest (per
`benchmark-protocol.md` §6) remains the evidence of record; the LocalMaxxing
JSON is a projection of selected headline numbers.

Leaderboard weights (VRAM 30 / model size 30 / run count 25 / TPS 15) reward
large-context, large-model, many-run submissions — not latency quality. Do not
optimize a production policy for the leaderboard score; optimize for real Pi
latency, then report to LocalMaxxing.

---

## 8. Where the custom B70 wrapper is still required

The custom wrapper is **not replaced** — it is demoted from primary latency tool
to the B70-specific layer the standard tools cannot cover. It must:

1. Run `vllm bench serve` (or call the OpenAI endpoint) as the latency core.
2. Snapshot `vllm:prefix_cache_*` and `vllm:spec_decode_*` counters around each
   phase, synchronized to request timestamps.
3. Sample hwmon `energy1_input` and all `temp*_input` on a 0.5–1.0 s cadence
   over the exact measured interval (protocol §6 real-power/thermal rules).
4. Read debugfs `visible_avail` after load and abort on capacity boundaries
   (<1 GiB unsafe; 1–2 GiB observation only; ≥2 GiB for staged C1; ≥3 GiB for
   mixed serving).
5. Replay exact rendered-token Pi multi-turn flows (cold → warm prefix → RAG
   append → follow-ups), not just synthetic random.
6. Separate C1 from Cn, retain all samples, and emit the §6 manifest.

This keeps the standard metrics where the standard tool owns them, and keeps
B70 telemetry/safety where only a host-side wrapper can.

---

## 9. Protocol gaps to fix in the docs phase (hands off to phase 2)

The current `benchmark-protocol.md` is llama.cpp-centric. These gaps must close
before the next vLLM campaign is considered protocol-compliant:

1. **§1.4/§1.5 tool selection is engine-blind.** Add an engine→tool decision:
   vLLM stack → `vllm bench serve` primary + `/metrics`; llama.cpp stack →
   `llama-bench` primary + HTTP for spec.
2. **§1.5 prefix-cache counter name is stale.** `prefix_cache_hits_total` → V1
   `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries` (token-level ratio).
3. **No vLLM decode standard.** Add: standard latency evidence is `vllm bench
   serve` client-observed TTFT/TPOT/ITL percentiles (these are end-to-end UX
   metrics by definition); `/metrics` latency histograms corroborate and the
   discrete counters (cache, spec) are authoritative for cache/MTP state.
4. **No TPOT-vs-ITL definition.** Add §3 of this audit as the metric glossary,
   including the "never use inclusive TPOT" rule.
5. **No real-world Pi workload matrix.** Add the 7 workload states and the
   user-facing metric list from the handoff (TTFT, TTFC, TPOT/ITL, latency,
   cache deltas, MTP acceptance, C1/Cn, power/thermal, p50/p95 under mixed load).
6. **No exact-rendered-token calibration rule.** Require
   `len(encoded["input_ids"])`, not `len(BatchEncoding)`, and shape-specific
   warmup before each distinct prompt length.

---

## 10. Sources

- vLLM serving benchmark: <https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py>, CLI docs <https://docs.vllm.ai/en/v0.11.0/cli/bench/serve.html>
- vLLM V1 metrics source: <https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/loggers.py>
- vLLM spec-decode metrics: <https://docs.vllm.ai/en/stable/api/vllm/v1/spec_decode/metrics/>
- NVIDIA GenAI-Perf / AIPerf metrics: <https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html>, <https://docs.nvidia.com/aiperf/dev/reference/ai-perf-metrics-reference>
- MLPerf Inference rules (TTFT/TPOT, p99 tail gates): <https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc>
- B70 cross-engine reality: `docs/cross-engine-comparison-2026-08-06.md`
- Existing protocol (llama.cpp-centric): `docs/benchmark-protocol.md`
