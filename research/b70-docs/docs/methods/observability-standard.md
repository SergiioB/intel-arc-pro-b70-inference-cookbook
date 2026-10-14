---
title: "B70 Benchmark Observability Standard"
status: living
type: method
date: 2026-09-15
---

---
id: b70-method-observability-standard
title: B70 Benchmark Observability Standard
document_type: method
status: canonical
version: "1.0"
effective_date: 2026-08-19
last_reviewed: 2026-08-19
authority: docs/methods/benchmark-protocol.md
scope: Runtime visibility, telemetry, counters, safety gates, and monitor artifacts for B70 campaigns
---

# B70 Benchmark Observability Standard

## Purpose

This standard defines the minimum visibility required to run, diagnose, validate, and publish B70 inference campaigns. An unobservable campaign cannot establish that the GPU was isolated, the intended phase ran, cache state was controlled, telemetry covered the measured interval, or cleanup completed.

## Principles

1. Observe the exact workload interval, not a convenient surrounding window.
2. Keep configured limits separate from measured behavior.
3. Use authoritative sources for each metric.
4. Preserve timestamped raw telemetry and cumulative-counter snapshots.
5. Fail closed when a required source is unavailable or unverified.
6. Run under a visible controlling process with declared progress and timeouts.

## Preflight visibility

Before every model load, retain or log the outcome of:

- no `llama-server`, `llama-bench`, `vllm`, or `vllm-xpu` process;
- no running inference container;
- empty-card `visible_avail` approximately 31,000 MiB or more;
- each card's pre-campaign observed `power1_cap` recorded (restore as found at cleanup; the 150 W default is retired 2026-09-14); and
- one declared harness and one engine instance.

If any check fails or cannot be observed, do not start. Diagnose the state first.

## Progress and control

The campaign controller must emit a timestamped milestone:

- before and after server load;
- after readiness and post-load VRAM checks;
- before and after each shape warmup;
- before and after each measured cell or workload phase;
- when changing scheduler, cache, speculation, concurrency, or power state;
- on retry, timeout, exclusion, interruption, and cleanup; and
- after restoring the 150 W cap and stopping the campaign engine.

Declare per-cell and campaign timeouts. Do not leave an unobservable detached campaign after OOM, stall, or driver error.

## Source map

| Observation | Authoritative source | Required treatment |
|---|---|---|
| llama.cpp synthetic `ppN`/`tgN` | Raw `llama-bench` repetition output | Retain every repetition |
| llama.cpp HTTP C1 engine decode | Response `timings` token count and predicted duration/rate | Report client streaming separately |
| vLLM TTFT/TPOT/ITL/E2EL | `vllm bench serve` client-observed output | Retain detailed per-request output |
| vLLM latency histograms | `/metrics` histograms | Corroborating/monitoring only |
| Prefix-cache state | Cumulative engine counters when available, diffed across the measured window | Uncached HTTP prefill requires zero hit delta |
| Speculative acceptance | Cumulative spec-decode counters diffed across the measured window | Acceptance is diagnostic, not the optimization objective |
| Cn aggregate generation | Successful generated tokens divided by monotonic campaign duration | Never sum per-request rates |
| VRAM | Verified B70 debugfs `visible_avail` and related fields | Record pre-load and post-load reserve |
| Configured power | Verified `power1_cap` | Label `configured_cap_W` |
| Average card draw | `energy1_input` delta over the exact interval | Use monotonic elapsed time |
| Peak power behavior | Fixed-cadence energy deltas | Label maximum `peak interval-average W` |
| Temperature | Verified `temp*_label`/`temp*_input` pairs | Record start, end, and maximum per named sensor |

Do not mix these sources in one unlabeled column.

## Timebase and synchronization

Use a monotonic clock for workload duration and local interval calculations. Telemetry records include a monotonic timestamp and, when useful for correlation, a UTC timestamp. Snapshot cumulative cache and speculation counters immediately before and after the measured phase.

For phase-specific power or temperature claims, align monitor samples to request-phase timestamps or run isolated phases. Unsynchronized campaign averages cannot establish prefill-versus-decode behavior.

## Power and temperature

Discover the B70 hwmon node from `/sys/class/hwmon/hwmon*/name` and verify its PCI mapping. Do not hard-code a historical `hwmon4` path in new portable evidence.

Compute average card draw as defined by the protocol:

```text
average_W = (energy_end_uJ - energy_start_uJ) / elapsed_seconds / 1,000,000
```

Retain the raw energy values, elapsed time, resolved path, configured cap, and sampler cadence. If the counter decreases or wraps without a documented correction, invalidate the power measurement.

Discover all temperature labels on the same verified device. Preserve at least 0.1 °C in raw evidence. If a sensor cannot be verified, report `not measured`; do not substitute a remembered value.

## VRAM and capacity guards

Record empty-card availability and post-load reserve. Apply the protocol thresholds:

- below 500 MiB: abort;
- 500–1,023 MiB: isolated C1 capacity observation only;
- 1,024–2,047 MiB: staged C1 research;
- at least 2,048 MiB: preferred C1 performance work; and
- at least 3,072 MiB: target for mixed or concurrent serving.

These are operational gates, not proof of successful context completion. Capacity still requires a real prompt-plus-output request.

## Cache observability

Record one protocol cache state: `cold_model_cold_prefix`, `warm_model_cold_prefix`, `warm_model_warm_prefix`, or `mixed_or_unknown`.

For uncached HTTP input work:

- place unique entropy at the beginning while preserving target length;
- record actual endpoint prompt tokens;
- snapshot cache queries and hits before and after;
- require zero hit delta for a valid cold-prefix phase; and
- investigate unexplained repetition acceleration.

To prove vLLM cache-off, retain startup evidence showing prefix caching disabled and run a repeated/resident-prefix control that still produces zero hits. A unique prompt with zero hits cannot prove cache-off.

## Failure and exclusion observability

For every request or cell, record completion, error, timeout, cancellation, early EOS, partial output, contamination, throttling indication, and retry relationship. Failed or excluded observations remain in the evidence trail with reasons. A missing log is `unknown`, not a successful run.

## Minimum monitor artifact

A campaign monitor should make these fields reconstructable:

- run ID and protocol version;
- monotonic and UTC timestamps;
- phase/cell and request identifiers;
- process/container and server readiness state;
- configured cap, energy counter, and labeled temperatures;
- VRAM availability;
- cache/spec counter snapshots;
- actual input/output token counts;
- request status and error;
- start/end boundaries for the measured interval; and
- cleanup outcome.

The manifest links the monitor artifact and states cadence, sensor domains, timing source, and any gaps.

## Publication limits

If observability coverage is partial, narrow the claim. Examples:

- configured cap without energy data supports only “at a configured cap,” not measured efficiency;
- campaign-wide power with unmatched phase coverage does not support a mode-to-mode power comparison;
- absent cache counters makes uncached HTTP prefill unpublishable unless the engine-specific protocol provides another sufficient guard;
- Cn without achieved concurrency and failures is not a serving-capacity result.

## Related authorities

- [B70 Benchmark Protocol](benchmark-protocol.md)
- [Evidence Lifecycle](../governance/evidence-lifecycle.md)
- [Hardware and Software Stack](../reference/hardware-software-stack.md)
