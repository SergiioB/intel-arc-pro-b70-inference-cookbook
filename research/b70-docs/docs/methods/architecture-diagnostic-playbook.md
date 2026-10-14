---
title: "B70 architecture economics and deciding experiments"
type: method
status: living
date: 2026-09-15
updated: 2026-10-08
---

# Price the removable time before writing a kernel

Run `python3 tools/research.py next`. Use this method for its selected question.
A model family suggests mechanisms; it does not establish a bottleneck.
The previous method incorrectly carried retired-route conclusions into new
engines and prescribed Mamba fusion after its measured budget had been rejected.
Historical method is preserved in the reset archive.

## 1. Establish the coordinates

1. Record exact GPU/driver/runtime, source or image digest, artifact revision,
   actual packed bytes and any mixed-precision exclusions.
2. Freeze task, prompt hash/content, actual input/output tokens, sampling,
   occupied context, concurrency and cold/warm cache state.
3. Separate resident weight bytes, active weight traffic, KV traffic, recurrent
   state, workspace, launch/synchronization costs and external transfers.
4. Measure a matched baseline under graph replay if production uses graphs.
   Pair kernel attribution with whole-request or whole-task wall time.
5. Name the removable term, a realizable change and a stopping threshold.
   No attributable budget means a bounded diagnostic first, not a kernel rewrite.

## 2. B70 core and memory facts

Intel's [B70 datasheet](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2026-03/datasheet-b70-gpu.pdf)
specifies **32 Xe2-HPG cores, 256 XMX matrix engines, 32 GB memory, 608 GB/s pin
bandwidth and 367 dense INT8 TOPS**. These are product ceilings. Board power
limits vary; the local reference-card maximum is recorded as 230 W, not an
instruction to change a cap. PCIe Gen5 x16 is the card interface, not this
machine's measured connection.

Intel's [Xe architecture guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html)
lists the B580 as its Xe2-HPG example: 8 vector engines/core, 8 hardware
threads/vector engine, 512-bit registers, 128/256-register modes, 128 KiB shared
local memory (SLM)/core and subgroup sizes 16/32. Do not copy its 20-core count or
cache capacity onto B70. Combining its per-core topology with B70's 32 cores
suggests **256 vector engines and 2048 hardware threads** in regular mode
[INFERENCE]; query the actual device before using those numbers to size a launch.
Registers and SLM limit resident work; a 256-register mode can reduce occupancy.

An XMX engine is a matrix unit, not an independent CPU core. A subgroup is a set
of lanes sharing instruction execution; a work-group can contain several
subgroups. One work-group per Xe core does not guarantee residency, balance or
safe grid-wide barriers. Count registers, SLM, work-items, spills and actual
issued instructions for the pinned kernel. Preserve capture and synchronization
correctness before timing it.

367 INT8 TOPS cannot be inserted as FP16 TFLOP/s. Packed INT4/EXL3/IQ3 weights
may require unpacking, scales, transforms and BF16/FP16 matrix operations.
FP8 storage does not prove native FP8 execution. Trace actual dispatch and its
instruction/rate class. Lower bits may reduce memory bytes while making compute
slower; the retained K3 result is a local example, not a format-wide verdict.

## 3. Equations with explicit units

`python3 tools/research.py price --input PATH` evaluates these equations. Inputs
must name provenance as measured, external claim or assumption. Its output is
always a bound or prediction, never a measured inference improvement.

### Memory and compute bound

For one step, let D be transferred bytes, F arithmetic operations, B sustainable
bytes/s, P operations/s and s non-overlapped serial seconds:

```text
t_step >= max(D / B, F / P) + s
rate <= useful_tokens_per_step / t_step
arithmetic_intensity = F / D
ridge_point = P / B
```

Use decimal GB/s (10^9 bytes/s), not GiB/s or gigabits/s. Weight-file size is not
necessarily DRAM traffic; cache reuse, re-reads, scales and unpack buffers matter.
Do not add a measured serial term that already includes the transfer/compute
window. Report uncertainty instead of filling unknowns with zero.

Illustration only: 27 billion parameters at exactly 4 bits require 13.5 GB before
metadata. At 608 GB/s that is 22.20 ms, a best-case 45.04 steps/s; at an assumed
500 GB/s it is 27 ms, 37.04 steps/s. This is neither an artifact size nor a B70
benchmark. A speculative step can emit multiple tokens; price its extra work.

### Whole-task benefit (Amdahl's law)

If fraction f of measured whole-task wall time becomes r times faster:

```text
speedup = 1 / ((1 - f) + f / r)
infinite-local-speed ceiling = 1 / (1 - f)
```

If the targeted operation owns 8% of task time, even removing it entirely caps
the gain at 8.70%. A 2x kernel makes the task 4.17% faster. An 8% device-time
share is not necessarily 8% of task wall time. Prefix reuse can dominate a
prefill-heavy task while barely moving base decode.

### Speculative decoding

Use measured marginal prefix-survival probabilities p_i, where p_i is the
probability that at least i proposed tokens survive verification:

```text
yield_per_round = 1 + sum(p_i for i = 1..k)
round_ms = total_draft_ms + target_verify_ms + overhead_ms
predicted_speedup = baseline_ms_per_token * yield_per_round / round_ms
```

Marginals must be nonincreasing. They are not conditional probabilities and
must not be multiplied again. Do not replace the measured profile with alpha^i
unless independence was established. Bonus-token accounting must match the
engine. Measure changed acceptance and round time together; an accepted draft
can still be slower. Sequential draft passes reread weights unless the actual
algorithm proves reuse. Never multiply speculative, kernel and cache speedups
that target overlapping time.

### KV cache and recurrent state

```text
full_attention_KV_bytes = 2 * attention_layers * kv_heads * head_dim
                          * occupied_context_tokens * batch * bytes_per_KV_element
resident_bytes = weights + full_attention_KV + recurrent_state + workspace
```

Count only full-attention layers in a hybrid model. Recurrent state is separate,
not context-linear by default. Add quantization scales, cache block slack,
allocator reserve, speculative snapshots and fragmentation from engine evidence.
For MoE, all resident experts count toward capacity; only selected experts plus
shared weights and misses count toward a step's weight traffic. An active-parameter
count alone does not capture routing, imbalance or offload.

### PCIe, RAM and storage

```text
transfer_ms = bytes_per_step / (GB_per_second * 10^6)
              + collective_count * per_collective_latency_us / 1000
```

Add transfer to the step budget unless an aligned trace proves overlap. Subtract
proven overlap once and never beyond either participating interval. Account for
payload direction, copies per layer, host staging and startup versus per-token
traffic. This host's retained direct-peer copy probe measured about 11–12 GB/s
with peer atomics absent (RL-059); it does not validate every collective.
A second card's capacity does not guarantee lower single-request latency.

MoE offload needs miss bytes * miss frequency, effective read pattern and RAM
cache residency. RQ-15's recorded NVMe bandwidth cannot reproduce a 26 GB/s RAID
host by changing a flag. SATA placement for RQ-14 must be priced, not ignored.

## 4. Select a mechanism from evidence

| Observed budget | Smallest discriminating experiment | Stop when |
|---|---|---|
| Decode weight traffic dominates | Same-task route/format comparison with quality first | Unpack/draft overhead consumes the byte saving |
| Cold prefill dominates whole task | Matched cache OFF/ON document sequence and counters | Savings cannot meet task gate or output changes |
| Expert offload dominates | Price measured cache misses and transfer pattern | Storage/link floor misses task budget |
| Launch or host synchronization dominates | One trace-backed capture/fusion change | It moves no meaningful wall time |
| Multi-user queueing dominates | Frozen task mix at C1/C4, then C8 | Tail latency/quality fails before useful aggregate gain |

Do not prescribe Mamba fusion, TP2, MTP depth, quantization or megakernels from the
model family alone. RQ-01..04 are operator-held regardless of theoretical appeal.

## 5. Decide without moving the goalposts

Declare minimum useful relative gain before the screen. The built-in assessor
uses paired gains, requires every confirmation pair to clear the declared gate,
and distinguishes rejection from a crossing/noisy result. This is a conservative
operational gate, not a confidence interval or proof of causality. Use raw pairs
to inspect drift and ordering. No cherry-picked maxima or selective exclusions.

A clean screen buys one fresh confirmation. A confirmed candidate still needs
a task-quality review and route decision; it does not rewrite the serving
registry automatically. Diagnostic closures, correctness fixes and capacity
results are useful outcomes but remain distinct from inference speed gains.
