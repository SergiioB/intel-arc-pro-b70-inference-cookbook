# SYCL peer-copy allreduce for `-sm tensor`: patch notes

Kept as a local patch in our own repos. Operator decision 2026-11-19 06:20 CET: nothing goes to
ggml-org/llama.cpp (no PR, issue, comment or fork push), because upstream requires disclosed, human-written
contributions. Files and apply steps: [patch/APPLY.md](APPLY.md). Full record: [REPORT.md](REPORT.md).

## Problem

On two Arc Pro B70s, upstream `-sm tensor` decodes slower than one card (RL-250: 0.86x on Nanbeige4.3-33B).
Each allreduce (two per layer) goes through pinned host memory and blocks the CPU until both cards have drained
(`ggml/src/ggml-sycl/ggml-sycl.cpp:7940-7941` in the shared-context path, `:7962-7965` otherwise, base
`78e05298f`). RL-250 measured about 275 us per call. Prefill-sized tensors (32,768 elements or more) are also
exchanged as truncated BF16 and each rank adds the other's truncated partial to its own full one
(`:8136-8156`), so the two ranks can hold different sums.

## What the patch changes

- New opt-in path, `GGML_SYCL_TP_P2P=1`. Used only when both queues are in-order Level Zero queues in one
  shared SYCL context and each device can access the other's memory.
- Per allreduce: a barrier on each queue; each card copies the peer's F32 partial into a local scratch buffer
  (`memcpy` that waits on the peer's barrier event); each card adds the scratch into its own partial after the
  peer's copy of its partial has finished. No host wait, no host memory, no BF16 truncation: both ranks
  compute `p0 + p1` in F32.
- Peer access is enabled once per device pair per process (llama-bench creates one context per test). If
  enabling fails, it logs a warning and keeps host staging.
- Growing the scratch buffer now waits for both queues first, so a pooled buffer is not reused while a queued
  add still reads it.
- F16 allreduces and the default path (switch unset) are unchanged.
- Size: `ggml-sycl.cpp` +68/-5, `docs/backend/SYCL.md` +1.

## Results

Pending: measurements queued as jobs 1060-1066 and 208-214 (Qwen3.8-27B Q4_K_M, q8_0 K/V, 40,960 context).
This section gets the measured tables (card 1 alone, stock `-sm tensor`, patched `-sm tensor`, skip-allreduce
bound; tg128, pp512, served decode at 0/8K/32K; per-call latency; perplexity; leak counts).

## Limits

- Two devices only (the SYCL comm path is N=2). Needs a shared context (`SYCL_ENABLE_DEFAULT_CONTEXTS=1`,
  Linux). Measured with the Level Zero v1 adapter (`UR_LOADER_USE_LEVEL_ZERO_V2=0`) only; v2 has a known
  multi-GPU crash in the host-copy path (Level Zero #556).
- Prefill sends F32 at every size, twice the bytes of upstream's BF16 branch.
- The path uses cross-queue event dependencies. #32058's author reports that cross-queue barriers leak Level
  Zero events on current runtimes; the exact-check job counts create/destroy calls and host RSS over 50,000
  calls to test that here.
- Measured on one host: PCIe 4.0 x8/x8, card 0 at 180 W and a 2,400 MHz clock cap (operator cooler profile),
  card 1 at 230 W.

## Related upstream work (reference only)

- [#32058](https://github.com/ggml-org/llama.cpp/pull/32058) (open): `comm_exchange_fast` V2, still
  host-staged with a host wait, as a stopgap for the event leak.
- [#30956](https://github.com/ggml-org/llama.cpp/pull/30956), [#29604](https://github.com/ggml-org/llama.cpp/pull/29604):
  the current pinned-host exchange.
- [#27550](https://github.com/ggml-org/llama.cpp/pull/27550): SYCL peer-to-peer copy API
  (`GGML_SYCL_DEV2DEV_MEMCPY`).
- [Level Zero #556](https://github.com/oneapi-src/level-zero/issues/556): event leak and v2 crash in the
  two-GPU host-copy pattern.
