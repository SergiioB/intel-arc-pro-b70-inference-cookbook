# Reference: memory bandwidth vs FLOPS in inference

**Source:** [Hikari article on X](https://x.com/Hikari_07_jp/status/2087188782827327596)  
**Retrieved:** 2026-08-11  
**Purpose:** A framework for reading concurrent inference measurements. This is not B70 evidence.

## What the article measured

One RTX PRO 6000 Blackwell Workstation Edition with 96 GB VRAM ran Qwen3.6-35B-A3B in NVFP4 through vLLM 0.26.0. It swept 1, 2, 4, 8, 16, and 32 concurrent streams across held contexts from 0K to 124K. Each cell generated 256 tokens once. The article labels this as a shape pass, not a median result.

At 0K held context, aggregate output rose from about 279 tok/s at C1 to 3,135 tok/s at C32. Per-stream speed fell from about 280 to 98 tok/s. At 124K and C32, per-stream output was about 26 tok/s and aggregate output was about 27% of the 0K C32 result.

## Technical reading

- Decode repeatedly reads weights. Short-context batching can share that traffic, so aggregate output rises while per-stream output falls.
- Higher concurrency does not multiply throughput linearly. It adds scheduler, memory-bandwidth, and compute contention.
- Long resident context adds prompt and state work that does not batch as efficiently. The article reports that its 16-to-32 stream aggregate gain nearly disappears at 124K.
- A single C1 tok/s number cannot distinguish bandwidth limits from compute, cache, or scheduler limits. Measure C1 and Cn separately.

## B70 use

Use this article as the interpretation rule for the B70 Nemotron campaign:

1. Report C1 decode separately from aggregate concurrency.
2. Test short-context serving and long-held-context serving as separate workloads.
3. Record actual prompt/output tokens, KV type, cache state, configured cap, measured draw, and VRAM reserve.
4. Do not transfer RTX PRO 6000 NVFP4 numbers or its concurrency curve to the Arc Pro B70. The GPU, engine, model format, memory capacity, and workload differ.

## Related B70 work

- `results/2026/08/nemotron-lightning-status-20260811T1915/STATUS.md`
- `docs/methods/benchmark-protocol.md` §§1.9, 1.10, 9, and 12
