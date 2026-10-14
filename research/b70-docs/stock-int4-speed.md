# Stock INT4 W4A16 speed re-measure — dense Qwen3.8-27B fast path on vLLM XPU v0.30.0

> Lane: `stock_int4_speed_20261011` | Host `b70-host` | **GPU1 only**
> (`0000:0f:00.0`, renderD129, hwmon6), cap **as-found 230 W** (never written).
> Deliverable for the assignment: *"does the historical ~1.62× W4A16-vs-EXL3
> decode advantage still exist under vLLM XPU v0.30.0?"*

## Verdict

**The 1.62× class does not hold on vLLM XPU v0.30.0.** On matched exact-512-token
prompts, C1, greedy, 128 generated tokens, usage-token decode rate:

| pairing | W4A16 (auto_gptq) | EXL3 4bpw | ratio W4A16/EXL3 |
|---|---|---|---|
| with native spec (MTP4 vs MTP3) | **71.46 tok/s** median | **79.89 tok/s** median | **0.89×** |
| without speculation | **32.85 tok/s** median | **32.53 tok/s** median | **1.01×** |

For reference (same era, different prompt sets — not cell-matched): the removed
champion recorded ~102.61 tok/s (MTP4, 230 W) vs EXL3 63.25 tok/s on the old
image, i.e. the ~1.62× class. On v0.30.0 the auto_gptq arm is ~30 % below its
old-image class value and ~11 % behind EXL3 spec-for-spec; at no-spec the two
formats are at decode parity (~1 %), consistent with both being memory-bandwidth
bound at ~4 bpw.

## Cells — usage-token decode rate

`decode_tok_s = usage.completion_tokens / (t_last_sse − t_first_sse)` —
post-first-token window per request. `usage` from `stream_options.include_usage`;
**not** SSE chunk counts (chunks recorded as diagnostic only). Sampling:
`temperature 0.0, max_tokens 128, ignore_eos=true, thinking off`. Cell =
p512/g128, C1 (`--max-num-seqs 1`), one discarded warmup + 3 measured per arm,
prompts identical across arms (exact-512 usage tokens, calibrated via /tokenize;
`evidence/prompts_exact512.json` sha256 `6cf8c2a2`).

| arm | warmup | cell 1 | cell 2 | cell 3 | median |
|---|---|---|---|---|---|
| w4a16-mtp4 | 71.60 | 71.46 | 61.55 | 72.55 | **71.46** |
| exl3-mtp3 | 85.62 | 80.09 | 79.89 | 79.62 | **79.89** |
| w4a16-nospec | 32.92 | 32.87 | 32.85 | 32.84 | **32.85** |
| exl3-nospec | 32.59 | 32.56 | 32.53 | 32.50 | **32.53** |

All cells: `prompt_tokens=512`, `completion_tokens=128`, `finish_reason=length`,
prefix-cache-hit delta = 0 (EXL3 prefix caching on-guarded: 512 queries, 0 hits
per request; W4A16 prefix caching OFF).

## Speculation counters (engine /metrics deltas)

| arm | accepted/drafted per request | acceptance | drafts per request |
|---|---|---|---|
| w4a16-mtp4 | 96–100 / 124 | **77–81 %** | ~31 |
| exl3-mtp3 | 86–89 / 123 | **70–72 %** | ~41 |
| nospec arms | spec counters absent (none emitted) | — | — |

Caveat: acceptance on these repetitive-calibrated prompts is a property of
(prompt × verifier) — the W4A16 arm's verifier is the **RTN speed-only
checkpoint**, so its acceptance need not equal a quality checkpoint's. The
nospec pair is the verifier-independent comparison; it shows kernel/format
decode parity.

## Power / energy (hwmon6, GPU1)

- Cap as-found `power1_cap=230000000 µW` at every record; never written.
- Per-request energy deltas (energy1_input): w4a16-mtp4 495–566 J,
  exl3-mtp3 432–438 J, w4a16-nospec ~970 J, exl3-nospec ~966 J per cell window.
- Measured cells avg power ≈ 230 W (all four arms); warmups 66–212 W (first
  request includes ramp).
- Package temps during cells ≈ 57–63 °C; campaign-end pkg ~70–73 °C transient.
- dmesg delta campaign start→end: **no new kernel messages, no resets/faults**.

## Artifacts and identities

- W4A16 checkpoint: `/mnt/<nvme-disk>/b70-run4-20261011/stock-int4/ckpt`
  — fresh RTN INT4 G128 symmetric, auto_gptq packing (`uint4b8`, `desc_act=false`,
  `lm_head` unquantized, `dynamic: {-:.*mtp.*: {}}`), 2399 tensors /
  400 quant quads, `ARTIFACTS.sha256` sha256 `0977ba76…` (40 files).
  **SPEED-ONLY — RTN, never a quality or serving artifact.**
- BF16 source: `/mnt/<nvme-disk>/b70-bf16-qwen38-20260926`
  (ARTIFACTS.sha256 verified; quantized tensors filtered out of physical shards —
  see findings).
- EXL3: `/mnt/<nvme-disk>/b70-exl3-qwen38-4bpw-20260926`
  (turboderp 4.00 bpw, native MTP3, fp8 KV config-of-record).
- Images: `vllm/vllm-openai-xpu:v0.30.0@sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d`,
  `ghcr.io/0xsero/exl3xpu@sha256:21412bdd7535e9c653eeb3d099dce3bc79a83d440def2cd9556111c79c870fa8`
  (image JSON captured in evidence).
- Runtime invariants: XPU graph capture ON (vllm `VLLM_XPU_ENABLE_XPU_GRAPH=1`;
  exl3 `FULL_DECODE_ONLY`), fp8 KV both, prefix caching OFF for W4A16.
- Serve-to-ready: w4a16 ~4.5 min, exl3 ~4–5 min (312 s / 252 s), all under one
  `flock /tmp/b70-gpu1.lock` hold with per-arm gpu_guard before/after.

## Cell integrity notes

- `chunks` (SSE content chunks) recorded as diagnostic: spec arms 32–39,
  nospec 118–123 (≈1 token/chunk without speculation) — **not** used as tok/s.
- EXL3 warmup `ttft=14.7 s` (first-request compile) — discarded per protocol.
- Wall rates (ct/(t_end−t_send), includes prefill) also recorded per cell in
  the raw JSONL; engine-side decode counters captured where exposed.

## Findings on the serving contract (recorded, fixed)

1. **Index-vs-file-header trap.** vLLM iterates safetensors *file headers*, not
   `model.safetensors.index.json`. A mixed BF16+quant layout that only drops the
   400 quantized `.weight` names from the index still streams the stale weights
   and crashes in `MergedColumnParallelLinear.load_weights`
   (`'MergedColumnParallelLinear' object has no attribute 'data'`). Fixed by
   physically filtering shards: index entries 2399, headers match exactly.
2. **EXL3 on GPU1** needs `--device=/dev/dri` (full dir), `-v
   /dev/dri/by-path:/dev/dri/by-path:ro` (oneCCL `ze_fd_manager` `opendir`), and
   `serve.py --gpu 1` (it sets `ZE_AFFINITY_MASK` itself — do not combine
   `ONEAPI_DEVICE_SELECTOR` with its mask).
3. v0.30.0 handles this model natively for auto_gptq: `-:.*mtp.*` dynamic
   exclusion + `XPUwNa16LinearKernel` (`int4_gemm_w4a16`), `method:"mtp"` spec
   draft from the same checkpoint — the historical patch stack is not needed.

## Evidence

`scripts/tmp/stock_int4_speed_20261011/` (client, quantizer, campaign,
diagnostics) and `/mnt/<nvme-disk>/b70-run4-20261011/stock-int4/evidence/`:
cells_*.jsonl (sha256 listed below), server/metrics/inspect logs per
arm, pm_* hwmon records, guard_*/dmesg_* files, monitor.log, image_*.json,
ARTIFACTS.sha256, campaign logs for failed-then-successful sequence.

- cells sha256: w4a16-mtp4 `7917f945`, exl3-mtp3 `9789d570`, w4a16-nospec
  `0d306462`, exl3-nospec `402fa22e`.
- Cleanup: all owned containers removed; guard clean at campaign end; GPU1
  released; no settings changed.

## Bottom line

On vLLM XPU v0.30.0 the stock auto_gptq W4A16 route for Qwen3.8-27B decodes at
~72 tok/s (MTP4, p512/g128, 230 W) — behind EXL3 4bpw's ~80 tok/s (MTP3) at
0.89×, and at no-spec parity (~32.7 tok/s both). The historical ~1.62× decode
advantage is **not** present on the current image.
