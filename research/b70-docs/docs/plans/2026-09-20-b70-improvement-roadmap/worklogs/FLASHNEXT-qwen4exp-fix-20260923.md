---
title: "Flash-Next qwen4exp load blocker — arch support found, real blocker is a oneAPI runtime mismatch"
type: plan
date: 2026-09-23
status: blocked-on-gpu-verification
---

# Flash-Next qwen4exp fix — 2026-09-23

> PROVISIONAL — NOT FOR PUBLIC HEADLINE
> No benchmark was completed. No correctness check or screen has run yet. GPU
> timeline: five load ATTEMPTS ran before the coordinator GPU hold arrived
> (all crashed instantly, §2); after the hold, zero GPU work was run from
> this lane, and the rebuilt 2026.1.1 server (§3) is CPU-built only and has
> NOT been GPU-loaded.
>
> Companion report: `FLASHNEXT-cascadia-pp-20260923.md` — Cascadia pipeline
> parallelism is UNSUPPORTED for qwen4exp (no sharding source, no arch
> support, no GenAI model class); this llama.cpp dual-card route is the only
> Flash-Next split path on this host.

## 1. The reported blocker and what it actually was

`/mnt/<nvme-disk>/models/signal-3.8-flash-next/smoke-server.log`
(2026-09-20) failed with `unknown model architecture: 'qwen4exp'`. That log came
from a binary WITHOUT the arch. The arch itself is not the blocker:

- **Upstream provenance:** qwen4exp was added to llama.cpp by commit
  `6c84c7d` — "model: add Qwen3.8-Flash-Next (qwen4exp) (#27742)" (found via
  `git log -S qwen4exp` inside the local trees; `src/llama-arch.cpp`,
  `src/models/qwen4exp.cpp`, `conversion/qwen4exp.py`).
- `~/llama.cpp` (pinned `1692f9e50`-dirty, the fused-gemm IQ3 tree)
  does **not** contain it (grep over `src/` and `gguf-py/` is empty) — that is
  the tree class the 2026-09-20 smoke most likely used.
- The campaign tree `/mnt/<sata-disk>/qwenflash-llama-pinned-9723942`
  (upstream `9723942` + uncommitted Flash-Next overlay on
  `src/llama-arch*.cpp`, `src/models/qwen4exp.cpp`, `llama-context`,
  `speculative.cpp`, SYCL mmvq, server) **supports qwen4exp**, and its existing
  binaries already have it:
  - `build-sycl/bin/llama-server` sha256 `f56a838b0e1a16aa8a1321ce2a217c8f3e972e961cf0aca706f76f4d6421c98b`
    — byte-identical to the cookbook's campaign decode (fused FP32) binary,
    built 2026-09-01 08:58, AFTER every overlay source mtime (latest
    2026-09-01 00:53).
  - `build-sycl-f16/bin/llama-server` sha256 `2da6e9fc89d9cb68368feb1a2e7ef1045ff705acb16721755ac1f72149dd1ee3`.
- Decision per priority (a): no cherry-pick needed; load-test with the existing
  campaign binary (rebuild only if it fails — it did fail, see §2, so a rebuild
  is now the fix candidate).

## 2. The real blocker: instant DEVICE_LOST on every launch attempt (5/5)

Target artifact:
`/mnt/<nvme-disk>/models/signal-3.8-flash-next/hf-dl/AP-Q4_K_XL/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf`
(94.2 GiB, arch `qwen4exp`, "Signal 3.8 Flash Next", 48 blocks, 512 experts /
10 used).

Composition (from GGUF metadata): experts `ffn_gate_exps` Q4_K 21.09 GiB +
`ffn_up_exps` Q4_K 21.09 GiB + `ffn_down_exps` IQ4_NL 21.09 GiB;
`per_layer_token_embd.weight` IQ4_NL **26.82 GiB** (the CPU-offloadable PLE);
attention/hybrid/`hc_*`/`indexer`/`ssm_*`/`ple_*` + output ≈ 4.2 GiB. Hybrid
SSM+MoE+PLE+indexer architecture — not a plain decoder.

All launches used the cookbook recipe (dual card, `--device SYCL0,SYCL1
--split-mode layer -ngl 99 -fa on -ctk q8_0 -ctv q4_1 -c 8192 -b/-ub 512 -t 6
-tb 14 --no-warmup --no-cache-prompt --jinja`, campaign env caps). Evidence:
`/mnt/<nvme-disk>/models/signal-3.8-flash-next/loadtest-20260923/`
(`server-load1.log` … `server-load5-m64class.log`).

| # | Variation vs cookbook | Result |
|---|---|---|
| 1 | PLE override accidentally dropped (multiple `-ot` deprecated — last wins) | DEVICE_LOST <1 s |
| 2 | `-ot "per_layer_token_embd=CPU,blk\.[0-3]\.ffn_(up\|gate\|down)_exps\.weight=CPU"` (correct comma syntax, verified in `common/arg.cpp` `string_split(value,',')`) | DEVICE_LOST <1 s, identical |
| 3 | blk 0-5 experts to CPU (GPU-side ≈ 59.6 GiB, ~1.4 GiB free/card) + `-ts 49,51` (GPU0 has ~0.9 GiB less avail — desktop) | DEVICE_LOST <1 s, identical |
| 4 | + `GGML_SYCL_ENABLE_OPT=0` (disables the SYCL reorder-extra path entirely) | DEVICE_LOST <1 s, identical |
| 5 | blk 0-11 experts to CPU (≈ 26 GiB/card — the memory class the M64 campaign ran at) | DEVICE_LOST <1 s, identical |

Failure signature every time:
`ggml_backend_sycl_buffer_init_tensor` → memset of padded region →
`UR_RESULT_ERROR_DEVICE_LOST`, before any weight bytes are read; dmesg gains
one `xe 0000:0f:00.0: [drm] VM worker error: -12` (ENOMEM) per attempt. GPUs
return clean after each crash (31705/32615 MiB `visible_avail`).

**Correction added 2026-09-23 after a source-mirrored GGUF inventory:** the
previous “NOT a placement/capacity problem” conclusion was unsupported. Under
the recorded `-ts 49,51`, llama.cpp's exact source split assigns blocks 0–24
to GPU0 and blocks 25–47 plus output to GPU1. The GGUF header has 1.3184 GiB
of expert tensors per block; before PLE, KV, and runtime buffers, baseline
weight use is about 34.59 GiB on GPU0 and 32.28 GiB on GPU1, exceeding both
31.85-GiB visible card budgets. Earlier CPU overrides offloaded blocks only
from the GPU0 side, so the failed attempts did not falsify a placement cause.

The oneAPI 2026.0→2026.1 mismatch remained a plausible but unproven hypothesis
at this stage. The matched 2026.1.1 rebuild was later loaded with the old
PLE/early-block placement and also failed; see §8. The GPU1 VM worker error
therefore cannot be attributed to runtime drift alone.

## 3. Fix in progress (CPU-only, no GPU needed to build)

Rebuild from the pinned tree into
`/mnt/<sata-disk>/qwenflash-llama-pinned-9723942/build-qwen4exp` with
the 2026.1 toolchain (icx/icpx, `-DGGML_SYCL=ON -DGGML_SYCL_F16=ON`, Release).
Driver log: `loadtest-20260923/rebuild-2026.1-driver.log`; build log
`build-qwen4exp/build-qwen4exp.log`. Fallback if the rebuild still
DEVICE_LOSTs: pin `LD_LIBRARY_PATH` to the 2026.0 compiler libs for the Sep-1
binary, or rebuild with 2026.0 explicitly.

**BUILD DONE 2026-09-23 (CPU-only, no GPU touched):**
`build-qwen4exp/bin/llama-server` sha256
`7c2d6adc4ac5dfa6f1deb8db8d545d8e4cae390ed9c1cd255d1e7f71a83e84a1`,
icx 2026.1.1 (2026.1.1.20260724) — build/runtime versions now MATCH, so the
mismatch class of failure is eliminated for this binary by construction.
First GPU load test deliberately NOT run: GPUs were held for the concurrent
Qwen3.8-27B Cascadia PP smoke (coordinator hold). Run §4 recipe when cleared.

## 4. Corrected placement discriminator (measured load/smoke; speed not measured)

A header-only reader (`GGUFReader`, read-only mmap) inventoried all 1224 tensors
in the AP-Q4_K_XL artifact and reproduced llama.cpp's `get_layer_buft_list`
split from `src/llama-model.cpp:1467-1492`. With `-ts 49,51`, the source assigns
blocks 0–24 to GPU0, blocks 25–47 and output to GPU1. Expert bytes are exactly
1.3184 GiB per block; PLE is 26.8222 GiB. Including non-expert layer/global
weights, the original PLE-on-CPU layout puts ~34.59 GiB on GPU0 and ~32.28 GiB
on GPU1 before KV/compute allocations. This explains why offloading only early
experts cannot fit this split.

The single changed variable is placement: keep PLE on CPU and offload the first
six GPU0-side and last six GPU1-side expert sets. Header/source arithmetic
estimates 26.68 GiB GPU0 / 24.38 GiB GPU1 before KV and compute buffers. The
one-shot load-only probe and separate bounded correctness smoke **PASSED**:

- `results/2026/09/flashnext-ap-placement-smoke-20260923T113433Z/` — server
  reached `/health` in ~90 s; post-load free VRAM 3372/7050 MiB.
- `results/2026/09/flashnext-ap-placement-correctness-20260923T114633Z/` —
  exact-token `B70-FLASH-OK`, arithmetic `703`, and city `Paris` all passed;
  no device-loss/OOM/segfault signatures in the server log. Cleanup left
  32051/32615 MiB free and no inference process/container.

This corrects the earlier attempt sequence, not the failure records: the five
old-binary variants and the matched-runtime attempt all used PLE/early-side
placements that overfilled at least one card. Corrected two-sided placement is
load-feasible. This is **not** a decode/prefill measurement, broad quality test,
MTP test, or speedup over the cookbook. Both cards reached 57–58 C after the
smoke, so the next performance cell remains gated on a new <=55 C cooldown and
all live empty-GPU/cap checks. The single-shot harness is
`scripts/tmp/flashnext-ap-placement-correctness-20260923.sh`.

## 5. Fused-kernel transfer verdict

- The P1 fused-gemm work (`~/llama.cpp` `fg_stage_a<block_iq3_xxs>`)
  does **not** transfer to this artifact: the GGUF has **zero IQ3-family
  tensors** (experts are Q4_K + IQ4_NL). P1 was withdrawn anyway (2026-09-21
  gate FAIL).
- The relevant fused-class path here is the pinned tree's own SYCL
  reorder/MMVQ overlay: `ggml_sycl_supports_reorder_mmvq` covers **Q4_K**
  (both gate/up expert tensors) and the K-quant family; `IQ4_NL` (down) is not
  in the reorder lists and rides the classic DMMV path. Toggle for an A/B
  exists: `GGML_SYCL_ENABLE_OPT=0` (verified present in this build).
- Expected effect is far below the K216 case (fused coverage ≈ 2/3 of expert
  bytes vs all of them on M64) — screen first, gate at ≥15 % e2e as usual.

## 6. MTP drafters (for the later decode cell)

Both draft GGUFs are arch `qwen4exp`, 49 blocks: `hf-dl/Qwen3.8-Flash-Next-MTP-Q8_0.gguf`
(4.1 GB) and `/mnt/<nvme-disk>/mtp-draft/Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf`
(2.5 GB). The overlay modifies `common/speculative.cpp`, so pairing is plausible
once the main model loads; VRAM headroom decides Q8_0 vs Q4_K_M.

## 7. Next gates

1. **COMPLETED 2026-09-23:** exact per-card split and tensor-byte audit; one
   corrected placement load-only probe; separate fixed correctness trio. See §9
   and `results/2026/09/flashnext-ap-placement-{smoke-20260923T113433Z,correctness-20260923T114633Z}/`.
2. **Performance remains unmeasured.** Next: after both cards are at <=55 C
   and empty process/container/VRAM/cap gates pass, run a narrow C1 p512/g128
   decode plus p8192/g1 cold-input screen using the same two-sided placement,
   cache guards, one discarded same-shape warmup, telemetry, and protocol stop
   rules. No broad sweep or MTP pairing before this screen.
3. If the corrected-placement launch ever again fails with both per-card
   predicted loads below budget and >=2 GiB reserve, capture the exact failing
   tensor/allocation before considering an upstream SYCL/Level-Zero report.
4. After any future GPU campaign, verify cleanup, preserve found per-card caps,
   and rerun process/VRAM gates.

Nothing here is citable as a speed claim. PROVISIONAL — NOT FOR PUBLIC HEADLINE.

## 8. 2026.1.1 GPU retest (2026-09-23) — FAILED under early-only placement

After explicit scope-owner clearance, the freshly rebuilt `build-qwen4exp/bin/llama-server`
(sha256 recorded in `loadtest-20260923/rebuild-2026.1-gpu-test/manifest.txt`, icx/icpx 2026.1.1)
was GPU-loaded once with §4 placement (`-ot per_layer_token_embd=CPU,blk.[0-5].ffn_(up|gate|down)_exps.weight=CPU`,
`-ts 49,51`, PLE present). It failed at the same point as the Sep-1/2026.0 binary:

- `common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort`
- then `level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)` at
  `ggml_backend_sycl_buffer_init_tensor` memset (`ggml-sycl.cpp:621`) — before model weights are read.
- Cleanup complete; both B70s returned to 31,651 / 32,615 MiB `visible_avail`, caps unchanged.

**Conclusion correction at that time:** matching the build to oneAPI 2026.1.1 did not fix the then-tested PLE/early-expert placement. Five old-binary failures plus this early-only matched-runtime failure remain valid historical samples. They did not test the later two-ended, per-card-corrected placement in §4/§9.

### 9. Corrected two-sided placement discriminator (2026-09-23)

A later header-only GGUF inventory and source replay of the 49/51 split found
that the prior placement underfilled GPU0 and overfilled GPU1; after including
all block bytes and output weights, it in fact exceeded each visible card's
budget before runtime buffers. The prior “not a placement/capacity problem”
and “six of six under the same placement class” interpretations are superseded
for placement by RL-055. The precise per-layer inventory and arithmetic are
in [`research/2026/qwen-flashnext-placement-and-kernel-screen-20260923.md`](../../../../research/2026/qwen-flashnext-placement-and-kernel-screen-20260923.md).

The 2026.1.1 matched binary loaded successfully with PLE and expert tensors in
blocks 0–5 and 42–47 assigned to CPU; a separate three-prompt smoke returned
B70-FLASH-OK, 703, and Paris. Raw evidence:

- `results/2026/09/flashnext-ap-placement-smoke-20260923T113433Z/`
- `results/2026/09/flashnext-ap-placement-correctness-20260923T114633Z/`

**Updated scope:** load-feasibility and bounded smoke coherence passed for
that corrected placement; decode, prefill, MTP, and broad task quality are not
measured. No upstream allocation issue is supported by the six overfilled
attempts. A narrow performance screen is next only after both cards cool to
<=55 C and all empty-GPU/reserve gates pass. Keep all six original failures as
historical evidence; do not rewrite their logs or claim they used the corrected
placement.
