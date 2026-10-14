---
title: "Qwen3.8-27B base vLLM XPU MTP4 optimization — 2026-09-05"
type: research
date: 2026-09-05
status: final
---

# Qwen3.8-27B base vLLM XPU MTP4 optimization — 2026-09-05

## Objective and status
Recover the existing base GPTQ INT4 sym G128 checkpoint with native BF16 MTP4 on a single B70; then profile and test the smallest real optimization. Status: RECOVERY IN PROGRESS; no new inference or performance measurements yet. No GGUF, FP8 target, DFlash, or model substitution. No promised throughput target.

## Correction of earlier unsupported claims
The earlier 120–125+ tok/s objective and 1.5 ms/draft bottleneck were hypotheses, not measurements. “Completed” meant only sketches were written, not compiled, integrated, or validated. The fused draft code returns token zero and omits native MTP attention/MLP/state and sampling semantics. It is invalid and quarantined with fail-closed guards and byte-preserved originals. The attempted Qwopus GGUF pivot was out of scope and abandoned. Historical Run40/48 figures are separate recorded workloads, not this campaign's baseline.

## Trials and results
| Phase | Trial | Result | Next |
|---|---|---|---|
| Governance | Live host + bench intent compiler | b70-host; compiler halt=false, native base GPTQ route; dirty repository preserved. Rock receive-only copy had orphaned ledger/stubs absent from authority host. | Publish corrected quarantine to authority host. |
| Recovery inventory | Actual mounts, checkpoint, Docker, launcher and historical summary | Base checkpoint is intact at /qw38-gptq-out/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16, five real shards. No remount needed. Docker reports zero images and containers. Launcher pins f01e24f6 digest. | Recover exact pinned image, never nightly substitution. |
| Memory preflight | Live RAM/swap/disk | 15 GiB RAM, about 11 GiB available, root 59 GiB free; existing 32 GiB swapfile verified as valid Linux swap and activated, total swap now about 47 GiB. No new swap allocation/deletion. | Capped one-worker load only after image and artifact gates. |
| Image recovery | systemd bounded docker pull | b70-qw38-mtp4-image-recovery-20260905.service; RuntimeMaxSec=1800, CLI timeout=1700, client MemoryMax=2G. Docker daemon extraction is not governed by that client cap. | Inspect pull outcome, disk and daemon resource state. |
| Quarantine TDD | Guard regression | Test failed on original Python import; after fail-closed guards test passes. Original source bytes retained with SHA256 manifest. | Do not integrate sketches. |

## Planned gated measurement (not executed)
A single C1 server, native MTP4, read-only base checkpoint, single XPU visibility, isolated name/port/cache. Keep baseline and candidate identical except the one profiled change. Host worker/build caps=1, reserve at least 3 GiB host RAM and 2 GiB GPU VRAM; bound load and campaign wall time; stop on memory pressure rather than retry loops. No production edits.

First: inspect pinned source, real MTP path, existing patches and traces. Load/smoke, repeated same-process correctness and greedy token/logit reference parity before any candidate performance claim. Baseline and candidate: p512/g128 and p8192/g128 decode plus p2048/g1 and p8192/g1 input anchors; one same-shape warmup + n=3 screen, fresh n=5 only for a finalist. Cache-off startup and counter proof; separate cache-on repeated-prefix lane. Preserve all raw prompts/responses, token counts, counter snapshots, failure samples, exact hashes/configs, thermals and timestamps. Client input/TTFT is cold input rate, not isolated prefill; no numerical parity or kernel speedup until actually measured.

## Next
Finish image recovery and capped import/source inspection. Recover base MTP4 before choosing an optimization. No public publication or commits authorized. Authority: b70-host:~/B70-DOCS.

- 2026-09-05T21:22:27.512778+00:00 **WAIT_IMAGE**: Waiting for exact pinned digest; deadline 1700s; no model load yet; evidence `results/2026/09/qw38-base-mtp4-load-20260905T212227Z`.

- 2026-09-05T21:35:28.613040+00:00 **SOURCE_AUDIT**: Reading installed pinned MTP source before load; no experimental kernel; evidence `results/2026/09/qw38-base-mtp4-load-20260905T212227Z`.

- 2026-09-05T21:35:38.041376+00:00 **LOAD**: Native BF16 MTP4; base GPTQ target; 12GiB container RAM/20GiB combined; workers=1; 150W; no speed comparison; evidence `results/2026/09/qw38-base-mtp4-load-20260905T212227Z`.

- 2026-09-05T21:36:58.560092+00:00 **BLOCKED**: RuntimeError('Server exited before ready'); evidence `results/2026/09/qw38-base-mtp4-load-20260905T212227Z`.

- 2026-09-05T21:36:59.253154+00:00 **CLEANUP**: Stopped only owned container if created; preserved logs/container, production untouched; power restored if changed; evidence `results/2026/09/qw38-base-mtp4-load-20260905T212227Z`.
