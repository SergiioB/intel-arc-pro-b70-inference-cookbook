---
title: "OpenVINO GenAI MTP long-context wall — upstream issue draft"
type: research
status: final
date: 2026-09-15
---

# OpenVINO GenAI MTP long-context wall — upstream issue draft

Draft for `openvinotoolkit/openvino.genai` (GPU-plugin triage requested).
Evidence: `results/2026/09/qw38-ov-mtp-20260914/` (u4-full sweep, mtp5 sweep,
ceiling probes, memory probes). Diagnosis reviewed against the failure ladder
2026-09-15; facts below are separated from hypotheses.

## Issue text (ready to file)

**Title:** MTP speculative decoding fails (-14 CL_EXEC) or collapses 3× at
>64K context on Xe2 (hybrid GDN model); no clean budget error before dispatch

**Environment:** openvino 2026.5.0-22967-38a575de45f, openvino.genai
2026.5.0.0-3412-8fa12145144, Intel Arc Pro B70 (BMG/Xe2, `xe` driver,
Level Zero GPU.1), VLMPipeline.

**Model:** 27B hybrid — 48 GDN (linear attention) + 16 full-attention layers,
4 KV heads, head_dim 256; export `Qwen3_5ForConditionalGeneration` (VL-style),
INT4 weights (24.7 GB resident of 32.6 GB). MTP = int4-ov draft head
(num_assistant_tokens=5). `enable_prefix_caching=false` (required for the
linear-attention verifier per mtp_strategy.cpp / pipeline_impl.cpp).

**Failure ladder (single-stream greedy, counting prompts):**

| Config | 512 | 65 536 | 98 304 | 131 072 | 196 608 / 262 144 |
|---|---:|---:|---:|---:|---:|
| MTP5 + u8 KV | 79.5 tok/s | 29.1 tok/s | **RuntimeError** `clWaitForEvents -14 CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST` (remote_tensor.cpp:77 / ocl_event.cpp:71) | — | — |
| no-MTP + u8 KV | 22.4 tok/s | **crash** (infer_request.cpp:224) | — | — | — |
| no-MTP + u4 KV, full protocol (warmup + n=1, `max_new_tokens=128`) | 22.4 tok/s | 18.6 tok/s | 17.2 tok/s | 15.9 tok/s | **`clFinish` error -5 `CL_OUT_OF_RESOURCES`** (ocl_stream.cpp:395; second length via ocl_common.hpp:62 with "any subsequent OpenCL API call may cause the application to hang") |
| no-MTP + u4 KV, light capacity probe (`max_new_tokens=40`) | 22.4 | 18.6 | 17.2 | 14.7 | 12.8 @196K / **11.3 @262K, ok** (3 812 MiB free at 262K steady state; ~33.4 GB in use during prefill on the failing full-protocol run) |
| MTP + long ctx (ceiling probe) | 3.4× vs no-MTP | — | **4.77 tok/s = 3.14× SLOWER than no-MTP (14.96)** | — | — |

The u4 full-protocol vs light-probe split at the same nominal lengths shows the
failure is decided by per-request reservation/workspace (output-length
reservation and prefill temporaries), not just resident KV — and the process
learns about it only as an OpenCL `-5` mid-prefill, never as a clean budget
error at admission.

**Reported problems:**

1. **Hard failure without a budget error.** With ~8 GB free at load, u8-KV
   requests beyond ~64–98K fail with `-14`, which reports a failed event
   dependency, not the original cause. Plain target-KV arithmetic (2×16
   layers×4 heads×256 dim = 32 768 B/token u8 → ≈3 GiB @98K) does **not**
   by itself explain exhaustion, so we suspect dynamic allocation /
   temporary-memory pressure or a shape-dependent GPU kernel fault. Expected:
   a descriptive allocation/budget exception **before** unsafe dispatch
   (load-time when the context requirement is declared, request-time
   otherwise).
2. **MTP becomes 3× slower than no-MTP at long context** on a
   linear-attention verifier (prefix caching necessarily off). Suspects:
   serial draft attention scans, multi-token verification overhead, draft-KV
   growth, state-copy cost, or reduced acceptance — not proven. Requested:
   phase instrumentation (t_draft / t_verify / t_rollback per committed
   token) and an adaptive draft-disable policy triggered on measured
   benefit, not a fixed context threshold.

**Diagnostics we can attach:** fresh-process matrix (MTP on/off × u8/u4) with
exact tokenized prompt counts, matched output lengths, acceptance stats,
live peak VRAM (debugfs `vram_mm`), `xe`/dmesg journal around the `-14`,
`ov::log::level` debug output and `ov::enable_profiling` traces on request.

## Filed

**https://github.com/openvinotoolkit/openvino.genai/issues/4483** (2026-09-15; cross-linked to #4481 cache-pool bounding). Checkbox items below remain open follow-ups.

## Filing checklist

- [ ] Attach minimal standalone repro (public GDN-hybrid model if available;
      else config + architecture JSON + hashes of the private artifact)
- [ ] Confirm behavior on latest nightly
- [ ] Link this evidence root
