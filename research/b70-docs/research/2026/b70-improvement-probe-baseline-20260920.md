---
title: "B70 improvement probe baseline — P0 (2026-09-20)"
type: research
date: 2026-09-20
status: provisional
---

# B70 improvement probe baseline — P0 (2026-09-20)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE**

P0 of `docs/plans/2026-09-20-b70-improvement-roadmap/`. Ledger **RL-030**. Resource window:
GPU 1 (`0000:0f:00.0`, hwmon6, 180 W cap) as the probe card; GPU 0 carried desktop work and
was not used for loads. No model load occurred — two of the three route artifacts are
**absent from the host** (below). Evidence root: `results/2026/09/b70-improvement-probes-20260920/`.
Worklogs: `docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/P0-20260920.md`,
`…/P0-inventory-20260920.md`, `…/P0-publication-inventory-20260920.md`.

## 1. Route baselines

| Route | Engine identity | Artifact state | Launchable |
|---|---|---|---|
| K216 (DeepSeek-V4-Flash-0731 REAP-K216 UD-IQ3_XXS) | llama.cpp `1692f9e50`, build `build-sycl-latest-qw38`, **tree dirty: 8 modified + 3 untracked files** (the IQ3 device-ID patch + a SYCL moe-cache v1 port) | 83 GiB GGUF **ABSENT** host-wide, no HF cache entry | **no** |
| Nemotron-3.5-Lightning-30B-A3B | `vllm-openai-xpu:1da0a954-det0123` (tag only — **no digest recorded**), vLLM 0.26.1rc1.dev668 | GPTQ-INT4 G64 target **ABSENT**; DFlash drafter 1.6 GB present | **no** |
| Qwen3.8-27B GPTQ-INT4 G128 + BF16 MTP4 | vLLM XPU | target 19 GB + FP8 29 GB present | **yes** |

Best recorded cells for re-association (not re-measured here): K216 L5 `-ncmoe 15 -ts 65,35`
= 8.87 engine tok/s (n=3) — **SUPERSEDED 2026-09-21 during this same phase: 9.25 clean / 9.58 patched** engine tok/s on the re-downloaded artifact; Nemotron p2048/g128 = 186.61 client post-first (n=5, 150 W);
Qwen3.8 p512/g128 = 102.61 (n=5, recommended sampling).

## 2. The decisive P0 result: K216 routed-IQ3 operator A/B

### 2.1 What the source actually says

The recorded framing ("IQ3_XXS is omitted from the fused SYCL `MUL_MAT_ID` MMVQ path") is
**true of upstream `1692f9e50`** and **false of the local working tree**:

- `ggml_sycl_mul_mat_id()` (`ggml/src/ggml-sycl/ggml-sycl.cpp:4770`) tries
  `ggml_sycl_mul_mat_id_mmvq_fused()` when `ne12 == 1`, else falls through to
  `stream->memcpy(ids_host, ids_dev, …)` + **`stream->wait()`** + a host per-expert loop.
- The fused path (`:4669`) requires F32 `src1`/`dst`, contiguous `src1`, `ids->ne[1] == 1`,
  `ne11 ∈ {1, n_ids_per_group}`; it passes **device-resident `ids_dev`** into
  `ggml_sycl_mul_mat_vec_q_id()` (`mmvq.cpp:2569`).
- **Uncommitted local work (2026-08-26) adds `case GGML_TYPE_IQ3_XXS` and
  `case GGML_TYPE_IQ3_S` to that switch** (`mmvq.cpp:2648`, `:2673`) — i.e. P1's repair
  already exists in the tree. Its own note records "built, not measured; smoke not run".
- **Graph capture cause:** `check_graph_compatibility()` disables SYCL graphs for
  `GGML_OP_MUL_MAT_ID` *because of that host wait* (`ggml-sycl.cpp:5687-5692`) — separately
  from the `device_count > 1` rule. The repair is a **prerequisite** for graph capture on
  this path, not a parallel lever.

### 2.2 Shape-matched A/B (three repetitions each)

Clean export of `1692f9e50` (`~/llama-clean-1692f9e50`) vs the patched pinned
tree. Both built with icpx 2026.0, Release, `GGML_SYCL_F16=ON`, `GGML_SYCL_TARGET=INTEL`;
both given identical added `MUL_MAT_ID` cases so the comparison is shape-matched. Device:
SYCL0 under `ZE_AFFINITY_MASK=1`. Operator shapes are K216's routed experts: 216 experts,
top-6, hidden 4096, expert FFN 2048 (3·h·f ≈ 25.2 M weights/expert).

| Case (m,n,k) | clean: host-ID fallback, median | patched: device-ID fused, median | Δ |
|---|---|---|---|
| gate/up decode `2048,1,4096` | 207.48 µs | 174.91 µs | **−32.57 µs (−15.7 %)** |
| down decode `4096,1,2048` | 201.88 µs | 168.93 µs | **−32.95 µs (−16.3 %)** |
| gate/up `2048,32,4096` | 6773.66 µs | 6765.82 µs | −7.8 µs (−0.12 %) |
| down `4096,32,2048` | 6912.71 µs | 6887.69 µs | −25.0 µs (−0.36 %) |

Both builds pass operator parity against the CPU reference at all four shapes (nmse ≤ 5e-4):
`test-backend-ops -b SYCL0 -o MUL_MAT_ID -p n_mats=216`. Raw: `operator-ab-summary.json`,
`perf-k216-{clean-fallback,patched-fused}.txt`, `parity-k216-{clean-fallback,patched-*}.txt`.

### 2.3 What that is worth per token — and what it is not

- GPU-routed routed ops per token = (43 − 15) layers × 3 projections = **84** (L5 `-ncmoe 15`).
- Saving = 28 × (2 × 32.57 + 32.95) = **2.75 ms/token** on a 112.74 ms token = **2.44 %**
  → 8.87 → **≈9.09 tok/s (≈1.025×)**.
- **The P1 20 tok/s aspiration is refuted**: it requires 62.74 ms/token; the measured
  removable time is 4.4 % of that.
- **The P1 engineering gate (≥10 %) is not met by the device-ID scope alone.**
- Batch (n=32) is unchanged — expected, since the fused gate is `ne12 == 1`. This repair is
  decode-only and says nothing about prefill (P3).
- Caveat recorded, not hidden: this is an **isolated-operator lower bound**. The fallback's
  `stream->wait()` is a device-wide sync whose cost inside a loaded, multi-op graph could
  exceed the isolated number. Confirming that requires the K216 GGUF, which is **absent**.

### 2.4 The larger, previously unmeasured fact in the same operator

Per-op weight traffic = 6 experts × 4096 × 2048 × 3.0625 bpw = **19.27 MB** → **31.7 µs** at
the single-card 608 GB/s class. Measured fused decode time is **174.9 µs = 5.52× off the
weight-read floor**. Per token the 84 GPU-routed ops read ≈1.62 GB of expert weights; at the
bandwidth floor that is ≈2.7 ms, but they actually cost ≈14.7 ms — a **~12 ms/token (≈10.5 %)**
gap that is *kernel efficiency*, not host synchronization.

The structural explanation is visible in the code: the non-fused path materializes each
expert as **FP16** before oneMKL (6 × 4096 × 2048 × 2 B ≈ 100 MB per op, 5.2× the IQ3 bytes).
So the real K216 lever is an IQ3 grouped/tile-local path that never materializes FP16 —
which is exactly P3's thesis, not P1's.

**P1 verdict (recorded, evidence-backed):** the repair is correct and should be kept (it also
unblocks graph capture), but its measured benefit is ≈1.025×, so the phase's headline target
is withdrawn; the K216 speed lever measured here belongs to P3/P5 (kernel efficiency), and
end-to-end confirmation of either is **blocked on the absent 83 GiB artifact**.

## 3. Probe status of the P0 batch

| Probe | Result |
|---|---|
| Graph capture vs compilation | **Answered (two engines, two answers).** (a) vLLM XPU: real capture/replay exists — `vllm/v1/worker/xpu_model_runner.py:61` aliases `torch.cuda.CUDAGraph = torch.xpu.XPUGraph`, so the same `CUDAGraphWrapper` runs graph capture on XPU, gated by `VLLM_XPU_ENABLE_XPU_GRAPH=1` and `supports_xpu_graph()` = torch ≥ `2.11.0.dev` (`vllm/utils/torch_utils.py:1018`); the local image carries **torch 2.13.0+xpu**, so it is available and **is not the same thing as `torch.compile`**. (b) llama.cpp SYCL: `GGML_SYCL_ENABLE_GRAPH` is off by default and graphs are refused both for `device_count > 1` and for any graph containing `GGML_OP_MUL_MAT_ID` (`ggml-sycl.cpp:5687`), i.e. the K216 dual-card path cannot capture until the host wait is gone. |
| Level Zero immediate lists | Not started — the routing trace it serves needs the absent K216 artifact. |
| Weight-layout inspection | Partially answered: IQ3_XXS expert GEMV reads one 98-byte IQ3 block per four lanes; the non-fused path instead materializes each expert as FP16 (≈100 MB/op, 5.2× the IQ3 bytes). |
| Per-expert-M distribution | Analytic answer: prefill ub=1024 ⇒ 1024 × 6 / 216 ≈ **28 rows per expert** — still small-M, so per-expert GEMM tiles stay small and the FP16 expansion dominates. |
| Dispatch separation | **Answered**: `ne12 == 1` (decode) takes the fused device-ID path; `ne12 > 1` takes the host fallback. The repair touches only the former, which the n=32 cells confirm (Δ ≤ 0.4 %). |
| Register/SLM accounting | **Answered (static)**: the routed IQ3_XXS GEMV is register-only — **no SLM staging**; `GGML_SYCL_MMV_Y = 1` so a 32-thread work-group computes **one row**, 8 blocks per row-iteration, and the inner cost is `vec_dot_iq3_xxs_q8_1`'s `iq3xxs_grid`/`ksigns64` table lookups. So the 5.52× gap to the bandwidth floor is **3-bit table-lookup dequantization on vector engines**, not spills or staging — a fix requires a different reconstruction schedule (P3/P5), not occupancy tuning. |
| Pure W4A4/W2A2 DPAS | **Answered — 2.0× per instruction.** `joint_matrix` cannot express sub-byte operands at all on this toolchain (its enum stops at 8-bit); the ESIMD `xmx::dpas` path can, and at identical shape (M=4, N=16, nacc=1) and chain depth (`reps=1`, nsub=1024) **s4×s4 = 33.91 TOPS vs s8×s8 = 16.96 = exactly 2.00×**, matching the `K_instr` 32 → 64 prediction at unchanged per-instruction cost. All absolute values are floors (the harness is issue-limited — its best verified bf16 cell is 58.6 vs the recorded ~170 TFLOPS class), and only 16/112 cells pass the probe's checksum (the M=4/N=16/nacc=1 family), so no absolute ceiling and no application-speed claim is made. The rate-class note was updated in place. Evidence: `results/2026/09/b70-improvement-probes-20260920/dpas-rate-probe/FINDINGS.md` + CSVs. |

## 4. Publication backlog (inventory, not a claim)

Catalog authority `~/intel-arc-pro-b70-inference-cookbook/data/benchmarks.v1.json` = 26
records (vLLM XPU 14, llama.cpp SYCL 12). **OpenVINO GenAI: 0 records** and **Cascadia: 0
records** despite on-disk evidence. OVMS has no live footprint. Merge risk flagged where
OV/Cascadia/llama series sit in one table/chart (`OPENVINO-CASCADIA.md:92-98`, the
head-to-head SVG) — the standard forbids merging those engines.

## 5. Stop-rule application

- P1: headline target **withdrawn** on measurement (2.44 % < 10 %); phase blocked on the
  absent artifact for the end-to-end gate.
- P2/P3: **not startable** in this window — their target artifacts are absent.
- Nothing in this note weakens a gate; no public number is created.
