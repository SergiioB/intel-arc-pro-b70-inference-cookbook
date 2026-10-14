---
title: "DeepSeek-V4-Flash-0731 REAP-K216 — HTTP request to output on dual B70 (2026-08-25)"
type: research
date: 2026-08-25
status: final
---

# DeepSeek-V4-Flash-0731 REAP-K216 — HTTP request to output on dual B70 (2026-08-25)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE**

This is the in-depth authority for how one llama.cpp SYCL C1 request becomes tokens on this host. The agent-contract pointer is in `AGENTS.md` §2 and the short operating summary in §16.

| Item | Value |
|---|---|
| Artifact | `heath0xFF/DeepSeek-V4-Flash-0731-REAP-K216-GGUF` UD-IQ3_XXS, 82.95 GiB, mmap only |
| Engine | llama.cpp SYCL `build-sycl-latest-qw38` commit `1692f9e50` |
| Placement | dual B70, `-sm layer`, `-dev SYCL0,SYCL1`, `-fa on`, `-t 8` |
| Campaign KV | `-ctk f16 -ctv f16` (not the fleet q8_0/q4_1 default) |
| Best no-spec decode | L5 `-ncmoe 15 -ts 65,35` p512/g128 median **8.87** engine tok/s (`n=3`) — **SUPERSEDED 2026-09-21: 9.25 clean / 9.58 patched** on the re-downloaded artifact |
| Best dedicated prefill | L3 `-b 8192 -ub 1024 -c 8448` p8192/g1 median **105.1** prompt tok/s |
| Evidence | `results/2026/08/dsv4-k216-20260825Tdual2/`, `results/2026/08/dsv4-k216-20260825T0707Z-prefill/`, `results/2026/08/dsv4-k216-20260825T1441Z-dspark/` |
| Related | `research/2026/dsv4-k216-prefill-ubatch-20260825.md`, `research/2026/dsv4-k216-dspark-and-public-numbers-20260825.md`, `research/2026/dsv4-k216-buun-moe-cache-20260825.md`, `research/2026/b450f-dual-b70-pcie-topology-20260824.md` |

Architect pass: `b70-architect` 2026-08-25 against this commit, host topology, and the measured cells above. Numbers below are local self-report, roughly evidence tier E2, not public headlines.

---

## 1. What actually limits this path

This workload is not limited by the B70’s nominal XMX/DPAS peak. It is limited by how llama.cpp’s present SYCL backend maps an irregular IQ3_XXS MoE graph onto two GPUs plus CPU-resident expert banks.

1. IQ3_XXS is not a native XMX operand. Xe2 DPAS is INT4/INT8/FP16/BF16. Multi-row IQ3 is expanded to FP16, then oneMKL.
2. C1 decode is tiny expert GEMVs (`M=1`, six routed experts per layer), not large GEMMs.
3. Fused SYCL `MUL_MAT_ID` MMVQ omits IQ3_XXS **upstream at `1692f9e50`** (the pinned tree `~/llama.cpp` carries the device-ID cases uncommitted — so on this host the fallback applies only to the multi-token path). The fallback copies expert IDs D2H, `stream->wait()`, then launches experts from a host loop. **(2026-09-21: the device-ID repair measured +3.54 % end-to-end, ≥10 % gate FAILED — sync is not the dominant cost; the fused op is 5.52× above its weight-read floor.)**
4. 43 layers × 3 routed projections × 6 experts is hundreds of small operations per token.
5. `-ncmoe N` puts the first N layers’ **routed experts** on CPU, not whole layers. Attention stays on GPU. That puts DDR4 and mmap page cache on the token critical path.
6. `-sm layer` is serial: GPU0 range, then a host-staged copy, then GPU1 range. No P2P.

Prefill and decode therefore have different walls:

- **Decode:** small-M IQ3 expert GEMV dispatch, host sync, CPU-MoE, serial heterogeneous scheduling.
- **Prefill:** host-orchestrated grouped expert dispatch plus IQ3→FP16 materialization before oneMKL; ubatch already captured ~1.26–1.30×.

HTTP, JSON, tokenize, sample, and socket write are not the wall.

---

## 2. Host buses that the request actually uses

From live topology (`research/2026/b450f-dual-b70-pcie-topology-20260824.md`):

| Bus | Observed | Role in this serving path |
|---|---|---|
| Loopback / TCP | localhost `127.0.0.1:8080` in campaigns | Request JSON in, token chunks out. Negligible vs ~113 ms/token. |
| DDR4-2133 dual channel | ~34 GB/s theoretical, 32 GiB total | Socket buffers, JSON, tokenizer, graph metadata, CPU-MoE expert GEMV, host-staged GPU0↔GPU1 copies, mmap page cache. One NUMA node. |
| 5700X3D L3 | large 3D V-Cache | Helps tokenizer tables, router arrays, one hot expert fragment. Cannot hold ~83 GiB weights or even 15 CPU MoE layers (~tens of GiB). |
| NVMe (CPU lanes, not chipset) | Gen3 x4 class | mmap major faults on cold expert pages. Competes with GPU copies for memory-controller bandwidth. |
| GPU0 PCIe | Gen3 **x8**, ~7.1 GB/s H2D/D2H | Weight upload at load; decode activations; expert-ID D2H (tiny bytes, expensive waits). |
| GPU1 PCIe | Gen3 **x4**, ~3.6 GB/s H2D/D2H | Same, slower leg. Layer-boundary H2D is the slower copy. |
| GPU0↔GPU1 | **no P2P** | Copies bounce `VRAM → D2H → DDR4 → H2D → VRAM`. |
| Resizable BAR | 32 GB both cards | Full VRAM BAR; does not create P2P. |

C1 decode activation crossing the GPU boundary is on the order of tens of KiB (hyper-connection ~64 KiB if 4096×4×F32). At 3.6 GB/s that is tens of microseconds of ideal transfer. The cost is the **serial backend boundary**, not the bytes. Prefill ubatch 1024 makes that copy tens of MiB and more visible.

---

## 3. Model geometry that drives the graph

GGUF metadata used by `deepseek4.cpp`:

| Property | Value |
|---|---:|
| Transformer layers | 43 |
| Hidden | 4,096 |
| Routed experts | 216 (REAP-K216) |
| Experts used | 6 |
| Expert FFN | 2,048 |
| Shared experts | 1 |
| Indexer top-K | up to 512 |
| Hyper-connection streams | 4 |

Per routed expert, three principal matrices (gate, up, down) are ~25.2 M weights ≈ **9.2 MiB** IQ3_XXS. Six selected experts ≈ **55 MiB/layer/token** of routed weight bytes, ≈ **2.3 GiB/token** across 43 layers (approximation: excludes shared expert, attention, temps, dequant expansion).

L5 `-ncmoe 15` split of that routed volume:

```text
CPU-routed: 15 × 55 MiB ≈ 827 MiB/token
GPU-routed: 28 × 55 MiB ≈ 1.51 GiB/token
```

Plus ~84 SYCL `MUL_MAT_ID` calls/token on the GPU-routed layers (28 × 3), each currently a blocking host wait.

---

## 4. Request path, in order

Sources: `tools/server/server-http.cpp`, `server-context.cpp`, `common/chat.cpp`, `src/llama-context.cpp`, `src/models/deepseek4.cpp`, `src/llama-graph.cpp`, `src/llama-kv-cache-dsv4.cpp`, `ggml/src/ggml-sycl/ggml-sycl.cpp`, `common/sampling.cpp`.

### 4.1 Socket and JSON (CPU only)

Client TCP → kernel socket buffers in DDR4 → `cpp-httplib` → `json::parse` → OpenAI-compat chat template → `handle_completions_impl()` queues a slot task.

No PCIe yet. Localhost C1: microseconds to low milliseconds.

### 4.2 Tokenize and slot (CPU, GGUF metadata via mmap)

`common_tokenize()` against tokenizer tables in the GGUF mapping. Prompt length vs `-c`, sampler chain, prefix-cache decision, `llama_batch`.

Cache-hit cells are a different measurement class. Cold HTTP cells in this campaign used unique entropy prefixes and `cache_n=0`.

### 4.3 Ubatch and graph

```text
effective ubatch = min(-b, -ub, remaining prompt, scheduler limits)
```

Raising `-ub` without `-b` is a no-op in this commit.

`llama_context::process_ubatch()` → `model.build_graph()` (`llama_model_deepseek4::build_arch_graph`) → `ggml_backend_sched_alloc_graph` assigns nodes to CPU / SYCL0 / SYCL1 → H2D of inputs → `ggml_backend_sched_graph_compute_async()`.

Graph **reuse** works (L5 logs hundreds of reused graphs). SYCL **command graphs do not**: multi-device capture is off, and `GGML_OP_MUL_MAT_ID` is rejected because it does a blocking queue wait (`check_graph_compatibility()` in `ggml-sycl.cpp`).

### 4.4 Dual-GPU layer pipeline

`-sm layer` assigns contiguous layer ranges. Not row-split (row-split segfaulted: R1/R2, do not retry).

```text
CPU scheduler
  → GPU0 layer range
  → D2H Gen3 x8 → DDR4 → H2D Gen3 x4
  → GPU1 layer range
  → logits / sample on host
```

### 4.5 Per layer (DeepSeek-V4)

Each main layer: hyper-connection pre (`build_hc_pre` / Sinkhorn) → RMS → one attention variant → routed MoE + shared expert → hyper-connection post.

Attention is **not** one dense algorithm:

- ratio-4: Lightning Indexer + compressed sparse attention (`ggml_lightning_indexer`, top-K ≤ 512, flash-attn over raw window + selected compressed K).
- ratio-128: hierarchical compressed attention.
- else: raw / sliding window 128 (`build_raw_attention`).

Indexer is DSA attention, **not** speculative decode. `-fa on` should stay on. At p512 it is not why decode is ~9 tok/s.

KV for this campaign is f16/f16 in `llama-kv-cache-dsv4.cpp` (raw recent K plus compressor/indexer state). Fleet `q8_0 K + q4_1 V` is unproven on this architecture; `q4_0` K remains forbidden.

### 4.6 Routed MoE — the hot path

Router: 216 logits → top-6. Then `build_moe_ffn()`:

1. `MUL_MAT_ID` up
2. `MUL_MAT_ID` gate
3. SwiGLU
4. `MUL_MAT_ID` down
5. weighted sum

IQ3 fused path (`ggml_sycl_mul_mat_id_mmvq_fused` / `ggml_sycl_mul_mat_vec_q_id`) **does not include IQ3_XXS**, even though non-routed `mmvq.cpp` has `mul_mat_vec_iq3_xxs_q8_1_sycl()`.

Fallback in `ggml_sycl_mul_mat_id()`:

**Decode (`n_tokens = 1`):**

```text
device expert IDs (6 × int32 ≈ 24 B)
  → D2H
  → stream->wait()
  → host loop of 6 × ggml_sycl_mul_mat()
```

The 24 bytes are not a PCIe bandwidth problem. The **~84 blocking waits/token** on L5 GPU-routed layers are. At 8.87 tok/s that is ~745 wait sites/s: the queue never keeps depth. **(Mechanism framing corrected 2026-09-21:** the device-ID repair measured only +3.54 % end-to-end, ≥10 % gate FAILED — the sync pathology is real but not the dominant decode cost; the fused routed op runs 5.52× above its weight-read floor, i.e. kernel efficiency. Current best: 9.25 clean / 9.58 patched.)

**Prefill (`n_tokens > 1`):** D2H IDs → wait → CPU counting sort (`mmid_counting_sort_rows`) → H2D row map → pack rows → host loop over active experts → scatter. Better than per-token GEMV, still host-orchestrated.

`ggml_sycl_supports_mmq()` is unconditionally `false`. Do not flip it: there is no IQ3 MMQ in `mmq.cpp`. Prefill therefore dequantizes IQ3→FP16 (~5.2× bytes) then oneMKL GEMM.

### 4.7 CPU-MoE (`-ncmoe N`)

`arg.cpp` / `llm_ffn_exps_block_regex()` overrides **routed expert tensors** of the first N layers onto the CPU backend. Attention and shared expert stay on the GPU assignment. Each of those layers is:

```text
GPU attention/shared  ↔  CPU routed GEMV  ↔  GPU continuation
```

Weights are mmap-backed. Selected experts fault in from NVMe if not in page cache. 15 layers × ~55 MiB selected ≈ 827 MiB/token of CPU routed weight traffic **plus** unpack and 8-thread accumulate on DDR4-2133. That is not “34 GB/s / 827 MiB”; it is latency + sync + dequant.

`--no-mmap` / `--mlock` on this 83 GiB file with 32 GiB RAM is forbidden.

### 4.8 Sample and socket

Last prompt ubatch requests logits → `llama_synchronize()` → `common_sampler_sample()` → detokenize → `send_partial_response()` → chunked HTTP.

Sampling is serial in the autoregressive chain and cheap next to ~113 ms of engine time.

Subsequent tokens reuse the graph topology and repeat §4.4–4.8 for one new token. Token T must be sampled before T+1 can start, unless a **compatible cheap draft** evaluates extra candidates.

---

## 5. Measured bottlenecks (PROVISIONAL)

### Decode

| Placement | Median engine tg p512/g128 |
|---|---:|
| L1 `-ncmoe 30` | 2.17 |
| L3 `-ncmoe 20` | 7.26 |
| L5 `-ncmoe 15 -ts 65,35` | **8.87** (max obs 9.27) — **SUPERSEDED 2026-09-21: 9.25 clean / 9.58 patched** |

≈ 113 ms/token at the best cell. Directionally, moving routed experts GPU-ward is the flag win. Splits are not a single-variable A/B.

Primary: IQ3 routed GEMV kernel efficiency (fused op 5.52× above its weight-read floor) + host `MUL_MAT_ID` waits (~84 GPU waits, ~504 GPU expert GEMVs/token on L5) — **2026-09-21: the device-ID repair is worth only +3.54 % end-to-end, so the waits are real but secondary; the kernel gap dominates**.

Secondary: CPU-MoE + CPU↔GPU backend crossings.

Tertiary: serial dual-device boundary (capacity, not 2× compute).

Not the wall: HTTP, tokenize, sample, socket, 24-byte ID copies as bandwidth, short-context KV, stock 150 vs 230 W as a 3× lever.

### Prefill

Dedicated long cells on L3, unique prefix, `n=3`:

| Cell | median pp |
|---|---:|
| p2048/g1 ub512 | 80.65 |
| p2048/g1 ub1024 | 104.55 |
| p8192/g1 ub512 | 83.72 |
| p8192/g1 ub1024 | **105.12** |

ub512→1024 ≈ **1.26–1.30×**. p512+g128 mixed cells are a bad prefill judge.

Primary remaining: host grouping + full IQ3→FP16 before oneMKL.

Secondary: CPU-MoE, host-staged GPU0→GPU1 activations (bytes scale with ubatch), expert load imbalance.

---

## 6. Packed DSpark probe (load only; 2026-08-25T17:04)

Goal was denser packing: ~1 GiB free on GPU0, fill GPU1, `--spec-draft-cpu-moe`, `--spec-draft-device SYCL1`, Unsloth DSpark Q8_0 (`dflash`, **256** experts) vs K216 **216**.

| Try | Result | Do not retry |
|---|---|---|
| GPU-resident DSpark on L3 (`-ngld auto`) | `UR_RESULT_ERROR_DEVICE_LOST` in `ggml_backend_sycl_buffer_set_tensor` | this pairing |
| CPU-MoE DSpark on sparse L3 | loaded; warmup accept 59/67 but **2.48 tg** (worse than no-spec 7.26/8.87; current best 9.25/9.58) | as a speed path |
| P15_65 `ncmoe=15 ts=65,35` + CPU-MoE draft + `-devd SYCL1` | **loaded** after 440 s; post-load free **GPU0 1897 MiB / GPU1 1720 MiB**. Server then exited without a measured decode cell in `campaign-packed.log`. | — |
| P15_60 `ts=60,40` | SYCL1 OOM `2370502656` B (~2.21 GiB) | this split + draft |
| P15_58 | launch started; log truncated / campaign gone | unknown; do not assume success |

GPU0 at 1897 MiB free is still above the “~1 GiB” packing target. One full routed expert bank is ~2 GiB IQ3; fragmented 1–2 GiB free does not buy another layer by itself. `-ts` is layer-boundary, not byte-equal.

Draft family remains wrong (256 vs 216). Acceptance on CPU-MoE draft was high and **still slower**.

Caps were left at 180/230 W after the packed script died; restored to **150 W** on both cards 2026-08-25T18:06. GPUs empty.

---

## 7. Ranked remaining speed work

Dead without a code or artifact change: `-ts 46,54` / `50,50` low ncmoe; `-sm row`; single GPU; GPU-resident this DSpark on L3/L5; `ggml_sycl_supports_mmq()` flip; SYCL graphs on dual+`MUL_MAT_ID`; `--no-mmap`/`--mlock`; q4_0 K; treating Unsloth 256-expert DSpark as a K216 recipe; CPU-MoE DSpark as a speed win.

### Decode (open)

1. **Fused IQ3_XXS `MUL_MAT_ID` MMVQ** — device-resident expert IDs, no D2H, no `stream->wait()`, one routed kernel (or tight set) for six experts. Files: `ggml-sycl.cpp`, `mmvq.cpp`. Architect: ~**9.6–11 tok/s**, not 30.
2. **Device-side grouping / event deps** for general `MUL_MAT_ID` (also helps prefill).
3. **Compatible 216-expert DSpark** that is cheap and GPU-resident beside the target. Architect typical **11–16**, stretch **~18**. Current Q8 draft is the wrong family and does not fit GPU-resident.
4. **Pack GPU0 toward ~1 GiB and fill GPU1** only if it unlocks `-ncmoe 14` without SYCL1 2.2 GiB alloc fails. Incremental at best; P15_60 already OOM with draft.
5. Bounded mmap readahead on CPU expert ranges (`MADV_WILLNEED` / `posix_fadvise`) — cold/tail, not median GPU kernel.
6. KV `q8_0/q4_1` as **capacity / long-context** experiment with DSV4-specific correctness, not a p512 decode 3×.
7. Narrow `-ncmoe 14/15/16` GPU0-heavy only.

### Prefill (open)

1. Device-resident grouped multi-row IQ3 expert GEMM (no whole-expert FP16 dump). This is the only plausible remaining **~2×** path (toward ~210 pp from ~105).
2. Tile-local IQ3→DPAS-compatible tiles in registers/SLM.
3. `-ub 1536/2048` with matching `-b`: expected **0–10%**, not 2×.
4. Fewer CPU-MoE layers if packing actually fits.

---

## 8. Why 30 tok/s C1 is not on this path

```text
8.87 tok/s ≈ 113 ms/token (2026-08-25 cell — SUPERSEDED 2026-09-21: 9.25 clean / 9.58 patched ≈ 104–108 ms/token)
30 tok/s   ≈ 33 ms/token
required   ≈ 3.38×  (drop ~79 ms/token, ~70% of current time)
```

Fused IQ3 MMVQ at 11 tok/s is still ~2.7× short. Compatible DSpark stretch at 18 tok/s is still ~1.7× short. Reaching 30 with unchanged target cost would need ~3.4 **accepted** tokens per target eval **and** a free draft — neither holds (CPU-MoE draft already proved high accept can still lose).

IQ3 is not native INT4 XMX; C1 is GEMV; IDs leave the device; 15 routed layers are on the Ryzen in the fastest measured fit; GPU1 is Gen3 x4; no P2P; graphs off.

**Defensible architectural range remains ~9.6–18 tok/s** with kernel fusion and a hypothetical compatible GPU draft. **30 tok/s is not a credible current-path target.**

Public directional only: same K216 family on GB10 CUDA ~16 gen / 33–48 pp. Unsloth B200 60→120 with DSpark is a different class. Dense 27B on this B70 ~25 tg / 613 pp is not a multiplier for 284B-class MoE IQ3 split.

---

## 9. Agent operating rules for this model

- Dual GPU until an explicit product-route change. Independent stacks (ADR-0008) are the dual-card **default** for two engines; this serving path is one **layer-split** engine, which is a different product.
- Never `--no-mmap` / `--mlock` this GGUF.
- Never q4_0 K.
- Do not retry the dead table in §7.
- `n_ubatch = min(n_batch, -ub)`.
- DSpark/MTP/DFlash are `--spec-type` modes in `libllama-common.so`; Lightning Indexer is not spec decode; DFlash2 is not a named mode here.
- Label all tok/s **PROVISIONAL — NOT FOR PUBLIC HEADLINE** until a claims audit.
- Restore 150 W on both cards after campaigns; discover hwmon dynamically.

## Relevant source (absolute)

- `~/llama.cpp/tools/server/server-http.cpp`
- `~/llama.cpp/tools/server/server-context.cpp`
- `~/llama.cpp/src/llama-context.cpp`
- `~/llama.cpp/src/llama-graph.cpp`
- `~/llama.cpp/src/models/deepseek4.cpp`
- `~/llama.cpp/src/llama-kv-cache-dsv4.cpp`
- `~/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp`
- `~/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp`
- `~/llama.cpp/common/speculative.cpp`
- `~/llama.cpp/common/sampling.cpp`
