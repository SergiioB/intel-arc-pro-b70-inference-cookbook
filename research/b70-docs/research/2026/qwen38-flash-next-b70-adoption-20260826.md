---
title: "Qwen3.8-Flash-Next on Dual B70 — Adoption and Porting Dossier"
type: research
date: 2026-08-26
status: final
---

# Qwen3.8-Flash-Next on Dual B70 — Adoption and Porting Dossier

**Status:** active engineering, 2026-08-26 UTC  
**Public-claim state:** PROVISIONAL — NOT FOR PUBLIC HEADLINE  
**Official model:** `Qwen/Qwen3.8-Flash-Next`  
**Pinned checkpoint revision:** `f5d08274bafd880402bd16f5e3e6c514136ec06c`  
**Evidence root:** `results/2026/08/qwen38-flash-next-adoption-20260826T133339Z/`

## 1. Objective and support boundary

The objective is a correct full-model local generation on two Intel Arc Pro B70
cards, followed by an AutoRound/GPTQ vLLM-XPU serving path. llama.cpp/GGUF remains
a possible first correctness route only if exact Qwen4-Exp converter, graph, QSA,
PLE, MTP, tokenizer, and multimodal support become verifiable.

Current outcomes:

- `official_tensor_inventory_complete` for names plus aggregate bytes;
- `transformers_reference_ready_dev_main`;
- `upstream_vllm_support_pending_pr`;
- initial local XPU package dispatch and portable HC implemented;
- upstream Qwen4-Exp reference tests pass in the pinned XPU image;
- `blocked_missing_xpu_kernel` for QSA and packed W2 routed MoE;
- `blocked_artifact_or_storage` for the full 335.28 GiB source and conversion;
- `blocked_capacity_or_bandwidth` for W4 full residency, host-N-gram-only W4, and
  unlabeled W4 expert-streaming decode;
- `blocked_capacity_or_bandwidth` for a resident INT4 PLE table on the current
  approximately 30 GiB host RAM;
- the capacity-first artifact *target* is W2 G128 routed experts plus a table-aware
  INT4 PLE table, with a selective W2/W4 2.5-bit-effective fallback on an explicitly
  declared FP8-small-domain series. That is not a packed, loaded, or quality-tested
  fit.

No model weights, conversion, full server launch, benchmark, publication, commit,
or push have occurred.

## 2. Official release evidence

Pinned metadata was downloaded without weight shards into the evidence root:

- `config.json`;
- `model.safetensors.index.json`;
- tokenizer, vocabulary, merges, and chat template;
- image/video processor configs;
- README, license, generation config, and HF API response.

The metadata directory is approximately 23 MiB. Its SHA-256 manifest is:

`results/2026/08/qwen38-flash-next-adoption-20260826T133339Z/manifests/metadata-sha256.txt`

Official artifact manifest:

`results/2026/08/qwen38-flash-next-adoption-20260826T133339Z/manifests/official-artifact.json`

The HF API reports 131 safetensors shards totaling 360,000,192,888 remote file
bytes. The index reports 359,999,963,128 tensor payload bytes.

## 3. Exact tensor inventory

The evidence root retains the 1,658-name index, config, and HF API parameter
totals. It does **not** retain per-tensor safetensors headers. Aggregate
parameter/byte totals are therefore `E2`; the per-family byte rows below are
reconstructed from config-closed families, name classes, and the index
`total_size`, and remain `E0/E1` until a header JSON is stored.

HF `safetensors.parameters` reports 179,999,981,424 BF16 plus 35 I64
parameters. Those 35 I64 parameters add 210 bytes versus a pure-BF16 total and
land on the PLE-support row.

| Subsystem | Parameters | Source bytes |
|---|---:|---:|
| Routed experts | 120,795,955,200 | 241,591,910,400 |
| N-gram tables | 51,200,245,760 | 102,400,491,520 |
| GDN/hyper-connections/support | 2,720,571,264 | 5,441,142,528 |
| MTP | 2,607,150,848 | 5,214,301,696 |
| Embedding and LM head | 1,271,398,400 | 2,542,796,800 |
| Full attention and QSA indexer | 617,358,336 | 1,234,716,672 |
| Vision | 448,931,056 | 897,862,112 |
| Shared experts | 236,052,480 | 472,104,960 |
| PLE support outside table | 32,839,715 | 65,679,640 |
| Other/classified metadata | 69,478,400 | 138,956,800 |
| **Total** | **179,999,981,459** | **359,999,963,128** |

The footprint inspector was corrected to recognize released names:

- prefix `mtp.*`;
- `ple.*` support tensors;
- N-gram table and integer metadata;
- `hyper_connection` tensors;
- fused `gate_up_proj` routed expert tensors.

The official 1,658-name index now classifies with zero unknowns:

| Family | Tensor count |
|---|---:|
| attention/GDN/GR | 819 |
| embeddings/head | 2 |
| N-gram table/index | 131 |
| MTP | 31 |
| norm/router/misc | 48 |
| PLE support | 6 |
| routed experts | 96 |
| shared experts | 192 |
| vision | 333 |

Permanent test:

`.agents/skills/qwen38-flash-next-adoption/scripts/test_official_tensor_names.py`

Evidence:

`results/2026/08/qwen38-flash-next-adoption-20260826T133339Z/tests/official-tensor-name-test.txt`

## 4. Capacity arithmetic

Two B70 cards expose 65,312 MiB combined, but the mixed-serving fixed-allocation
ceiling with 3 GiB reserve per card is 59,168 MiB, approximately 62.04 GB decimal.

Whole-checkpoint optimistic payloads:

- BF16: 360.0 GB;
- FP8/INT8: approximately 180.0 GB;
- uniform W4: approximately 90.0 GB.

Routed experts at symmetric W4 with one FP16 scale per group at G128:

- parameters: 120,795,955,200;
- raw W4: 60.398 GB;
- G128 scales: 1.887 GB;
- routed expert packed planning estimate: 62.285 GB.

All other non-N-gram source tensors, including shared experts and PLE support,
total approximately 16.008 GB. The experts-first GPU-resident estimate is therefore
approximately 78.293 GB before runtime, cache, graph, activation, recurrent state,
communication, and per-card imbalance. Host N-gram alone does not solve fit.

The corrected inspector separates:

- quantized GPU tensors;
- GPU-preserved source tensors;
- host-tier N-gram source bytes;
- estimated GPU-resident weights;
- estimated total artifact.

Its default metadata assumption is now explicitly scale-only `Bmeta=2`; `Bmeta=4`
must be declared separately rather than silently mixing zero-point storage into
the same candidate.

## 5. Storage and host-memory gate

Live storage at the start of adoption:

| Mount | Free |
|---|---:|
| `/` | 38.4 GiB |
| `/mnt/<models-disk>` | 31.7 GiB |
| `/mnt/<models-disk>` | 35.5 GiB |
| `/mnt/<other-disk>` | 319.2 GiB |

Aggregate free space was about 424.8 GiB, insufficient for source plus output,
temporary packing, and rollback. No single mounted volume holds the source.

The unmounted `/dev/nvme0n1p2` was inspected read-only. It is the Windows system
volume, 931 GiB total, 831 GiB used, only 100 GiB free, with Windows, Users,
Program Files, pagefile, hiberfil, games, and user files. It was unmounted after
inspection and must not be repurposed.

Full conversion remains blocked until one safe workspace provides roughly 500 GiB
or more free. NTFS may be considered for immutable source storage only after its
semantics and write integrity are validated; it is not suitable by assumption for
atomic publication, symlink-sensitive staging, or a nearly-full conversion.

Host memory after stopping ComfyUI:

- 30 GiB total;
- approximately 24 GiB available;
- 15 GiB swap, approximately 12 GiB free at observation.

N-gram table host lower bounds:

- BF16: 95.37 GiB, impossible;
- INT8: 47.68 GiB, impossible;
- INT4: 23.84 GiB plus metadata, barely capacity-plausible but leaves insufficient
  margin for workers, pinned staging, caches, and OS.

Therefore a host PLE table likely needs INT4 or lower, memory mapping/disk backing,
and strict ownership; a broad host expert cache cannot coexist casually on the
current RAM budget.

A read-only reclaim audit found no safe path to the approximately 500 GiB workspace
needed for full source plus conversion. Current non-Windows mounts provide about
106 GiB free in aggregate. Strictly conservative reclaim is only about 0.25 GiB;
conditional cache, dangling-image, and user-debug candidates total roughly 45–52
GiB and require separate deletion authorization. Registered model artifacts,
conversion lineage, and result evidence were excluded. No files were deleted.

For repeated full conversion work, the architecture review recommends one dedicated
NVMe filesystem with at least 1.5 TiB free; 2 TiB is safer, and a 4 TB device is the
practical repeated-candidate target. Storage improves conversion and cold loading,
but does not remove the GPU 1 Gen3 x4 serving bottleneck.

## 6. Upstream source pins

Transformers reference checkout:

- path: `~/worktrees/qwen38-flash-next/transformers`;
- commit: `36bc98ef9dd009569366f5e253ec1876ecafd925`.

vLLM architecture PR worktree:

- path: `~/worktrees/qwen38-flash-next/vllm-pr53896`;
- PR: `vllm-project/vllm#53896`;
- commit: `d4d0f73ef171154eac6f1914dca47001d662cfbb`.

Pinned runtime image used for tests:

`vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`

The PR contains common architecture plus explicit NVIDIA and AMD component
packages. `model.py`, `mtp.py`, and `model_state.py` are identical between those
backends; QSA, PLE, HC ops, and low-latency hooks differ.

## 7. Reference test results

The upstream PR's config, QSA reference, PLE, and weight-loading tests were run
inside the pinned XPU image without GPU access:

- 80 passed;
- 27 skipped.

After adding the local XPU dispatch and portable HC tests, the same suite on GPU 0
reported:

- 82 passed;
- 27 skipped;
- one visible Intel Arc Pro B70.

The public package correctly resolved:

- `Qwen4ExpForCausalLM` to `vllm.models.qwen4_exp.xpu.model`;
- `Qwen4ExpForConditionalGeneration` to the XPU model module;
- `Qwen4ExpMTP` to `vllm.models.qwen4_exp.xpu.mtp`.

This proves config, registry, mapping, PLE shard logic, QSA reference logic, and HC
formulas. It does not prove full QSA execution, PLE execution, model forward, MTP,
or serving.

## 8. Local XPU overlay

Current uncommitted worktree changes:

- modify `vllm/models/qwen4_exp/__init__.py` to dispatch XPU explicitly;
- add `vllm/models/qwen4_exp/xpu/`;
- add `tests/models/qwen4_exp/test_xpu_portable_hc.py`.

XPU package policy:

- reuse the common model/MTP/model-state source;
- retain existing vLLM XPU GDN dispatch;
- use standard vLLM linears;
- implement Gated Residual glue in ordinary PyTorch first;
- use the ordinary BF16 PLE path initially;
- reject serialized FP8 PLE on XPU;
- reject QSA explicitly rather than silently replacing sparse attention with dense
  attention;
- keep NVIDIA low-latency GEMM disabled;
- defer MTP claims until target/draft scheduling passes.

Portable HC implements:

- grouped Gemma RMSNorm;
- HC SiLU with division by branch count;
- sigmoid gated branch mean;
- `2 * sigmoid(injection / hc_count)` residual injection;
- materialized combine followed by per-branch RMSNorm.

## 9. Tiny XPU construction attempts

Harness:

`results/2026/08/qwen38-flash-next-adoption-20260826T133339Z/tests/tiny_xpu_construct.py`

### Attempt 1

The config and platform initialized, but `initialize_model_parallel` failed because
the standalone harness did not enter `set_current_vllm_config`. This was a fixture
error, not a model/backend failure.

Log:

`results/2026/08/qwen38-flash-next-adoption-20260826T133339Z/logs/tiny-xpu-construct.log`

### Attempt 2

After fixing the config context:

- architecture resolved as `Qwen4ExpForCausalLM`;
- max model length 64;
- XCCL rank 0/TP1 initialized;
- Qwen GDN construction began;
- the tiny config selected Triton/FLA prefill and Triton decode because it used
  K/V dimensions 8 rather than the real 128 and did not satisfy the fused kernel
  contract;
- ordinary full attention emitted an XPU dtype/backend fallback warning;
- the process did not complete and was stopped after the controlling command was
  cancelled.

Log:

`results/2026/08/qwen38-flash-next-adoption-20260826T133339Z/logs/tiny-xpu-construct-v2.log`

The next fixture must use production-relevant K/V=128, BF16, a minimal but valid
head topology, and bounded vocabulary/layer count. It should test construction
before forward, then bind cache metadata for one deterministic forward separately.

A cancelled tool call left one test container running. Its logs were captured and
only that exact container was stopped. VRAM returned to approximately 32,040 MiB
free on GPU 0 and 32,615 MiB on GPU 1. Both power caps remain 150 W.

## 10. GPU quiescence procedure

The active workload before tests was not idle ComfyUI: it included the server,
memory watcher, an active H3 queue, and a waiter prepared to launch another chain.
The user explicitly authorized freeing the GPUs. Those exact controllers and the
server were stopped, preventing automatic relaunch.

After quiescence:

- no ComfyUI, llama.cpp, or vLLM engine remained;
- no inference Docker container remained;
- GPU 0 exposed about 31.84 GiB free initially;
- GPU 1 exposed about 32.62 GiB free;
- both cards were restored to 150 W.

Do not restart ComfyUI while the adoption campaign is actively using the GPUs.
When restoring it later, reconstruct the previous command and queue intent from the
preflight evidence rather than launching a different workload silently.

## 11. Current kernel gaps

### Existing useful XPU paths

- GDN has an explicit `forward_xpu` route and registered XPU op;
- local vLLM-XPU kernels include the Xe2 grouped-GEMM atomic-zero fix;
- existing INT4 MoE work provides expert-by-expert GPTQ representation conversion;
- mixed GDN v5 provides the compact/split/scatter correctness pattern.

### New required work

QSA is the principal architecture blocker. A correct XPU implementation needs:

1. index Q/K projection;
2. compressed key cache update;
3. average-pool-before-RMSNorm/partial-RoPE semantics;
4. block-causal scoring;
5. top-k complete block selection;
6. token expansion/truncation and mandatory incomplete tail;
7. sparse paged attention;
8. QSA index reuse under MTP;
9. eager/graph determinism and cache metadata integration.

The common PR metadata builder is reusable, but no current XPU kernel provides the
selection or sparse-attention computation.

PLE requires:

- official hashing/address parity;
- table ownership and quantized row format;
- sparse gather/dequantization;
- short convolution/fusion;
- host prefetch/hot cache;
- traffic and wait counters.

Generic CPU weight offload is not equivalent to PLE prefetch.

## 12. Architecture fit decision

**Evidence:** `E0 estimate_or_extrapolation`; official tensor inventory plus
analytical packing/capacity/bandwidth modeling. No lower-bit artifact has been
packed, loaded, quality-tested, or benchmarked. **PROVISIONAL — NOT FOR PUBLIC
HEADLINE.**

### Capacity-first artifact

The selected first complete artifact target is:

- routed experts: symmetric **W2 G128** planning representation;
- PLE N-gram table: dedicated table-aware **INT4**, one authoritative host copy;
- GDN/HC, QSA/indexer, embeddings/head, shared experts, PLE support, MTP, vision,
  and miscellaneous neural tensors: BF16 initially;
- MTP retained in the artifact but disabled for first no-spec text bring-up;
- vision retained but lazy-loaded until the text path passes;
- complete layers owned by one card, not per-layer tensor parallelism.

Optimistic neural payload excluding the host PLE table:

- **48,093,986,598 bytes = 44.791 GiB**;
- ideal balanced placement: approximately 22.396 GiB/card;
- remainder versus the 2 GiB/card combined ceiling: 14.990 GiB;
- remainder versus the 3 GiB/card combined ceiling: 12.990 GiB.

Those remainders are packed-weight payload versus the combined reserve ceiling.
They are **not** unused KV/runtime headroom and are **not** a dual-B70 fit
result. Runtime allocations have not been estimated.

This candidate is dual-blocked on the current host:

- `blocked_missing_xpu_kernel`: the proven local Xe2 grouped-MoE route is W4A16,
  not packed W2A16;
- `blocked_capacity_or_bandwidth` for resident INT4 PLE: the 23.84 GiB table plus
  metadata cannot coexist with workers, pinned staging, OS, and caches on
  approximately 30 GiB RAM / 24 GiB available.

W2 checkpoint quality is also untested.

### Quality fallback

The preferred fallback is selective W2/W4 experts rather than a nominal W3
format. Xe2 naturally exposes INT2 and INT4 DPAS operands, not INT3. Keep the
payload series explicit:

- 25% W4 / 75% W2, **all other neural tensors BF16**: 51.822 GiB, remainder
  5.959 GiB versus the mixed-serving ceiling;
- 25% W4 / 75% W2, **GDN/QSA BF16 and other small domains at unvalidated ideal
  FP8**: 47.477 GiB, remainder 10.304 GiB.

Do not quote 47.477 GiB / 10.304 GiB as the same-basis companion of the 44.791
GiB all-BF16-other W2 candidate. The 2.8-bit 51.696 GiB / 6.086 GiB row belongs
only to the FP8-small-domain series.

Promotion must use calibration sensitivity and capability-oriented quality tests;
routing frequency alone can erase rare-language or rare-domain experts.

### Why W4 expert tiering is not the primary route

Each routed expert contains 4,915,200 parameters. Including provisional G128 FP16
scales, a complete top-10 miss across 48 layers transfers approximately:

- W4: 1.216512 GB/token;
- W3-equivalent: 0.921600 GB/token;
- W2: 0.626688 GB/token.

On GPU 1's previously observed 3.6 GB/s x4 host link, the **ideal H2D-only** W4
full-miss floor is about 338 ms/token, or 2.96 tok/s, with zero compute, DMA
setup, fragmentation, or wait included. A 90% hit rate would raise that same
ideal H2D ceiling to about 29.6 tok/s. That is not a C1 decode rate, and the
required locality is not established; a uniform cache would approach full expert
residency.

Therefore GPU 1 must own self-contained later blocks and their expert banks. GPU 0
owns embeddings, early blocks, PLE ingress, and unavoidable host traffic. The
research execution model is a topology-aware pipeline with one host-staged
activation crossing at a block boundary, no per-layer allreduce, no remote expert
ownership, and no stock vLLM TP2.

### Host upgrades that materially change the route

- 64 GiB RAM makes an INT4 PLE table operationally more comfortable.
- **128 GiB RAM is the recommended minimum upgrade**: it can hold the 47.684 GiB
  INT8 table with substantial runtime headroom, or INT4 plus a bounded expert
  cache and larger pinned buffers.
- 192–256 GiB permits a BF16 PLE reference table and meaningful W4 host cache.
- Faster/stabler memory matters because current DDR4-2133 also constrains host
  lookup and staging.

### VRAM maximization and selective mixed-precision strategy

To maximize utilization of the 64 GiB physical VRAM (2x 32,656 MiB) on dual B70s:
- **Base non-expert tensors** (Embeddings, Attention, GDN, MTP, Shared Experts, HyperConnections): ~13.5 GiB in BF16 (~8 GiB if non-GDN small domains use validated FP8).
- **Driver/OS & Runtime workspace reserve**: ~3–4 GiB total (1.5–2.0 GiB per card).
- **Target weight payload**: ~51–55 GiB total (~25.5–27.5 GiB per card).
- **KV cache & recurrent GDN headroom**: ~9–13 GiB total across cards (permits 64K–128K context).

| Expert Quantization Strategy | Expert Weight Payload | Base Tensors (BF16) | Total VRAM Footprint | KV Cache Headroom | Feasibility / Quality Target |
|---|---:|---:|---:|---:|---|
| **Uniform W4 (4.0 bits)** | 60.4 GiB | 13.5 GiB | **73.9 GiB** | *Overflow* (>64 GiB) | Infeasible on dual B70 |
| **W3 Mixed (50% W4 / 50% W2)** | 45.3 GiB | 13.5 GiB | **58.8 GiB** | ~5.5 GiB (short context) | High quality (~3.0 bits avg) |
| **W2.5 Mixed (25% W4 / 75% W2)** | 37.7 GiB | 13.5 GiB | **51.2 GiB** | **~13.1 GiB (64K–128K context)** | **Recommended balance (~2.5 bits)** |
| **Uniform W2 (2.0 bits)** | 30.2 GiB | 13.5 GiB | **43.7 GiB** | ~20.6 GiB (large margin) | Baseline capacity candidate |

Promotion policy: Allocate W4 to top/frequent/critical experts (identified via activation calibration) and early layer blocks; allocate W2 to deep/sparse experts.

### Storage audit and conversion execution strategy

The official BF16 release is 335.28 GiB (131 shards of ~2.7 GiB each).
- Host mount `/mnt/<other-disk>` exposes **320 GB free**.
- Rather than downloading all 335 GiB unquantized at once (which exceeds a single mount without deleting past artifacts), the official conversion pipeline will use **shard-by-shard streaming**:
  1. Stream/download individual shard (~2.7 GiB).
  2. Quantize target expert blocks to W4A16/W2A16.
  3. Emit compressed Safetensors shard (~600 MiB) to target workspace.
  4. Discard raw input shard.
- Peak scratch disk usage with streaming is under **15 GiB**, preserving all existing model artifacts and datasets without requiring destructive disk reclaim.

## 13. Execution gates status

1. **[PASSED]** Promote production-dimension construction fixture to permanent tests (`test_xpu_construct.py` passes with and without PLE).
2. **[PASSED]** Execute deterministic tiny XPU PLE forward on hardware:
   - Full pipeline validated: N-gram hash arithmetic, embedding lookup, linear projections, grouped RMSNorm, gated non-linearity, and dilated short-convolution fallback op.
   - Outputs verified on `xpu:0`: shape `[4, 512]`, dtype `bfloat16`, finite, bitwise deterministic across repeated forwards, sample norm ~0.5537.
   - Hardware test suite passes 5/5 tests in 21.40s (`test_xpu_portable_hc.py`, `test_xpu_construct.py`, `test_xpu_ple.py`).
   - Logs archived in `results/2026/08/qwen38-flash-next-adoption-20260826T133339Z/logs/`.
3. **[NEXT]** Add explicit QSA config rejection test, followed by XPU QSA reference op.
4. **[PENDING]** Tiny deterministic eager end-to-end model forward with bound GDN/KV state.
5. **[PENDING]** AutoRound routed-expert W4A16 slice save/reload proof.
6. **[PENDING]** Packed W2A16 Xe2 MoE DPAS kernel implementation & slice quality validation.
7. **[PENDING]** Topology-aware dual-card pipeline loader (layer ownership without cross-card expert streaming).
8. **[PENDING]** Shard-by-shard streaming conversion pipeline execution.
9. **[PENDING]** Complete no-spec generation, parity, MTP, context scaling, and claims audit.

## 14. Hard stops

- Do not download 335.28 GiB of weights onto current mounts.
- Do not use or modify the Windows system volume for this campaign.
- Do not relabel QSA as dense attention.
- Do not use stock TP2 on this host.
- Do not use `--no-mmap` or `--mlock` for an oversized tiered GGUF.
- Do not quantize PLE through ordinary AutoRound expert calibration.
- Do not publish performance claims from construction or synthetic fixtures.
- Do not delete model artifacts, results, caches, or user files without an exact
  reviewed reclaim manifest and separate authorization.
