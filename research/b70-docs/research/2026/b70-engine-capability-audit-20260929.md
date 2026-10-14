---
title: "B70 engine capability audit — EP, GDN state, flag-slot sync (static source)"
type: research
date: 2026-09-29
status: provisional
author: "substitute runtime engineer, dispatched 2026-09-29"
---

# B70 engine capability audit — 2026-09-29

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.**

Static source audit only: no GPU, no builds, no containers started, no
services. Prices three breakthrough candidates from
`research/2026/b70-breakthrough-architect-20260928.md` (B10/B11/B14/B15)
against the actual on-host engine trees. Every verdict cites file:line
from these trees, not memory or upstream docs.

## Trees audited

| Tree | Identity |
|---|---|
| `~/vllm-src/vllm` | HEAD `066ae99e` ("[Bugfix][Hardware][Intel] Disable compiled grouped_topk on XPU"), 0.27.2rc1-era, carries `VLLM_DRAFT_INT4` (`vllm/model_executor/model_loader/draft_int4.py`) |
| `~/vllm-src/vllm-xpu-kernels` | `v0.1.13-5-g4043e76` (post-0.1.13, *not* the pinned 0.1.12.3 wheel — no installed 0.1.12.3 tree found on disk; the pinned wheel lives inside the docker images) |
| `~/inference/research/src/vllm-xpu-kernels` | second checkout, `13013c5` (grouped topk #527) |
| exl3xpu fork | image-internal; `/opt/exl3xpu` extracted read-only via `docker create`+`docker cp` on `unix:///run/<exl3-docker>.sock` (container never started, created object removed). Plugin source: `/tmp/exl3xpu-top/exl3xpu/` — `vllm_patches.py` (358 lines), `vllm_plugin.py`, `exl3_ops.sycl`. In-image vLLM is `0.26.1.dev0+g568afb3a1` — an *older* generation than the pinned tree |
| `~/llama.cpp` | SYCL fork; carries `LLAMA_SPLIT_MODE_TENSOR` meta device and SYCL `moe-cache.cpp` |
| `~/llama-iq3-reorder` | exists; not needed beyond SYCL MoE question |
| oneCCL | `/opt/intel/oneapi/ccl/latest -> 2022.1` headers |
| Level Zero | `/usr/include/level_zero/ze_api.h` |

---

## Audit A — Expert-parallel (EP) on XPU (candidate B14)

### Question A1 — Does the pinned vLLM tree have EP dispatch usable on XPU?

**The EP plumbing exists; the only XPU-usable dispatch is the
AllGather+ReduceScatter collective path, which requires `dp_size > 1` —
and that is not the topology B14 priced.**

Evidence:

- `FusedMoEParallelConfig.use_ep`, `ep_size`, `ep_rank` are all wired
  (`vllm/model_executor/layers/fused_moe/config.py:1035-1254`);
  `--enable-expert-parallel` exists (`vllm/config/parallel.py:164`);
  `enable_ep_weight_filter` exists (`parallel.py:166-172`).
- The XPU communicator only instantiates an all2all manager for
  `"naive"`/`"allgather_reducescatter"`; every other backend logs a
  warning and falls back to `AgRsAll2AllManager`
  (`vllm/distributed/device_communicators/xpu_communicator.py:29-46`).
  `"naive"` itself is removed upstream and remapped to
  `allgather_reducescatter` (`vllm/config/parallel.py:466-472`).
- All device-native dispatch backends are import-gated to
  `is_cuda_alike()` — deepep HT/LL/V2, mori, nixl_ep, flashinfer NVLink
  (`vllm/model_executor/layers/fused_moe/all2all_utils.py:42-57`).
  Selecting one on XPU would pass `use_*_kernels` string checks and then
  `NameError` in `maybe_make_prepare_finalize` (`all2all_utils.py:159+`)
  because the classes are never imported — nothing XPU-side re-gates
  `all2all_backend` (`vllm/platforms/xpu.py:287-388` has no EP check).
- `use_all2all_kernels` requires `use_ep AND (dp_size>1 OR pcp_size>1 OR
  sequence-parallel)` (`config.py:1054-1058`); `use_all2all` requires
  `dp>1` or sequence-parallel-MoE (`tp>1 AND dp>1`) or EP+pcp
  (`parallel.py:688-694`). So **TP2+EP alone (dp=1) never activates the
  all2all manager**: experts shard via `expert_map`, each rank computes
  masked-local experts on all tokens, output reduces through the normal
  TP path — same collective-per-layer cost class as plain TP2.
- The only true EP dispatch on XPU: `AgRsAll2AllManager.dispatch` =
  `dist_group.all_gatherv` of hidden+topk to *all* ranks, then
  `combine` = `reduce_scatterv` of outputs
  (`vllm/distributed/device_communicators/all2all.py:43-149`), wrapped by
  `MoEPrepareAndFinalizeNaiveDPEP*`
  (`prepare_finalize/naive_dp_ep.py:158-164, 207-208`). Transport is
  torch.distributed → ProcessGroupXCCL → oneCCL — host-staged on this
  host (~102 µs/4 KB class, per RL-075 pricing).
- The XPU fused-MoE kernel itself is EP-aware
  (`vllm-xpu-kernels/vllm_xpu_kernels/fused_moe_interface.py:186-305`:
  `ep_rank`, `ep_size`, `expert_map`, `init_expert_map` op at :301) —
  local-expert masking is not the gap; dispatch is.
- EPLB is hard-gated off XPU: `parallel.py:492-497` —
  `"Expert parallelism load balancing is only supported on CUDA devices
  or ROCm devices now."`
- DeepSeek-V4's MegaMoE — the one model-native EP dispatch in the XPU
  model dir, symmetric-buffer + `fp8_fp4_mega_moe` over `get_ep_group()`
  (`vllm/models/deepseek_v4/xpu/model.py:465-554`) — is **stubbed off on
  XPU**: `_check_runtime_supported` raises
  `NotImplementedError("DeepSeek V4 MegaMoE is not supported on XPU.")`
  (`xpu/model.py:424-425`). The `#56096` lesson verbatim: the dispatch
  exists in the tree but is not available here.
- Bonus XPU-only path: `batched_triton` MoE gets a *local* regroup
  `BatchedPrepareAndFinalize` — explicitly "no all-to-all"
  (`all2all_utils.py:130-138`). Not EP.

### Question A2 — Does the exl3xpu fork change the answer?

No. Zero matches for `ep_size|expert_parallel|all2all|deepep|
expert_map` in the extracted plugin (`/tmp/exl3xpu-top/exl3xpu/*.py`,
`csrc/exl3_ops.sycl`). The fork is single-GPU qwen3_5 dense/hybrid
(EXL3 trellis + MTP3); its patches are GDN metadata, fp8-KV prefill,
and mamba page alignment. In-image vLLM is 0.26.1 — older than pinned.

### Question A3 — llama.cpp SYCL expert splitting?

- `-sm tensor` (this fork's EXPERIMENTAL meta-device TP mode,
  `src/llama.cpp:160-210`, `ggml-backend-meta.cpp`) splits weight
  tensors along matmul axes 0/1 only; `MUL_MAT_ID` goes through
  `handle_mul_mat` which knows only axis-0/axis-1/mirrored/partial
  splits (`ggml-backend-meta.cpp:577-597, 873-875`) — the expert axis
  (ne[2]) is never split. This is TP of each expert matrix with
  partial-result reduce, not expert placement. Supported arch list
  includes QWEN35MOE/DEEPSEEK4/NEMOTRON_H_MOE
  (`src/llama-arch.cpp:1012-1039`).
- SYCL `--moe-cache`: LRU cache of *host-resident* experts in leftover
  VRAM; `moe_cache_pick_device` selects **one** device (largest budget)
  per node (`ggml/src/ggml-sycl/moe-cache.cpp:382-393`); dispatch is
  host-staged memcpy + `mul_mat_vec_q` + D2H
  (`moe-cache.cpp:555-591`). The `expert_parallel` field in
  `ggml_moe_cache_config` (`ggml-backend-moe-cache.h:22`,
  set at `ggml-backend.cpp:1916`) is **never consumed** by the SYCL
  provider — dead parameter.
- `-ot` places whole `_exps` tensors per device but cannot split one
  expert tensor across devices.
- `ggml-backend.cpp:1665-1750` copies only *used* expert ranges during
  split offload — CPU→GPU caching, not cross-GPU routing.

### Audit A verdict

B14's arithmetic (activations-only crossings, µs-class) describes a
**device-resident per-token dispatch** that no engine on this host
implements. What exists is collective-class: AgRs EP moves *all* tokens
through allgather/reduce_scatter on host-staged xccl — per-MoE-layer
cost is the same 100 µs-class latency that killed TP2, i.e. the
existing XPU EP would not beat TP2, it *is* TP2's transport wearing
expert metadata.

| Claim | Status | Evidence | Pricing impact |
|---|---|---|---|
| EP dispatch primitive exists on XPU | ENGINE-ANCHOR EXISTS but wrong class | `xpu_communicator.py:29-46`, `all2all.py:43-149`, `naive_dp_ep.py:158-208` | A priceable baseline exists (DP2+EP2 via AgRs), but its transport is xccl collectives — it cannot realize B14's µs-class crossings |
| deepep/native all2all usable on XPU | NO | `all2all_utils.py:42-57` import-gate; `xpu_communicator.py:36-46` fallback | CUDA-only |
| Model-native EP (MegaMoE symm-buffer) | NO — stubbed | `deepseek_v4/xpu/model.py:424-425` NotImplementedError | Would need enabling + it targets deep_gemm/fp4 path |
| Per-token device-resident routing (B14's mechanism) | BUILD-PROGRAM REQUIRED | no counterpart exists in any audited tree | Sized below |
| llama.cpp EP | NO | `ggml-backend-meta.cpp` axis-0/1 only; `moe-cache.cpp:382-393` single device | `-sm tensor` is TP; nothing to anchor |
| Placement verdict for next MoE | ARITHMETIC-ONLY | — | Record: for >64 GB MoE on dual B70, EP is the topology that matches copy-only P2P, but it must ride on B11's flag-slot primitive |

**Build-program size (honest):** router-metadata exchange (topk_ids,
~bytes), per-token dispatch/combine kernels driven by flag-slot sync
(B11), expert residency split via `expert_map` (exists), plus a
weight-loader shard path (`enable_ep_weight_filter` exists,
`parallel.py:166`). The unfamiliar/hard parts are the dispatch kernel +
ordering protocol, not the config plumbing. Realistically a vllm-xpu
EP prototype = new prepare/finalize backend + P2P copy/flag-slot
transport — weeks, not days; gated on B11 microbench first.

---

## Audit B — GDN constant-state handling (candidate B15)

Audited on the pinned 0.27.2rc1-era tree; qwen3_5 is
`vllm/model_executor/models/qwen3_5.py` (`QwenGatedDeltaNetAttention`
at `qwen_gdn_linear_attn.py:348-393`, dedicated `forward_xpu` at
:381-382).

### B1 — Is the per-request GDN state a first-class object?

**Yes — it is a paged KV-cache spec, not an ad-hoc buffer.**

- `MambaSpec` (`vllm/v1/kv_cache_interface.py:747-797`) — each GDN
  layer registers `get_kv_cache_spec` returning MambaSpec with
  `(conv_state, ssm_state)` shapes
  (`vllm/model_executor/layers/mamba/abstract.py:62-78`;
  shape math `mamba_utils.py:247-268`).
- State size is per-*page*: `page_size_bytes` = sum over shapes
  (`kv_cache_interface.py:756-765`). The exl3 fork's patch comment
  confirms ~3.1 MiB/layer on qwen3.8-class GDN
  (`/tmp/exl3xpu-top/exl3xpu/vllm_patches.py:221-222`) → ~150 MB/req
  across 48 layers, consistent with B15's number.
- Three residency modes via `cache_config.mamba_cache_mode`
  (`MambaSpec.max_memory_usage_bytes`, `kv_cache_interface.py:767-776`):
  - `"none"`: 1 + num_speculative_blocks pages per request.
  - `"align"`: 2 + num_speculative_blocks resident; position-indexed
    block table, skipped states nulled
    (`kv_cache_interface.py:778-788`, `single_type_kv_cache_manager.py:1442-1459`).
  - `"all"`: one state snapshot per block boundary — mamba prefix
    caching.
- `num_speculative_blocks = num_speculative_tokens`
  (`abstract.py:73-77`) — MTP3 ⇒ 3 extra state slots per request for
  spec-decode state rollback.

### B2 — Is there a checkpoint/serialize/migrate path?

**Partially — two real paths exist:**

- **Prefix-cached snapshots** (mode `"all"`, and `"align"` partial-hit
  CoW): a mamba hit needs exactly the single state block ending on the
  boundary (`single_type_kv_cache_manager.py:1403-1422`). This *is*
  constant-cost checkpoint/resume: resume from a boundary = one ~3.1
  MiB/layer block, not a KV replay. `reachable_block_mask`
  (`:1373-1429`) already implements *sparse retention* — keep one state
  per `retention_interval` plus proven-reuse boundaries.
- **Connector offload**: `_pending_partial_tail_offloads` →
  `kv_cache_manager.take_partial_tail_offloads`
  (`kv_cache_manager.py:857-879`) → mooncake worker
  `_maybe_offload_partial_tail` moves the mamba boundary CoW block
  (`distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:566-620`).
  NIXL also handles hybrid-mamba prefix state
  (`kv_cache_manager.py:300-322` comment; `nixl_integration/
  test_mamba_prefix_cache.py` exists in tests).
- **Preemption**: worker-side `cleanup_mamba_state_idx` pops
  finished/preempted/resumed ids
  (`v1/worker/mamba_utils.py:1110-1125`); a preempted request that
  loses blocks recomputes, or resumes from a cached boundary state if
  prefix caching kept one. No general "freeze live request to host"
  path — but the building blocks (page-addressable state + connector
  offload) exist.
- Generic `v1/kv_offload/` tiering does **not** reference mamba — the
  only offload lanes are the connector paths above.

### B3 — The external "E6" variant (1 state block per request)?

**Mostly already in this tree.** "align" mode is exactly the
sparse-resident layout: 1 running + num_spec blocks resident
(position-indexed table, earlier states nulled). With MTP3 that's
4 resident + optional CoW; without spec it's 2. E6's "1 block" is one
notch tighter than align (drops/merges spec slots) but the mechanism —
single resident state + boundary snapshots — is native here, not
fork-specific. Also note `Qwen3_5MTP` *rejects* `mamba_cache_mode="all"`
(`qwen3_5_mtp.py:224-228`) — align is the MTP-required mode. The MTP
draft layers are `full_attention` only (`qwen3_5_mtp.py:115-122`) —
**the draft carries no GDN state at all**, which matters for B10/B15d.

### B4 — Hard assumptions for tiering / cross-device state

- **"Mamba state is replicated across DCP/PCP ranks, never sharded"**
  (`kv_cache_interface.py:779-780`); MambaManager asserts
  `dcp_world_size == 1` and `pcp_world_size == 1`
  (`single_type_kv_cache_manager.py:1310-1311`). Cross-device tiering
  cannot ride DCP; it must go through a KV-connector-style block copy —
  pages are ordinary pool tensors, so a P2P-copy connector to GPU1 is
  writable but does not exist.
- Worker-side state indexing is per-request dict + device tensors
  (`mamba_utils.py:1162-1290`, `mamba_state_idx`); nothing pins state
  to a NUMA/peer-visible allocation class — GPU1 tiering needs an
  allocator/pool change or connector path.
- Graph capture: GDN builder supports `UNIFORM_BATCH` cudagraph
  (`gdn_attn.py:83`) with preallocated index tensors (:126-165); a
  tiering copy inside capture is not supported today (copy would need
  fixed addresses).
- exl3xpu fork evidence (older base): it had to *patch* mask-index
  host syncs and skip a mamba-align preprocess sync
  (`vllm_patches.py:5-69, 265-291`) — the pinned tree carries the
  corresponding machinery natively (async `prepare_chunk_indices`
  h2d at `gdn_attn.py:189-207`, align preprocess in
  `mamba_utils.py:863-1226`); whether the pinned tree still
  host-syncs on XPU needs a runtime check, not derivable statically.

### Audit B verdict

| Claim | Status | Evidence | Pricing impact |
|---|---|---|---|
| GDN state is first-class, constant-size | CONFIRMED | `kv_cache_interface.py:747-776`, `mamba_utils.py:247-268` | (a) is real: checkpoint = copy of 1+spec pages/req |
| Checkpoint/migrate path exists | PARTIAL | boundary snapshots + connector offload (`single_type_kv_cache_manager.py:1373-1429`, `mooncake worker.py:566-620`); no generic live-request tiering | (a)/(b) need a new connector lane; not a green field |
| E6 "1 block/req" novelty | MOSTLY PRESENT | align mode = sparse-resident native (`kv_cache_interface.py:773-787`); MTP forces "align" (`qwen3_5_mtp.py:224-228`) | The +53 % KV-token class win is already the align design; remaining delta is spec-block trimming |
| Draft state sharing for cross-device draft | NO HARD BLOCKER for B10 | draft = full-attn only (`qwen3_5_mtp.py:115-122`); spec GDN rollback slots are target-side | B10's "hidden state + ids cross" design is consistent with the tree |
| Cross-device GDN tiering | HARD ASSUMPTION | `single_type_kv_cache_manager.py:1310-1311` asserts; "replicated never sharded" | Needs connector/P2P-copy lane, not DCP |

---

## Audit C — Flag-slot (seqno) sync, graph-safety pre-check (candidate B11)

### C1 — XPU graph capture: what breaks / is forbidden?

- Capture is `torch.xpu.graph`/`XPUGraph` aliased into the CUDA wrapper
  (`v1/worker/xpu_model_runner.py:59-63`); gate:
  `VLLM_XPU_ENABLE_XPU_GRAPH` default **0** (`envs.py:309, 2094-2095`)
  and `supports_xpu_graph()` needs torch ≥ 2.11.0.dev
  (`utils/torch_utils.py:1018-1019`).
- Platform explicitly warns: "XPU Graph support is experimental and
  currently only supports single-GPU execution"
  (`platforms/xpu.py:311-314`) — a **warning, not a hard gate**. No code
  in the tree force-disables graphs under TP>1 on XPU.
- Fusion passes disabled on XPU include `fuse_gemm_comms` ("Async TP")
  (`xpu.py:320-335`) — the only capture-adjacent collective path.
- Capture-safe collectives: TP all_reduce goes through the opaque
  custom op `torch.ops.vllm.all_reduce`
  (`distributed/parallel_state.py:151-160, 680-688`) →
  `XpuCommunicator.all_reduce` → `dist.all_reduce` (xccl). Whether xccl
  ops record into an XPUGraph depends on ProcessGroupXCCL enqueueing on
  the capture stream — **not statically determinable** (torch XPU
  wheel not vendored). Positive signal: oneCCL 2022.1 headers
  acknowledge graph recording — "If the communicator was used during
  graph recording, [finalize] will log a warning and do nothing. Users
  must manually synchronize graph submissions"
  (`/opt/intel/oneapi/ccl/latest/include/oneapi/ccl.h:216-218`) — i.e.
  the collective library has a graph-recording mode, so recorded
  collectives are at least an API possibility.
- Standard stream-capture invariants apply: host syncs
  (`.item()/.tolist()`/D2H), event queries, dynamic shapes break or
  corrupt capture. The tree carries capture-detection guards that
  *silently skip* on non-CUDA —
  `fused_batched_moe.py:40-42` notes `is_current_stream_capturing` was
  unavailable on XPU torch (guarded by `is_cuda_alike()`) — gaps exist
  where capture-awareness is compiled out for XPU.
- `torch.xpu.Event` lacks the `blocking` kwarg — wrapper drops it
  (`xpu_model_runner.py:53-58`). Minor but shows the shim layer is thin.

### C2 — Device-side spin/poll precedent

- **No spin/poll/flag kernels in either vllm-xpu-kernels tree**
  (grep over `csrc/` clean — only unrelated "flag" constants).
- XPU kernels launch via `at::xpu::getCurrentXPUStream().queue()` —
  same-stream SYCL submissions (e.g.
  `vllm-xpu-kernels/csrc/xpu/grouped_gemm/xe_2/grouped_gemm_xe2_interface.hpp:188`)
  → a custom op whose kernel *spins on its own VRAM* is capture-recordable
  in principle (it's just a kernel node on the capture stream).
- oneCCL path contains no discoverable flag-slot primitive in headers;
  the finalize note implies graph integration exists at 2022.1.

### C3 — Memory-ordering of P2P zeCopy (static determinability)

- **In-order execution is specified**: `ZE_COMMAND_QUEUE_FLAG_IN_ORDER`
  / `ZE_COMMAND_LIST_FLAG_IN_ORDER` — "commands appended to this command
  list are executed in-order, with driver implementation enforcing
  dependencies between them" (`/usr/include/level_zero/ze_api.h:3036-3044,
  3369-3374`). A data copy followed by a seqno copy on one in-order list
  is ordered at the API level; cross-queue ordering is available via
  `hSignalEvent`/`phWaitEvents` on `zeCommandListAppendMemoryCopy`
  (`ze_api.h:4051-4061`).
- **What headers do NOT answer**: whether a *concurrently spinning*
  kernel on the reader device observes peer-VRAM writes (copy-engine
  writes vs. reader's L2/uncached loads) — a device-scope memory-model
  question. `ZE_MEMORY_ACCESS_CAP_FLAG_*` (ze_api.h:2329-2333) covers
  capability bits (B70: RW yes, ATOMIC no — verified RL-059), not
  ordering-to-concurrent-observer semantics.
- Microbench ordering test cases needed: (1) flag-after-data visibility
  under spin load; (2) partial-torn observation of the data payload when
  flag lands (does flag imply data?); (3) stale-read rate under L2
  pressure; (4) same behavior through `zeCommandListAppendMemoryCopy`
  vs. a device-side memcpy kernel writing peer VA; (5) seqno monotonicity
  under back-to-back rounds.

### Audit C verdict

**AMBER** — no hard blocker found; two named unknowns.

| Claim | Status | Evidence | Pricing impact |
|---|---|---|---|
| Something forbids flag-slot pattern in XPU graphs | NO hard block found | graphs are warning-gated single-GPU (`xpu.py:311-314`), default OFF (`envs.py:309`); no multi-GPU force-disable in code | Microbench + a two-rank capture test decide |
| Spin/poll precedent in fleet kernels | NONE | grep clean over both vllm-xpu-kernels `csrc/` | Greenfield kernel; keep it a standalone op outside captured graphs initially |
| Writer ordering data→flag on one in-order list | SPECIFIED | `ze_api.h:3369-3374` in-order + event chaining `:4051-4061` | Design constraint satisfied at API level |
| Reader-side visibility of peer writes to a spinning kernel | UNKNOWN — not in headers | `ze_api.h:2329-2333` covers access caps only | This IS the microbench's core question |
| xccl collectives inside XPU graph capture | UNKNOWN | oneCCL 2022.1 graph-recording note (`ccl.h:216-218`); torch-side enqueue path not auditable statically | If xccl is capturable, hybrid flag-slot+xccl fallback is possible |

**Practical guidance for B11:** build the spin-copy microbench outside
vLLM as planned (pure SYCL/L0, two in-order queues); the riskiest item
is reader-side visibility semantics, not ordering. A 60 µs kill
threshold stands.

---

## Cross-audit findings

- **B14 ↔ B11 are coupled harder than the breakthrough doc states**:
  the only XPU EP dispatch (AgRs) is collective-class — B14's advantage
  doesn't exist without B11's cheap sync. B14 pricing = "build EP on
  flag-slots"; there is no middle rung (stock EP on xccl ≈ TP2 transport).
- **B10 gets a free fact**: the qwen3_5 MTP draft is full-attention-only
  (`qwen3_5_mtp.py:115-122`) — cross-device drafting needs no GDN-state
  sharing; only the target needs GDN spec blocks (which exist:
  `num_speculative_blocks`, `abstract.py:73-77`).
- **The pinned tree already carries most of B15(a)**: mamba "align"
  sparse residency + boundary CoW + connector offload — the E6 external
  variant is incremental over what this tree does, not novel.
- **vllm-xpu-kernels pinned wheel 0.1.12.3 not on disk** — vendored
  trees are 0.1.13-dev; for any kernel-level audit against the exact
  pinned wheel, extract from the image as done for exl3xpu.
