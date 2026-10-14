---
title: "humble-b70-llm dual-B70 TP2 stack vs this host"
status: final
type: research
date: 2026-08-24
---

---
id: b70-research-humble-b70-llm-dual-tp2-20260824
title: humble-b70-llm dual-B70 TP2 stack vs this host — 2026-08-24
document_type: research
status: complete
authority: docs/methods/benchmark-protocol.md
scope: External JP-devv/humble-b70-llm recipe captured as dual-card serving evidence; comparison to b70-host topology and Qwen3.8 routes; not a local campaign and not a public headline
truth_verdict: PARTIAL
evidence_tier: E0
publication_gate: PROVISIONAL_NOT_FOR_PUBLIC_HEADLINE
observed_at: 2026-08-24
source_url: https://github.com/JP-devv/humble-b70-llm
---

# humble-b70-llm dual-B70 TP2 stack vs this host

Captured 2026-08-24 from the public repository
[JP-devv/humble-b70-llm](https://github.com/JP-devv/humble-b70-llm) (Apache-2.0)
and compared to live `b70-host` topology plus the current Qwen3.8-27B
vLLM route.

This note is **external-source research**, not a B70 campaign. Their tok/s
figures are **their** published rows. They are not this host's measurements,
not a matched comparison, and not a public headline for Intel Arc Pro B70
on this board. Local topology evidence remains
[`research/2026/b450f-dual-b70-pcie-topology-20260824.md`](b450f-dual-b70-pcie-topology-20260824.md).
Serving policy is [`decisions/adr-0008-dual-b70-independent-default.md`](../../decisions/adr-0008-dual-b70-independent-default.md).

## 1. What the repository is

A pinned, quality-gated **vLLM XPU** stack for one checkpoint:

`johannrplaster/Qwen3.8-27B-Uncensored-int4-AutoRound`

Dual-card here means **tensor-parallel 2** (one model split across both GPUs),
not two independent servers. llama.cpp is not in the repo. Docker is not the
serve path. Data-parallel / multi-user continuous batching is not the
published product (`--max-num-seqs 1`).

Layout at capture:

```text
README.md  Makefile  LICENSE
bench/realistic-suite-v1.json
bench/reference/*.json
docs/{QUICKSTART,drivers,hardware,metrics-contract,optimizations,troubleshooting}.md
patches/vllm/0001-humble-b70-production-deltas.patch
patches/vllm-xpu-kernels/0001-int8-w8a8-lmhead-backport.patch
scripts/{setup-host,build,model,serve,verify-install,bench-strict,quality-gate}.*
```

Pinned bases recorded in `patches/*/BASE_COMMIT.txt`:

| Tree | Commit |
|---|---|
| vLLM | `8e6d8e4f6a0c84db0d79129ab648492edf640fe2` |
| vllm-xpu-kernels | `27214a3d99c8c4f60f4a15ecb7d3c2febef5bc57` |

AOT SYCL wheels are smoke-tested, not hash-gated.

## 2. How they launch two cards

Native host venv, not Docker:

```bash
bash scripts/setup-host.sh
bash scripts/build.sh
bash scripts/model.sh fetch
bash scripts/model.sh verify
bash scripts/serve.sh            # default --tp 2
bash scripts/serve.sh --tp 1     # one card
bash scripts/verify-install.sh
bash scripts/bench-strict.sh
bash scripts/quality-gate.sh
```

### 2.1 Serve defaults (`scripts/serve.sh`)

| Axis | Default |
|---|---|
| Parallelism | `--tensor-parallel-size 2` (`--tp 1` for one card) |
| Spec | MTP3, `method=qwen3_5_mtp` |
| Weights | `--quantization gptq`, `--dtype float16` |
| KV | FP16 unless `--kv fp8` |
| LM head | load-time INT8 W8A8 (`VLLM_XPU_LM_HEAD_INT8=1`) |
| Context | `--max-model-len 8192` |
| Batch | `--max-num-seqs 1`, `--max-num-batched-tokens 8192` |
| Util | `0.95` TP2 / `0.88` TP1 |
| Graphs | `VLLM_XPU_ENABLE_XPU_GRAPH=1` |
| Bind | `0.0.0.0`, port `19622` |
| Served name | `Qwen3.8-27B-UNC-G128-AR` |
| TP1 device | `ZE_AFFINITY_MASK=<gpu>`, `ONEAPI_DEVICE_SELECTOR=level_zero:0` |

Checkpoint notes from their docs: AutoRound INT4 W4A16 g128, MTP head kept,
linear-attention / GDN projections left FP16. Approximate weight size ~18 GB.

### 2.2 Dual-GPU environment (the load-bearing part)

Always set:

```text
VLLM_TARGET_DEVICE=xpu
VLLM_XPU_ENABLE_XPU_GRAPH=1
VLLM_XPU_HOST_STAGED_COLLECTIVES=1
CCL_TOPO_P2P_ACCESS=0
CCL_ZE_IPC_EXCHANGE=pidfd
CCL_ATL_TRANSPORT=ofi
ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
```

Optional: `VLLM_XPU_LM_HEAD_INT8=1`, `KV_CACHE_DTYPE` / `--kv-cache-dtype`,
`VLLM_XPU_HOST_STAGED_MIN_BYTES` (default 1 MiB).

**Host-staged collectives:** `XpuCommunicator` copies large XPU tensors into
pinned CPU buffers, runs the collective on the gloo/`cpu_group` path, copies
back. Small tensors stay on the device group. Motivation in the patch comment:
cross-device VRAM IPC is unavailable (`zeMemOpenIpcHandle` /
`ZE_RESULT_ERROR_INVALID_ARGUMENT` on consumer boards).

**Graph rule:** those host-staged comms must stay **outside** captured XPU
graphs (piecewise capture). Putting them inside capture is how they lost the
engine on TP2 hosts without peer IPC (“kernel timeout ~270 GPU submissions
into the decode capture”). `CUDAGraphWrapper` may use `graph_pool = None`;
`VLLM_XPU_SEPARATE_GRAPH_POOLS` exists because graph-pool sharing across
pieces corrupted batched replay. `weak_ref_output` flipped to false.

### 2.3 Production patch surface (`patches/vllm/0001-humble-b70-production-deltas.patch`)

Files touched: `cuda_graph.py`, `xpu_communicator.py`,
`vocab_parallel_embedding.py`, `qwen3_5_mtp.py`, `gdn_attn.py`,
`gpu_model_runner.py`, `mamba_utils.py`.

Env knobs introduced or used: `VLLM_XPU_HOST_STAGED_COLLECTIVES`,
`VLLM_XPU_HOST_STAGED_MIN_BYTES`, `VLLM_XPU_SEPARATE_GRAPH_POOLS`,
`VLLM_XPU_GRAPH_DUMP` / `_PATH`, `VLLM_XPU_LM_HEAD_INT8` (+ scope/chunk/scale),
`VLLM_XPU_DRAFT_LM_HEAD_INT4` (+ group/chunk/scale/fallback),
`B70_MTP_BF16_DRAFT`, GDN debug flags, `VLLM_XPU_VERIFY_TRACE`,
`VLLM_XPU_STEP_TIMING`.

Other B70-relevant fixes in the same patch: incomplete last speculative group
rerouted onto the non-spec prefill path; Mamba `_to_s64` stores device pointers
as signed int64 because XPU VMM addresses can exceed `2**63-1`.

Kernel-side: `patches/vllm-xpu-kernels/0001-int8-w8a8-lmhead-backport.patch`
for the INT8 W8A8 lm_head GEMM.

INT8 conversion is **row-chunked** (`VLLM_XPU_LM_HEAD_INT8_CHUNK_ROWS`, default
4096). A full `float()` copy of the untied Qwen3.8-27B head (~2.54 GB FP16,
248320×5120) is a ~5 GB spike that OOMs one card (~2.5 GB per TP2 rank).

## 3. Their published numbers

Two protocols. Do not mix them with each other or with this host.

### 3.1 Warm LocalMaxxing (external, APPROVED on their submissions)

Warm-up, then median of 3 greedy iterations, client post-first:

| Config | Decode tok/s | TTFT | tokSPrefill | Submission |
|---|---:|---:|---:|---|
| TP2 — 2× B70 | 136.4 | 841 ms | 755 | `cmt0vu76q0fvtms017exhssx9` APPROVED |
| TP1 — 1× B70 | 97.8 | 356 ms | 1782 | `cmt0vrxjk0fvpms01n4i9cmns` APPROVED |

LocalMaxxing `APPROVED` means accepted/published self-report, not independent
reproduction on this host.

### 3.2 Cold one-shot (in-repo harness)

Prefix/KV reuse off (`--no-enable-prefix-caching`); each call must show
`prompt_tokens_details.cached_tokens == 0`. Headline rate is the median
inter-token interval between generated-token timestamps 1 and 100 after TTFT
(`scripts/bench-strict.py` against `bench/realistic-suite-v1.json`).

| Serving config | Decode tok/s (cold) | Reference JSON |
|---|---:|---|
| TP2 + MTP3 + FP16 KV + INT8 lm_head | 94.58 | `tp2-mtp3-int8-head-94.58.json` |
| TP2 + MTP3 + FP16 KV + FP16 head | 81.81 | `tp2-mtp3-fp16-head-81.81.json` |
| TP1 + MTP3 + FP16 KV + INT8 lm_head | 66.84 | `tp1-mtp3-int8-head-66.84.json` |

INT8 W8A8 lm_head at load is the shipped +15.6% decode vs FP16 head on that
cold protocol. Quality canaries (including `sum(i*i for i in range(4)) → 14`)
are recorded as passing for both heads.

Rejected on the same cold protocol:

| Experiment | Result | File |
|---|---|---|
| MTP4 instead of MTP3 | 79.84 vs 81.81 (FP16 head) | `tp2-mtp4-fp16-79.84-REJECTED.json` |
| INT4 draft LM head + INT8 target head | 91.84 vs 94.58 | `tp2-mtp3-int8+int4draft-91.84-REJECTED.json` |

MTP4 lost at ~45% accept on mixed prompts. INT4 draft saved bandwidth but
dropped accept enough to lose. Warm same-shape repeats on one card can look
like 80+ tok/s; they treat that as repeat traffic, not a cold headline.

Reproduction target in the README is the **cold** row (about 45 minutes if
the host is ready), band default ±3% on the median. `torch.compile` caches
are not byte-identical across machines.

### 3.3 Hardware they claim for those rows

- Verified GPU: Arc Pro B70 32 GB only. Other Arc/XPU SKUs are not claimed.
- TP2 cold 94.58 / 81.81: 2× B70 + ReBAR. Second slot **may be PCIe x4**.
- TP1 cold 66.84: 1× B70 + ReBAR, any working slot.
- Host: Ubuntu-family Linux (26.04 tested); **64 GB RAM recommended** (SYCL
  compile ~7–12 GB per job, default 6 jobs; 32 GB with `MAX_JOBS=3` is
  unvalidated); ~90 GB disk; Battlemage-capable `xe` + Intel compute runtime.
- P2P is optional. Host-staged collectives still reproduce the TP2 cold row.
  Working peer IPC can fold comms into graphs and raise decode; that gap is a
  motherboard/CPU property.
- Power: hardware cap reports ~230–275 W depending on platform. Dense prefill
  scales with the cap; decode is comparatively flat.
- Long context: FP16 KV at 128K does not fit one dense 27B card; use FP8 KV.
  200K+ can load without safe headroom; 256K is called infeasible.

## 4. This host, side by side

Live topology:
[`research/2026/b450f-dual-b70-pcie-topology-20260824.md`](b450f-dual-b70-pcie-topology-20260824.md).
Qwen3.8 route: [`models/qwen/qwen3.8-27b/`](../../models/qwen/qwen3.8-27b/).

| Axis | humble-b70-llm (external) | b70-host (this repo) |
|---|---|---|
| Board class | consumer, no GPU P2P required; 2nd slot may be x4 | ASUS B450-F; GPU 0 Gen3 **x8** (~7.1 GB/s), GPU 1 Gen3 **x4** (~3.6 GB/s); no ACS / no usable P2P |
| ReBAR | required, full 32 GB | Active on both cards, BAR 2 = 32 GB |
| RAM | 64 GB recommended for the SYCL kernel build | 4×8 GB DDR4-2133, ~30 Gi usable |
| Dual product | one model, TP2, both cards | independent stacks, `ZE_AFFINITY_MASK` |
| Engine packaging | patched vLLM in a venv | Docker vLLM XPU + llama.cpp SYCL |
| Qwen3.8 checkpoint | uncensored AutoRound INT4 W4A16 g128 | `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` |
| Spec default | MTP3 | MTP4 on the current dense route |
| KV default | FP16 (fp8 for long context) | FP8 KV on the recorded 131K stack |
| Prefix cache | off for the cold headline | on in Run 47/48 champion |
| Concurrency | `--max-num-seqs 1` | C1 and C5 campaigns |
| LM head | INT8 W8A8 at load | stock BF16 MTP / FP16-class head |
| Collectives | host-staged patch | stock image; TP2 not a serving lane |
| GPU 1 idle | both cards occupied by TP2 | `b70-gpu1 off` power-remove path exists |
| Desktop | not specified | GPU 0 carries display + compositor |

Their TP2 existence proof is aimed at **this class of board**: no P2P, x4
second slot, ReBAR on. The earlier topology note that “vLLM TP=2 is not
recommended… expect a large fraction of single-GPU throughput” was a physics
guess without their measured host-staged path. It is corrected in §6 of the
topology document and by ADR-0008.

### 4.1 Scaling they measured (still their numbers)

On the same cold protocol: 94.58 / 66.84 ≈ **1.41×** decode TP2 vs TP1.
On the warm LMX protocol: 136.4 / 97.8 ≈ **1.39×** decode.
TTFT **worsens** (841 vs 356 ms). LMX prefill **worsens** (755 vs 1782).

That pattern matches this board: per-token allreduce is small enough that
decode can still win; prefill activations bounce through host RAM over the
x4/x8 links and do not scale.

### 4.2 Do not race their rows against Run 48

This host’s current Qwen3.8 recommended non-thinking C1 is **102.61 tok/s**
(Run 48 n=5, prefix-on, later champion stack). Comparison class:
`not_comparable` against their cold 66.84 or warm 97.8.

Different checkpoint, converter (GPTQ vs AutoRound), MTP depth, KV dtype,
prefix-cache policy, sampling, statistic, and (for TP2) card count. Keep
the numbers in separate tables.

## 5. Dual-inference product matrix for this host

Three different products. Pick the objective first.

### A. Independent stacks (current default)

- One engine per card via `ZE_AFFINITY_MASK=0` / `1`.
- Zero inter-GPU traffic. Each card keeps its full 32 GB.
- GPU 1 can be PCI-removed when idle (`b70-gpu1`).
- Fits two users, two models, or a clean research card plus desktop.
- x4 on GPU 1 only hurts **cold load** (~2× vs GPU 0), not resident decode.

### B. vLLM TP2, host-staged (external existence proof; local unmeasured)

Use when the objective is **one faster C1 Qwen3.8-class stream** and both
cards may be occupied.

Required, copied from their working recipe — not yet reproduced here:

1. Patched vLLM + kernels at or equivalent to the pinned commits, including
   host-staged collectives and piecewise graph capture.
2. `VLLM_XPU_HOST_STAGED_COLLECTIVES=1` and the CCL/ZE env block in §2.2.
3. Do not capture host-staged allreduce inside XPU graphs.
4. `--tensor-parallel-size 2`, `--max-num-seqs 1` until a local Cn gate
   exists.
5. INT8 lm_head if chasing their 94.58-class cold row; convert row-chunked.
6. Empty-GPU gate on **both** cards; no second engine.
7. Expect ~1.4× decode vs one card, slower TTFT, slower prefill, both cards
   busy, no GPU 1 idle-off.
8. Compile RAM is tight on 32 GB; `MAX_JOBS=3` is unvalidated even for them.

Local derived images are not public recipes (`AGENTS.md` §7). A reproduction
campaign must pin digest or source commit, record patches as upstream /
required / local-only, and keep prefix-cache counters at zero for any cold
headline.

### C. llama.cpp layer/tensor split (topology-viable, not this repo’s product)

Per-token cross-card traffic is one activation. Decode-focused split can
open combined ~64 GB VRAM for models that do not fit one card. Prefill will
scale well below 2×. Not present in humble-b70-llm. Not a current launcher
profile.

### Anti-patterns

- Stock Docker vLLM TP2 **without** the host-staged / piecewise-capture
  patches on this board: IPC failure or graph-capture device loss is the
  expected failure, not a silent 2× win.
- Calling TP2 “dual inference” when the user wanted two independent
  endpoints.
- Summing two per-card C1 rates as a TP2 wall aggregate, or publishing their
  136.4 / 94.58 as this host.
- Mixing their cold ITL-median with LocalMaxxing warm post-first or with
  this repo’s prefix-on n=5 medians.

## 6. What they ship that this host has not measured

Open local gates, in increasing cost:

1. Confirm Level Zero peer IPC is still absent (`zeMemOpenIpcHandle` / no
   ACS on the GPU bridges). Topology already says no usable P2P.
2. Build or overlay the host-staged collective + INT8 lm_head patches
   against a pinned vLLM/kernel pair. Record RAM peak; 32 GB may force
   `MAX_JOBS=3`.
3. Load `johannrplaster/Qwen3.8-27B-Uncensored-int4-AutoRound` **or** apply
   only the serving patches to the current GPTQ artifact — those are
   different experiments.
4. TP1 matched cold cell on GPU 0, then TP2, with prefix-cache hit delta
   = 0, quality canary, and energy-aligned power.
5. Separate INT8 vs FP16 head A/B on **this** checkpoint.
6. Do not promote TP2 to a launcher profile or public headline until that
   campaign exists and the claims audit passes.

INT8 lm_head is independently interesting on **one** card (their TP1 66.84
vs our GPTQ BF16-head route). That A/B does not require TP2.

## 7. Quality and correctness they require

Before a rate counts, their quality gate must pass: canary
`sum(i*i for i in range(4)) → 14`, plus copy, fact, schema, long-context
needle, and stability checks (`scripts/quality-gate.sh`,
`bench/reference/quality-fp16-head.json` and `quality-int8-head.json`).
Speculation is allowed only when the target re-scores every kept draft
token (vLLM MTP treated as exact).

A local reproduction must run an equivalent gate. Their canary is not a
substitute for this repo’s correctness validation method.

## 8. Operational notes for agents

- Dual-card **default** remains independent stacks. See ADR-0008.
- TP2 occupies both cards. The empty-GPU gate and “one declared engine”
  rule still apply: TP2 is one engine, two devices, not a second campaign.
- After any TP2 experiment: stop that server, restore 150 W on both cards,
  re-run process/container/VRAM gates, and remember GPU 1 render/hwmon
  minors can renumber.
- Do not `b70-gpu1 off` during TP2; both cards are in use.
- 32 GB host RAM plus a SYCL kernel compile plus a 27B load is a memory
  risk. Discover `free -h` before `build.sh`.
