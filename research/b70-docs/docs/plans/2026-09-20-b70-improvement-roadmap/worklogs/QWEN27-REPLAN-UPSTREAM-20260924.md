---
title: "QWEN27 REPLAN — upstream primary-source research (Level Zero IPC/P2P + post-pin kernels)"
type: worklog
date: 2026-09-24
status: final
---

# QWEN27 REPLAN — upstream primary-source research (2026-09-24)

Owner: b70-upstream-researcher. Intent: research. Read-only; no GPU probes, no
builds, no config/git changes, no upstream posts. Scope: (1) dual-B70 TP2 via
Level Zero IPC / DMA ring buffers across X570-F CPU x8/x8 root ports — with
the operator's qualification that ">1 TB/s" means **pooled local GDDR read
bandwidth** (2 × 608 GB/s aggregate local weight streaming under weight-split
TP), not cross-card PCIe bandwidth; (2) post-pin upstream kernel fixes
relevant to the Qwen3.8-27B GPTQ INT4 dense hybrid-GDN vLLM C1 champion.
Revision 2 (same day): operator qualifications (pooled-local reading, IPC
same-device precision, E1 screenshot class, #935 applicability to THIS host,
"currently blocked unless capability evidence" verdict), coordinator fixes
(frontmatter, pinned links, experiment independence, external-evidence
wording, no auto oneCCL changes).

Local working set (12-file budget, 10 used): `registry/research-ledger.json`
(row scan), `models/qwen/qwen3.8-27b/README.md`,
`models/qwen/qwen3.8-27b/serving.md`,
`research/2026/x570f-dual-b70-pcie-topology-20260830.md`,
`decisions/adr-0008-dual-b70-independent-default.md`,
`research/2026/qwen38-fp8-tp2-speed-attack-and-ceilings-20260829.md`,
`research/2026/humble-b70-llm-dual-tp2-20260824.md`,
`docs/plans/2026-09-20-b70-improvement-roadmap/OUTCOMES.md`,
`research/2026/exl3-analysis-and-next-inference-levers-20260924.md`, this file.

---

## Q1 — Dual-B70 TP2 via Level Zero IPC / DMA ring buffers

### 1.1 Level Zero semantics (primary: oneapi-src/level-zero-spec, master `143853c1`, fetched 2026-09-24; floating master, no release pin)

- [memory.yml @ 143853c1 (level-zero-spec)](https://github.com/oneapi-src/level-zero-spec/blob/143853c129b66d337d3823db7f1ec1abd61c6940/scripts/core/memory.yml) — `zeMemOpenIpcHandle`:
  > "The 'hDevice' parameter identifies the device into which the allocation
  > is imported. **If this device differs from the one where the memory was
  > originally allocated, it must have P2P access to the exporting device.
  > P2P accessibility can be verified using zeDeviceGetP2PProperties or
  > zeDeviceCanAccessPeer.**"

  **Precise reading (operator qualification):** IPC itself is an
  **inter-process** mechanism and works on a **single** device — exporting a
  device/host allocation (`zeMemGetIpcHandle`: device 1.0, host 1.5) to
  another process and opening it there needs no second GPU. What carries the
  P2P precondition is only the **cross-device** case: importing on a device
  other than the exporter. IPC never creates peer routing; it inherits
  whatever P2P access exists. This matches humble-b70-llm's
  `ZE_RESULT_ERROR_INVALID_ARGUMENT` (their board had no usable P2P to the
  exporter).
- [device.yml @ 143853c1 (level-zero-spec)](https://github.com/oneapi-src/level-zero-spec/blob/143853c129b66d337d3823db7f1ec1abd61c6940/scripts/core/device.yml) — `zeDeviceCanAccessPeer`: true when same device / same sub-device / same parent, or "connected by a direct or indirect scale-up fabric **or over PCIe (same root complex or shared PCIe switch)**". Query of routing, not of bandwidth; the spec nowhere promises PCIe P2P approaches local-memory bandwidth. `zeDeviceGetP2PProperties`: core struct has only `flags` (`ACCESS`, `ATOMICS`); numeric bandwidth/latency exist only in the 1.4 experimental extension (`ze_device_p2p_bandwidth_exp_properties_t`, logical/physical bandwidth + units) — driver-dependent diagnostics, not a pooled-bandwidth promise.

### 1.2 Driver/kernel reality, and what applies to THIS host

Primary thread: [intel/compute-runtime#935](https://github.com/intel/compute-runtime/issues/935)
("[BMG / Arc Pro B70] Is cross-root-port P2P expected? …", GSD-12874,
created 2026-06-06, closed 2026-08-20; reproducer
[TSUMUGI-XE/b70-dual-tp2](https://github.com/TSUMUGI-XE/b70-dual-tp2), ~85-line L0 probe).

- **What #935 actually showed (their hardware, do not generalize):** on a
  consumer **Intel** host (Rocket Lake-S `8086:4c43`), dual B70 on separate
  root ports: direct `zeCommandListAppendMemoryCopy` dev0→dev1 = `0x70000003`;
  kernel refuses at `pci_p2pdma` ("do not share an upstream bridge or
  whitelisted host bridge"); forcing the host bridge into
  `pci_p2pdma_whitelist` yields SUCCESS with **silent corruption** (all-1s,
  PCIe Unsupported-Request) — that Intel root complex does not route peer
  TLPs. Host-staged copies PASS. Their switch experiment (PEX880xx, one
  shared upstream): P2P clean **only with ACS redirect disabled** on the
  switch ports (ACS on: 10.89 GB/s but corrupt). Intel acknowledged the
  `OUT_OF_DEVICE_MEMORY` code is misleading and said they are investigating
  reporting unsupported P2P more accurately.
- **Applicability check to b70-host (done this session, static +
  local):** the deciding code is **CPU-vendor-generic, in PCI core**, not in
  xe and not Intel-specific. Upstream
  [drivers/pci/p2pdma.c](https://github.com/torvalds/linux/blob/f2c53ea949c5048f96b3dbb5a5ee7131ce4ff2de/drivers/pci/p2pdma.c)
  (master `f2c53ea9`, fetched 2026-09-24):
  `calc_map_type_and_dist()` sets `NOT_SUPPORTED` only when
  `!cpu_supports_p2pdma() && !host_bridge_whitelist(...)`, and
  `cpu_supports_p2pdma()` is documented in-source as "Any AMD CPU whose
  family ID is **Zen or newer** supports p2pdma" (`x86_vendor == AMD &&
  x86 >= 0x17`). Verified locally: `AMD Ryzen 7 5700X3D`, **CPU family 25
  (0x19)** ≥ 0x17; kernel `7.0.0-28-generic` with `CONFIG_PCI_P2PDMA=y`;
  host bridge `1022:1480` (AMD Starship/Matisse Root Complex); both GPUs on
  CPU-attached GPP root ports `00:03.1`/`00:03.2` of that one root complex;
  no shared switch → path resolves to `PCI_P2PDMA_MAP_THRU_HOST_BRIDGE`,
  **and the whitelist refusal branch is bypassed on this CPU**. I.e. the
  exact kernel gate that blocked #935's Intel host **does not block ours**;
  two devices with no common upstream bridge land in the same trusted
  THRU_HOST_BRIDGE class. In-thread corroboration for the AMD class:
  third-party AM5 (Zen4) dual-B70 cross-card P2P verified checksum-PASS
  (mjsabby 2026-06-28; TSUMUGI-XE relay) — same GPUs, same xe-KMD
  requirement, different AMD socket than ours; **our platform remains
  unmeasured**. Residual unknowns that only a probe answers: whether the
  Zen2/Starship root complex actually routes peer TLPs (the AMD blanket
  trust exists because AMD RCs are expected to), whether NEO's
  `Device::canAccessPeer → queryPeerAccess`
  ([shared/source/device/device.cpp](https://github.com/intel/compute-runtime/blob/master/shared/source/device/device.cpp),
  floating master) returns true for two discrete BMG here, and whether any
  ACS redirect sits on our root ports (locally recorded as none, ADR-0008
  era; the kernel prints a `pci=disable_acs_redir=` hint if one is found).
  Noteworthy static detail: no `/sys/bus/pci/devices/0000:0b:00.0/p2pdma`
  node exists locally — as on #935's host; that node only appears once a
  driver publishes p2pdma pages, so its absence is not dispositive.

### 1.3 vLLM / oneCCL collectives state (primary: vLLM main, fetched 2026-09-24)

- [`vllm/distributed/device_communicators/xpu_communicator.py`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/xpu_communicator.py):
  plain `torch.distributed` collectives on `device_group` (oneCCL);
  `self.ca_comm: None` — **no custom allreduce, no IPC/SHM path, no
  host-staged option**; `gather` works around XPU+ray via `all_gather`;
  optional `VLLM_BATCH_INVARIANT` rank-ordered gather. The humble-b70
  host-staged patch set is **still not upstream**; ADR-0008's
  reconsideration trigger ("upstream merge of host-staged collectives or
  working peer IPC") is **not met**.
- XPU graph capture is CUDA-API aliasing onto `torch.xpu.XPUGraph`
  ([`vllm/v1/worker/xpu_model_runner.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/xpu_model_runner.py));
  piecewise capture around comms is a local-patch discipline, not upstream.
- Stock TP2-with-capture remains broken upstream on this hardware class:
  [vllm#41663](https://github.com/vllm-project/vllm/issues/41663) (dual-B70
  TP2: GP fault + xe BCS engine reset; closed without fix; July tail
  reports even `--enforce-eager` corrupting under sustained ragged load on
  that reporter's host);
  [vllm#58388](https://github.com/vllm-project/vllm/issues/58388) OPEN
  (eager all-reduce after XPU graph capture leaves the stream "Recording" →
  engine dies on first large prefill);
  [vllm#48946](https://github.com/vllm-project/vllm/issues/48946) OPEN
  (TP=2 piecewise XPU-graph corruption even on Data Center GPU Max 1100).
- oneCCL practice on this class: `CCL_TOPO_P2P_ACCESS=0` + ofi transport
  (humble env block); Puget saw RxErr/engine resets with
  `CCL_TOPO_P2P_ACCESS=1` (riser-attributed, never re-validated).

### 1.4 Verdict — capability split (operator-qualified)

- **Pooled local GDDR read bandwidth >1 TB/s: a valid capacity statement.**
  Under weight-split TP each card streams its half of the weights from its
  own local 608 GB/s GDDR6 — aggregate local read capacity 2 × 608 =
  1,216 GB/s. This needs no P2P and no cross-card bandwidth. What is
  **unsupported** is *delivered C1 scaling / transport headroom on this
  host*: per-step activation collectives must cross between cards, stock
  vLLM XPU TP2 is upstream-broken with capture (#41663/#58388/#48946), and
  the only externally matched TP2 evidence on this GPU class is
  host-staged-collective decode scaling of **~1.39–1.41× with worse TTFT and
  prefill** (humble cold 94.58/66.84, warm 136.4/97.8 — their host, their
  checkpoint; an external evidence class, **not** an expected local number),
  and 1.99× at TP4 on an 8B via LLM Scaler
  ([Puget Systems, Jun 2026, corrected Sep 2026](https://www.pugetsystems.com/blog/intel-arc-pro-b70-multi-gpu-ai-inference-performance/)).
  No local TP2 measurement exists. Cross-card transport on this board is
  per-link Gen4 x8 ≈ ≤16 GB/s theoretical regardless (measured floors
  8.85/7.89 GB/s here; ~14 GB/s on a cleaner host; ~11 GB/s via PEX switch),
  so ring buffers can only shave host-staged latency, never approach local
  GDDR rates.
- **Zero-copy (true peer/IPC) transport: CURRENTLY BLOCKED, not impossible
  forever.** The #935 failures are consumer-Intel-RC results and do not
  transfer to this host's decisive layer: on 5700X3D the kernel-side p2pdma
  gate is bypassed (§1.2). What is missing is *capability evidence* at the
  NEO/L0 layer on THIS platform — `zeDeviceCanAccessPeer(0,1)`,
  `zeDeviceGetP2PProperties` flags, and one verified cross-card copy have
  never been measured locally. Status changes only on changed capability
  evidence: a local probe PASS, an upstream NEO/vLLM capability change, or
  a topology change (shared switch with ACS off — externally verified — or
  a whitelisted/server host). Note humble's `INVALID_ARGUMENT` came from a
  different board/slot class (chipset-attached x4 second slot) and does not
  predict this topology.

### 1.5 Deciders and sequencing (independent branches; no auto changes)

1. **Static (done, this session):** spec, kernel source, local kernel/CPU
   checks pinned above. No further static work exists.
2. **GPU microprobe (orchestrator-assigned window, ~1–2 min, both cards, no
   model load):** TSUMUGI 85-line L0 probe (or equivalent) → print
   `zeDeviceCanAccessPeer` both directions + P2P flags + one 256 MiB
   dev0→dev1 copy **with checksum** + host-staged control.
   - FAIL (canAccessPeer=0 / error / corrupt): zero-copy stays
     capability-blocked on this host as-built; any TP2 work continues only
     via the gated host-staged campaign path (ADR-0008). External ~1.4×
     decode class remains the only scaling reference, explicitly not a
     local expectation.
   - PASS: capability exists — but **no automatic oneCCL or server
     changes follow**. oneCCL `CCL_TOPO_P2P_ACCESS=1`, peer-atomics
     behavior, and capture interaction are a **separate correctness gate**
     (collective correctness + determinism + graph-capture rules of §1.3),
     to be proposed as its own bounded, canaried experiment if pursued.
     The probe's own measured number (≈8–14 GB/s class) is the only
     transport figure anyone may quote; pooled-local capacity (§1.4) stays
     a capacity statement, never a delivered-throughput claim.

---

## Q2 — Post-pin upstream kernel state vs the dense GPTQ hybrid-GDN champion

Pinned stack: vLLM `0.27.2rc1.dev77+gac7509e2b` (**ac7509e2b = 2026-08-14**,
[commit](https://github.com/vllm-project/vllm/commit/ac7509e2b)) +
vllm-xpu-kernels `0.1.12.3`, image digest `f01e24f6…`. Upstream changes
after 2026-08-14 are post-pin.

### 2.1 Concrete post-pin changes that differ from killed hypotheses

| Change | Identity | Relevance to our dense GPTQ route |
|---|---|---|
| [vLLM #56096](https://github.com/vllm-project/vllm/pull/56096) "[XPU] Route to fused_qk_rmsnorm_rope_gate triton kernel" | MERGED 2026-09-16, head [`b9914f58`](https://github.com/vllm-project/vllm/pull/56096/commits/b9914f580efa827c3cde32179819b1411e4b95bc); one-line diff in `qwen3_next.py`: `use_fused_qk_norm_rope_gate` true on `is_xpu()` when `attn_output_gate` + neox rotary + dtype + (text_only or mrope) | Merged form of [#53989](https://github.com/vllm-project/vllm/pull/53989) (still OPEN, needs-rebase — superseded; do not re-screen #53989). Author-measured on Qwen3.5-9B/B70: QK-norm+RoPE+gate region device time −55 % decode bs=8, −23 % prefill. Post-pin. Our earlier local screen was protocol-invalid and **dispatch was never traced** (RL-054). |
| vllm-xpu-kernels [v0.1.14/0.1.14.1](https://github.com/vllm-project/vllm-xpu-kernels/releases/tag/v0.1.14) (2026-08-31) | Release notes | "**Fixed mixed speculative and non-speculative GDN workloads**, corrected token-indexed convolution-state layout for causal-conv1d speculative decode" — GDN spec-decode *correctness* class, adjacent to #54785's surviving suspects; RMSNorm/fused-add RMSNorm now round once in FP32 (numerics change); fused Gemma RMSNorm "for Qwen3.5". |
| vllm-xpu-kernels [v0.1.15](https://github.com/vllm-project/vllm-xpu-kernels/releases/tag/v0.1.15) = PyPI **0.1.15.4** ([wheel](https://pypi.org/project/vllm-xpu-kernels/), 2026-09-22; **latest** — nothing newer exists today) | Release notes | Optimized top-k/top-p sampler + per-row RNG + `compact_sampling_mask`; fused LayerNorm family; widened `act_and_mul` vectorization; Xe2 BlockFP8 grouped-GEMM accumulator scaling (FP8 path — **not** our GPTQ W4A16 route); MoE routing fixes (not dense). |
| [vllm#54785](https://github.com/vllm-project/vllm/issues/54785) (XPU graph capture + MTP k=4 corruption) | OPEN, last activity 2026-09-01 | Narrowed in-thread to k+1=5-row spec step + short-context dependence; surviving suspects ESIMD `gated_delta_rule_spec_kernel` / 5-row verify mapping; upstream mitigation remains the k=3 clamp only. Local stance unchanged (canary PASS at k=4; clamp costs +5.45 %, L12). |
| [vllm#56431](https://github.com/vllm-project/vllm/pull/56431) DFlash context-key normalization | MERGED 2026-09-17, `5905059a` | Already known (RL-027 lineage). DFlash2 failures not reopened unchanged per task scope. |
| [llama.cpp #29245](https://github.com/ggml-org/llama.cpp/pull/29245) "sycl: add grouped MoE XMX GEMM" | OPEN, head `950f1c4a`; 5 files (`ggml-sycl/fused-gemm.*`, `common.hpp`, `ggml-sycl.cpp`, `SYCL.md`) | **No direct dispatch on a dense GPTQ model — confirmed at source level, three independent ways:** (a) routed `MUL_MAT_ID` path — dense models call `MUL_MAT`; (b) dequantizes GGUF `IQ4_NL`/`IQ3_S`, not GPTQ; (c) llama.cpp SYCL engine, not vLLM. Its +34.37 % pp / +0.95 % tg (Qwen3.8-Flash-Next, MoE) **does not transpose** to the dense vLLM champion. (K216 archival per 2026-09-23 scope change.) |
| FastMTP — [arXiv:2509.18362](https://arxiv.org/abs/2509.18362) (2025-09-16) | Paper verified | Fine-tunes a single MTP head with **position-shared weights on self-distilled data**; aggregate 2.03× vs NTP claimed; abstract **silent** on preserving early-position calibration of an already-strong factory head. L15's local refutation (Δα −0.04, −2.5 % tps) stands; no basis to reopen D11-class light retraining. |

### 2.2 Deciders (independent branches — neither gates the other)

1. **#56096 dispatch trace — static, zero-GPU:** on upstream ≥ `b9914f5` or
   the rebased `~/vllm-src` tree, evaluate the four
   `use_fused_qk_norm_rope_gate` conditions for our exact artifact (GPTQ
   W4A16, bf16 activations, text-only): expect `attn_output_gate=True` +
   neox for the qwen3-next/GDN class; the open items are `supports_dtype`
   and the Triton import path under our pinned torch. If all four hold, a
   bounded matched p512/g128 n=3 screen (acceptance counters + spec-canary)
   is justified **independent of Q1**. Prior: region is 2.42→1.09 ms at
   bs=8 on a 9B; at C1 M=1–5 on 27B decode effect likely small, prefill
   region (−23 %) the more plausible TTFT-side effect.
2. **Q1 GPU microprobe (§1.5)** — independent of the QK branch; decides the
   zero-copy capability question for less than two minutes of both cards.
3. **Do NOT re-run the kernels wheel bump:** 0.1.15.4 is still latest; the
   sampler regression verdict is measured (settled 2026-09-23). If the
   v0.1.14 GDN spec-decode correctness fixes ever matter, that is a full
   determinism+acceptance canary campaign on a pinned overlay (numerics
   changed), not a quick screen; nothing in the champion's measured
   behavior currently demands it (W12 canary PASS, greedy hashes stable).
4. **Stay on:** stock MTP weights, k=4, greedy drafting, fp8 KV,
   presence_penalty 1.5 (re-validated by L12–L15 this week).

---

## Side-task — @0xSero EXL3 screenshot provenance

`research/2026/exl3-analysis-and-next-inference-levers-20260924.md` cites an
`@0xSero` post + screenshot with **no primary URL**. Two targeted web
searches (2026-09-24) found **no primary artifact** — X posts are not
indexed by the engines available here. Classification for the shared note
(operator-corrected): **E1 — self-reported external claim lacking raw data
and provenance** (E0 stays reserved for model estimates). No benchmark
comparability. The external "555 GB/s lm_head microbench" is unverified; the
internal "538.7 GB/s / 96.2 % of practical ceiling" row is *derived
arithmetic* (21.44 GB step bytes ÷ 39.8 ms step) layered on that unverified
608/560 GB/s framing — not a local bandwidth measurement, and must not carry
`truth_verdict: VALIDATED`. Same treatment for Frontier 4's "zero-copy"
promise (refuted by the spec text, §1.1) and its delivered-scaling wording
(capacity vs delivered split, §1.4). External anchors that DO exist for the
surrounding physics: [Puget Systems B70 multi-GPU](https://www.pugetsystems.com/blog/intel-arc-pro-b70-multi-gpu-ai-inference-performance/)
(608 GB/s per card; the 1,074 GB/s sanity check; TP4 1.99× on 8B) and
compute-runtime #935 (§1.2). No further social scraping attempted (per
coordinator instruction).

---

## Source limitations

- #935's failures are consumer-Intel-RC measurements; the AMD datapoint
  (AM5/Zen4) is third-party and reported in-thread, not ours; our
  X570-F/Zen3 result is unknown until the §1.5 probe runs. The kernel-source
  verification (§1.2) pins the *gate logic* and our CPU/config facts, not
  the runtime outcome.
- Puget tested TP1/TP4 only (no TP2), via LLM Scaler/oneCCL; direct P2P not
  re-tested after their riser fix. All external scaling figures are their
  hosts' numbers — never quoted as local expectations.
- #56096 numbers are the PR author's B70 measurements on a 9B — external
  until measured here; the one-line diff makes the dispatch trace decisive.
- Level Zero spec and p2pdma.c links are floating-master pins (`143853c1`,
  `f2c53ea9`, fetched 2026-09-24); re-verify on any later fetch. oneCCL
  graph-capture restriction is evidenced by upstream vLLM code state + open
  bugs + humble patch comments; no oneCCL doc page was fetched.
- Nothing was posted, submitted, or commented upstream. No local files
  changed except this worklog.

## Next deciding experiments (for the orchestrator; independent)

1. L0 P2P microprobe (both cards, ~1–2 min) — decides the zero-copy
   capability branch. No automatic follow-on: any oneCCL/capture change is
   a separate correctness-gated proposal.
2. #56096 static dispatch trace (CPU-only) — gates the fused QK-RoPE screen
   branch on its own.
3. Each branch, if passed, proceeds to its own bounded matched screen per
   protocol §1.9; neither depends on the other.
