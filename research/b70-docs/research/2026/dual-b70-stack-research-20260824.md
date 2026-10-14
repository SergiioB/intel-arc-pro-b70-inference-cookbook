---
title: "Dual-B70 inference stack research — llama.cpp / vLLM / SGLang / Cascadia / OpenVINO (2026-08-24)"
type: research
date: 2026-08-24
status: final
---

# Dual-B70 inference stack research — llama.cpp / vLLM / SGLang / Cascadia / OpenVINO (2026-08-24)

Status: `active`. All benchmark numbers below are **PROVISIONAL — NOT FOR PUBLIC HEADLINE**
(self-reported, E2 tier) until independently reproduced. Campaign evidence roots:
`results/2026/08/qw38-dualcard-20260824T123000Z/` (llama.cpp),
`results/2026/08/qw38-fp8-20260824T130000Z/` (vLLM FP8), plus this file's linked logs.

## 1. Question

Can two Arc Pro B70s on an ASUS ROG STRIX B450-F (GPU0 Gen3 x8 ≈ 7.1 GB/s, GPU1 Gen3
x4 ≈ 3.6 GB/s, no usable P2P) form a successful **dual-GPU inference stack**, which
runtime achieves it, is the x4 link a blocker, and what needs converting?

## 2. Model checkouts used (exact identities)

| Artifact | Revision / hash | Role |
|---|---|---|
| `unsloth/Qwen3.8-27B-GGUF` Q4_K_M | rev `430473d9`, SHA256 `7b2aec3b…89f1b` verified post-download | Phase B llama.cpp |
| `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` | local `/mnt/<models-disk>/...nogidx` (registry ART-QWEN38-GPTQ-G128-MTP-BF16) | champion reference (single card) |
| `Qwen/Qwen3.8-27B-FP8` @ `017b9c7a…` | 66 shards + outside.safetensors = 30.87 GB, CRC manifest PASS (3 text files mismatch upstream's own stale CRC — non-blocking) | Phase C vLLM |
| `unsloth/Meta-Llama-3.1-8B-Instruct` | downloaded by `cascadia shard` | Cascadia pipeline demo |

**Conversion answer: no conversion required anywhere.** GGUF serves llama.cpp directly;
the official FP8 loads natively in vLLM XPU (`XPUFp8BlockScaledMMKernel for Fp8LinearMethod`
selected, 28.51 GiB resident); Cascadia's own `shard` tool exports INT4 OpenVINO IR from HF
safetensors itself.

## 3. llama.cpp dual-card results (Qwen3.8-27B Q4_K_M, build `1692f9e50`, KV q8_0/q4_1, fa on, 230 W declared)

### Screens n=3 then fresh finalist n=5

| Cell | Control GPU0 | Layer-split ×2 | Δ |
|---|---|---|---|
| tg128 decode (n=5) | 25.78 ± 0.01 t/s | **26.03 ± 0.01 t/s** | **+0.97 %** |
| pp512 prefill (n=5, layer) vs (n=3, control) | 863.98 ± 1.38 | 840.50 ± 6.28 | −2.7 % |
| pp8192 prefill | 876.76 ± 2.90 (n=3) | **888.53 ± 3.60** (n=5) | +1.3 % |

HTTP serving cells (llama-server, cache off, entropy-prefixed prompts):

| Cell | Control median | Layer median |
|---|---|---|
| real workload p380/g256 engine t/s | 25.31 (n=5) | **25.58** (+1.1 %) |
| cold input ~2130 tok | 639.1 t/s | **642.1 t/s** (+0.5 %) |
| cold input ~8260 tok | **759.6 t/s** | 747.0 t/s (−1.7 %) |

Power during g256 decode: control 224 W on GPU0 + 46 W idle GPU1; layer-split
134 W + 167 W (≈ same total, spread across both cards).

Findings:

1. **Layer-split wins or ties every decode and short-prefill cell**; loses only long cold
   prefill (−1.7 %). The second card adds effective weight-read bandwidth even over x4,
   because boundary traffic is one activation per token (few KB–MB).
2. **Row split segfaults (SIGSEGV, exit 139)** on this build/topology — bounded diagnostic
   closed as negative; do not pursue.
3. Greedy deterministic replay **diverges occasionally by one token** on SYCL Xe2
   (reduction-ordering noise). Coherence unaffected. Do not promise bit-exact replay for
   llama.cpp SYCL claims.
4. Harness notes: this llama-bench has no `-c` flag (context derives from p+n);
   `-ts` uses `/` separators; scripts must source oneAPI via `set +u … set -u`.

## 4. vLLM XPU results (image `vllm/vllm-openai-xpu@sha256:f01e24f6…`, vLLM 0.27.2rc1.dev77)

| Experiment | Outcome |
|---|---|
| Official FP8 single GPU0, 16K ctx | **Loads natively** (28.51 GiB, no dequant blow-up) but KV init left 0.09 GiB vs 1.32 GiB needed → capacity-blocked for C1 |
| FP8 PP=2 (`--tp 1 --pp 2`) | Accepted (`world_size=2`, MultiprocExecutor); **blocked_host_ram**: two ranks staging ≈ 29 GiB host RAM on a 30 GiB box. Watchdog killed both attempts at 1405/310 MiB MemAvailable — zero kernel OOM events, host unharmed |
| Container launch requirements | `--ipc=host` (shm ring), `-v /dev/dri:/dev/dri:ro` (oneCCL fd manager), `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`; xccl warmup allreduce runs even at world_size=1 |

Conclusion: vLLM dual-rank needs ≥ 64 GB host RAM as-is; not a GPU or topology failure.
The single-card GPTQ-Int4+MTP champion (81–112 t/s class) remains the vLLM production route.

## 5. SGLang (research verdict; empirical lane deferred)

- Docs declare optimization for **Intel Arc Pro B-Series**; dedicated in-tree XPU backend
  (intel_xpu attention, GDN kernels, graph runner, nightly Intel CI).
- **Dual-GPU is documented first-class**: `--tp 2  # using multi GPUs` +
  `--attention-backend intel_xpu`.
- Community state: PR #34574 (open) documents that XPU TP decode collectives go through
  plain oneCCL `dist.all_reduce` with **no small-message fast path**; decode-shape
  messages (~57 KB) are dominated by fixed collective cost. The PR adds torch
  symmetric-memory one-shot all-reduce (multicast unsupported on Xe). Unmerged.
- Qwen3.8 actively supported upstream (MXFP4/NVFP4/DFlash2 work streams).
- Expectation for this host: mechanically works, but per-layer host-bounced allreduce over
  x4 will tax decode — same physics that motivated layer-split. Re-test empirically after
  #34574 merges. PP on XPU is not documented in SGLang.

## 6. Cascadia (cascadia.to / labscommunity/cascadia)

Alpha Rust runtime, Apache-2.0, **explicitly supports Arc B-series Battlemage dGPUs**;
ships Linux bundles with embedded OpenVINO; `cascadia shard` exports INT4 per-stage OV IR
from HF models; pipeline stages run as `worker --rank N --total 2` processes relaying
activations over TCP; OpenAI-compatible API.

On this host: `cascadia doctor` reports **CPU, GPU.0, GPU.1 (both 0xe223 dGPUs)**.
Llama-3.1-8B two-stage INT4 sharded export + dual-worker pipeline test: see §6.1.

### 6.1 Pipeline test results

Setup: `cascadia shard --model Qwen/Qwen2.5-1.5B-Instruct --num-stages 2 --quantization int4`
(export deps: transformers 5.4 / torch 2.13+cpu / OpenVINO 2026.3 venv). Workers:
`--rank 1 --device GPU.1 --listen :9100` started first, then
`--rank 0 --device GPU.0 --next 127.0.0.1:9100 --api :8000`. VRAM after load:
GPU0 1365 MiB (embed + half layers), GPU1 649 MiB (half layers + head) — true per-stage
placement on separate B70s.

| Config | e2e tok/s (client wall, 80-token greedy completions, n=3–4) |
|---|---|
| 1-stage `ov-runtime` on GPU.0 (control) | **median 85.8** (69.9–86.8) |
| 2-stage pipeline GPU.0→TCP→GPU.1 | median 70.0 (55.9–77.4), ≈ −18 % |

Findings:

1. **Dual-B70 pipeline works end-to-end**: coherent OpenAI-compatible completions relayed
   GPU.0 → TCP localhost → GPU.1, first-try bring-up in under two minutes.
2. On a 1.5B model the pipeline tax (−18 %) dominates — as theory predicts; the runtime's
   value is capacity sharding, not small-model speed. A ≥27B INT4 pair (≈14 GiB/stage)
   is the intended operating point but its HF-source export does not fit this host's
   30 GiB RAM safely (Llama-3.1-8B export worked; 27B BF16 source load would not).
3. This test doubles as the **OpenVINO dual-B70 datapoint**: `ov-runtime`/`ov-genai`
   (OpenVINO 2026.3) compiled and executed transformer shards on both Battlemage cards.
4. Ops notes: worker order matters (last stage first); `ov-genai` refuses staged trees;
   `--layer-start/--layer-end` exists for asymmetric splits (maps naturally onto x8/x4).

## 7. OpenVINO standalone

Covered through Cascadia's `ov-genai`/`ov-runtime` engines (OpenVINO 2026.3 runtime inside
the bundle sees both cards). A separate direct OVMS GenAI serving datapoint exists
historically for Qwen3.6 MoE (`archive/legacy-dirs-20260915/benchmarks/ovms-vs-llamacpp-qwen35b-moe-2026-08-04.md`);
no additional standalone dual-GPU OV run was executed in this campaign beyond the
Cascadia engines.

## 8. The x4-link verdict

**The x4 link is NOT a blocker for the working dual stack — but it IS a hard ceiling for
tensor parallelism.**

- Working path (layer/pipeline split): inter-stage traffic is one hidden-state vector per
  token (~10 KB/token bf16 at hidden 5120) — negligible against 3.6 GB/s. Measured:
  decode +1 %, prefill −1.7 % to +1.3 % versus single-card. The x4 link costs us nothing
  observable at C1 decode.
- Blocked paths: any scheme inserting per-layer collectives (vLLM/SGLang TP=2). With no
  P2P, every allreduce bounces through host RAM twice (x4 out + x4 back ≈ 2×3.6 GB/s
  shared), and fixed collective latency dominates small decode messages. Row-split in
  llama.cpp crashes outright today.
- Secondary ceilings: vLLM dual-rank host-RAM staging (>30 GB needed); idle power of the
  always-on second card (44.7 W card domain, §7 of the topology doc).

## 9. Recommended production stack (this host)

1. **Two independent single-card stacks** for throughput isolation (champion GPTQ+MTP on
   one card, second model/engine on the other) — unchanged conclusion.
2. **Capacity route**: llama.cpp SYCL layer-split (`-sm layer -ts 1/1 -mg 0`,
   `ZE_AFFINITY_MASK=0,1`) — decode parity-to-plus, doubles VRAM for larger models/context.
   Asymmetric splits via `--layer-split`-style ratios only if VRAM pressure demands.
3. **FP8 route**: official Qwen3.8-27B-FP8 works natively once host RAM allows PP=2, or on
   future hardware with ≥36 GB single cards.
4. Keep GPU1 powered via `b70-gpu1` workflow; runtime PM must stay pinned `on`.

## 10. Kernel/runtime fixes catalogued this session

- xe driver: GPU1 whole-chain runtime-suspend kills PCIe link → pin `power/control=on`;
  automated in `b70-gpu1 on`.
- vLLM XPU container: oneCCL fd-manager needs `/dev/dri` bind-mount + `--ipc=host` +
  COMPOSITE hierarchy (xccl warmup allreduce otherwise dies with "could not open device
  directory").
- HF Xet content must be fetched with the `hf` client; aria2 ranged downloads through the
  xet-bridge CDN produced silently corrupt bytes twice (hash mismatch, same size).
- llama-bench (build 1692f9e50): no `-c` option; `-ts N/M`; needs oneAPI `libsvml.so`
  sourced at runtime.
- Prior-session fixes still in effect for the vLLM champion: GDN mixed-split v5 patch,
  MTP nightly + boundary patches (see qwen38 dossier).

## 11. Next steps

1. Empirical SGLang `--tp 2` Qwen3.8 cell after PR #34574 merges (or as a bounded screen
   now if time allows).
2. vLLM PP=2 on a ≥64 GB RAM host to unlock FP8 two-card serving.
3. Cascadia 27B INT4 sharding (export RAM profile must be profiled first — BF16 load of a
   55 GB model does not fit this host; requires streaming exporter or a smaller dtype).
