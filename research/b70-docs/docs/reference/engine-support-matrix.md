---
title: "B70 Engine Support Matrix"
status: living
type: reference
date: 2026-08-08
---

---
id: b70-ref-engine-support-matrix
title: B70 Engine Support Matrix
document_type: reference
status: canonical
version: "1.0"
effective_date: 2026-08-19
last_reviewed: 2026-08-24
authority: docs/methods/benchmark-protocol.md
scope: Evidence-based support status for inference engines and benchmark tooling on Intel Arc Pro B70
---

# B70 Engine Support Matrix

## Purpose

This matrix records support as an evidence status, not a permanent hardware promise. “Supported” means a bounded path has completed the stated gate on the B70. It does not imply every architecture, quantization, modality, context, scheduler, or concurrency mode works.

## Status vocabulary

| Status | Meaning |
|---|---|
| `production` | Current operational path with documented launch, safety, and bounded correctness/performance evidence |
| `validated_research` | Repeatable research path with raw evidence but not the default operational path |
| `experimental` | A bounded smoke or patched path works; quality, breadth, reproducibility, or upstream status remains incomplete |
| `blocked` | A required architecture, kernel, build, memory, or artifact gate is known to fail or be absent |
| `historical` | Previously observed path retained for history; prerequisites or current relevance have changed |
| `not_measured` | No adequate B70 evidence |

## Engine matrix

| Engine family | Current status | Demonstrated scope | Primary benchmark path | Important limits |
|---|---|---|---|---|
| llama.cpp SYCL | `production` | GGUF text serving; quantized KV with flash attention; synthetic pp/tg; HTTP paths for supported speculation; selected multimodal paths when an mmproj and decoder are available | `llama-bench` for synthetic prefill/decode; `llama-server` HTTP for MTP/spec paths | Build/model/quant specific; `llama-bench` cannot test speculative decoding; context claims require real boundary completions |
| vLLM XPU | `validated_research` | OpenAI-compatible serving for exact supported architectures/checkpoints; GPTQ-INT4 and other exact quant paths that have loaded; C1/Cn latency; prefix-cache and spec counters; patched MTP/DFlash paths in named campaigns | `vllm bench serve` for TTFT/TPOT/ITL/E2EL; controlled HTTP for exact workloads; `/metrics` for cache/spec counters | Architecture and quant packing are exact gates; local overlays may be non-reproducible; cache, graph, mixed-token, and long-context behavior require separate validation |
| EXL3-XPU (vLLM fork, ExLlamaV3 trellis) | `validated_research` | Qwen3.8-27B trellis 4bpw + native MTP3 + fp8 KV; OpenAI-compatible serving on isolated dockerd; full native 262,144 ctx on one card; prefix caching; **primary route since 2026-09-27 (ADR-0010)** | `CAM-2026-AUTOROUND-VS-EXL3` harness (`results/2026/09/autoround-vs-exl3-20260926/`) | fp8 is the only KV dtype (bf16 rejected); MTP chunk-rate undercounts tokens; needs isolated dockerd + `serve.py models/<id>` config | 
| OpenVINO Model Server / OpenVINO | `historical` | Bounded engine sweep observations on selected models | Engine-specific report | Not the current canonical serving or benchmark path; no broad current support claim |
| OpenVINO GenAI (Python) | `validated_research` | Qwen3.8-27B INT4-GDN8 via `VLMPipeline`; grafted MTP5 speculative decode (79 tok/s @512 client post-first, single-card champion, n=3, 2026-09-15 sweep); composite serving curve to 128K full-protocol | Campaign harness (`results/2026/09/qw38-ov-mtp-20260914/`) | Hybrid-GDN only; draft off ≥98K (u8 -14 crash / 3.1× MTP regression); ≥196K fails CL_OUT_OF_RESOURCES (openvino.genai#4483); 40-token capacity probes reach 256K |
| Cascadia (separate engine — Rust server + IR surgery) | `experimental` | `qwen35` surgery runtime on Qwen3.8-27B: 4.95 tok/s P512/G128 (stock 0.2.3 + OV 2026.2, INT4-GDN8, 2026-09-11 bakeoff); 23.78 tok/s qwen35 / 30.4 tok/s ov-genai class on the lab's patched 0.2.4 + OV 2026.5.0.0beta1 build with the official int4-ov IR (2026-09-22 fleet session); PP2 across both B70s retains ~100% of 1-stage (RL-051, −0.26%) | Bakeoff harness (`research/2026/ov-cascadia-qwen38-bakeoff-20260911.md`) + `results/2026/09/cascadia-pp-qwen38-20260923/` + fleet lab `~/projects/cascadia/` | Never merge numbers with OpenVINO GenAI rows; binary × OV × artifact define the class — do not mix the two rows; capacity/multi-node pipeline-parallel is its actual use case; `ov-genai` VLM SIGSEGV fixed by OV 2026.5 bump (upstream #160/#161, open); probe listener binds 0.0.0.0:9101 |
| SGLang XPU | `blocked` | No usable B70 path in the recorded engine sweep | None | Recorded absence of a usable XPU image/build path; reassess only with new upstream evidence and a successful B70 smoke |

Sources: [benchmark protocol](../methods/benchmark-protocol.md), [standards audit](../../archive/legacy-docs-20260915/benchmark-standards-audit-2026-08-08.md), [cross-engine comparison](../../archive/legacy-docs-20260915/cross-engine-comparison-2026-08-06.md), and [engine sweep](../../archive/legacy-docs-20260915/b70-engine-sweep-results-2026-07-05.md). Dated comparison numbers remain historical evidence and are not repeated here.

## Capability matrix

| Capability | llama.cpp SYCL | vLLM XPU | Notes |
|---|---|---|---|
| GGUF model weights | Demonstrated | Not the native checkpoint path | Use exact engine format |
| Hugging Face GPTQ-INT4 | Not the native checkpoint path | Demonstrated for exact architectures/checkpoints | Loader and XPU kernel dispatch must both pass |
| Quantized KV | Demonstrated with engine-supported K/V types and flash attention | Engine/config dependent | Record exact KV dtype and capacity |
| Synthetic engine pp/tg | Demonstrated through `llama-bench` | Not applicable | `llama-bench` launches its own GGUF model |
| Standard serving latency | HTTP available; report engine timing and client UX separately | Demonstrated through `vllm bench serve` | TTFT/TPOT/ITL/E2EL are client-observed |
| Cn continuous batching | Available but must be measured as aggregate Cn | Demonstrated for named server campaigns | Never compare Cn aggregate to C1 engine rate |
| MTP/speculative decoding | Demonstrated on exact supported GGUF/model paths through HTTP | Demonstrated on named patched/native checkpoint paths | Screen draft depth at matched anchors; acceptance is diagnostic |
| Prefix caching | Engine/path dependent; counter visibility may be limited | Demonstrated with authoritative V1 counters | Cold and warm serving are separate campaigns |
| Dual-card independent stacks (`ZE_AFFINITY_MASK`) | Demonstrated as the host default | Demonstrated as the host default | One engine per card; GPU 1 may be powered off |
| Dual-card tensor parallel (TP=2) | Topology-viable split; not a launcher product | External existence proof only; local `not_measured` | Requires host-staged collectives and piecewise graph capture on this board; see [ADR-0008](../../decisions/adr-0008-dual-b70-independent-default.md) |
| Multimodal serving | Demonstrated for selected mmproj/model combinations | Exact architecture and vision path required | Text support does not imply vision support |
| Long-context capacity | Demonstrated on named model/config completions | Demonstrated on named model/config completions | Configured max length is not evidence of completion |

## Support decision rules

An engine or capability may move to a stronger status only when:

1. a reproducible stack identity exists;
2. the exact architecture and artifact load through the intended kernel path;
3. the declared correctness scope passes;
4. safety and observability requirements pass;
5. a claim-sized benchmark or capacity campaign is retained; and
6. operational prerequisites are current and available.

A successful patch, load, or coherent response normally supports `experimental`, not `production`.

## Benchmark tool selection

- Use `llama-bench` only for llama.cpp synthetic pp/tg.
- Use `llama-server` controlled HTTP when a llama.cpp speculative path cannot be measured by `llama-bench`.
- Use `vllm bench serve` as the primary vLLM serving latency tool.
- Use vLLM `/metrics` latency histograms only for corroboration; use its discrete cache/spec counters as authoritative cumulative counters when diffed across the measured window.
- Use a host-side wrapper for B70 power, temperature, VRAM guards, synchronized counters, and exact real-workload replay.

## Reassessment triggers

Review this matrix after an upstream engine release, new XPU image, kernel registration, architecture merge, quant loader change, reproducible public image, or retained B70 campaign that changes a gate. A repository name, open pull request, or successful conversion alone does not change support status.
