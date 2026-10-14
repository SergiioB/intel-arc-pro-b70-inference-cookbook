---
id: b70-ref-hardware-software-stack
title: "B70 Hardware and Software Stack Reference"
type: reference
status: living
date: 2026-08-27
updated: 2026-10-09
version: "1.0"
authority: AGENTS.md
scope: Stable B70 host identity and required software-stack provenance fields
---

# B70 Hardware and Software Stack Reference

## Purpose

This reference separates stable host identity from campaign-specific software provenance. It intentionally avoids duplicating current benchmark results and rapidly changing model inventory. For live operational state, consult `AGENTS.md` and verify the host directly before running or publishing.

## Stable host identity

| Component | Recorded host identity | Verification expectation |
|---|---|---|
| GPU | Two Intel Arc Pro B70, 32 GB class GDDR6; 32,656 MiB visible each in the recorded host snapshot | Re-read debugfs for each campaign and after GPU 1 power cycles |
| Dual-card topology | X570-F since 2026-08-27, CPU-attached x8/x8. RL-059 verified checksum-clean direct peer copies about 11–12 GB/s; peer atomics absent | [Topology note](../../research/2026/x570f-dual-b70-pcie-topology-20260830.md); [ADR-0008](../../decisions/adr-0008-dual-b70-independent-default.md). Revalidate after hardware/driver changes |
| CPU | AMD Ryzen 7 5700X3D, 8 cores / 16 threads; 8 is the established llama.cpp worker-thread choice, not total CPU threads | Record host CPU and chosen threads in each manifest |
| System memory | Historical campaigns disagree (30 GB and 60 GiB snapshots) | Read live MemTotal and available RAM; do not size a run from either snapshot |
| Operating system family | Ubuntu 26.04 on x86_64 | Record exact kernel and OS for each campaign |
| GPU driver interface | Linux `xe` driver; debugfs for VRAM and hwmon for energy/temperature | Verify device paths rather than copying historical nodes |
| Storage | Root plus three data disks mounted at `/data/nvme`, `/data/sata1`, `/data/sata2` (layout: `docs/operations/storage-layout.md`) | Use live `df -h` and `python3 tools/storage_inventory.py check`; historical free space is not authoritative |

Hardware marketing values and remembered telemetry are not substitutes for a campaign manifest. If a field matters to a claim, capture it during that campaign.

## Supported software families

Engine families are recorded separately in `registry/model-status.json` and
`engines/`: llama.cpp SYCL, vLLM XPU, EXL3-XPU, OpenVINO GenAI, Cascadia and
other researched engines. Presence in a dossier does not mean adoption.
`research.py next` selects a candidate; it does not replace a serving route.

## Campaign provenance requirements

Every campaign must record the observed stack rather than referring only to “the B70 stack.”

### Host and driver

- hostname or host ID;
- GPU, CPU, RAM, OS, and kernel;
- driver/runtime versions when observable;
- verified PCI identity for debugfs and hwmon;
- empty-card VRAM check; and
- storage mount and availability when artifact acquisition or conversion is involved.

### llama.cpp SYCL

- repository and exact commit/build number;
- binary identity and path;
- compiler, oneAPI, and relevant build options;
- complete command and relevant environment variables;
- model GGUF file, size, and hash when available;
- GPU/MoE offload, context, batch/microbatch, threads, flash attention, K/V cache types, mmap policy, and speculation settings; and
- timing source (`llama-bench` raw repetitions or `llama-server` response timings).

The current operator snapshot records a production llama.cpp SYCL b10255+ family, but each new result must retain its own exact build identity.

### vLLM XPU

- pullable image repository plus immutable digest, or exact source build;
- observed vLLM and `vllm-xpu-kernels` versions inside the runtime;
- ordered patch list with hashes and application method;
- model repository/path, exact revision and quantization metadata;
- context, GPU memory utilization, KV dtype, scheduler limits, graph/eager mode, prefix-cache policy, concurrency, and speculative configuration;
- startup log evidence for effective configuration and dispatch; and
- timing and counter sources.

A local-only image tag or overlay may support internal research but is not a public reproduction prerequisite.

## Environment and device selection

For llama.cpp SYCL, the repository's protocol currently records oneAPI activation and Level Zero device selection variables. Capture their effective values in the campaign manifest rather than copying them without verification. Do not disable the SYCL optimization path except for a declared debug experiment.

Dual-card default is one independent engine per GPU (`ZE_AFFINITY_MASK`). Direct
peer copy support does not authorize stock TP2 or imply remote atomics. A custom
collective requires correctness, capture and matched end-to-end evidence; use the
current program and ADR-0008 rather than a historical topology assumption.

## Telemetry interfaces

| Need | Interface | Rule |
|---|---|---|
| Empty/post-load VRAM | B70 debugfs `vram_mm` fields | Verify PCI node; retain `visible_avail` |
| Configured cap | hwmon `power1_cap` | Record as-found before the campaign; restore that value at cleanup |
| Average/interval power | hwmon `energy1_input` | Use energy delta over a monotonic interval |
| Temperatures | hwmon `temp*_label` and `temp*_input` | Discover labels on the verified device |
| vLLM cache/spec state | OpenMetrics `/metrics` counters | Diff cumulative counters over the measured phase |

`xpu-smi` is not the recorded telemetry authority for this stack. Follow the [observability standard](../methods/observability-standard.md).

## Current-state sources

- `docs/operations/current-host-state.md`: dated inventory pointers; live discovery remains required.
- [B70 Benchmark Protocol](../methods/benchmark-protocol.md): normative campaign configuration and provenance requirements.
- [Recommended Configurations](../operations/recommended-configs.md): current per-model choices, subject to artifact availability.
- [Benchmark History](../../archive/legacy-docs-20260915/benchmark-history.md): dated stack identities attached to campaigns.
- [Launcher Guide](../../archive/legacy-docs-20260915/compat-stubs/launcher-guide.md): launcher architecture and operational behavior; verify examples against the live installation.

## Interpretation rule

A stack identity is a coordinate, not a performance claim. Never carry a number from one build, image, model artifact, quantization, tokenizer/template, scheduler, cache policy, or speculation path into another without a matched comparison and linked evidence.

## Product specification versus measured capability

Intel's [B70 datasheet](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2026-03/datasheet-b70-gpu.pdf)
specifies 32 Xe2 cores, 256 XMX engines, 32 GB, 608 GB/s and 367 dense INT8 TOPS.
Use `architecture-diagnostic-playbook.md` under `docs/methods/` for equations and
rate-class cautions. `b70-hardware.json` is a legacy single-card catalog descriptor,
not a live topology, RAM, free-memory or power reading.

Datapath facts that decide quantization format (DPAS operand types and rates,
FP8 absence, joint_matrix vs ESIMD exposure, SLM/GRF limits) and the
measured bandwidth anchors are kept in one place:
[B70 format and lever guide](b70-format-and-lever-guide.md) §1.
