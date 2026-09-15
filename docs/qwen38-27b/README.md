# Qwen3.8-27B on Intel Arc Pro B70

This directory is the family hub for the distinct Qwen3.8-27B serving routes in
this cookbook. Choose the artifact and hardware topology first; the recipes do
not share a numeric authority or an interchangeable patch list.

## Choose a route

| Route | Hardware | Artifact and serving goal | Start here |
|---|---|---|---|
| GPTQ-INT4 + native MTP | One B70 | General single-card Linux serving, long context, optional draft-INT4, and the established Pi/agent path | [QWEN38-VLLM-XPU.md](QWEN38-VLLM-XPU.md) |
| FP8 W8A16 + native MTP8 | Two B70 cards in TP2 | Research recipe for the Xe2 FP8 small-M kernel path; C1 evidence only, self-reported E2 | [FP8-TP2-W8A16.md](FP8-TP2-W8A16.md) |
| Windows standalone | One display-attached B70 | Docker Desktop and experimental WSLC packaging with display-safe VRAM budgets | [WINDOWS-STANDALONE.md](WINDOWS-STANDALONE.md) |
| Pi agent backend | Qwen3.8 server | Tool calling, provider configuration, and recommended thinking/non-thinking sampling | [PI-AGENT-BACKEND.md](PI-AGENT-BACKEND.md) |
| Draft INT4 overlay | One B70 | Optional draft LM-head and MTP-linear RTN INT4 overlay for the GPTQ route | [DRAFT-INT4-S-M1.md](DRAFT-INT4-S-M1.md) |
| OpenVINO GenAI / Cascadia | One B70 | Official INT4-OV VLM research. Hub INT4 failed; custom mixed-INT4 works but is bottlenecked by host logits / GDN (15.8 tok/s). | [OPENVINO-CASCADIA-REPORT.md](OPENVINO-CASCADIA-REPORT.md) · [Architecture](../architecture/openvino-and-cascadia.md) |

## Shared topology and compatibility references

- [Quantization and Drafters](../QUANTIZATION-AND-DRAFTERS.md) — how weight
  formats and speculative drafters were chosen on Xe2, including the DFlash2
  vs native MTP closure and the drafter-selection rule for quantized targets.
- [Dual-B70 TP2 / PP2](../DUAL-B70-TP2.md) explains worker affinity,
  `SYS_PTRACE`, and the oneCCL simple-threshold configuration required on
  non-P2P desktop platforms.
- [Image and patch matrix](../IMAGE-AND-PATCH-MATRIX.md) is the compatibility
  denylist and ordered patch reference. Do not apply GPTQ-only draft patches to
  the FP8 target path, or FP8 W8A16 patches to GPTQ weights.
- [Benchmark catalog](../BENCHMARK-CATALOG.md) is generated from
  [`data/benchmarks.v1.json`](../../data/benchmarks.v1.json), the public numeric
  catalog authority.

## Evidence and claim boundary

The FP8 TP2 route is an official-lab self-report at E2 with public raw evidence;
it has not been independently reproduced. Its greedy random-token cells are
diagnostic maxima, while the model-card-sampling cells are sampling-sensitivity
diagnostics rather than natural-content quality evaluations. The GPTQ route has
separate evidence, sampling conditions, cache states, context limits, and
hardware scope. Do not compare or combine numbers across the routes without a
matched comparison.

Qwen3.8-Flash-Next is a different model:
[docs/qwen38-flash-next/](../qwen38-flash-next/README.md).
