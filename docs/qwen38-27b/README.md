# Qwen3.8-27B on Intel Arc Pro B70

This directory is the family hub for the distinct Qwen3.8-27B serving routes in
this cookbook. Choose the artifact and hardware topology first; the recipes do
not share a numeric authority or an interchangeable patch list.

## Choose a route

| Route | Hardware | Artifact and serving goal | Start here |
|---|---|---|---|
| EXL3 (exl3xpu engine) | One B70 | **Primary route:** trellis 4bpw + native MTP3 + fp8 KV; only route at full 262,144 context; clean correctness | [EXL3-XPU.md](EXL3-XPU.md) |
| GPTQ-INT4 + native MTP | One B70 | General single-card Linux serving, long context, optional draft-INT4, and the established Pi/agent path — **legacy/reference: the W4A16 class fails the exact-answer battery on this model (all exports answer 30); prefer EXL3** | [QWEN38-VLLM-XPU.md](QWEN38-VLLM-XPU.md) |
| TensorFold MLX 4-bit + DFlash2 | One B70, or two for one stream (`--tp 2`, upstream PR #3586) | Lossless speculation (drafted == non-drafted); fastest measured at short prompts; prefix attention kernel on our tag for 8K–131K; two cards 1.25–1.5× one card (screen 2026-11-18) | [TENSORFOLD-B70.md](TENSORFOLD-B70.md) |
| TensorFold on two B70s, root causes and fixes | Two B70s | Why two cards gain 1.25–1.5× and not 2×, two fork fixes, decode by prompt length to 245,760 tokens (screen 2026-11-19) | [TENSORFOLD-TWO-CARDS-20261119.md](TENSORFOLD-TWO-CARDS-20261119.md) |
| 250K context and warm sessions | One or two B70s | What fits at 262,144 context, int8 against bf16 K/V, retrieval, warm 250K follow-ups in 1.1–1.6 s on two cards (screen 2026-11-19) | [LONG-CONTEXT-250K-20261119.md](LONG-CONTEXT-250K-20261119.md) |
| FP8 W8A16 + native MTP8 | Two B70 cards in TP2 | Research recipe for the Xe2 FP8 small-M kernel path; C1 evidence only, self-reported E2 | [FP8-TP2-W8A16.md](FP8-TP2-W8A16.md) |
| Windows standalone | One display-attached B70 | Docker Desktop and experimental WSLC packaging with display-safe VRAM budgets | [WINDOWS-STANDALONE.md](WINDOWS-STANDALONE.md) |
| Pi agent backend | Qwen3.8 server | Tool calling, provider configuration, and recommended thinking/non-thinking sampling | [PI-AGENT-BACKEND.md](PI-AGENT-BACKEND.md) |
| Draft INT4 overlay | One B70 | Optional draft LM-head and MTP-linear RTN INT4 overlay for the GPTQ route | [DRAFT-INT4-S-M1.md](DRAFT-INT4-S-M1.md) |
| OpenVINO GenAI | One B70 | INT4-GDN8 + grafted MTP5: single-card short-ctx champion (79 t/s @512); draft off ≥98K, full-protocol ceiling 128K | [OPENVINO-CASCADIA-REPORT.md](OPENVINO-CASCADIA-REPORT.md) · [Tuning](OPENVINO-B70-TUNING.md) |
| Cascadia (separate engine) | One B70 | Rust server + IR surgery; `qwen35` ~5 t/s measured — multi-node pipeline-parallel research path, not a B70 speed path | [Architecture](../architecture/openvino-and-cascadia.md) |

**Latest matched evidence (2026-11-11):** a five-pair one-card confirmation
compares the deployed EXL3 4.00 bpw + MTP3 and TensorFold 4-bit g64 + DFlash2
stacks. TensorFold is faster at 0–8K (the 8K non-thinking cold cell is
unresolved); EXL3 is faster from 16K and is the long-context/cache choice. The confirmation does not reproduce the earlier
screen's 40-item battery ordering under the stricter grader, so quality remains
unresolved. [ROUTE-CONFIRMATION-20261111.md](ROUTE-CONFIRMATION-20261111.md).

**Since then:** the TensorFold prefix attention kernel (2026-11-13) and a sized prompt cache (2026-11-18) change
the 16K+ and multi-turn picture: 90 tok/s at 32K and 51 at 131K on one card, follow-up turns in 0.3–1.1 s up to
131K, and two cards for one stream. See [TENSORFOLD-B70.md](TENSORFOLD-B70.md).

**Same session, greedy (2026-11-19):** TensorFold is faster for one user to 32K (1.35–1.54× on code) and level at
131K; exl3xpu reaches the first token 1.5–1.6× sooner from 32K, leads on one card at 250K (36.7 against 20.7 tok/s)
and at 8–32 streams on one card; only TensorFold repeats its text.
[ROUTE-GREEDY-SAME-SESSION-20261119.md](ROUTE-GREEDY-SAME-SESSION-20261119.md). The route decision of 2026-11-10
was made at the model card's sampling, and that rerun is pending. EXL3 stays the primary route until it reports.

Task quality against Nemotron Bolt 74B-A9B and Qwen3.6-35B-A3B on the same 94 tasks (EXL3
4.0bpw route, one card): [MODEL-COMPARISON.md](../MODEL-COMPARISON.md).

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
- [Re-measure tasks](RE-MEASURE-TASKS.md) names the pending like-for-like
  re-measurements that gate ranking reuse and benchmark-class promotion
  (P10 decisions, 2026-09-23).

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
