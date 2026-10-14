---
status: living
title: "Serving"
type: model-dossier
date: 2026-09-15
---

---
schema_version: 1
model_id: facebook/Muse-Glimmer-30B
model_slug: muse-glimmer-30b
document: serving
as_of: 2026-08-19
status: experimental
truth_verdict: PARTIAL
evidence_tier: E2
engines: [llama.cpp-sycl, vllm-xpu]
primary_evidence:
  - ../../../results/2026/08/muse-dflash-sweep-manual-20260810T164735/FINAL-TABLE.md
  - ../../../results/2026/08/muse-glimmer-int4-screen-20260812T192845Z-835270/manifest.txt
  - ../../../archive/legacy-docs-20260915/benchmark-history.md
---

# Serving

## llama.cpp SYCL

Required build property: `GGML_SYCL_F16=ON`. The 128K-context campaign used Q4_K_XL + mmproj + DFlash draft, F16-enabled SYCL build, one slot, q8_0 K / q4_1 V KV, `-ub 8192`, and a configured 230 W cap.

| Context configuration | Cell | Result | Sample class |
|---|---|---:|---|
| 64K | p512/g128 | 29.20 tok/s | n=5, corrected F16 build |
| 64K | p8192/g128 | 22.28 tok/s | n=4; fifth repetition lost |
| 128K | p512/g128 | 26.8 tok/s | n=5, DFlash n2 |
| 128K | p8192/g128 | 22.9 tok/s | n=5, DFlash n2 |
| 128K | p32768/g128 | 21.1 tok/s | long-context comparison |

These are engine-reported decode rates. Do not mix the 64K and 128K p512 figures. DFlash n2 was the winner; n8 aborted the server at launch. A 128K context configuration fit, but the campaign did not complete a near-128K boundary request.

Avoid port 8765 when the vLLM profile manager is active; it killed the Muse server during the recorded work.

## vLLM XPU

The compressed-tensors route is an n=3 text-only provisional screen on a local patched image. It is not the default recipe, and its text/vision smoke does not make the overlay production-ready.
