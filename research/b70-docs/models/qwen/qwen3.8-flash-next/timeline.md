---
status: living
title: "Timeline"
type: model-dossier
date: 2026-08-31
---

---
schema_version: 1
model_id: Qwen/Qwen3.8-Flash-Next
model_slug: qwen3.8-flash-next
document: timeline
as_of: 2026-09-01
status: research_foothold
evidence_tier: E2
---

# Timeline

| Date | Event |
|---|---|
| 2026-08-26 | Official inventory + XPU construction fixtures; conversion blocked on storage/kernels (`research/2026/qwen38-flash-next-b70-adoption-20260826.md`) |
| 2026-08-31 | Dual-B70 llama.cpp GGUF foothold: PLE-on-CPU, 8K then 16K/128K C1 screens |
| 2026-08-31 | Decode wall isolated as `-t 14` oversubscription; recipe `-t 6 -tb 14` |
| 2026-08-31 | ngram / pin / FA-off rejected |
| 2026-09-01 | Fused IQ3_S/IQ4_NL `MUL_MAT_ID` MMVQ + F16 prefill binary; n=3 provisional |
| 2026-09-01 | Kernel acceptance + GPTQ ABI freeze (GGUF fused ≠ vLLM Xe2 INT4) |
| 2026-09-01 | Synthetic `XpuFusedMoe` Flash-Next-shape probe passed; QSA still NIE; conversion gates closed without download |
| 2026-09-01 | n=5 C1 confirmation: fused-FP32 c8 p512 **23.38** decode; fused-F16 c16 p9096 **594.49** cold input |
| 2026-09-01 | GPU MUL_MAT_ID grouping (no expert-ID D2H): +0.81% prefill n=3 vs disable |
| 2026-09-01 | Private COOKBOOK-DRAFT written; not public |
| 2026-09-10 | Upstream main 52d4268 supersedes local patches: no-spec c8 decode 26.86 n=3 (+15% free); F16 prefill 585.9 c16 p9096 |
| 2026-09-10 | qwen4exp MTP ported (dzannotti patch + manual walk/ctor + 2 drift fixes); draft-mtp working with dzannotti Q4_K_M head |
| 2026-09-10 | MTP tuned (n-max3, backend sampling, draft@SYCL1): 29.9 n=3 / 26.5 n=5; acceptance 87.7% natural text |
| 2026-09-10 | Local multi-token verify kernel `_mt` added (correct; gain within noise — draft-step cost binds) |
| 2026-09-10 | Quality battery PASS: 42/42 code, divergence inside OFF/OFF envelope, prose vs code both hold |
