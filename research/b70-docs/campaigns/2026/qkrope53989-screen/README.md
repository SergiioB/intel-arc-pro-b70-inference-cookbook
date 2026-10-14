---
status: provisional
title: "Qwen3.8-27B fused QK-RoPE dispatch screen"
type: campaign
date: 2026-09-23
---

---
id: CAM-2026-QKROPE53989-SCREEN
status: provisional
---

# Qwen3.8-27B fused QK-RoPE dispatch screen

- Model: Qwen3.8-27B GPTQ-INT4 G128 + MTP4, TP1 vLLM XPU champion route.
- Hypothesis: enabling the existing fused QK-RMSNorm + RoPE + gate Triton path on text-only XPU improves C1 decode without prefill regression.
- Treatment: local one-file dispatch hunk on the f01e24f6 champion image, kernels 0.1.12.3 unchanged; control uses the unpatched image.
- Evidence: `results/2026/09/qkrope53989-abba-20260923/screen/` (`../../../results/2026/09/qkrope53989-abba-20260923/screen/`), especially `RESULTS.md` (`../../../results/2026/09/qkrope53989-abba-20260923/screen/RESULTS.md`) and `comparison-authoritative.json` (`../../../results/2026/09/qkrope53989-abba-20260923/screen/comparison-authoritative.json`).
- Verdict: **failed promotion / no speed claim**. Corrected p512/g128 median was -10.97%, p8192/g128 +11.68%; ranges overlap at both cells, acceptance shifts exceed the predeclared guard, and thermal readings violated the <=55 C cooldown requirement.
- Upstream status: equivalent XPU dispatch is already merged as vLLM #56096; no duplicate PR is warranted. The local screen did not trace actual fused-op dispatch.
- Limits: the records are diagnostic only; they do not establish a performance regression or gain. No fresh finalist was run.
