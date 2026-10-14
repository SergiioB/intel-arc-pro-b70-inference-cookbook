---
status: final
title: "Qwen3.8-27B Official DFlash2 objective campaign"
type: campaign
date: 2026-08-30
---

---
id: CAM-2026-QWEN38-DFLASH2-OBJECTIVE
doc_id: DOC-CAM-2026-QWEN38-DFLASH2-OBJECTIVE
status: provisional
---

# Qwen3.8-27B Official DFlash2 objective campaign

- Decision authority: [Qwen3.8-27B Official DFlash2 Objective Campaign — research](../../../research/2026/qwen38-dflash2-objective-campaign-20260830.md)
- Model: [Qwen3.8-27B dossier](../../../models/qwen/qwen3.8-27b/README.md) / [serving route](../../../models/qwen/qwen3.8-27b/serving.md)
- Draft artifact: [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) (2B BF16, block size 8) — pinning record appended at step 2
- Implementation: upstream [vLLM PR #52816](https://github.com/vllm-project/vllm/pull/52816) lineage (merged, head `19c93519`) applied to the pinned XPU generation
- Evidence root: `results/2026/08/qwen38-dflash2-objective-20260830/` (`../../../results/2026/08/qwen38-dflash2-objective-20260830/`) with per-phase subroots (`unit-gates/`, `correctness/`, `screens/`, `finalist/`, `cleanup/`)
- Evidence tier / publication state: E2 self-reported raw evidence — **PROVISIONAL — NOT FOR PUBLIC HEADLINE**

## Lane

C1, TP2/PP1, official FP8 target, W8A16/small-M verifier stack, per-worker `ZE_AFFINITY_MASK` + `SYS_PTRACE`, `VLLM_XPU_ENABLE_XPU_GRAPH=0`, 230 W configured caps, cache off (zero-hit-required), GDN mixed-split v5.

- Tuning cells: p512/g128 and p1024/g128.
- Stability diagnostic: p2048/g128 (not a guaranteed W8A16 production route).
- Spec depths: DFlash2 k=7 (primary) vs native MTP7 (matched control) vs MTP8 (incumbent, operational) vs no-spec (correctness control).

## Acceptance of the objective

Official DFlash2 is lossless (greedy parity over the five fixed prompts) and beat matched MTP7 on fresh n=5 medians at p512/g128 and p1024/g128, with acceptance ≥ MTP7 + 0.30, or the campaign records a completed no-go naming the failed gate.

## Status ladder

- [x] 2026-08-30 — Research authority + campaign package created; registry entry `CAM-2026-QWEN38-DFLASH2-OBJECTIVE`
- [x] Step 2 — artifact pinned (revision `dedf8df6…`, sha256 `67fc76d6…`, 81 tensors / 1.924B params, manifest under `artifact-pin/`) + PR #52816 lineage classified and ported (13-file 3-way merge onto the image generation `ac7509e2b`)
- [x] Step 3 — adoption layer in `scripts/tmp/`: `dflash2-objective-port/`, `qwen38-dflash2-objective-apply.py` (port install + XPU adaptation: torch selector walk, torch SDPA non-causal draft attention, NaN guard), `qwen38-dflash2-objective-launch.sh` (TP2 wrapper, `--workdir /` for engine-tree consistency), `qwen38-dflash2-objective-client.py` (ref/check/bench, coherent-prefix comparator + greedy-tie probe), `qwen38-dflash2-objective-unittests.py`
- [x] Step 4 — static gate 29/29 PASS; unit gates 7/7 PASS (`unit-gates/`)
- [x] Step 5 — target-load correctness: no-spec ref → MTP7 (4/5 exact + verified tie) → **DFlash2 k=7 (5/5 coherent-prefix parity)** under `correctness-*/`
- [x] Step 6 — matched n=3 screens p512/p1024/p2048 g128: **DFlash2 k=7 loses to matched MTP7 on every cell** (11.12 vs 22.14 / 11.86 vs 21.90 / 10.94 vs 18.86 e2e tok/s; acceptance 0.10–0.12 vs 0.35–0.41); no-spec and MTP8 recorded as controls
- [ ] Step 7 — fresh n=5 finalist confirmation **NOT RUN (by design)**: the n=3 screen did not win; the plan requires step 7 only for a real finalist
- [x] Step 8 — retraining decision: **completed no-go** recorded (decision log A4 in the research authority); cleanup evidence under `cleanup/` (150 W restored, both cards empty)

## Result (PROVISIONAL — NOT FOR PUBLIC HEADLINE)

**Official DFlash2 k=7 on the FP8 W8A16 TP2 stack: completed no-go.** Correctness PASS (first clean DFlash2 correctness on this host), but acceptance (≈0.7 mean accepted length vs MTP7 ≈2.8) and draft-cycle latency both lose to matched native MTP7; the leading mechanism is FP8 target-state mismatch against the BF16-trained draft. Full gate table and loss decomposition: §11 of the research authority. DFlash2 adoption on this stack is closed; MTP4 / the k=8 record remain the serving spec, and a future retry depends on an Inco/z-lab DFlash2 trainer retrained on W8A16-decode hidden states (SpecForge capture remains the diagnostic fallback).

**DSpark follow-up (decision log A5/A6):** the FP8-trained community DSpark drafter
(Doopeworld/Qwen3.8-27B-DSpark-vLLM) also fails acceptance (~2%, both layouts, both FP8 and
INT4 targets) — broadly refuting the target-state hypothesis and pointing at the vLLM 0.27.2
DSpark serving contract (first-cycle trace vs the SpecForge reference is the named next step).
Architect ceiling: ≈35–60 tok/s even with recovered acceptance; 80+ requires draft-side
cycle-time kernels. Evidence: `results/2026/08/qwen38-dflash2-objective-20260830/dspark-*`,
`dspark-fillin-correctness/`, `dspark-int4-correctness/`.