---
title: "Standing rules: always on, always try first, closed classes"
type: governance
status: living
date: 2026-10-09
updated: 2026-10-09
---

# Standing rules

Living authority for the invariants every campaign inherits. Promoted on
2026-10-09 from `research/2026/b70-breakthrough-architect-20260928.md` §A (that
dated note stays as history) and corrected where routes changed since. AGENTS.md
and the operating guide win on conflict; model-status wins on route facts.
Format and engine choice per workload: `docs/reference/b70-format-and-lever-guide.md`.

## 1. Always on

1. **Graph capture on for vLLM-family serving** (vLLM XPU, exl3xpu). A decode
   number with capture off is diagnostic only. New kernels must be capture-safe
   and evaluated under replay; eager traces miss replayed kernels (RL-072).
2. **The route's speculation stack is locked per route.** Qwen3.8-27B: native
   MTP3 on exl3xpu. Flash-Next: Strata MTP `--spec 4 --spec-min-p 0.5`.
   Nemotron: DFlash n7. Any change is a named variable with a predeclared gate.
3. **Exact-answer battery and canary before any speed claim.** Coherence,
   replay and perplexity all passed the fluent wrong answer "30" (RL-087). Run
   the canary at boot and after any stack change.
4. **Prefix-cache policy follows AGENTS.md §7.** Off on vLLM-XPU W4A16 and
   speculative stacks until the #53912 class is fixed. On for exl3xpu with
   declared per-cell state, the zero-hit-delta guard and ≥64 characters of
   leading entropy. Any new engine × draft × cache combination gets one cache-on
   multi-turn corruption check first.
5. **Output preservation.** Every decode-path change is lossless by
   construction or proven byte-identical (greedy) / same modal class (sampled).
   No approximate verify.
6. **Metric naming is a claims rule.** Name the counted quantity: engine tokens,
   client post-first tokens or SSE chunks. Chunk rate is not tok/s (the "24.6"
   EXL3 figure was chunks). C1 and Cn never share an unlabeled column.
7. **Acceptance bookkeeping.** Report accepted tokens per round and per-position
   marginal survival separately; never treat marginals as conditionals.
8. **One measuring client per C1 server**, cache and spec counters bracketing
   each cell.
9. **Placement.** Independent per-card stacks (ADR-0008) unless a reviewed route
   decision says otherwise (ADR-0011 is one card; a two-card single-engine
   route needs its own decision). Never a stock Docker TP2 launch.
10. **Quantization axes are independent:** target weights, draft body, draft
    head, verify LM head, activations, KV. A verdict on one never transfers.
11. **Attribution honesty.** "At floor" names the floor it was proven against;
    active-kernel share is not wall share.
12. **Thermal and power.** The ≤55 °C cooldown is unreachable on this host
    (RL-065, RL-069): declare the sustained regime instead. Record as-found caps
    and clocks per campaign and restore those, not a remembered 150 W.
13. **Allocation safety.** Never allocate beyond live free VRAM − 4 GiB; never
    run arena probes on the display card (RL-123).

## 2. Always try first (cheap before expensive)

1. Capture on (or the engine equivalent) before reading a decode number.
2. Price before building: `research.py price` with declared provenance. The
   k=3 clamp adopted on arithmetic alone cost −5.45 %.
3. Canary first, speed second.
4. Read the counter: tokens, chunks or rounds?
5. **Residency before streaming.** For MoE, compare total expert bytes to the
   VRAM you can give them (one card or a two-card layer split) before tuning any
   streaming path. A miss stream at 11.5 GB/s is a third of a Flash-Next round.
6. **Block alignment before kernel blame.** Xe2 splits misaligned 16-byte loads;
   Strata measured 2.3–4.7× from aligned access. IQ4_XS (136 B), IQ3_XXS (98 B)
   and Q8_0 (34 B) blocks are not 16-byte aligned.
7. **Rebase before tuning.** Compare the engine pin to upstream (the local
   llama.cpp tree was ~2 months behind master on 2026-10-09, RL-126). A fix that
   already exists upstream is not a research question.
8. Pull the per-position acceptance profile before assuming a shape; the
   bimodal 0.98/0.98/0.12/0.09 profile is not universal (RL-093).
9. If a drafter is separately quantizable, test the draft-side pair (+39.6 /
   +48.6 % on the removed GPTQ stack); the EXL3 MTP head has no separate pair.
10. Price the KV read at the campaign's context: 32–38 KiB/token fp8 on the 27B
    (RL-085), co-dominant with weights past ~100K.
11. Placement arithmetic before "one card": hidden states are ~10–40 KB per
    round over 11–12 GB/s copy-only P2P. Ask what could run on the idle card.

## 3. Closed — do not retry without a named changed variable

| Class | Evidence | Re-entry condition |
|---|---|---|
| Symmetric INT4 W4A16 (GPTQ, AutoRound ×2) on Qwen3.8-27B | RL-087; defect is in the weights (RL-092) | Operator reopening of RQ-09 with a new calibration class |
| DFlash2 / DSpark drafters on quantized 27B targets | DFlash2 campaign §12: 25 vs 51 tok/s | Drafter trained on the quantized target's hidden states, ≥1.5× native-head acceptance on held-out text first |
| EXL3 kernel probe program (VERIM4, thread sweep, residual split, K3) | RL-097, RL-101, RL-112; operator stop RL-122 | Operator reopening |
| Stock TP2 (oneCCL / xccl), EXL3 TP2 | RL-075 −45…−71 %; RL-103 | RQ-06 copy-only collective ≤60 µs |
| PP2 (Cascadia), llama.cpp RPC split | RL-051 −0.26 %; RL-083 −46 % | New transport |
| Acceptance frontier on the old GPTQ stack: MTP-head small-corpus retrain, probabilistic drafting, bf16 draft KV, foreign shortlist, verify-head INT4, depth >4 | D11/D12/D14, RL-074, RL-041 | Per-item new mechanism |
| W4A8 prefill | closed by attribution, ≤1.27× | New attribution showing larger GEMM share |
| QuaRot → INT3, s4×s4 decode, megakernel | no INT3 kernel; decode not compute-bound; ADR-0009 | Compute-bound decode or new kernel class |
| Grouped-IQ3 from-scratch kernel | 1.39–1.41× slower | Upstream kernel delta |
| Strata host-staged second-card expert cache | RL-125: 33.44 vs 48.03 tok/s | DMA path between cards |
| Storage-tier Flash-Next build (sybil) | RL-124 | ≥26 GB/s NVMe array or ≥92 GB free |
| Invented Q3DT (lossless IQ3 container), M5X | `b70-quant-lever-hunt-20261009-summary.md` | New entropy evidence / priced issue saving |
| vllm-xpu-kernels 0.1.15, #29107, fused QK-RoPE #53989/#56096, host-prep cache, max-num-batched-tokens, `-sm row`, CPU-MoE DSpark | ledger | Named change |
