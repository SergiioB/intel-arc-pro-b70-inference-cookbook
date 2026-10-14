# Measured Lever Verdicts — What External Research Should Not Re-Propose

> For agents and researchers proposing optimization levers for the Arc Pro
> B70 stack. This host has already measured most of the obvious ideas.
> Verdicts below are measured on the reference dual-B70 host unless marked
> external. Publication-grade numbers live in `data/benchmarks.v1.json`;
> figures marked *(internal)* are lab-campaign measurements not yet promoted
> to the public catalog. Last updated: 2026-10-14.

## TL;DR

Before proposing a lever, check the status table. Roughly two-thirds of the
levers that external deep-research reports generate for this hardware have
already been measured here — about half of those are dead, a quarter are
already adopted, and the rest are open with defined gates. External
benchmark numbers from other hosts are hypotheses, not evidence.

## Hardware premises external reports keep getting wrong

| Premise in external reports | Measured on this host |
|---|---|
| "No P2P between the two B70s (`p2p_access: 0`)" | **Wrong.** Direct-peer DMA copies verified bidirectional at ~11–12 GB/s, checksum-clean, ACCESS flag set. What is absent: peer **atomics**. Collectives must be copy-based, not atomic-synchronized. |
| "INT4 = 734 TOPS, so 4-bit is a compute lever" | **Wrong for serving.** s4×s4 doubling exists only in an isolated microbenchmark. Every W4A16 path dequantizes to FP16/BF16 before DPAS — INT4 weights run in the ~160–183 TFLOPS BF16 class. INT4 is a bandwidth/capacity lever, never a compute multiplier. |
| "Both cards on PCIe 5.0 x16" | The reference dual-B70 board runs CPU-attached **x8 + x8**. Measured H2D ~8–9 GB/s (probe-limited floor, not link bandwidth). |
| "Sub-8-bit operands add TOPS" | On Xe2, any ≤8-bit integer mix with an 8-bit operand ties the INT8 rate class. No rate gain below INT8 in any served mix. |
| "FP8 is native" | No native FP8 XMX DPAS on Battlemage. FP8 weights/KV run software conversion paths with real latency cost. |

## Lever status table

| Lever (as typically proposed) | External claim | Status here | Measured verdict |
|---|---|---|---|
| EXL3 4.00bpw via exl3xpu for Qwen3.8-27B | "fixes GPTQ corruption, faster" | **ADOPTED** | Already the primary dense-27B route on this host (dedicated container stack). External speed figures not reproduced yet. |
| Draft-side INT4 (draft LM head + MTP body) | — | **ADOPTED (WIN)** | Champion lever: +40–49% single-stream decode vs BF16 draft *(internal, n=5)*; greedy output byte-identical. Only the **pair** pays; each half alone is inert. Requires graph capture on. |
| XPU graph capture | — | **ADOPTED (WIN)** | ~1.68× on the champion cell by itself; without it the draft-INT4 pair measures ≤1.5%. |
| Concurrency scaling | — | **WIN** | Aggregate throughput ~2.8× at B=32 vs B=1 on the champion cell; knee not saturated. Aggregate and single-stream numbers are different metrics — never cross-compare. |
| Ubatch 1024 (llama.cpp SYCL MoE prefill) | — | **WIN** | 1.29× on the measured MoE cell. Above 1024 is saturated; 4096 OOMs. Exhausted. |
| TP=2 via `CCL_ENABLE_SYCL_KERNELS=0` + env triple | "444–896 tok/s, fixes TP2 crashes" | **OPEN — gated** | Direction is right (host-staged/copy collectives match the measured no-atomics hardware), but env vars alone are not the fix: the working recipe also needs the per-worker affinity patch and `SYS_PTRACE`. Pre-declared gate +20–50% vs matched 1-card control. Honest modeled band ~1.2–1.5× single-stream; measured serial-collective cost (~112µs × ~128 calls) puts naive launches near the bottom of it. See `docs/` TP2 notes and the break-even math below. |
| Hybrid-GDN prefix-cache pinning (16-token boundary) + disabling async scheduling | "hit rate 0–38% → >85%, warm TTFT 22.6s → <1.5s" | **OPEN — hypothesis** | Mechanism is coherent (spec-decode rejections straddle block boundaries on the 48 GDN layers; stale-index gather defect is a real upstream bug class), but unverified on this host. Cheapest open test: flag-level A/B on a C=5 multi-turn harness with bracketed prefix-cache counters. |
| TurboQuant KV (tq4_nc / tq3_nc) | "302–367K context, 2.3× long-ctx throughput, lossless" | **SCOPE-LIMITED** | Only 16 of 64 Qwen3.8-27B layers are GQA softmax attention; the other 48 are GDN recurrent state that TurboQuant cannot compress. At 131K context the entire GQA FP8 KV is only ~2 GiB — tq4_nc saves ~0.6 GiB. Not worth its Triton decode overhead here. Interesting only for 100% full-attention models. "Bitwise-identical PPL" claims for lossy 3-bit KV should be read as "identical when rounded." |
| MTP/draft-head retraining (QLoRA distillation, FastMTP-style, small corpora) | "α 0.48 → 0.88" | **DEAD — measured** | Retrained head reduced acceptance and end-to-end speed on this stack *(internal)*. External acceptance numbers on other stacks are not evidence. |
| Alternative draft sampling (probabilistic) / lookahead k≠4 | — | **DEAD — measured** | Probabilistic drafting failed its n=5 finalist; k=3 and k>4 both measured worse than fixed k=4. Depth is closed absent a changed mechanism. |
| Verify-pass LM-head INT4 | — | **DEAD — measured, retired** | Weight-only INT4 on the verify head moves the target distribution itself (~72% top-1 agreement vs ≥99% promotion line). Same reason fp8 recipes exclude `lm_head`. |
| Triton kernel rewrites for decode (large-GRF, block2d, slice swizzling) | "2.8× geomean, up to 35× MoE" | **DEAD for decode / OPEN for MoE prefill** | Single-card dense decode is measured at the GDDR6 byte floor — 87.7% of step time is GEMM, no hidden waste to reclaim. The external 2.8× figure is on un-tuned baselines on other models. MoE IQ3_XXS prefill remains open — see below. |
| Grouped-GEMM for IQ3_XXS MoE prefill (llama.cpp #29245 extension) | — | **OPEN — isolated win only** | Native fused grouped op measured ~3.6× faster than the fallback on the isolated operator *(internal)*; no end-to-end gain established yet. This is the only live kernel lever on the llama.cpp SYCL MoE path. |
| SGLang-XPU (runtime pinning fix) | "version mismatch root cause; pinning unblocks" | **DEAD — screened** | Screened to do-not-attempt after garbage-output failures. A pinned-runtime retest is possible but low priority. |
| NVMe / host-RAM expert offload for 125B-class MoE | "serve 125B on one card" | **DEAD as a speed route** | Works as a capacity demo only; measured several-fold slower than the dense champion *(internal)*. DDR4 and NVMe staging cannot reach interactive decode on this topology. |
| BITCOS sub-ternary packing | "125B fits one card → 180–240+ tok/s" | **NOT APPLICABLE** | BITCOS packs *natively ternary* checkpoints (BitNet b1.58-class). The Flash-Next artifact is a low-bit quant of a non-ternary model — re-encoding to {-1,0,+1} is a quality collapse, not a repack. The "theoretical ceiling → realized" projection is roofline arithmetic, not measurement. |
| vllm-xpu-kernels bump (≥0.1.15 fused sampler) | "free upstream win" | **DEAD — measured regression** | −4% to −14.5% throughput on this stack at matched cells *(internal)*. Kernel-bump levers must be screened, not assumed. |
| IQ3 dot-product reorder upstream port | "+75–95%" | **DEAD for our path** | The external win is on dense models where all decode weights reorder. Measured +0.08% end-to-end here — the routed `MUL_MAT_ID` path is untouched by that PR *(internal)*. |

## TP=2 break-even math (so the next proposal doesn't redo it)

Single-card champion step: ~52 ms per MTP4 verification cycle, ~87.7% in GEMM
(byte-floor reads). Sharding weights across two cards halves per-card GEMM
bytes: Δ ≈ −22.8 ms. Cost side: ~128–130 all-reduce collectives per step
(2/layer × 64 + MTP), each moving ~51 KB at M=5 tokens.

- Pure wire time over the measured ~11.5 GB/s peer path: ~1.2 ms — negligible.
- The whole question is per-collective launch+sync latency `t_sync`:
  - `t_sync` must be < ~170 µs for TP2 to beat single-card at all.
  - Measured serial collectives here run ~110 µs/call → ~14.3 ms tax →
    lands ~1.2×, at the edge of the +20–50% adoption gate.
  - Reaching the 1.4–1.5× band needs XPU-graph/persistent collectives to cut
    `t_sync` below ~30 µs. Unproven — that is the experiment.

## Rules for proposing levers on this host

1. State the **cell** (model, engine+version, quant, prompt/gen tokens,
   concurrency) and keep engine-rate and client-rate in separate labeled
   fields.
2. External numbers are hypotheses. A lever needs a matched control on this
   host and a pre-declared effect-size gate before it counts as a win.
3. A lever measured dead stays dead unless the mechanism changed — new code,
   new kernel, new artifact, not a rerun.
4. Do not propose: retraining the stock MTP head on small corpora, draft-depth
   or draft-sampling changes, verify/LM-head quantization, decode-side kernel
   hunting on the dense champion, host/NVMe offload as a speed route, or
   ternary repacking of non-ternary checkpoints.
5. Live open tracks: the gated TP2 campaign, the IQ3_XXS grouped-GEMM
   end-to-end test, and the hybrid-GDN prefix-cache hypothesis.
