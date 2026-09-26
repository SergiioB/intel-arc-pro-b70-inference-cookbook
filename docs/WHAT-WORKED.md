# B70 inference: what to run, what to stop, what to test next

Updated 2026-09-26. Start with this page, then follow **one** model-family route. A speed result applies only to its stated image, model, patch stack, workload, cache state, power cap, and timing method. The [benchmark catalog](BENCHMARK-CATALOG.md) is the generated numeric view; [`data/benchmarks.v1.json`](../data/benchmarks.v1.json) is the authority for public benchmark records. This page is a decision index, not another benchmark table.

## Use now (measured for the stated route)

| Need | Route | Evidence boundary |
|---|---|---|
| One-card GPTQ serving and tool calls | [Qwen3.8-27B vLLM](qwen38-27b/QWEN38-VLLM-XPU.md) | The published fast C1 cells are self-reported, not a universal sustained or quality score. [Draft-only INT4](qwen38-27b/DRAFT-INT4-S-M1.md) has a matched single-card speed win; its target remains unchanged, but draft logits change. Use the exact pinned stack. |
| GGUF or local vision on one card | [Muse-Glimmer llama.cpp](muse-glimmer/MUSE-GLIMMER-B70.md) | Measured on its own model, engine, and context; do not rank its engine-native rate against vLLM client post-first rates. |
| MoE with long context | [Qwen3.6-35B-A3B](qwen36-35a3/QWEN36-MOE-VLLM-XPU.md) | The historical MTP4 record belongs to its original pinned session. A [later same-family comparison](qwen36-35a3/MOE-INT4-THREE-ENGINE-CEILING-20260917.md#speculation-verdict-engaged-zero-gain-one-net-loss) did not reproduce its short-prompt speed. Treat MTP4 as stack-specific, not the default win. |
| Two cards for independent users | Run [one server on each card](DUAL-B70-TP2.md) | This avoids treating tensor parallelism (TP2) as a guaranteed single-request speedup. The [FP8 TP2 route](qwen38-27b/FP8-TP2-W8A16.md) remains research evidence on its own stack, not the default deployment. |

**Correctness gate:** A passing [four-canary smoke](../scripts/canary.py) checks basic syntax and arithmetic, not exact code execution or broad answer quality. Do not describe the GPTQ checkpoint as quality-equivalent to another quantization based on those four probes. Before using it for coding decisions, compare exact answers with a reference artifact on your own task set. Do not infer quality from throughput or greedy replay alone.

## Do not repeat unchanged

| Attempt | Evidence and scope | Decision |
|---|---|---|
| Native MTP4 on the later Qwen3.6 MoE stack | [Short-prompt matched sweep](qwen36-35a3/MOE-INT4-THREE-ENGINE-CEILING-20260917.md#speculation-verdict-engaged-zero-gain-one-net-loss) shows engaged speculation without useful net speed; llama.cpp draft loses at short prompts but helps at long loaded context. | Test by context and stack; do not copy the earlier record as a new default. |
| DFlash2 in place of native MTP4 for the dense GPTQ route | [DFlash2 comparison](qwen38-27b/MTP-DEPTH-DFLASH2-KNORM-20260914.md) leaves it not ready; changed target precision and topology confound any recovery attribution. | Keep the native route unless a new, matched quality-and-speed comparison passes. |
| Verify-head quantization or mixed GGUF precision as a free speed win | [Draft and quantization review](QUANTIZATION-AND-DRAFTERS.md) and [format comparison](qwen38-27b/QUANT-FORMAT-SHOWDOWN-B70.md) record quality or decode tradeoffs. | Do not lower target verification precision or mix quants on a speed forecast alone. |
| TP2, pipeline parallelism, and large-model placement as automatic decode gains | [FP8 TP2](qwen38-27b/FP8-TP2-W8A16.md), [engine comparisons](qwen38-27b/ENGINE-COMPARISON-CTX-20260914.md), and [Flash-Next](qwen38-flash-next/README.md) are different artifacts, workloads, and capacity goals. | Use two independent servers by default. Research TP2 only against a same-session one-card control. |

Recent lab screens of the external shortlist, the rebuilt own-corpus shortlist, and host-staged TP2 did **not** earn a replacement public recipe. The own-corpus screen failed its matched thermal gate; no finalist ran. Their raw runs are not published in this repository. Do not quote private rates, treat an external B70 result as local reproduction, or change the image/patch matrix from those screens.

## Next decisions, in order — 2026-09-26 status

1. **Protect answer quality before chasing speed — MEASURED 2026-09-26 (screen).** The exact-answer battery now has three arms on this host: the serving GPTQ checkpoint fails `sum(i*i for i in range(4))` (answers 30, correct 14, deterministic ×2); a same-source Q4_K_M GGUF answers 14 but fails 9×7−3 (answers 59, correct 60); an EXL3 4 bpw checkpoint answers every case correctly. No route is exempt from exact-answer gating, and none of these batteries is broad enough to crown a quality default by itself. Keep the battery as the standing drift gate for every serving change.
2. **Long-context sparse attention for Flash-Next — MEASURED, screen PASS.** The merged, opt-in llama.cpp [SYCL sparse-attention change](https://github.com/ggml-org/llama.cpp/pull/28796) measured **+7–9 % decode at occupied 28K** on our two-card placement, with kernel engagement proven and no answer drift. Caveats: the pinned build needed a local port of the upstream `n_kv_max` graph plumbing to engage at all, and the win is muted because this placement's step is CPU-expert-bound. Screen-level (n=3); not a recipe until rebased and confirmed at a deeper cell.
3. **Quantized-KV decode dispatch for dense GGUF — MEASURED, screen PASS.** The merged [Xe2 TILE dispatch](https://github.com/ggml-org/llama.cpp/pull/26689) measured **+13 % median at 32K occupied depth** on the Qwen3.8 Q4_K_M champion cell (q8_0 K / q4_1 V, `-fa on`), every interleaved pair positive. Cheapest confirmed lever of the day: it is a 6-line dispatch change on the recorded champion build commit. Still to check before a recipe: MTP/speculative serving, where verification was already TILE and the external full-serving result was ~0 %.
4. **Measured draft-INT4 win stands; do not repeat its A/B.** Already confirmed publication-grade. The own-corpus shortlist screen was thermally invalid and below gate; unchanged repeats stay blocked.
5. **EXL3 on one B70 — quality-first alternative, MEASURED (screen).** First local run of the [EXL3 XPU plugin](https://github.com/0xSero/exl3xpu) (pinned image + revision-pinned 4 bpw checkpoint): **the only arm with a clean exact-answer sheet**, and decode in the same class as the GPTQ champion (~65–71 tok/s post-first implied at p512/p8192 vs 72–73). It is a changed checkpoint and engine, not a speed win. Details: [EXL3-XPU-EVALUATION-20260926](EXL3-XPU-EVALUATION-20260926.md). Before any recipe: prose-corpus decode campaign, repeat-stability, needle, and agent traffic.

Closed today without GPU spend: Intel `0.26.0-b2` (our pinned 0.27.2rc1 already carries the relevant fixes; the kernels bump already measured as a loss) and Cascadia 0.2.4 (pipeline retains ~100 % of one card, exactly as our PP2 run measured; llama.cpp RPC split retains only 54 % on the same cards — recorded as the never-RPC-split datum; independent per-GPU servers remain the two-card default).

Publish new numbers only through the [benchmark format](BENCHMARK-FORMAT.md) and [recipe confirmation process](CONFIRM-A-RECIPE.md); the three screens above are lab-grade and labeled as such. Preserve old result generations; a failed new stack does not erase a valid historical measurement.

