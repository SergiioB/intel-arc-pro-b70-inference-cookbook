# RETIRED: b70-dynamic-power.sh (2026-08-07)

This reactive power-cap manager (boost to 230W on prefill bursts, relax to 165W
on decode) was built on the hypothesis that MoE+MTP **prefill** scales with the
power cap (+16-22% claimed).

**That hypothesis was refuted by a paired, alternating 150W-vs-230W A/B**
(Run 24): p2k/p4k/p8k prefill all came back flat at ±0.2%. The earlier "+16-22%"
result was prefix-cache contamination — the benchmark harness used a constant
filler string, so reps 2-3 hit the cache and reported inflated prefill (up to
5×). With honest cold prefill (unique random prefix per call), raising the cap
above 150W gives zero gain on MoE+MTP. The grouped-GEMM is bandwidth-gated, not
power-gated.

**Do not use this for MoE+MTP.** It adds complexity and heat for no benefit.
(It may still help dense llama.cpp decode, which genuinely scales +18-30% from
150→230W — but that is a different workload.)

The paired 150 W vs 230 W A/B above is the refuting evidence.
