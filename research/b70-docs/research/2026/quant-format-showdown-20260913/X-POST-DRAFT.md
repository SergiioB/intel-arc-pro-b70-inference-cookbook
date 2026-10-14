---
title: "X post draft — quant-format showdown night (2026-09-13/14)"
type: research
date: 2026-09-13
status: final
---

# X post draft — quant-format showdown night (2026-09-13/14)

Status: DRAFT — posting requires separate user authorization (AGENTS.md §12).
Claims-audit applied 2026-09-14 (audit report in campaign tree); all numbers below
match the corrected FINDINGS.md (canonical B0 = clean-host set, 24.41/23.89).
Not "breakthrough"-worded; route-reconfirmation framing per audit §C.

---

## Option A — full thread (recommended), 6 posts

**1/6**
One night, one Intel Arc Pro B70 (32GB, 150W), one question: what's actually the
best weight format for a dense 27B hybrid-GDN model — the AutoRound INT4 route
everyone recommends, or classic GGUF quants?

Ran a full format showdown on Qwen3.8-27B. Representative numbers fresh n=5;
diagnostics labeled. One card, serial arms, protocol-controlled (one disclosed
host-contention slip, corrected + superseded in the writeup). 🧵

**2/6**
Baseline: the registered serving route — GPTQ INT4 sym G128 (W4A16) on vLLM XPU,
compile-only, prefix cache off, recommended sampling.

Spec-off decode (p512/g128): 24.4 tok/s — identical at 150 W and the stock 230 W
cap (decode is launch-gap-bound; the cap can't add what the workload won't draw).
Cold input though: 1047 → 1586 tok/s (+53%) going 150 W → 230 W. The eco cap was
strangling prefill, not decode.

Add native MTP-4: 45.7 tok/s (+87%) at p512, 37.1–38.0 at p8192; mean accepted
length ≈3.1 (±~10% across content draws). (45.7 = endpoint-token throughput —
under spec decode tokens arrive in bursts, not a smooth SSE stream.)

**3/6**
Landmine found first: `--enforce-eager` on this stack = 10.3 tok/s. Compile-only is
2.36× faster decode at similar wall power — roughly 2.4× worse energy per token if
you leave eager on.

If you benchmark vLLM XPU in eager and walk away sad — you measured the mode, not
the hardware. Execution mode is part of the config. (A/B retained, n=5.)

**4/6**
Then the GGUF matrix — same BF16 source, fresh quantizations, llama.cpp SYCL,
KV q8_0/q4_1, llama-bench r=5:

- Q4_K_M (16.8GB): 20.0 tok/s decode
- Q4_K_M + Q6/Q8 GDN/attn rescue (18.6GB): 18.9
- Q5_K_M (19.5GB): 17.0
- Q8_0 (29GB): 13.6

Decode tracks streamed weight bytes. Long-prompt prefill (pp2048/pp8192): flat
844–871 tok/s across ALL formats — kernel-bound, not bandwidth-bound. (Short-prompt
pp512 sits lower, 611–677 — launch-bound, different regime.)

**5/6**
Sleeper detail: Q8_0 streams ~396 GB/s effective vs Q4_K_M's ~336 — the trivial
dequant path utilizes GDDR6 better per byte than K-quant super-blocks. The decode
curve is kernel efficiency × size, not just size.

And a warning for hybrid-GDN models: wikitext perplexity came back NON-monotonic
in precision — the full F16 reference scored worse than Q4_K_M on 7/8 chunks.
We root-caused it (recurrent-state accumulation in the eval, not kernels) and
filed it upstream: ggml-org/llama.cpp#28879. Don't rank quant formats with
llama-perplexity on linear-attention hybrids.

**6/6**
Verdict: on one B70, AutoRound-INT4 + MTP-4 on vLLM stays the C1 decode king for
this model (route re-confirmed, not a new record); GGUF Q4_K_M is the llama.cpp
workhorse; mixed-precision "rescue" tensors cost 6% decode and bought no measurable
quality — the Q4_K_M preset already allocates precision well on this GDN hybrid.

Full writeup + evidence: ggml-org/llama.cpp#28879 (upstream) +
cookbook records (PR #19, intel-arc-pro-b70-inference-cookbook). PROVISIONAL,
self-reported, raw samples retained — independent repro welcome.

---

## Option B — single post

One B70, one night, one dense 27B: the INT4 format showdown 🧵

AutoRound/GPTQ INT4 G128 (W4A16) + MTP4 on vLLM XPU: 45.7 tok/s decode C1 (37.1
at p8192), +87% over spec-off — the route of record, re-confirmed fresh n=5.

Same-source GGUFs on llama.cpp: Q4_K_M 20.0, mixed Q4/Q6/Q8 18.9, Q5_K_M 17.0,
Q8_0 13.6 tok/s. Decode = streamed bytes; long-prompt prefill flat 844–871 across
all — kernel-bound. And `--enforce-eager` is a 2.36× decode trap.

All representative numbers n=5, 150W, one card, raw evidence + claims audit
retained. Full writeup: [link]

---

## Compliance notes (post-audit, 2026-09-14)

- Audit fixes applied: canonical B0 set (24.41) with +87%/+55%; eager ratio 2.36×
  (retained pair); power claims reduced to per-token energy statement; prefill-flat
  qualified (long-prompt only); "buy quality" deleted (§7.4 says the opposite);
  burst semantics surfaced in-thread; acceptance given as log-derived ≈3.1 range;
  "representative numbers fresh n=5" wording; slip disclosure line in post 1/6.
- All figures single-host self-reported (E2); label as such; no LocalMaxxing
  submission without separate authorization.
- 45.7/37.1 are client-side endpoint-token medians under MTP4 burst arrival
  (documented in FINDINGS §3).
- Cross-engine rows (vLLM vs GGUF) are directional only, not matched claims.
- Do not post the word "breakthrough" (audit §C); this is a re-confirmation + novel
  negative/characterization findings night.
- Posting still requires separate user authorization.
