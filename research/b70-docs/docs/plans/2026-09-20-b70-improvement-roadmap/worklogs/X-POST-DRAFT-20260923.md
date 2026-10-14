---
title: "X post draft — draft-side INT4 on one Arc Pro B70 (DRAFT ONLY, NOT POSTED)"
type: worklog
date: 2026-09-23
status: draft
---

# X post draft — 2026-09-23

**DRAFT ONLY. NOT POSTED. Posting requires Sergio's separate explicit authorization.**
Deliverable rule: exactly ONE post, not a thread. Plain and human — no emoji, no
bold-label/sectioned formatting, no thread-marketing voice.

Numbers are the 2026-09-22 publication-grade re-measure (fresh n=5 per arm, client
post-first rate, measured energy). While the campaign banner reads PROVISIONAL — NOT
FOR PUBLIC HEADLINE, this draft stays unpublished.

---

## The post (one, plain)

Quantized the drafter instead of the model and got 1.4x decode on one Intel Arc Pro B70.

Qwen3.8-27B GPTQ-INT4 target with a 4-token MTP drafter. Only the drafter's weights go to INT4: its 2.5 GB LM head and its body. The 27B target is untouched.

52.6 to 73.4 tok/s at p512/g128 and 48.8 to 72.6 at an 8192-token prompt. n=5 each, medians, client post-first rate. Same card, same 230 W cap, measured draw unchanged at about 229 W both arms. So roughly +40% speed and +32% tokens per joule at the short cell.

Output is byte-identical to the unpatched run under greedy decoding. Acceptance rate wanders run to run. Throughput does not.

Two honest caveats. XPU graph capture has to be on or the patches do nothing, and upstream vLLM has an open bug where graph capture with MTP4 can emit wrong logits at temperature 0. Validate determinism for your own case.

Recipe and raw numbers in the cookbook.

---

## Why this wording (claims hygiene)

- No naked number: every figure carries its cell (p512/g128 and 8192-token/g128), n, and the metric class (client post-first rate, stated as such).
- Configured cap (230 W) and measured draw (~229 W) are separated explicitly; tokens/J is derived from the measured energy counter, and the window caveat (includes 5-request prefill) lives in the full write-up.
- Comparison class is matched A/B with a disclosed tokenizer delta, so "1.4x" is licensed; the post says "1.4x decode on one Intel Arc Pro B70" not "fastest anything".
- Correctness is scoped: "byte-identical under greedy decoding" — no distribution-preservation claim, no unqualified "correctness verified".
- The acceptance-rate variance is stated flat rather than hidden.
- Both disclosures travel ON the post, not behind a link: the graph-capture enabler and upstream issue #54785.
- C1 single-stream only. The concurrency numbers (2.79x aggregate at B=32) are deliberately NOT here because aggregate never shares a line with single-stream.

## Not-yet-authorized actions

Posting to X/Twitter. Also not implied: IntelInside, XeCores, leaderboard submissions of any
other kind. The LocalMaxxing submission is separately authorized by the operator's
"PUBLISH NEW LOCALMAXXING EVAL" instruction and has its own payload set at
`scripts/tmp/lmx-submit-20260922/` plus its own claims audit
(`worklogs/CLAIMS-AUDIT-LMX-20260922.md`).
