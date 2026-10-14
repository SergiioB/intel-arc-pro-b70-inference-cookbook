# PLAN v2: SC5 int8 prefill vs fp16 prefill on chat, two passes per server (2026-10-14)

Written after v1. v1 is **invalid under its own rule**: the predeclared determinism bound was a KL
of at most 1e-6 between repeated passes, and repeated passes measured 5e-4 to 5e-3. That
predeclared bound is not lowered for v1. v1's numbers stay as an unsealed diagnostic.

## What v1 showed about the noise (read before the effect)

- Repeated passes on one server are bit-identical up to a token index that is a multiple of 16
  (160 to 800), then drift. With fp16 prefill the drift is tiny in the first 1024-token chunk (mean
  KL 1.5e-5 to 2.4e-5). It is much larger in the second chunk, which has a cached prefix (1e-3 to 2e-3).
- Most of the second-chunk drift comes from a few hard tokens. In one wikitext row, around positions
  1676 to 1805, the observed-token NLL is 7 to 11 nats in every arm.
- int8 prefill drift is 30 to 50 times the fp16 drift in the first chunk (7.9e-4 to 8.7e-4). int8
  prefill therefore adds run-to-run noise. That noise is part of int8's error against fp16.
- On v1's chat panel the coarse KL is accurate (tail mass 0.0003). SC5 fp16 to int8 measured
  0.00109 [0.00096, 0.00122], against SC5's whole weight-quantization KL to BF16 of 0.00141 (RL-179).
  No fp16 noise floor was measured on chat, which is the gap v2 closes.

## Design

- **Arms:** sc5_fp16 and sc5_int8 only. Same image, flags, limits and safety as v1.
- **Panel:** `panels_v2.json` (sha256 `215b40bd...a5fb`), built by `prep_panels_v2.py`. It holds all 18
  held-out qbench rows with at least 1024 response tokens: 44,473 scored positions, top-128.
- **Passes:** two full passes per server load (rep1, rep2). Raw output goes to `raw_v2/`.

## Decision rule (predeclared, unchanged thresholds from v1)

- **Effect:** E is the mean of the rep1 and rep2 coarse-KL lower bounds from fp16 to int8 on P2.
- **K:** E divided by 0.00141.
- **Validity:** the fp16 noise floor (rep1 to rep2 KL) must be at most 0.1 × E. Otherwise the result is
  inconclusive.
- **Material** if the lower CI of K is at least 0.25. Then:
  - recommend `EXL3_INT8_PREFILL=0` for the SC5 quality tier;
  - price its TTFT cost separately before any launcher change.
- **Negligible** if K < 0.10.
- **Otherwise minor.**

Scoring script: `analyze_v2.py` (written before the run).
