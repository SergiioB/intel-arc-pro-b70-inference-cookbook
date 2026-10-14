---
title: "Speculative Decoding on Intel Arc Pro B70 (SYCL)"
type: method
date: 2026-07-03
status: living
---

# Speculative Decoding on Intel Arc Pro B70 (SYCL)

**Validated**: 2026-07-02 | **Build**: SYCL #9853

> **Scope correction 2026-10-09.** The MoE branch of the decision tree below
> ("Do NOT use speculative decoding") applied to one 35B MoE GGUF without MTP
> layers on llama.cpp. It is not a rule for MoE: Nemotron 3.5 Lightning runs
> DFlash n7 (186.6 tok/s, ADR-0002) and Flash-Next runs Strata MTP `--spec 4`
> (ADR-0011). Use the route's own speculation stack
> (`docs/governance/standing-rules.md` §1.2) and the
> [format and lever guide](../reference/b70-format-and-lever-guide.md).

## Results Summary

| Strategy | Model | Speedup | Draft Accept | Verdict |
|----------|-------|---------|-------------|---------|
| MTP-4 | 27B Dense | **+45-50%** | 93.8% | Use always |
| ngram-mod-4 | 35B MoE | -1.3% | 75.0% | No benefit |
| ngram-mod-8 | 35B MoE | -1.1% | 87.5% | No benefit |

## MTP-4 on 27B Dense (Corrected Numbers)

| Power | BASE t/s | MTP-4 t/s | Gain | Efficiency |
|-------|----------|-----------|------|------------|
| 165W | 17.56 | 25.50 | +45.1% | 0.155 t/s/W |
| 180W | — | 27.22 | — | 0.151 t/s/W |
| 230W | 20.18 | 30.36 | +50.4% | 0.132 t/s/W |

Draft acceptance: 93.8%, mean accepted length 2.6-3.2 tokens.

## Ngram on 35B MoE (Negative Result)

Despite 87.5% draft acceptance with ngram-mod-8 (7/8 accepted, mean len 8.0),
zero speedup (-1.1%). This proves the compute-bound boundary:

The MoE at 3B active params is compute-saturated. Each accepted draft token
still requires expert routing + verification. Verification cost = generation cost.

## Compute vs Bandwidth Framework

```
Model              Bottleneck       Speculation?
27B Dense (19GB)   Bandwidth        YES (+45-50% with MTP-4)
35B MoE (3B act)   Compute          NO (0% despite 87% acceptance)
9B Dense (7GB)     Near-boundary    NO (already at 51 t/s ceiling)
```

## Configuration

### MTP-4 (27B dense — recommended)
```bash
llama-server -m Qwen3.6-27B-MTP-Q5_K_M.gguf \
    -ngl 99 -fa on -ctk q5_0 -ctv q4_1 -t 8 --no-mmap -c 131072 \
    --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.75
```

### Ngram (NOT recommended on MoE)
```bash
# Do NOT use on MoE — proven 0% gain despite 87% acceptance
```

## Decision Tree

```
Model is dense?
├── YES + bandwidth-bound (>10GB) → MTP-4 (+45-50%)
├── YES + small (<10GB) → Already saturated, no benefit
└── NO (MoE) → Do NOT use speculative decoding
```

## Stacking Analysis: MTP + Ngram Combined (Validated 2026-07-03)

### 27B Dense: Stacking is a Clean Negative

| Config | Power | Wall t/s | vs MTP-4 Alone |
|--------|-------|----------|----------------|
| MTP-4 alone | 180W | 27.22 | — |
| MTP-4 alone | 230W | 30.36 | — |
| MTP+ngram stack | 180W | 26.65 | **-2.1%** |
| MTP+ngram stack | 230W | 29.67 | **-2.3%** |
| ngram-4 alone | 180W | 18.71 | -31.2% |

**Verdict**: Do NOT stack. Ngram drafts compete with MTP for verification slots,
adding overhead without improving acceptance. Use MTP-4 alone.

### 35B MoE: MTP Architecturally Impossible + Ngram Zero Gain

The 35B MoE checkpoint has **no MTP layers** — context creation fails:
```
context type MTP requested but model does not contain MTP layers
```
Ngram-map-4: 56.22 t/s, 35.5% acceptance — no speedup (compute-bound, verified).

### Updated Decision Tree

```
Model is dense?
├── YES + bandwidth-bound (>10GB) → MTP-4 alone (+45-50%). Do NOT stack with ngram.
├── YES + small (<10GB) → Already saturated, no benefit
└── NO (MoE) → Do NOT use speculative decoding
    └── MTP impossible (no MTP layers). Ngram gives zero gain (compute-bound).
```
