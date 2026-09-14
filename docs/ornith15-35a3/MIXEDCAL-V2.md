# MixedCal-v2 conversion (Ornith-1.5-35B-A3B)

Why this GPTQ exists. Speed numbers live on
[CLAIMS.md](CLAIMS.md). Serve from [ORNITH-VLLM-XPU.md](ORNITH-VLLM-XPU.md).

**Self-reported E2.** Isolated C1 confirmed speed **parity** at 150 W.
MixedCal-v2 is not a tok/s win.

## What changed

There is no official Ornith GPTQ. MixedCal-v2 is a local GPTQ INT4
symmetric G128 (`desc_act=false`) with MTP left in BF16. Format and tensor
scope match the WikiText-calibrated control so the only experimental axis is
**calibration coverage**.

| Axis | WikiText GPTQ (control) | MixedCal-v2 |
|---|---|---|
| Format | GPTQ INT4, G128, `sym=true`, `desc_act=false` | same |
| Quantized tensors | routed-expert `gate_proj` / `up_proj` / `down_proj` only | same |
| Expected expert `qweight` | 40 × 256 × 3 = **30,720** | same |
| MTP | 785 tensors, **0** `qweight` | same |
| Calibration | WikiText | mixed-domain, 128 × 1,536 tokens |
| Public id | not published | [`SergiioB/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2`](https://huggingface.co/SergiioB/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2) |

The official BF16 source is `ornith-ai/Ornith-1.5-35B-A3B`, revision
`fbb995a79eedd569a5edc5f2af9644c0fa1124fc`. MixedCal-v2 does not overwrite it.

## Experts-only contract

Quantize **only** routed experts. Do not quantize:

- `lm_head`
- embeddings (`model.language_model.embed_tokens`)
- attention / GDN
- router (`mlp.gate`)
- shared experts
- MTP draft tensors
- vision

GPTQModel 7.3.2 dynamic exclusions:

```python
{
    "-:.*attn.*": {},
    "-:.*mlp\\.gate$": {},
    "-:.*mtp.*": {},
    "-:.*shared_expert.*": {},
    "-:.*visual.*": {},
    "lm_head": {},
    "model.language_model.embed_tokens": {},
}
```

Published artifact: **30,720** expert `qweight`, **785** MTP tensors with
**0** quantized, **0** forbidden `qweight`. Size **24,454,916,052** bytes,
6 shards.

## Calibration (the actual change)

| Input | Value |
|---|---|
| Samples | 128 |
| Tokens per sample | 1,536 |
| Total tokens | 196,608 |
| Seed | `15035260819` |
| Corpus SHA-256 | `e88ccd5f999a05a3c78bb9ec86a82256255a9ad97068c787dfc564907a377381` |
| Quantizer | GPTQModel 7.3.2 |
| Image | `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f` |

The corpus spans code, reasoning, agentic/tool content, SQL, networking,
technical prose, general text, and systems debugging. Routing is **natural**:
the process does not force router outputs or expert IDs.

WikiText under-covers late-layer experts on this 256-expert MoE. GPTQ then
cannot form a usable Hessian for those matrices and falls back to
round-to-nearest (RTN). MixedCal is a broader natural-routing set so more
expert matrices get a real GPTQ solution.

## What RTN fallback means

RTN here is **GPTQModel's weight-only fallback**, not a CPU/XPU kernel
fallback and not a serving-mode switch. A matrix that hits RTN is still
INT4 G128; it just was not GPTQ-solved from calibration.

Lower RTN % is the conversion win. It is **not** a decode or prefill
throughput claim.

## Observed percentages

| Artifact | RTN fallbacks | Rate |
|---|---:|---:|
| WikiText GPTQ | 7,605 / 30,720 | **24.76%** |
| MixedCal-v2 | 3,186 / 30,720 | **10.37%** |

Relative to WikiText, MixedCal-v2 removed **58% of RTN fallbacks**
(`1 − 10.37 / 24.76`).

Per-projection MixedCal split is even: **1,062 / 1,062 / 1,062**
(`gate` / `up` / `down`).

Late-layer hotspot (layer 39, 768 expert matrices):

| Artifact | Layer 39 RTN | Rate |
|---|---:|---:|
| WikiText | 297 / 768 | **38.7%** |
| MixedCal-v2 | 75 / 768 | **9.8%** |

Worst MixedCal layers after the change: 34 (201), 37 (174), 32/36 (156).
The conversion did not zero RTN; it cut the WikiText tail, especially late
experts.

## What it does **not** improve

n=5 no-spec confirmation at 32K, 150 W, cache off, instance-median of three
loads:

| Cell | WikiText GPTQ | MixedCal-v2 |
|---|---:|---:|
| p512/g128 client post-first | 70.80 | 70.74 |
| p8192/g128 client post-first | 64.86 | 64.95 |
| p2048/g1 cold input | 6935 | 6968 |
| p8192/g1 cold input | 6926 | 6963 |

Do **not** call MixedCal-v2 faster. That table is speed **parity**.

Prefill **~9.7k** is a **configured 230 W** cell on the same WNA16 nightly,
not a MixedCal-vs-WikiText effect.

There is still no BF16 logit/KL or held-out task-quality suite. RTN coverage
is a conversion metric, not a substitute for KL.

## Download check

```bash
huggingface-cli download SergiioB/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2 \
  --local-dir "$HOME/models/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2"
```

Verify after download:

- 6 shards, 24,454,916,052 bytes
- 30,720 routed-expert `qweight`
- 785 `mtp.*` tensors, **zero** MTP `qweight`
- `quantize_config.json`: 4-bit, `sym=true`, `group_size=128`, `desc_act=false`
