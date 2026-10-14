---
title: "KV Cache Quantization Reference"
type: reference
date: 2026-09-15
status: living
---

---
doc_id: DOC-REF-KV-CACHE
old_doc_id: DOC-LEGACY-KV-CACHE-QUANTIZATION
source_path: docs/reference/kv-cache-quantization.md
doc_type: canonical_reference
title: KV Cache Quantization Reference
lifecycle: stable
created: 2026-08-19
updated: 2026-08-19
last_verified: 2026-08-19
canonical_for:
  - kv-cache-quantization
---

# KV Cache Quantization Reference

The full retained analysis remains at [`docs/reference/kv-cache-quantization.md`](../reference/kv-cache-quantization.md) during migration. This page is the concise canonical current view.

## Current llama.cpp fleet default

```text
K cache: q8_0
V cache: q4_1
Flash attention: on
```

Key precision dominates attention selection; value precision is more tolerant because values are aggregated. The asymmetric policy protects K while compressing V.

## Validity boundaries

| Configuration | Current interpretation |
|---|---|
| q8_0 K + q4_1 V + FA on | Recommended fleet default; near-lossless measured class |
| q8_0 K + q8_0 V + FA on | Conservative, higher-VRAM option |
| q5_0/q5_1 K + q4_1 V + FA on | Acceptable measured trade-off when capacity requires it |
| q4_0 K | Prohibited; catastrophic quality cliff |

Do not confuse **model-weight Q4_0** with **KV-cache q4_0 K**. A model-specific Nemotron weight-format speed result does not relax the KV prohibition.

## vLLM distinction

vLLM KV dtype is an independent decision from checkpoint weight format. For example, dense Qwen3.8 uses GPTQ-INT4 target weights and fp8 KV to reach the 128K capacity target. This does not mean the model weights are FP8.

## Claim discipline

Every capacity or performance row must declare:

- engine and exact checkpoint;
- K/V or vLLM KV dtype;
- context coordinate;
- flash-attention state where applicable;
- post-load reserve;
- correctness scope.
