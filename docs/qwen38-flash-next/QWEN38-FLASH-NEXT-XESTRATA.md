# Qwen3.8-Flash-Next on two Arc Pro B70s: XeStrata (122.2 tok/s)

[Family index](README.md) · [dual-B70 topology and oneCCL setup](../DUAL-B70-TP2.md) · [image/patch matrix](../IMAGE-AND-PATCH-MATRIX.md)

This route serves the `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (IQ3_XXS) model on two Intel Arc Pro B70 cards with expert parallelism and native MTP 4 speculation using **XeStrata** (Strata 0.1.45 SYCL).

Evidence tier: E2 (community verified, run `cmvb7j24x01hzle01zgemx67l` on [LocalMaxxing](https://www.localmaxxing.com/en/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF?run=cmvb7j24x01hzle01zgemx67l)).

## 1. Performance Headline

| Metric | Measured Value | Notes |
|---|---:|---|
| **Decode Throughput** | **122.2 tok/s** | C1 greedy, reasoning on (512 max output tokens) |
| **Prefill Rate** | **469.2 tok/s** | 333 prompt tokens, TTFT 727.88 ms |
| **MTP Acceptance** | **54.36%** | MTP depth 4, mean accepted length 2.64 tokens/step |
| **Hardware** | **2× Arc Pro B70 (32 GB)** | Card 0 (primary) + Card 1 (expert parallel) |
| **Power Draw** | **180 W cap** | As-found desktop workstation profile |

## 2. Why it is faster than earlier dual-card routes

1. **True Expert Parallelism (`--expert-parallel-device 1`)**: Unlike the older Strata 0.1.40.3 helper cache (which staged misses through host memory at 33.4 tok/s), XeStrata 0.1.45 splits the routed experts across both B70 cards natively in VRAM.
2. **Native MTP4 Speculative Decoding**: Emits up to 4 speculative tokens per window, achieving an average accepted step length of 2.64 tokens.
3. **No PCIe Bottleneck**: Weights stay in pinned RAM and VRAM tiers, streaming without host stalls.

## 3. Stack and Artifacts

| Item | Value |
|---|---|
| Model | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (IQ3_XXS, 76 GB in 2 shards) |
| Engine | XeStrata 0.1.45-1 (`strata` SYCL binary on Level Zero) |
| MTP Layer | `mtp-rt` (BF16 dense + Q2_0 experts, ~800 MB) |
| Host Memory | ~60 GB RAM resident |
| VRAM | 32 GB × 2 (Card 0: 0000:0b:00.0, Card 1: 0000:0f:00.0) |

## 4. Serving Commands

### Via B70 Launcher (Recommended)

```bash
b70 launch qwen38-flashnext -e xestrata
```

Dry-run plan:
```bash
b70 launch qwen38-flashnext -e xestrata --dry-run
```

### Direct Server Invocation

```bash
/data/sata1/engines/xestrata/.venv/bin/python /data/sata1/engines/xestrata/serve/server.py \
  --engine strata \
  --config ~/.config/xestrata/xestrata-iq3_xxs.json \
  --port 8000 \
  --host 127.0.0.1 \
  --gpu 0,1
```

Client endpoint exposes standard OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) APIs on port 8000.
