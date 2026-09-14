# Qwen3.6-35B-A3B vLLM XPU MoE Recipe

Native MTP 1/2/4 speculative decoding, 128K context on Intel Arc Pro B70.

![B70 phase-separated input and decode dashboard](../assets/b70-prefill-decode-dashboard.svg)

## Quick Start (3-Step Setup)

### Step 1: Pull the image
```bash
export IMAGE='vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97'
docker pull "$IMAGE"
```

### Step 2: Download the preserved-MTP model
```bash
export MODEL_DIR="$HOME/models/Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4"
huggingface-cli download llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4 \
  --local-dir "$MODEL_DIR"
```

### Step 3: Launch server & verify health
```bash
bash benchmarks/qwen36-35a3/launch-vllm-128k-mode.sh "$MODEL_DIR" mtp2 on 8000
curl -f http://127.0.0.1:8000/health
```

---

## 1. Stack & Provenance

| Component | Tested Value |
|---|---|
| Public image | `vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97` |
| vLLM in image | `0.26.1rc1.dev457+gc810e5ee9.xpu` |
| `vllm-xpu-kernels` | `0.1.12` |
| MoE Model | `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4` |
| Patches | `patches/patch_mtp_nightly.py`, then `patches/patch_mtp_boundary.py` |
| Context / Scheduler / Util | 131,072 / 8,192 / `gpu-memory-utilization=0.85` |

---

## 2. Whole Analysis & Benchmark Tables

**Scope for every table below:** one Intel Arc Pro B70, C1, median of `n=5` after one same-output same-shape warmup, prefix cache enabled, unique entropy-first cold prefixes, zero cache-hit delta, scheduler 8,192, context 131,072, configured cap 165 W, client monotonic SSE timing.

### Cold prefill proxy: actual input tokens / TTFT, tok/s

| Mode | p512 | p2048 | p4096 | p6144 | p8192 | Full p131071 |
|---|---:|---:|---:|---:|---:|---:|
| No spec | 5,156 | 6,674 | 7,197 | 7,451 | 7,576 | 3,144 |
| MTP1 | 4,840 | 7,377 | 6,999 | 7,189 | 7,264 | 2,679 |
| MTP2 | 4,843 | 7,341 | 7,002 | 7,140 | 7,229 | 2,683 |
| MTP4 | 4,532 | 7,401 | 6,868 | 7,057 | 7,197 | 2,678 |

This rate includes scheduling, uncached prompt processing, and first-token work. It is not isolated engine prefill and is not llama-bench `pp`.

### Decode at p512: client post-first tok/s

| Mode | g32 | g128 | g256 | g512 |
|---|---:|---:|---:|---:|
| No spec | 97.43 | 96.79 | 96.60 | 96.13 |
| MTP1 | 122.21 | 124.57 | 123.82 | 120.58 |
| MTP2 | 162.90 | 153.17 | 148.31 | 141.80 |
| MTP4 | 178.34 | 170.91 | 167.85 | 148.35 |

### Decode at p8192: client post-first tok/s

| Mode | g32 | g128 | g256 | g512 |
|---|---:|---:|---:|---:|
| No spec | 85.92 | 90.34 | 90.91 | 91.26 |
| MTP1 | 108.41 | 118.41 | 118.49 | 117.45 |
| MTP2 | 143.95 | 145.43 | 143.82 | 135.61 |
| MTP4 | 156.28 | 164.36 | 163.89 | 138.03 |

### Historical control: p9445/g128

| Mode | Client post-first median (tok/s) |
|---|---:|
| No spec | 89.68 |
| MTP1 | 116.85 |
| MTP2 | 142.02 |
| MTP4 | 160.42 |

### Full-context decode

| Mode | p130944/g128 (tok/s) | MTP accept | p130560/g512 (tok/s) | MTP accept |
|---|---:|---:|---:|---:|
| No spec | 57.35 | n/a | 57.14 | n/a |
| MTP1 | 84.88 | 89.22% | 82.74 | 85.32% |
| MTP2 | 101.64 | 85.81% | 94.01 | 76.45% |
| MTP4 | 93.53 | 66.91% | 93.83 | 59.81% |

---

## 3. Scheduler and context findings

`--max-num-batched-tokens` is a cap, not a target: at p4096 both 8,192 and 16,384 prefill in one chunk, yet the larger budget is measurably faster at the same 128K recipe, same seqs 64, same prompts (exact palmfuture checkpoint, 230 W, MTP4):

| Budget | p4096 prefill | g128 decode |
|---|---:|---:|
| 8,192 | 6,525 t/s | 133.1 t/s |
| 16,384 | 7,672 t/s | 149.1 t/s |
| Δ | **+17.6%** | **+12.0%** |

Prefill is essentially **flat across context** (p4096 input rate, batch 16,384, seqs 16): 8K ctx 7,727 t/s · 16K 6,622 · 32K 7,740 · 128K 7,672.

---

## 4. Mode Selection Guide

1. **Short C1 responses:** MTP4 is fastest in p512 and p8192 g32/g128 cells.
2. **Exact 128K, g128:** MTP2 is fastest at 101.64 client post-first tok/s.
3. **Exact 128K, g512:** MTP2 and MTP4 tie at 94.01 and 93.83 tok/s.
4. **Mixed long prefill + short requests:** Use no-spec on this stack.
