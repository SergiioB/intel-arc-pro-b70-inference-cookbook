# Qwen3.6-27B vLLM XPU Dense Recipe

Dense GPTQ-INT4 + MTP4 with FP8 KV cache, 128K context on Intel Arc Pro B70.

![B70 dense 27B 4-mode dashboard](../assets/b70-dense27-4mode-dashboard.svg)

## Quick Start (3-Step Setup)

### Step 1: Pull the image
```bash
export IMAGE='vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97'
docker pull "$IMAGE"
```

### Step 2: Download the model
```bash
export DENSE_DIR="$HOME/models/Qwen3.6-27B-MTP-Preserved-GPTQ-Int4"
huggingface-cli download llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4 \
  --local-dir "$DENSE_DIR"
```

### Step 3: Launch server & verify health
```bash
bash benchmarks/qwen36-27/launch-dense27-128k-mode.sh "$DENSE_DIR" mtp4 on 8000
curl -f http://127.0.0.1:8000/health
```

> [!IMPORTANT]
> **FP8 KV Cache is required:** Dense 27B attention needs ~9.5 GiB FP16 KV at 128K, which does not fit in 32 GB VRAM. FP8 KV cache halves it to ~4.75 GiB.

---

## 1. Stack & Provenance

| Component | Tested Value |
|---|---|
| Public image | `vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97` |
| vLLM in image | `0.26.1rc1.dev457+gc810e5ee9.xpu` |
| `vllm-xpu-kernels` | `0.1.12` |
| Dense Model | `llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4` |
| Patches | `patches/patch_mtp_nightly.py`, then `patches/patch_mtp_boundary.py` |
| Context / Scheduler / Util | 131,072 / 8,192 / `gpu-memory-utilization=0.88` (MTP4) |
| KV Cache Dtype | `fp8` (**required**) |

---

## 2. Whole Analysis & Benchmark Tables

**Scope for every table below:** one Intel Arc Pro B70, C1, median of `n=5` after one same-output same-shape warmup, prefix cache enabled, unique entropy-first cold prefixes, zero cache-hit delta, scheduler 8,192, context 131,072, `--kv-cache-dtype fp8`, configured cap 230 W, client monotonic SSE timing.

### Cold input rate: actual input tokens / TTFT, tok/s

| Mode | p2048 | p4096 | p6144 | p8192 |
|---|---:|---:|---:|---:|
| No spec | 1,781 | 1,813 | 1,782 | 1,742 |
| MTP1 | 1,816 | 1,776 | 1,747 | 1,713 |
| MTP2 | 1,812 | 1,767 | 1,744 | 1,711 |
| MTP4 | 1,755 | 1,693 | 1,683 | 1,654 |

### Decode at p512: client post-first tok/s

| Mode | g32 | g128 | g256 | g512 |
|---|---:|---:|---:|---:|
| No spec | 32.90 | 32.85 | 32.78 | 31.54 |
| MTP1 | 50.00 | 50.47 | 50.19 | 48.88 |
| MTP2 | 62.15 | 63.59 | 61.45 | 59.95 |
| MTP4 | 72.78 | 69.30 | 64.06 | 64.13 |

### Decode at p8192: client post-first tok/s

| Mode | g32 | g128 | g256 | g512 |
|---|---:|---:|---:|---:|
| No spec | 31.48 | 31.46 | 31.45 | 31.42 |
| MTP1 | 48.08 | 46.90 | 47.97 | 47.33 |
| MTP2 | 63.98 | 60.73 | 59.62 | 57.10 |
| MTP4 | 67.44 | 64.11 | 65.87 | 57.79 |

### Historical control: p9445/g128

| Mode | Client post-first median (tok/s) |
|---|---:|
| No spec | 31.35 |
| MTP1 | 48.41 |
| MTP2 | 60.12 |
| MTP4 | 67.25 |

### Full-context decode (exact 131,072 total tokens)

| Mode | p130944/g128 (tok/s) | MTP accept | p130560/g512 (tok/s) | MTP accept |
|---|---:|---:|---:|---:|
| No spec | 23.14 | n/a | 23.05 | n/a |
| MTP1 | 36.77 | 90.9% | 37.21 | 93.6% |
| MTP2 | 42.67 | 91.1% | 36.18 | 87.8% |
| MTP4 | 47.61 | 89.2% | 42.56 | 75.9% |

---

## 3. Real Pi Workload (Resident 32K Document Session)

| Step | Prompt tokens | Cache hits | Hit % | TTFT (s) | Post-first (tok/s) |
|---:|---:|---:|---:|---:|---:|
| 1. First read (cold) | 32,640 | 0 | 0% | 38.191 | 41.2 |
| 2. Follow-up 1 | 32,789 | 29,952 | 91.3% | 4.069 | 41.0 |
| 3. Follow-up 2 | 32,884 | 29,952 | 91.1% | 4.162 | 48.0 |
| 4. Follow-up 3 | 32,961 | 29,952 | 90.9% | 4.241 | 46.7 |
| 5. Follow-up 4 | 33,054 | 29,952 | 90.6% | 4.491 | 49.3 |

![Dense 27B resident 32K session](../assets/b70-dense27-resident-session.svg)

---

## 4. Key Constraints & Mode Selection

- **FP8 KV is mandatory** for 128K.
- **128K is the safe ceiling:** 200K loads but leaves <5 MiB free (abort zone).
- **MTP4 needs `gpu-memory-utilization=0.88`** (0.90 exhausts VRAM with spec buffers).
- **Power:** Dense prefill scales with power (+52% at 230 W vs 165 W).
- **Selection:** MTP4 is fastest in short C1 (69.3 t/s) and full-context 128K (47.6 t/s). Use no-spec for mixed long prefill + decode workloads.
