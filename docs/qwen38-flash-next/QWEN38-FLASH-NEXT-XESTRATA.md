# Qwen3.8-Flash-Next on two Arc Pro B70s: XeStrata (2,026 tok/s prefill, 55–68 tok/s decode, 262K context)

[Family index](README.md) · [dual-B70 topology and oneCCL setup](../DUAL-B70-TP2.md) · [image/patch matrix](../IMAGE-AND-PATCH-MATRIX.md)

This route serves `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (IQ3_XXS) across two Intel Arc Pro B70 cards using **XeStrata 0.1.45 SYCL**, K=24 layer pipelining, and native MTP 4 speculation.

Evidence tier: E1 (measured locally and verified independently on IntelInside.ai [Result #240](https://www.intelinside.ai/results/240)).

## 1. Measured Performance Breakdown

All numbers measured directly on an AMD Ryzen 7 5700X3D workstation with 2× Intel Arc Pro B70 (180 W cap each), oneAPI 2026.1, compute-runtime 26.05.

### Prefill Scaling

| Prompt Length | Prefill Rate | Total Prefill Time | Hardware Execution |
|---:|---:|---:|---|
| **217 tokens** | 307.5 tok/s | 705.6 ms | XMX GEMM, single batch |
| **772 tokens** | 431.1 tok/s | 1,790.8 ms | XMX GEMM, single batch |
| **5,276 tokens** | 1,415.5 tok/s | 3,727.4 ms | XMX GEMM, helper GPU assisted |
| **9,652 tokens** | 1,621.6 tok/s | 5,952.2 ms | 8K auto prompt chunk |
| **19,252 tokens** | 1,935.3 tok/s | 9,947.7 ms | 32K auto prompt chunk |
| **32,452 tokens** | **2,026.5 tok/s** | 16,013.9 ms | 32K auto prompt chunk |

### Decode Speeds by Workload Type

Autoregressive base speed without speculative drafting is **~25–28 tok/s**. Native MTP4 drafts up to 4 tokens per step:

| Task / Workload | Decode Throughput | Draft Acceptance | Notes |
|---|---:|---:|---|
| **Complex Python Code Generation** | **64.2 tok/s** | 65.1% (224 / 344) | Multi-threading, locking, benchmarks |
| **Long Context Prose / Reasoning** | **67.5 tok/s** | 75.9% (123 / 162) | 4.9K prompt history + reasoning |
| **Structured Numerical Sequences** | **93.3 tok/s** | 80.0% (344 / 430) | Roman numerals, structured lists |
| **Synthetic / Peak Repetitive** | **122.2 tok/s** | >90% | Theoretical ceiling reported in synthetic bench |

### Hardware and Memory Footprint

| Component | Usage / Setting |
|---|---|
| **Context Window** | **350,000 tokens** (YaRN 1.335× scaling over 262K native with `int8` KV cache) |
| **GPU 0 (0000:0b:00.0)** | 23.7 / 32 GiB (Layers 0–23 dense weights + 12,288 expert slots) |
| **GPU 1 (0000:0f:00.0)** | 29.9 / 32 GiB (Layers 24–47 dense weights + 12,288 expert slots + MTP draft head) |
| **Interconnect** | PCIe 4.0 x16 (14.4 GB/s measured bandwidth) |
| **Host System RAM** | 60.7 GiB available, 5.7 GiB resident before model load; mmap expert backing |

## 2. Architecture and Tuning Details

1. **Layer Split & Pipelining (`--layer-split 24 --pipeline-windows 2`)**:
   - The 48 layers are partitioned symmetrically: GPU 0 executes layers 0–23 and GPU 1 executes layers 24–47.
   - Pipelined verify windows overlap stage 0 (next token window drafting) with stage 1 (prior token verification), eliminating card stall time.
2. **Native MTP4 Speculative Decoding**:
   - Emits up to 4 speculative tokens per window, achieving 2.64 mean accepted tokens per step on complex tasks.
3. **Dynamic Context Chunking**:
   - Batched prefill automatically sizes chunks up to 32,768 tokens, borrowing cache slots dynamically for 2,026 tok/s prefill while keeping full 262K context resident.

4. **Dynamic Reasoning Effort Control (No Relaunch Needed)**:
   - Thinking is configurable dynamically per-request via `"reasoning_effort": "none" | "low" | "medium" | "high" | "xhigh"` or `{"chat_template_kwargs": {"enable_thinking": false}}`.
   - When `"none"`, thinking tokens are skipped completely and the answer is returned directly (~50 ms).
   - In B70 Launcher: specify `--effort none` (or `--effort high`) at launch to set the default, or select variants directly in OpenCode and OMP.
5. **Multimodal Vision (`mmproj`)**:
   - Supports image inputs via the model's 27-layer ViT projector (`mmproj-Qwen3.8-Flash-Next-BF16.gguf`, 866 MB).
   - In B70 Launcher: toggle with `--vision` (enabled) or `--no-vision` (text-only).
   - Compatible with both OpenAI vision format (`{"type": "image_url", ...}`) and Anthropic image blocks (`{"type": "image", ...}`).
6. **YaRN Context Extension (350K)**:
   - Extends the native 262,144 context by 1.335× to **350,000 tokens** using YaRN (`--rope-scaling yarn --rope-scale 1.33514 --yarn-orig-ctx 262144`).
   - Keeps perplexity degradation negligible while fitting cleanly inside the dual 32 GB VRAM footprint (23.7 GB on GPU 0, 29.9 GB on GPU 1).
## 3. Serving Commands

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
source /opt/intel/oneapi/setvars.sh --force
/data/sata1/engines/xestrata/.venv/bin/python /data/sata1/engines/xestrata/serve/server.py \
  --engine strata \
  --config ~/.config/xestrata/xestrata-iq3_xxs.json \
  --port 8000 \
  --host 127.0.0.1 \
  --gpu 0,1
```

Client endpoint exposes standard OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) APIs on port 8000.
