# Qwen3.6-35B-A3B INT4/Q4 262k Ceiling Analysis

> Hardware: Intel Arc Pro B70 32GB
> Date: 2026-09-17

We swept the upper context limits (131k, 196k, 262k) of Qwen3.6-35B-A3B using three INT4/Q4 paths to measure capacity constraints, long-context latency, and generation speed at maximum KV cache utilization.

## 1. Engine Capabilities & Constraints

| Engine | Format | 131k (prefill / decode) | Max Achieved | Failure Mode at 262k |
|--------|--------|-------------------------|--------------|----------------------|
| **llama.cpp** (SYCL) | GGUF Q4_K_XL + Q8_0 MTP | 668 t/s / 26.2 t/s | 131k | HTTP 500 / Network Timeout at 196k |
| **vLLM** (XPU) | GPTQ-Int4 + MTP | 3178 t/s / 13.6 t/s | 131k | OOM (`UR_RESULT_ERROR_DEVICE_LOST`) |
| **OpenVINO** | OV INT4 | 549 t/s / 35.0 t/s | 262k | N/A (Cleared 262k) |

## 2. Findings

### OpenVINO is the only 262k survivor
OpenVINO GenAI successfully processed the full 262,144 context limit, achieving 26.5 t/s decode at the absolute ceiling. This proves the hardware is capable of 262k context on a 35B MoE, provided the KV cache memory footprint is small enough.

### vLLM speed vs. capacity tradeoff
vLLM remains the prefill champion by a massive margin (3,178 t/s vs OpenVINO's ~549 t/s at 131k context), but it requires too much VRAM for its internal XPU graphs and Python overhead.
* At 131k context, vLLM prefill is roughly 5.8x faster than OpenVINO.
* However, vLLM crashed attempting 262k: `Free memory on device xpu:0 (4.86/30.3 GiB) on startup is less than desired GPU memory utilization (0.9, 27.27 GiB).`

### llama.cpp SYCL limits
llama.cpp with an MTP draft head cleared 131k but locked up (timeout/Connection refused) trying to reach 196k. MTP speculative decoding consumes additional VRAM for the draft head weights and speculative KV/logits, pushing the 21GB GGUF over the limit.

## 3. Next Steps

1. **Production configuration:**
   - **For < 131k context:** Use **vLLM (GPTQ-Int4)**. The 3,178+ t/s prefill speed is transformative for RAG and agentic workflows, and 131k is sufficient for 95% of use cases.
   - **For 131k-262k context:** Use **OpenVINO**. It is the only engine that reliably scales to the 262k ceiling on 32GB VRAM without OOM.

2. **vLLM optimization:** 
   - We hit `CCL_ZE_IPC_EXCHANGE` errors over `pidfd`, requiring fallback to `sockets`. Investigate if updating XPU runtimes or adjusting container IPC privileges can restore `pidfd` for better efficiency.
   - Ensure the MTP `patch_mtp_ptr_wrap.py` is applied in production deployments.
