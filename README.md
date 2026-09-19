# Intel Arc Pro B70 Inference Cookbook

Repeatable, production-ready vLLM XPU and llama.cpp SYCL recipes for Intel Arc Pro B60/B70 GPUs.

> 🚀 **New to the cookbook? Start with [`START.md`](START.md) for the 30-minute quick start guide.**  
> 🤖 **Automated agent or bot? See [`AGENTS.md`](AGENTS.md) and [`data/system-map.v1.json`](data/system-map.v1.json).**

---

## What This Cookbook Is (and Is NOT)

- **IS:** Verified, reproducible, containerized recipes and bare-metal builds for Intel Arc Pro B60/B70 (Battlemage BMG-G21).
- **IS:** Exact image digests, ordered patch sequences, cold/warm cache telemetry, power envelope benchmarks, and failure mode documentation.
- **IS NOT a CUDA translation layer:** This repo targets native Intel SYCL and Level Zero runtimes.
- **IS NOT a 4-card server cluster guide:** 4× GPU TP4 is not supported due to Xe driver peer-memcpy limitations. The cookbook focuses on 1× B70 (desktop/workstation) and 2× B70 (TP2).
- **IS NOT for consumer Arc A-series or B50:** Recipes rely on BMG-G21 XMX integer hardware and 32 GB VRAM per card.

---

## The 30-Second Chooser

| Your Setup | Goal | Recommended Recipe | Headline Decode Speed | Guide |
|---|---|---|---|---|
| **1× B70 (32 GB)** | Chat, agent tools, maximum single-card speed | **Qwen3.8-27B GPTQ-INT4** (vLLM XPU) | **106.7 tok/s** (C1, p512/g128, cache-zero @ 230W) | [Recipe](docs/qwen38-27b/QWEN38-VLLM-XPU.md) |
| **1× B70 (32 GB)** | GGUF quants, zero Docker, rock-solid stability | **Muse-Glimmer-30B** (llama.cpp SYCL) | **26.8 tok/s** (C1, 128K context) | [Recipe](docs/muse-glimmer/MUSE-GLIMMER-B70.md) |
| **1× B70 (32 GB)** | Raw MoE throughput record | **Qwen3.6-35B-A3B** (vLLM XPU, Pi digest) | **170.9 tok/s** (C1, MTP4) | [Recipe](docs/qwen36-35a3/QWEN36-MOE-VLLM-XPU.md) |
| **2× B70 (64 GB)** | Dense FP8 / large models across 2 cards | **Dual-B70 TP2** (vLLM XPU + oneCCL) | **~108 tok/s** (TP2 W8A16) | [Recipe](docs/DUAL-B70-TP2.md) |
| **Windows 11** | Desktop display on single B70 | **Qwen3.8-27B** (Docker Desktop kit) | **~70–84 tok/s** (util 0.75 display headroom) | [Windows Guide](docs/qwen38-27b/WINDOWS-STANDALONE.md) |

---

## Model Family Hub

The cookbook enforces strict family isolation: **never mix patch lists or image digests across families**.

### Tier 1: Supported & Production-Tested Recipes
Maintained, canary-verified, and recommended for production use.

| Family | Engine & Stack | What Is Proven | Headline Metric & Workload | Status | Route Link |
|---|---|---|---|---|---|
| **Qwen3.8-27B** | vLLM XPU (nightly digest) | Dense GPTQ-INT4 + MTP4; optional draft-INT4; 100K–128K ctx; tool calling | C1 **106.7 tok/s** decode (p512/g128, cache-zero, 230W) · C5 realistic **127.4 tok/s** Σ-streams | Official lab | [Recipe](docs/qwen38-27b/QWEN38-VLLM-XPU.md) · [Hub](docs/qwen38-27b/README.md) |
| **Qwen3.6-35B-A3B** | vLLM XPU (Pi digest) | Native MTP 1/2/4; 128K context; FP16 KV cache | MTP4 p512/g128 **170.91 tok/s** client post-first n=5 (cache-zero) | Official lab | [Recipe](docs/qwen36-35a3/QWEN36-MOE-VLLM-XPU.md) |
| **Muse-Glimmer-30B** | llama.cpp SYCL | Vision + DFlash n=2; 128K context; zero Docker dependencies | **26.8 tok/s** engine decode at p512/g128 (128K ctx, n=5) | Official lab | [Recipe](docs/muse-glimmer/MUSE-GLIMMER-B70.md) |
| **Qwen3.6-27B** | vLLM XPU (same Pi digest) | Dense GPTQ-INT4 + MTP4; FP8 KV cache required | MTP4 p512/g128 **69.30 tok/s** n=5 (cache-zero) | Official lab | [Recipe](docs/qwen36-27/QWEN36-DENSE-VLLM-XPU.md) |

### Tier 2: Experimental & Research Routes
Active research, custom kernel overlays, or comparative benchmarks.

| Family | Engine & Stack | Experimental Focus | Headline / Observations | Route Link |
|---|---|---|---|---|
| **Nemotron-3.5-30B-A3B** | vLLM XPU (newer digest) | DFlash n=7 speculative decoding; native MTP 0% | **186.61 tok/s** C1 decode (p2048/g128) · **7160 tok/s** cold prefill | [Recipe](docs/nemotron35-30a3/NEMOTRON-DFLASH-B70.md) |
| **Ornith-1.5-35B-A3B** | vLLM XPU (nightly digest) | Local MixedCal-v2 GPTQ-INT4, MTP1 + DraftINT4 default | **108.4 tok/s** decode (230W) · **9073 tok/s** cold prefill | [Recipe](docs/ornith15-35a3/ORNITH-VLLM-XPU.md) · [Converters](docs/ornith15-35a3/AUTOROUND-VS-GPTQ.md) |
| **Qwen3.8-Flash-Next** | llama.cpp SYCL (dual B70) | MTP + fused multi-token `MUL_MAT_ID` kernel | C1 n=5: FP32 **33.25 tok/s** decode (8K ctx) · F16 cold prefill **585.9 tok/s** | [Recipe](docs/qwen38-flash-next/QWEN38-FLASH-NEXT-LLAMACPP.md) |
| **Qwen3.8-27B FP8 TP2** | vLLM XPU (dual B70) | Source-built `vllm-xpu-kernels 0.1.12.3` W8A16 | C1 **60.13 tok/s** decode · prefill **1736 tok/s** (research baseline) | [Recipe](docs/qwen38-27b/FP8-TP2-W8A16.md) |
| **OpenVINO & Cascadia** | Bare metal / Rust | Native INT4 VLM, NNCF surgery, Paged Attention (`--cb`) | 15.8 tok/s OpenVINO 230W vs 4.95 tok/s Cascadia | [Report](docs/qwen38-27b/OPENVINO-CASCADIA-REPORT.md) · [Architecture](docs/architecture/openvino-and-cascadia.md) |

---

## Engine Selection Matrix

| Engine | Best For | Why Choose It | Trade-off / Limitation |
|---|---|---|---|
| **vLLM XPU** (Docker) | Maximum decode speed, speculative MTP, OpenAI API | Flash attention crushes prefill; MTP decode kernels deliver 100–170 tok/s | Docker `--privileged` required; FP16 models OOM (must use INT4 or TP2) |
| **llama.cpp SYCL** (C++) | GGUF models, low VRAM footprint, zero Docker | Native C++ compilation; fits large models in 32 GB; zero container overhead | Slower prefill than vLLM; layer-split multi-GPU does not scale decode speed |
| **OpenVINO GenAI** | Edge deployments, Intel Core Ultra NPU integration | JIT cache (`--ov-cache-dir`) drops cold-start to ~1s; single Rust binary | Model coverage lags behind vLLM/llama.cpp; linear-attention GDN kernels missing |

---

## Quick Start (3-Step Setup: Qwen3.8-27B)

### 1. Pull the pinned container
```bash
export IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
docker pull "$IMAGE"
```

### 2. Download the model weights
```bash
export MODEL_DIR="$HOME/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
huggingface-cli download SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 \
  --revision 9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e \
  --local-dir "$MODEL_DIR"
```

### 3. Launch server & verify health
```bash
RENDER_GID="$(stat -c '%g' /dev/dri/render* | sort -u | sed -n '1p')"
docker rm -f b70-qwen38 >/dev/null 2>&1 || true

docker run -d --name b70-qwen38 -p 8000:8000 \
  --device /dev/dri --group-add "$RENDER_GID" \
  -v /dev/dri:/dev/dri:ro -v "$MODEL_DIR:/model:ro" \
  -v "$PWD/patches/patch_mtp_nightly.py:/patch_mtp.py:ro" \
  -v "$PWD/patches/patch_mtp_boundary.py:/patch_boundary.py:ro" \
  -e VLLM_TARGET_DEVICE=xpu -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e B70_MTP_BF16_DRAFT=1 -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash "$IMAGE" -lc \
  'set -e; python /patch_mtp.py; python /patch_boundary.py; exec vllm serve /model --quantization gptq --dtype float16 --max-model-len 100000 --gpu-memory-utilization 0.88 --kv-cache-dtype fp8 --port 8000 --max-num-seqs 1 --max-num-batched-tokens 8192 --no-enable-prefix-caching --served-model-name qwen38 --language-model-only --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":4}"'

curl -f http://127.0.0.1:8000/health
```

Run golden canaries to verify correctness:
```bash
python3 scripts/canary.py --port 8000 --model qwen38
```

---

## Production Reliability & Failure Cards

Under sustained load, Intel Xe driver ring resets can wedge the userspace context.

| Symptom | Root Cause | Fix |
|---|---|---|
| `curl /health` hangs indefinitely | **Xe driver ring wedge** | Driver wedged. Inspect with `dmesg \| grep -iE 'xe\|wedged'`. Recover with `python3 scripts/doctor.py --recover` or reboot. See [RELIABILITY-REPORT.md](docs/RELIABILITY-REPORT.md). |
| `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` | **VRAM KV overflow** | Reduce `--max-model-len` (e.g. 100000 -> 65536) or lower `--gpu-memory-utilization` to 0.85. |
| `zeMemOpenIpcHandle` on dual-card TP2 | **Missing oneCCL thresholds** | Motherboard lacks P2P direct access. Export the 4 oneCCL threshold variables in [DUAL-B70-TP2.md](docs/DUAL-B70-TP2.md). |
| Output corruption or repeating loops | **APC + MTP conflict** | Prefix caching combined with MTP causes silent token corruption on this kernel build. Keep `--no-enable-prefix-caching`. |

Install the automated watchdog to automatically detect wedges and restart the container:
```bash
sudo bash watchdog/install-watchdog.sh --container b70-qwen38
```
Full documentation: [watchdog/README.md](watchdog/README.md).

---

## Connecting Clients & Power Policy

- **Client Setup:** See [CONNECTING-CLIENTS.md](docs/CONNECTING-CLIENTS.md) for Pi, Hermes, Open WebUI, and API key configurations.
- **Power Optimization:** See [POWER-SWEET-SPOTS.md](docs/POWER-SWEET-SPOTS.md) for 150 W eco vs 230 W stock-cap trade-offs.
- **Image & Patch Pins:** See [IMAGE-AND-PATCH-MATRIX.md](docs/IMAGE-AND-PATCH-MATRIX.md) for patch orders and denylists.
- **Reproducing & Confirming:** See [CONFIRM-A-RECIPE.md](docs/CONFIRM-A-RECIPE.md) to submit community verification reports.

---

## Canonical Public Benchmark Catalog

The authoritative numeric catalog lives in [`data/benchmarks.v1.json`](data/benchmarks.v1.json).
The human-readable catalog [`docs/BENCHMARK-CATALOG.md`](docs/BENCHMARK-CATALOG.md) is generated from it:

```bash
python3 scripts/render-benchmark-catalog.py
python3 scripts/render-benchmark-catalog.py --check
```

---

## Agent Control Plane (L0–L4 Architecture)

> **For autonomous agents and automated maintainers:**  
> Start with [`data/system-map.v1.json`](data/system-map.v1.json) and [`AGENTS.md`](AGENTS.md).

The repository follows a five-layer authority model:
- **L0 (Control):** `AGENTS.md`, `data/system-map.v1.json`
- **L1 (Contracts):** `docs/BENCHMARK-FORMAT.md`, `docs/IMAGE-AND-PATCH-MATRIX.md`, `docs/RELIABILITY-REPORT.md`
- **L2 (Family Recipes):** `docs/<family>/`, `benchmarks/<family>/`
- **L3 (Evidence):** `results/`, `data/benchmarks.v1.json`
- **L4 (Views):** `README.md`, `START.md`, `docs/BENCHMARK-CATALOG.md`, `docs/architecture/architecture.md`

Run all repository integrity gates:
```bash
python3 scripts/check-repo.py
```

---

## License

Code is MIT licensed. Measurement reports and prose are CC BY 4.0. See [LICENSE](LICENSE).
