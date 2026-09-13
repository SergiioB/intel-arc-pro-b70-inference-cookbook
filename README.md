# Intel Arc Pro B70 Inference Cookbook

Repeatable vLLM XPU and llama.cpp SYCL recipes for Intel Arc Pro B60/B70 GPUs.

## Agent control plane

Agents should begin with [`data/system-map.v1.json`](data/system-map.v1.json),
choose the closest task intent, and load only that route's authorities. Routes
cover reading, setup, reproduction, client connection, operation, submission,
publication, and triage. The design and change-routing model is
[`docs/AGENT-SYSTEM.md`](docs/AGENT-SYSTEM.md). Add or change a family route
through [`docs/ADDING-A-RECIPE.md`](docs/ADDING-A-RECIPE.md).

The repository is a five-layer system: control -> contracts -> family recipes
-> evidence -> generated reader views. Edit the owning authority, then render
and validate downstream views. Do not maintain copied benchmark or patch tables.

Run all GPU-free repository gates with `python3 scripts/check-repo.py`.

This is **one cookbook with one page per model family**. Do not start a second
repo when a new architecture lands: add `docs/<family>/` + `benchmarks/<family>/`
and pin that family's image digest. **Do not mix patch lists or numbers across
families.** The image that serves Qwen3.6 Pi is not the image that served
Nemotron DFlash.

## Model family hub

| Family | Engine | What is proven | Headline | Page |
|---|---|---|---|---|
| **Qwen3.6-35B-A3B** | vLLM XPU (Pi digest) | Native MTP 1/2/4, 128K | MTP4 p512/g128 **170.91** client post-first n=5 | [QWEN36-MOE-VLLM-XPU](docs/qwen36-35a3/QWEN36-MOE-VLLM-XPU.md) |
| **Qwen3.8-27B** | vLLM XPU (nightly digest) | Dense GPTQ-INT4 + MTP4; optional draft-INT4; concurrent serving via mixed-split v5; separate dual-B70 FP8 TP2 research route | C1 **106.7** n=5 current GPTQ stack (LMX `cmt03mj040eh8ms01trjvhm75`); cache-off 112.65 (`cmszpqy000e8fms014ty6i5x3`), BF16-draft 83.7 (`cmsur82fz06svms01ga1f0z83`). Concurrent (v5 + draft-INT4, prefix on): **C5 realistic 127.4** Σ-streams / 25.5 per-user (`cmt03mjo60ehbms0117c5i745`), short-prompt **C5 203.8 / C32 224.2** (lmx harness), C32 Σ-streams 903. **Prefix reuse largely fails at C5 on this build** (0–38% hits vs 91% at C1) — warm-session TTFT at Cn is an open issue. FP8 TP2 values remain on their dedicated E2 page/catalog records. | [Family hub](docs/qwen38-27/README.md) · [GPTQ recipe](docs/qwen38-27/QWEN38-VLLM-XPU.md) · [FP8 TP2](docs/qwen38-27/FP8-TP2-W8A16.md) · [Windows 11](docs/qwen38-27/WINDOWS-STANDALONE.md) |
| **Qwen3.6-27B** | vLLM XPU (same Pi digest) | Dense GPTQ-INT4 + MTP, fp8 KV | MTP4 p512/g128 **69.30** n=5 | [QWEN36-DENSE-VLLM-XPU](docs/qwen36-27/QWEN36-DENSE-VLLM-XPU.md) |
| **Nemotron-3.5-Lightning-30B-A3B** | vLLM XPU (**newer** digest) | DFlash n=7; native MTP **0%** | **186.61** C1 client post-first at p2048/g128 n=5; **cold input 7160** (prompt/TTFT) at p8192/g1 | [NEMOTRON-DFLASH-B70](docs/nemotron35-30a3/NEMOTRON-DFLASH-B70.md) |
| **Muse-Glimmer-30B** | llama.cpp SYCL | Vision + DFlash n2; vLLM still experimental | **26.8** engine t/s at p512/g128 **128K** n=5 | [MUSE-GLIMMER-B70](docs/muse-glimmer/MUSE-GLIMMER-B70.md) |
| **Qwen3.8-Flash-Next** | llama.cpp SYCL (two B70s, C1) | MTP + fused multi-token `MUL_MAT_ID` kernel. 8K/16K/128K are context windows. FP32 and F16 are separate compile-time binaries. | C1 n=5: FP32 **33.25** tok/s p512/g128 at 8K (+42%); F16 cold input **585.9** at p9096/g128, 16K. | [Recipe](docs/qwen38-flash-next/QWEN38-FLASH-NEXT-LLAMACPP.md) |
| **Ornith-1.5-35B-A3B** | vLLM XPU (Qwen3.8 nightly digest) | Local GPTQ-INT4 MixedCal-v2, **MTP1 + DraftINT4 default**; 262K C1; 150↔230 W prefill A/B | Self-reported E2: combined 230 W LMX `tokSOut` **108.4** / `tokSPrefill` **9073** (`cmt2tdx5q0hy0mv01koh4xwpw`); host p512/g128 **106.64**. BF16-draft MTP1 150 W **96.43**. No-spec 230 W prefill **9780** (`cmt2sr6gq0himmv01ogieh0c8`) | [ORNITH-VLLM-XPU](docs/ornith15-35a3/ORNITH-VLLM-XPU.md) |
| **Ornith-1.5-35B-A3B** (converters) | vLLM XPU (nightly digest) | GPTQ→AutoRound converter head-to-head; MTP1; **reference logprob parity vs BF16** | MTP1 96.4 t/s n=5 @150 W; AutoRound equal-or-best parity (self-report, E2) | [AUTOROUND-VS-GPTQ](docs/ornith15-35a3/AUTOROUND-VS-GPTQ.md) |
| **Qwen3.8-27B** | OpenVINO GenAI & Cascadia | Native INT4 VLM, NNCF surgery, Paged Attention (`--cb`) | 15.8 tok/s OpenVINO 230 W (single-user limit) vs 4.95 Cascadia. Fast pipeline-parallel but bottlenecked by GDN. | [OpenVINO vs Cascadia](docs/qwen38-27/OPENVINO-CASCADIA-REPORT.md) · [Architecture Guide](docs/architecture/openvino-and-cascadia.md) |

Image + patch pin: [IMAGE-AND-PATCH-MATRIX.md](docs/IMAGE-AND-PATCH-MATRIX.md).

**Reliability map (what breaks and which layer owns it):** [RELIABILITY-REPORT.md](docs/RELIABILITY-REPORT.md) — single-card baseline, multi-GPU failure modes, Linux bring-up, ranked fix list. Evidence-linked as of 2026-08-31.

Issue/PR checks: [MAINTAINER-PRECHECKS.md](docs/MAINTAINER-PRECHECKS.md).

Every speed cell is C1 unless a table says otherwise. LocalMaxxing `APPROVED`
means the payload was accepted into the public leaderboard.

### One benchmark source for the cookbook and XeCores

Public summary records live in [`data/benchmarks.v1.json`](data/benchmarks.v1.json).
XeCores reads that file directly and keeps a vendored fallback. To add or change
a public result, edit the JSON once, then run:

```bash
python3 scripts/render-benchmark-catalog.py
python3 scripts/render-benchmark-catalog.py --check
```

The command regenerates [`docs/BENCHMARK-CATALOG.md`](docs/BENCHMARK-CATALOG.md).
Complete numeric rows require an exact workload, sample count, metric definition,
and commit-pinned evidence. Working recipes without those coordinates stay as
capability records and do not enter benchmark rankings.

## Engine Quick Selection

The B70 has 32 GB of VRAM and 608 GB/s bandwidth. The right engine depends on your model architecture and deployment goal.

**1. vLLM XPU (Docker + Python)**
- **Best for:** Absolute maximum speed, speculative decoding (MTP), OpenAI API serving, multi-user throughput.
- **Why:** XPU flash attention crushes prefill (often 5-10× faster than llama.cpp). MTP kernels provide massive decode boosts for Qwen.
- **Catch:** Dense FP16 models OOM on a single card. You must use GPTQ/AWQ INT4 formats or TP2 across two cards. Docker + `--privileged` required.
- **Guide:** [vLLM XPU Setup](docs/FULL-SETUP-COMMANDS.md)

**2. llama.cpp SYCL (Bare Metal C++)**
- **Best for:** Low VRAM footprint, GGUF compatibility, interactive local chat, running dense models that don't fit in vLLM.
- **Why:** Highly optimized `Q4_K_M` quants fit huge models (e.g., 35B) comfortably in 32 GB. Zero dependencies, compiles natively.
- **Catch:** Slower prefill. Layer-split multi-GPU does not scale decode speed.
- **Guide:** [llama.cpp SYCL vs vLLM XPU](docs/engine-comparison.md) (from our early benchmarks)

**3. OpenVINO GenAI & Cascadia (Bare Metal / Rust)**
- **Best for:** Edge deployments, NPU utilization (Core Ultra), pipeline-parallel across machines over the network.
- **Why:** JIT cache (`--ov-cache-dir`) drops cold-start to ~1 second. Cascadia provides a single Rust binary for serving.
- **Catch:** Model coverage lags behind vLLM/llama.cpp (e.g., Qwen3.8 GDN kernels missing/broken).
- **Guide:** [OpenVINO vs Cascadia Architecture](docs/architecture/openvino-and-cascadia.md)

## Quick Start (3-Step Setup)

### Step 1: Pull the image
```bash
export IMAGE='vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97'
docker pull "$IMAGE"
```

### Step 2: Download the model
**MoE (Qwen3.6-35B-A3B):**
```bash
export MODEL_DIR="$HOME/models/Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4"
huggingface-cli download llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4 \
  --local-dir "$MODEL_DIR"
```

**Dense (Qwen3.6-27B):**
```bash
export DENSE_DIR="$HOME/models/Qwen3.6-27B-MTP-Preserved-GPTQ-Int4"
huggingface-cli download llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4 \
  --local-dir "$DENSE_DIR"
```

### Step 3: Launch server & verify health
**MoE (128K context, MTP2, FP16 KV):**
```bash
bash benchmarks/qwen36-35a3/launch-vllm-128k-mode.sh "$MODEL_DIR" mtp2 on 8000
curl -f http://127.0.0.1:8000/health
```

**Dense (128K context, MTP4, FP8 KV required):**
```bash
bash benchmarks/qwen36-27/launch-dense27-128k-mode.sh "$DENSE_DIR" mtp4 on 8000
curl -f http://127.0.0.1:8000/health
```

Both launchers include tool-calling flags (`--enable-auto-tool-choice --tool-call-parser qwen3_coder`) out of the box for Pi, omp, and OpenAI agent clients.

**Serving with Pi / omp / agents:** point the client at `http://127.0.0.1:8000/v1`
and use the served model name (`Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4` or
`Qwen3.6-27B-MTP-Preserved-GPTQ-Int4`). Tool calling is enabled by the
launchers, so `tool_choice: "auto"` works out of the box. For a persistent
server, wrap either launcher in your own systemd unit or process supervisor —
the scripts are self-contained and portable (no host-specific paths).

### Windows 11 hosts (WSLC / Docker Desktop)

<img src="docs/assets/windows-11-logo.svg" alt="Windows 11" width="22" align="top"> Qwen3.8-27B also runs on **Windows 11** with the same image digest.
Two standalone PowerShell kits (Docker Desktop — proven, ~70 tok/s class on
the 2026.08.18 BF16-draft measure; Microsoft WSLC — experimental, 2.4–2.8×
slower) devised and tested by Ian Hudson (aitesthive.com). They reserve GPU
memory for the Windows desktop (`gpu-memory-utilization 0.75` + explicit
4.25 GiB fp8 KV) because a single-B70 Windows machine drives its display
from the same 32 GB card. Image **2026.08.19** adds draft-INT4 S+M1 and
turns **prefix cache on** for real sessions. If you already have the
18 August kit: `.\Upgrade-Qwen38-Docker.ps1` — do not re-download the
model. Guide:
**[docs/qwen38-27/WINDOWS-STANDALONE.md](docs/qwen38-27/WINDOWS-STANDALONE.md)** (kits in [`windows/`](windows/)).

### Connecting Pi / omp / Hermes

See **[CONNECTING-CLIENTS.md](docs/CONNECTING-CLIENTS.md)** for the full
client quick start: Hermes `config.yaml` provider block, omp base URL,
Pi client settings, the port table (8000 launcher / 8765 bridge), the `active`
model alias, API key setup, and a copy-paste tool-call smoke test.

**Exact software versions (do not substitute):**

| Component | Exact tested value |
|---|---|
| Image | `vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97` |
| vLLM | `0.26.1rc1.dev457+gc810e5ee9.xpu` |
| `vllm-xpu-kernels` | `0.1.12` |
| Tool-call parser | `qwen3_coder` (`Qwen3EngineToolParser`) |

Use [Full setup commands](docs/FULL-SETUP-COMMANDS.md) for the render-device check, model download and verification, package check, patch hashes, endpoint checks, and full matrix.

Benchmark graphics are rendered from the canonical `summary.json` with the public renderer [`benchmarks/render-prefill-decode-svg.py`](benchmarks/render-prefill-decode-svg.py) (dashboard + method diagram).

## Serve reliably: Xe2 wedge watchdog (production)

Under sustained Level-Zero load the `xe` driver can reset a compute/copy engine
and wedge the userspace context permanently - the server stops responding until
the container is restarted (intel/compute-runtime#948, vllm-project/vllm#41663).
The watchdog detects the wedge (health poll + kernel engine-reset signatures)
and restarts the serving container for you:

```bash
sudo bash watchdog/install-watchdog.sh --container vllm-serve
```

(adjust `--container` to the name your launcher uses - the launch scripts read
it from `$CONTAINER`; for non-docker deploys pass
`--recovery-cmd "systemctl restart <your-unit>"`). Container launchers now set
`--restart unless-stopped`, so a clean engine exit is recovered automatically
and the watchdog covers the GPU-wedge case where that exit never happens. Full docs:
[watchdog/README.md](watchdog/README.md).

## Model Architecture Guides

Every model has its own dedicated recipe and benchmarks in `docs/<family>/`:

1. **[Qwen3.6-35B-A3B (MoE)](docs/qwen36-35a3/QWEN36-MOE-VLLM-XPU.md):** Native MTP 1/2/4, 128K context, 170.9 tok/s peak decode.
2. **[Qwen3.6-27B (Dense)](docs/qwen36-27/QWEN36-DENSE-VLLM-XPU.md):** Dense GPTQ-INT4 + MTP4, FP8 KV cache required, 69.3 tok/s decode.
3. **[Qwen3.8-27B family hub](docs/qwen38-27/README.md):** choose the single-B70 GPTQ/MTP route, the separate dual-B70 FP8 TP2 research route, Windows packaging, or Pi agent integration without mixing their patch lists or numeric authorities.
4. **[Nemotron-3.5-Lightning-30B-A3B](docs/nemotron35-30a3/NEMOTRON-DFLASH-B70.md):** DFlash $n=7$ speculative decoding, 186.6 tok/s decode, 7160 tok/s cold prefill.
5. **[Ornith-1.5-35B-A3B](docs/ornith15-35a3/ORNITH-VLLM-XPU.md):** MixedCal-v2 local GPTQ-INT4, default MTP1 + DraftINT4, 108.4 tok/s decode.
6. **[Muse-Glimmer-30B](docs/muse-glimmer/MUSE-GLIMMER-B70.md):** llama.cpp SYCL vision + reasoning, DFlash $n=2$, 26.8 tok/s decode.

## Reproduce the matrix

The runner does not change host power. `CONFIGURED_CAP_W` records the cap selected by the operator.

```bash
CONFIGURED_CAP_W=165 \
  bash benchmarks/b70-pi-prefill-decode-matrix.sh "$MODEL_DIR"
```

Dense 27B (same matrix contract, fp8 KV, 230 W, GPU util 0.88 for MTP4):

```bash
CONFIGURED_CAP_W=230 \
  bash benchmarks/qwen36-27/launch-dense27-128k-mode.sh "$DENSE_DIR" mtp4 on 8000
```

Evidence and format:

- [Machine-readable phase-separated result](results/prefill-decode-matrix-20260809-summary.json)
- [Dense 27B machine-readable result](results/qwen36-27/prefill-decode-matrix-20260809-dense27-summary.json)
- [Dense 27B dashboard SVG](docs/assets/b70-dense27-4mode-dashboard.svg)
- [Stable cross-model benchmark format](docs/BENCHMARK-FORMAT.md)
- [Current result plus prior Pi campaigns](docs/REAL-WORLD-PI-BENCHMARKS.md)
- [Image and patch compatibility](docs/IMAGE-AND-PATCH-MATRIX.md)
- [Dual-B70 multi-GPU serving (TP2 / PP2)](docs/DUAL-B70-TP2.md)
- [Connecting Pi / omp / Hermes clients](docs/CONNECTING-CLIENTS.md)
- [Historical campaign log](docs/CAMPAIGN-LOG.md)

## vLLM runtime decisions — what this stack uses (both MoE and dense)

The pinned image runs vLLM V1 (0.26.1rc1.dev457+gc810e5ee9.xpu) on a single-socket single-GPU host. Of the five
runtime decisions commonly discussed, here is exactly where this stack stands
(verified from the running server's own config log, 2026-08-10):

| Decision | This stack | Evidence |
|---|---|---|
| **NUMA binding** | **N/A — single socket.** `Socket(s): 1`, `NUMA node(s): 1`. There is no inter-socket link to cross; the "wrong socket" problem cannot occur on one NUMA node. vLLM's `--numa-memory-tracking` / node pinning is irrelevant here and would change nothing. | `lscpu` |
| **Chunked prefill** | **Already ON (V1 default).** Server config: `enable_chunked_prefill=True`. `--max-num-batched-tokens 8192` is the chunk cap; large prompts are sliced and decode interleaves between chunks. Scheduler-budget probes on the MoE (+17.6% at 16,384) and dense (flat) show the cap also shapes throughput — see the scheduler findings above. | server config log |
| **Recompute instead of swap** | **Already the V1 behavior.** vLLM V1 has no KV swap path — evicted/recomputed requests rebuild from the prompt (recompute) rather than moving KV to CPU. `swap_space` is a V0 concept; on this V1 build there is nothing to set to 0. The `vllm:prefix_cache_*` counters confirm hits are served from GPU KV, not CPU. | V1 source + metrics |
| **Skip memory profiling** | **Not used — and not worth it here.** We pass `--gpu-memory-utilization 0.88` (dense) / `0.85` (MoE); the memory-profile/warmup phase costs **0.40 s + 0.03 s** of a **139.77 s** engine init (compilation 106.37 s). `--kv-cache-memory` would skip ~0.4 s of a 140 s boot — 0.3%. Startup is dominated by Triton JIT + CUDA graph capture, not profiling. | server log |
| **Eager mode** | **Not used — correct for serving.** `enforce_eager=False`, `cudagraph_mode: FULL_AND_PIECEWISE` with capture sizes 1-256. Graph capture is the 106 s of the 140 s boot, and it is what makes steady-state decode fast (MTP4 69.3 t/s dense, 170.9 MoE). `--enforce-eager` would cut boot but trade away most decode throughput — only sensible for throwaway dev loops, not the production profile. | server config log |

**Tool calling (Pi / omp / OpenAI clients):** both model paths must run with
`--enable-auto-tool-choice --tool-call-parser qwen3_coder` (the
`Qwen3EngineToolParser` in this build). Without them, clients that send
`tool_choice: "auto"` (Pi, omp, most agents) get
`400: "auto" tool choice requires --enable-auto-tool-choice and
--tool-call-parser to be set`. The launcher profiles for both models include
these flags; the raw launcher scripts in `benchmarks/` include them for the
serve command.

**Bottom line:** of the five levers, this stack already uses chunked prefill
and V1 recompute (both defaults), does not need NUMA (single socket), and
correctly skips eager mode and `--kv-cache-memory` — the profiling saving is
0.3% of boot while the eager trade would cost most decode throughput. The
actionable runtime lever measured here was the scheduler budget (see MoE
scheduler findings) and prefix caching (see the resident-session section).

## Correctness limitation

Prompt hashes match across no-spec, MTP1, MTP2, and MTP4. Output parity does not. Depending on the longer-decode cell, only 0 to 4 of 5 repetitions matched exact output text across all four modes. The campaign shows speed and completed exact token shapes, not token, logit, KL, or task-quality parity. Do not use speed as correctness proof.

## Repository map

Architecture diagrams: [docs/architecture/architecture.md](docs/architecture/architecture.md) · Runtime env-flag registry: [docs/FEATURE-FLAGS.md](docs/FEATURE-FLAGS.md)

```text
benchmarks/
  qwen36-35a3/       MoE Qwen3.6-35B-A3B launchers and model-specific campaigns
  qwen36-27/         Dense Qwen3.6-27B launchers (launch-dense27-128k-mode.sh)
  nemotron35-30a3/   Nemotron DFlash + no-spec graph launchers
  ornith15-35a3/     Ornith-1.5 MixedCal-v2 MTP1 launcher
  <root>             shared: matrix runner, harness, monitor, prompt generation, compiler, renderers
windows/             Windows 11 standalone kits (WSLC + Docker Desktop) — see docs/qwen38-27/WINDOWS-STANDALONE.md
patches/             family-tagged patches — see IMAGE-AND-PATCH-MATRIX.md
docs/
  qwen36-35a3/       MoE-specific reference (QUANTIZATION-QUALITY.md)
  qwen36-27/         Dense-specific reference (DENSE-FP8-GAP.md)
  nemotron35-30a3/   Nemotron DFlash + no-spec recipes
  ornith15-35a3/     Ornith MixedCal-v2 recipe + measured tables
  muse-glimmer/      Muse llama.cpp recipe
  qwen38-flash-next/ Flash-Next llama.cpp dual-B70 C1 recipe (not Qwen3.8-27B)
  <root>             shared: setup, benchmark contract, methodology, compatibility, history
results/
  qwen36-35a3/       MoE machine-readable summaries and engine grids
  qwen36-27/         Dense summaries (dense27 model card, llama.cpp grids)
  <root>             shared cross-model summaries
research/            kernel and quantization investigations
submissions/         historical LocalMaxxing payloads
watchdog/            production reliability: Xe2 wedge detection + auto-recovery
```

Model-specific files live under the family directory; cross-model contracts
(benchmark format, setup, image/patch matrix) stay at the shared root. A new
architecture gets a new family folder, not a new cookbook repo.

Code is MIT licensed. Measurement reports and prose are CC BY 4.0. See [LICENSE](LICENSE).
