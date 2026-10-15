# Nemotron 3.5 Bolt 74B-A9B on one B70 (llama.cpp SYCL)

NVIDIA's 24GB GGUF of Nemotron 3.5 Bolt runs on one Intel Arc Pro B70 with
no offload. The route is llama.cpp `620d2225c` (tag `b11669`) with one
cookbook patch, served from a prebuilt image. Measured on 2026-10-15:
28.2 tok/s decode at short context, 23.4 tok/s at 65K, and 1357.2 to 1430.7
tok/s prefill on 2K to 8K prompts. Evidence tier E2 (one host, one card,
not independently reproduced).

## Stack

| Item | Value |
|---|---|
| Model | [`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF), file `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf` (20.5 GiB) |
| Model SHA-256 | `91d63363e782c8a3d5f4be42816eba8e0d42dd2530bd07a7d0d2dc440599af75` |
| Quantization | IQ2_XXS routed and shared experts, NVFP4 Mamba projections, Q8_0 for the rest |
| Vision | `mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf` (BF16, 1.6 GiB), same repo |
| Architecture | 88 layers: 40 Mamba-2, 40 latent MoE (up to 496 experts per layer, top-9), 8 GQA attention (32 query / 2 KV heads). About 9B active parameters. |
| Engine | llama.cpp [`620d2225c`](https://github.com/ggml-org/llama.cpp/commit/620d2225cc436ab0b001e3200e590e3b286ef9ab) (tag `b11669`) + [`llama.cpp-620d2225c-b70-bolt.patch`](../../patches/llamacpp-sycl/llama.cpp-620d2225c-b70-bolt.patch) |
| Image | `ghcr.io/sergiiob/llama.cpp-sycl-b70:b11669-bolt` (upstream `.devops/intel.Dockerfile`, target `server`, `GGML_SYCL_F16=ON`) |
| Card | One B70 32 GB at its as-found 230 W cap. Server resident VRAM 23.9 GiB at the flags below (DRM fdinfo). |
| Required env | `GGML_SYCL_GROUPED_MAX_ROWS=0` |

Image and patch compatibility lives in
[IMAGE-AND-PATCH-MATRIX.md](../IMAGE-AND-PATCH-MATRIX.md). Do not apply the
Flash-Next llama.cpp patches or any vLLM patch on this pin.

## Run it

### 1. Download

```bash
export MODEL_DIR="$HOME/models/nemotron35-bolt-74b"
huggingface-cli download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF \
  NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf \
  mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf \
  --local-dir "$MODEL_DIR"
echo "91d63363e782c8a3d5f4be42816eba8e0d42dd2530bd07a7d0d2dc440599af75  $MODEL_DIR/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf" | sha256sum -c
```

### 2. Launch

```bash
bash benchmarks/nemotron35-bolt-74b/launch-nemotron-bolt-llamacpp.sh "$MODEL_DIR" 8080
curl -f http://127.0.0.1:8080/health
```

The launcher refuses to start while another llama.cpp or vLLM process runs.
`GPU=1` selects the second card (`ONEAPI_DEVICE_SELECTOR=level_zero:1`). The
measurement used the headless card, not the one driving a display.

The launcher runs this inside the image:

```text
llama-server -m <gguf> --mmproj <mmproj> -ngl 999 --fit off -c 65536 -np 1 \
  -b 8192 -ub 4096 --flash-attn on -ctk q8_0 -ctv q4_1 -t 8 --jinja --metrics
```

| Flag | Why |
|---|---|
| `-ngl 999 --fit off` | All layers on the card; skip the automatic fit search. |
| `-c 65536 -np 1` | 64K context, one slot. This is the tested shape. |
| `-b 8192 -ub 4096` | `-ub 4096` measured 6% faster than 2048 on a 16K prompt. |
| `-ctk q8_0 -ctv q4_1` | Quantized KV cache, as in the measurement. |
| `--jinja` | Uses the model's chat template, including the thinking switch. |
| `GGML_SYCL_GROUPED_MAX_ROWS=0` | Patch knob: grouped XMX MoE GEMM at every batch width. Measured +7% on pp8192. |

### 3. Thinking on or off

Thinking is on by default (chat template). Turn it off per request:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 17*23?"}],
       "chat_template_kwargs":{"enable_thinking":false},"temperature":0}'
```

With thinking on, the server returns the reasoning in a separate field.

## Rebuild the image

The image is the upstream Intel Dockerfile with the patch applied. To build it
yourself from a cookbook checkout:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 620d2225cc436ab0b001e3200e590e3b286ef9ab
git apply /path/to/cookbook/patches/llamacpp-sycl/llama.cpp-620d2225c-b70-bolt.patch
docker build -f .devops/intel.Dockerfile --target server \
  --build-arg GGML_SYCL_F16=ON -t llama.cpp-sycl-b70:b11669-bolt .
```

Then pass `IMAGE=llama.cpp-sycl-b70:b11669-bolt` to the launcher.

Patch SHA-256: `51fe97fd2f76ee04adddcf4fc2c0fcb4158fed49d411edee0145a5b736f972f7`
(12 files: 11 engine files plus a row-strided `OUT_PROD` case in
`tests/test-backend-ops.cpp`). What it changes:

- F16 build: the XMX dequant-GEMM paths take f16 activations.
- NVFP4 Mamba projections use a oneDNN BF16 route instead of the F32 fallback.
- `GGML_SYCL_GROUPED_MAX_ROWS` knob for the grouped MoE GEMM.
- Staged `ssm_scan` for long prompts, one 64-dim head per work-group.
- Fused squared ReLU (`relu` + `sqr`) for the MoE FFN.
- Top-9 expert ids compacted on the device before the host copy.
- `OUT_PROD` accepts a row-strided F32 source, as CUDA does. Stock SYCL
  aborted on the first DFlash draft step. Other layouts now fall back to CPU
  instead of aborting.

`test-backend-ops` on the patched build: SSM_SCAN 12/12, MUL_MAT_ID 937/937,
MUL_MAT 2277/2277, OUT_PROD 87/87, SQR 6/6.

## Why there is no drafter

NVIDIA publishes a DFlash drafter for this model
(`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF`). With the patch
it runs, keeps greedy output identical, and 87% of drafted tokens are
accepted. Decode is still slower: 25.7 tok/s with the drafter against 28.1
without it. On SYCL a 7-token verify step costs about 5x one decode step, so
the accepted tokens do not pay for the verify. The recipe serves without a
drafter.

## Numbers

### llama-bench, C1 (record `nemotron35-bolt-74b-llamacpp-c1-v1`)

`-ngl 999 -fa 1 -ctk q8_0 -ctv q4_1 -ub 4096 -b 8192`, built-in same-shape
warmup, 5 reps, median, synthetic tokens, cold prefix, 230 W cap. "Stock" is
unpatched `620d2225c` built with `GGML_SYCL_F16` off. The two builds ran in
separate sessions. token/s:

| Cell | Stock | Patched (this route) |
|---|---:|---:|
| pp2048 | 581.3 | 1357.2 |
| pp8192 | 664.1 | 1430.7 |
| pp4096 at depth 16384 | | 1310.7 |
| pp4096 at depth 32768 | 625.5 | 1247.1 |
| pp4096 at depth 61440 | | 1137.2 |
| tg128 at depth 512 | 28.17 | 28.24 |
| tg128 at depth 16384 | | 26.82 |
| tg128 at depth 32768 | 25.47 | 25.58 |
| tg128 at depth 65280 | | 23.43 |

The patch makes prefill 2.0 to 2.3x faster and leaves decode unchanged.

### llama-server, C1

Server `timings`, recipe flags, greedy, thinking off, cold prompt. A needle
placed at 50% depth was found in every long prompt. token/s:

| Prompt tokens | Prefill | Decode |
|---:|---:|---:|
| 886 (128 out) | 840 | 28.1 |
| 8,022 | 1230.8 | 27.36 |
| 16,012 | 1308.9 | 26.79 |
| 32,018 | 1305.8 | 25.65 |
| 59,976 | 1262.9 | 23.67 |

Evidence: [`llamacpp-c1-20261015-summary.json`](../../results/nemotron35-bolt-74b/llamacpp-c1-20261015-summary.json).

## Quality

- 6 greedy smoke cases (arithmetic, capital city, letter count, code,
  translation, image) pass. The text is byte-identical to the stock build.
- KL divergence against stock F32-path logits on wikitext-2 (10 x 2048
  tokens): mean 0.00139, same top token 98.56%, perplexity ratio
  1.00085 ± 0.0007. This is f16 activation rounding; upstream
  `GGML_SYCL_F16=ON` images have the same.
- A 6-case smoke does not establish general quality.

## Not tested

- More than one concurrent request, or more than one card.
- Context above 65,536 tokens.
- The published image itself. The numbers above come from a local
  `GGML_SYCL_F16=ON` build of the same commit and engine changes; the
  published patch adds only the `OUT_PROD` test case on top.
- The upstream `ghcr.io/ggml-org/llama.cpp:server-intel-b11669` image (runs
  the model without the fixes; not measured).
- Task-quality benchmarks beyond the smoke and the KLD screen.
- Power caps other than 230 W.
