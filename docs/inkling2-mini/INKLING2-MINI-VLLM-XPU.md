# Inkling 2 Mini on Intel Arc Pro B70: vLLM XPU + `vllm-inkling2-xpu`

**Primary one-card route (measured 2026-11-22).** Serves the unmodified [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) checkpoint (native NVFP4/MXFP8 routed experts, BF16 dense/attention/`lm_head`, BF16 KV cache) up to **200,000 tokens of context** on a single 32 GB Intel Arc Pro B70, with **3,172 to 1,711 tok/s cold prefill** (2K to 196K) and **54.5 to 26.3 tok/s decode** across the full reachable window (0 to 198K, model-card sampling).

Credit: out-of-tree XPU plugin by [`mgaruccio/vllm-inkling2-xpu`](https://github.com/mgaruccio/vllm-inkling2-xpu), XPU FP32 `lm_head` fast path from upstream [vLLM PR #69186](https://github.com/vllm-project/vllm/pull/69186).

## 1. Pinned Stack

| Component | Exact Pin |
|---|---|
| Hardware | 1x Intel Arc Pro B70 (32 GB, non-display card, `xe` driver, 230 W cap) |
| Model | `thinkingmachines/Inkling-2-Mini` @ `ed40e7da625f594fc9c4990f5f66be24873c22d7` (`model-00001-of-00001.safetensors` sha256 `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`) |
| vLLM XPU | `vllm-0.34.0rc1.dev1133+g647a76866.xpu` (upstream `647a76866b4559a74e27d1c6b55fbffe85c07ce4`, wheel sha256 `7f16548be2192e6ce7cf2d77a96b4b04418648a5178b3bfa9c15e13f042520fb`) |
| Runtime | `torch==2.15.0+xpu`, `triton==3.9.0+xpu`, `vllm-xpu-kernels==0.1.17` |
| Plugin | `mgaruccio/vllm-inkling2-xpu` @ `2e9d28b04ff23f5d9996a221ffaa0ae3a440d8f9` (`--no-deps`) |
| Required patch | [`patches/inkling2-mini-vllm-xpu/pr69186-logits-processor.patch`](../../patches/inkling2-mini-vllm-xpu/pr69186-logits-processor.patch) (2-line XPU FP32 `lm_head` fast path; skip once PR #69186 merges) |

Do not convert this checkpoint to GPTQ INT4 or EXL3 (see `DNR-009`, `DNR-035`, `DNR-041`). Experts stay in their original NVFP4 (layers 0–19) and MXFP8 (layers 20–23) storage and unpack to BF16 operands in Triton (`W4A16`/`W8A16`); Battlemage has no native FP4/FP8 DPAS hardware.

## 2. Install and Patch

```bash
uv venv --python 3.14 vllm-inkling-venv
uv pip install --no-build --python vllm-inkling-venv/bin/python \
  --index https://download.pytorch.org/whl/xpu \
  --index-strategy unsafe-best-match \
  --find-links https://vllm-project.github.io/vllm-xpu-kernels/whl/vllm-xpu-kernels/ \
  "https://wheels.vllm.ai/647a76866b4559a74e27d1c6b55fbffe85c07ce4/vllm-0.34.0rc1.dev1133%2Bg647a76866.xpu-cp38-abi3-manylinux_2_34_x86_64.whl" \
  "torch==2.15.0+xpu" \
  "triton @ https://wheels.vllm.ai/647a76866b4559a74e27d1c6b55fbffe85c07ce4/triton-3.9.0%2Bxpu-py3-none-any.whl" \
  "vllm-xpu-kernels==0.1.17"

uv pip install --python vllm-inkling-venv/bin/python --no-deps \
  "git+https://github.com/mgaruccio/vllm-inkling2-xpu@2e9d28b04ff23f5d9996a221ffaa0ae3a440d8f9"
```

Apply PR #69186 following [`patches/inkling2-mini-vllm-xpu/APPLY.md`](../../patches/inkling2-mini-vllm-xpu/APPLY.md). Keep `VLLM_RPC_BASE_PATH` on a short directory such as `/tmp/ink2-ipc-*` so ZeroMQ IPC socket paths stay under Linux's 107-byte `sun_path` limit.

## 3. Serve

Use [`benchmarks/inkling2-mini/launch-inkling2-mini-vllm-xpu.sh`](../../benchmarks/inkling2-mini/launch-inkling2-mini-vllm-xpu.sh):

```bash
# Balanced daily profile: up to 90,112 context (~4.0 GiB live VRAM headroom)
VLLM_BIN=./vllm-inkling-venv/bin/vllm MODEL=/path/to/Inkling-2-Mini \
  CTX=90112 GPU_MEM_UTIL=0.88 \
  benchmarks/inkling2-mini/launch-inkling2-mini-vllm-xpu.sh

# Maximum single-card context profile: up to 200,000 context (~1.8 GiB live VRAM headroom)
VLLM_BIN=./vllm-inkling-venv/bin/vllm MODEL=/path/to/Inkling-2-Mini \
  CTX=200000 GPU_MEM_UTIL=0.95 \
  benchmarks/inkling2-mini/launch-inkling2-mini-vllm-xpu.sh
```

Required serving flags and why:
- `--language-model-only --dtype bfloat16`: text-only XPU path; BF16 activations and KV cache.
- `-cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY`: disables unsupported `torch.compile` while enabling XPU decode graph replay.
- `--no-enable-prefix-caching`: prefix caching stays off until multi-turn corruption checks pass on this stack.
- `--reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling`: separates `message.reasoning` from `message.content`; without it, raw reasoning leaks into `content`.
- First requests after a cold Triton cache compile 8 Triton kernels; persist `TRITON_CACHE_DIR` across restarts.

## 4. Measured Performance (1x B70, 230 W, C1, Cold Cache)

All measurements use one client-side streaming OpenAI `/v1/chat/completions` observer capturing both `delta.reasoning` and `delta.content`, with `vllm:prefix_cache_hits_total` delta verified at `0.0` on every request. Raw receipts: [`results/inkling2-mini/vllm-xpu-20261122/`](../../results/inkling2-mini/vllm-xpu-20261122/README.md).

### Same-session PR #69186 attribution (greedy, 256 tokens out, `n=3`, `CTX=90112`, `u=0.88`)

| Arm | Depth | Decode (median) | TPOT | Cold TTFT | Decode J/tok | Greedy parity |
|---|---:|---:|---:|---:|---:|---|
| Base (no #69186) | `0` (76 tok) | 39.71 tok/s | 25.18 ms | 0.163 s | 4.705 | — |
| **+ PR #69186** | `0` (76 tok) | **56.44 tok/s (+42.1%)** | **17.72 ms** | 0.135 s | **3.269 (-30.5%)** | short near-tie variance |
| Base (no #69186) | `8k` (9,172 tok) | 37.49 tok/s | 26.67 ms | 3.105 s | 5.057 | 3/3 self-identical |
| **+ PR #69186** | `8k` (9,172 tok) | **51.99 tok/s (+38.7%)** | **19.23 ms** | 3.054 s | **3.577 (-29.3%)** | **3/3 byte-identical to base** |

Memory effect: eliminating the per-step FP32 `lm_head` cast (~1.72 GiB temporary) freed **+1.22 GiB of KV cache** at `u=0.88` (`6.60` to `7.82 GiB`, raising reported KV capacity from `90,540` to `107,351` tokens) and raised the single-card ceiling to **200,000 tokens** at `u=0.95`.

### Cold prefill anchors (`pN/g1`, `n=3`, unique entropy-first prefixes, no-spec + #69186)

| Target prompt | Median TTFT | Cold input rate | Avg card power |
|---:|---:|---:|---:|
| 2,048 | 0.646 s | **3,172 tok/s** | 232.8 W |
| 8,192 | 2.648 s | **3,094 tok/s** | 230.5 W |
| 16,384 | 5.477 s | **2,992 tok/s** | 229.8 W |
| 32,768 | 11.696 s | **2,802 tok/s** | 229.7 W |
| 65,536 | 26.394 s | **2,483 tok/s** | 229.6 W |
| 131,072 | 64.693 s | **2,026 tok/s** | 229.5 W |
| 196,608 | 114.920 s | **1,711 tok/s** | 229.6 W |

### Full-context ladder (model-card sampling: `temperature=1.0`, `top_p=1.0`, `reasoning_effort=0.9`, `max_tokens=2048`, `n=3` seeds)

`0` through `max-88k` ran on the `CTX=90112, u=0.88` profile (`min_vram = 4,002 MiB`); `128k` and `max-198k` ran on `CTX=200000, u=0.95` (`min_vram = 1,832 MiB`). `256k` **does not fit** on one 32 GB B70 (largest proven context: `200,000`).

| Depth | Prompt tok | Cold TTFT | Cold input rate | Decode tok/s (min–max) | Decode W | Decode J/tok | Empty answers |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0` | 178 | 0.181 s | 1,014 tok/s | **54.51** (54.50–54.51) | 188.9 W | 3.47 | 3/3 (thinking-only in 2K) |
| `8k` | 9,268 | 3.104 s | **2,986 tok/s** | **51.44** (51.42–51.44) | 190.7 W | 3.71 | 3/3 (thinking-only in 2K) |
| `32k` | 39,062 | 14.437 s | **2,705 tok/s** | **44.66** (44.64–44.67) | 196.3 W | 4.40 | 3/3 (thinking-only in 2K) |
| `64k` | 65,648 | 26.590 s | **2,469 tok/s** | **39.92** (39.92–39.94) | 200.2 W | 5.02 | 3/3 (thinking-only in 2K) |
| `max-88k` | 87,936 | 38.303 s | **2,296 tok/s** | **36.83** (36.82–36.98) | 202.2 W | 5.50 | **0/3** |
| `128k` | 131,171 | 64.785 s | **2,025 tok/s** | **31.66** (31.64–31.66) | 205.6 W | 6.50 | 3/3 (thinking-only in 2K) |
| `max-198k` | 197,823 | 116.218 s | **1,702 tok/s** | **26.31** (26.28–26.35) | 208.5 W | 7.96 | **0/3** |

At `reasoning_effort=0.2` (`0` and `8k`, `n=3`, `max_tokens=2048`), all **6/6 requests returned non-empty final answers** at **54.51 / 51.44 tok/s** (`effort02-cells.jsonl`).

## 5. Quality, Retrieval, and Claim Boundaries

Deterministic chat/code/JSON/multi-turn and needle-in-a-haystack retrieval suite (`scenarios.jsonl`, `temperature=0`, `seed=4242`, `--reasoning-parser inkling`): **8/9 PASS**:
- **PASS:** `math-return-trip` (2.0 s), `structured-dispatch` (7.0 s), `code-boundary` (5.5 s), `conversation-correction` (13.2 s), `retrieval-64-records` (1.4K, 2.8 s), `retrieval-384-records` (8.5K, 5.3 s), `retrieval-1536-records` (33.8K, 15.8 s), and **`retrieval-6144-records` (135,291 tokens, 72.9 s)**.
- **FAIL (1/9):** `code-state-trace` returned `[2, 4, 5, 6, 7, 1]` instead of `[1, 2, 5, 6, 7, 4]`.

**Explicit limits (not claimed):**
- Single B70, text-only, `C1` (`--max-num-seqs 1`); multi-user batching and vision inputs are untested.
- `W4A16`/`W8A16` BF16 activations (not native Blackwell NVFP4 activations).
- `256k` context does not fit on one 32 GB B70 with BF16 KV cache.
- DSpark `k=7` speculative decoding passed a preliminary 64K fit probe (`26.7 s` TTFT, `41.87 tok/s` at 64K, `min_vram = 3,438 MiB`, `110,976`-token estimated ceiling at `u=0.93`) and is not yet promoted until its full-context ladder and parity checks complete.
