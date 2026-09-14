# Nemotron-3.5-Lightning + DFlash on the B70 (vLLM XPU)

Isolated C1 n=5 Lane 1 card. Copy numbers only from [CLAIMS.md](CLAIMS.md).

This page is the **DFlash** recipe. The no-spec graph path (93 / 87 t/s) stays
on [NEMOTRON-B70.md](NEMOTRON-B70.md). Do not mix those tables.

## What this is

NVIDIA Nemotron-3.5-Lightning-30B-A3B (hybrid Mamba2 + LatentMoE, 3B active)
served on one Intel Arc Pro B70 32 GB with:

1. a **local symmetric GPTQ INT4 G64** target
2. a **local NVFP4→BF16 DFlash** draft
3. vLLM `method=dflash`, `num_speculative_tokens=7`
4. XPU graphs (PIECEWISE+FULL) + native grouped-topk v2 + SSU B8/W4

Native MTP on this stack historically accepts **0%**. DFlash is the working
speculator.

## Stack (do not substitute)

| Component | Exact tested value |
|---|---|
| Public image digest | `vllm/vllm-openai-xpu@sha256:1da0a95485455f08588c11080b9718992fd7d434c6a965d74654903a9d999c57` |
| vLLM | `0.26.1rc1.dev668+g3ee2df303` |
| `vllm-xpu-kernels` | `0.1.12.3` |
| Target | [`SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym`](https://huggingface.co/SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym) |
| Draft | [`SergiioB/Nemotron-3.5-Lightning-30B-A3B-DFlash-BF16`](https://huggingface.co/SergiioB/Nemotron-3.5-Lightning-30B-A3B-DFlash-BF16) |
| Source BF16 | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` |
| Source DFlash | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash` |
| Runtime (in launcher) | `patches/patch_xpu_grouped_topk_native_v2.py` + `patches/ssu-b70-b8w4/` (still required on this `0.1.12.3` image) |
| Upstream (2026-09-05) | [vllm#52159](https://github.com/vllm-project/vllm/pull/52159) closed without merge. Fused XPU grouped_topk is [vllm#53580](https://github.com/vllm-project/vllm/pull/53580) (open). [vllm-xpu-kernels#524](https://github.com/vllm-project/vllm-xpu-kernels/pull/524) (`at::zeros`) still open. Muse tuple already in kernels `main` via [#526](https://github.com/vllm-project/vllm-xpu-kernels/pull/526). |
| Context / batch / seqs | **120,000** serving limit / 8,192 / `max_num_seqs=1`. Speed card remains the isolated 16K n=5 matrix |
| Cache | **explicitly off** (`--no-enable-prefix-caching`) |
| Power | configured **150 W** |

## Measured (isolated n=5, C1, cache off, 150 W)

Timing is **client monotonic SSE**. Decode = `(completion_tokens-1)/(end-first)`.
Input = `prompt_tokens / TTFT` (**cold input rate**, not isolated engine prefill).

Canonical evidence (private host log):
`results/nemotron-dflash-bf16-n7-n5-20260813T082203Z-1407994/`
(benchmark-history Run 36).

![Nemotron DFlash advanced card — speed, matched speculation, capacity](../assets/b70-nemotron-dflash-advanced-dashboard.svg)

![Nemotron DFlash isolated n=5 dashboard](../assets/b70-nemotron-dflash-dashboard.svg)

| Cell | Metric | n | median | min | max | acceptance |
|------|--------|--:|-------:|----:|----:|-----------:|
| p2048/g1 | cold input (tok/s) | 5 | 6455.6 | 6262.9 | 7353.3 | — |
| p8192/g1 | cold input (tok/s) | 5 | **7160.1** | 7117.3 | 7226.3 | — |
| p512/g128 | C1 client post-first | 5 | 194.61 | 140.20 | 220.01 | 45.1% |
| **p2048/g128** | C1 client post-first | 5 | **186.61** | 174.60 | 201.83 | 56.5% |
| p8192/g128 | C1 client post-first | 5 | 157.92 | 143.50 | 170.25 | 53.0% |

- Representative decode for any single scalar: **186.6 t/s** (p2048/g128).
  Do **not** headline p512 194.6 (41% family range).
- Window DFlash acceptance **1830/3521 = 52.0%**.
- After load (16K speed card): 5826 MiB `visible_avail`.
- Context **capacity** (Run 38, not a speed card): `max-model-len=120000`
  loaded (5328 MiB free, KV 295,000) and completed staged C1 requests through
  **p119872+g32 = 119,904** tokens. **128K was not run.**
  `n` is repeats, not concurrency — both campaigns are **C1**.

![Speed card vs 120K capacity ladder](../assets/b70-nemotron-dflash-context-dashboard.svg)

| Campaign | `max_model_len` | Prompt / gen | Metric | n | Value |
|---|---:|---|---|--:|---:|
| Speed (Run 36) | 16384 | p8192/g1 | cold input median | 5 | **7160** |
| Speed (Run 36) | 16384 | p2048/g128 | C1 client post-first median | 5 | **186.61** |
| Speed (Run 36) | 16384 | p8192/g128 | C1 client post-first median | 5 | 157.92 |
| Speed (Run 36) | 16384 | p512/g128 | C1 client post-first median | 5 | 194.61 (do not headline) |
| Capacity (Run 38) | 120000 | 32768+32 | completed | 1 | 32800 |
| Capacity (Run 38) | 120000 | 65536+32 | completed | 1 | 65568 |
| Capacity (Run 38) | 120000 | 98304+32 | completed | 1 | 98336 |
| Capacity (Run 38) | **120000** | **119872+32** | **completed** | 1 | **119904** |

Run 38 g32 post-first rates (103.8 / 75.6 / 52.1 / 37.1) are diagnostics only.

- Cell-window draw ~149–160 W; peak interval-average 179.3 W; pkg max 68.0 °C.
  Raw `energy1_input` / temp samples are in the private Run 36 directory
  (`monitor.jsonl`); not mirrored in this public repo.
- Deterministic raw-completion replay smoke: `exact_match=true` (smoke only).

### Matched-except-speculation

Same target, same image family, no-spec n=5 p8192/g128 = **87.25 t/s**.
DFlash p8192/g128 = **157.92 t/s** → **1.81× on that cell only**.

### The “10k prefill” number

An earlier no-spec n=3 screen showed ~10,349 tok/s from p8192/**g128** TTFT.
That is a different cell, n=3, and it is still a cold input rate. Isolated
DFlash Lane 1 input is **7160** at p8192/**g1** n=5. Do not say “DFlash hit 10k
prefill.”

## Quick Start (3-Step Setup)

### Step 1: Pull the public image
```bash
export IMAGE='vllm/vllm-openai-xpu@sha256:1da0a95485455f08588c11080b9718992fd7d434c6a965d74654903a9d999c57'
docker pull "$IMAGE"
```

The launcher applies `patches/patch_xpu_grouped_topk_native_v2.py` and SSU configurations automatically.

### Step 2: Download the target and draft models
```bash
# Target ~18 GB
export TARGET="$HOME/models/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym"
huggingface-cli download SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym \
  --local-dir "$TARGET"

# Draft ~1.7 GB
export DRAFT="$HOME/models/Nemotron-3.5-Lightning-30B-A3B-DFlash-BF16"
huggingface-cli download SergiioB/Nemotron-3.5-Lightning-30B-A3B-DFlash-BF16 \
  --local-dir "$DRAFT"
```

### Step 3: Launch server & verify health
```bash
# Capacity profile (120,000 max-model-len):
bash benchmarks/nemotron35-30a3/launch-nemotron-dflash.sh "$TARGET" "$DRAFT" 8001
curl -f http://127.0.0.1:8001/health
```

### Smoke test
```bash
curl -s http://127.0.0.1:8001/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nemotron-gptq-dflash7","prompt":"Return exactly this: SMOKE","max_tokens":32,"temperature":0,"seed":42}'
```

Speculative counters (`vllm:spec_decode_num_accepted_tokens_total`) will increment on valid drafting.

## Convert it yourself (optional)

The published HF repos are the reproduction default. Optional local conversion
is not required to follow this recipe.

Draft conversion contract:

- E2M1 low-nibble-first × `float8_e4m3fn` × F32 `weight_scale_2`, group 16, linear
- copy attention / embeddings / norms (already BF16)
- strip `quantization_config` and rename leftover ModelOpt json

## LocalMaxxing

Self-reported record `cmsr9po4w000ams01e4fc5qhj` (2026-08-13T08:40:16Z),
status `APPROVED`.

Displayed: `tokSOut=186.6`, `tokSPrefill=7160`,
`GPTQ-INT4-G64-sym-local+DFlash-BF16-local`, engine `vllm`.

Platform parser limitation: `specDecoding=false` / `specMethod=null` even
though the command snippet and notes carry `method=dflash` n=7.

## Changelog

- 2026-08-13: isolated n=5 DFlash proof. n=3 screen superseded.
- 2026-08-13: HF artifacts published under `SergiioB/` (canonical two-i account).
- 2026-08-13: upstream PRs opened — [vllm#52159](https://github.com/vllm-project/vllm/pull/52159),
  [vllm-xpu-kernels#524](https://github.com/vllm-project/vllm-xpu-kernels/pull/524).
  Source copies live in `patches/vllm-xpu-kernels/`. Launcher already applies
  the Python router patch; kernel `at::zeros` still needs a rebuild.
- 2026-09-05: [vllm#52159](https://github.com/vllm-project/vllm/pull/52159)
  closed without merge. Muse paged-decode tuple is in kernels `main`
  ([#526](https://github.com/vllm-project/vllm-xpu-kernels/pull/526)); do not
  apply `0002`. `#524` (`at::zeros`) and [vllm#53580](https://github.com/vllm-project/vllm/pull/53580)
  remain open. Python router patch still required on this pinned image.
