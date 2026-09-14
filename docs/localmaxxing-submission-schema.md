# localmaxxing.com submission schema

The platform uses a flat JSON object. Keep the richer internal manifest because one flat payload cannot preserve separate decode, prefill, latency, cache, and concurrency observations.

## Schema (flat, working)

```json
{
  "hfId": "Qwen/Qwen3.6-35B-A3B",
  "hardware": {
    "hwClass": "DISCRETE_GPU",
    "gpuName": "Intel Arc Pro B70",
    "gpuCount": 1,
    "vramGb": 32.0,
    "os": "Linux"
  },
  "engineName": "vllm",
  "engineVersion": "v0.26.1rc1.dev457+gc810e5ee9 XPU",
  "quantization": "GPTQ-INT4",
  "backend": "xpu",
  "tokSOut": 204.6,
  "tokSPrefill": 8153.0,
  "contextLength": 16384,
  "batchSize": 1,
  "notes": "Public image vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97. Observed vLLM v0.26.1rc1.dev457+gc810e5ee9; vllm-xpu-kernels 0.1.12. Patches: patch_mtp_nightly.py then patch_mtp_boundary.py. MTP4, single-stream. 204.6 tok/s is a short g32 maximum-observed cell; 8153 tok/s is cold p4k with unique prefix. Independent reproduction pending.",
  "engineFlags": {
    "commandSnippet": "vllm serve /model --quantization gptq --dtype float16 --max-model-len 16384 --gpu-memory-utilization 0.85 --max-num-seqs 64 --max-num-batched-tokens 8192 --enable-prefix-caching --language-model-only --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":4}",
    "kvCacheDtype": "auto",
    "flashAttn": true,
    "numParallel": 1
  }
}
```

## Required fields

| Field | Type | Example | Notes |
|-------|------|---------|-------|
| `hfId` | string | `Qwen/Qwen3.6-35B-A3B` | HuggingFace org/model. Must match leaderboard model registry. |
| `hardware` | object | (see above) | `hwClass` ∈ {DISCRETE_GPU, IGPU, ...}. `gpuName` must match leaderboard hw registry. |
| `engineName` | string | `vllm` or `llama.cpp` | |
| `engineVersion` | string | `v0.26.1rc1.dev457+gc810e5ee9 XPU` | Include the observed version and immutable image digest in notes. |
| `quantization` | string | `GPTQ-INT4` / `Q4_K_XL` / `Q5_K_M` | |
| `backend` | string | `xpu` | |
| `tokSOut` | float | `204.6` | Platform output field. Preserve timing formula and exact prompt/output coordinate in the internal manifest. |
| `tokSPrefill` | float | `8153.0` | Use an actual cold prompt of at least 1,024 tokens and retain zero cache-hit delta. |
| `contextLength` | int | `16384` | Max context the server was configured for. |
| `batchSize` | int | `1` | 1 for single-stream. |
| `notes` | string | | **Disclose patches + setup.** Reproducibility matters. |
| `engineFlags.commandSnippet` | string | | The exact launch command. |
| `engineFlags.kvCacheDtype` | string | `q8_0/q4_1` / `auto` | |
| `engineFlags.flashAttn` | bool | `true` | |
| `engineFlags.numParallel` | int | `1` | |

## Leaderboard weights (as of Jul 2026)

VRAM 30% / Model size 30% / Run count 25% / TPS 15%.

So submitting **multiple quants/models** (each valid run) helps the score more
than one high-TPS run. The vLLM MTP run is the standout (high TPS + large model
+ 32GB VRAM), but the llama.cpp MoE + dense runs add run-count weight.

## Honesty rules

localmaxxing values reproducibility. Every submission MUST disclose:

- **Patches applied** — link the repo if non-stock engine.
- **Speculative decoding** — note MTP/ngram + acceptance if known.
- **Power cap** — note if non-stock.
- **Single-stream vs concurrent** — `tokSOut` is single-stream. If you measured
  multi-user aggregate, that goes in notes, NOT `tokSOut`.

Historical payloads in `submissions/` disclose their old four-patch local-image stack. Do not copy those notes into a current nightly submission.

## POST endpoint (curl fallback)

```bash
curl -sS -X POST https://www.localmaxxing.com/api/speed-tests \
  -H "Authorization: Bearer $LOCALMAXXING_API_KEY" \
  -H "Content-Type: application/json" \
  -d @submission.json
```

`LOCALMAXXING_API_KEY` (format `bhk_...`) from localmaxxing.com account settings.
Never commit the key.
