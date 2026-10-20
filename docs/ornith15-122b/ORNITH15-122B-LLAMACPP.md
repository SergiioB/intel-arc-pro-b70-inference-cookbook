# Ornith-1.5-122B UD-IQ3_XXS — llama.cpp SYCL, two B70s

One request at a time (`-np 1`). 32K is the default context so KV fits next to ~49.5 GB weights on 64 GB. 128K is opt-in with the same KV types, not a second recipe.

Image pin (launcher generation): `ghcr.io/ggml-org/llama.cpp:server-intel-b11429@sha256:3353d968ed8987612b7859828d14d205413567baafed00a90d71d0c23e63d7ba`. A host SYCL `llama-server` of the same generation is equivalent if you pass the same flags.

## Download

```bash
hf download unsloth/Ornith-1.5-122B-GGUF Ornith-1.5-122B-UD-IQ3_XXS.gguf \
  --local-dir /path/to/models
```

Expected size: **49,515,667,712 bytes**. Do not launch while a `.incomplete` sibling exists.

## Launch

```bash
bash benchmarks/ornith15-122b/launch-ornith-122b-iq3xxs.sh \
  /path/to/Ornith-1.5-122B-UD-IQ3_XXS.gguf 32768 8000
```

Flags the script pins:

| Flag | Value | Why |
|---|---|---|
| GPUs | 0,1 | 49.5 GB weights |
| `--split-mode` | `layer` | dual-card default for this GGUF |
| `--tensor-split` | `50,50` | even fill on two 32 GB cards |
| `-ngl` | `99` | all layers on GPU; abort if it does not fit |
| `-fa` | `on` | Flash attention |
| `-ctk` / `-ctv` | `q8_0` / `q4_0` | KV budget for 32K |
| `-np` | `1` | C1 |
| `-b` / `-ub` | `512` / `512` | MoE decode batch |
| `-t` / `-tb` | `6` / `14` | host threads vs GPU work |
| sampling | temp 0.6, top_p 0.95, top_k 20 | Ornith coding defaults |

Optional: `CTX=131072` for long context. If load fails, stay at 32K; do not offload experts until that is measured.

## Not this recipe

- vLLM XPU GPTQ / MTP patches from Ornith-1.5-35B-A3B
- SGLang `--speculative-algorithm DFLASH` + `incoai/Ornith-1.5-122B-DFlash2`
- Q4_K_M, UD-Q3_K_XL, EXL3 4 bpw, FP8, MLX

## Evidence

Load on two Arc Pro B70 32 GB: GGUF present at 49,515,667,712 bytes. Throughput is **unmeasured**. Publish tok/s only through a new catalog record after a n=5 cell.
