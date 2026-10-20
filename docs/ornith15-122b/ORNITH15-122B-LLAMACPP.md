# Ornith-1.5-122B UD-IQ3_XXS — llama.cpp SYCL, two B70s

One request at a time (`-np 1`). Default context is **131072**. Weights ~49.5 GB plus q8/q4 KV fit on two 32 GB cards.

Measured numbers below are **llama-bench engine rates** on a host SYCL binary with `GGML_SYCL_F16=ON` (`1692f9e50`, b361). They are not vLLM client post-first rates and not the Docker `server-intel-b11429` pin.

## Download

```bash
hf download unsloth/Ornith-1.5-122B-GGUF Ornith-1.5-122B-UD-IQ3_XXS.gguf \
  --local-dir /path/to/models
```

Expected size: **49,515,667,712 bytes**. Do not launch while a `.incomplete` sibling exists.

## Launch (measured flags)

Host binary, both cards, 150 W MoE cap:

```bash
llama-server -m /path/to/Ornith-1.5-122B-UD-IQ3_XXS.gguf \
  --host 0.0.0.0 --port 8000 \
  -c 131072 -np 1 -ngl 99 \
  --split-mode layer --tensor-split 50,50 \
  -fa on --cache-type-k q8_0 --cache-type-v q4_0 \
  -b 2048 -ub 2048 -t 6 \
  --temp 0.6 --top-p 0.95 --top-k 20
```

Docker helper (same flags, **unmeasured** image pin):

```bash
bash benchmarks/ornith15-122b/launch-ornith-122b-iq3xxs.sh \
  /path/to/Ornith-1.5-122B-UD-IQ3_XXS.gguf 131072 8000
```

| Flag | Value | Why |
|---|---|---|
| GPUs | 0,1 | 49.5 GB weights |
| `--split-mode` | `layer` | dual-card default |
| `--tensor-split` | `50,50` | even fill |
| `-ngl` | `99` | all layers on GPU |
| `-fa` | `on` | Flash attention |
| `-ctk` / `-ctv` | `q8_0` / `q4_0` | KV for 131K |
| `-np` | `1` | C1 |
| `-b` / `-ub` | `2048` / `2048` | prefill (512/512 was ~half the pp rate) |
| `-t` | `6` | host threads |
| sampling | temp 0.6, top_p 0.95, top_k 20 | Ornith coding defaults |
| power | 150 W/card | MoE sweet spot |

## Measured (2026-10-20)

llama-bench, n=5, `-b/-ub 2048`, FA on, q8/q4 KV, layer 50/50, 150 W/card.

| Cell | Median tok/s |
|---|---:|
| pp512 | 460.50 |
| pp2048 | **700.07** |
| pp4096 | 697.87 |
| tg128 | **28.10** |

Serving check: `llama-server -c 131072` with these flags loads. Short-prompt chat decode n=5 / g128 is 27.6 tok/s. Do not mix that with the llama-bench medians.

Raw: [results/ornith15-122b/iq3xxs-llamacpp-bench-20261020/](../../results/ornith15-122b/iq3xxs-llamacpp-bench-20261020/).

Prefill note: `-ub 512` on the chat server was ~336 tok/s at ~2.6k prompt tokens. llama-bench `-ub 2048` is the prefill recipe.

## Not this recipe

- vLLM XPU GPTQ / MTP patches from Ornith-1.5-35B-A3B
- SGLang DFlash2 (`incoai/Ornith-1.5-122B-DFlash2`)
- Q4_K_M, UD-Q3_K_XL, EXL3 4 bpw, FP8, MLX
- MTP: GGUF still has unused `blk.48.nextn.*` tensors; this route is no-spec
