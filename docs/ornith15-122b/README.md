# Ornith-1.5-122B on two Arc Pro B70s

Family index. Keep Ornith-1.5-35B-A3B on [its own page](../ornith15-35a3/README.md).

| Path | What it is |
|---|---|
| [ORNITH15-122B-LLAMACPP.md](ORNITH15-122B-LLAMACPP.md) | Recipe: Unsloth UD-IQ3_XXS GGUF, llama.cpp SYCL, two cards |
| `benchmarks/ornith15-122b/launch-ornith-122b-iq3xxs.sh` | Default serve: layer split 50/50, 131K, `-b/-ub 2048`, q8 K / q4 V, C1 |

## Artifact

- Target: `unsloth/Ornith-1.5-122B-GGUF` file `Ornith-1.5-122B-UD-IQ3_XXS.gguf` (~49.52 GB)
- Base: `ornith-ai/Ornith-1.5-122B` (MoE ~122B, ~9B active, MIT)

This is not an official Ornith GPTQ. There is no published AutoRound/GPTQ that fits 64 GB.

## Warnings

1. **Two cards.** Weights alone are ~49.5 GB. One 32 GB B70 cannot hold them.
2. **UD-IQ3_XXS is the fit pick.** Official Q4_K_M (~75.8 GB), Unsloth UD-Q3_K_XL (~60 GB), EXL3 4 bpw (~65.5 GB / 63.4 GiB load), FP8, and BF16 do not leave KV room on 2×32 GB.
3. **DFlash 2 is not this route.** `incoai/Ornith-1.5-122B-DFlash2` is an SGLang NVIDIA drafter (GB300 TP2). Do not mix it into llama.cpp SYCL or vLLM XPU apply lists.
4. Measured llama-bench (host F16 SYCL, n=5): pp2048 **700.07** tok/s, tg128 **28.10** tok/s. Do not copy 35B MixedCal or Flash-Next tok/s onto 122B.
