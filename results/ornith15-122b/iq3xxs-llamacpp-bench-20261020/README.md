# Ornith-1.5-122B UD-IQ3_XXS llama.cpp SYCL bench (2026-10-20)

Self-reported, one host, two Arc Pro B70 32 GB. Engine: llama-bench, not a client SSE rate.

- Binary: `llama.cpp` `1692f9e50` / b361, `GGML_SYCL_F16=ON`
- GGUF: `unsloth/Ornith-1.5-122B-GGUF` `Ornith-1.5-122B-UD-IQ3_XXS.gguf` (49,515,667,712 bytes)
- Flags: `-ngl 99 -fa on -ctk q8_0 -ctv q4_0 -b 2048 -ub 2048 -t 6 -sm layer -ts 50/50`
- Power cap: 150 W per card (MoE)
- n=5 after llama-bench warmup

| Cell | Median tok/s |
|---|---:|
| pp512 | 460.50 |
| pp2048 | 700.07 |
| pp4096 | 697.87 |
| tg128 | 28.10 |

Serving check: `llama-server` `-c 131072` with the same `-b/-ub 2048` loads and answers. Chat-completions decode n=5 at short prompt / g128 is 27.6 tok/s (not mixed into the llama-bench median).

Raw: `llama-bench.json` in this folder.
