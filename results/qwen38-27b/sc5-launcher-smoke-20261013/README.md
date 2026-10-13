# SC_5.00bpw_H6 launcher smoke — 2026-10-13

Correctness smoke only; no speed figures.

- Launcher: b70-launcher 0.5.2 (`b70 launch qwen38-27b-sc5 -e exl3 --gpus 1`); status fix verified on 0.5.3.
- Image: `ghcr.io/sergiiob/exl3xpu-mixedk:20261012c@sha256:7bb0c5cfed8a35da8f4db0034338d8dbe5f53731253d96ea8f1db2b5bddd4a40`, pulled by digest.
- Model: `turboderp/Qwen3.8-27B-exl3` branch `SC_5.00bpw_H6` @ `dfe2f4fd71bf9a1994051d4d47e6460fa96dd723`.
- Hardware: one Arc Pro B70, 230 W cap. Recipe defaults: max_model_len 65,536, gpu_memory_utilization 0.90, fp8 KV, MTP3, prefix caching on.
- `canary.txt`: 6/6 exact-answer canaries (greedy) + 3/3 long prompts (8,161–8,162 prompt tokens).
- `multiturn.txt`: 4-turn session on a shared long context with prefix caching on, prompts up to 15,600 tokens: PASS (facts recalled on every turn).
