---
title: "P0 Baseline Inventory — 2026-09-20 (READ-ONLY)"
type: plan
date: 2026-09-20
status: final
---

# P0 Baseline Inventory — 2026-09-20 (READ-ONLY)

READ-ONLY artifact. No model loaded, no server started, no docker touched, nothing modified except this file.
Working directory: `~/B70-DOCS`. Every fact cites its source. `MISSING` = searched, not found, never guessed.

## ROUTE 1 — K216 (DeepSeek-V4-Flash-0731 REAP-K216, llama.cpp SYCL)

| field | value | source path or command | verified present? |
|---|---|---|---|
| checkout HEAD | `1692f9e50bb20fd96b963af38a282daf78feea64` | `git -C ~/llama.cpp rev-parse HEAD` | yes |
| checkout log | `1692f9e50 ggml: recurrent state rollback (#26623)` / `4c1a0af40` / `77918caf3` | `git -C ~/llama.cpp log --oneline -3` | yes |
| local source mods | `M common/arg.cpp, common/common.cpp, common/common.h, ggml/include/ggml-backend.h, ggml/src/ggml-backend.cpp, ggml/src/ggml-cpu/ggml-cpu.c, ggml/src/ggml-sycl/ggml-sycl.cpp, ggml/src/ggml-sycl/mmvq.cpp`; untracked `ggml-backend-moe-cache.h`, `ggml-sycl.cpp.bak-fix`, `moe-cache.cpp` | `git -C ~/llama.cpp status --short \| head -20` | yes — DIRTY tree |
| server binary | `~/llama.cpp/build-sycl-latest-qw38/bin/llama-server` (note: `bin/`, no `build/bin/`) | `ls -la` + `stat` | yes |
| binary mtime/size | `2026-08-26 12:48:21 +0200`, 755976 bytes, `-rwxrwxr-x` | `stat -c '%y %s %n'` | yes |
| cmake flags | `CMAKE_BUILD_TYPE=Release`, `GGML_SYCL=ON`, `GGML_SYCL_DEVICE_ARCH=` (empty), `GGML_SYCL_DNN=ON`, `GGML_SYCL_F16=ON`, `GGML_SYCL_GRAPH=ON`, `GGML_SYCL_HOST_MEM_FALLBACK=ON`, `GGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON`, `GGML_SYCL_TARGET=INTEL` | `build-sycl-latest-qw38/CMakeCache.txt` (grep `^GGML_SYCL\|^CMAKE_BUILD_TYPE`) | yes, verbatim |
| model shards | `MISSING` — `/mnt/<models-disk>/DeepSeek-V4-Flash-0731-REAP-K216-UD-IQ3_XXS/` gone; `find /mnt ~ -iname *REAP-K216*` empty; `*REAP*` hits are only node_modules/comfyui false positives | `timeout 120 find` commands | ABSENT |
| HF cache copy | `MISSING` — no `models--heath0xFF--*` in `~/.cache/huggingface/hub/` (has Qwen, Signal, rmacy drafter, Unsloth, OpenVINO entries only) | `ls -d .../hub/models--heath0xFF--*` + `ls hub/` | ABSENT |
| best-cell flags (L5) | `-c 4096 -ngl all -fa on -ctk f16 -ctv f16 -t 8 --ignore-eos -b 256 -ub 256 -dev SYCL0,SYCL1 -sm layer -ncmoe 15 -ts 65,35` | `results/2026/08/dsv4-k216-20260825Tdual2/campaign.log` (`[launch] L5` line) | recorded |
| best-cell model path | `/mnt/<models-disk>/DeepSeek-V4-Flash-0731-REAP-K216-UD-IQ3_XXS/...-00001-of-00003.gguf`, `n_slots=1`, `n_ctx_slot=4096` | `.../server-L5.log` | recorded (path now dead) |
| best-cell number | p512/g128 `median_engine_tg=8.871`, max 9.268, min 7.043, n=3 | `.../SUMMARY.txt` lines 8–9 | recorded |
| GPU placement | dual `-dev SYCL0,SYCL1`, `-ts 65,35`, `-sm layer` | campaign.log L5 line | recorded |
| power cap | campaign opened at 230 W, later pinned 150 W; per-cell L5 cap UNCERTAIN (see MISSING) | campaign.log `[cap]` lines; `research/2026/dsv4-k216-request-to-output-pipeline-20260825.md`:256 | uncertain |
| KV policy | f16/f16 for this campaign (`-ctk f16 -ctv f16`); fleet `q8_0 K + q4_1 V` unproven on DSV4; `q4_0` K forbidden | campaign.log; pipeline doc `:148` | recorded |

## ROUTE 2 — Nemotron-3.5-Lightning-30B-A3B (vLLM XPU, GPTQ-INT4 G64 + DFlash n7)

| field | value | source path or command | verified present? |
|---|---|---|---|
| docker image | `vllm-openai-xpu:1da0a954-det0123` — TAG ONLY, no digest recorded | `results/2026/08/nemotron-dflash-bf16-n7-n5-20260813T082203Z-1407994/manifest.txt` | recorded, digest MISSING |
| vLLM version | `0.26.1rc1.dev668+g3ee2df303` (in-container, from server log) | `.../server.log` (`version 0.26.1rc1...`) | recorded |
| model dir (recorded) | `/mnt/<models-disk>/nemotron-lightning-gptq-sym64-20260812` | manifest.txt `model=`; `cookbook:models/nemotron/.../artifacts.md:31` | ABSENT on disk today |
| drafter (recorded) | `/mnt/<models-disk>/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-DFlash-BF16` | manifest.txt `draft=` | PRESENT (see sizes) |
| drafter size | 1.6 GiB (`model.safetensors` 1666731848 B + config/README) | `du -sh` + `ls -la` on `/mnt/<models-disk>/...-DFlash-BF16` | yes |
| launch flags | `XPU_GRAPH=1`, async-scheduling, `method=dflash num_speculative_tokens=7`, `served_model=nemotron-gptq-dflash7`, `max_seq_len=16384`, TP1/PP1, `dtype=torch.float16`, `quantization=auto_gptq`, `kv_cache_dtype=auto`, prefix caching OFF, C1 | manifest.txt + server.log engine config line | recorded |
| patch | `patch_xpu_grouped_topk_native_v2.py` + SSU B8/W4 | manifest.txt `patch=`; file at `scripts/tmp/patch_xpu_grouped_topk_native_v2.py` | file PRESENT |
| best numbers | p2048/g128 186.61 (174.60–201.83, 56.5%) / p8192/g128 157.92 (143.50–170.25, 53.0%) / p512/g128 194.61 / p8192/g1 cold 7,160.1 / window accept 52.0% | `cookbook:models/nemotron/.../serving.md:30-40`; evidence `.../n5-summary.json` | recorded |
| GPU placement | TP1 single card; `ZE_AFFINITY_MASK` NOT recorded (harness.log grep empty; manifest hwmon=hwmon4 only) | `.../harness.log`, manifest.txt | MISSING |
| power cap | 150 W configured | manifest.txt `configured_cap_W=150`; serving.md:28 | recorded |
| KV policy | `auto`, prefix caching explicitly off | server.log (`kv_cache_dtype=auto`, `enable_prefix_caching=False`) | recorded |

## ROUTE 3 — Qwen3.8-27B GPTQ-INT4 G128 + BF16 MTP4 (vLLM XPU)

| field | value | source path or command | verified present? |
|---|---|---|---|
| target artifact | `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` (target GPTQ INT4 sym G128; native MTP tensors BF16; 400 qweight + 15 BF16 MTP tensors; 19.56 GB contract) | `models/qwen/qwen3.8-27b/README.md`:27; `artifacts.md:28-38` | recorded |
| image digest (FP8 recipe) | `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f` | `models/qwen/qwen3.8-27b/serving.md`:106 | recorded |
| Run 40 base stack | vLLM `0.27.2rc1.dev77+gac7509e2b`, XPU kernels 0.1.12.3, GPTQ G128, BF16 MTP, fp8 KV, graphs on, cache off, U=0.88, sched budget 8192, 230 W; p512/g128: 32.9 no-spec / 52.0 MTP1 / 65.8 MTP2 / 83.7 MTP4 | serving.md:32 | recorded |
| INT4 TP2 champion | TP2 p512/g128 84.00 (TPOT 9.199 ms, 69.1%) / p2048 38.35 / p8192 24.52, n=3, 230 W, FP8 KV, full graphs, no-g_idx artifact | serving.md:290-298; evidence `results/2026/08/qw38-int4-tp2-mtp4-230w-graph-confirm-20260828T120423Z/` | recorded |
| INT4 TP1 control | 150 W: p512/g128 49.08, p8192/g128 12.51, n=3 | serving.md:300-305; `results/2026/08/qw38-int4-sc-mtp4-champ-20260828T112032Z/` | recorded |
| Run 48 non-thinking sampling | temp 0.7, top_p 0.80, top_k 20, presence penalty 1.5, thinking off → 102.61 tok/s (Run 48 n=5) | serving.md:36-42 | recorded verbatim |
| Run 48 thinking sampling | temp 1.0, top_p 0.95, top_k 20, presence 0, thinking on → 99.32 tok/s | serving.md:40 | recorded verbatim |
| INT4 target on disk | `/mnt/<models-disk>/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16-nogidx` = 19 G (also under `/mnt/<sata-disk>/`) | `du -sh` | PRESENT |
| FP8 artifact on disk | `/mnt/<models-disk>/Qwen3.8-27B-FP8` = 29 G; recorded path `/mnt/<models-disk>/hf/qw38-fp8` ABSENT | `du -sh` + `ls` | PRESENT (new path) |
| HF cache | `models--Qwen--Qwen3.8-27B` = 12 K, refs only, no blobs | `du -sh` + `ls` | refs-only |
| GPU placement | INT4 TP2: both cards, per-worker affinity patch + `SYS_PTRACE`; TP1 control: single card | serving.md:119-123; champ manifest `tensor_parallel_size=1, pipeline_parallel_size=2` (PP2 validation file — not the TP2 speed launch) | recorded |
| power cap | INT4 TP2 230 W; TP1 control 150 W; 09-14 caveat: both MTP-depth arms ran 230 W | serving.md:382-422, 586-592 | recorded |
| KV policy | fp8 KV (INT4/FP8 routes); prefix cache off on record cells, on in Run 47/48 champion stack (do not blend) | serving.md:32, 42, 106-117 | recorded |

## MISSING / UNCERTAIN

1. K216 GGUF shards ABSENT everywhere (`/mnt/<models-disk>/...` gone; no `heath0xFF` HF cache entry) — Route 1 not launchable without re-download.
2. Nemotron GPTQ-INT4 G64 target dir (`/mnt/<models-disk>/nemotron-lightning-gptq-sym64-20260812`) ABSENT; no copy found under `/mnt/*/ ` — Route 2 target not launchable; drafter (1.6 G) survives.
3. Nemotron image digest MISSING (only tag `1da0a954-det0123` in manifest; no `sha256:` anywhere in dossier).
4. Nemotron `ZE_AFFINITY_MASK` / card assignment MISSING (harness.log grep empty).
5. Qwen INT4 TP2 full verbatim `docker run` + flag line MISSING as one block (flags scattered across serving.md; champ manifest on file is a TP1/PP2 validation file, not the TP2 speed launch).
6. Qwen recorded FP8 path `/mnt/<models-disk>/hf/qw38-fp8` ABSENT (superseded by `/mnt/<models-disk>/Qwen3.8-27B-FP8`, 29 G).
7. K216 L5 per-cell power cap UNCERTAIN (230 W at campaign open vs 150 W pins later in campaign.log).
8. Qwen HF cache is refs-only (12 K); K216 HF cache entry absent entirely.
