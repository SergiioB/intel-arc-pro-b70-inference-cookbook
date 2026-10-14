---
title: "spiritbuun `--moe-cache` vs dual-B70 SYCL K216 (2026-08-25)"
type: research
date: 2026-08-25
status: final
---

# spiritbuun `--moe-cache` vs dual-B70 SYCL K216 (2026-08-25)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE**

**Verdict: skip as a current speed option.** Worth a SYCL kernel port only after fused IQ3 `MUL_MAT_ID` exists. Do not copy the NVIDIA command line onto this host.

Claim examined (social, `directional_only`): DeepSeek-V4-Flash-0731 on **one RTX 5090 32 GB + 96 GB DDR5**, spiritbuun llama.cpp fork, ~90 GB GGUF, **25 → up to 50.6 tok/s** depending on quant, advertised 256K context. Recipe:

```text
-ngl 99 -sm layer -fa on -c 262144 --no-mmap -np 1
-ub 4096 -b 4096 -ctk f16 -ctv f16
-ot 'exps=CPU' --moe-cache on
-t 16 -tb 16
```

Fork: [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) (~781★). README names the feature **CUDA MoE cache** (`GGML_CUDA_MOE_CACHE_RESERVE_MB`, `ggml-cuda/moe-cache.cu`). SYCL is listed as a generic backend; it is **not** wired to this cache.

Local baseline to compare against: llama.cpp SYCL `1692f9e50`, REAP-K216 UD-IQ3_XXS, dual B70, best no-spec **8.87** engine tok/s (`n=3`, L5) — **SUPERSEDED 2026-09-21: 9.25 clean / 9.58 patched**. Evidence and pipeline: `research/2026/dsv4-k216-request-to-output-pipeline-20260825.md`.

Comparison class vs the 5090 post: **`directional_only`** (CUDA vs SYCL, 96 GB DDR5 vs 32 GB DDR4-2133, `--no-mmap` vs required mmap, unpruned/IQ2-class vs REAP-K216 IQ3, 256K vs our p512 cells, likely DSpark in the 50 tok/s “up to”).

---

## What `--moe-cache` actually is

Not a synonym for `-ncmoe N`.

| | Our `-ncmoe N` | Fork `--moe-cache` + `-ot 'exps=CPU'` |
|---|---|---|
| Expert home | First N layers’ routed experts **always** CPU | **All** routed experts CPU-resident |
| GPU experts | Remaining layers fully GPU | Spare VRAM holds an LRU of **hot expert tensors** (quantized bytes, including IQ3_XXS on CUDA) |
| Misses | N/A | CPU GEMV |
| Hits | N/A | CUDA quantized MMV, optional up/gate fusion |
| Backend | ggml-sycl / CPU | **ggml-cuda only** |

Fork README also: `--moe-cache on` = single GPU; `--moe-cache auto` = ≥2 GPUs conservative; default CUDA safety reserve **~3 GiB**; dual 3090 + DSpark ~51.5 tok/s mean; single 3090 + DSpark ~31.5 vs ~24 target-only; IQ2_M+DSpark RSS **~95.3 GiB**; **128 GiB RAM recommended**.

Our current SYCL binary has `-ot` and `-ncmoe`. It does **not** parse `--moe-cache`.

---

## Why their flags are a bad copy-paste here

1. **`--no-mmap` is forbidden** on this ~83 GiB GGUF with 32 GiB RAM (`AGENTS.md`). Their 96 GB box can pin the expert bank; ours cannot. Live host at write-down: ~30 GiB RAM, **~15 GiB swap already in use**.
2. **`-ot 'exps=CPU'` without the CUDA cache** is “all 43 layers’ routed experts on the Ryzen.” That is worse than L5, not better. L1 `-ncmoe 30` already measured **2.17** tok/s vs L5 **8.87** (SUPERSEDED: 9.25/9.58). All-CPU experts ≈ 2.3 GiB selected IQ3 bytes/token; DDR4-2133 theoretical 34 GB/s is an *impossible* ~14 tok/s roof before unpack, faults, and sync. Expect low-single-digit actual.
3. **Spare VRAM is the cache working set.** Packed P15_65 left **1897 / 1720 MiB** free. CUDA’s normal **3 GiB reserve leaves zero cache**. Even a 1 GiB reserve leaves ~0.7–0.9 GiB/card before scratch. One IQ3 expert triple ≈ 9.2 MiB; 1 GiB ≈ 111 layer-expert entries. One token selects **258** (43×6). Perfect 1 GiB IQ3 coverage ≤ ~43% of *one* token’s experts, and routing changes. FP16 copies (~48 MiB/expert) are worse (~21 entries/GiB).
4. **GPU1 is Gen3 x4, no P2P.** Dual-GPU `auto` + expert-parallel is written for NVIDIA peer/NVLink-class boxes. Host-staged fills on x4 would tax the miss/fill path.
5. **Building the fork with SYCL does not enable moe-cache.** Architect: cache provider and MMV live under `ggml-cuda`. A real port is slab alloc, async H2D fill, IQ3 cached MMV, hit compaction, dual-device policy **without** more `stream->wait()` on `MUL_MAT_ID`. Shallow CUDA→SYCL of the manager while keeping today’s host-ID fallback can easily lose to L5.
6. **30 tok/s is still not implied.** Their 50.6 is an NVIDIA “up to” on a different memory system. Local architect stretch with fused IQ3 MMVQ + compatible 216-expert GPU draft remains ~9.6–18.

---

## Ranked action

1. **Skip** — do not replace L5 with `-ot 'exps=CPU'`, do not `--no-mmap`, do not expect `--moe-cache on` on `build-sycl-latest-qw38`.
2. **Research-port (later)** — only as part of fused IQ3 SYCL `MUL_MAT_ID` MMVQ, then a quantized hot-expert slab in leftover VRAM. Not a weekend flag sweep.
3. **Try-build** — lowest value. Compile-for-SYCL of buun-llama-cpp does not answer the speed question.

If someone later ports it: keep mmap; do not drop below post-load VRAM gates; measure C1 p512/g128 n=3 against L5 on the same artifact; treat 5090 25→50.6 as `directional_only`.

---

## SYCL v1 port (2026-08-26) — built, not measured

Local llama.cpp tree `~/llama.cpp`, binary `build-sycl-latest-qw38`. Default remains **off**. This is **not** the CUDA 3800-line fused cache; it is a Xe-shaped v1:

| Piece | Where |
|---|---|
| Public flag | `--moe-cache off\|auto\|on\|N` (`N` = MiB budget per device) |
| Provider | `ggml/src/ggml-sycl/moe-cache.cpp` registered from `ggml_backend_sycl_reg()` |
| CPU hook | `ggml_compute_forward_mul_mat_id` in `ggml-cpu.c`: skip hit rows, GPU MMV, CPU miss fallback |
| IQ3 MMV | `ggml_sycl_mul_mat_vec_q_id` now handles `IQ3_XXS` and `IQ3_S` |
| Reserve | **256 MiB** leftover (not CUDA’s 2–3 GiB). Override `GGML_SYCL_MOE_CACHE_RESERVE_MB` |

Intended load (when GPUs are empty): keep mmap; put routed experts on CPU (`-ot 'exps=CPU'` or high `-ncmoe`); `--moe-cache on` or `--moe-cache 1024`. Expect log `[moe-cache] enabled SYCL v1`. No fused gate/up. Hits still H2D-fill then SYCL quantized MMV; misses stay CPU.

**Smoke not run 2026-08-26 12:48 local.** Empty-GPU gate failed: ComfyUI `main.py --listen 0.0.0.0 --port 8188` (PID 553117) holds both render nodes. GPU0 used ~30077 MiB / free ~2579 MiB; GPU1 used ~16992 MiB / free ~15664 MiB. Caps observed 180 W (left as-is; Comfy owns the cards). Do not race Comfy or the named inference services.

Still **PROVISIONAL**. Compare later C1 p512/g128 n=3 against L5 8.87 (SUPERSEDED 2026-09-21: 9.25 clean / 9.58 patched). Do not claim 30 tok/s. All-CPU experts without leftover VRAM leaves the cache dormant.
