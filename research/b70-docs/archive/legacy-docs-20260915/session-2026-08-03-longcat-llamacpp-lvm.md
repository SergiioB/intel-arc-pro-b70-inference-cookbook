# B70 Session Report — 2026-08-03
## LongCat-Sparse fit, llama.cpp SYCL update assessment, /dev/sdc LVM expansion, full re-benchmark

**Scope of this session (4 user requests):**
1. Does LongCat-Flash-Lite-Sparse fit with a decent quant on the B70?
2. Are there useful new SYCL commits in llama.cpp — should we pull?
3. Benchmark ALL models (MoE + dense) at 128K and 256K context.
4. Add the new `/dev/sdc` 500GB disk to the LVM for more model storage.

---

## 1. LongCat-Flash-Lite-Sparse — DOES NOT RUN on B70 today

**Model:** `meituan-longcat/LongCat-Flash-Lite-Sparse` (Meituan)

| Attribute | Value |
|-----------|-------|
| Architecture class | `LongcatCausalLM` (custom) |
| Total parameters | **68.5B** (~69B) |
| Activated parameters / token | **~3B** (MoE) |
| Embedding allocation | **31.4B** (46% of params) — N-gram embedding MoE |
| Native context | 256K (YaRN, `max_position_embeddings` 983040) |
| Attention | **LSA** (LongCat Sparse Attention) — extends DeepSeek Sparse Attention |
| # experts | 256 routed + 128 zero/identity, top-12 routing |
| Hidden size | 3072, 14 layers, 32 heads, MLA |
| MTP layers | 3 (with `dsa_mtp_cli`) |
| License | Meituan LongCat (see HF repo) |
| Files on HF | 26× safetensors × 5.32 GB = **~138 GB** (FP16). **No GGUF exists.** |

### Verdict: CANNOT run on the B70 — three blockers

1. **No GGUF, and no `LongcatCausalLM` arch in llama.cpp.**
   I verified the llama.cpp source (`src/llama-arch.cpp`): supported archs include `deepseek`, `deepseek2`, `deepseek2-ocr`, `deepseek32`, `deepseek4`, `qwen2`, `qwen2moe`, `qwen3`, `qwen3moe`, `qwen35`, `qwen35moe`, `gemma`, `gemma2`, `gemma3`, … but **there is NO `LLM_ARCH_LONGCAT`**. The model's `LongcatCausalLM` has custom layers (`LSA` sparse attention with Streaming-Aware/Cross-Layer/Hierarchical indexing, `zero_expert`, N-gram embedding tables) that have no loader.

2. **The DSv4/Lightning-Indexer ops it resembles are CUDA/Metal only — no SYCL kernels.**
   Recent llama.cpp merged DeepSeek-V4 hyper-connections, the Lightning Indexer, and DSpark sidecar (commits `596a5795b`, `3b5321936`, `0dc74e332`, `1464c62d8`, `fffbcbdb9`, `3581ba0cf`). These implement the *kind* of sparse-attention/index machinery LongCat uses. **But every one of these is `cuda:` or `metal:` only** — there is no `ggml-sycl` implementation of `LIGHTNING_INDEXER`, `HYPER_CONN`, or `DSPARK` ops. grep of `ggml/src/ggml-sycl/` returns nothing for these.

3. **Even the Sparse offload trick targets 24 GB GPU + large RAM, not 32 GB alone.**
   The Sparse variant's selling point is offloading ~30B of N-gram lookup tables to host RAM so the active compute fits in 24 GB VRAM. On the B70 (32 GB VRAM, but **only 30 GB system RAM** with 18 GB already in use), the 31.4B embedding table at Q4 would still need ~15 GB and the 256-expert MoE weights another ~10 GB at Q4 — borderline, and llama.cpp can't load it anyway.

### What would change this
- Meituan (or a third party) shipping a GGUF with a supported arch mapping, AND
- SYCL backend gaining Lightning-Indexer / sparse-attention kernels (currently CUDA/Metal only).

**Recommendation: skip. Re-check in 1–2 months when/if SYCL DSv4 ops land.** The architecture is interesting and would be a strong fit on paper (~3B active params → would be compute-bound and very fast like our other MoE), but the software stack isn't there. Track `ggml/src/ggml-sycl/` for `lightning`/`indexer` files appearing.

---

## 2. llama.cpp SYCL — pull assessment: OPTIONAL, low benefit now

**Current local HEAD:** `a7a6d0d26` — **b10222**, built 2026-08-01 (`build-sycl-0801`).
**Documented baseline in AGENTS.md:** `7af4279f4` — **b9853** (`build-sycl-latest-l0`, 2026-07-01).
So the tree has **already advanced 369 builds** past the documented baseline; we are essentially current.

**`origin/master` is only 20 commits ahead of local HEAD.** Of those 20:

| Commit | Backend | Relevant to us? |
|--------|---------|-----------------|
| `272700b36` sycl: fix classification of iGPUs (#26105) | **SYCL** | Marginal — B70 is a **d**GPU, so the iGPU-classification path doesn't apply. Harmless, no perf impact. |
| `4ed2b13f7` model: MTP support for Qwen3-Next (#25589) | core | Only relevant if we run Qwen3-Next (we don't). |
| `2b63e0610` llama: MTP support for DeepSeek V3.2 (#26457) | core | Only for DSv3.2 models. |
| `1464c62d8` metal: DSv4 Lightning Indexer | metal | Not us. |
| `fffbcbdb9` metal: DSv4 hyper-connections | metal | Not us. |
| `bb4e0e1b3`/`3581ba0cf`/`596a5795b` DSv4/DSpark | cuda/core | CUDA only; SYCL ops absent. |
| `0b14b87d7` server: default port change 8080→9931 notice | server | Heads-up only, no action. |
| rest (CUDA data-races, opencl GLU, metal SILU_BACK, webgpu…) | other | Not us. |

**Verdict: No need to pull right now.** The single SYCL commit fixes iGPU classification (we're a discrete Arc Pro B70, so it's a no-op for us). None of the 20 commits add SYCL kernels or fix SYCL perf/correctness for our archs (`qwen35`, `qwen35moe`, `laguna`).

**When to pull next:** when a commit with `sycl:` in the title touches MoE reorder, attention, or KV-cache quant kernels — or when SYCL DSv4/Lightning-Indexer ops land (which would also unlock LongCat). Set a reminder to check `git log origin/master --oneline | grep sycl` weekly.

---

## 3. /dev/sdc added to LVM — +435 GB for models/quants

The new 500 GB T-FORCE SSD is now part of the `vg_data` volume group and mounted persistently.

**Before:** `/mnt/<models-disk>` (`/dev/sdb3`, raw ext4, **not** LVM) was 84% full (35 GB free); root `/` (`/dev/sdb4`) was 96% full (8.7 GB free). No room for additional quants.

**What was done (all verified):**
```
wipefs -a /dev/sdc                      # clear empty GPT
pvcreate /dev/sdc                       # → new PV
vgextend vg_data /dev/sdc               # add to existing VG
lvcreate -l 100%FREE -n lv_models vg_data  # → 465.76 GB LV
mkfs.ext4 -L models2 /dev/vg_data/lv_models
mount /dev/vg_data/lv_models /mnt/<models-disk>
chown sergio:sergio /mnt/<models-disk>
# fstab entry added (defaults,nofail)
```

**Result:**
```
/dev/mapper/vg_data-lv_models  458G  2.1M  435G  1%  /mnt/<models-disk>
```

**Final LVM state:**
```
VG      PVs                                   VSize     LVs
vg_data /dev/sda3 (49.6G) + /dev/sdc (465.8G) 515.34G   lv_data (49.6G), lv_models (465.8G)
```

**Usage guidance:** `/mnt/<models-disk>` (435 GB free) is the place to download additional quants and model variants. The original `/mnt/<models-disk>/gguf` still holds the production models (102 GB). Both are owned by `sergio`. Consider a symlink `~/models2-cache -> /mnt/<models-disk>` to mirror the existing `~/models-cache -> /mnt/<models-disk>/gguf` convention.

---

## 4. Full re-benchmark — 128K + 256K, all models

**Methodology:** per AGENTS.md §8 — sequential only, warmup discarded (SYCL JIT), 4 diverse prompts (essay/technical/design/comparison), engine decode rate from `timings.predicted_per_second`, cooldown ≤55 °C between tests, power set 3 s before launch. Build **b10222** (`build-sycl-0801`). KV cache **q5_0-q4_1** (production config) for all unless noted.

**Models tested (as actually on disk — note drift from AGENTS.md §5):**

| File | arch | Size | Type |
|------|------|------|------|
| Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf | qwen35moe | 20 GB | MoE 35B-A3B |
| Qwen3.6-35B-A3B-UD-Q5_K_M.gguf | qwen35moe | 24 GB | MoE 35B-A3B |
| ThinkingCap-Qwen3.6-27B-Q4_K_M.gguf | qwen35 | 15 GB | Dense 27B (+MTP) |
| Laguna-S-2.1-UD-IQ2_XXS.gguf | laguna | 34 GB | Dense, IQ2_XXS dynamic quant |

> Note: `dflash-draft-3.6` and `laguna-s-2.1-DFlash-BF16` are speculative-decoder heads (arch `dflash`/`dflash-draft`) and are **not standalone models** — they require a host model. They were excluded.

### Results — b10222 (build-sycl-0801) vs documented baseline (b9853/b10053)

6 of 8 planned tests completed with valid data. The run stopped when loading Laguna-S-2.1 IQ2_XXS (34 GB model) — see crash note below. **No test was retried** to avoid further hardware stress after the reboot.

| Model | Quant | Type | Context | Power | Decode t/s | Prefill t/s | VRAM free | Doc baseline | Δ vs doc |
|-------|-------|------|---------|-------|-----------|-------------|-----------|--------------|----------|
| Qwen3.6-35B-A3B | Q4_K_XL | MoE | **128K** | 150W | **69.8** | 140.5 | 7950 MiB | 68.8 | **+1.5%** |
| Qwen3.6-35B-A3B | Q4_K_XL | MoE | **256K** | 150W | **70.0** | 142.8 | 6103 MiB | 68.8 | **+1.7%** |
| Qwen3.6-35B-A3B | Q5_K_M | MoE | **128K** | 150W | **67.9** | 132.4 | 4061 MiB | 67.1 | **+1.2%** |
| Qwen3.6-35B-A3B | Q5_K_M | MoE | **256K** | 150W | **67.7** | 144.3 | 2197 MiB | 67.1 | **+0.9%** |
| ThinkingCap Qwen3.6-27B | Q4_K_M | Dense+MTP | **128K** | 180W | **29.0** | 76.3 | 6319 MiB | ~30.4 (230W) | in line |
| ThinkingCap Qwen3.6-27B | Q4_K_M | Dense+MTP | **256K** | 180W | **29.4** | 63.0 | **1135 MiB** | n/a (new) | new data |
| Laguna-S-2.1 | IQ2_XXS | Dense | 128K | 150W | **CRASH** | — | — | — | did not run |
| Laguna-S-2.1 | IQ2_XXS | Dense | 256K | 150W | **CRASH** | — | — | — | did not run |

*(Decode = `timings.predicted_per_second`, avg of rounds R1–R4 after a discarded warmup. All with `-fa on -ctk q5_0 -ctv q4_1 -ngl 99 -ncmoe 0 -t 8 --no-mmap`.)*

### Key findings

1. **Context scaling penalty = ZERO, re-confirmed on b10222.** Qwen 35B Q4_K_XL is *faster* at 256K (70.0) than 128K (69.8) — within noise. Q5_K_M: 67.9 → 67.7. This holds the documented property that q5_0-q4_1 KV cache keeps decode speed flat from 64K through 256K.

2. **b10222 is ~1–1.7% faster than the documented b9853 on MoE models.** Small but real and consistent across both quants. The newer build carries no regression; the production server (`llama-profile.service`) is already running `build-sycl-0801`, so this is the live configuration.

3. **27B dense + MTP at 256K works but is VRAM-tight (1135 MiB headroom).** It completed all 4 rounds at 29.4 t/s without error, but 1135 MiB is the danger zone — any spike in activation memory would OOM. This is a **Q4_K_M** (15 GB) model; the documented Q5_K_M 27B at 200K is the safer ceiling. Do **not** run 27B Q4 at 256K in production alongside vision/mmproj without watching VRAM.

4. **Dense 27B is bandwidth-bound: 180W → 29 t/s, vs MoE 35B compute-bound at 150W → 68-70 t/s.** Power scaling matters for dense, not MoE — reconfirmed.

### ⚠️ CRASH NOTE — Laguna-S-2.1 IQ2_XXS caused a system reboot

The benchmark run crashed the **entire PC** (hard reboot) when it tried to load `Laguna-S-2.1-UD-IQ2_XXS.gguf` (34 GB file, arch `laguna`). The system came back up cleanly after 1 minute with all LVM volumes and the production service auto-restored.

**Root cause (most likely):** a 34 GB model cannot fit in 32 GB VRAM. Even though IQ2_XXS is a 2-bit dynamic quant, the file is 34 GB on disk and the GGUF reported `visible_size: 32656 MiB`. With weights + KV cache + compute overhead, this overflows VRAM and the xe driver / GPU hit a fatal error that took the whole machine down (not a cleanable OOM).

**Action / rule:** Do **not** attempt to load any single GGUF larger than ~30 GB into VRAM on this B70, even with `--no-mmap`, unless using `-ngl` partial offload to keep some layers on CPU. Laguna-S-2.1 IQ2_XXS should either be run with reduced `-ngl` (CPU offload of tail layers) or removed from the active model set. The crash is reproducible by design — it was not retried here to protect the hardware.

Full raw results with per-round timings and VRAM: `results/all-models-128k-256k-20260803-1452.txt`.
