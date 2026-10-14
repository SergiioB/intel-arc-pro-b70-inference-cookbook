# B70 Optimization Deep Audit — 2026-07-21

## CRITICAL FINDING: Production binary is 200 commits behind

Your production llama.cpp binary is build b9853, commit `7af4279f4` from **2026-07-01**.
Your source checkout is at `0dc74e332` from **2026-07-17** — 200 commits ahead.

**The most impactful SYCL/Battlemage optimizations merged AFTER your build date:**

| Commit | Date | What it does | Impact on your models |
|--------|------|-------------|----------------------|
| `32b741c33` | Jul 15 | **Flash Attention via oneDNN XMX** (Battlemage-gated) | Qwen27 Q8 prefill: **x1.21 at pp512, x4.26 at pp80K**. Your 32K prefill bottleneck (305 t/s) could improve dramatically |
| `c1063ac9d` | Jul 14 | **FA vec threads = 256 for Battlemage** (was 128) | Better FA kernel occupancy on Xe2 |
| `efb3036c1` | Jul 13 | **Fused top-k MoE** (`GGML_SYCL_ENABLE_FUSION=1` default) | Fewer kernel launches for Qwen35/Ornith MoE routing |
| `d3fba0c79` | Jul 15 | **Fix get_rows Q2_K, Q4_K, Q5_K** | Correctness fix for your exact quant types |
| `d20908615` | Jul 7 | **K_QUANTS_PER_ITERATION=1 on DMMV** | Performance change that exposed row/Q5_K correctness bugs later fixed by PR #25690 |
| `0e148a573` | Jul 15 | **USM buffer size increase** | Stability fix for system allocations |
| `26145b3db` | Jul 7 | **Env var rename** | `GGML_SYCL_DISABLE_OPT` → `GGML_SYCL_ENABLE_OPT` (default 1) |
| `3d4cbdf18` | Jul 7 | **AOT double type fix** | Fixes AOT compilation issues |

### What this means concretely

Your profiles already pass `--flash-attn on`, but your **old binary doesn't have the oneDNN XMX FA path**. You're running the pre-July FA kernel. The new one is Battlemage-specific and claims:
- 32% prefill improvement at pp512
- **4.26x prefill at 80K context**
- Your measured 32K prefill was 305 t/s (down from 620 at 4K). This is exactly the bottleneck the new FA targets.

The fused top-k MoE is also new and directly relevant to your Qwen35/Ornith profiles.

## Current measured performance (production b9853)

| Model | TG t/s | PP4K t/s | Power | Context |
|-------|-------:|--------:|------:|--------:|
| Ornith 35B Q5_K_M | **69.27** | 1,726 | 150W | 256K |
| Qwen35 Q4_K_XL | **62.76** | 1,682 | 150W | 256K |
| Qwen35 Q5_K_M | **61.54** | 1,690 | 150W | 256K |
| Qwen27 MTP Q5 (180W live) | **26.11** | 620 | 180W | 128K |
| Qwen27 MTP Q5 (165W) | **25.07** | 613 | 165W | 200K |

### Known bottlenecks
1. **Long-context prefill**: 620 t/s at 4K → 305 t/s at 32K (50.77% drop). 107s TTFT at 32K.
2. **Reasoning exhaustion**: unlimited `--reasoning-budget -1` causes empty content on short prompts.
3. **Vision transcription**: misreads small text, never reaches final content under reasoning.
4. **Storage**: /mnt/<models-disk> was 98% (now 89% after Gemma cleanup). / is 93%.

## Recommended action plan (ranked by impact)

### 1. REBUILD llama.cpp from current source (HIGHEST IMPACT)
```bash
cd ~/llama.cpp
# Source is already at 0dc74e332 (Jul 17)
# Build to a NEW directory to preserve b9853
cmake -B build-sycl-0717 \
  -DGGML_SYCL=ON -DGGML_SYCL_F16=ON \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl-0717 -j4
```
Then A/B test the new binary against b9853 with identical profiles.

**Expected gains:**
- Prefill: potentially 30-400% improvement at long context (the 4.26x claim is at 80K)
- MoE decode: small improvement from fused top-k (fewer kernel launches)
- Stability: get_rows fix, USM fix, DMMV multi-turn fix

**Risk:** Low. Build to separate directory. Production b9853 binary stays untouched. Rollback = point service back to old binary.

### 2. Test new env vars with rebuilt binary
After rebuild, test:
- `GGML_SYCL_FA_ONEDNN=1` (default, should be on)
- `GGML_SYCL_ENABLE_FUSION=1` (default, fused MoE)
- `GGML_SYCL_ENABLE_GRAPH=1` (command graphs, default OFF — test carefully)
- `GGML_SYCL_ENABLE_OPT=1` (default, replaces old DISABLE_OPT)

### 3. Fix reasoning budget (no rebuild needed)
Change profiles from `--reasoning-budget -1` to a bounded value (e.g., 2048) or use `--reasoning off` for non-reasoning tasks. This fixes the empty-content bug immediately.

### 4. Storage cleanup
- Archive P4.3A experimental GGUFs (22.36 GB each) to /mnt/<other-disk>
- This frees ~45 GB on /mnt/<models-disk>

### 5. Second B70 (if purchased)
- Use as independent server, NOT layer-split or TP
- Best use: Qwen3-Coder-Next 80B-A3B Q4 (~43 t/s on 2× B70 with CPU-direct x8/x8)
- Your B450 second slot is chipset x4 — adequate for independent servers, poor for TP

## What the community is doing (July 2026)

### Best single-B70 results found
- **70.54 t/s tg128** on Qwen3.6-35B-A3B Q4_K_M (SYCL, F16, q8_0 KV, FA on, threads=1, cache-ram 0)
- **977 t/s pp512** same config
- **81.12 t/s tg128** on same model via Vulkan with Mesa 26.2-dev NV_coopmat2
- **100.75 t/s tg128** on Windows Vulkan (Intel driver 32.0.101.8737)
- **63 t/s sustained** via Docker with seq_rm hybrid memory patch + CPU pinning

### Key community flags not in your setup
- `--cache-ram 0` (force all KV in VRAM, no host spill)
- `--threads 1` with CPU pinning (zero contention for full-offload)
- `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`
- `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` (you have =0)

### Multi-GPU findings
- Layer split: capacity only, ~4% slower for models that fit one card
- vLLM TP=4: 965 t/s aggregate at 64 concurrent users (WRX80 platform)
- vLLM TP=2: BCS reset bugs on some platforms, requires specific env stack
- MachX engine: claims 63 t/s decode on Qwen3.6-35B Q8 across 2× B70 (1.49× vs llama.cpp layer-split)
- **No credible evidence of >80 t/s single-stream decode on one B70 with SYCL**

### What NOT to do
- Don't switch to Vulkan based on Windows numbers (different driver stack)
- Don't trust "200 tok/s" Reddit titles (aggregate concurrency, not per-user)
- Don't run vLLM Docker builds on 30 GB RAM host
- Don't apply `GGML_SYCL_DISABLE_OPT=1` globally (renamed to ENABLE_OPT, default on)

## A/B benchmark results (2026-07-21, executed)

Model: Qwen3.6-35B-A3B-UD-Q4_K_XL, SYCL0, `-ngl 99`, f16 KV (llama-bench default), 3 runs unless noted.

| Test | OLD b9853 (7af4279f4) | NEW b10053 (0dc74e332) | Δ |
|------|----------------------:|-----------------------:|--:|
| pp512 | 1043.5 t/s | 1039.3 t/s | -0.4% (noise) |
| tg128 | 70.56 t/s | **72.57 t/s** | **+2.8%** |
| pp8192 FA on | 936.9 t/s | **1028.6 t/s** | **+9.8%** |
| pp32768 FA on | 724.1 t/s | **967.5 t/s** | **+33.6%** |

### Dense Qwen27 MTP Q5_K_M (raw, no MTP in llama-bench)

| Test | OLD b9853 | NEW b10053 | Δ |
|------|----------:|-----------:|--:|
| pp512 | 513.2 t/s | **849.0 t/s** | **+65.4%** |
| tg128 | 19.62 t/s | **20.45 t/s** | +4.2% |
| pp8192 FA on | 551.9 t/s | **816.5 t/s** | **+47.9%** |
| pp32768 FA on | 298.8 t/s | **754.0 t/s** | **+152.4%** |

The 32K prefill bottleneck (measured 305 t/s live, 107s TTFT) should now be ~750 t/s (~43s TTFT).

### Additional MoE tests (new binary only)

| Test | Fusion OFF | Fusion ON | Δ |
|------|----------:|----------:|--:|
| pp512 | 648.2 t/s | 650.9 t/s | +0.4% (noise) |
| tg128 | 67.68 t/s | **69.55 t/s** | **+2.8%** |

Fusion is default ON in b10053. `--threads 1` vs `-t 8`: noise (69.11 vs 68.82). Keeping `-t 8`.

### Profile fixes applied
- MoE profiles (qwen35-q5, qwen35-q4, ornith35): `reasoning: off` (was `auto` with budget 256, caused empty content).
- Dense profiles (qwen27 variants): `reasoning: auto`, budget 256 (was -1, caused empty content on short prompts).
- All profiles: `IMMEDIATE_COMMANDLISTS=1`, `RELAXED_ALLOCATION_LIMITS=1` in run-profile env.

Build flags used for new binary:
```bash
cmake -B build-sycl-0717 \
  -DGGML_SYCL=ON -DGGML_SYCL_F16=ON \
  -DGGML_SYCL_DEVICE_ARCH=bmg-g31 \
  -DGGML_SYCL_DNN=ON \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_BUILD_TYPE=Release
```

### QUALITY GATE: b10053 NO-GO for production

Under production-like flags (`ctx=262144`, `-b 8192 -ub 4096`, FA on, `ctk=q8_0 ctv=q4_1`):

| Binary | Short "2+2" | Count 1..20 | Verdict |
|--------|-------------|-------------|---------|
| **NEW b10053** (`0dc74e332`) | `Four` OK | **garbage tokens** @ ~66 t/s | **NO-GO** |
| **OLD b9853** (`7af4279f4`) | `Four` OK | `1..20` exact @ ~63 t/s | **PASS** |

Evidence:
- New binary chat longer generation → PEG parse failure: `The model produced output that does not match the expected peg-native format`
- Short answers can look correct and mask the bug — do not use single-word checks as quality gate
- FA_ONEDNN=0 / FUSION=0 did not fix the long-generation garbage under smaller-ctx retests consistently; production-like path remains broken on b10053

**Production decision (2026-07-21):** rolled back to b9853.
- `run-profile` exec: `~/llama.cpp/build-sycl-latest-l0/bin/llama-server`
- Keep env fixes already proven safe on b9853: `IMMEDIATE_COMMANDLISTS=1`, `RELAXED_ALLOCATION_LIMITS=1`
- Keep profile fixes: MoE `reasoning=off`; dense `reasoning=auto` + budget `256`
- Keep new binary for research only: `~/llama.cpp/build-sycl-0717/`
- Backups: `run-profile.bak-pre-0717`, `profiles.json.bak-pre-0717`

Next research gate before any re-promotion of b10053+:
1. Exact multi-token quality A/B (count/list/code, temp=0, ≥40 tokens)
2. MoE issue #25455 / get_rows / FA+quant KV interactions
3. Only then re-measure prefill gains under quality-safe flags

## Subagent research findings (deleg_b57cdb27, 2026-07-21)

### CRITICAL: MTP on SYCL — model-dependent
- **Dense models (Qwen27): MTP IS a win.** +37-47% decode (15-25 → 31-37 t/s). Your Qwen27 MTP profiles are correct.
- **MoE models (Qwen35/Ornith): MTP is a NET LOSS.** -21% to -43% slower despite 100% draft acceptance. Root cause: SYCL kernel dispatch overhead 100-500µs vs ~5µs on CUDA. Your MoE profiles correctly omit MTP.
- Source: https://github.com/ggml-org/llama.cpp/issues/23533

### CRITICAL: SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS
- Your current: `=0`. Intel docs confirm B-series L0 V2 adapter ONLY supports immediate command lists.
- **Setting `=0` may be forcing a fallback path.** Change to `=1` or unset after rebuild.
- Source: Intel L0 docs, lemongravy.me

### CRITICAL: UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
- Not set in your profiles. Level Zero caps single allocations at ~4GB by default.
- Your 256K context KV caches likely exceed this. **Add `=1` to all profiles.**
- Source: lemongravy.me, sparrowzhou/arc-pro-b70-ubuntu-gpu-speedup-bugfixes

### XMX Flash Attention (PR #25222) — KV cache requirement
- The new oneDNN XMX FA path **requires f16 KV cache** (`--cache-type-k f16 --cache-type-v f16`).
- Your profiles use q8_0/q4_1 KV. To use XMX FA, you'd need to switch to f16 KV (more VRAM).
- **PR #25312 (OPEN, not merged)** extends XMX FA to quantized KV: 2.48x prefill at 32K with Q8_0 KV. Watch this.
- Source: PR #25222, PR #25312

### Vulkan with Mesa 26.1+ — now competitive
- Mesa 26.1+ with NV_coopmat2: Vulkan decode **76 t/s** single-stream (vs SYCL 70-76).
- 8-stream aggregate: Vulkan 170 t/s vs SYCL 100 t/s. Vulkan scales near-linearly; SYCL plateaus.
- **Coopmat stability**: causes TDR/device-lost on MoE (MUL_MAT_ID). Fix: `GGML_VK_DISABLE_COOPMAT=1` if crashes.
- Your kernel 7.0 + Mesa version needs checking. Ubuntu 26.04 may ship Mesa 25.x.
- Source: jonathanmann.tech, issue #22034

### MoE prefill bug (issue #25455)
- Active bug: MUL_MAT_ID prefill produces wrong results on B70. 28/792 test failures.
- Could affect your Qwen35/Ornith prefill correctness. Monitor this issue.
- Source: https://github.com/ggml-org/llama.cpp/issues/25455

### oneAPI 2026.1 (released July 7)
- ABI/API breaking from 2026.0. Must recompile all SYCL binaries.
- Enhanced free-function kernels, device-type queries.
- **Rebuild with oneAPI 2026.1** + AOT (`-DGGML_SYCL_DEVICE_ARCH=bmg-g31`).
- Source: Intel release notes

### Linux 7.1 kernel
- Phoronix-confirmed free B70 performance gains on BMG-G31. Xe2 driver scheduling improvements.
- Upgrade when stable.
- Source: phoronix.com/review/linux-71-arc-pro-b70

### B450 x4 slot — confirmed severe bottleneck
- PCIe 3.0 x4 = ~3.9 GB/s. All working dual-B70 setups use CPU-direct x8/x8 minimum.
- 23% throughput collapse measured on 70B Q4_K_M in x4 slot.
- **Not recommended for multi-GPU. Independent servers only.**
- Source: craftrigs.com, PMZFX benchmarks

### Recommended env vars (consolidated best practice)
```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export ZES_ENABLE_SYSMAN=1
export SYCL_CACHE_PERSISTENT=0
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1  # CHANGED from 0
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1          # NEW
# No GGML_SYCL disable variable in the validated b10053+#25690 production path.
# Do NOT set: ZEX_NUMBER_OF_CCS, SYCL_UR_USE_LEVEL_ZERO_V2
```

### Recommended build flags
```bash
cmake -B build-sycl-0717 \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_F16=ON \
  -DGGML_SYCL_DEVICE_ARCH=bmg-g31 \
  -DGGML_SYCL_DNN=ON \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_BUILD_TYPE=Release
```

### Full subagent reports
- Task 1 (SYCL config): `~/.hermes/cache/delegation/subagent-summary-0-20260721_140426_365097.txt`
- Task 2 (multi-GPU): `~/.hermes/cache/delegation/subagent-summary-1-20260721_140426_365473.txt`
- Task 3 (optimizations): `~/.hermes/cache/delegation/subagent-summary-2-20260721_140426_365756.txt`

## Authoritative correction and production promotion — 2026-07-22

### Verdict

**b10053 + upstream PR #25690 passed and replaced the temporary MoE-reorder guard in production.**

The original b10053 commit predates the merged fix. PR #25690 (`0bd0ec609`) modifies only `dmmv.cpp` (+72/−63) and fixes reordered Q5_K plus K-quant row selection. No `convert.cpp` patch or dual weight layout is required for this exact failure.

### RED → GREEN

- RED, original b10053 with reorder enabled: request 1 correct; request 2 corrupted; assertion failed.
- GREEN, b10053+#25690 with reorder enabled: Q5 26/26 requests passed.
- Q4_K_XL validation: 10/10 requests passed.
- Post-promotion production validation: 6/6 passed.
- Extended production soak: **100/100 passed**, mean 69.979 t/s, median 70.048, range 69.638–70.142.

### Backend gates

- `MUL_MAT`: 970/970.
- `MUL_MAT_ID`: 790/790.
- No Xe reset, GPU fault, hang, or OOM in captured kernel logs.

### Matched 150 W A/B

Identical Q5_K_M model, no-mmap, batch 8192, ubatch 4096, FA on, q8_0/q4_1 KV, five repetitions:

| Test | Guard control | PR #25690 | Delta |
|---|---:|---:|---:|
| pp8192 | 1461.104 t/s | 1454.926 t/s | −0.42% |
| pp32768 | 942.516 t/s | 942.360 t/s | −0.02% |
| tg128 | 61.008 t/s | **67.218 t/s** | **+10.18%** |

Matched 20-request real-server decode:

- Guard: 63.252 t/s mean, 95% CI 63.192–63.313.
- PR #25690: 70.156 t/s mean, 95% CI 69.887–70.426.
- Gain: approximately 10.92%.

### Storage/design analysis

- Model tensor payload: 24.629 GiB.
- Affected expert weights: 21.338 GiB.
- A dual layout would need 45.967 GiB before KV/work buffers; impossible on 32 GiB.
- Restore+repack per request has a 301–362 ms bandwidth lower bound and would collapse tg128 to roughly 60.3–62.0 t/s.
- Corrected reordered kernels are the only practical zero-extra-VRAM solution.

### Production and rollback

Production points to the isolated b10053+#25690 binary. `GGML_SYCL_DISABLE_MOE_REORDER` is absent, so expert reorder remains enabled. The original b10053+guard and b9853 remain available as ordered fallbacks. Patch, hashes, scripts, JSON, and raw logs are preserved in the B70 lab workspace.
