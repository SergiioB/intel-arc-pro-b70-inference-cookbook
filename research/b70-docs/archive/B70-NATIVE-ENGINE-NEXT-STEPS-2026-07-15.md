# B70 Native Engine: exact next steps and revised performance plan

Date: 2026-07-15  
Target: one Intel Arc Pro B70 32 GB, single-user local LLM latency first  
Primary models: Qwen3.6-35B-A3B and Ornith-1.0-35B  
Stable control: llama.cpp SYCL b9853, commit `7af4279f4`  
Prior work:

- `B70-MEGAKERNEL-ARCHITECTURE-2026-07-14.md`
- `b70-native-engine-investigation-2026-07-14.md`

## 0. Implementation status after P4.2

P0, P1.1/P1.2, P2, P3, P5, P4.0, P4.1, and the P4.2 exact routed-plus-shared MoE block have now been executed. The current authoritative continuation documents are in the B70 research repository:

- `~/b70-lab/b70-engine/docs/HANDOFF-2026-07-15.md`
- `~/b70-lab/b70-engine/docs/EXPERIMENT_PROTOCOL.md`
- `~/b70-lab/b70-engine/docs/experiments/2026-07-15-p0-fp8-aot.md`
- `~/b70-lab/b70-engine/docs/experiments/2026-07-15-p1-q8-ab.md`
- `~/b70-lab/b70-engine/docs/experiments/2026-07-15-p1-1-q8-correctness.md`
- `~/b70-lab/b70-engine/docs/experiments/2026-07-15-p1-2-q8-repeats.md`
- `~/b70-lab/b70-engine/docs/experiments/2026-07-15-p2-shared-q8-projections.md`
- `~/b70-lab/b70-engine/docs/experiments/2026-07-15-p3-routing-d2m.md`
- `~/b70-lab/b70-engine/docs/experiments/2026-07-15-p5-logits-boundary-roofline.md`
- `~/b70-lab/b70-engine/docs/experiments/2026-07-15-p4-0-gdn-fp32.md`
- `~/b70-lab/b70-engine/docs/experiments/2026-07-15-p4-1-gdn-attention.md`
- `~/b70-lab/b70-engine/docs/experiments/2026-07-15-p4-2-moe-block.md`
- `~/b70-lab/b70-engine/results/raw/p1-ab/20260715T193036Z/`

Current decision:

- P0 FP8: PASS, AOT, zero spills, 485.52 GB/s direct, 91.61% of the local D2D roof.
- P1 Q8: PARTIAL PASS; deterministic random correctness, AOT and zero-spill gates passed.
- P1.1 correctness: PASS; literal 34/36-byte ABIs, bit-exact AoS/SoA round trips, independent double oracles, required adversarial vectors, poisoned Q8_1 `s`, ULP reporting, and disclosed forward-error bounds passed under CTest and ASan/UBSan.
- P1.2 repeated A/B: FAIL promotion gate; the three provisional direct candidates measured 1.0921x, 1.0736x, and 1.0903x paired device-p50 speedup, all below 1.10x.
- P2 shared-input Q8: FAIL promotion gate; shared quantization reached 1.0211x and shared quantization plus one two-matrix GEMV reached 1.0576x device p50. The projected saving is only 0.091 ms/token.
- P3 routing/D2M: CLOSED; the exact model reaches the fused device-resident `MUL_MAT_ID` return. The one D2M/token is the F32 `[248320]` logits tensor, exactly 993,280 bytes, not expert ids.
- P5 logits boundary: STOP; perfect zero-cost removal of the entire D2M is bounded at 1.0891x/75.74 tok/s from stable device time and 78.78 tok/s from the corrected control, both below 82 tok/s.
- P4.0 FP32 GDN: COMPONENT PASS / PROGRAM FAIL; 1.3021x device p50, exact 4,096-update state, BMG AOT/zero spills, but only 0.09063 ms/token (0.63%) projected saving.
- P4.1 Q8 GDN attention: COMPONENT PASS / PROGRAM FAIL; 1.2353x device and 1.2432x direct-host p50, exact 4,096-update FP32 state, BMG AOT/zero spills, but only 0.761991 ms/token (5.30%) projected saving.
- P4.2 routed-plus-shared MoE: PROJECTION PASS; exact IDs/weights, 1.5964x direct p50, and 1.929159 ms/token projected saving. P4.1+P4.2 projects 18.72% latency reduction, 85.56 tok/s from the stable profile or 89.46 tok/s from the corrected control.
- `[2048,512]` improved at direct p50 but regressed to 0.7127x at p99; `[512,2048]` remained at 0.8662x direct p50. Both remain fallback/graph-only candidates.
- Weighted selected-plus-fallback standalone device reduction: 9.86%, below the 15% projected end-to-end threshold.
- No Q8 kernel has been integrated into llama.cpp or production; direct integration is stopped.

P1 and P2 direct/fusion gates failed. Do not integrate those candidates. P3 disproved the assumed expert-routing wait, P5 failed its hard ceiling, and P4.0/P4.1 proved valid but individually insufficient GDN components. P4.2 closes the standalone projection gap at 18.72%. P4.3 must now measure actual model tokens/s through guarded integration, five-run A/B, differential correctness, integrated unitrace, and soak; projections are not a final promotion result. This status section supersedes the historical host-state narrative below where they conflict.

## 1. Revised verdict

The best practical route to “beast mode” on one B70 is:

> Incubate a Qwen35/Ornith-specific fast path inside a pinned llama.cpp SYCL fork, execute a fixed model plan through a reusable Level Zero command list, and replace the expensive graph islands with BMG-AOT ESIMD kernels. Extract a standalone runtime only after this path proves at least 1.25x correct end-to-end decode speed.

Do not begin with a literal 40-layer cooperative mega-kernel. The production design should behave like a mega-kernel from the host’s perspective, but internally remain approximately five globally ordered kernels per layer.

The revised priority is:

1. Restore correctness on experimental llama.cpp branches.
2. Measure actual per-kernel device time, host gaps, launch count, spills and memory traffic.
3. Capture/replay the fixed token graph so the host submits once per token.
4. Fuse each Qwen block into model-specific kernel islands.
5. Optimize the dominant dense Q8 paths, not only the MoE expert path.
6. Add a BMG-native mixed W4/W6 pack only after exact-GGUF kernels are correct.
7. Revisit MTP only after ordinary decode is fast and state rollback is implemented on-device.

The critical new conclusion is that the MoE experts are not the majority of active bytes. The real Q4_K_XL tensor manifest shows that GDN projections, full-attention projections and the LM head account for about 70% of active weight traffic. A fast expert kernel alone cannot produce the desired step change.

## 2. Current host state

The remote B70 host is reachable and normal production inference is restored.

- `llama-profile.service`: active.
- `dev.xe.observation_paranoid`: 1 after profiling.
- Stable b9853 source commit: `7af4279f4579094cbe121cccb3c28357396e55d0`.
- Corrected experimental PR #25690 commit: `45ee550fc64f875291a0d3b0f26acf0ea4a8aad8`.
- Standalone research code commit before the P1 documentation update: `33f138394e208f1ef991258a9f1d49a4814b95ab`.

The stable worktree has four pre-existing local modifications unrelated to this research. Do not reset, stage, commit, or overwrite them. The PR #25690 worktree has one uncommitted test-only change adding seven exact Q8 fixtures to `tests/test-backend-ops.cpp`; its perf CSV did not expose usable timing rows.

Use `~/b70-lab/b70-engine/docs/HANDOFF-2026-07-15.md` as the live operational state. Continue diagnosing and recording host/GPU health before every heavy benchmark, and restore production service through a trap-protected transaction.

## 3. Important correctness change discovered today

llama.cpp PR #25063 was merged on 2026-07-07. It changed the SYCL DMMV path to `K_QUANTS_PER_ITERATION=1` and improved B70 Q4_K decode substantially, but later multi-turn corruption was traced to an incorrect row calculation. Q5_K also had a half-block processing bug.

PR #25690 opened on 2026-07-14 and contains four fixes:

- Correct the row calculation for `K_QUANTS_PER_ITERATION=1`.
- Use the KQPI setting in the non-reordered Q5_K kernel.
- Process the missing second half of reordered Q5_K blocks.
- Correct several row guards.

Status at research time: open, mergeable, not yet merged.

Consequences:

- The preserved b9853 daily driver predates #25063 and remains the safe control.
- Never benchmark a current-master MoE build for promotion unless #25690 is included.
- Run `MUL_MAT`, `MUL_MAT_ID`, Q4_K/Q5_K multi-turn and deterministic output gates before reading performance results.
- Do not use `GGML_SYCL_ENABLE_OPT=0` as the final fix. It is only a diagnostic fallback.

## 4. Exact model traffic from the real Q4_K_XL GGUF manifest

I downloaded only the first 16 MiB of the public GGUF, expanded it as a sparse file and parsed its 733 tensor descriptors. This avoids downloading the complete 22.36 GB artifact while giving exact tensor shapes and quantization types.

Artifact:

- Repository: `unsloth/Qwen3.6-35B-A3B-GGUF`
- File: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`
- Size: 22,360,456,160 bytes, 20.82 GiB
- LFS SHA-256: `707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450`
- GGUF architecture: `qwen35moe`; Qwen3.6 reuses the Qwen3.5-MoE architecture implementation.

### 4.1 Active model dimensions

- Hidden size: 2,048
- Layers: 40
- GDN layers: 30
- Full-attention layers: 10
- Routed experts: 256
- Selected routed experts: 8
- Shared experts: 1
- Expert intermediate size: 512
- Vocabulary: 248,320
- GDN key heads: 16 x 128
- GDN value heads: 32 x 128

Exact active parameter count per generated token, excluding the one-row embedding lookup:

- Routed selected experts: 1,006,632,960
- Shared expert: 125,829,120
- Router and shared gate: 21,053,440
- GDN projections and parameters: 1,011,553,920
- Full attention projections: 272,634,880
- LM head: 508,559,360
- Layer and final norms: 165,888
- Total: 2,946,429,568 active parameters

### 4.2 Actual active weight bytes in Q4_K_XL

| Subsystem | Active bytes/token | Share |
|---|---:|---:|
| GDN projections and parameters | 1,089.477 MB | 39.6% |
| Selected routed experts | 614.728 MB | 22.3% |
| LM head | 540.344 MB | 19.6% |
| Full-attention projections | 289.772 MB | 10.5% |
| Shared expert | 134.021 MB | 4.9% |
| Router | 83.886 MB | 3.0% |
| Norms and embedding lookup | 0.338 MB | negligible |
| **Total active weights** | **2,752.566 MB** | **100%** |

The model is not “uniform Q4.” The XL quant uses:

- Q4_K and Q5_K for most selected expert matrices, plus a small Q6_K portion.
- Q8_0 for the GDN, full attention, shared expert and LM head.
- F32 for the router and several GDN parameters.

By quant type, active bytes are approximately:

- Q8_0: 2.033 GB
- Q4_K: 0.368 GB
- Q5_K: 0.219 GB
- F32: 0.105 GB
- Q6_K: 0.028 GB

This changes the optimization order. The dominant target is the Q8_0 dense projection path plus launch elimination, not only Q4/Q5 expert GEMV.

### 4.3 Mutable-state and KV traffic

GDN recurrent state per layer:

`32 value heads x 128 value dimension x 128 key dimension = 524,288 elements`

Across 30 GDN layers:

- 15,728,640 elements
- FP32 read + write: 125.829 MB/token
- FP16 read + write: 62.915 MB/token

The official configuration requests FP32 Mamba/SSM state. Use FP32 as the correctness reference. Intel’s current `llm-scaler` sequential ESIMD donor uses FP16 state, so it must not be copied unchanged into the reference path.

Convolution-state read/write adds approximately 2.95 MB/token.

With daily q8_0 K and q4_1 V KV:

- Ten full-attention layers
- 512 K elements and 512 V elements per prior token per full-attention layer
- Q8_0 K: 34/32 bytes per element
- Q4_1 V: 20/32 bytes per element
- Decode reads approximately 8,640 bytes per prior context token

KV traffic per generated token is therefore approximately:

- 16K context: 141.6 MB
- 32K context: 283.1 MB
- 80K context: 691.2 MB
- 128K context: 1,132.5 MB

### 4.4 Lower-bound traffic roofline

Using exact active Q4_K_XL weight bytes, FP32 GDN state and q8_0/q4_1 KV:

| Context | Lower-bound bytes/token | 205 GB/s | 316 GB/s | 507 GB/s | 574 GB/s | 608 GB/s |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.881 GB | 71.1 t/s | 109.7 | 176.0 | 199.2 | 211.0 |
| 16K | 3.023 GB | 67.8 t/s | 104.5 | 167.7 | 189.9 | 201.1 |
| 32K | 3.164 GB | 64.8 t/s | 99.9 | 160.2 | 181.4 | 192.1 |
| 80K | 3.573 GB | 57.4 t/s | 88.5 | 141.9 | 160.7 | 170.2 |
| 128K | 4.014 GB | 51.1 t/s | 78.7 | 126.3 | 143.0 | 151.5 |

These are not performance promises. They exclude redundant reads, cache misses, intermediate traffic, device dispatch latency, synchronization and imperfect dequantization. They are a ceiling and diagnostic model.

At roughly 68 t/s, the historical baseline corresponds to only about 196–205 GB/s of lower-bound traffic, depending on starting context. That leaves substantial headroom, but the missing time can be launch gaps, repeated traffic, vector inefficiency, synchronization or state work. Unitrace must separate them.

## 5. Mathematical impact ranking

Historical single-stream latency at 68 t/s is approximately 14.71 ms/token.

At 16K context, the 507 GB/s lower-bound traffic time is approximately 5.96 ms/token. The difference, approximately 8.75 ms/token, is the maximum budget available to remove through better kernels, fewer launches, less redundant movement and graph replay. It is not all host overhead.

### 5.1 Why expert-only optimization is insufficient

The selected expert weights are 0.615 GB/token.

If their effective bandwidth improves from the donor-documented 316 GB/s DPAS path to 574 GB/s wide-load ESIMD:

- Old idealized expert time: 1.95 ms
- New idealized expert time: 1.07 ms
- Saving: 0.88 ms/token
- End-to-end change if everything else is unchanged: about 68 to 72 t/s

The expert path remains important, especially because fusion removes launches, but it cannot alone deliver 100+ t/s.

### 5.2 Dominant dense Q8 target

GDN + full attention + LM head are approximately 1.920 GB/token, mostly Q8_0. They represent about 70% of active weight bytes.

Highest-impact kernel work is therefore:

1. BMG wide-load Q8_0 GEMV for fixed Qwen shapes.
2. Residual/RMSNorm/multi-projection fusion around those GEMVs.
3. GDN projection and recurrence fusion.
4. LM-head GEMV plus on-device top-k/sampling.
5. Command-list replay to eliminate host gaps between these phases.

### 5.3 Native mixed quantization options

Approximate active-weight traffic after offline repacking:

| Pack | Active weights/token | Saving vs current | Risk |
|---|---:|---:|---|
| Current Q4_K_XL manifest | 2.753 GB | reference | known quality |
| Routed W4 group-128; dense/shared/LM W6; router F32 | 2.177 GB | 20.9% | medium |
| Same, router FP16 | about 2.135 GB | 22.4% | medium |
| Everything W4 group-128 | 1.519 GB | 44.8% | high |

The best first native pack is mixed W4/W6, not all-W4. Preserve sensitive dense projections and LM head at W6 initially. Calibrate from official BF16 weights with a reproducible quantizer; do not blindly transcode the GGUF and call it equivalent.

## 6. Best engine architecture

### 6.1 Incubator

Use a pinned llama.cpp SYCL fork because it already provides:

- GGUF metadata and tensor loading
- tokenizer and chat templates
- KV and recurrent-state lifecycle
- Qwen/Ornith graph construction
- MTP support
- sampling and server
- backend correctness tests
- CPU/Vulkan/SYCL controls

Add one explicitly gated path:

`--b70-native-qwen35`

Required controls:

- Build flag to compile the experimental backend.
- Runtime kill switch.
- Automatic fallback to ordinary ggml-SYCL for unsupported shapes.
- Per-layer debug mode to compare outputs and state.
- No change to the daily binary or source tree.

### 6.2 Scheduler

The final fast path should use:

- Stable VRAM addresses for weights, hidden buffers, KV, convolution and GDN state.
- A pointer table generated once at model load.
- BMG AOT modules compiled for the installed toolchain/device target.
- One in-order Level Zero-backed SYCL queue.
- A fixed command graph finalized once and replayed once per token.
- Device-resident token position, sequence metadata, expert IDs and sampling state.
- One small host readback: selected token ID and stop status.

Use the existing SYCL command-graph path over Level Zero first. P3 runtime tracing established that the corrected exact-model `MUL_MAT_ID` already takes the fused device-routed path and does not execute the generic ids-copy/wait fallback. The remaining graph boundary is the 993,280-byte logits readback and host sampling, plus any independently measured graph exclusions.

Keep all pointers stable and write changing scalars into device memory. Test direct Level Zero regular command-list replay only if the SYCL graph is measurably slower or unreliable; do not adopt mutable/experimental Level Zero extensions as the first dependency.

### 6.3 Kernel count target

Target approximately five kernels per layer, not one kernel for the whole model.

GDN layer:

1. Residual + RMSNorm + Q/K/V/Z/alpha/beta multi-projection.
2. Conv1d + GDN recurrence + state update + z gate/norm.
3. Output projection + residual + next RMSNorm + router + TopK.
4. Selected routed/shared gate-up projection + activation.
5. Down projection + routing-weighted reduction + shared expert + residual.

Full-attention layer:

1. Residual + RMSNorm + Q/K/V/gate projection + Q/K norm + RoPE.
2. Quantized paged decode attention + KV write.
3. Output projection + residual + next RMSNorm + router + TopK.
4. Selected routed/shared gate-up projection + activation.
5. Down projection + weighted reduction + shared expert + residual.

End of model:

1. Final norm + LM-head GEMV.
2. GPU top-k/sampling and token-ID writeback.

This gives approximately 202 device dispatches per token. A reusable command list removes host submission between them. Further fusion is allowed only when profiling shows a gain without spills or occupancy collapse.

### 6.4 Why not one literal kernel

A 40-layer kernel would require device-wide barriers after projections, recurrent updates, attention reductions, dynamic routing and expert reductions. Cooperative launch residency would cap the number of work-groups, while different phases need incompatible work-group sizes and GRF/SLM footprints.

Expected failure modes:

- GRF spills and large-GRF occupancy loss
- instruction-cache pressure
- work-groups waiting at software barriers that cannot all be resident
- underutilization during scalar/router phases
- difficult rollback for speculative GDN/KV state
- GPU hangs that are hard to localize
- compiler sensitivity to one giant ESIMD control-flow graph

A literal cooperative kernel should be attempted only after the captured graph is correct and fast. Kill it if it gains less than 5% over the captured command-list executor.

## 7. Kernel strategy by workload

| Rows | Preferred path | Reason |
|---:|---|---|
| M=1 | Wide-load ESIMD GEMV | Decode is bandwidth/launch bound; DPAS tiles are underfilled |
| M=2–4 | Split-K or batched ESIMD GEMV | More parallelism without DPAS setup overhead |
| M=8–64 | DPAS/XMX GEMM | Enough rows to fill natural matrix tiles |
| Prefill | oneDNN SDPA + XMX GEMM, later custom ESIMD where justified | Strong existing B70 evidence |
| Concurrent MoE | Route/group tokens by expert, then grouped DPAS | Reuse expert tiles across rows |
| MTP verification | DPAS only when accepted-candidate work amortizes state management | Existing local MoE MTP is neutral/negative |

Do not force XMX into batch-1 GEMV. Intel’s current donor source documents about 316 GB/s for a narrow-load MoE DPAS path versus about 574 GB/s for BMG wide-load decode GEMV.

## 8. Weight-format strategy

### Stage 1: exact GGUF compatibility

Implement or reuse direct kernels for the current manifest:

- Q8_0 dense GEMV
- Q4_K routed gate/up
- Q5_K routed down
- small Q6_K routed subset
- F32 router

Preserve the exact quantized values and compare layer outputs against b9853. This path is the correctness and integration baseline.

Intel’s current INT4 donor is symmetric group-128 INT4. It is not a drop-in implementation of GGUF Q4_K/Q5_K. Do not silently reinterpret the formats.

### Stage 2: BMG-native pack

Generate from official BF16/safetensors weights:

- Routed experts: symmetric W4, group 128, FP16 scales.
- Shared expert: W6 first.
- GDN and attention projections: W6 first, then selectively test W4.
- LM head: W6 first.
- Router: FP16 or F32.
- Decode layout: contiguous 64-byte cache-line streaming along K.
- Prefill layout: separate VNNI/DPAS-friendly order.

Keep separate decode and prefill layouts if the VRAM cost remains acceptable. Do not compromise M=1 layout to make DPAS convenient.

## 9. Donor source status captured today

Pinned research snapshots:

- Intel `llm-scaler` current snapshot: `f72576f79c5b3df9f6df3d0d639d4e91018dddb7`
- Intel Qwen3.5 symmetric-INT4 ESIMD MoE donor commit: `ea6a0d61bfb9df0a4f447ab9f63d1f01016aa7ac` (`Add INT4 ESIMD MoE Kernels for Qwen3.5 sym_int4 Quantization`, #360)
- `vllm-xpu-kernels`: `eb07423780c2f9547e24ee69c85ef33310f1558c`
- llama.cpp master observed: `12127defda4f41b7679cb2477a4b0d65ee6a0c8f`

Useful donor files:

- `fp8_GEMV_v2.h`: BMG M=1 GEMV, source claims about 97% bandwidth utilization on large shapes.
- `fp8_GEMV_bmg.h`: wide vector/tail handling.
- `int4_GEMV.h`: symmetric group-128 W4, FP32 accumulation.
- `resadd_norm_gemv_int4.h`: residual + RMSNorm + W4 GEMV.
- `moe_decode_gemv.h`: explains the 316 vs 574 GB/s load-width issue.
- `moe_int4.sycl`: fused router and INT4 MoE paths.
- `gdn_conv_fused_seq.h`: sequential Qwen35 GDN path, including H=16/HV=32 large-H dispatch.
- `vllm-xpu-kernels/csrc/xpu/gdn_attn/xe_2`: official Xe2 GDN kernels and FP32-state test support.
- `vllm-xpu-kernels/csrc/xpu/attn/xe_2`: Xe2 paged decode and FMHA.
- `vllm-xpu-kernels/csrc/xpu/grouped_gemm/xe_2`: Xe2 grouped GEMM.

Important donor caveat: `gdn_conv_fused_seq.h` currently stores the SSM state as FP16. Use it as an algorithm/layout donor, but retain an FP32-state reference and quality gate.

## 10. Exact desktop execution sequence

Run this after work, only after SSH is healthy. These commands intentionally preserve the daily b9853 tree.

### M0. Recover and identify the host

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 desktop-b70 '
set -e
hostname
uname -a
uptime
lspci -nn | grep -Ei "VGA|Display"
readlink -f /sys/class/drm/card*/device/driver 2>/dev/null || true
'
```

If this still hangs during banner exchange, recover the desktop locally before continuing. An open TCP socket is not sufficient.

### M1. Capture the exact software and device state

```bash
ssh desktop-b70 '
set -e
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
mkdir -p ~/b70-lab/state
{
  date -Is
  hostnamectl
  uname -a
  sycl-ls
  icpx --version
  ocloc --version || true
  xpu-smi --version || true
  xpu-smi discovery || true
  dpkg -l | grep -E "level-zero|intel-opencl|intel-level-zero|oneapi" || true
  git -C ~/llama.cpp rev-parse HEAD
  git -C ~/llama.cpp describe --tags --always
  grep -E "^(GGML_SYCL|CMAKE_BUILD_TYPE)" ~/llama.cpp/build-sycl-latest-l0/CMakeCache.txt || true
} | tee ~/b70-lab/state/stack-$(date +%F-%H%M).txt
'
```

### M2. Confirm the preserved control

```bash
ssh desktop-b70 '
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
~/llama.cpp/build-sycl-latest-l0/bin/llama-server --version
~/llama.cpp/build-sycl-latest-l0/bin/llama-bench --list-devices
'
```

Do not rebuild `~/llama.cpp/build-sycl-latest-l0`.

### M3. Create isolated worktrees

Before any commit, verify:

```bash
git config user.name
git config user.email
```

Required identity:

```text
SergiioB <<owner-email>>
```

Then:

```bash
cd ~/llama.cpp
git fetch origin --tags
git fetch origin pull/25690/head:refs/remotes/origin/pr-25690
mkdir -p ~/b70-lab/worktrees

git worktree add -b b70-master-correct \
  ~/b70-lab/worktrees/llama-master origin/master

git -C ~/b70-lab/worktrees/llama-master merge --ff-only origin/pr-25690 || true
```

If the fast-forward is impossible, create a separate PR worktree rather than forcing the merge:

```bash
git worktree add -b b70-pr25690 \
  ~/b70-lab/worktrees/llama-pr25690 origin/pr-25690
```

Do not stack performance PRs until this correctness branch passes.

### M4. Build the correctness branch

First recover the known architecture selector from the preserved cache:

```bash
ARCH=$(grep '^GGML_SYCL_DEVICE_ARCH:' \
  ~/llama.cpp/build-sycl-latest-l0/CMakeCache.txt \
  | cut -d= -f2)
printf 'ARCH=%s\n' "$ARCH"
```

Build:

```bash
source /opt/intel/oneapi/setvars.sh --force
SRC=~/b70-lab/worktrees/llama-pr25690
BUILD=~/b70-lab/build/llama-pr25690

cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_F16=ON \
  -DGGML_SYCL_DNN=ON \
  -DGGML_SYCL_DEVICE_ARCH="$ARCH" \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=ON

cmake --build "$BUILD" -j"$(nproc)"
```

### M5. Correctness gate before performance

```bash
source /opt/intel/oneapi/setvars.sh --force
BUILD=~/b70-lab/build/llama-pr25690

"$BUILD/bin/test-backend-ops" test -b SYCL0 -o MUL_MAT
"$BUILD/bin/test-backend-ops" test -b SYCL0 -o MUL_MAT_ID
```

Also require:

- Q4_K model deterministic 20-turn conversation.
- Q5_K model deterministic 20-turn conversation.
- Same fixed prompts on b9853, corrected current branch and Vulkan/CPU control.
- No gibberish on turn 2 or after context reuse.
- Stable tool-call JSON.

Any failure stops performance work.

### M6. Profile b9853 without changing it

Start with summaries:

```bash
source /opt/intel/oneapi/setvars.sh --force
BIN=~/llama.cpp/build-sycl-latest-l0/bin
MODEL=/absolute/path/to/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf

unitrace -h -d -s -v -o ~/b70-lab/profile/b9853-qwen35.txt \
  "$BIN/llama-bench" \
  -m "$MODEL" -ngl 999 -fa 1 -p 64 -n 128 -r 1
```

Then capture host/device gaps:

```bash
unitrace \
  --chrome-call-logging \
  --chrome-sycl-logging \
  --chrome-kernel-logging \
  -o ~/b70-lab/profile/b9853-qwen35-trace.json \
  "$BIN/llama-bench" \
  -m "$MODEL" -ngl 999 -fa 1 -p 64 -n 32 -r 1
```

Run metrics separately because metric collection perturbs timing:

```bash
unitrace --metric-list
unitrace -q -g ComputeBasic \
  --chrome-kernel-logging \
  -o ~/b70-lab/profile/b9853-compute.csv \
  "$BIN/llama-bench" \
  -m "$MODEL" -ngl 999 -fa 1 -p 64 -n 16 -r 1
```

Extract:

- Total kernels/token.
- Median host-to-device gap.
- Top 20 kernels by total device time.
- Kernel call count.
- SLM/work-group, private memory and spill bytes.
- GTI read/write bytes and effective bandwidth where metrics expose them.
- Time in Q8 DMMV, Q4/Q5 expert DMMV, GDN state, attention, copies and reductions.

This profile decides the first implementation. Do not infer it from model architecture alone.

### M7. Reproduce donor kernels on exact shapes

Use current source pins, but keep this outside the daily environment:

```bash
cd ~/b70-lab
git clone https://github.com/intel/llm-scaler.git
git -C llm-scaler checkout f72576f79c5b3df9f6df3d0d639d4e91018dddb7

git clone https://github.com/vllm-project/vllm-xpu-kernels.git
git -C vllm-xpu-kernels checkout eb07423780c2f9547e24ee69c85ef33310f1558c
```

Use Intel’s pinned container/toolchain first. For Qwen3.5/3.6 the current README still recommends the 0.14.0-b8.3.x line; do not assume the 0.21 image has identical Qwen kernels without inspection.

Minimum exact microbench matrix:

| Kernel | Shape |
|---|---|
| Dense GDN qkv projection | N=8192, K=2048, M=1 |
| GDN z/gate projection | N=4096, K=2048, M=1 |
| GDN output projection | N=2048, K=4096, M=1 |
| Attention Q+gate projection | N=8192, K=2048, M=1 |
| Attention K/V | N=512, K=2048, M=1 |
| Attention output | N=2048, K=4096, M=1 |
| Router | N=256, K=2048, M=1 |
| Expert gate/up | 8 experts, N=1024, K=2048 |
| Expert down | 8 experts, N=2048, K=512 |
| Shared expert | N=1024/2048, K=2048/512 |
| LM head | N=248320, K=2048, M=1 |
| GDN recurrence | H=16, HV=32, K=V=128, TP=1, FP32 state |

Microbench gates:

- Exact CPU/reference agreement.
- No private-memory spill.
- Stable results after 10,000 iterations.
- Large streaming matrices target at least 500 GB/s effective weight bandwidth.
- Compare queue submission and kernel-only timing separately.

### M8. First integration candidate

Choose from the profile:

1. If host/launch gaps exceed 20% of token time: implement reusable Level Zero command-list replay first.
2. If dense Q8 projections dominate: implement fixed-shape Q8_0 ESIMD multi-GEMV fusion first.
3. If MoE expert kernels dominate: implement Q4_K/Q5_K wide-load two-phase MoE first.
4. If GDN recurrence dominates: port the Xe2 GDN kernel with FP32 state first.
5. If LM head dominates: add Q8/W6 LM-head GEMV and fused GPU sampling.

Do not choose the visually most interesting kernel; choose the largest measured Amdahl term.

## 11. Recommended first fused block

The most likely winning first full-block prototype is one Qwen35 GDN+MoE layer with five phases:

1. Residual/RMSNorm and combined QKVZ/alpha/beta projections.
2. FP32-state GDN recurrence and convolution.
3. GDN output projection, residual, RMSNorm, router and TopK.
4. Routed plus shared gate/up projection and SiLU multiplication.
5. Down projections, weighted reduction and residual.

Reasons:

- GDN appears in 30 of 40 layers.
- GDN projection weights are the largest active-byte subsystem.
- The block includes router/TopK and MoE launch overhead.
- The exact shape is fixed and already represented in Intel donor code.
- A single block provides a hard A/B before implementing all 40 layers.

Promotion gate for the block:

- Layer output within the agreed tolerance of the exact-GGUF SYCL path.
- FP32 state drift remains bounded through at least 4,096 generated-state updates.
- At least 1.20x layer latency improvement.
- No spills.
- At least 15% projected end-to-end gain from measured layer occurrence and Amdahl analysis.

If the block does not meet these gates, stop the custom runtime branch and keep upstream-focused patches.

## 12. Prefill track remains separate

For long-context prefill, the strongest evidence remains:

- #25025: dequantize K/V and use oneMKL GEMMs plus online softmax.
- #25222: oneDNN Graph fused SDPA for native F16 K/V.
- #25312: quantized-KV dequantization into oneDNN SDPA.

Current status:

- #25025: open.
- #25222: open.
- #25312: open and currently reports a dirty/conflicting merge state.

Test each in an isolated worktree with #25690 correctness fixes applied. Do not stack all three first.

Order:

1. #25025 as the broad all-KV control.
2. #25222 with F16 KV.
3. A clean quantized-KV adapter based on #25312’s concept after provenance/conflicts are resolved.

These paths primarily improve prefill and TTFT. They are not the main answer for 68 t/s single-token decode.

## 13. On-device sampling and LM head

The LM head alone is 540 MB/token in the current Q8_0 pack, nearly 20% of active weight bytes.

The exact path still needs all 248,320 logits for greedy/top-k sampling unless an approximation is accepted. Do not use an unverified vocabulary shortlist as the default.

Safe optimization:

- Stream LM-head rows once.
- Maintain per-work-group top-k candidates while producing logits.
- Reduce candidates on-device.
- Apply repetition/frequency penalties and sampling on-device.
- Return token ID only.

This saves the full logits write/read and a host synchronization, but not the weight read. A W6 LM head is the safer traffic-reduction experiment before W4.

## 14. MTP decision

Do not budget an MTP multiplier for Qwen35/Ornith yet. Existing local MoE MTP was neutral or negative.

Revisit only when:

- Ordinary token graph is already captured.
- Verification can execute M=2–4 in batched ESIMD or M>=8 in DPAS.
- GDN, conv and KV state use device-side transactional buffers.
- Accepted-prefix state can commit without host copies.
- Mean accepted tokens per target pass exceeds the break-even point measured on the B70.

MTP promotion gate:

`effective accepted output tok/s >= ordinary decode tok/s x 1.15`

with equal quality and no tail-latency regression.

## 15. Performance targets

Targets are now split into gates rather than hype:

### Exact current Q4_K_XL format

- Baseline: approximately 67–68 t/s historical.
- First promotion gate: 82+ t/s, correct, at the same context/power.
- Strong graph/fused path: 95–120 t/s.
- Stretch: 120–140 t/s if the profile confirms large launch and dense-Q8 inefficiency.

### BMG-native mixed W4/W6 pack

- Promotion only after the exact-format engine is correct.
- Target: 110–150 t/s at short/medium context.
- 128K context roof is lower because q8/q4.1 KV traffic exceeds 1.1 GB/token.

### MTP

- Do not quote 180+ effective t/s until acceptance, rollback cost and state traffic are locally measured.

### Power

- Optimize at 165 W first; historical data showed the best efficiency there.
- Characterize 230 W briefly only after kernels are stable.
- Report tok/J, not just tok/s.

## 16. Kill criteria

Stop or narrow the custom engine if any is true:

1. Corrected current llama.cpp plus upstream PRs reaches within 10% of the custom block.
2. Captured command-list replay gains less than 5% and launch gaps are below 10% of token time.
3. The first fused GDN+MoE block gains less than 20% at block level.
4. The projected correct end-to-end gain is below 15%.
5. Required kernels spill or lose enough occupancy that fusion regresses performance.
6. FP32 GDN state cannot be supported without losing the measured gain.
7. Mixed W4/W6 fails perplexity, KL, coding, tool-use or multi-turn quality gates.
8. Maintenance requires an LLVM/IGC fork for ordinary operation.
9. GPU reset/recovery is unreliable after kernel faults.

## 17. Verification matrix

Every promotion candidate must pass:

1. `test-backend-ops` for affected ops and formats.
2. Per-layer output comparison against b9853.
3. FP32 GDN-state comparison over long deterministic sequences.
4. Next-token logits KL and top-k overlap.
5. Perplexity on a fixed public dataset.
6. Twenty-plus deterministic multi-turn exchanges.
7. Context reuse, slot restore and repeated-prefix tests.
8. 32K and 128K retrieval.
9. Tool calls and strict JSON schema.
10. Coding patch plus real test execution.
11. Qwen and Ornith separately.
12. Batch/concurrency 1, 4, 8, 16 and 32.
13. 150, 165 and 180 W; 230 W characterization only.
14. Clocks, thermals, VRAM, errors and reset behavior.
15. Source, model, compiler, driver and command hashes in every result.

## 18. Final recommendation

The B70 has enough bandwidth and explicit-SIMD control for a materially faster fixed-model engine. The exact GGUF analysis proves that the ordinary 68 t/s path is far below the 507 GB/s purpose-built roofline, even after accounting for FP32 GDN state and long-context KV traffic.

The best option is not “an INT4 MoE mega-kernel.” It is a launch-minimized Qwen35 block engine that attacks all three dominant terms:

1. Q8 GDN/attention projections.
2. Q4/Q5 selected experts.
3. Q8 LM head and sampling.

Build it inside llama.cpp first, replay it through Level Zero, and require a 20% block win before expanding. That is the shortest route to a professional result and the only route that separates real B70 headroom from theoretical TOPS marketing.

## 19. Additions from the independent implementation review

Two independent workers completed after the main plan was written. Their useful implementation findings are incorporated below. Their planning roofline assumed approximately 1.907 GB of active weight reads because the exact tensor inventory was unavailable to them. The parsed GGUF value of 2.753 GB is authoritative for the current Q4_K_XL artifact; the more optimistic planning roofline is therefore not used.

### 19.1 Smallest toolchain prototype

Before porting a complete Qwen block, create a standalone native C++ microbenchmark containing one extracted BMG ESIMD FP8 GEMV:

```text
M=1, N=4096, K=4096
FP16 input
FP8 E4M3 row-major weights
FP32 accumulation
FP16 output
```

This prototype is not the final model format. Its purpose is to prove, independently of Torch, GGUF and MoE:

- BMG AOT compilation rather than JIT.
- ESIMD execution on the Level Zero backend.
- Direct versus SYCL command-graph replay.
- Unitrace timing and AOT/spill reporting.
- IGC assembly capture.
- Stable USM pointers and 10,000 correct graph replays.
- Reproducible effective-bandwidth measurement.

Acceptance:

- No spill or private-memory allocation.
- No NaN/Inf.
- Cosine similarity at least 0.9999 against a CPU reference using the same dequantized FP8 values.
- Normalized RMSE no worse than 1e-3.
- At least 75% of a local streaming-read benchmark, or within 15% of the pinned donor kernel.
- Graph and direct submission produce equivalent output.

Only after this passes should the project add Q8_0, Q4_K/Q5_K and native W4/G128 kernels.

### 19.2 Standalone library, not standalone model runtime

Organize the first implementation as a small reusable library and harness:

```text
b70-engine/
  include/b70/          public API, plan and packed-weight ABI
  src/runtime/          queue, graph, events and Level Zero interop
  src/format/           pack reader/writer and quantization
  src/kernels/esimd/    one translation unit/module per kernel family
  benchmarks/           copy, GEMV, GDN, attention, MoE and graph replay
  tests/                format, kernel, graph and adversarial tests
  tools/                pack, inspect and benchmark CLIs
  integrations/llama/   narrow patch series and smoke tests
  third_party/          provenance manifest, licenses and origin records
```

This is not a second tokenizer/server/runtime. It is a kernel/runtime library linked into an isolated llama.cpp fork. llama.cpp continues to own model loading, graph semantics, KV/state lifecycle, tokenization, sampling initially, server behavior and fallbacks.

### 19.3 Packed sidecar ABI

Do not modify the source GGUF. Generate a versioned sidecar:

```text
model.gguf
model.gguf.b70pack
```

The header and each tensor descriptor must include:

- Source GGUF SHA-256.
- Format and kernel-ABI version.
- Source tensor name, shape and GGML type.
- Packed dtype and layout ID.
- Group size, pack factor and scale layout.
- Payload offsets, lengths, alignment and checksum.
- Model metadata hash.

Reject stale sidecars when the source hash, tensor table or ABI differs. Align payloads to at least 64 bytes and large tensor starts to 4 KiB.

Initial independent formats:

1. Dense row-major W4/G128 for M=1 contiguous K streaming.
2. Expert K-packed/VNNI-style W4/G128 for MoE/prefill.
3. FP8 row-major for the toolchain prototype.

Do not label the donor’s group-128 format “GGUF Q4_0.” It is not binary-compatible with llama.cpp Q4_0/Q4_K.

### 19.4 llama.cpp integration seam

Link the standalone library into `ggml-sycl` in the experimental fork and enqueue on the existing backend queue. Do not create a second Level Zero context or unordered queue.

Minimal internal seam:

```cpp
bool b70_supports(const ggml_tensor * dst, const b70_runtime_caps & caps);

ggml_status b70_enqueue(
    sycl::queue & queue,
    b70_weight_registry & weights,
    b70_scratch & scratch,
    ggml_tensor * dst);
```

Integration order:

1. Register sidecar pointers during model/SYCL-buffer initialization.
2. Intercept supported batch-1 `GGML_OP_MUL_MAT` shapes.
3. Add fused projections sharing the same input.
4. Preserve the already-active device-routed `GGML_OP_MUL_MAT_ID` path and make logits reduction/sampling device-resident.
5. Preserve the unchanged b9853 path for every unsupported shape and for prefill until the custom prefill path exists.

P3 disproved the model-relevant `MUL_MAT_ID` host-wait premise. The immediate whole-graph capture enabler is removing the full-logits readback/host-sampling boundary while preserving exact sampling semantics.

### 19.5 Build discipline

Use:

```text
-fsycl
-O3
-std=c++20
-fsycl-targets=spir64_gen
-fsycl-device-code-split=per_kernel
-Xsycl-target-backend=spir64_gen "-device bmg"
```

But query and reuse the architecture selector accepted by the installed compiler/baseline rather than assuming the alias. Confirm driver expansion with `icpx -###`.

Maintain three presets:

- Correctness: no global `-ffast-math`.
- Release: fast math only for individually validated kernels.
- Profile: line tables, Unitrace and IGC dumps.

Compile large-GRF kernels in separate modules. Never force large GRF for the entire engine.

### 19.6 Additional profiler gates

For each benchmark report:

- Device median, p90, p99 and minimum.
- Host-wall median, p90 and p99.
- Submission overhead and graph-replay overhead.
- Packed bytes and effective GB/s.
- AOT/JIT status.
- SLM, private memory and spill bytes.
- Compiler, driver, device, kernel ABI and packed-format versions.
- Numerical metrics and hash of the raw JSON result.

Capture IGC dumps in a dedicated profile build and inspect for scratch accesses, scalarized loads, missing LSC/block loads, missing DPAS where expected, excessive barriers and spill/fill sequences.

### 19.7 Revised schedule

| Week | Deliverable | Hard exit gate |
|---:|---|---|
| 1 | Baseline/toolchain lock and one-kernel AOT probe | Unitrace proves Level Zero and AOT |
| 2 | Standalone FP8 GEMV | Correct, reproducible, zero spill |
| 3 | Read-bandwidth roofline and kernel tuning | Meets bandwidth gate |
| 4 | Versioned sidecar and first native W4 GEMV | Deterministic pack/unpack and CPU agreement |
| 5 | SYCL graph replay | 10,000 replays, no rebuild/leak, at least 3x lower submission overhead |
| 6 | First llama.cpp M=1 integration | Real model uses selected kernels with fallback |
| 7 | Fused dense projections and RMSNorm | Measurable end-to-end improvement |
| 8 | Fixed-shape decode attention/GDN | Correct across target context/state cases |
| 9 | Device-routed persistent MoE operation | No host expert loop or readback |
| 10 | Full fixed-shape decode graph and sampling boundary | 10,000-token soak |
| 11 | End-to-end power/latency review | At least 1.25x or stop broad engine work |
| 12 | Three-day whole-model persistent spike, then hardening | Continue only for at least 15% over graph, zero spill |

### 19.8 Refined kill gates

- Stop standalone AOT work if the BMG toolchain cannot be made reliable in three engineering days.
- Stop extracted FP8 work if it remains below 60% of local read bandwidth after two tuning iterations or the donor stays materially faster.
- Do not integrate native INT4 unless it beats the corresponding b9853 matvec by at least 1.25x on most target shapes and 1.10x on every target shape.
- Abandon graph capture if replay is over 3% slower than direct submission, fails the 10,000-replay soak or reduces submission overhead by less than 3x.
- Do not productize below 1.25x batch-1 throughput, with no more than 10% extra VRAM and no correctness regression.
- Kill the whole-model persistent experiment unless it adds at least 15% over the completed graph engine, remains spill-free and retains at least 50% viable occupancy.

## 20. Sources checked for this continuation

- Qwen3.6 config and GGUF manifest: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- Public GGUF: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF
- Transformers Qwen3.5-MoE implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
- llama.cpp #25025: https://github.com/ggml-org/llama.cpp/pull/25025
- llama.cpp #25063: https://github.com/ggml-org/llama.cpp/pull/25063
- llama.cpp #25222: https://github.com/ggml-org/llama.cpp/pull/25222
- llama.cpp #25312: https://github.com/ggml-org/llama.cpp/pull/25312
- llama.cpp #25455: https://github.com/ggml-org/llama.cpp/issues/25455
- llama.cpp #25690: https://github.com/ggml-org/llama.cpp/pull/25690
- Intel llm-scaler: https://github.com/intel/llm-scaler
- vLLM XPU kernels: https://github.com/vllm-project/vllm-xpu-kernels
- Intel PTI unitrace: https://github.com/intel/pti-gpu/tree/master/tools/unitrace
