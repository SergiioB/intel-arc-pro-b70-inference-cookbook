---
title: "FP8 TP2 Speed Attack — Consolidated Record and Current Status"
type: research
date: 2026-08-30
status: final
---

# FP8 TP2 Speed Attack — Consolidated Record and Current Status
**Dates:** 2026-08-28/30 · **Host:** dual Arc Pro B70, vLLM XPU TP2, per-worker `ZE_AFFINITY_MASK` + `SYS_PTRACE`, 230 W campaign caps
**Chronological evidence trail:** `research/2026/qwen38-fp8-tp2-speed-attack-and-ceilings-20260829.md` (authority; all raw addenda) · routing tables: `models/qwen/qwen3.8-27b/serving.md`
**Status: PROVISIONAL — NOT FOR PUBLIC HEADLINE**

> **2026-08-30 planning correction (supersedes §7 multiplier 1):** the official
> [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
> checkpoint (2B BF16, block size 8) plus merged upstream [PR #52816](https://github.com/vllm-project/vllm/pull/52816)
> change the DFlash route: **adoption of the official DFlash2 checkpoint precedes retraining**,
> and the SpecForge capture/retrain project is a diagnostic fallback, not the next multiplier.
> Published Qwen3.8-27B mean acceptance length is **4.80 vs MTP 4.28 vs DSpark 3.62 at block size 8**
> (external H200/BF16/FA3 evidence, E3 — not a B70 projection, and not evidence for length 8–10).
> The old 24.7% DFlash2 result was an implementation failure (off-by-one proposer offsets), and the
> later 90.5% p2048 cell was a narrow corrected-stack observation on the rmacy image with the
> community drafter — not a representative official-DFlash2 result. The decision authority is
> [`research/2026/qwen38-dflash2-objective-campaign-20260830.md`](qwen38-dflash2-objective-campaign-20260830.md),
> campaign `CAM-2026-QWEN38-DFLASH2-OBJECTIVE`.
>
> **Campaign outcome (same day): completed no-go.** Official DFlash2 k=7 passed correctness on
> the FP8 W8A16 TP2 stack but lost the matched n=3 screen to native MTP7 on every cell
> (p512 e2e 11.12 vs 22.14 tok/s; acceptance 0.10 vs 0.40; mean accept length ≈0.7 vs ≈2.8).
> DFlash2 stays closed on this stack; MTP4 / k=8 MTP remain the FP8 route of record. §7
> multiplier 1 is now: DFlash2 **retraining on W8A16-decode hidden states** (trainer pending
> from Inco/z-lab), with SpecForge capture as the diagnostic fallback.

---

## 1. Where things stand now (current state)

- **FP8 route of record (≤1536-token workloads):** pinned image `f01e24f6` + full custom-wheel package mount + `patch_fp8_stack.py` (M≤40 → custom `xe2_block_fp8_small_m` op → `fp8_gemm_w8a16` → env-forced W8A8), tuned flags (`--max-num-seqs 1 --async-scheduling --block-size 64 --mamba-ssm-cache-dtype float16`, BF16 KV, CCL tmp-buf set), **k=8 native MTP**.
  **Records: 51.54 tok/s (p512), 56.85 tok/s (p1024, accept len 6.63).** Power 90–99 W/card.
- **FP8 ≥2048-token workloads:** same stack + `B70_FP8_MODE=hybrid` (`patch_fp8_hybrid.py`), **k=6**: p2048 **28.5–32.2 tok/s** with acceptance **49–61%** — the long-context acceptance collapse is fixed operationally.
- **Host overall champion:** INT4 GPTQ-nogidx TP2 MTP4 = **84.0 / 84.21 tok/s** (BF16/FP16 parity), 24.5–24.9 at p8192.
- **dflash2/DSpark (rmacy stack):** best = k=4, **39.79 tok/s** (p2048, accept 90.5%). ABI-locked to torch 2.11 (new kernels need `libsycl.so.9`); cannot take the custom kernel work. **Status 2026-08-30:** superseded as the DFlash route by the official-DFlash2 objective campaign (see the planning-correction block above; rmacy numbers remain the community-drafter reference, not the official checkpoint).
- **Cn-MTP4:** upstream-broken (GDN `causal_conv1d` spec/non-spec crash on 0.21.1, 0.27.2 and 0.28.0). No aggregate MTP4 numbers until fixed.
- **Host state:** restored — 150 W caps, zero containers, VRAM empty (31.6/32.6 GiB), swappiness 60. SpecForge XPU branch staged at `/mnt/<models-disk>/specforge-xpu`.

## 2. What was tried — the complete matrix

### ✅ Adopted (each measured, in the route of record)

| # | Lever | Mechanism | Measured effect |
|---|---|---|---|
| 1 | **W8A16 reroute** (`patch_fp8_stack.py`) | BF16 activations direct into oneDNN `fp8_gemm_w8a16`; `apply_input_quant=False` deletes all per-linear act-quant kernels | p512 +34% (43.34 vs 32.44 same draw); TPOT 26.5→19.9 ms; TTFT −27%; power −20 W |
| 2 | **Scratchpad pooling** (`apply_scratchpad_patch.py`) | `thread_local` pooled scratchpad per GEMM primitive (3 sites + helper in `onednn_ext.h`) | compiled+landed; E0 1–4 ms/step (inside combined records) |
| 3 | **Xe2 small-M kernel** (`apply_smallm_patch.py`, 7 sub-patches) | E=1 reuse of Xe2 persistent grouped-GEMM scheduler + new block-scaled K=128 inner loop (per-`B_scale[k,n]` apply — old grouped FP8 was scalar-per-expert); M-tiles 16/32; op registered in `_xpu_C.abi3.so` | **parity PASS** (max abs err 0.0078 vs w8a16 at M=5); inside all k=6/k=8 records |
| 4 | **k-ladder 4→6→8** | Native MTP head generalizes to deep speculation; small-M op absorbs M≤40 verify rows (k=8 → 9 rows) | monotone ≤1536: p512 46.09→48.89→51.54; p1024 47.49→53.83→56.85; accept len 3.91→4.40→6.63 |
| 5 | **HYBRID routing** (`patch_fp8_hybrid.py`) | decode/small-M → fast path; prefill/large-M → inline-quant W8A8 (MTP head's training numerics) | fixes long-context acceptance collapse: p2048 0%→49/61%, 11.6→28.5/32.2 tok/s (+145%) |
| 6 | Tuned flags | seqs 1, async-scheduling, block 64, mamba-fp16, BF16 KV, CCL tmp-buf set | adopted into every tuned baseline |
| 7 | INT4 GPTQ nogidx artifact (prior session) | W4A16 oneDNN path = half the weight traffic | 84 tok/s host champion (unchanged tonight; FP16-dtype parity confirmed) |

### ❌ Refuted / no-effect (measured or code-proven — do not retry)

| Lever | Verdict | Proof |
|---|---|---|
| XPU graphs on TP2 (FULL or PIECEWISE) | **impossible** — upstream caps capture at single-GPU (`xpu.py:307`); compilation-config on TP2 is a no-op (27.41 ≈ 27.12 baseline) | `qw38-fp8-tp2-pwgraph-*` server.log warning |
| `VLLM_XPU_SPEC_DECODE_MAX_QLEN=64` | no-op — gate is per-sequence `q_len`=5, not the 40 total rows; fast path already taken | architect source read |
| `CCL_SYCL_FORCE_PCIE=1` | no measurable effect (within noise, same draw) | `qw38-fp8-tp2-forcepcie-*` |
| oneDNN descriptor/primitive caching patch | already implemented upstream (`matmul_primitive_create_and_cache`); only scratchpad was poolable | architect source read |
| `paged_decode` full-matrix rebuild | unnecessary — Qwen tuples already compiled with `grf_size<256>` | source read |
| `VLLM_XPU_FORCE_XE_DEFAULT_KERNEL` | dense model: zero effect (grouped-GEMM-only knob) | source read |
| Natural content as a dflash speed lever | no unlock — 0.21 stack is step-bound at ~40 tok/s regardless of content | `qw38-fp8-tp2-dflash2-rmacy-*/natural` |
| dflash k=6 on the k=4-block-trained drafter | **refuted** — accept 11.3–32.2% vs 90.5% at k=4; drafter degrades wholesale beyond trained depth | `qw38-fp8-tp2-dflash2-k6-*` |
| dflash2 on stock images | two independent failures (`hc_mult` schema; `confidence_head` mapping) — rmacy image is the only working host | `qw38-fp8-tp2-dflash2-fixed-*`, `dspark-native-*` |
| dflash2 + custom wheel | **ABI-rejected** — wheel needs `libsycl.so.9` (oneAPI 2026); rmacy image ships older SYCL (torch 2.11) | container import test |
| Native `method:dspark` (pinned) for this checkpoint | `hc_mult` missing — checkpoint schema older than pinned implementation | `qw38-fp8-tp2-dspark-native-*` |
| FP8 per-tensor/per-channel conversion | E0 5–25% only — still ≥30% below INT4; lossy requant of lossy artifact; ADR-0007 sanctions INT4 W4A16 as the conversion baseline instead | architect analysis |
| PIECEWISE config as an acceptance fix | the acceptance swing was content, not fusions (27.41 ≈ 27.12 same-draw) — corrected earlier hypothesis | `pwgraph` vs `forcepcie` same-draw cells |

### 🚫 Blocked upstream

| Blocker | Detail |
|---|---|
| XPU graphs on TP2 | single-GPU capture cap in all three builds (0.21.1 / 0.27.2 / 0.28.0) |
| Cn-MTP4 GDN crash | `causal_conv1d does not support spec-decode and non-spec tokens in the same invocation` — engine dies on 0.27.2 AND 0.28.0 |
| W8A16 long-context acceptance collapse | draw-dependent, >1536 tokens; hybrid mode is the operational fix, engine-side acceptance guard is the clean fix (proposer-counter wiring) |

## 3. Root causes established (the "why" layer)

1. **Acceptance is content-driven.** Identical builds/flags: MTP4 82.1%↔36.7% by prompt draw; W8A16 0%↔90.8%. Step time is nearly constant (~59–77 ms) — accepted tokens/step is what moves. The "morning regression", the "nightly collapse", and the "nightly vs Reddit" deltas were all content artifacts. **Rule: every speculative number ships with acceptance + content provenance.**
2. **The p2048 collapse was never an M=2048 structural bug.** Boundary sweep p2040–2056 showed no collapse on a fresh draw. Mechanism: W8A16 prefill numerics shift the KV/GDN state the MTP head consumes; on some draws the shift compounds past the head's generalization at longer contexts. Hybrid routing (prefill W8A8) restores it.
3. **The dflash 24.7% cap was the off-by-one proposer bug** (verified in our images at `utils.py:556-557`; paired fix `dflash.py num_query_per_req = k`). Our 2026-08-19 DFlash2 NO-GO measured the bug, not the method.
4. **FP8-vs-INT4 decode gap = weight traffic at small M.** Verify GEMMs (M≈5–40) are bandwidth-bound: FP8 streams 2× INT4's bytes plus block-act-quant. INT4 W4A16 oneDNN step ≈37–40 ms vs FP8 ≈70–77 ms. This is why INT4 (84 tok/s) beats FP8 (56.9) and why the small-M FP8 kernel is the only kernel-level fix.
5. **Cn-MTP4 crash is upstream** (GDN spec/non-spec mixing), identical on 0.21.1/0.27.2/0.28.0.
6. **The model is DENSE** (`qwen3_5`) — those MoE kernel libraries in the wheel are inert for this model; any MoE framing of qwen3.8 serving is wrong.

## 4. Prefill impact (documented per rung)

| Rung | p512 TTFT | p2048-class TTFT | Verdict |
|---|---:|---:|---|
| W8A8 baseline | 527–577 ms | 941–998 ms | reference |
| W8A16 reroute | **421 ms** | 945–975 ms | **prefill faster** (act-quant removed for all M) |
| + small-M kernel | 381–555 ms | — | no regression (kernel is decode-only dispatch) |
| HYBRID k=6 | ~583 ms | **1504 ms** | intentional: prefill reverts to W8A8 numerics — the price of the acceptance fix at ≥2048 |

## 5. Power

FP8 records now run at **79–145 W per pair** (was 105–186 W): act-quant removal, faster steps, INT4's halved traffic. Cards remain utilization-bound — never cap-bound (draw 75–186 W against a 230 W cap); 230 W symmetric is correct for campaigns, 150 W restored at close (verified).

## 6. Tooling + ops lessons (so nothing is re-learned)

Scripts (`scripts/tmp/`): `qw38-tp2-variant-launch.sh` (image/patch/drafter/shim/mount-parametrized), `qw38-tp2-n3-powerbench.sh` (upstream+fork flag sets, per-card cooldowns — GPU0 is the display card, ~56 °C idle floor), `qw38-natural-content-bench.py`, `patch_fp8_stack.py`, `patch_fp8_hybrid.py`, `patch_dflash_offset.py`, `patch_dspark_quant.py`, `patch_vllm_worker_affinity_v021.py`, `apply_smallm_patch.py`, `apply_scratchpad_patch.py`, `apply_fake_xpu_ops.py`, `fix_smallm_placement.py`.
Kernel pipeline: `/mnt/<models-disk>/vxk-build/{src,out}` — baseline + incremental scripts, extracted package `out/extracted2/vllm_xpu_kernels` (mount over image package dir; all libs rebuild on fresh cmake, so mount the whole dir, never a single .so).
Pitfalls catalogued: `set -u` vs setvars; `bash script.sh` not exec-bit; source-tree `build/` shadows the `build` module (run from `/`); setuptools 77.0.3 + packaging ≥26.2 pins; git `safe.directory`; template-header/name insertion split (shadowing errors); per-patch markers must be unique; `torch.compile` can't trace `globals()`/imports (log from class bodies); `--save-result` opt-in on 0.28; fork bench drops `--seed/--tokenizer`; containerd GC deletes untagged images under disk pressure (**tag local images** — the nightly build was lost); lspci LnkSta lies on runtime-suspended devices (measure H2D: 8.85/7.89 GB/s verified); runtime-PM re-pin after every power event.

## 7. 150 tok/s — honest ladder status

Achieved: FP8 56.85 (p1024) / 51.54 (p512), +75% over the session start. INT4 84 remains the host crown.
Remaining multipliers, in order: (1) **official DFlash2 k=7 adoption on the FP8 W8A16 TP2 stack** — the running objective campaign (`CAM-2026-QWEN38-DFLASH2-OBJECTIVE`); k≥6/8 community-drafter retraining (SpecForge XPU branch staged at `/mnt/<models-disk>/specforge-xpu`, XPU training probes PASS; capture script is the missing piece — ≈50 GB per 1M tokens, models LV bounds the corpus) is now the diagnostic fallback, pending the official checkpoint outcome and Inco/z-lab trainer availability; (2) root-cause the draw-dependent acceptance collapse → engine-side acceptance guard; (3) W8A16-class prefill without the acceptance tax (removes the hybrid ≥2048 penalty); (4) TTFT amortization; (5) GDN GRF256. Each is a full session. 150 needs step ≈30 ms — beyond every currently-compiled kernel; the ladder documents exactly what remains.
