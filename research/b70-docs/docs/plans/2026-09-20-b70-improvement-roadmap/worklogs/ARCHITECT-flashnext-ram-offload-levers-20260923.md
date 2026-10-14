---
title: "Flash-Next M64 RAM-offload recipe — architecture review of post-re-acquisition levers (P-3b transfer verdict)"
type: worklog
date: 2026-09-23
status: analysis-storage-blocked
scope: qwen3.8-flash-next (AtomicChat M64, qwen4exp) only; qwen3.8-27b / MiMo-v2.6-flash context noted where shared; DeepSeek/DSV4 retired — no K216 e2e work is proposed
author-role: B70 hardware architect (read-only review; no GPU loads, no builds, no state changes)
---

# Flash-Next M64 RAM-offload levers — 2026-09-23

> **BOXED STATUS — READ FIRST.**
>
> - The original M64 artifact (AtomicChat `Qwen3.8-Flash-Next-GGUF` rev
>   `142262902a46f7daed19c79d0771534c8106ad59`, 33 shards, 94,525,394,976 B /
>   88.03 GiB) **and its SHA identity manifest were deleted in the 2026-09-21
>   disk reclaim** (`campaign-20260831/artifact/` is now empty — verified
>   2026-09-23). The staged NVMe/SATA trees and `shard*-gguf-dump.json` are gone
>   with them.
> - **Re-download is storage-blocked by the 2x headroom policy**: the largest
>   free mount is `/dev/nvme1n1p1` at **105 GiB** (then sata-tforce 87 GiB,
>   sata-wd 84 GiB — `df` 2026-09-23); 88.03 GiB wants ~176 GiB.
> - **Therefore no cell in this memo can currently be re-measured.** Everything
>   below is the ranked plan for *once the artifact is safely re-acquired*,
>   plus the subset that needs **no artifact at all** (test-backend-ops and
>   CPU-side work — marked NOW).
> - All numbers are lab-internal. **PROVISIONAL — NOT FOR PUBLIC HEADLINE.**
>   No public claims, no XeCores/catalog writes, nothing citable.

## 1. What stands (archived, not re-measurable today)

Leader route = upstream `52d4268` + `qwen4exp-mtp-draft-head.patch` +
`sycl-fused-mmvq-mt.patch`, dzannotti Q4_K_M draft on SYCL1, PLE pre-warmed
(cookbook `intel-arc-pro-b70-inference-cookbook/docs/qwen38-flash-next/QWEN38-FLASH-NEXT-LLAMACPP.md`, 2026-09-11
update). Recipe skeleton: dual B70 layer-split `--tensor-split 1,1`, PLE-only
CPU (`-ot per_layer_token_embd=CPU`), q8_0 K + q4_1 V, FA on, `-t 6 -tb 14`,
195 W.

| Cell (C1 engine timings) | Value | Evidence |
|---|---:|---|
| Decode p512/g128 8K, code / prose (n=5) | **33.25 / 30.68** | cookbook MTP table; LocalMaxxing `cmtwyygmq0ah0ps01vvutjf7u` |
| Same, no-spec new build (n=3) | 26.86 | cookbook |
| Verify round after multi-token kernel | 118 → ~85 ms | cookbook "Why it is faster" |
| Prefill 16K actual p9096/g128, F16 (n=3 / n=5 old build) | **585.9 / 594.49** | cookbook; `campaign-20260831/run/n5-fused-f16-t6-tb14-c16k-b3072-20260901T064001Z/` |
| Short-prompt cold input p512 8K (n=5) | 183.98 | `…/run/n5-fused-fp32-t6-tb14-c8k-20260901T061439Z/` |
| 128K near-boundary P~120.7K decode (n=5) | 18.38 (MTP ≈ 0% vs 18.53 no-spec) | cookbook c128 record |
| 128K p9096 cold input, b1024 forced | 354.96 (vs 479.51 at 16K b3072) | `…/run/sweep-decode-t6-tb14-c131072-b1024-20260831T213045Z/` |

Already banked in the leaders (do not re-propose): fused IQ3_S/IQ4_NL MMVQ
decode (`+11.5%` n=3 isolation, upstreamed), multi-token MoE kernel (M=2..8),
`-t 6 -tb 14` thread split, GPU-side expert grouping (+0.81% prefill). Killed
generic sweeps stay killed: 195→230 W, tensor-split 48/52 *as a speed lever*,
PLE SATA-vs-NVMe, F16-vs-FP32 decode, n-gram spec, CPU pin (`COOKBOOK-DRAFT`
§7, `HANDOVER` §8-9).

## 2. The P-3b transfer question (the load-bearing analysis)

**Measured (trust):** the `fg_stage_a<block_iq3_xxs>` fused-gemm extension on
llama.cpp #29245 measured **3.60x same-session ABBA** (fused 9,530/9,367 µs vs
stock 33,929/34,041 µs; gate ≤14,400 µs passed with 1.5x margin; down op 3.70x;
distributions disjoint), **plus a decode no-regression bonus** (n=1: 166.6 vs
192.2 µs ≈ 13% faster; n=32: 1.47x). Evidence:
`results/2026/09/b70-improvement-probes-20260920/p3b-fg-gate-20260923/`
(manifest, perf-A1/A2/B1/B2, rebuild-abba.log); verdicts in roadmap
`LEVERS.md` (P-3b) and `OUTCOMES.md` (W14). The K216 e2e rung is retired with
DSV4 — **this evidence is used here only as kernel-class proof, never as a
proposal to load anything DeepSeek.**

**Why the 3.60x does NOT automatically transfer to M64 (theory, must be
measured at-shape):**

| Factor | K216 gate cell (where 3.60x was won) | M64 prefill (qwen4exp) |
|---|---|---|
| Mean rows/expert at serving ubatch | 14.2 (n=512, 216 exp, top-6) | **40** at b512, **240** at b3072 (128 exp, top-10) — 3-17x better GEMM M-regime |
| Stock-path effective rate | 1.52 TF (25x below dense) | ~9.4 TF-class sustained (est: 594.49 tok/s × ~15.7 GFLOP/token from the recorded geometry H=2560, I=640, G=128, topk=10 — `campaign-20260831/probe/flashnext-int4-moe-probe.json`; layer count re-derive from GGUF dump at re-acquisition, PLE dim implies 160) |
| F16 materialization expansion | 5.22x (IQ3_XXS 3.0625 bpw) | 4.65x (IQ3_S 3.4375) gate/up, 3.56x (IQ4_NL 4.5) down |
| Fused kernel's own measured rate | 5.45 TF at n=512 | **unknown at M64 shapes — the deciding number** |

The pathology P-3b deleted has three parts (materialization traffic, small-M
GEMM inefficiency, launch structure). At M64's M-regime the **small-M part is
largely absent** (oneMKL per-expert GEMM at M≈240 is respectable), and the
fused kernel's 5.45 TF is *below* M64's estimated stock operating point — if
the fused path cannot scale its compute with M, it **loses** at b3072. What
remains deletable is the materialization round-trip, which is **fixed per
layer per ubatch** (all 128 experts touched every ubatch): at the recorded
geometry ≈ 2.5 GB/layer F16 round-trip ≈ 4.1 ms/layer — ~0.8 ms/token at
n=512 but only ~0.13 ms/token at n=3072. **Verdict: conditional transfer —
the win, if any, concentrates in the short-prompt (p512) prefill/TTFT cell,
not the 16K b3072 cell.** The at-shape decision cell is nearly free and needs
no artifact.

Note: the P-3b decode bonus does **not** apply to M64 decode — M64's
`ne12==1` path is already fused (upstreamed MMVQ); different dispatch.
Also: the failed Signal AP-Q4_K_XL artifact is a different GGUF (Q4_K/IQ4_NL
experts, 512-expert, 6/6 DEVICE_LOST); its no-transfer conclusion
(`OUTCOMES.md` line 93) says nothing about M64, whose gate/up **are** IQ3_S.

## 3. Ranked levers (A-F)

### Rank 1 — (A) Expert-kernel transfer: IQ3_S + IQ4_NL templates on the #29245 fused-gemm path (prefill `ne12>1` only)

- **Measured headroom:** P-3b 3.60x/3.70x at the analogous op (§2). M64 has
  the IQ-family tensors the transfer needs.
- **Theory (label as such):** at-shape fused 1.5-3x on the n=512 routed op if
  the kernel's compute scales with M; e2e **p512 cold-input (TTFT) +15-40%**
  (184 → ~210-260), p9096 +10-25%; b3072 cell likely small.
- **Cheapest probe (NOW — zero artifact, zero model load):** add ~10 lines of
  M64 cases to `tests/test-backend-ops.cpp` (pattern: existing K216 cases)
  — `MUL_MAT_ID(type_a=iq3_s,type_b=f32,n_mats=128,n_used=10,b=0,m=640,n=512,k=2560)`
  (+ n=3072, and `type_a=iq4_nl,m=2560,k=640` twins) — run stock baseline
  (`test-backend-ops perf -b SYCL -o MUL_MAT_ID -p 'type_a=iq3_s'`,
  `ZE_AFFINITY_MASK=0`, minutes/cell). Then extend `fg_block_traits` with
  `block_iq3_s`/`block_iq4_nl` + the bit-parity oracle (Gate-1 pattern:
  0-mismatch FP16 vs `dequantize_block_iq3_s<half>`) and rerun fused.
- **Stop rules:** (i) stock at-shape ≥ ~10 TF effective → **ledger-close the
  lever** (negative result, cheap); (ii) fused at-shape < 1.5x vs stock after
  one bounded revision → stop (P-3b rule analog); (iii) two root-cause parity
  failures → stop; (iv) e2e gate: matched p512 cold-input + p9096 n=3 with
  ≥15% → fresh n=5, mandatory decode p512/g128 no-regression; any VRAM-gate
  breach → revert hook behind env guard.

### Rank 2 — (B) MTP draft-path schedule re-sweep (post-kernel) + draft placement

- **Measured adjacent:** MTP tuning banked +28% pre-kernel (26.86→29.90→33.25
  chain); the kernel then cut verify 118→85 ms — **n-max / p-min / draft-device
  were never re-swept after the cost structure changed.** Draft is Q4_K_M
  2.44 GiB on SYCL1 (cookbook recipe delta).
- **Theory bound (excluded):** "h-to-h crossover" — the layer-split host-staged
  activation copy is ~20 KB/verify-round against an 85 ms round → ≤0.5%;
  **do not spend a session on it.** The schedule levers that can matter are
  n-max/p-min economics and draft placement (SYCL1 vs SYCL0 vs CPU).
- **Expected:** +3-10% decode (33.25 → ~34-36 class); placement variant doubles
  as the VRAM liberator for Rank 3.
- **Cheapest probe (post-re-acquisition):** one load, trimmed grid — n-max
  {2,3,4} at p-min 0.75 draft SYCL1, then winner × draft-device {SYCL0, CPU};
  matched p512/g128 code+prose n=3, standard cache/entropy guards.
- **Stop rule:** no cell ≥ +5% median over 33.25/30.68 → retain current config,
  no n=5; any acceptance/quality regression → revert.

### Rank 3 — (E) 128K long-context batch-fit recovery

- **Measured headroom:** same recipe measures **354.96** (c131072, b1024
  forced) vs **479.51** (c16, b3072) cold input — a **+35% gap** that exists
  only because `-b 1536+` fails the memory-fit guard at 128K with `-ngl 99`
  pinned (`HANDOVER` §9: "Capacity: 128K + b1536 rejected; b1024 loads").
  MTP@128K is capacity-only (GPU1 transient free 391 MiB).
- **Theory:** KV at 128K is small (~1.35 GiB/card from post-load free deltas
  3916→2538 MiB), so the **compute buffer** is the binding term; freeing GPU
  VRAM (draft relocation from Rank 2, or letting the guard fit) should admit
  b1536/b2048 → 128K prefill **355 → ~430-480**.
- **Cheapest probe:** no-spec c131072 b1536 with `--tensor-split 48,52`
  (speed-neutral: +0.06% measured) and `-ngl` omitted so the guard fits;
  audit how many layers landed on CPU (abort-request if decode p512 regresses
  >3%); then the MTP variant with draft on the freer card.
- **Stop rule:** both placement variants rejected by the guard, or post-load
  free <2,048 MiB (contract §5), or decode regression >3% → 128K stays b1024
  capacity-class; record and stop.

### Below the cut

- **(F) qwen4exp oneAPI runtime fix — GATE-ZERO precondition, not a speed
  lever.** Last M64-class load was 2026-09-11; oneAPI 2026.1 landed 2026-09-17.
  The Signal failures (6/6 DEVICE_LOST incl. a matched-2026.1.1 rebuild,
  `worklogs/FLASHNEXT-qwen4exp-fix-20260923.md` §2/§8) proved that crash class
  is *not* runtime drift — but whether **M64** binaries still load today is
  simply untested. Probe (first GPU act after re-acquisition, ~30 min): one
  smoke load of the MTP-route binary + correctness trio; on DEVICE_LOST →
  rebuild `52d4268`+patches on 2026.1.1, retest, then 2026.0 lib pin. Nothing
  else in this memo runs before this gate passes.
- **(C) CPU PLE mmap/NUMA/readahead — rank 4, variance-only.** Host is now
  **60 GiB (52 available)**, so the 35.76 GiB Q5_1 PLE fits page cache; the
  MTP route already pre-warms it. Median upside ~0 (NVMe-vs-SATA A/B was
  +0.4-1.4%, not material); residual value is tail hygiene — the 13.45 tok/s
  PLE-miss outlier class (`HANDOVER` §2) and 2.9-8.0 MiB/request reads under
  memory pressure (swap currently 6.1 GiB used). Cheap check only: PLE
  residency + per-sample disk-read deltas during long cells; swap discipline
  during campaigns. NUMA is void — single-socket 5700X3D. Draft-token PLE
  prefetch during verify is a theory note, not a scheduled lever.
- **(D) partial CPU expert offload — DEAD for speed (measured):** 15 CPU
  expert layers → 7.75-13.41 tok/s vs 17-23 all-GPU (`HANDOVER` §4); rejected
  in `COOKBOOK-DRAFT` §7. Retain only as the emergency capacity fallback if
  Rank 3's fit juggling fails.

## 4. Sequencing

**NOW (no artifact, no model loads):** Rank 1 at-shape baseline cells and the
`fg_block_traits` IQ3_S/IQ4_NL templates + parity oracle (single-GPU
test-backend-ops only); this either banks the transfer or kills it cheaply
before the artifact question is even solved. Shared infrastructure note: the
templates are reusable for any IQ/K-quant mixed GGUF in scope (notably if
MiMo-v2.6-flash lands in that representation — its FP8 dense-MLP layer is
unrelated).

**POST-RE-ACQUISITION:** (F) gate-zero smoke → Rank 2 grid (one load) →
Rank 3 fit probes (no-spec load + MTP variant) → Rank 1 e2e A/B only if the
at-shape gate passed → fresh n=5 on winners, claims audit, then the usual
publication gates. Every campaign: empty-GPU gate before, 150 W restoration
and gates after (contract §4/§13).

## 5. Evidence index

- P-3b kernel proof: `results/2026/09/b70-improvement-probes-20260920/p3b-fg-gate-20260923/`
  (manifest.txt, perf-A1/A2/B1/B2, rebuild-abba.log); `LEVERS.md` P-3b row;
  `OUTCOMES.md` W14; spec `worklogs/KERNEL-grouped-iq3-spec-20260923.md`;
  #29245 review `worklogs/P3-review-pr29245-20260922.md`.
- M64 recipe + archived cells: `/mnt/<nvme-disk>/campaign-20260831/COOKBOOK-DRAFT-QWEN38-FLASH-NEXT.md`,
  `…/HANDOVER.md`, public page
  `~/intel-arc-pro-b70-inference-cookbook/docs/qwen38-flash-next/QWEN38-FLASH-NEXT-LLAMACPP.md`.
- Geometry: `campaign-20260831/probe/flashnext-int4-moe-probe.json`
  (H=2560, I=640, G=128, topk=10; re-verify layer count from GGUF dump at
  re-acquisition).
- Signal/oneAPI blocker (distinct artifact — do not conflate):
  `worklogs/FLASHNEXT-qwen4exp-fix-20260923.md`.
- Storage state (2026-09-23): `df` — nvme1n1p1 105 GiB free (largest);
  `campaign-20260831/artifact/` empty.

Ongoing label for everything here and everything it cites:

> PROVISIONAL — NOT FOR PUBLIC HEADLINE
