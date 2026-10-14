---
title: "B70 next quant-format lever — audit of what worked/failed + ranked new-format candidates (2026-10-09)"
type: research
date: 2026-10-09
status: provisional
owner: b70-quantization-engineer
---

# B70 next quant-format lever — audit and ranked candidates (2026-10-09)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Zero-GPU source audit: registry + ledger reads,
engine-source dispatch verification, arithmetic. No downloads, no conversions, no model
loads, no authority edits. Tasking: parent orchestrator (find the next BIG quant-format
lever; evidence-first; rank candidate NEW formats; flag operator-held items).

## 0. Method and evidence base

Law (all pricing obeys it): `decode tok/s = yield × achievable_BW / bytes_per_token`.
Pin floor 608 GB/s/card. Measured per-byte efficiencies on this host: Q8_0 ≈ 396 GB/s,
Q4_K_M ≈ 336 GB/s (showdown §4), champion GPTQ auto_gptq packing ~430 GB/s (RL-087),
EXL3 4bpw 381 GB/s in the M=4 verify pass at 230 W (probe-1 attribution) / 255 GB/s at
150 W on the RL-087 harness. **Format choice moves achievable_BW (dequant cost per byte),
not just bytes** — candidates are priced on both.

Bandwidth-base discipline (lead correction, carried): the B_FAKE 636 GB/s figure is
**logical-byte** throughput (can exceed the 608 pin; counts cache hits). The EXL3
instruction-issue tax is therefore cited two ways: **37.3 % pin-based (1−381/608)** and
**40.1 % vs the B_FAKE logical base (1−381/636)**.

Read this session: `registry/model-status.json`; ledger RL-004/034/041/070/087/092/094/
097/113/114/115/117/121/122/125; `research/2026/b70-format-route-lessons-20260930.md`;
`b70-sub4bpw-and-margin-requant-plan-20260929.md` (+ addendum);
`flashnext-strata-source-math-20261009.md`; `quant-format-showdown-20260913/FINDINGS.md`;
`iq3-xxs-upstream-contribution-package-20260923.md`; `xmx-dpas-rate-classes-20260915.md`;
authorities (quantization decision record v1.3, KV reference, engine support matrix).
**Engine-source verification (this session, decisive for ranking):**

1. **Strata v0.1.40.3** (`/mnt/<sata-disk>/strata/repo/strata-0.1.40.3`,
   `include/strata/kernels/iq_kernels.hpp` + `native_mmvq.hpp` + `artifact/dequant.hpp`):
   expert/MMVQ type set = **IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, Q2_0**
   (iq path) + **Q2_0, Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS** (native
   path). Header comment: UD-Q4_K_XL mixes (gate/up Q4_K/Q5_K, down Q5_1/Q8_0, Q8_0 embed)
   explicitly supported. **Q2_K is NOT in the expert set.** `native_expert_supported()`
   gates every layer at startup (fail-closed, before allocation).
2. **llama.cpp fleet fork** (`~/llama.cpp` @`1692f9e50`, the showdown build):
   SYCL MMVQ dense decode dispatch (`ggml/src/ggml-sycl/mmvq.cpp:2238-2600`) covers
   Q4_0…Q8_0, Q1_0, Q2_0, Q2_K…Q6_K, **IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
   IQ3_S, IQ4_NL, IQ4_XS, MXFP4, NVFP4** (multi-col ≤8 and single-col paths). FP4 vec_dot
   (`vec_dot_mxfp4_q8_1` / `vec_dot_nvfp4_q8_1`) present in this fork's lineage.
3. **vllm-xpu-kernels** (`~/vllm-src/vllm-xpu-kernels`): `csrc/quantization/`
   contains **fp4 (mxfp4_quant, fused_silu_mul_mxfp4_quant), fp8, fused** — **no INT4/W4A16
   linear kernels here**; MXFP4 GEMM is a true W4A4 path (`torch.ops._xpu_C.fp4_gemm`,
   e2m1x2). **TurboQuant KV does not exist in this kernels repo** — it lives in vLLM core
   as Triton ops (`vllm/v1/attention/ops/triton_turboquant_*`), with **no XPU gating found
   in those files** → XPU dispatch unverified; treat as CUDA/CPU-likely.

## 1. Verified state of the two adopted routes (numbers with cells)

### Dense — EXL3 4bpw Qwen3.8-27B, exl3xpu, one card (ADR-0010)

- 63.25 t/s p512/g128 usage-token-rate n=5 @230 W; MTP3 accept 2.93–3.16; fp8 KV.
- Round bytes (derived from safetensors headers, sub4bpw plan §0): body 12.2295 +
  head 2×0.9541 + MTP 0.2126 + fp16 embed 2.5428 = **16.89 GB/round** (r_h=2). Closure
  check: 381 GB/s × 3.0 / 16.89 = 67.7 t/s vs 63.25 measured — consistent within yield
  uncertainty. Ceilings: 608/16.89×3 = 108 t/s (pin), 636/16.89×3 = 113 t/s (logical).
- **Instruction tax 37.3 % pin-based / 40.1 % B_FAKE-based**; trellis state-build +
  codebook decode + operand moves ≈ 336 instr per 512 B tile-group (~8.6 G instr/pass
  at M=4); ccs0 ~88 % busy. B_FAKE (+18/+21/+31 % GB/s on down/gate-up/qkv cells) and
  A_ONCE (+1.6/+2 %) bracket it: activations are not the wall, trellis decode work is.
- Sub-4bpw quality is PROVEN at 3.00 bpw (RL-094: battery 6/6 incl. code_execution 14×2,
  tasks 9/10, replay stable — weights identical to what ESIMD K=3 would serve) — but the
  K=3 ESIMD kernel program is OPERATOR-STOPPED (RL-122) and the Triton fallback served
  10.25 t/s (diagnostic-only). int8-prefill is K∈{4,6}-only → 3bpw also loses prefill.

### MoE — Strata IQ3_XXS GSQ-RCO Flash-Next, one card (ADR-0011, 2026-10-09)

- 48.03 t/s p512/g128 engine decode-wall n=5 (47.77–48.12); 8K prefill 989.11 t/s;
  MTP --spec 4 --spec-min-p 0.5; int8 KV; battery 3/3 exact at gen=64 only.
- Expert store: 24,576 experts × **1.75 MB** (IQ3_XXS; calibrated: 14,544 resident
  slots in the 23.60 GiB cache, 10,032 mirrored in 16.37 GiB pinned host RAM — both
  reproduce 1.75 MB/expert; INTEL.md's 1.97 MB is the IQ3_S figure). Residency 59 %.
- Round model: measured round ≈ 56 ms (=2.7 accepted / 48.03 t/s), external E0 class
  65–89 ms. Miss stream at the sensitivity point: 135 misses × 1.75 MB = 236 MB
  @11.5 GB/s = **20.6 ms/round ≈ 33–37 % of the round** (INTEL.md IQ3_S basis: 266 MB,
  23.1 ms). Dequant per expert: 42 µs gate/up vs GEMM 20 µs — dequant ≈ 2/3 of expert
  time. **Both timings come from the same A750-measured INTEL.md sentence (capture
  sha256-a394b896, "Measured on the A750") — E0 cross-card, not B70-measured**; the
  derived "~200 GB/s effective vs a 608 GB/s card" therefore compares an A750 rate to
  B70 pin and is indicative only. Their fused dequant+XMX kernel is 0.22× of the pair
  (same A750 context, E0). The directional claim (dequant dominates the per-expert
  pair) stands; B70 dequant bandwidth is unmeasured. (Provenance corr. 2026-10-09
  post-audit.)
- Active bytes/token ≈ 2.30 GB (≈6 B active × 3.0625 bpw). No-spec pin ceiling 264 t/s;
  the measured 48 t/s is dominated by the miss stream + dequant instruction tax, not by
  pin bandwidth. Prefill is compute-bound (989 vs ≤1,766 transfer bound → ≥1.8× margin).

## 2. Deficiency list — where current formats leave performance on the table

- **D1 — Dense EXL3 instruction tax (the frontier).** 37.3 % pin / 40.1 % logical of
  bandwidth lost inside kernels at unchanged bytes. A same-bytes format with cheaper
  dequant would recover it (Q8_0 396, GPTQ ~430 GB/s per-byte vs EXL3 381) — but every
  cheaper-dequant format measured on this model failed quality (W4A16 class: 30-defect,
  RL-087/092/117) or is loader-blocked (AWQ RL-034). The kernel-side fix (VERIM4,
  ~76–79 t/s ceiling) is operator-stopped with RL-122.
- **D2 — Dense bytes floor locked behind the same stop.** body 4→3 bpw is 92 % of the
  available byte saving (−3.04 GB/round, ideal ×1.219–1.283) and its QUALITY is already
  proven (RL-094); only the ESIMD K=3 path and int8-prefill coverage are missing.
  Operator-held; do not plan.
- **D3 — MoE Flash-Next pays three format-addressable taxes:** (a) 41 % of experts
  non-resident → 20.6 ms/round PCIe miss floor at the sensitivity point (33–37 % of the
  round); (b) dequant = 2× the GEMM per expert (instruction frontier again); (c) 16.37 GiB
  pinned host RAM consumed by the mirror. Fewer bytes/expert shrinks all three at once.
- **D4 — No admissible KV lever at current cells.** exl3xpu KV is fp8-only (engine);
  Strata int8 KV is 0.2 GB at the adopted 10K cell (negligible); q4_0 K prohibited
  (fleet policy, catastrophic cliff); Strata k8v4 measured −11 % on prefill (881 vs 989)
  with decode untested. KV moves only matter at ≥100K ctx cells and only with K protected.
- **D5 — Quality gates open on the adopted MoE route.** ADR-0011 secondary gates
  (representative task quality, long/multi-turn, replay drift) rest on 3 exact-answer
  cases. This gates every sub-IQ3 candidate below and makes UD-Q4_K_XL (already supported
  by Strata per source) the pre-wired quality fallback.
- **D6 — MoE vllm-xpu (GPTQ-INT4) is dispatch-bound, not format-bound.** Nemotron 186.6
  t/s = 19 % of its no-spec pin ceiling; K216 lesson (84+ host syncs/token; grouped GEMM
  1.79 vs ~46 TFLOPS dense). No weight-format change fixes dispatch; that route's next
  lever is kernel/fusion work, outside this audit.
- **D7 — Per-byte kernel efficiency is a hidden format axis.** Measured spread 336→430
  GB/s across formats at similar bpw (Q4_K_M → GPTQ packing). Ranking candidates by bpw
  alone overprices LUT-heavy formats; each candidate below carries an explicit efficiency
  assumption.

## 3. Ranked candidate list (NEW formats)

### C1 — IQ2_XXS / IQ2_S experts for Flash-Next on Strata (rank 1)

- **Bytes/token:** experts 1.75 → 1.18 MB (IQ2_XXS 2.0625 bpw) / 1.42 MB (IQ2_S 2.5 bpw);
  active weights 2.30 → 1.55 / 1.88 GB per token. All-expert store 43.0 → 29.0 / 34.8 GB.
- **Effect on the measured walls (IQ2_XXS):** resident slots in the same 23.60 GiB cache:
  14,544 → **~21,500 (59 % → 87.5 % residency)**; pinned mirror 16.37 GiB → **~3.4 GiB**
  (frees ~13 GB host RAM); sensitivity miss stream 236 → 159 MB = 20.6 → **13.8 ms/round**
  → **+11–14 % decode bound** at the sensitivity point (more when miss counts exceed 135;
  content-dependent). Prefill chunk stream 23.6 → 15.9 GB (bound ×1.48, but prefill is
  compute-bound → modest).
- **Xe2 dequant cost class:** same vecdotq lineage as IQ3_XXS (grid+sign LUT) — per-byte
  instruction cost ≥ IQ3_XXS, but −33 % bytes → net fewer instructions AND fewer bytes
  per token; does not fix the 42 µs-vs-20 µs dequant ratio itself.
- **Quality risk: HIGH.** 2.06 bpw post-train on a non-QAT model. Local precedent: EXL3
  3bpw passed the full battery on dense (RL-094) but IQ3_XXS itself has only 3 exact
  cases (RL-125). MoE expert redundancy helps; nothing below a full battery clears it.
- **Engine port path:** Strata — **kernels already in-engine (verified from source**;
  `iq_supported` covers IQ2_XXS/XS/S/IQ1_M; startup gates every layer). Not Q2_K. Not a
  llama.cpp speed route (21.6 t/s placement class, RL-070/121).
- **Cheapest falsifying probe:** (zero-GPU) HF artifact-existence check for a Flash-Next
  IQ2-class GGUF (GSQ-RCO or ggml-org/Unsoth lineage) + confirm the pack loads through
  `iq_supported`/`native_expert_supported` — if no artifact exists, the route is
  storage-gated (own conversion needs the BF16 source ≈250 GB staging; live discovery
  required, root disk was 8.3 GB free on 2026-10-09). Then **quality-first**: exact
  battery + replay in one bounded window BEFORE any speed claim (RL-094 ordering).
  Kill: any wrong exact answer.

### C2 — IQ4_XS GGUF for Muse-Glimmer on llama.cpp SYCL (rank 2)

- **Bytes/token:** −12 % vs Q4_K_XL (4.25 vs ~4.83 bpw) → ideal +14 %; priced at Q4_K-class
  per-byte efficiency with an IQ-LUT haircut → **+8–12 %**: 26.8 → ~29–30 t/s p512/g128
  @128K. Modest, but on the only route with no EXL3 alternative.
- **Xe2 dequant cost class:** IQ4_NL 16-entry LUT ×2 lookups/byte — mid class; **SYCL
  MMVQ dense decode verified in the fleet fork** (`mmvq.cpp:2548`, single- and multi-col).
- **Quality risk: LOW-MEDIUM** at 4.25 bpw with imatrix. Rules: **never run IQ quants
  without imatrix** (showdown); per-position logit parity vs the Q4_K_XL reference, not
  PPL (PPL non-monotonic on hybrid-GDN; Muse class unproven → use parity regardless).
- **Engine port path:** llama.cpp SYCL (fleet fork @1692f9e50) — no port needed.
- **Cheapest falsifying probe:** imatrix build + 315-position teacher-forced logit-parity
  vs the incumbent Q4_K_XL artifact (CPU-class or one bounded window); kill on
  top-1 agreement loss beyond the pre-declared budget, speed never measured.

### C3 — Official QAT artifacts (Qwen INT4-QAT class) for the dense 27B (rank 3)

- **Bytes/token:** GPTQ-INT4 class ≈ 16.5 GB/round — **no byte win vs EXL3**; the prize is
  **per-byte efficiency + vLLM-native stack**: champion-class auto_gptq packing measured
  ~430 GB/s (RL-087) vs EXL3's 381 → 430×3/16.5 ≈ **78 t/s potential** (vs 63.25), plus
  no isolated dockerd/fork serving constraints.
- **Xe2 dequant cost class:** fused dequant-in-GEMM feeding DPAS (the MoE GPTQ path's
  measured class) — the cheapest per-byte class measured on this host.
- **Quality risk: THE gate.** ADR-0010 closed *post-train* W4A16 on this model (30-defect
  proven weight-inherent, RL-092; margin-requant repairs failed transfer, RL-113/114/115
  — scale-only repair retargets represented trajectories 4/4). **QAT is a different
  artifact basis, not a reopen of the closure**: weights trained into the INT4 grid.
  Still must clear the full 6/6 battery incl. code_execution 14×2 + ≥8/10 tasks.
- **Engine port path:** vllm-xpu W4A16 compressed-tensors (kernel PR #33973 merged
  2026-09-23 per worklog; **verify current image dispatch from source/logs before any
  run** — my charter; the vllm-xpu-kernels repo has no INT4 linear kernels in
  `csrc/quantization`, so the serving image's actual dispatch must be confirmed).
- **Cheapest falsifying probe:** (zero-GPU) watch HF for an official Qwen3.8-27B QAT INT4
  release; when it exists, CPU-side faithful-dequant battery screen (RL-092 method,
  fp32 oracle) before any GPU minute.

### C4 — MXFP4-native QAT artifacts (gpt-oss-20B class; MiMo MXFP4 GGUF) (rank 4)

- **Bytes/token:** MXFP4-G32 ≈ **+3.0 % vs INT4-G128 — no byte win** (scale-traffic math,
  showdown §1). The value is artifact-side: QAT-native MXFP4 weights are native precision
  ("QAT-lossless" per the MiMo audit), opening an FP4-QAT model class, not a speed
  multiplier on current routes.
- **Xe2 dequant cost class:** E2M1→FP16 LUT + E8M0 scale — cheap-ish; **no `dpas.f4` on
  Xe2 → no rate-class win** (rate-classes law; NVFP4's 2× is Blackwell-only). llama.cpp
  SYCL: dense MXFP4 MMVQ verified in the fleet fork; **grouped/MoE expert path (MUL_MAT_ID)
  MXFP4 coverage unverified** — the named open bit from the MiMo worklog. vllm-xpu:
  `fp4_gemm` is true W4A4 (activation FP4 quant → Decision-6-class quality risk).
- **Quality risk:** LOW for QAT-native artifacts; **HIGH for post-train MXFP4** (KLD ~2×
  worse than GPTQ-INT4, retired locally; 10.4 t/s Triton-era measurement — engine, not
  format). Post-train MXFP4 stays rejected.
- **Engine port path:** llama.cpp SYCL MXFP4_MOE (upstream B70 row exists, +24 % class on
  expert paths) via the fleet fork; verify grouped-path coverage first.
- **Cheapest falsifying probe:** static — grep/trace the fork's grouped MoE path for
  MXFP4 expert dispatch (zero-GPU); then one gpt-oss-20B MXFP4 GGUF battery window
  (16 GB-class model, fits easily).

### C5 — KV variants below int8/fp8 (rank 5 — cell-gated)

- **Bytes/token:** dense exl3xpu fp8 KV at 262K = 8.59 GB/step (co-dominant); a K-protected
  4-bit KV would halve it → long-ctx ideal ×1.19–1.22 — but exl3xpu is **fp8-only**
  (engine; escalate, not plan). vllm-xpu exposes `turboquant_k3v4_nc`/`3bit`/`4bit`/
  `int4_per_token_head` in the arg parser (local build, 2026-10 sweep log), but the
  kernels are vLLM-core **Triton ops with no XPU gating found** → likely not XPU-served.
- **Quality risk:** q4_0-K cliff stands; any candidate must protect K (k8/k3-v4 shape or
  per-token-head scales). Strata k8v4 already measured **−11 % prefill** (881 vs 989);
  decode cell untested.
- **Cheapest falsifying probe:** zero-GPU source read of the vLLM attention-backend
  wiring for turboquant on XPU (is the Triton kernel compiled/dispatched on XPU at all?);
  if absent → blocked on kernel port, close the candidate.

### C6 — TQ ternary / BitNet-class (rank 6 — watch-only)

- **Bytes/token:** ~2.1 bpw incl. scales → the largest byte cut available; dequant is the
  cheapest possible (sign+FMA, no LUT) — the only format class whose instruction tax
  plausibly DROPS below INT4.
- **Blocker:** requires QAT from scratch; no artifact for any fleet model class; no
  engine path locally (llama.cpp TQ1_0/TQ2_0 are BitNet-arch only). Re-entry trigger: a
  QAT ternary release we actually want to serve. No probe worth a GPU minute today.

## 4. Operator-held — requires operator reopening, NOT planned here

- **EXL3 3.00/3.50bpw + SC variants on ESIMD (K∈{2,3,5} all-codebooks rebuild, VERIM4
  instruction-issue rewrite, K=3 nt_of sweep):** RL-122 blocked_pattern
  `exl3xpu.*(kernel.?probe|repull|image.?pull).*(rerun|resume|restart)` unless the
  operator reopens. Standing facts if reopened: 3bpw quality already 6/6 + 9/10 (RL-094);
  K=3 P3 correctness had an intermittent MR8 excursion (RL-097); int8-prefill is
  K∈{4,6}-only (prefill regression axis); ideal ×1.219–1.283 traffic, decode-only prize.
- Nothing in C1–C6 touches the stopped program: C1/C2/C4/C5 use existing kernels in
  engines we already run; C3 waits on an artifact.

## 5. Rejected with local precedent (do not re-litigate without new evidence)

| Candidate | Why rejected | Precedent |
|---|---|---|
| NVFP4 (as target format) | +9.1 % bytes vs INT4-G128, materially worse KLD, no Xe2 rate class (no `dpas.f4`) | showdown §1; rate-classes; kept only as NVFP4→BF16 draft reconstruction |
| W4A8 / activation quant | same 363-TOPS class as W8A8 (no compute win); activations ≈ +2 % (A_ONCE); quality divergence history | Decision 6; rate classes; M4 ablation |
| Post-train MXFP4 | KLD ~2× worse than GPTQ-INT4 at iso-bytes | quantization-format-strategy; 10.4 t/s local |
| AWQ (community artifact) | 1.079× champion bytes (not iso-byte); AWQ loader path never exercised | RL-034 |
| LM-head INT4 | top-1 agreement 72 % vs 99 % gate; head is the verified distribution | RL-041 |
| UD/AP dynamic per-tensor mixes as speed lever | +10.7 % bytes → −5.8 % decode, no quality gain on hybrid-GDN | showdown G2 |
| Requant-from-quantized (any family) | approximation-of-approximation; compounds correlated error; violates lineage rule | Decision Record; sub4bpw §A2.4 |
| FP8 small-M block GEMM | xe2 grouped/dense small-M FP8 path absent | RL-004 |
| 3bpw EXL3 as a *speed* route on pinned image | silent Triton fallback = undeclared-path serving; 10.25 t/s | RL-094 addendum; RL-097 |

## 6. Cheapest next probes (decision tree, all quality-gated per §11)

1. **Now, zero-GPU:** (a) HF artifact-existence sweep for Flash-Next IQ2-class GGUFs
   (C1); (b) ADR-0011 open quality gates — close representative-task/long-turn/replay on
   the INCUMBENT IQ3_XXS before spending anything on sub-IQ3 (D5: if IQ3_XXS fails deeper
   quality, C1 dies and UD-Q4_K_XL — already loadable in Strata — becomes the fallback);
   (c) static turboquant-XPU dispatch read (C5); (d) static grouped-MoE MXFP4 coverage
   read in the fleet fork (C4).
2. **One bounded GPU window each, only after (1):** C1 quality battery (then speed cells);
   C2 logit-parity; C4 gpt-oss-20B battery.
3. **Watch:** official QAT INT4 for Qwen3.8-27B (C3) — falsify via CPU faithful-dequant
   battery before any GPU work (RL-092 method).

Storage note (no acquisition authorized this phase): any artifact acquisition needs live
discovery; last live figures 2026-10-09: T-FORCE 128 GB free, SN770 43 GB, root 8.3 GB
(tight). A own-conversion IQ2 route (~250 GB staging) is storage-blocked today.

## 7. Evidence log (this session, all read-only)

- Registries/ledger: `registry/model-status.json`; RL-004/034/041/070/087/092/094/097/
  113/114/115/117/121/122/125 from `registry/research-ledger.json`.
- Research notes: `research/2026/b70-format-route-lessons-20260930.md`;
  `b70-sub4bpw-and-margin-requant-plan-20260929.md`; `flashnext-strata-source-math-20261009.md`;
  `quant-format-showdown-20260913/FINDINGS.md`; `iq3-xxs-upstream-contribution-package-20260923.md`;
  `xmx-dpas-rate-classes-20260915.md`.
- Results: `results/2026/09/b70-m4-flag-ablation-20260930T143631Z/ABLATION-RESULT.json`
  (bfake +18/+21/+31 % GB/s; aonce +1.6/+2 %); `results/2026/10/rq14-strata-matched-screen-20261009/`
  (via RL-125); `results/2026/10/b70-sweep/kv-bf16.json` (local vLLM KV dtype list).
- Engine sources: Strata v0.1.40.3 headers under `/mnt/<sata-disk>/strata/repo/`;
  llama.cpp fork `~/llama.cpp` @`1692f9e50` `ggml-sycl/mmvq.cpp`;
  `~/vllm-src/vllm-xpu-kernels/csrc/`; `~/vllm-src/vllm/v1/attention/ops/`.
- Authorities: `docs/reference/quantization-decision-record.md` (v1.3),
  `docs/reference/kv-cache-reference.md`, `docs/reference/engine-support-matrix.md`.

PROVISIONAL — NOT FOR PUBLIC HEADLINE. No GPU touched, no artifact modified, no
authority edited, nothing downloaded.
