---
title: "Ornith-1.5-35B-A3B AutoRound-vs-GPTQ head-to-head — 2026-08-21/22 (COMPLETE)"
type: research
date: 2026-08-22
status: final
---

# Ornith-1.5-35B-A3B AutoRound-vs-GPTQ head-to-head — 2026-08-21/22 (COMPLETE)

## Outcome

| Gate | Status |
|---|---|
| Smoke conversion (iters=2) | ✅ contract 30720/0/785, 17 min |
| Full conversion ar-r1 (200 iters, 128×1536 MixedCal-v2) | ✅ 3 h 26 m, contract 30720/0/785, 24.45 GB, zero RTN fallback |
| vLLM XPU serve (`--quantization gptq`) | ✅ champion image/pattern, KV 169,622 tok @ 32K (identical to control) |
| No-spec C1 screen | ✅ parity: AR 70.1 vs GPTQ 69.0 t/s (p512/g128) |
| MTP1 speculative decoding | ✅ works with same patches; accept 78.5% vs 80.3% pos0; t/s parity |
| Behavioral quality probe | ✅ 8/8 coherent both artifacts |
| **Reference parity vs BF16 (2026-08-22)** | ✅ **AR equal-or-best on all three metrics** — see §4b |

**Verdict (diagnostic):** AutoRound is a drop-in quality-motivated replacement
for the GPTQ conversion — identical serving behavior and speed class, with the
24.75% RTN-fallback debt eliminated and every expert calibrated by signSGD.
Not a speed change; not a proven accuracy gain until a reference-parity gate
(KL/perplexity vs BF16) is run.

**PROVISIONAL n=3 — NOT FOR PUBLIC HEADLINE.**
Evidence: `results/ornith15-ar-headtohead-*/` (candidate) and
`results/2026/08/ornith15-35b-20260819T1630Z/` (GPTQ control).

## 1. Motivation

The GPTQ MixedCal-v2 control conversion (2026-08-19) recorded GPTQModel's
low-observation **RTN fallback for 7,605 / 30,720 expert projections (24.75%)**,
rising to 38.7% in layer 39. AutoRound signSGD tuning is the candidate fix:
same dynamic scope (experts-only), same calibration corpus (MixedCal-v2,
sha `e88ccd5f…`), same W4A16 symmetric group-128 target with MTP left BF16 —
**the converter is the only changed variable**.

## 2. Candidate identity

| | AutoRound candidate | GPTQ control |
|---|---|---|
| Artifact | `…-AutoRound-W4A16-sym-G128-MTP-BF16` | `…-GPTQ-Int4-sym-G128-MTP-BF16` |
| Converter | auto-round 0.14.2 (pinned), signSGD 200 iters, 128×1536 MixedCal-v2 | GPTQModel (image default) |
| Packing | auto_gptq qweight/qzeros/scales, desc_act=false | gptq, desc_act=false |
| Scope | 30,720 expert qweights; attn/GDN/shared/router/visual/lm_head/MTP BF16 | identical |
| Contract | 30720 qweights / 0 leaks / 785 MTP BF16 / 24.45 GB | 30720 / 0 / 785 / 24.42 GB |
| Fallback | none — all experts tuned (per-block loss logged) | 24.75% RTN |

Smoke (`ar-smoke`, iters=2) validated the full pipeline first: contract
30720/0/785, published atomically; smoke artifact preserved at
`/mnt/<other-disk>/ar-artifacts/…-SMOKE-iters2`.

## 3. Getting it to run at all on a 30 GiB-RAM host (failure ledger)

Fixed in order, each verified by a fresh run:

1. `scheme` dict → string `"W4A16"` (auto-round 0.14.2 `parse_scheme`
   `KeyError: 'data_type'` on dict input).
2. `ignore_layers` list → comma-separated string
   (`utils.py` `set_layer_config` does `str.replace/split`).
3. Calibration samples `{"input_ids": [list]}` → batched `torch.int64`
   tensor (MLLM mode forwards dicts verbatim into `model(**data)`).
4. **Host OOM reboot**: `qwen3_5_moe` is absent from auto_round's
   `MODEL_CONFIG`, so `apply_model_monkey_patches` returns False and
   `data_driven.py` *silently* disables `low_cpu_mem_usage` — no per-block
   release — host RAM ratchets ~2 GiB/block (24.2 GiB peak at block 17),
   kernel OOM cascade killed sshd/dockerd and the `oom-watchdog-reboot`
   service emergency-rebooted the host (evidence:
   `/mnt/<models-disk>/ornith15-quant/mixedcal-v2/runs/ar-smoke-killed-oom-reboot`, probe logs under
   `/mnt/<models-disk>/ornith15-quant/mixedcal-v2/runs/ar-mem-probe/`). Fix: force `ar.model_context.is_model_patched = True`
   after init — the flag's only consumer is the offload gate; compute path
   unchanged (probe2: anon flat 7.7 GiB through block 10).
5. `AR_WORK_SPACE` defaults to a *relative* `ar_work_space` (EROFS under
   the read-only `/quant` CWD) and its disk check needs 1.2× the full 72 GiB
   block state: pointed it at the NTFS volume
   (`/mnt/<other-disk>/ar-work-space`, 159 GiB free).
6. Save-phase OOM ×2: `save_quantized(inplace=False)` deep-copies, and even
   inplace packing-assembles the ~24 GiB model in RAM. Fix:
   `quantize_and_save(output_dir, format="auto_gptq", inplace=True)` — binds
   output_dir/formats *before* `post_init`, activating
   `is_immediate_saving`: each quantized block streams to shards via
   ShardWriter during tuning. Peak RAM 16.3 GiB flat.
7. Contract check: `quantize_and_save` nests the artifact as
   `<STAGING>/<slug>-w4g128/`; verifier + publish now resolve the nested dir.
8. Host protection added: `--memory 24g --memory-swap 34g` cgroup cap on the
   conversion container + `ar-host-watchdog.sh` (docker-stops the exact
   container below 7 GiB MemAvailable+SwapFree). No further host incidents.

Post-reboot recovery: NTFS backing volume (sdb2) does not auto-mount; the
80 GiB ext4 loop workspace and its backing image survived the unclean
shutdown (journal replay), all run evidence intact.

## 4. Head-to-head results (PROVISIONAL)

Server: champion vLLM XPU image `f01e24f6…`, `--quantization gptq
--dtype float16`, U=0.85, block-size 64, prefix-caching off, 230 W cap,
C1, unique-entropy prefixes, cache-hit deltas recorded by `c1-screen.py`.

| Cell | post-first median t/s | cold-input rows t/s |
|---|---:|---|
| GPTQ no-spec p512/g128 n=3 | 69.0 | 6436 5809 6257 |
| **AR no-spec p512/g128 n=3** | **70.1** | 6484 6152 6740 |
| GPTQ no-spec p2048/g1 n=3 | 9,537 (med) | 8572 9537 9616 |
| **AR no-spec p2048/g1 n=3** | — | 8271 9685 9560 |
| GPTQ MTP1 accept filler n=3 | 94.3 | 94.2 95.4 94.3 |
| **AR MTP1 accept filler n=3** | **95.2** | 92.6 96.4 95.2 |
| GPTQ MTP1 accept real n=3 | 94.3 | 98.4 90.0 94.3 |
| **AR MTP1 accept real n=3** | **96.3** | 96.5 87.6 96.3 |
| GPTQ MTP1 screen p512/g128 | 94.3 | — |
| **AR MTP1 screen p512/g128** | **92.4** | 5756 5306 6301 |

- **Speed: parity.** Same packing ⇒ same throughput class; deltas (±2%) are
  inside n=3 spread. KV footprint identical: 169,622 tokens @ 32K config.
- **MTP1 acceptance:** AR 78.5% pos0 (mean accepted len 1.79) vs control
  80.3% pos0 — slightly lower, decode throughput unaffected.
- **Quality (behavioral, not reference parity):** 8/8 prompts produced
  coherent on-task reasoning traces on both artifacts; 0/8 identical greedy
  texts (expected for different 4-bit weight sets). Neither artifact
  degenerate. Tuning-loss evidence: deep-block losses cut 70–83% by signSGD
  (e.g. 0.0094→0.0016, 0.0038→0.0012) with **zero RTN fallback**
  (30,720/30,720 experts tuned) vs the control's 24.75% fallback.
- Conversion cost: 3 h 26 m wall (40 blocks, 200 iters, 128×1536 calibration),
  peak host RAM 16.3 GiB flat, peak VRAM 11.5 GiB, artifact 24.45 GB.

## 4b. Reference parity vs BF16 — exact numbers (2026-08-22)

Method: the 72 GiB BF16 source cannot be served whole on this host, so the
reference is a **streaming teacher-forced forward** (meta-device skeleton,
one block materialized at a time from file-backed memmaps of the source
shards; peak cgroup anon well under the 20 GiB cap) over 12 fixed
out-of-calibration prompts (315 predicted positions). The two quantized
artifacts were served by the pinned vLLM nightly (identical champion
pattern, WNA16 MoE backend + Flash Attention confirmed in both server logs)
and queried with `prompt_logprobs=20` on the **same token ids**.

| Metric vs BF16 (315 positions) | AutoRound | GPTQ MixedCal-v2 |
|---|---:|---:|
| mean \|Δlogprob\| actual tokens (primary) | **0.20059** | 0.20250 |
| median \|Δlogprob\| | **0.07931** | 0.10391 |
| p95 \|Δlogprob\| | 0.70245 | **0.66293** |
| top-1 argmax agreement | **0.9143** | 0.9048 |
| mean top-20-union KL (renormalized approx.) | **0.27747** | 0.31911 |
| missing-actual entries | 0 | 0 |

**Result: AR is equal-or-best on all three primary metrics** (mean |Δ| ratio
AR/GPTQ = 0.991; median 24% closer; top-1 +0.9 pts; KL 13% lower). GPTQ's
only edge is the p95 tail (0.663 vs 0.702) — AR's worst-case deviations are
marginally larger while its typical-case deviations are clearly smaller.
Magnitude sanity: both artifacts sit at ~0.20 mean |Δ| with ~91% top-1
agreement to BF16 — consistent 4-bit MoE quantization noise, neither
degenerate.

Evidence: `results/2026/08/ornith15-ar-parity-20260822/` (reference-bf16.json,
logprobs-{ar,gptq}.json, parity-summary.json, server logs, scripts,
reference-run.log incl. the two cgroup-OOM iterations that motivated the
memmap reader).

Limits: 12 short prompts / 315 positions; teacher-forced logprobs, not a
task suite; single run each (deterministic seeds; no n>1 spread measured);
KL truncated to top-20 union.

## 5. Analysis and conclusions

**Baseline precision (three-way, not two-way).** The campaign brief framed the
control as "GPTQ with 24.75% RTN fallback", but that figure belongs to the
*original* (Aug-19) GPTQ conversion. The current reference artifact is
**MixedCal-v2 GPTQ, RTN fallback 10.37%** (dossier n=5: 70.74 no-spec /
96.43 MTP1 p512/g128 @ 150 W). The correct converter ladder is:

| Conversion | RTN fallback | Speed class (p512/g128) |
|---|---:|---|
| Original GPTQ (Aug-19) | 24.76% | 69.0 t/s n=3 (this campaign's control cells) |
| MixedCal-v2 GPTQ (current reference) | 10.37% | 70.74 t/s n=5 @ 150 W |
| **AutoRound ar-r1 (this campaign)** | **0%** (30,720/30,720 experts signSGD-tuned) | 70.1 t/s n=3, MTP1 92.4–96.3 |

**Conclusion 1 — conversion route.** AutoRound with the same corpus and scope
completes the calibration-debt ladder to zero fallback: every expert
projection received gradient-guided rounding (deep-block losses cut 70–83%),
versus 1-in-10 expert projections (MixedCal-v2) or 1-in-4 (original) getting
untuned RTN weights. Conversion cost is real but acceptable: 3 h 26 m vs
GPTQ's ~6.4 h on this host, and it required host-specific fixes (§3) that are
now encoded in the retained scripts.

**Conclusion 2 — serving is a drop-in.** Same `--quantization gptq` load path,
same champion pattern, same KV footprint (169,622 tokens @ 32K), speed inside
n=3 spread of both GPTQ generations, MTP1 works with the same nightly/boundary
patches (acceptance 78.5% vs 80.3% pos0 — slightly lower, decode t/s
unaffected). No engine-side changes needed anywhere.

**Conclusion 3 — reference parity is now MEASURED (2026-08-22): AR equal-or-best.**
The §4b gate ran after the initial write-up: on 315 teacher-forced positions
AR beat GPTQ on mean/median |Δlogprob|, top-1 agreement, and top-20-union KL
against the BF16 source (GPTQ retains only a slight p95-tail edge). The
zero-fallback structural argument now has matching distributional evidence.
Remaining limit: this is logprob parity on 12 prompts, not a task-quality
suite — "equal-or-better calibrated rounding" is confirmed, "better end-task
accuracy" is still unclaimed.

**Recommendation.** AutoRound artifact confirmed **equal-or-best** vs the
GPTQ control (ADR-0007 default-converter decision stands on measured
evidence). The remaining step before swapping the live serving reference is
the n=5 confirmation set at the declared cap plus upload/publication
authorization; MixedCal-v2 GPTQ stays the published/LocalMaxxing reference
until then.

## 6. Reproduction

```bash
# conversion (after fixes):
ORNITH_AR_RUN_ID=ar-r1 /mnt/<models-disk>/ornith15-quant/b70-ornith-autoround-run.sh
# (requires: workspace loop mounted, NTFS ar-work-space mounted, GPU empty)

# head-to-head serve:
nohup /mnt/<models-disk>/ornith15-quant/ar-headtohead-serve.sh &
# control probe re-serve:
nohup /mnt/<models-disk>/ornith15-quant/ar-headtohead-control.sh &
```

## 7. Open gates

- n=5 confirmation set on the AR artifact (the remaining pre-swap gate; reference parity passed 2026-08-22)
- task-quality suite (logprob parity ≠ end-task accuracy; 12 prompts / 315 positions only)
- Cn concurrency, 64K/128K ladder — unchanged from control campaign
- fp8-KV + TRITON_ATTN and bf16-activation A/B cells (see Intel-skills analysis doc, `research/2026/intel-vllm-xpu-skills-vs-b70-observations-20260822.md`)
