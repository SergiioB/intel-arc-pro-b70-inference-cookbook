---
title: "Exp17C: Qwen3.6-35B-A3B hybrid GDN+MoE quantization KL, shipped GPTQ audit, MoE BiasCorr gate"
status: complete
type: research-result
date: 2026-11-25
---

# Exp17C (Wave 9A, Gate 3), GPU1

## 1. Result

On `Qwen/Qwen3.6-35B-A3B@345d467b` (40 layers, causal SDPA, 2,044 scored tokens each of WikiText-2 test,
`difflib.py` and OOD chat), data-free pooled KL vs BF16 (nats per token) is: HF-INT4-G16 W4A16 0.126, NVFP4 0.130,
MLX G64 0.174, shipped GPTQ 0.177, SymINT4 G128 RTN 0.249. With the 2.54 MiB shared-only BiasCorr, HF-INT4-G16
reaches 0.092 and passes both the 3-check and the OOD gate. Correcting all 256 experts (122.5 MiB) is not
separable from shared-only in any format.

**The local GPTQ checkpoint does not quantize this base.** It quantizes a heretic-abliterated derivative.
Its residual against the official weights has one shared direction (|cos| 0.991 to 1.000 across expert
`down_proj`, layers 10 to 39). Its KL therefore mixes abliteration with quantization. All its BiasCorr gates fail,
and no GPTQ sidecar was saved.

## 2. What you need

| Item | Value |
|---|---|
| GPU | 1x Arc Pro B70, host GPU1 (redacted-pci-gpu1), 230 W, 2,800 MHz |
| Peak own VRAM (DRM fdinfo) | 14.4 GiB audit, 10.2 GiB main; cap `set_per_process_memory_fraction(0.55)` |
| Peak host RSS | 6.8 GiB (main); GPU spill to host at most 0.10 GiB; no watchdog kill |
| Disk | 67 GiB stage on `data-root/samsung980` (85 GiB free before, 18 GiB during, 85 GiB after) |
| Image | `vllm/vllm-openai-xpu:v0.31.0` (torch 2.14.0+xpu, transformers 5.17.0) |
| Plugin code used | `vllm-hf-int4-g16` v0.2.0 `format.py` and `converter.dry_run_layer_mapping` (read-only mount) |

## 3. Timeline (local CET, 2026-11-25)

| Time | Step | Outcome |
|---|---|---|
| 17:59-18:01 | Read 17B scripts, GPTQ card, HF file list | card says the GPTQ base is abliterated |
| 18:02-18:23 | Download, plain HTTP (`HF_HUB_DISABLE_XET=1`) | ok, 21 min; 26/26 shard SHA-256 match |
| 18:11 | Unit check on GPU1 (quantizers, packing, GPTQ unpack) | ok |
| 18:23-18:28 | Audit, attempt 1 | exact checks ok; **direction flag rule was wrong** (fix 1) |
| 18:28-18:32 | Audit rerun saving all top vectors, then analysis | found the signal |
| 18:33-18:37 | Audit, final | ok, 261 s |
| 18:37-18:44 | Identity gate | 11/11, 392 s |
| 18:44-18:47 | Preregistration, 4-layer smoke | ok |
| 18:47-19:05 | **Main run, 40 layers, 42 arms** | ok, 1,068 s |
| 19:05-19:15 | Validate, floor check, post-hoc pairs, cleanup | ok |

About 76 min total. About 9 min went to the audit rework. No crash, no OOM, no reboot.

## 4. Steps

```bash
cd study-root/results/exp17c_scripts
# 0. stage: run download.py in the image with HF_HOME=/stage/hf_cache HF_HUB_DISABLE_XET=1 (command in download.py's docstring is the pinned revision)
./run_exp17c_gpu1.sh audit      # about 4.5 min; census, GPTQ audit, direction estimate
./run_exp17c_gpu1.sh identity   # about 7 min; must print "passed": true
./run_exp17c_gpu1.sh smoke      # about 2.5 min; plumbing only
./run_exp17c_gpu1.sh main       # about 18 min; refuses to start if identity failed
docker run --rm --network none --entrypoint python3 -v $PWD/..:/results:ro -v $PWD/../../publish-bundle:/gate:ro \
  vllm/vllm-openai-xpu:v0.31.0 /results/exp17c_scripts/validate_exp17c.py /results/exp17c_qwen36_35b_a3b.json
```

The download container was started with `docker run -d --user $(id -u):$(id -g) --memory=12g
-e HF_HOME=/stage/hf_cache -e HF_HUB_DISABLE_XET=1 ... python3 /work/download.py`.

## 5. Every fix

| # | Symptom | Cause | Fix | Verified by |
|---:|---|---|---|---|
| 1 | Flag rule found 17 of 120 target matrices, mostly shared `down_proj`; the direction held 54% of flagged energy, cos 0.26 to 0.39 | my absolute cut (share > 0.10) sat above the expert `down_proj` signal (0.05 to 0.08 against an RTN floor of 0.001) | no threshold: λ1-weighted eigenvector over expert `down_proj` residuals; report all 120 cosines | 99.4% of weighted rank-1 energy; changed before any quantized forward pass. Attempt 1: `exp17c_audit.attempt1-flagrule.json` |
| 2 | `exp17c_validation.json` was not valid JSON | I teed container stderr warnings into it | stripped to the JSON object | `json.load` succeeds |

## 6. Verification

| Check | Result | Raw file (under `results/`) |
|---|---|---|
| pread loader vs `safetensors` | 712/712 equal | `exp17c_scripts/exp17c_identity.json` |
| HF `Qwen3_5MoeForCausalLM.forward` vs streaming, 40 layers | max diff 0.0; repeat 0.0 | same |
| Two streaming passes, all eval and cal inputs | 0.0 | same |
| Per-token causality, token-independent experts, cut at token 97 (mid GDN chunk) | prefix 0.0; suffix control 26.6 | same |
| Stock `grouped_mm` noise on the unchanged prefix | KL mean 0.0057, max 0.196, top-1 97.9% | same |
| W4A8 expert loop vs stock kernel, BF16 weights, layer 0 | 0.0 | receipt `meta` |
| Zero-bias repeat arm, 40 layers | 0.0, KL 0 | receipt `meta` |
| Calibration vs eval 6-gram overlap (WT2, code, OOD) | 0 / 0 / 0 | receipt `meta.corpus` |
| Re-derived from per-token arrays | 516 mean/PPL/CI, 30 verdicts, 129 pairs, 0 failures | `exp17c_scripts/exp17c_validation.json` |
| Script hashes equal the pre-run freeze | 5/5 | `exp17c_scripts/exp17c_frozen_before_main.sha256` |
| v0.2.0 packer plus reference unpacker vs harness int8 and d_row | equal on 3 matrices | receipt `v020_packing_roundtrip` |

`from_pretrained()` was never called. The loader check is the HF forward with layers loaded one at a time.

## 7. Results and claim boundary

BF16 PPL: WT2 7.72, Code 1.60, OOD chat 4.00. Code is below the 2 to 4 range the task expected. HF's own forward
matches exactly, and earlier Qwen runs on this file span 1.41 to 6.79.

`cal_token_sha256` = `05ed10460d34ada25fb68a7834a6a1c7d0ebce6e1eb5c7b40b25b378b906af2d`.

### Structure and fit

- 40 layers: 30 GDN, 10 full attention (every 4th). Hidden 2,048. 256 experts, top-8, expert width 512.
- `gate_up_proj` [256, 1024, 2048]; `down_proj` [256, 2048, 512].
- Every quantized K is a multiple of 128, so Had64 is never used.
- BF16 66.97 GiB: 62.62 quantizable, 1.94 other text, 0.83 vision, 1.57 MTP.
- HF-INT4-G16 projection: 18.82 GiB text (13,215 MiB headroom against 32,482 MiB idle free), 19.65 GiB with BF16
  vision. MTP is excluded.
- KV/state arithmetic: 20 KiB per token for full layers plus 61.4 MiB GDN state per sequence;
  5.06 GiB at 262,144 tokens. Not measured in an engine.
- v0.2.0 dry run: 310 dense candidates and 40 expert banks. It would also pack `in_proj_a/b`, which GPTQ and this
  study keep BF16. vLLM 0.31.0 registers both `Qwen3_5Moe` classes. No engine load was tried.

### Scoreboard: KL vs BF16 (WT2 / Code / OOD), top-8 and CIs are in the receipt

| Format | Data-free | BC shared (2.54 MiB) | BC all experts | `-b` (all) | 3-check, 4-check (shared; all) |
|---|---|---|---|---|---|
| Shipped GPTQ | .064/.250/.216 | .063/.239/.220 | .063/.235/.203 | .071/.270/.239 | fail,fail; fail,fail |
| SymINT4 G128 | .120/.385/.243 | .090/.289/.210 | .087/.283/.205 | .288/.826/.589 | pass,**fail**; pass,pass |
| MLX asym G64 | .075/.237/.210 | .049/.174/.156 | .048/.174/.167 | .157/.445/.357 | pass,pass; pass,pass |
| NVFP4 4/6 | .051/.185/.153 | .040/.170/.131 | .040/.167/.111 | .092/.313/.243 | **fail**,fail; pass*,pass* |
| HF-INT4-G16 W4A16 | .056/.166/.157 | .036/.129/.111 | .035/.126/.120 | .118/.303/.310 | pass,pass; pass,pass |
| HF-INT4-G16 W4A8 | .058/.171/.157 | .035/.132/.118 | .036/.127/.113 | .119/.302/.294 | pass,pass; pass,pass |
| Rounding floor | .0047/.0186/.0117 | | | | |

- `threshold >= 64` passes both gates for MLX and HF W4A16 (HF .036/.128/.116; MLX .051/.171/.175).
- \* NVFP4 all-experts: the Code gain is 0.95x the Code floor and its CI upper bound is -0.0003. Under
  preregistration item 4 the Code gain is **not interpretable**. Do not quote this pass without that note.
- Other publishable gains are 2.0x to 7.0x their domain floor (`exp17c_floor_check.json`).

### Paired answers (pooled over 3 domains unless stated; delta KL, 95% blk32 CI)

| Question | Answer |
|---|---|
| HF W4A16 vs NVFP4, data-free | not separable: -0.003 [-0.020, +0.014] |
| HF W4A16 vs MLX, data-free | HF lower: -0.047 [-0.066, -0.031] |
| All experts vs shared-only | not separable. HF +0.002 [-0.003, +0.006]; MLX +0.004 [-0.004, +0.011]; NVFP4 -0.007 [-0.016, +0.001] |
| MLX threshold-64 vs shared-only | slightly worse: +0.006 [+0.000, +0.012], below the floor |
| W4A8 vs W4A16 | not separable: +0.002 [-0.003, +0.007]; W4A8 dense + W4A16 experts: +0.003 [-0.004, +0.010] |
| K-only vs V-only, 10 full layers | V more sensitive: -0.009 [-0.012, -0.007]. V .0095/.028/.031; K .0054/.019/.017, K is about at the floor |
| HF data-free vs shipped GPTQ | -0.050 [-0.070, -0.032], **includes the abliteration** |
| HF data-free vs GPTQ with the direction restored (post hoc) | -0.032 [-0.045, -0.019]; WT2 alone not separable |

### GPTQ audit

- Coverage: all 30,970 modules equal the harness set: experts, shared expert, q/k/v/o, GDN `in_proj_qkv`,
  `in_proj_z`, `out_proj`. `lm_head`, `in_proj_a/b`, conv1d, norms, router, embeddings and vision are BF16.
- 363 text, 333 vision and 19 MTP unquantized tensors are bit-identical to the official base.
- Packing: G128, `g_idx` contiguous, every stored zero is 7. Wrong nibble order gives relative error 2.0
  against 0.017 for the right order.
- Expert `down_proj` residual top singular share: 0.001 to 0.002 at layers 0 to 9, 0.049 to 0.082 at layers 10 to 39
  (plain RTN at most 0.0075). GDN `out_proj` reaches |cos| >= 0.9 on 19 layers and `o_proj` on 4. The shared-expert
  `down_proj` reaches it on none.
- BiasCorr computed against the official base puts 9% to 31% of its energy along that direction on the target
  families (chance 0.049%; other formats 0.03% to 0.12%). So it partly reverses the abliteration. Behaviour
  (refusal rate) was not measured.
- The direction restore is a rank-1 heuristic. The card describes a norm-preserving ablation, which is not exactly
  rank-1, so some base mismatch may remain. It lowers pooled KL by 0.018 [0.004, 0.034].

### Sidecars (`results/exp17c_sidecars/`)

Shared-only: 250 tensors, 2,662,400 tensor bytes (2.54 MiB), 2,694,776 file bytes. All experts would be 122.5 MiB.
Saved for HF W4A16 and MLX (both gates pass) and SymINT4 (3-check only; its OOD CI crosses 0). NVFP4 shared failed.
They fit only this harness's quantized weights. They are not the v0.2.0 converter's `bias_corr` tensors.

Calibration routing: 10,240 expert slots, median 78 positions per slot, 180 with none, 4,514 below 64.
Slots with 64 or more hold 91.3% of routed positions.

Not tested:

- Any engine load, generation, task accuracy, refusal behaviour, speed or engine KV memory.
- A real packed HF-INT4-G16, MLX, NVFP4 or SymINT4 checkpoint. Those arms dequantize in memory.
- The shipped GPTQ against its own heretic BF16 base (not on disk; no space for a second 67 GiB).
- vLLM's GPTQ kernel arithmetic. GPTQ weights run here as BF16 dequantized weights.
- More than one corpus draw. Intervals cover 32-token blocks only. Windows are raw token slices.

Allowed claim: in this harness and on these windows, open-loop BiasCorr from dense layers alone lowers causal KL
for HF-INT4-G16 and MLX on all three domains, and negating it raises KL.

## 8. If you do this again (about 55 min including download)

1. Read the model card of any "shipped" quant first, and run the weight-space audit before comparing KL.
2. Stream the base by layer. Keep the 0.55 VRAM cap, deterministic mode and `grouped_mm`.
3. Run `audit`, `identity`, `smoke`, `main`, then the validator.

## 9. Adapting it

- Carries over: pread reader, GPTQ batched unpack, rank-1 residual fingerprint, arm table with exact `-b`,
  watchdog, floor arm, validator.
- Change: dense module lists per layer type, K widths, chat template, and abliteration target names.
- To isolate GPTQ quantization error, stage the heretic BF16 base and rerun the GPTQ arms against it.

Not done: ledger entry and `research/SESSIONS.md` registration (outside this task's write scope).
