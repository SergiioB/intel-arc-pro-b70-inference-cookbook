---
title: "Exp17B: Gemma-4-26B-A4B-it MoE quantization KL, BiasCorr gate, HF-INT4-G16 fit"
status: complete
type: research-result
date: 2026-11-25
---

# Exp17B (Wave 8B), GPU0

## 1. Result

All four 4-bit formats pass the 3-check KL gate with `BC_all_experts` on `google/gemma-4-26B-A4B-it`
(30 layers, causal SDPA, 2,044 WikiText-2 and 2,044 Code targets, **chat-wrapped windows, primary**).
Combined KL vs BF16: `HF-INT4-G16` 0.158 to 0.113, `NVFP4` 0.169 to 0.115, `MLX G64` 0.262 to 0.169,
`SymINT4 G128` 0.449 to 0.290. The gain comes from the dense layers: per-expert correction is not separable
from attention + shared-MLP-only correction at this calibration size. The projected HF-INT4-G16 text weights are
13.6 GiB. Main limits: this is a harness (no engine run), and the task's raw-text windows are not usable on this
`-it` checkpoint (BF16 PPL 40,294).

## 2. What you need

| Item | Value |
|---|---|
| GPU | 1x Arc Pro B70 (host GPU0, redacted-pci-gpu0), 230 W, 2,800 MHz |
| Peak use | VRAM 8.6 GiB reserved (full run), 11.6 GiB (identity gate); host RSS 6.4 GiB / 11.9 GiB |
| Disk | 49 GiB stage (51.6 GB weights) on `data-root/samsung980`, deleted afterwards |
| Image | `vllm/vllm-openai-xpu:v0.31.0@sha256:95ac815038c4b3537b173798bfe3b89bdf525a52bd653cd14d4deb01f1062be7` |
| Software | torch 2.14.0+xpu, transformers 5.17.0, kernel 7.0.0-29-generic |
| Model | `google/gemma-4-26B-A4B-it` @ `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` (not gated) |

## 3. Timeline (local CET, 2026-11-25)

| Time | Step | Outcome |
|---|---|---|
| 12:17-12:42 | Download, container capped at 8 GiB | FAILED: OOM-killed at 49.24 of 49.91 GB in shard 1 |
| 12:43-12:44 | Resume same partial file by HTTP range | ok, both SHA-256 match |
| 12:44-12:48 | Identity gate v1 | FAILED 4 checks |
| 12:48-12:57 | Kernel diagnostic | **FAILED badly: rebooted the host** (OOM from 12:53:45, back 12:57:37) |
| 12:59-13:01 | VRAM cap verified, guarded diagnostic | kernels match references |
| 13:00 | Second agent assigned the same job | it stood down, no edits by it |
| 13:02 | Layer diagnostic | killed by my first watchdog (false positive) |
| 13:03-13:12 | Per-layer checks, synchronize A/B | layers correct; sync theory falsified |
| 13:13-13:23 | CPU FP32 reference, knowledge probes, chat vs raw | model and harness fine; raw text is the outlier |
| 13:24-13:28 | Determinism search | cause found and fixed |
| 13:29-13:37 | **Identity gate v2** | **11/11, exact zeros** |
| 13:38-13:44 | 6-layer smoke | 1 crash and 1 useless control, fixed |
| 13:44-13:55 | **Full run, 30 layers, 2 formats** | **671 s, receipt valid** |
| 13:56-14:00 | Validate, prune, clean up | done |

Total about 1 h 45 min. About 49 min (12:48-13:37) went to the failed gate, the crash and the diagnosis.

## 4. Steps

```bash
cd study-root/results/exp17b_scripts
# 0. Stage must hold the model at the revision above, plus train/test parquet of Salesforce/wikitext
#    wikitext-2-raw-v1 (@b08601e04326c79dfdd32d625aee71d232d685c3) in the stage root.
./run_exp17b_gpu0.sh identity   # about 7.5 min; must print "passed": true
./run_exp17b_gpu0.sh main       # about 11 min; refuses to start if the identity JSON did not pass
docker run --rm --entrypoint python3 -v $PWD/..:/results:ro -v $PWD/../../publish-bundle:/gate:ro \
  vllm/vllm-openai-xpu:v0.31.0 /results/exp17b_scripts/validate_exp17b.py /results/exp17b_gemma4_26b_a4b.json
```

## 5. Every fix

| # | Symptom (literal) | Cause | Fix | Verified by |
|---:|---|---|---|---|
| 1 | download container `exit 137`, `OOMKilled true` | 8 GiB cap too small for the Xet download | `curl -C -` on the same partial file | SHA-256 equals the LFS object id |
| 2 | `[TTM] Buffer eviction failed`, global OOM, reboot | my diagnostic ran HF `batched_mm`: about 31 GiB at 512 tokens, spilled to host shmem, not bounded by `docker --memory` (inferred from size match and cgroup allocation failures) | removed `batched_mm`; `torch.xpu.memory.set_per_process_memory_fraction` 18 GiB; `watchdog.py`; `--oom-score-adj=1000` | cap probe raises `XPU out of memory`; full run shmem max 0.33 GiB |
| 3 | `[watchdog] KILL exp17b-layers-gpu0 shmem_kB=11625732` | v1 watched host-wide shmem (another GPU job was running) | per-container DRM `fdinfo` accounting | later runs: own gtt+system at most 0.16 GiB, no kill |
| 4 | `size of tensor a (16) must match ... b (8)` | my SDPA checker ignored grouped-query attention | expand KV heads in the reference | 30/30 layers, SDPA rel L2 about 1.7e-3 |
| 5 | `seq1536 max_abs_diff 22.5`, causality `15.3`, passes disagree | XPU float matmuls nondeterministic (13-16 distinct results in 16 calls); raw-text input amplifies it | `torch.use_deterministic_algorithms(True)` | 16/16 identical; identity gate exact 0.0 |
| 6 | BF16 WT2 PPL `40812` | not a harness bug: `-it` checkpoint on non-chat text (inferred) | chat-wrapped primary format, raw kept as secondary, preregistered | chat probes `Paris`/`banana`; chat PPL 24.6 / 11.9 |
| 7 | `AmbiguousGlobalPerLayerAttributeError` | census read `cfg.head_dim`, which is per layer | read `per_layer_config` | smoke and full run complete |
| 8 | rounding control gave bit-identical output | BF16 expert loop equals `grouped_mm` in deterministic mode | control uses FP32 expert matmul | non-zero control KL (table below) |

Also tested and rejected: "free without `synchronize()` corrupts weights". Both modes gave the same bad PPL.

## 6. Verification

| Check | Count | Raw file (under `results/`) |
|---|---|---|
| Identity gate | 11/11 | `exp17b_identity.json` |
| Tensor reads vs `safetensors` | 657/657 | `exp17b_scripts/exp17b_probe.result.log` |
| Zero-bias repeat arm | 0.0 max diff, 30 layers, KL 0 | receipt `meta` |
| Means, PPL, CIs re-derived from arrays | 120/120 | `exp17b_scripts/exp17b_validation.json` |
| Gate verdicts re-derived | 18/18 | same |
| Calibration/eval 6-gram overlap | 0 | receipt `meta.corpus` |
| Run script hashes equal the pre-run freeze | 3/3 | `exp17b_scripts/exp17b_frozen_before_main.sha256` |

Crash evidence: `exp17b_scripts/exp17b_host_oom_excerpt.log`.

## 7. Results and claim boundary

Chat format (primary). KL vs BF16 in nats per token, mean over 2,044 targets, one deterministic run.

| Arm | WT2 DF to BC | Code DF to BC | Code/WT2 `-b` KL | Verdict |
|---|---|---|---|---|
| HF-INT4-G16 W4A16 | 0.139 to 0.103 (-26.1%) | 0.176 to 0.122 (-30.6%) | 0.365 / 0.333 | PUBLISHABLE |
| NVFP4 4/6 | 0.158 to 0.102 (-35.5%) | 0.180 to 0.129 (-28.2%) | 0.339 / 0.342 | PUBLISHABLE |
| MLX asym G64 | 0.223 to 0.160 (-28.4%) | 0.301 to 0.179 (-40.6%) | 0.693 / 0.511 | PUBLISHABLE |
| SymINT4 G128 | 0.424 to 0.265 (-37.5%) | 0.475 to 0.315 (-33.7%) | 1.421 / 1.548 | PUBLISHABLE |

Rounding floor (same BF16 math, FP32 expert matmul): 0.0057 WT2, 0.0110 Code. Every chat primary gain exceeds it.

| Question (chat, combined KL) | Answer |
|---|---|
| HF-G16 vs NVFP4, data-free | not separable: -0.011, CI -0.025 to +0.003 |
| Per-expert BC vs attention + shared MLP only | not separable: HF +0.0010 (CI -0.005 to +0.007), MLX -0.0014 (CI -0.008 to +0.005) |
| W4A8 vs W4A16 | +0.0028, CI -0.002 to +0.007 |
| K-only vs V-only | 0.034 vs 0.050; sliding layers only: 0.019 vs 0.050. V-sensitive |
| Experts + `mlp.down_proj` left BF16 | 0.121 vs 0.158 data-free, but 45.0 GiB: does not fit |

Raw format (secondary): all four primary gates also report PUBLISHABLE, but the floor is 0.40 / 0.64.
`MLX` and `NVFP4` Code gains are smaller than the floor, so those two verdicts are **not interpretable**.
`HF-INT4-G16` (code ratio 1.2) and `SymINT4` clear it, HF only narrowly.

Structure: 30 layers (25 sliding, 5 full), hidden 2,816, 128 experts, top-8, expert width 704, shared MLP 2,112.
`mlp.down_proj` K=2,112 and `experts.down_proj` K=704 are 64 mod 128. BF16 48.07 GiB. HF-INT4-G16 projection:
13.63 GiB text, 14.70 GiB with vision in BF16. vLLM 0.31.0 registers both Gemma4 classes, but the published
plugin rejects packed experts, requires K%128==0 and skips `router,experts`.

Sidecars (chat only, 64 MiB each, above the requested 15-30 MiB): `exp17b_sidecars/`. The raw-format sidecars
were removed to save disk; their SHA-256 stays in the receipt.

Not tested:

- vLLM or any engine load, serving speed, KV memory use in an engine.
- A real quantized checkpoint (all arms dequantize in memory, except the live W4A8 int8 path).
- More than 2,048 calibration tokens (212 of 3,840 expert slots got none; median 88 per slot).
- Sliding-window truncation in the scored windows (they are shorter than 1,024; checked only in the identity gate).
- A third-party engine on raw text, other chat prompts, generation or task accuracy.

Allowed claim: in this harness, on these windows, open-loop BiasCorr lowers causal KL for all four formats and the
negated control raises it. The intervals cover 32-token blocks, not document, prompt or calibration choice.

## 8. If you do this again (about 45 min after download)

1. Give the download at least 16 GiB of container RAM (guess; 8 GiB was not enough).
2. Set both guards first: the VRAM fraction cap and `torch.use_deterministic_algorithms(True)`.
3. Never run `batched_mm` experts on a model this wide.
4. For an `-it` checkpoint, probe chat vs raw PPL before choosing the eval layout.
5. Run `identity`, then `main`, then the validator.

## 9. Adapting it (for example Qwen3.6-35B-A3B, 71.9 GB)

- Carries over: pread reader, watchdog, VRAM cap, determinism, rounding-floor arm, gate and validator.
- Re-check: model math and layer keys, expert kernel default, K % 128, per-layer config attributes, prompt format.
- Disk: 72 GiB was free on `data-root/samsung980` at cleanup while `exp17a_stage` still existed. 71.9 GB is about 67 GiB.
- Earlier XPU studies without deterministic mode can differ run to run at rounding level. Whether that changed any
  earlier verdict is **not checked**.

Not yet done: ledger entry and `research/SESSIONS.md` registration (standard rule 7), outside this task's write scope.
