# Gate 4B: Exp 19A (KV quantization) and Exp 19B (Ternary-Bonsai-2-27B), 2026-11-25

All KL values are next-token KL against Qwen3.8-27B BF16 (nats). "all3" pools WikiText-2, code and
ood_chat (3 x 2,044 tokens). "bpw" means stored bits per weight.

## 1. Result

- **19A.** The TurboQuant-style 1-bit QJL sketch loses to the same rotated quantizer using those bits
  directly: all3 KL 0.0529 vs 0.0169 for K=3 / V=4, and 0.0640 vs 0.0242 for K=3 / V=3.
- **19A.** "V gets more bits than K" did not beat the alternatives for this quantizer.
- **19A.** KV quantization composes with HF-INT4-G16 + BiasCorr weights without compounding. The
  stacked KL was 0.93 to 0.98 of the sum of the separate KLs.
- **19B.** Bonsai as shipped scores 0.653. That is below Exp18 ternary G64 + BC (2.689) and INT2-G16 + BC
  (1.138), and above INT3-G16 + BC (0.271).
- **19B.** Both BiasCorr variants (plus the open residual form) fail the 4-check gate on Bonsai.
  Ship no sidecar.
- **19B.** PrismML's training method is not stated. "QAT" is not verified.

## 2. What you need

- 1 x Arc Pro B70, host GPU1 (`redacted-pci-gpu1`, `card1`/`renderD129`), 230 W cap, `max_freq` 2800.
- `vllm/vllm-openai-xpu:v0.31.0` (torch 2.14.0+xpu, transformers 5.17.0).
- `data-root/sata1/models/Qwen3.8-27B-bf16`, `models/Qwen3.8-27B-HF-INT4-G16-BC`, the v1 sidecar
  (`afafdc60...`), and the Exp18 corpus and receipt.
- 19B download: `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` revision `24b5937a...`, `model.safetensors`
  8,595,477,990 bytes, sha256 `130de592...`. `hadamard.json` is also required to dequantize.

## 3. Timeline (local clock, 2026-11-25)

| time | step | outcome |
|---|---|---|
| ~21:47 | read Exp18 / HF-G16 code, check GPU1 and disks | GPU1 idle, 32,482 MiB free |
| 21:50-21:53 | Bonsai download (22 GB free before) | hash matches manifest and LFS id |
| 21:55-22:00 | 19A attempt 1 | **failed**: XPU out of memory in the KV precheck (~5 min lost) |
| 22:01-22:05 | cycle fix and 16-layer smoke | memory flat |
| 22:05-22:19 | 19A full | pass |
| ~22:20-22:24 | 19B smokes | **norm convention bug caught before any scoring** |
| 22:24-22:34 | 19B run 1 | completed, **host RSS 14.57 GiB, over the 14 GiB limit**; rejected by validator (~10 min lost) |
| 22:36-22:46 | 19B run 2 | pass, RSS 8.00 GiB, all 441,504 per-token values equal to run 1 |
| 22:47 | publish receipts, delete download | `data-root/samsung980/work/exp19b_bonsai2` removed |
| 22:48-22:55 | post-hoc PTQ sparsity control | pass |

Total about 70 min. About 15 min was lost to the two failed or rejected runs.

## 4. Commands

```bash
cd study-root/exp19-kv-bonsai
./launch_gpu1.sh exp19a_kv.py exp19a_full.log
./launch_gpu1.sh exp19b_bonsai.py exp19b_full.log          # needs the download mounted at /bonsai
./launch_gpu1.sh exp19b_posthoc_sparsity.py exp19b_posthoc.log
# validator: CPU only, same image: validate_exp19.py A.json B.json EXP18.json OUTDIR
```

## 5. Fixes

| # | symptom | cause | fix | verified by |
|---|---|---|---|---|
| 1 | `XPU out of memory`, 16.46 GiB allocated, 23,084 MiB in kernel | `m.forward = saved_bound_method` makes a reference cycle, so streamed layers stay alive | `restore_forwards()` removes the instance attribute | CPU weakref reproduction; allocated flat at 2.78 GiB |
| 2 | norm weights off by ~1.0; output-mean matching gap 0.22 on Bonsai vs 0.002 on PTQ | HF `Qwen3_5RMSNorm` computes `x*(1+w)`; the MLX pack stores `1+w` | load `w-1` (FP32) for that module class only | gap 0.0019, equal to the PTQ rounding floor |
| 3 | host peak RSS 14.57 GiB | Exp18 precheck keeps two shards mapped, plus the embedding mapping | `LayerBoundedReader`, release embedding first, in-script peak check | peak 8.00 GiB; identical results |

The Exp18 and 19A runs stayed under the limit (13.23 and 13.35 GiB).

## 6. Verification

- Exact 0.0 repeat, future-prefix and other-row checks: stock BF16 (64 layers); 34 KV paths in 19A; Bonsai
  weights; control weights. Each had a non-zero suffix control.
- Hooked unquantized KV equals stock SDPA with max abs 0.0. `BF16_KV` KL is 0.0.
- BF16 PPL 7.79822 / 2.37436 / 3.67308. Per-token NLL equals Exp18. Calibration is the same 4,088 positions
  with 0 shared 6-grams.
- v1 gate anchor: six published HF-G16 KL values reproduced with difference 0.0.
- QJL self-test: inner-product slope over 400 sketches 0.9998; relative MSE 1.566 (theory 1.567).
- Bonsai dequantization: GPU unpack equals an independent numpy unpack; folded dense weight equals the
  rotate-activation path to 7.8e-7 relative (FP32); all 402 modules have bias == -scale and no code 3
  (`embed_tokens` on used rows only); median row cosine to BF16 0.884 with the rotation, 0.0002 without.
- `validate_exp19.py` re-derives means from per-position arrays and recomputes 6 gates.
- Receipts in `../results/`: `exp19a_turboquant_kv_qwen38.json` (`de6ca3f1...`),
  `exp19b_ternary_bonsai2_27b.json` (`ab653594...`),
  `exp19b_posthoc_ptq_sparsity_control.json` (`e61cb236...`).

## 7. Results and claim boundary

Full tables, paired blk32 95% intervals, per-token arrays and the 256K footprint are in the receipts.

Not tested: task accuracy, generation, speed, any packed KV or ternary kernel, real VRAM at 256K context,
contexts longer than 512 tokens, multimodal input.

- 19A "PolarQuant" is Hadamard + signs + Gaussian Lloyd-Max, as specified. It is not the polar-coordinate
  transform in the PolarQuant paper. The TurboQuant result applies to this implementation.
- KL to the BF16 teacher penalizes any divergence from Qwen3.8-27B, including useful divergence in a
  re-optimized model. PrismML's task-benchmark claims were not tested.
- The footprint is arithmetic from shapes and headers. It excludes work buffers.
- The sparsity control is post-hoc.
- The v1 sidecar uses its shipped 1,022-token calibration, not the 4,088-token stream.

## 8. If you do this again

About 40 min: 4 min download, 14 min 19A, 10 min 19B, 7 min control. Run the output-mean invariant before
trusting any converted checkpoint. It found the norm convention.

## 9. Adapting it

- Another MLX or GGUF pack: check norm conventions per module class, and compare every unquantized tensor
  with the reference before scoring.
- Another KV codec: add one branch in `KVQuant.__call__`, then keep the per-path precheck.
- `bonsai_rotation_signs.json` is the one artifact kept from the download. It is derived from `hadamard.json`.
