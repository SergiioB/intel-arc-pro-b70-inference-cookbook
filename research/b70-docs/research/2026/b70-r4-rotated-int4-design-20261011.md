---
title: "R4: residual-rotated INT4 on the stock B70 W4A16 path - move the quality fix offline, keep the fast kernel (2026-10-11)"
type: research
date: 2026-10-11
status: provisional
---

# R4 — residual-rotated INT4 for the dense Qwen3.8-27B

**PROVISIONAL - NOT FOR PUBLIC HEADLINE.** Design and kill ladder only. No conversion,
serving or GPU run backs this note yet. Program question: RQ-27.

## 1. Why this format, from what this host measured

| Observation | Evidence | Consequence |
|---|---|---|
| The plain INT4 W4A16 path is fast on the B70. The removed GPTQ route ran 102.61 tok/s with MTP4 at 230 W against EXL3's 63.25 tok/s (both p512/g128). The kernel streams ~430-495 GB/s with auto_gptq packing. | RL-087, ADR-0010, `registry/model-status.json` | 1.62× decode already exists in a kernel nobody has to write |
| That route died only on quality: every symmetric INT4 G128 export flips `sum(i*i for i in range(4))` from 14 to 30. The cause is a ~-3.3 logit margin inversion in the weights, not the kernel or the accumulation dtype. | RL-092 | The fix must change the weights, not the kernel |
| An offline residual Hadamard rotation cut W4 G128 block KL to 0.40× (GDN) and 0.58× (attention) at identical bytes, and lowered reconstruction error on every tensor. | RL-067 | Rotation is a quality lever at zero byte cost |
| HQ4 tried to buy the same quality with per-linear online Hadamards inside a new kernel. Its kernel reached 24 GB/s, undiagnosed, and its block metric was blind on 4 of 6 blocks. | RL-145, RL-150, RL-157 | A new kernel and a block metric are the two things to avoid |

R4 keeps the fast kernel untouched. All of the quality work moves offline, into the
weights, and is judged by an end-to-end metric that includes a known-bad control.

## 2. Format

1. **Residual rotation, folded offline.** Q = H_5120 = D·(K_20 ⊗ S_256)/√5120 (seed 20260925), as in `scripts/tmp/rotation_t1t2_common.py`.
   - Readers of the residual stream get W' = W·diag(1+g)·Q, with the preceding RMSNorm gain folded and the norm weight set to 0. This model's norm computes (1+w)·x̂.
   - Readers are q/k/v, gate/up, in_proj_qkv/z, and the BF16 in_proj_a/b.
   - Writers get W' = Qᵀ·W. Writers are o_proj, out_proj and down_proj.
   - This is exact in real arithmetic because RMSNorm without gain commutes with an orthogonal Q.
2. **Rotation sandwich at the boundaries.**
   - Entry: x' = Qᵀ·embed(x).
   - Exit: h = Q·h' before `model.norm`.
   - Because of this, embeddings, final norm, lm_head and the MTP head (which reads the post-norm hidden state and the shared embedding) all see the original basis, so MTP stays exact.
   - Cost: two BF16 5120×5120 matmuls per step, about 105 MB per round, roughly 0.6 % of round bytes. It needs a small capture-safe vLLM model patch.
3. **Rounding.** GPTQ W4 G128, symmetric, desc_act off, computed in the rotated basis, with auto_gptq packing. These are the champion contract settings, so `XPUwNa16` serves them unchanged. lm_head, embeddings, norms, conv, in_proj_a/b and mtp.* stay BF16. Variant: G64.

Bytes are the same as the removed GPTQ route (19.56 GB checkpoint), plus 105 MB.

## 3. Kill ladder

| Stage | Work | Hardware | Pre-declared gate |
|---|---|---|---|
| R0 screen | One sequential layer-streamed pass. GPU1 runs torch-xpu inside `vllm/vllm-openai-xpu:v0.30.0`, no serving. It keeps parallel hidden-state streams: BF16, EXL3 4bpw (bit-exact `ref.py`), GPTQ-plain, R4-G128 and R4-G64. Calibration: 64×512 wikitext-2 test tokens [0,60 %). Eval: 2×832 [62 %,…). Canaries: r01/h03/n01/n02/n03 from the margin-requant heldout manifest. | GPU1 | **Sanity first:** (s1) R4 with quantization disabled reproduces BF16 final logits (top-1 ≥ 99.9 %, KL ≤ 1e-4); (s2) GPTQ-plain reproduces the defect (r01 margin wrong sign); (s3) EXL3 answers r01 correctly. If any sanity check fails, the harness is invalid and nothing is scored. **Gate:** the best R4 arm has every canary margin with the correct sign, final-logit KL on eval ≤ EXL3's, and top-1 ≥ EXL3 − 0.1 pt |
| R1 checkpoint | Export the passing arm (auto_gptq, `dynamic` keeps mtp.* BF16). Add the sandwich patch. Boot interactively first, then run a sealed exact-answer battery incl. the 14-vs-30 case, plus replay | GPU1 | Battery passes with the same set EXL3 passes |
| R2 route | Paired cells against EXL3 4bpw (exl3xpu) at p512/g128 and p8192/g128. Graph capture on, MTP on, the same cap | GPU1 | ≥ 1.30× decode in every confirmation pair; MTP acceptance within 0.05 of EXL3 |

Stop rule:
- R0 below the gate after one recipe retry (G64 or a calibration change) closes R4 for this model. Record the measured KL gap and the margins.
- R0 with a failed sanity check is `blocked`, never `failed`.

## 4. What this is not

- It is not HQ4: no new kernel, no per-linear online transform, no block metric.
- It is not QuaRot W4A4: activations stay FP16; only the weight-only residual fold is used.
- It is not requantization from a quantized artifact (RL-113..115). It starts from BF16.
- Nearest literature: QuaRot (arXiv 2404.00456), SpinQuant (arXiv 2405.16406). The
  host-specific part is placing all of the quality work offline, so that the one kernel
  this card already runs fast serves it.

## 5. Risks

- The rotation may fix block error and still miss the 14-vs-30 margin, which is fragile (BF16 margin only +0.40). R0 measures exactly that.
- The v0.30.0 image is not the champion image (f01e24f6, gone). The 102.61 tok/s anchor will not transfer automatically; R2 re-measures it.
- Host RAM: R0 must not run beside a Strata server (RL-160). It runs in a container with a memory cap.
