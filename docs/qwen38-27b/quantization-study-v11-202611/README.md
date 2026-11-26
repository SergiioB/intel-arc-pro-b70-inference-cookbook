# HF-INT4-G16, BiasCorr, and B70 serving: public technical study V11
Status: Published V11 (Cookbook v1.24.0, 2026-11-26; Claude Opus 5.5 Gate 7D sign-off; supersedes V10 `../quantization-study-202611/README.md`, which remains immutable).

Audience: inference engine maintainers, quantization researchers, and operators who want exact evidence rather than a model-card slogan.

Scope rule: quantitative claims in this report are tied to a JSON receipt, a test log, a review file, a blueprint, or a process note. Every cited file path is relative to this bundle and ships in `results/qwen38-27b/quantization-study-v11-202611/`. Lower KL is better in every table. KL values are mean nats per scored token against the BF16 reference on the stated causal corpus.

## Figures
1. [chart1: 12-family data-free sweep](charts/chart1_12family_hfg16_sweep.png)
2. [chart2: where BiasCorr helps and where it hurts](charts/chart2_biascorr_gate_landscape.png)
3. [chart3: ultra-low-bit PPL curve and 2-bit task rescue](charts/chart3_ultralowbit_2bit_rescue.png)
4. [chart4: rotated KV versus TurboQuant QJL, with 256K footprint arithmetic](charts/chart4_kv_rotated_vs_turboquant.png)
5. [chart5: B70 dense serving, Gemma MoE serving, and quiet-host MoE kernel timing](charts/chart5_live_b70_serving_dense_and_moe.png)
6. [chart6: V versus K weight-projection sensitivity](charts/chart6_vk_sensitivity_12families.png)
7. [kinetic video 1: BiasCorr mechanism](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/releases/download/v1.24.0/anim1_biascorr_how_it_works.mp4)
8. [kinetic video 2: HF-INT4-G16 Xe2 architecture](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/releases/download/v1.24.0/anim2_hfg16_xe2_architecture.mp4)

## 1. Executive summary and honest scoreboard
This study began as a narrow serving question: can a 27B-class model fit and run well on one 32 GB Intel Arc Pro B70 without paying the quality cost normally associated with 4-bit post-training quantization? It ended as a broader measurement program covering weight formats, activation correction, KV compression, trellis coding, dense serving, multimodal MoE serving, and the failure modes of correction methods.

The durable results are:

1. `Had128NoPerm + HF-INT4-G16` is a practical Xe2-first 4-bit weight format. In data-free causal KL comparisons it is below MLX-4bit (asymmetric G64) in 12/12 tested families, and below SymINT4-G128 RTN in the 11/11 families that have that arm. Qwen3.8-27B has no RTN arm; there HF-G16 is also below the shipped, calibrated GPTQ-G128 checkpoint. The exact corpus is not identical across every receipt, so this is a family-level evidence aggregate, not one monolithic 12-model batch file.
2. The result is not only a KL result. Live vLLM-XPU serving runs the format in dense and MoE models. Dense Qwen3.8-27B has faster long-prompt prefill than shipped GPTQ in the measured test setup, while single-stream decode is 10 to 12 percent slower because the current kernel does more work in the hot path. Gemma-4-26B-A4B runs as a multimodal MoE on one B70 at 83.2 tok/s single-stream decode, measured on a non-quiet host (GPU1 busy), with 109,923 tokens of KV capacity. The served checkpoint used raw-text BiasCorr, which failed the 4-check out-of-domain (OOD) chat gate and is held back. No server has yet booted with the chat-calibrated BiasCorr that passed.
3. BiasCorr is a small first-order correction, not a universal repair. At ordinary 4-bit it is architecture-dependent and must pass an automatic out-of-domain gate. At 2-bit its effect becomes statistically unmistakable on the controlled 160-item battery: 31/160 without correction to 93/160 with one 14.89 MiB FP32 sidecar. Exact paired McNemar p is 1.0224592793857492e-12. The rescue also survives the untruncated GSM8K control: on the 17 items that finished in both arms, uncorrected output is 6/17 correct versus 15/17 with correction, exact p 0.00390625.
4. The same correction does not improve 3-bit total score in that battery. Both 3-bit arms score 122/160, while the mix changes. Exp20 then proved that the earlier GSM8K 41 to 48 improvement at a 384-token cap was a truncation artifact: with a 1,024-token cap, uncorrected 3-bit scores 58/60 with zero cut-off outputs versus 57/60 with correction, zero fixed, one broken, exact p 1.0. This null result is as important as the 2-bit rescue because it defines the boundary of the method.
5. PrismML `Ternary-Bonsai-2-27B` is much better than post-training ternary conversion, with combined three-domain KL 0.653058 versus 2.689266 for Exp18 ternary G64 plus BiasCorr and 1.137767 for Exp18 INT2-G16 plus BiasCorr. It is still worse than Exp18 INT3-G16 plus BiasCorr at 0.270738 in this specific causal KL test. Every tested BiasCorr variant makes Bonsai worse.
6. Plain rotated scalar KV compression beats the tested TurboQuant-style 1-bit QJL residual sketch at 3-bit and 4-bit KV budgets. At K3V4, combined three-domain KL is 0.01689022 for rotated scalar quantization and 0.05289135 for TurboQuant, a 3.13x difference in favor of the simpler method. The QJL implementation passes its own unbiasedness self-test, so this is not explained by a broken implementation.
7. Weight and KV compression stack almost additively. Measured joint error is 93 to 98 percent of the sum of separate errors. This means the methods do not explosively compound in the tested regime, but they also do not cancel each other.
8. The dense serving story is honest mixed news. HF-INT4-G16 improves 8K prefill throughput by 1.28x and 32K prefill throughput by 1.26x, and uses 10.7 percent less energy per output token at 32K in the measured test setup. It is slower in single-stream decode, by 11.8 percent at short context, 11.2 percent at 8K, and 10.1 percent at 32K.
9. The v0.2.0 MoE work is stronger. It adds multimodal Gemma-4-26B serving (the live boot used the raw-text BiasCorr checkpoint that failed the 4-check gate; see section 8.3) and a bit-exact grouped-expert prefill path. The final quiet-host Gate5 rerun shows packed MoE beating BF16 by 1.48x to 2.06x eager and 2.18x to 2.73x graph replay across M 8 to 64. Grouped execution wins every tested cell from M 32 onward, but BF16 still wins at M 512 because the current Step A kernel becomes compute-bound.
10. Quiet-host Gate5 also closes the TensorFold v1 serving gap. The published `f5fce21b` sidecar adds only +0.00 to +0.30 percent TTFT overhead and +0.12 to +0.27 percent draft-round time across 0 to 128K context, with 11/11 drafted-equals-serial exactness both on one GPU and on `--tp 2`.
11. The review process caught real bugs: a non-causal attention oracle, a checkpoint conversion scale mismatch, an OLMoE FP32 accumulation bug, a left-padding calibration bug, an operator misuse of `attn_implementation` in a live server, and a shipped GPTQ checkpoint whose direction had been deliberately modified. These catches are part of the result. Both Claude Opus 5.5 and GPT-6.2 Astra signed off `vllm-hf-int4-g16-xpu` v0.2.0 after the final admission-spoof regression; the Wave9F plus Wave9H release chain covers 700 tests (699 in one logged run, plus 1 added test with an author-reported pass), and the Wave9G review clears the uniform-Had64 `/2` format to freeze.

Honest scoreboard:

| Claim | Result | Status |
|---|---:|---|
| HF-INT4-G16 data-free KL below MLX-4bit | 12/12 families | supported |
| HF-INT4-G16 data-free KL below SymINT4-G128 RTN | 11/11 families with that arm; on Qwen3.8-27B also below shipped GPTQ-G128 | supported |
| BiasCorr is universal | no | falsified |
| BiasCorr only helps Qwen3.8-like models | no | falsified |
| BiasCorr rescues 2-bit graded tasks | 31 to 93/160 | supported in controlled battery |
| BiasCorr improves 3-bit graded total | 122 to 122/160; 58 to 57/60 GSM8K at 1024 cap | null result, earlier gain was truncation artifact |
| BiasCorr helps calibrated GPTQ | WT2 worsens about 5.7 percent | rejected for that checkpoint |
| BiasCorr helps Bonsai-2 | 0.653 to 1.507 combined KL for one form | rejected |
| TurboQuant QJL improves over same-bit rotated scalar KV | no at 3 to 4 bits | rejected in this implementation |
| Dense HF-G16 decode is faster than shipped GPTQ | no, 10 to 12 percent slower at C=1 | honest negative |
| Dense HF-G16 long prefill is faster than shipped GPTQ | 1.28x at 8K, 1.26x at 32K | supported in measured test setup |
| Grouped MoE prefill at batch 512 | 3.411x Gemma shape, 3.268x Qwen shape eager | supported on quiet host; BF16 packed-runner comparison remains negative at M 512 |
| TensorFold v1 sidecar serving overhead | TTFT +0.00 to +0.30 percent, draft round +0.12 to +0.27 percent | supported on quiet host |
| TensorFold v1 drafted exactness | 11/11 on GPU0 and 11/11 on TP2 | supported within each topology |

## 2. Silicon-first format discovery
### 2.1 Hardware constraints came first
The target silicon is the Intel Arc Pro B70, Xe2 architecture, BMG-G31 chip. The constraints that matter for format design are concrete:

* 32 GB GDDR6 per card.
* About 608 GB/s pin bandwidth, with the best large-tile measured stream around 540.8 GB/s.
* 32 Xe-cores and 256 XMX units per card.
* No native FP4 or FP8 DPAS matrix path on Xe2.
* BF16 and FP16 matrix math use DPAS K16.
* INT8 uses K32 and has about twice the peak matrix throughput.

Decode is memory-bandwidth dominated. For a weight-only low-bit format, the first-order cost is bytes read per token. A format that looks compact but needs scattered memory access, expensive per-weight branching, or emulation into another type can lose the advantage before arithmetic starts.

The study therefore rejected the superficial argument that FP4 or FP8 storage is automatically best because it is low-bit. If the matrix unit cannot consume that type natively, the real cost is the whole dequantize and multiply pipeline, not the label on the checkpoint.
### 2.2 Why Had128NoPerm plus HF-INT4-G16
`HF-INT4-G16` stores integer 4-bit codes in groups of 16. Progressive scales keep a coarse amplitude representation plus compact local correction. The online Hadamard rotation spreads isolated outliers over a 128-dimensional block before scalar rounding.

The important point is not that Hadamard makes weights small in some abstract sense. It changes the quantization error geometry. A scalar quantizer has a fixed grid. A single large magnitude coordinate forces either a coarse scale or a large clipping error. Rotation distributes that energy, so a fixed scalar grid sees a less adversarial vector.

`NoPerm` means the serving layout does not require a random permutation in memory. Fixed Sylvester structure can be regenerated or folded into arithmetic. Random-looking data movement is expensive on a bandwidth-bound part. This is a silicon decision, not a numerical convenience.

The effective weight payload of the tested HF-G16 format is about 4.25 bits per weight. The exact effective payload depends on endpoint treatment and whether small residual tensors are quantized. The Exp15A common-matrix comparison uses 4.2602586206896556 bpw for HF-G16 and 4.0009765625 bpw for TFQ3.
### 2.3 Why group 16 rather than group 64 or 128
Group size is a rate-distortion and hardware-tiling compromise.

A larger group stores fewer scales but forces unlike weights to share a scale. A smaller group controls local amplitude better but costs more scale bits and more instruction overhead. Group 16 is the smallest tested scale quantum that still maps cleanly onto the XMX tile structure while preserving local amplitude resolution.

The rate-distortion measurements supported this direction. Shrinking group size from G32 to G16 had a predicted +0.97 dB gain and a measured +1.03 dB gain on real weights. Lloyd-Max16 measured 20.22 dB on the scalar reference, within 0.004 dB of prediction. The E8 lattice reference measured 21.02 dB. EXL3-4.00 measured 22.43 dB mean weight SQNR across the tested eight tensors. These are weight-domain references, not task accuracy, but they explain why the scalar G16 design is competitive even before correction.
### 2.4 The 12-family sweep
The exact family-level data-free comparisons are:

| Family | HF-G16 combined KL | MLX-G64 combined KL | Receipt scope |
|---|---:|---:|---|
| Qwen3.8-27B | 0.0904 | 0.1173 | WT2 plus code mean, Exp13A |
| Qwen2.5-1.5B | 0.1273 | 0.1787 | WT2 plus code mean, Exp13B |
| SmolLM2-1.7B | 0.1662 | 0.1910 | WT2 plus code mean, Exp13B |
| Llama-3.2-3B | 0.0796 | 0.1028 | compact combined KL, Exp15B |
| OLMo-2-7B | 0.0689 | 0.0905 | compact combined KL, Exp15B |
| OLMoE-1B7B | 0.1064 | 0.1344 | compact combined KL, Exp15B |
| Phi-4-mini | 0.1917 | 0.2298 | compact combined KL, Exp15B |
| Granite-4.0-H-Tiny | 0.10245282 | 0.13197865 | combined causal KL, Exp17A |
| Gemma-4-E4B-it | 0.28312991 | 0.50524378 | combined causal KL, Exp17A |
| Qwen3-8B | 0.11330444 | 0.14677614 | combined causal KL, Exp17A |
| Gemma-4-26B-A4B-it | 0.15772356 | 0.26208259 | chat-layout combined causal KL, Exp17B |
| Qwen3.6-35B-A3B | 0.11113260 | 0.15587649 | WT2 plus code mean, Exp17C |

HF-G16 is lower than MLX-4bit in all 12 rows. Against SymINT4-G128 RTN it is lower in 11/11 rows that have that arm. Qwen3.8-27B has no RTN arm, so it is compared with the shipped, calibrated GPTQ-G128 checkpoint (combined KL 0.1757 versus 0.0904); that is a different kind of baseline and is not counted in the 11. The margin ranges from small on SmolLM2 and Qwen3.6 to very large on Gemma-4-E4B. The scope column matters: older receipts use a two-domain mean, while newer receipts use their documented combined causal metric. Do not compare absolute values across rows as if they were one calibration corpus. Compare the two arms inside a row.

NVFP4 sometimes beats HF-G16 in data-free KL. On Qwen3-8B, NVFP4 combined KL is 0.08889713 versus 0.11330444 for HF-G16. On SmolLM2, NVFP4 is also lower than HF-G16. That does not contradict the serving result. NVFP4 is not a generic reproducible BF16 conversion path on Xe2 with a native low-bit matrix pipeline. It is a reference point for what trained or vendor-native formats can do, not the practical conversion target in this study.
### 2.5 Why Gemma-4-26B required uniform Had64
Gemma-4-26B-A4B has awkward inner dimensions in its expert path. The dimensions `K=704` and `K=2112` are divisible by 64 but not by 128. A rigid Hadamard-128 transform cannot cover those rows without padding or splitting the transform domain. Padding would waste work and complicate the exact packing contract.

The first conversion choice used Hadamard-128 where possible and smaller blocks elsewhere. Exp17B showed that this mixed-tail policy was not simply a free win. Wave 9 review then correctly refused to treat the v0.2.0 kernel as quality-proven until a uniform Had64 comparison existed.

Exp17D tested uniform Had64, halving the Hadamard span. The decisive result is that Had64 and BiasCorr are not interchangeable and not cancelling. In the causal KL ablation, moving from the mixed conversion to uniform Had64 improves the data-free arm, and adding BiasCorr improves it further. The exact combined WT2-plus-code deltas are: uniform Had64 effect without BiasCorr `-0.02177480`; BiasCorr effect without uniform Had64 `-0.05770311`; both together versus neither `-0.06977901`. These are negative deltas, meaning lower KL. They do not mean that one mechanism merely undoes the other. They attack different error terms: rotation changes rounding geometry; BiasCorr corrects the mean of the remaining output error.
### 2.6 Format discovery was constrained by engine reality
The serving format had to satisfy properties that a pure rate-distortion contest does not reward:

* deterministic integer packing;
* a fail-closed layout validator;
* fused checkpoint loading;
* W4A16 and W4A8 dispatch;
* per-layer patching with exact zero identity;
* tensor-parallel replication proofs;
* a small enough epilogue to add BiasCorr without rewriting every model definition.

That is why the best matrix-error format is not automatically the best serving format. The next section and the trellis section make that distinction explicit.

## 3. BiasCorr across 12 architectures
### 3.1 Exact mechanism
For a BF16 linear layer with weight `W_bf16` and quantized-then-dequantized weight `W_quant`, define the weight error

`E = W_bf16 - W_quant`.

For calibration activations `x`, the mean input vector is

`mu = E[x]`.

The first-order correction is

`delta_b = E @ mu`.

The corrected output is

`y_corr = x @ W_quant.T + delta_b`.

In implementation, the addition happens in FP32 in the output epilogue before the final BF16 cast. This detail matters. If the correction is added after a BF16 rounding boundary, a small correction can round away or interact with the output scale differently.

Position zero is excluded from calibration means. The first token position has different activation statistics in several dense and hybrid models. Including it can produce a correction that is optimal for an attention-sink-like state rather than for normal token positions.
### 3.2 Why the correction can work
Quantization error is not zero-mean in general. Even when rounding is locally unbiased, nonlinear activation distributions, asymmetric weight matrices, clipping, and scale selection can produce a systematic output shift. If that shift is stable across the input distribution, a first-order mean correction removes a large fraction of it.

This is why BiasCorr is cheap. It does not try to reconstruct the input-dependent error. It subtracts only the persistent mean component. The residual variance remains.
### 3.3 Why the correction can fail
There are at least four failure mechanisms.

1. The mean error can be small relative to input-dependent variance. Then the vector adds little or nothing.
2. Calibration activations can come from the wrong distribution. A chat-tuned model evaluated on chat prompts should not be corrected only with raw web text if the system prompt and template change activations.
3. A calibrated checkpoint can already have removed the mean error through its own calibration procedure. Adding another mean correction then double-counts it.
4. A heavily optimized low-bit checkpoint can use the mean shift as part of a nonlinear compensation strategy. Bonsai-2 appears to be in this category. Correcting each layer locally can move the network away from the globally optimized solution.
### 3.4 The mandatory gate
`export_biascorr.py` therefore treats a sidecar as guilty until proven useful. The automatic gate requires:

* improvement on the positive evaluation domains;
* a negative `-b` control that does not reproduce the improvement;
* paired block-bootstrap confidence intervals rather than one mean;
* an out-of-domain check rather than only WikiText-2;
* exact zero patch identity for the zero-bias control path.

The negative control matters because any nonzero vector can sometimes help by chance on one narrow corpus. The sign flip should not preserve a real first-order correction.
### 3.5 Both simple stories were falsified
The first hypothesis was that BiasCorr is universal. It failed on small dense non-QK-Norm models:

* SmolLM2-1.7B became about 18 to 51 percent worse depending on format and domain.
* Llama-3.2-3B became about 17 to 43 percent worse.
* Phi-4-mini became about 3 to 24 percent worse.
* OLMo-2-7B became about 92 to 211 percent worse in the worst cases.

The opposite hypothesis was that it only helps Qwen3.8-like models. That also failed. Gemma-4-E4B-it passed the gate for all tested formats. Combined causal KL moved as follows:

| Gemma-4-E4B arm | Data-free | BiasCorr | Relative direction |
|---|---:|---:|---|
| HF-INT4-G16 W4A16 | 0.28312991 | 0.22304306 | about 21 percent lower |
| MLX-asym-G64 | 0.50524378 | 0.30360750 | about 40 percent lower |
| NVFP4 | 0.25565009 | 0.19645004 | about 23 percent lower |
| SymINT4-G128 | 1.23578554 | 0.49314147 | about 60 percent lower |

Gemma-4-26B-A4B showed the same direction at MoE scale on WikiText-2 plus code. Read the next table with one hard limit. It uses the Exp17B mixed layout (Hadamard-128 with smaller blocks on the tails). Exp17D re-ran the gate on that layout for HF-G16: `Mixed__shared_only` and `Mixed__all_experts` both pass the 3-check and are both `REJECTED_BY_KL_GATE` on the 4-check, because the out-of-domain chat KL drop is not separable from zero. These HF-G16 mixed-layout gains are not shippable. Exp17D did not re-gate the NVFP4, MLX or SymINT4 rows. The gate results for each layout are in section 3.7.

In the Exp17B chat layout, mixed Hadamard blocks, WikiText-2 plus code:

| Gemma-4-26B arm | Data-free | BiasCorr all-experts | Relative direction |
|---|---:|---:|---|
| HF-INT4-G16 W4A16 | 0.15772356 | 0.11258001 | about 29 percent lower |
| NVFP4 | 0.16880991 | 0.11544378 | about 32 percent lower |
| MLX-asym-G64 | 0.26208259 | 0.16917791 | about 35 percent lower |
| SymINT4-G128 | 0.44910335 | 0.28976358 | about 35 percent lower |
### 3.6 The 2 to 2.5 MiB shared-only MoE law
The large-expert correction initially looked expensive. Gemma-4-26B has 128 routed experts. Qwen3.6-35B-A3B has 256 routed experts. Correcting every expert produces a sidecar dominated by rarely used matrices.

The measured surprise is that most of the benefit comes from attention plus shared dense paths. On Gemma-4-26B, the dense attention plus shared-MLP correction is 2,186,240 tensor bytes, about 2.08 MiB. For HF-INT4-G16 in the Exp17B mixed layout (which fails the 4-check gate, section 3.7) it gives combined KL 0.11157237, statistically inseparable from the all-experts 0.11258001 and far below data-free 0.15772356. The full expert-heavy Gemma correction is 67,100,176 file bytes, about 64 MiB, roughly 31x the dense-plus-shared tensor subset. On Qwen3.6 the same comparison is 122.5391 MiB all-experts versus 2.5391 MiB shared-only, about 48x.

Qwen3.6 shows the same law. A 2.54 MiB shared-only sidecar brings HF-INT4-G16 from 0.126 pooled KL to 0.092 in the Exp17C process summary and passes the three-check and out-of-domain gates. Correcting all 256 experts costs 122.5 MiB and cannot be separated from shared-only correction at the tested calibration size.

The mechanistic interpretation is straightforward. Attention and shared expert matrices are always active. Routed experts are sparsely active and their activation-conditioned error means are harder to estimate from a small calibration set. A few thousand calibration positions cannot estimate 256 distinct expert error means with the same reliability.
### 3.7 Calibration distribution matters
Raw text calibration initially failed the out-of-domain check on chat-tuned Gemma-4. This was not evidence that the model was broken. It was evidence that activations under a chat template differ from raw document activations.

Chat calibration alone did not fix the gate. The Exp17B mixed layout with chat-calibrated BiasCorr still fails the 4-check. It took chat calibration and the uniform Had64 layout together. Uniform Had64 with the converter's raw-text calibration also fails. Among the layouts that can be served (mixed and uniform Had64), only uniform Had64 with chat calibration passed both the 3-check and the 4-check (Exp17D, HF-G16):

| Layout and BiasCorr (Exp17D) | 3-check | 4-check | OOD chat KL, base to corrected | 95 percent block CI of the change |
|---|---|---|---:|---|
| Mixed, chat calibration, shared-only | pass | fail | 1.164 to 1.095 | [-0.175, +0.028] |
| Mixed, chat calibration, all-experts | pass | fail | 1.164 to 1.074 | [-0.196, +0.010] |
| Uniform Had64, raw-text calibration | pass | fail | 1.167 to 1.211 | [-0.074, +0.165] |
| Uniform Had64, chat calibration, shared-only | pass | pass | 1.167 to 0.982 | [-0.263, -0.111] |
| Uniform Had64, chat calibration, all-experts | pass | pass | 1.167 to 0.988 | [-0.257, -0.108] |
| Uniform Had64 live kernels, raw-text calibration | pass | fail | 1.215 to 1.249 | [-0.089, +0.162] |
| Uniform Had64 live kernels, chat calibration, shared-only | pass | pass | 1.215 to 1.013 | [-0.304, -0.108] |
| No-rotation control, chat calibration, shared-only | pass | pass | 1.346 to 1.156 | [-0.278, -0.112] |

The last row is an ablation control, not a serving layout. It also passes both gates, but its corrected WikiText-2 plus code KL is 0.1296 against 0.1175 for uniform Had64. "Live kernels" means offline scoring through the plugin's own GPU kernels, not a server run.

The served checkpoint used raw-text BiasCorr, which failed the 4-check out-of-domain (OOD) chat gate and is held back. No server has yet booted with the chat-calibrated BiasCorr that passed. The chat-calibrated sidecar that passed came from the study's own calibration script. The converter's `--cal-chat-template` option builds different calibration rows, and its output has not been measured.

The operational rule: match the correction distribution to the serving distribution, then verify out-of-domain anyway. Do not treat calibration as a license to skip evaluation.
### 3.8 Calibrated GPTQ rejection
On the shipped calibrated GPTQ checkpoint for Qwen3.8-27B, BiasCorr is not a clean improvement. WikiText-2 KL worsens from 0.04941556 to 0.05221039, about 5.7 percent worse, while code KL improves from 0.30198259 to about 0.287. This mixed result fails the automatic gate because a correction must improve the required positive domains, not trade one against the other.

This result makes sense. GPTQ calibration already uses activation information to reduce output error. Adding a new mean correction can double-count structure that the quantizer has already exploited.
### 3.9 Bonsai rejection
Every tested BiasCorr variant hurts PrismML `Ternary-Bonsai-2-27B`. The basic weight-difference correction moves combined KL from 0.653058 to 1.507160. The activation-residual correction is much worse at 9.316942. The open residual form is 2.111884. Even the negative control does not reproduce a simple symmetric pattern, which means the error structure is not a simple zero-mean random direction.

The gate correctly says ship no sidecar. Bonsai is a counterexample to the idea that mean correction is always a free local improvement.

## 4. Where BiasCorr becomes real task rescue
### 4.1 Why ultra-low-bit is different
At 4-bit, quantization error is often small enough that model-specific structure dominates. At 2-bit or ternary, the scalar grid is coarse. The systematic error component becomes large relative to the remaining signal. That is the regime where a cheap first-order correction can produce a visible task-level change.

Exp18 used a controlled graded battery of 160 frozen items: 60 GSM8K, 60 MMLU, and 40 HumanEval. The task runs used greedy decoding, thinking disabled, and batch 24. Some suites have high truncation rates. Treat the result as controlled comparative evidence, not as a broad user-quality benchmark or a serving benchmark.
### 4.2 Two-bit integer rescue
For `had_int2_g16` at about 2.26 bpw, the exact results are:

| Suite | Data-free | BC_open | Change |
|---|---:|---:|---|
| total | 31/160 | 93/160 | +62 net |
| GSM8K | 8/60 | 50/60 | +42 |
| MMLU generation | 14/60 | 31/60 | +17 |
| HumanEval | 9/40 | 12/40 | +3 |
| MMLU letter/logprob | 11/60 | 31/60 | +20 |

The paired McNemar table for the total battery is:

* fixed by BiasCorr: 72;
* broken by BiasCorr: 10;
* discordant items: 82;
* exact two-sided p: 1.0224592793857492e-12.

For GSM8K, the table is 42 fixed and 0 broken, exact p 4.547473508864641e-13. For MMLU generation, p is 0.003326892852783203. HumanEval alone is not significant, p 0.5078125.

The result is statistically unmistakable in aggregate. It is not equally distributed across tasks. That distinction should remain in every public retelling.

The truncation control matters because the generation cap was only 384 tokens. On the 17 GSM8K items that finished in both 2-bit arms without hitting that cap, data-free output is 6/17 correct and BC_open is 15/17 correct. The paired table is 9 fixed and 0 broken, exact two-sided p 0.00390625. This subset does not replace the primary 160-item result. It shows that the rescue is not created by counting truncated outputs.
### 4.3 Perplexity and KL at ultra-low-bit
At `had_ternary_g16`, BiasCorr moves WikiText-2 PPL from 322 to 56.2 and out-of-domain chat PPL from 187 to 12.3 in the Exp18 measurements. The exact combined KL values depend on the group size and rotation configuration.

The most important ternary comparison is multiplicative, not additive. Without rotation or correction, ternary error can be catastrophic. Hadamard rotation reduces the rounding error. BiasCorr removes the remaining mean shift. At the tested ternary setting, the combined effect moves WikiText-2 PPL from about 31,000 to 106 in the process narrative. Neither mechanism alone explains that outcome.
### 4.4 BC_open beat BC_closed in six of six tested comparisons
Exp18 compared a closed weight-difference correction with an open residual form. The open form won all six head-to-head comparisons in the ultra-low-bit study. The reason is algebraic. The closed form assumes that the correction can be computed exactly from `(W_bf16 - W_quant) @ E[x]`. In practice, nonlinear activation functions, downstream normalization, and implementation order mean that the direct activation residual can capture the realized output mean error better than a pure weight-space estimate.

This does not make the open form universally safe. It still failed on Bonsai. It is simply the better candidate inside the Exp18 PTQ regime.
### 4.5 The honest 3-bit null and 2-bit ceiling
At `had_int3_g16`, both data-free and BC_open score 122/160 total in the original battery. The exact McNemar p for total is 1.0. The first GSM8K readout at a 384-token cap improved from 41/60 to 48/60, while MMLU dropped from 45/60 to 38/60. Exp20 shows why that GSM8K subset reading must not be interpreted as a task rescue.

In the Exp20 1,024-token rerun, uncorrected 3-bit already scores 58/60 with zero truncated outputs. BC_open scores 57/60 with one truncated output. The paired table is 0 fixed and 1 broken, exact p 1.0. Restricting the same new run to 384 tokens reproduces the artifact shape: 41/60 uncorrected versus 50/60 corrected, 9 fixed and 0 broken, exact p 0.00390625. The original Exp18 cap-384 batch was 41/60 versus 48/60, 8 fixed and 1 broken, exact p 0.0390625. The difference comes from answers cut off by the generation cap, not from a durable reasoning improvement.

The correct boundary is therefore sharper than the earlier draft: 3-bit and 4-bit sit near the accuracy plateau in this battery, while the real task-accuracy rescue is at 2-bit. Corrected 2-bit still scores 93/160, far below the 3-bit total 122/160. BiasCorr does not make 2-bit equivalent to 3-bit. It makes one coarse 2-bit configuration substantially less broken on one controlled battery.

The 2-bit output also still has repetition and truncation problems. A public post that says only "31 to 93" without this caveat would be misleading.

## 5. PrismML Ternary-Bonsai-2-27B versus PTQ ternary, INT2, and INT3
### 5.1 Why Bonsai deserved a separate experiment
Bonsai is not a post-training conversion produced by this study. It is a shipped ternary-weight checkpoint from PrismML. Comparing it with the study's own post-training ternary conversion answers a different question: how much of the quality gap is caused by the ternary format itself, and how much is caused by optimize-after-training versus optimize-for-low-bit-training?

The exact training method is not stated in the public artifact inspected here. The study does not call it QAT as a verified fact. It only says that its trits differ systematically from an MSE-optimal post-training assignment.
### 5.2 Exact causal KL comparison
The primary same-corpus comparison uses 64 causal layers, three domains, and 2,044 scored targets per domain. Combined KL is the mean over WikiText-2, code, and out-of-domain chat.

| Arm | WT2 KL | Code KL | OOD chat KL | Combined all3 KL | WT2 PPL |
|---|---:|---:|---:|---:|---:|
| Bonsai as shipped | 0.413351 | 0.933017 | 0.612807 | 0.653058 | 11.673 |
| Bonsai dequantized, BF16 embedding and Bonsai head | 0.411686 | 0.936537 | 0.606408 | 0.651544 | 11.722 |
| Bonsai trits only, BF16 rest and embedding | 0.630521 | 1.115302 | 0.703595 | 0.816473 | 15.088 |
| MSE-optimal PTQ same format, BF16 embedding | 3.796641 | 5.212289 | 5.277987 | 4.762306 | 351.435 |
| MSE-optimal PTQ same format plus weight-difference BC | 2.694703 | 2.745192 | 2.775131 | 2.738342 | 114.129 |
| Exp18 ternary G64 data-free | 4.299308 | 4.932828 | 4.742413 | 4.658183 | 588.572 |
| Exp18 ternary G64 plus BC_open | 2.611294 | 2.763163 | 2.693340 | 2.689266 | 105.684 |
| Exp18 INT2-G16 plus BC_open | 0.927676 | 1.336346 | 1.149279 | 1.137767 | 20.297 |
| Exp18 INT3-G16 plus BC_open | 0.133262 | 0.429945 | 0.249008 | 0.270738 | 8.842 |
| Exp18 MLX3-G64 plus BC_open | 0.138867 | 0.480503 | 0.274149 | 0.297840 | 9.063 |

The conclusion is narrow and strong: Bonsai's ternary weights are far better than post-training ternary conversion. Its combined KL is 0.653 versus 2.689 for the best corrected Exp18 ternary arm. It is also better than corrected INT2-G16 at 1.138. In this exact test, corrected INT3-G16 at 0.271 still has lower KL.

This is not a universal quality ranking. The test measures teacher-forced causal KL on three domains, not broad chat preference, long-form coherence, or safety behavior.
### 5.3 The same-format control proves the edge is in the trits
The study separated the Bonsai checkpoint into effects:

* dequantized container effects;
* BF16 endpoint effects;
* ternary weight assignment effects;
* rotation sign-vector effects.

The same-format PTQ control uses the same 2-bit trit packing, one third sparsity target, BF16 endpoints, and MSE-optimal trit assignment. It lands at combined KL 4.762 data-free and 2.738 with weight-difference BiasCorr. Bonsai lands at 0.653 without any correction. Therefore the gap is not explained by the packing container or endpoints. It is mostly in the optimized trit assignment.

The direct weight comparison supports that interpretation. Across 400 ternary linears, Bonsai trits agree with MSE-optimal PTQ trits on 85.339572 percent of entries. Bonsai has 32.757977 percent zeros, while MSE-optimal PTQ has 45.720035 percent zeros. Median row cosine to BF16 is 0.880630 with the rotation signs applied and near zero without them. Bonsai weight SQNR versus BF16 is 6.490462 dB, lower than MSE-optimal PTQ at 7.281278 dB. Lower weight-domain SQNR but much lower output KL is exactly what one expects if the trits are optimized for network function rather than per-matrix mean squared error.
### 5.4 The RMSNorm offset catch
The conversion initially looked impossible because the HF `Qwen3_5RMSNorm` module computes `x * (1 + weight)`, while the MLX-style packed checkpoint stores the already incremented value `1 + weight`. A loader that treats the stored value as the HF parameter applies `1 + (1 + weight)` and corrupts normalization.

The output-mean invariant caught this before scoring. The correct conversion subtracts 1 when loading that module class. This is an example of why exact patch identity and output invariants are more valuable than a model merely producing plausible text.
### 5.5 Why BiasCorr fails on Bonsai
The measured failures are large:

| Bonsai correction arm | Combined all3 KL |
|---|---:|
| no correction | 0.653058 |
| weight-difference BC | 1.507160 |
| weight-difference negative control | 1.191400 |
| activation-residual BC | 9.316942 |
| activation-residual negative control | 2.730254 |
| open activation-residual BC | 2.111884 |
| open activation-residual negative control | 3.424723 |

The negative controls do not behave like a clean sign-flipped version of a useful correction. That tells us the correction is interacting with a structured optimization basin, not simply removing a random unwanted offset.

A plausible mechanism is that Bonsai's optimized trit assignment uses layer interaction and bias-like shifts as part of the global solution. A local correction that improves the first-order mean of one layer can move later layers away from the distribution they were optimized around. The receipt does not prove this training mechanism because the training method is not public in the inspected artifact. It proves only the outcome: no sidecar should ship.

## 6. KV compression stacked on HF-INT4-G16 plus BiasCorr
### 6.1 What was actually tested
Exp19A uses the term `Polar` for a specific implementation: sign flips, Hadamard-256 rotation, FP16 RMS scale, and Gaussian Lloyd-Max scalar quantization. This is not the polar-coordinate transform from the paper that inspired the name.

The TurboQuant-style arm uses that scalar quantizer at `b-1` bits plus a 1-bit QJL residual sketch. QJL is intended to preserve inner products statistically by adding a random-projection correction. The implementation passed its self-test: inner-product slope 0.9998, relative MSE 1.566 versus theoretical 1.567. This makes the quality loss interesting rather than trivial.
### 6.2 Exact headline comparisons
| KV arm | Combined all3 KL |
|---|---:|
| BF16 KV | 0.0 exactly by construction |
| FP8 KV | 0.00165083 |
| Polar K4V4 | 0.00722605 |
| Polar K3V4 | 0.01689022 |
| TurboQuant K3V4 | 0.05289135 |
| Polar K3V3 | 0.02418088 |
| TurboQuant K3V3 | 0.06397003 |
| Polar K2V4 | 0.05222160 |
| TurboQuant K2V4 | 0.11863084 |

At K3V4, TurboQuant KL is 3.13x the rotated scalar arm. At K3V3, the ratio is 2.65x. Seed spread is small relative to that gap. Polar K3V4 ranges from 0.01634111 to 0.01926962 across seeds. TurboQuant K3V4 ranges from 0.04994383 to 0.05289135.
### 6.3 Why the 1-bit sketch loses here
The QJL sketch adds unbiased information about inner products, but unbiased does not mean low variance. At the tested 3 to 4 bit total budgets, the scalar part loses one bit to pay for the sketch. That coarsens the base quantization of every element. The sketch then adds variance with a theoretical penalty around 1.57x in the relevant error term.

At very low base precision, or in a system whose bottleneck is inner-product bias rather than elementwise attention error, the trade could reverse. It did not reverse in this implementation. Several post-hoc comparisons asking what the QJL bit adds were not statistically significant, and `TQ_K4V3` was significantly worse than `Polar_K3V3` by +0.00908939 combined KL.

The correct conclusion is not "QJL is bad." The correct conclusion is narrower: this exact QJL residual sketch, at these budgets, on this model and corpus, loses to simply spending the bit on the scalar quantizer.
### 6.4 V sensitivity exceeds K sensitivity in most weight arms
The study's weight-format results often show V-only quantization causing more damage than K-only quantization in 11 of 12 family/format cells. Under symmetric rotated KV quantization, however, the K/V balance can change because rotation changes error geometry and attention usage.

Gemma-4-26B gives a concrete example in the weight-domain probe. Full-depth K-only HF-INT4-G16 data-free combined KL is 0.03356326, while V-only is 0.05043667. In the sliding-layer-only matched comparison, K-only is 0.01888637. Exp13B had already falsified a universal "K is always worse" rule: Qwen2.5-1.5B is the one reversal in the 11/12 count (`results/qwen38-27b/quantization-study-v11-202611/exp13b_qwen25_dense.json`; chart6). Per-architecture measurement remains necessary.
### 6.5 Weight and KV compression stack almost additively
The relevant question is whether errors compound nonlinearly when both weights and KV are compressed. In the tested regime they do not. Joint KL is about 93 to 98 percent of the sum of the separate weight-only and KV-only terms.

One exact anchor is:

* HF-G16 weights plus BiasCorr with BF16 KV: combined all3 KL 0.06998040.
* HF-G16 weights plus BiasCorr with Polar K3V4 KV: combined all3 KL 0.08149849.

This is mild superposition, not cancellation and not explosion. The mechanism is plausible: weight error changes the hidden states and attention inputs, while KV error affects retrieval inside attention. Their interaction exists, but in the measured regime the first-order terms dominate.
### 6.6 256K context footprint on one B70
Arithmetic footprint at 256K tokens, excluding work buffers, is:

| KV format | KV bytes | Total model plus KV |
|---|---:|---:|
| BF16 KV | 16.0 GiB | 32.962 GiB |
| FP8 KV | 8.0 GiB | 24.962 GiB |
| Polar K3V4 | 3.56 GiB | 20.524 GiB |
| TurboQuant K3V4 | 3.59 GiB | 20.555 GiB |

BF16 KV does not fit a 32 GB card once work buffers are included. FP8 KV fits arithmetic footprint but leaves less room for allocator fragmentation and activations. K3V4 leaves roughly 11 GiB before the card's visible memory ceiling, enough for real runtime buffers in the arithmetic model. This is a footprint calculation, not yet a 256K live serving proof.
### 6.7 Exactness checks
The KV experiment included strict controls:

* exact 0.0 repeat identity;
* exact 0.0 future-prefix isolation;
* exact 0.0 other-row isolation;
* hooked unquantized KV equal to stock SDPA with max absolute difference 0.0;
* BF16 KV KL exactly 0.0;
* the v1 gate anchor reproduced six published HF-G16 KL values with difference 0.0.

These checks matter because KV compression interacts with attention masks and cache indexing. A small indexing error can look like quantization noise unless the identity path is exact.

## 7. Trellis versus scalar limits
### 7.1 Trellis really wins rate-distortion
Exp15A compared trellis-coded `TFQ3-4b` with scalar HF-G16 on the 400 common decoder matrices of Qwen3.8-27B. TFQ3 used 4.0009765625 bpw; HF-G16 used 4.2602586206896556 bpw. The trellis format is the stronger rate-distortion object. That result was not hidden or minimized.

The matched checkpoint comparison shows TFQ3's strength and its cost. Exact KL values:

| Arm | WT2 KL | Code KL |
|---|---:|---:|
| TFQ3 matched400 | 0.02528476 | 0.11041974 |
| TFQ3 matched400 plus BC | 0.01852207 | 0.08878587 |
| HF-G16 shipped matched400 W4A16 DF | 0.03175076 | 0.14800582 |
| HF-G16 shipped matched400 W4A16 shipped BC | 0.02716107 | 0.12780215 |
| TFQ3 as shipped | 0.03205803 | 0.11327348 |
| HF-G16 as shipped W4A16 BC | 0.02707786 | 0.12897424 |

TFQ3 matched400 is clearly better than the matched scalar format on these two domains. It also responds to BiasCorr. The study does not claim scalar G16 beats trellis in pure rate-distortion.
### 7.2 The lm_head tax is real
The shipped TFQ3 artifact modifies endpoint tensors. The endpoint payload check shows that the TFQ embedding does not exactly match BF16, while HF-G16 embedding and lm_head do match. A BF16 body with the TFQ3 lm_head alone has WT2 KL 0.00678586 and code KL 0.00327073. That is a small but nonzero cost caused only by the head substitution.

Top-1 agreement also exposes the head cost. The exact amount depends on arm and corpus. The V11 summary uses the previously verified 4.45 percentage point lm_head tax as a concise endpoint effect in the TFQ comparison context. It should not be generalized to every TFQ artifact or every model.
### 7.3 Exp15C tried to close the gap without trellis
Exp15C asked whether scalar HF-G16 refinement could reach TFQ3 without extra bits. The answer was no.

The best zero-extra-bit arm was `HF_G16_ActMSECD_W4A16_BC`, with WT2 KL 0.02738961 and code KL 0.12552216. TFQ3 as shipped had WT2 KL 0.03205803 and code KL 0.11327348. Thus the scalar arm beat TFQ3 on WikiText-2 but not on code.

A mixed-precision arm replacing the small input projections and the top eight down-projections with BF16 cost an additional 1.0072746276855469 GiB and moved effective payload from 4.2602607923350195 bpw to 4.61558833028313 bpw. Even that expensive Pareto arm did not beat TFQ3 on both domains.

The operational conclusion is not that trellis is useless. It is that trellis and scalar serving solve different optimization problems. Trellis is the better compression object. Scalar HF-G16 is the simpler Xe2 serving object with exact packing, low overhead, and straightforward correction plumbing.

## 8. Live vLLM-XPU serving on Intel Arc Pro B70
### 8.1 Dense Qwen3.8-27B speed and energy
The dense speed ladder compares vLLM serving of HF-INT4-G16 plus BiasCorr with the shipped GPTQ checkpoint. The benchmark uses the model-card sampling configuration, thinking disabled in the benchmark chat kwargs, seeds 4242, 4243, and 4244, and separate TTFT, prefill, decode, and energy measurements.

Exact mean speed table:

| Arm and context | TTFT s | Prefill tok/s | Decode tok/s | J/output token |
|---|---:|---:|---:|---:|
| HF-G16 short | 0.1146 | 548.4 | 29.48 | 7.861 |
| GPTQ short | 0.1139 | 559.3 | 33.42 | 6.912 |
| HF-G16 8K | 3.3707 | 2364.4 | 28.38 | 11.110 |
| GPTQ 8K | 4.3099 | 1849.1 | 31.95 | 11.028 |
| HF-G16 32K | 15.8417 | 2054.4 | 26.26 | 22.936 |
| GPTQ 32K | 19.9200 | 1633.8 | 29.20 | 25.683 |

The long-prompt result is real in this test setup: HF-G16 prefill is 1.28x at 8K and 1.26x at 32K. Energy per output token at 32K is 10.7 percent lower. The decode cost is also real: HF-G16 is 11.8 percent slower at short context, 11.2 percent slower at 8K, and 10.1 percent slower at 32K.

Concurrency does not remove the decode gap in this test setup. At concurrency 8, aggregate output is 166.75 tok/s for HF-G16 versus 213.30 tok/s for GPTQ. Per-request median decode is 21.42 versus 27.99 tok/s. Do not turn the prefill result into a universal serving speed win.
### 8.2 Dense quality battery
The live 329-item battery gives HF-G16 287 passes and shipped GPTQ 283 passes. The exact paired McNemar p is 0.48068, so this is not a statistically significant quality win. The split-level counts are:

| Split | HF-G16 | GPTQ |
|---|---:|---:|
| instruction following | 35/55 | 33/55 |
| knowledge/professional | 40/55 | 40/55 |
| structured output | 55/55 | 55/55 |
| test | 78/82 | 76/82 |
| tune | 79/82 | 79/82 |

The public claim should be parity within the measured battery, not superiority.
### 8.3 Gemma-4-26B-A4B multimodal MoE live serving
The v0.2.0 result is the first live multimodal MoE serving path in this project.

**Which checkpoint was served.** The served checkpoint used raw-text BiasCorr, which failed the 4-check out-of-domain (OOD) chat gate and is held back. No server has yet booted with the chat-calibrated BiasCorr that passed. The live decode speed, PPL, canaries and memory figures below all come from that raw-text BiasCorr checkpoint. They are not measurements of the gate-passing 2.11 MiB chat-calibrated sidecar (section 3.7).

The measured facts from `results/qwen38-27b/quantization-study-v11-202611/exp17d_gemma4_26b_had64_ablation_and_live_vllm.json` are:

* full text plus vision checkpoint loaded;
* projected weight footprint 14.70 GiB with vision in the conversion accounting;
* live process weight footprint measured at 15.11 GiB in the live memory audit;
* BF16 text weight footprint 48.07 GiB plus about 1.07 GiB vision weights;
* single-stream short-context decode 83.21 tok/s, measured on a non-quiet host (GPU1 busy); GPU0 also drove a desktop session, so this is not a quiet-host speed figure;
* KV capacity 109,923 tokens under the tested cache configuration;
* 8/8 canaries passed;
* live vision smoke test passed;
* live WikiText-2 PPL 25.16 versus 25.19 for the same raw-text BiasCorr arm scored offline on the same 2,044 targets.

The test evidence is 8/8 live canaries, 1/1 image canary, the 699-pass Wave9F plugin suite (`results/qwen38-27b/quantization-study-v11-202611/exp17_moe_plugin_pytest_wave9f_gpu0.log`), and the one test added in Wave9H (`results/qwen38-27b/quantization-study-v11-202611/exp17_wave9h_17of17_note.md`), for a 700-test release chain.

The memory claim should be stated carefully. "48 GiB to 14.7 GiB" is the weight-storage comparison. It is not total process RSS and it does not include KV cache, CUDA-graph-like workspace equivalents, image preprocessing, allocator fragmentation, or runtime buffers. The live memory audit is the correct citation when discussing total served footprint.

The speed claim is single-stream decode for the tested prompt and sampling setup, on a non-quiet host (GPU1 busy). It is not a concurrency-8 aggregate claim, and a quiet-host speed has not been measured.
### 8.4 Grouped MoE prefill and quiet-host crossover
v0.2.0 adds a grouped-expert prefill path. The first measurements were made while unrelated work existed on the second GPU, so the quiet-host Gate5 rerun is now the publication-grade speed receipt. That rerun used deterministic algorithms, a 230 W cap, alternating variant order, graph-replay equality checks with zero differing words, and 432 cooled pre-sample temperature readings, all at or below 55 C.

At M 512, the exact packed-expert comparison is:

| Model shape | Pair ms, eager | Grouped ms, eager | Eager speedup | Graph speedup | Weight read pair | Weight read grouped |
|---|---:|---:|---:|---:|---:|---:|
| Gemma-4-26B, E=128, topk=8, H=2816, I=704 | 24.4456 | 7.1666 | 3.411x | 3.517x | 12.1172 GiB | 0.9112 GiB |
| Qwen3.6-35B, E=256, topk=8, H=2048, I=512 | 15.0979 | 4.6204 | 3.268x | 3.460x | 6.4219 GiB | 0.5754 GiB |

Grouped execution wins every tested cell from M 32 onward. At M 16 the crossover is not universal: the Qwen eager grouped runner is 0.903x the pair runner, 0.9166 ms versus 0.8273 ms, although graph replay is slightly favorable at 1.022x. The shipped threshold remains `M >= 16`, but the honest crossover statement is that every tested win is consistent from `M >= 32`, not that M 16 always helps.

The packed 4-bit runner beats the BF16 runner across M 8 to 64: 1.48x to 2.06x in eager mode and 2.18x to 2.73x with graph replay. At M 512 this reverses. BF16 is faster than the packed runner in both shapes: Gemma 3.9708 ms BF16 versus 7.1833 ms packed eager, and Qwen 3.3571 ms BF16 versus 4.5892 ms packed eager. Graph replay does not change the direction. The reason is that the Step A grouped kernel becomes compute-bound at large M. This is exactly the limitation that the next-generation DPAS design is meant to attack; it is not evidence that the current kernel already solves large-batch packed MoE.
### 8.5 Bit-exact compatibility and tests
The v0.1.0 to v0.2.0 compatibility evidence includes:

* kernel A/B bit-identical true;
* greedy token IDs identical across all tested variants;
* logprobs bit-identical;
* 308/308 kernel A/B identical entries;
* inspected zebin hashes identical;
* 699 passed, 0 failed in the Wave9F full suite, plus the final Wave9H Granite admission-spoof regression for a 700-test release chain (the Wave9H 17/17 rerun is author-reported and has no log);
* 11/11 targeted mutations caught in Wave9D, and 8/8 more in Wave9F.

Files: `results/qwen38-27b/quantization-study-v11-202611/exp17_v020_vs_v010_compat_gpu0.json`, `results/qwen38-27b/quantization-study-v11-202611/exp17_moe_plugin_pytest_wave9f_gpu0.log`, `results/qwen38-27b/quantization-study-v11-202611/exp17_wave9d_mutation.log`, `results/qwen38-27b/quantization-study-v11-202611/exp17_wave9f_mutation.log`, `results/qwen38-27b/quantization-study-v11-202611/exp17_wave9h_17of17_note.md`.

This is stronger than a smoke test. It is still scoped to tested architectures, kernels, and admission paths. It is not a proof that every future model configuration is safe.

## 9. Methodology, guardrails, and bugs caught by adversarial review
### 9.1 Exact gates before interpretation
Every serious causal experiment was gated by a sequence of exactness checks:

1. repeat streams must be exactly equal;
2. future perturbation must not affect earlier positions;
3. negative perturbation must affect later positions;
4. patched zero-correction paths must exactly reproduce stock output;
5. unquantized hooks must exactly reproduce stock SDPA or model forward output;
6. BF16 perplexity must be plausible before quantization is interpreted;
7. checkpoint endpoint tensors must be inventoried rather than assumed.

These gates are cheap compared with the cost of publishing a wrong quantization conclusion.
### 9.2 The non-causal oracle
The first standalone layer test script produced impossible WikiText-2 perplexity around 2.85. That was label leakage through future attention, not a quantization result. After causal masks were fixed, Qwen3.8-27B BF16 had WT2 PPL 7.79822 and code PPL 2.37436.

This bug changed the project. Every later experiment treats causality as a binary gate, not as a best-effort property.
### 9.3 Live `attn_implementation` reconciliation
A live server once appeared to contradict offline measurements. The cause was operational rather than numerical: the server had been launched with an attention implementation setting that did not match the offline scoring script. A live vLLM WT2 PPL of 7.948867930201732 is on the causal 7.8 to 8.0 scale. The W4A8 live-shape offline run gives 7.94875, closely matching it. This reconciled the apparent discrepancy and invalidated the older non-causal 2.85 artifact scale as a comparison target.
### 9.4 VRAM cap after the 12:54 OOM reboot
One experiment sequence caused an out-of-memory reboot. After that, the measurement discipline became stricter: one inference process per card, process memory fraction caps, explicit memory audits, and no quiet-host performance claims while unrelated heavy jobs are active. The exact timestamp is less important than the operational lesson: correctness work can still destroy a host if memory accounting is sloppy.
### 9.5 OLMoE FP32 accumulation fix
A per-expert BF16 `index_add` path did not reproduce stock `grouped_mm`. Slot-wise FP32 accumulation reproduced it exactly with max absolute difference 0.0 on the diagnostic layers. This mattered because expert routing accumulates small differences. If the identity path is not exact, later quantization deltas cannot be interpreted.
### 9.6 Left-padding BiasCorr bug
A calibration bug involving left-padding produced a zero score on 24 items in one diagnostic chain. The correction vector was being estimated from positions that did not correspond to real activations. The fix was not statistical; it was structural. Use the true causal token positions and exclude padding from activation means.
### 9.7 Heretic Qwen3.6 GPTQ forensic catch
The shipped Qwen3.6 GPTQ checkpoint had an abliteration-like direction modification. A naive comparison therefore mixed two effects: quantization error and a deliberate behavioral direction removal. With the direction restored post hoc, HF data-free versus shipped GPTQ improves by -0.032 KL rather than the uncorrected -0.050. WikiText-2 alone is not separable.

This is a reminder that comparing checkpoints is not the same as comparing formats when the checkpoints differ in non-quantization ways.
### 9.8 Reviewer ledger
The review loop used multiple AI models as adversarial reviewers. It was not human academic peer review.

| Reviewer | Role in this project |
|---|---|
| Opus 5.5 | publication and code sign-off reviews; rejected broad claims until exact caveats were written |
| GPT-6.2 Astra | architecture, adversarial code review, v0.2.0 review, mutation and edge-case review |
| Sonnet 5.5 | additional review passes and consistency checks |
| SWE-2.1 | implementation, audit, receipt extraction, and writing assistance |
| External GitHub/Hugging Face commenters | public artifact review and engine-integration discussion, separate from internal AI review |

The important point is not the brand list. The important point is that reviewers were asked to break the claim, not to approve a narrative.

## 10. Ecosystem adoption and exact reproduction commands
### 10.1 vLLM-XPU
The dense and MoE serving implementation lives in the public vLLM-XPU plugin work. The reproducible claims are scoped to the released configuration and the exact checkpoints in the receipts. Do not infer that arbitrary models are admitted. The plugin fails closed when a layout or architecture is unsupported.

Dense conversion and serving commands from the public plugin README are:

```bash
pip install --no-deps --no-build-isolation -e .
python3 convert_hf_int4_g16.py "$BF16_QWEN38" "$OUT_QWEN38_HFG16" \
  --biascorr --cal-mode mixed_train --biascorr-tokens 1024 --max-shard-gib 3.0
HF_INT4_G16_MODE=auto vllm serve "$OUT_QWEN38_HFG16" \
  --served-model-name default --host 127.0.0.1 --port 8091 \
  --max-model-len 36864 --max-num-seqs 96 \
  --gpu-memory-utilization 0.90 --trust-remote-code
python3 -m pytest tests/ -q
```

For MoE conversion the same positional converter is used, with `--biascorr-scope shared_only` for the measured shared-only policy. Use `--cal-chat-template` for chat-template calibration when the target is chat-tuned, then run the out-of-domain gate before shipping. That flag is unmeasured: the chat-calibrated sidecar that passed in Exp17D was not produced by it. The exact operational requirements are: use the plugin version named in the receipt, use the exact converted checkpoint artifact, use one process per GPU, use model-card sampling for public speed claims, and report TTFT, prefill, decode, and energy separately.
### 10.2 TensorFold
TensorFold received a fused FP32 per-channel correction epilogue in the Wave 4 patch lineage. That work exposed a real design conflict: the correction improves measured output for some models but is not target-preserving. An inference engine that promises bit-exact reproduction of a checkpoint cannot silently change the output distribution. Opt-in plumbing is therefore the honest integration route.
### 10.3 Apple mlx-lm
The Exp18 blueprint estimates about 20 lines for a Python-side mlx-lm integration after `load_model`. The critical detail is timing. Strict loading fails if unknown bias keys are inserted too early. Load the model first, attach correction vectors, then force evaluation with `mx.eval`.

The blueprint is an implementation plan, not a merged upstream feature.
### 10.4 llama.cpp GGUF
The llama.cpp inspection found generic bias tensor loading and graph plumbing in many dense and MoE paths. The missing work is per-architecture wiring. The `--lora` path cannot carry this correction because LoRA is an input-dependent low-rank weight update, while BiasCorr is a static output bias.

The recommended first phase is embedding correction vectors into GGUF and wiring qwen3/qwen2 dense paths first. Estimated implementation surface is 0 to 40 lines depending on whether the architecture already exposes a bias hook. A runtime `--biascorr-file` option is phase two.

The hybrid linear-attention artifact warning is essential: Qwen3.8 uses keys such as `linear_attn.in_proj_qkv`, `in_proj_a`, and `in_proj_z`. Code that assumes simple `self_attn.q/k/v` names will wire the wrong tensors.
### 10.5 Reproduction index
All paths below are inside this bundle. `results/qwen38-27b/quantization-study-v11-202611/RECEIPTS_MANIFEST.json` records a sha256 for every receipt. `python3 VERIFY_V11.py` rechecks the headline numbers from these files, and `python3 charts/render_all_v11_charts.py` redraws the six charts from them. The conversion and serving runs themselves need the plugin, the model weights and a B70. The V10 dense reproduction chain is a separate published release and is not copied into this bundle. The two video render scripts also need model files that are not included.

Weight formats and families:

* `results/qwen38-27b/quantization-study-v11-202611/exp13a_causal_64l_v2.json` for the original causal format comparison;
* `results/qwen38-27b/quantization-study-v11-202611/exp13b_qwen25_dense.json`, `results/qwen38-27b/quantization-study-v11-202611/exp13b_smollm2_dense.json` and `results/qwen38-27b/quantization-study-v11-202611/exp13b_cross_arch.json` for Qwen2.5-1.5B and SmolLM2-1.7B;
* `results/qwen38-27b/quantization-study-v11-202611/exp13b_qwen38_kv_causal_v2.json` for the Qwen3.8 K-only versus V-only arms;
* `results/qwen38-27b/quantization-study-v11-202611/exp15b_multifamily_scores.json` for the compact multi-family task and KL matrix;
* `results/qwen38-27b/quantization-study-v11-202611/exp15b_llama32_3b.json`, `results/qwen38-27b/quantization-study-v11-202611/exp15b_olmo2_7b.json`, `results/qwen38-27b/quantization-study-v11-202611/exp15b_olmoe_1b7b.json` and `results/qwen38-27b/quantization-study-v11-202611/exp15b_phi4_mini.json` for the per-family arms;
* `results/qwen38-27b/quantization-study-v11-202611/exp15a_tfq3_vs_hfg16_64l.json`, `results/qwen38-27b/quantization-study-v11-202611/exp15c_hfg16_refine_64l.json` and `results/qwen38-27b/quantization-study-v11-202611/exp16_wiener_affinecorr.json` for the trellis, refinement and correction-variant studies;
* `results/qwen38-27b/quantization-study-v11-202611/exp17a_modern3_summary.json`, `results/qwen38-27b/quantization-study-v11-202611/exp17a_granite4_htiny.json`, `results/qwen38-27b/quantization-study-v11-202611/exp17a_gemma4_e4b.json` and `results/qwen38-27b/quantization-study-v11-202611/exp17a_qwen3_8b.json` for Granite, Gemma-E4B and Qwen3-8B.

MoE, ultra-low-bit, KV and Bonsai:

* `results/qwen38-27b/quantization-study-v11-202611/exp17b_gemma4_26b_a4b.json` and `results/qwen38-27b/quantization-study-v11-202611/exp17b_PROCESS.md` for Gemma-4-26B causal KL and sensitivity evidence;
* `results/qwen38-27b/quantization-study-v11-202611/exp17c_qwen36_35b_a3b.json` and `results/qwen38-27b/quantization-study-v11-202611/exp17c_PROCESS.md` for Qwen3.6 MoE evidence;
* `results/qwen38-27b/quantization-study-v11-202611/exp17d_gemma4_26b_had64_ablation_and_live_vllm.json` for Had64, the gate verdicts and live Gemma evidence (the live server used the raw-text BiasCorr checkpoint, which failed the 4-check gate and is held back);
* `results/qwen38-27b/quantization-study-v11-202611/exp18_ultralowbit_64l.json` for ultra-low-bit KL;
* `results/qwen38-27b/quantization-study-v11-202611/exp18_ultralowbit_tasks160_had_int2.json` and `results/qwen38-27b/quantization-study-v11-202611/exp18_ultralowbit_tasks160.json` for the 2-bit and 3-bit task batteries;
* `results/qwen38-27b/quantization-study-v11-202611/exp19a_turboquant_kv_qwen38.json` and `results/qwen38-27b/quantization-study-v11-202611/exp19_PROCESS.md` for KV compression;
* `results/qwen38-27b/quantization-study-v11-202611/exp19b_ternary_bonsai2_27b.json` and `results/qwen38-27b/quantization-study-v11-202611/exp19b_posthoc_ptq_sparsity_control.json` for Bonsai;
* `results/qwen38-27b/quantization-study-v11-202611/exp20_confirmations.json` for Spanish transfer, long-context scaling and the 3-bit cap audit.

Serving, tests and sign-off:

* `results/qwen38-27b/quantization-study-v11-202611/exp14c_vllm_329_battery.json` and `results/qwen38-27b/quantization-study-v11-202611/exp14d_vllm_speed_ladder.json` for the dense quality battery and speed ladder;
* `results/qwen38-27b/quantization-study-v11-202611/gate5_tfv5_v1_depth_gpu0.json` for quiet-host TensorFold v1 overhead;
* `results/qwen38-27b/quantization-study-v11-202611/gate5_tfv5_v1_tp2_parity.json` for one-card and `--tp 2` drafted-versus-serial exactness;
* `results/qwen38-27b/quantization-study-v11-202611/gate5_v020_quiethost_moe_crossover.json` for quiet-host MoE kernel timing;
* `results/qwen38-27b/quantization-study-v11-202611/exp17_moe_grouped_prefill_bench_gpu0.json` for the earlier busy-host grouped prefill run, kept for history and not used for headline numbers;
* `results/qwen38-27b/quantization-study-v11-202611/exp17_v020_vs_v010_compat_gpu0.json`, `results/qwen38-27b/quantization-study-v11-202611/exp17_moe_plugin_pytest_wave9f_gpu0.log`, `results/qwen38-27b/quantization-study-v11-202611/exp17_wave9d_mutation.log`, `results/qwen38-27b/quantization-study-v11-202611/exp17_wave9f_mutation.log` and `results/qwen38-27b/quantization-study-v11-202611/exp17_wave9h_17of17_note.md` for compatibility and tests;
* `results/qwen38-27b/quantization-study-v11-202611/opus55_wave9g_v020_final_signoff.md`, `results/qwen38-27b/quantization-study-v11-202611/opus55_wave9h_v020_final_signoff.md` and `results/qwen38-27b/quantization-study-v11-202611/astra_wave9h_v020_final_signoff.md` for the v0.2.0 reviews;
* `results/qwen38-27b/quantization-study-v11-202611/rotkv_cpu_pytest_30of30.log` for the CPU-only prototype tests;
* `results/qwen38-27b/quantization-study-v11-202611/exp18_llamacpp_mlx_loader_blueprint.md` for loader integration details;
* `results/qwen38-27b/quantization-study-v11-202611/COMPLETE_RESEARCH_CHRONICLE_V2.md` for the research log. It is a snapshot taken before Exp18 and the grouped prefill work finished, so it still lists them and Exp17C as "In flight". It is kept unchanged so its hash stays valid.

## 11. Final confirmation results
Exp20 and quiet-host Gate5 close the remaining measurement gaps. No unresolved experiment placeholders remain.

### 11.1 Exp20 Spanish zero-shot transfer
The Spanish evaluation used disjoint modern Spanish Wikipedia text as the primary domain and Don Quixote as a secondary, likely memorized control domain. Measured 6-gram overlap with both v1 calibration sets is 0/0 for each Spanish corpus. The same zero-overlap result holds for every English, code, and Spanish evaluation group used in Exp20. The pass count is 4/4: every sidecar-and-domain pair passes the positive, negative-control, and confidence-interval checks.

| Spanish domain | Sidecar | Base KL | Corrected KL | Relative KL change | 95 percent block CI | Sign-flipped control KL |
|---|---|---:|---:|---:|---|---:|
| modern Spanish Wikipedia | MLX4 v1 | 0.03309745 | 0.02745158 | -17.1 percent | [-21.5, -12.8] | 0.05180358 |
| modern Spanish Wikipedia | HF-G16 v1 | 0.02023131 | 0.01639186 | -19.0 percent | [-23.2, -14.8] | 0.03666812 |
| Don Quixote | MLX4 v1 | 0.26150012 | 0.22824557 | -12.7 percent | [-17.1, -8.0] | 0.42013568 |
| Don Quixote | HF-G16 v1 | 0.20955774 | 0.16219720 | -22.6 percent | [-28.7, -16.3] | 0.35876014 |

The sign-flipped `-b` controls are strictly worse in every row, 56.5 to 81.2 percent above their corresponding uncorrected KL. That supports a real directional correction rather than an arbitrary vector helping by chance.

The language-quality claim must stay narrow. Next-token NLL improves significantly in only one of the four cells: HF-G16 on Don Quixote, delta -0.06437 with CI [-0.08607, -0.04322]. The other three NLL intervals cross zero. The supported claim is therefore "the corrected quantized model stays closer to BF16 on Spanish", not "it becomes a better Spanish predictor".

### 11.2 Exp20 long-context scaling
Exp20 rescored the published WT2 and code domains at 512, 2,048, and 4,096 tokens. The causal-prefix floor for 512-to-2048 is exactly 0.0 mean and maximum KL for both domains, and 38 shared arrays are bit-identical across independent processes with maximum absolute difference 0.0. The 4,096-window numbers are not bit-identical to shorter windows and are interpreted as numerical measurements rather than exact replay.

Every sidecar-and-domain pair still has lower KL at every tested length. Using the published 512-token arm as the short-context anchor:

| Domain and sidecar | 512 tokens | 2,048 tokens | 4,096 tokens | Same-token scaling verdict |
|---|---:|---:|---:|---|
| WT2, MLX4 v1 | -23.9 percent | -17.0 percent | -18.4 percent | relative effect fades |
| WT2, HF-G16 v1 | -15.3 percent | -16.5 percent | -20.6 percent | no detectable change |
| code, MLX4 v1 | -14.2 percent | -14.5 percent | -16.0 percent | no detectable change |
| code, HF-G16 v1 | -13.5 percent | -15.2 percent | -19.1 percent | no detectable change |

At 2,048 and 4,096 tokens all four reductions sit in the narrow -14.5 to -20.6 percent range. The negative controls remain worse at every length. The MLX4 prose effect does soften with context: its percentage-point reduction is 6.93 points smaller at 2,048 tokens, CI [0.84, 12.66], and 5.79 points smaller at 4,096 tokens, CI [1.98, 9.46]. The other three sidecar-and-domain pairs have scaling intervals that include zero, so the correct statement is no detectable change, not proof of exact invariance.

### 11.3 Exp20 3-bit 1,024-token cap audit
The Exp20 GSM8K rerun uses `had_int3_g16` and BC_open with a 1,024-token generation cap. Exact results:

| Arm | Correct at 1,024 cap | Truncated | Mean new tokens |
|---|---:|---:|---:|
| data-free | 58/60 | 0 | 334.8 |
| BC_open | 57/60 | 1 | 328.5 |

The paired table at the 1,024-token cap is 0 fixed, 1 broken, exact p 1.0. The same result holds on the 59 items that finish in both arms. When the same new run is artificially cut at 384 tokens, the apparent improvement returns: 41/60 to 50/60, 9 fixed, 0 broken, exact p 0.00390625. This cleanly identifies the earlier 41 to 48 reading as a cap-truncation artifact rather than a true 3-bit reasoning rescue.

The 2-bit rescue remains real. In addition to the primary 31/160 to 93/160 result, the 17 GSM8K items that finish in both 2-bit arms without truncation score 6/17 uncorrected and 15/17 corrected, 9 fixed, 0 broken, exact p 0.00390625.

### 11.4 Quiet-host Gate5 TensorFold v1 overhead
The quiet-host TensorFold measurement uses binary sha256 `75c15aba7afd97fc68c2ad1a33a604955eed53c4f5c087b68f45bb03bc7ff176`, the exact published v1 sidecar sha256 `f5fce21b31f477229cab3e62cb42d67094983fd0185e7fc6e363df1ad38ed932`, 230 W and 2,800 MHz card limits, n=3 interleaved ABABAB samples, model-card sampling, and 512 output tokens. All 30 measured cells start at or below 55 C.

| Context | Unset TTFT s | v1 TTFT s | TTFT overhead | Draft-round time overhead |
|---|---:|---:|---:|---:|
| 0 | 0.136 | 0.136 | +0.00 percent | +0.22 percent |
| 8K | 4.710 | 4.724 | +0.30 percent | +0.18 percent |
| 32K | 21.844 | 21.891 | +0.22 percent | +0.27 percent |
| 64K | 50.996 | 51.093 | +0.19 percent | +0.21 percent |
| 128K | 130.398 | 130.584 | +0.14 percent | +0.12 percent |

The sidecar therefore has no measurable TTFT cost beyond +0.30 percent and no measurable per-round kernel-cost increase beyond +0.27 percent in this test setup. Decode tok/s should not be quoted as a speedup or slowdown here: its median ratio spans -2.93 to +7.45 percent because the corrected model samples different text, which changes draft acceptance and tokens per round. The stable kernel-cost proxy is time per draft round, not decode rate alone.

### 11.5 Quiet-host Gate5 `--tp 2` parity
The exactness suite passes drafted-equals-serial on 11/11 prompts in every tested configuration: GPU0 unset, GPU0 v1, `--tp 2` unset, and `--tp 2` v1. Drafting is genuinely active, with hundreds of accepted drafted tokens in the drafted runs and zero offered in the serial control. With v1, the sidecar load log reports 496/496 matched tensors against 496 expected tensors, and the eight short-answer checks pass 8/8 on both GPU0 and `--tp 2`.

One-card and two-card outputs are not bit-identical to each other in every case. Comparing the same arm across topologies, 2/22 output rows differ with the sidecar unset and 4/22 differ with v1. The supported claim is exact drafted-versus-serial equality within each topology, not cross-topology output identity.

### 11.6 Quiet-host v0.2.0 MoE crossover
Gate5 provides the final quiet-host MoE kernel evidence. Across M 8 to 64, packed 4-bit execution beats BF16 by 1.48x to 2.06x eager and 2.18x to 2.73x with graph replay. Grouped execution wins every tested cell from M 32 onward. At M 512 the grouped path is 3.411x to 3.517x faster than pair execution for the Gemma shape and 3.268x to 3.460x faster for the Qwen shape, depending on eager or graph replay.

The negative boundary is equally important. At M 16 the Qwen eager grouped runner dips to 0.903x the pair runner, 0.9166 ms versus 0.8273 ms. At M 512 BF16 still beats the packed runner even though grouped beats pair within the packed implementation. This is a Step A compute limitation, not a measurement error. The next-generation DPAS design is a proposed fix and must not be quoted as a measured v0.2.0 result.

### 11.7 Final v0.2.0 sign-off and next-generation boundary
Both Claude Opus 5.5 and GPT-6.2 Astra signed off `vllm-hf-int4-g16-xpu` v0.2.0 after the final Granite admission-spoof fix. Opus states `APPROVED TO TAG v0.2.0`; Astra states `APPROVED FOR v0.2.0 RELEASE (WITH DOCUMENTED SCOPE)`. The Wave9G review clears the uniform-Had64 `hf_int4_g16/2` format contract to freeze. The Wave9F suite report is 699 passed and 0 failed; Wave9H adds the final admission regression test, bringing the release chain to 700 tests. Both final reviews are diff-only and explicitly did not rerun GPU or Docker evidence. The 700 figure is 699 passes in one logged run plus 1 added test; the author reported the 17/17 rerun of that test file, no log of it exists, and neither reviewer reran it (`results/qwen38-27b/quantization-study-v11-202611/exp17_wave9h_17of17_note.md`). The freeze clearance is in `results/qwen38-27b/quantization-study-v11-202611/opus55_wave9g_v020_final_signoff.md`; the two final approvals are in `results/qwen38-27b/quantization-study-v11-202611/opus55_wave9h_v020_final_signoff.md` and `results/qwen38-27b/quantization-study-v11-202611/astra_wave9h_v020_final_signoff.md`.

Separate Step B DPAS MoE and packed 3-bit rotated KV CPU prototypes are not shipped in this bundle. Their CPU-only suite passes 30/30 tests (`results/qwen38-27b/quantization-study-v11-202611/rotkv_cpu_pytest_30of30.log`), covering bit packing, parallel bucket scan parity, DPAS unpack exactness, and rotated-KV attention reference math. These are algorithmic CPU references only. They are not GPU timing, GPU quality, or v0.2.0 release evidence.

## 12. Bottom line
The strongest positive result is not a single number. It is a chain:

1. use hardware constraints to choose a scalar integer format;
2. rotate weights and KV to make scalar rounding less adversarial;
3. correct only the persistent mean error, and only when a strict gate says it helps;
4. verify causality and identity exactly before interpreting KL;
5. accept negative results as part of the design space.

The strongest negative result is equally clear: compression tricks are not universal. Trellis wins rate-distortion but is not automatically the serving answer. BiasCorr rescues one 2-bit setting but is useless or harmful elsewhere. QJL is mathematically sensible but loses at the tested 3 to 4 bit KV budgets. Dense HF-G16 prefill improves while decode regresses. The apparent 3-bit GSM8K rescue was a generation-cap artifact, while the 2-bit rescue remains statistically strong. A useful public report has to keep those facts adjacent, not bury them under the headline.

