# Complete Research Chronicle V2

## Scope, evidence rule, and date

This is the second end-to-end chronicle of the quantization study. It covers the work from the first broken attention oracle through Wave 9C review, the post-publication multi-family experiments, and the current Wave 10 integration and ecosystem work. It updates, but does not replace, `COMPLETE_RESEARCH_CHRONICLE.md`.

The compile time for this document is 2026-11-25, after the Exp17B Gemma-4-26B-A4B receipt was complete and while the Exp17A Gemma-4-E4B task arm was still running. I wrote it on CPU only. I did not run a GPU job, start Docker, or modify any source artifact. Paths in this document are study-relative paths, not host paths. GitHub and Hugging Face names are public artifact names, not local checkout paths.

The evidence rule is simple: measurements come from receipts, review claims come from signed review files, and integration claims come from the public repository state or release bundles. When the task brief supplied a number that I could not verify as a standalone receipt, I label it as an interpretation or as operator-supplied context instead of silently upgrading it into a measurement.

The review loop used multiple AI agents as adversarial reviewers. Those rounds were automated adversarial reviews, not human academic peer review. External humans did participate later on the public TensorFold pull request, and their comments are separated from the internal AI review ledger.

## 0. Timeline in one page

| Phase | What happened | Durable output |
|---|---|---|
| Wave 1 | The first standalone layer harness was found to be non-causal. The impossible WikiText-2 perplexity scale around `2.85` was not a quantization miracle; it was label leakage through future attention. The harness was rebuilt with causal masks, exact BF16 identity, perturbation causality, functional probes, and realistic BF16 perplexity prechecks. | Corrected causal oracle. `COMPLETE_RESEARCH_CHRONICLE.md`, `opus55_wave1_audit.md` |
| Wave 2 | The Intel kernel and serving design was written before the converter existed. Hadamard rotation, native Xe2 packed layouts, progressive G16 scales, W4A16/W4A8 dispatch, BiasCorr plumbing, and fused-checkpoint loading were specified as one system. | `vllm_plugin_design_astra.md` |
| Wave 3 | First real code review was intentionally adversarial. It rejected broad sign-off even though one dense TP1 WT2 result looked good, then forced independent oracles, coverage tests, alignment checks, rounding fixes, TP proofs, and scheduler honesty. | `astra_wave3_code_review.md`, `astra_wave3_rereview.md` |
| Wave 4 | TensorFold received a fused FP32 per-channel correction epilogue. The work exposed a hard upstream design question: an opt-in correction improves model output but is not target-preserving. | TensorFold patch lineage and review discussion |
| Wave 5 | The vLLM plugin was hardened into an installable XPU-only package rather than a demo script. Dense linears, Had128NoPerm, fail-closed layout validation, strict or opt-in mixed precision, and separate Triton/native paths were documented. | `astra_wave5a_code_signoff.md` |
| Wave 6 | Code sign-off remained narrow. Opus accepted the implementation only inside the tested dense scope and kept strict serving, replication, and speculative interactions out of the claim. | `opus55_wave6a_code_signoff.md` |
| Wave 7 | Publication review caught arm and sidecar mismatches that the raw tables did not show. The train-split sidecar was distinguished from the predecessor WT2-tail predecessor used in much live evidence. Final publication was approved only after those caveats were written into the public study. | `opus55_wave7b_publication_signoff.md`, `opus55_wave7d_final_signoff.md`, `publish-bundle/PUBLIC_STUDY_V10.md` |
| Wave 8 | Exp17A and Exp17B moved the question away from Qwen-only dense models: Granite-4.0-h-tiny, Gemma-4-E4B-it, and Gemma-4-26B-A4B-it added hybrid attention, large expert banks, shared expert paths, K/V sharing, tail widths, and grouped-matmul rounding effects. | `results/exp17a_granite4_htiny.json`, `results/exp17a_gemma4_e4b.json`, `results/exp17b_gemma4_26b_a4b.json` |
| Wave 9 | vLLM-HF-INT4-G16 v0.2.0 added MoE admission, Gemma and Granite mappings, and Had64 support. Reviews found that the implementation had outrun the evidence: uniform Had64 had not yet been quality-gated against the Exp17B mixed-tail receipt, kernel rewrites lacked bit-exact A/B proof, and several admission edge cases were unsafe. | `opus55_wave9c_v020_review.md`, `astra_wave9c_v020_review.md` |
| Wave 10 | The current integration phase is not a single receipt-labeled wave in the study. I use Wave 10 here for the post-v0.2.0 work: fixing the MoE/Had64 evidence gap, measuring grouped-expert prefill, finishing Exp17A, starting Exp17C, and mapping BiasCorr into llama.cpp, MLX, and ultra-low-bit research. | `exp18_llamacpp_mlx_loader_blueprint.md`, public PR discussion on TensorFold |

## 1. How we found the formats: silicon physics first

### 1.1 The hardware dictated the search space

The target card was the Intel Arc Pro B70, Xe2 architecture, BMG-G31 chip. Its relevant facts were not vague preferences. They were constraints:

- 32 GB GDDR6 per card, with about 608 GB/s pin bandwidth. The best measured large-tile stream reached about 540.8 GB/s.
- 32 Xe-cores and 256 XMX units per card.
- No native FP4 or FP8 DPAS path on Xe2.
- BF16 and FP16 matrix math use DPAS K16. INT8 uses K32 and has about twice the peak matrix throughput.
- Decode is memory-bandwidth dominated. Weights dominate memory traffic. A format that cannot be read efficiently is unattractive even if its offline KL is excellent.

Those facts explain why the study converged on integer 4-bit weights rather than native FP4 or FP8. If the hardware cannot multiply FP4 or FP8 natively, a format that stores FP4-like values still has to be dequantized or emulated before the matrix unit can use it. The useful question became: how few bytes can be moved per weight while keeping the dequantize and multiply path simple enough for XMX?

`Had128NoPerm + HF-INT4-G16` emerged from that question. It stores 4-bit integer codes in groups of 16 with progressive scales and uses an online FWHT rotation. The net weight payload is about `4.25 bpw`. The rotation is Hadamard-based, but the serving format does not need a scatter permutation in memory. "NoPerm" mattered: random-looking data movement is expensive on a bandwidth-bound part, while the fixed Sylvester rotation can be regenerated and applied in the arithmetic pipeline.

The Intel-native packing also mattered. The two hand-check native words were `0x73625140` and `0xfe10`; Wave 3 review verified those declared words by CPU round trip and fixed integer dot products. That low-level check was not ceremonial. A nibble order or `(q,u)` pair mistake can look fine in a smoke test while silently destroying the least significant bits of every projection.

### 1.2 Why group size 16 and progressive scales won

The practical competitors were:

| Format | Approximate weight bits | Data needed | Main weakness |
|---|---:|---|---|
| `HF-INT4-G16` / Had128NoPerm | `4.25 bpw` | Usually data-free or only BiasCorr calibration | More scaffolding than plain RTN, but shaped for Xe2 |
| `MLX-asym-G64` | `4.50 bpw` | No calibration | Coarser error control on many 2026 architectures |
| `SymINT4-G128` data-free | `4.25 bpw` equivalent class | No calibration | Large groups lose too much local amplitude information |
| Native NVFP4/MXFP8 checkpoints | model dependent | trained or released by model vendor | not reproducible as a generic BF16 conversion on Xe2 |
| Trellis-coded TFQ3/TFQ4 | about `4.001 bpw` for the tested TFQ3 configuration | codebook/calibration machinery | best rate-distortion, but serving integration is a separate problem |

The rate-distortion references came before the serving arguments. The 4-bit Shannon ceiling was `24.082 dB`; Lloyd-Max16 measured `20.22 dB`, within `0.004 dB` of prediction; the E8 lattice reference measured `21.02 dB`; EXL3-4.00 measured `22.43 dB` mean weight SQNR across the tested eight tensors. Shrinking group size from G32 to G16 had a predicted `+0.97 dB` gain and a measured `+1.03 dB` gain on real weights. Those are weight-domain references, not task accuracy, but they kept the format search honest.

The group-16 scale layout was not chosen because 16 looks special. It is the smallest scale quantum that still maps cleanly onto the XMX tile shape while keeping enough local resolution to control outliers. The progressive scale tree stores coarse amplitude information once and local corrections compactly. The Hadamard rotation spreads isolated outliers across a block, so scalar integer rounding sees a less adversarial distribution.

Across the completed family set, HF-INT4-G16 beat MLX-asym-G64 in the primary causal KL comparisons at lower weight bits, and it beat data-free SymINT4-G128 by a wider margin. The original public study established `7/7` family wins. Waves 8 and 9 extended the same pattern to Granite-4.0-h-tiny, Gemma-4-E4B-it, and Gemma-4-26B-A4B-it. Counting the original seven plus those three gives the `10/10` family-level sweep cited in this chronicle. That is an aggregate across Exp13B, Exp15B, Exp17A, and Exp17B receipts, not a single monolithic harness run.

The qualifier is that this was a KL sweep of the tested conversion arms, not a proof that every downstream task score must improve. Exp15B explicitly refuted that stronger inference.

### 1.3 Why trellis coding won the rate-distortion argument and still did not become the serving answer

Exp15A compared trellis-coded quantization with scalar HF-INT4-G16 on the 400 common decoder matrices of Qwen3.8-27B. The trellis result was real: `TFQ3-4b` at about `4.001 bpw` beat scalar G16 per bit on matrix-level error and KL-oriented measures. We did not ignore that result. Exp15C tested whether zero-extra-bit or modest mixed-width refinements could close the practical gap while staying near the G16 serving format.

They could not. The conclusion was uncomfortable but clear: rate-distortion math favors trellis coding when the comparison is purely about representing the weight matrices. The study stopped trying to disprove that. The reason HF-INT4-G16 remained the Xe2 serving candidate is different. It has a scalar integer data path that can be implemented directly in DPAS-oriented kernels, loaded compactly, checked bitwise, and corrected with a small sidecar. A better compression representation is only useful after an engine can serve it at competitive latency and memory cost.

### 1.4 Gemma and Nemotron forced the Had64 issue

The published v0.1.0 plugin was built around Had128. Gemma-4-26B-A4B exposed dimensions that do not divide cleanly by 128. In the Exp17B receipt, `mlp.down_proj` has `K=2112` and `experts.down_proj` has `K=704`; both are `64 mod 128`. The experiment therefore used Had128 blocks plus one Had64 tail block for those matrices. That mixed layout was a measurement artifact, not a shipped frozen format.

NemotronH added a harder shape, `K=1856`, and also has ungated `relu2` behavior that is outside the normal gated MLP assumptions. The v0.2.0 implementation chose uniform Had64 (`hf_int4_g16/2`) partly to avoid tail complexity. Wave 9C review correctly objected that this engineering choice had not yet been proven equivalent in the exact quality gate that motivated it. Uniform Had64 may still be right, but the required evidence is a rerun of the Exp17B KL gate using the actual v0.2.0 converter and layout. Until then, v0.2.0 is not publication-grade for the Gemma tail claim.

### 1.5 Inkling-2-Mini is a boundary case, not a failure to convert

Inkling-2-Mini stayed on its official native NVFP4/MXFP8 checkpoint. No BF16 expert source was released for the relevant expert path, so the study could not honestly convert it from BF16 and compare like with like. DNR-035, DNR-040, and DNR-041 record that decision. The useful lesson is negative in a good way: a conversion study must not pretend that a vendor-native low-bit checkpoint is evidence for or against a BF16 conversion format when the BF16 source does not exist.

## 2. How we found BiasCorr, and how ten families refuted both extremes

### 2.1 The original idea

BiasCorr is a first-order correction for the mean activation error introduced by weight quantization. For a linear layer with BF16 weight `W_bf16`, quantized-then-dequantized weight `W_quant`, and calibration activations `x`, the correction vector is

`delta_b = (W_bf16 - W_quant) @ E[x]`

The implementation stores this vector in a small sidecar and adds it in the output epilogue. The key numerical detail is that the addition happens in FP32 before the final BF16 cast. Adding after the BF16 output would often round the benefit away or introduce a different error term.

Position 0 is excluded from calibration. The reason is empirical and architectural, not mystical. Early dense and hybrid models often use the first position as an attention-sink-like state with activation statistics unlike the rest of the stream. Including it made the mean less representative of normal token positions and sometimes worsened the correction.

### 2.2 The first hypothesis died: BiasCorr is not universal

The first attractive story was "quantization bias is systematic, so this correction should help everywhere." That story is false.

On Qwen3.8-27B the shipped GPTQ arm was calibrated rather than data-free, and BiasCorr made its WikiText-2 KL worse by about `5.7%` (`0.04941556` to `0.05221039`) even though its code KL improved. Small dense non-QK-Norm models were often worse:

- SmolLM2-1.7B: about `+18%` to `+51%` worse KL depending on format and domain.
- Llama-3.2-3B: about `+17%` to `+43%`.
- Phi-4-mini: about `+3%` to `+24%`.
- OLMo-2-7B: about `+92%` to `+211%` in the worst cases.

The controls were not decorative. In the earlier Qwen rigor chain, the sign-flipped `-b` correction worsened KL by `+84.00%` on WikiText-2 and `+68.66%` on code. Matched random-direction controls also worsened rather than helped. The zero-bias path proved bit-neutral load/add only; it did not prove channel order. Those distinctions are why later reviews trusted the positive gate more than a single attractive mean.

That is why `export_biascorr.py` now has the mandatory three-check automatic KL gate. The gate requires improvement on both domains, requires the negative `-b` control to fail or at least not reproduce the benefit, and uses block-bootstrap confidence intervals rather than a single mean. A sidecar is not presumed innocent because it is small.

### 2.3 The second hypothesis also died: it is not only a Qwen3.8 trick

After the small dense failures, the opposite story became attractive: perhaps BiasCorr only helps Qwen3.8-like models or very large models. That story is false too.

Gemma-4-E4B-it, a 2026 model, passed the three-check gate for all four tested formats in Part A. Combined KL moved as follows:

| Format arm | Data-free combined KL | BiasCorr combined KL | Direction |
|---|---:|---:|---|
| HF-INT4-G16 W4A16 | `0.28312991` | `0.22304306` | about `21%` lower |
| MLX-asym-G64 | `0.50524378` | `0.30360750` | about `40%` lower |
| NVFP4 | `0.25565009` | `0.19645004` | about `23%` lower |
| SymINT4-G128 | `1.23578554` | `0.49314147` | about `60%` lower |

Receipt: `results/exp17a_gemma4_e4b.json`. This file is still marked `partB_in_progress` because task scoring is incomplete, but Part A causal KL evidence is present.

Gemma-4-26B-A4B-it repeated the result at MoE scale. In the primary chat-wrapped format:

| Format arm | Data-free combined KL | BiasCorr all-experts combined KL | Direction |
|---|---:|---:|---|
| HF-INT4-G16 W4A16 | `0.15772356` | `0.11258001` | about `29%` lower |
| NVFP4 | `0.16880991` | `0.11544378` | about `32%` lower |
| MLX-asym-G64 | `0.26208259` | `0.16917791` | about `35%` lower |
| SymINT4-G128 | `0.44910335` | `0.28976358` | about `35%` lower |

The same receipt contained a K/V sensitivity probe. Full-depth K-only HF-INT4-G16 data-free combined KL was `0.03356326`; V-only was `0.05043667`. On the sliding-only matched comparison, K-only was `0.01888637`. This does not restore a universal "K is always worse than V" law. Exp13B had already falsified that law across architectures. It does show that Gemma-4-26B has its own asymmetric K/V quantization sensitivity and that per-architecture measurement beats inherited rules.

The most surprising part is how little correction data was needed for most of the benefit. The complete all-experts sidecar is large because 128 experts dominate it, about 64 MiB. The dense attention plus shared-MLP subset is only `2,186,240` tensor bytes, about 2 MiB. For HF-INT4-G16 it produced combined KL `0.11157237`, statistically inseparable from the all-experts arm at `0.11258001` and far below the data-free `0.15772356`. The same pattern held for MLX within noise. The receipt is careful here: dense layers explain most of the gain, and per-expert correction cannot be separated from attention plus shared-MLP correction at this calibration size.

### 2.4 Alternatives that were tried and killed

The study did not stop at the first correction that worked.

- Rank-AR was tested as a low-rank residual correction and rejected. It added complexity without buying enough reliable KL improvement.
- WienerAffineCorr was tested in Exp16 and rejected. It was more expressive but less reliable and harder to bound.
- Position-0 inclusion was rejected after causality and activation-distribution checks showed that it polluted the mean.
- The naive BF16 accumulation loop for MoE experts was itself a bug source. In the OLMoE patch identity work, a per-expert BF16 `index_add` path did not reproduce stock `grouped_mm`; a slot-wise FP32 accumulation path reproduced it exactly (`0.0` max difference) on the diagnostic layers. The receipt is `results/exp15b_olmoe_patch_identity_diag.json`.

BiasCorr survived because it is small, linear, easy to fuse, and easy to reject when the gate fails. It is not a universal patch and the public artifacts now say so.

### 2.5 The GGUF-tier intuition and why it is only an intuition

A useful bench intuition came from Qwen3.8-27B: moving from about 4.5-bit class storage toward 8-bit class storage lowers KL by roughly a factor of `2.1` per additional bit per weight. That number is obtained by comparing the Qwen KL table across the 4.5 to 8.0 bpw range, not by fitting a universal law. It gives a scale for what a small correction is worth: if a tiny sidecar cuts KL by a factor comparable to a fraction of a bit of extra storage, it is worth engine attention.

This is not a claim that BiasCorr is equivalent to GGUF Q8, or that every model has the same slope. It is a communication aid for engine maintainers. The measured fact is narrower: on the families that pass the gate, a few megabytes of correction can buy a KL reduction that would otherwise require a much larger weight payload.

### 2.6 Why engine adoption is cheap

The Exp18 blueprint inspected llama.cpp and Apple mlx-lm rather than guessing. llama.cpp already has generic GGUF bias tensor loading and graph plumbing for many dense and MoE paths. The main gap is per-architecture wiring. The `--lora` path cannot carry this correction because LoRA is a low-rank input-dependent weight update, not a static output bias.

The proposed llama.cpp phase 1 is to embed bias vectors into GGUF and wire qwen3/qwen2 dense first. The estimated implementation surface is roughly 0 to 40 lines depending on whether the target architecture already exposes a bias hook. A runtime `--biascorr-file` is phase 2, after format and scope are pinned.

The MLX path is even simpler. The correction must be injected after `load_model`, because strict loading fails if unknown bias keys are inserted too early. After loading, a Python integration can attach the vectors and force evaluation with `mx.eval`; the blueprint estimates about 20 lines. Receipt: `exp18_llamacpp_mlx_loader_blueprint.md`.

One caveat decides whether the integration works: the inspected Qwen3.8 artifact uses hybrid linear-attention keys such as `linear_attn.in_proj_qkv`, `in_proj_a`, and `in_proj_z`. It is not a one-to-one `self_attn.q/k/v` model. Any integration that assumes a simple dense attention layout will wire the wrong tensors.

## 3. Research and measurement methodology: every guardrail we built

### 3.1 Correctness before quantization

No quantized arm was supposed to run until the BF16 oracle was believable. The checks were:

1. Exact identity: the harness output had to match the reference implementation with max absolute difference `0.0` on the tested windows.
2. Repeat identity: two independent streaming passes had to be exactly equal.
3. Future perturbation: changing a future token could not change earlier logits.
4. Negative perturbation control: changing the suffix had to change the suffix, proving the test was not trivially insensitive.
5. Functional probes: simple recall questions had to produce sane top tokens.
6. Realistic BF16 perplexity: if BF16 perplexity was absurd, quantization scores were meaningless.

This mattered because the first harness failed exactly here. The non-causal attention path made WikiText-2 perplexity look impossibly low. After the causal rewrite, Qwen3.8-27B BF16 had WikiText-2 PPL `7.79822` and code PPL `2.37436`, with final hidden-state max absolute diff `0.0` against the HF reference on the tested path.

Exp17B applied the same discipline to a much harder model. The Gemma-4-26B identity gate was `11/11` exact zeros for the streaming-repeat checks, but it also revealed a subtle expert batching effect: stock `grouped_mm` results depend deterministically on which other tokens share an expert batch. That is why the run added a BF16 rounding-floor arm using FP32 expert matmuls rounded to BF16. Its combined KL floor was `0.00835045`; small quantization differences below that scale should not be overinterpreted.

### 3.2 Calibration discipline

The mature calibration rules were:

- Use train-split text, not evaluation text.
- Require zero 6-gram overlap between calibration and evaluation content. The strict target was `0/0` overlapping 6-grams for both WikiText-2 and code.
- Exclude position 0 from means.
- Record exact corpus hashes and token starts.
- Use the same chat or raw wrapping for calibration and evaluation when the model is instruction tuned.
- Use the exact negative `-b` sidecar as a sign-flip control.
- Use 95% block-bootstrap confidence intervals with block size 32.

The honesty problem in Wave 7 was that some live TensorFold evidence used predecessor sidecar `5719cfb4...`, calibrated on the WT2 test tail and sharing one code 6-gram, while the public train-split sidecar was `f5fce21b...`. That did not make the measurements fabricated, but it did mean they were not the same arm. The final public study had to say so plainly.

### 3.3 Live serving discipline

Live serving measurements used a different rulebook from offline KL. The important parts were:

- Both cards capped equally at 230 W and 2,800 MHz.
- Model-card sampling and default thinking mode for externally quoted speed.
- Interleaved `ABABAB` style comparisons where practical, rather than all baseline runs followed by all experimental runs.
- One warm-up discarded before timing.
- Cooling between cells.
- A hard 18 GiB VRAM cap in the deep-study harness after the 12:54 out-of-memory reboot.
- `torch.use_deterministic_algorithms(True)` after Gemma nondeterminism was isolated.
- Drafted output must equal serial output for any speculative speed claim. If hashes differ, the speed is labeled lossy speculation.

The 12:54 incident is worth keeping in the record because it changed methodology. A Gemma-4-26B run filled memory badly enough to reboot the host. After that, the harness gained a stricter memory cap and a spill guard. This is the kind of operational detail that often disappears from polished papers, but it explains why later receipts are shaped the way they are.

### 3.4 Metrics are not interchangeable

The study uses at least four different measurement types:

| Metric | What it means | What it does not mean |
|---|---|---|
| Offline causal KL | How close quantized next-token distributions are to BF16 on fixed windows | Task accuracy, user-visible quality, or serving speed |
| Offline perplexity | Exponentiated mean NLL on fixed windows | Direct proof of causality or correctness by itself |
| Binary task score | Correct/incorrect on the frozen 160-item suite | Fine-grained distributional closeness |
| Kernel microbenchmark | Time and bandwidth for a fixed shape | End-to-end serving throughput |

Exp15B made the distinction unavoidable. Across 20 measured points, KL did not reliably predict task-score drop: Pearson `r=-0.1554473`, Spearman `-0.1159203`. In plain language, a format can look much closer to BF16 in KL and still answer more binary questions incorrectly, or vice versa. This does not mean KL is useless. It means KL is a distribution-level forensic metric, not a task benchmark.

The task table also refuses to flatten into one story:

| Model | BF16 | HF-G16 DF | HF-G16 BC | MLX DF | MLX BC | SymINT4 DF |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.2-3B | `104/160` | `93` | `84` | `94` | `92` | `90` |
| OLMoE-1B-7B | `83` | `82` | `77` | `68` | `78` | `74` |
| Phi-4-mini | `104` | `107` | `102` | `95` | `91` | `100` |
| OLMo2-7B | `90` | `89` | `93` | `99` | `92` | not shown here |
| Granite-4.0-h-tiny | `100` | `110` | `105` | `99` | `106` | `96` |
| Gemma-4-E4B partial tasks | `113` | `110` | incomplete | `104` | `107` | `84` |

Sources: `results/exp15b_multifamily_scores.json`, `results/exp17a_granite4_htiny.json`, and the still-running `results/exp17a_gemma4_e4b.json`. The Gemma row is not final.

One example is enough to show why pairwise statistics matter. OLMoE HF-G16 data-free beat MLX data-free by `+14` correct answers with reported `p=0.024`. But on the Qwen serving batteries, HF-G16-BC `287/329`, TensorFold MLX plus predecessor sidecar `284/329`, vLLM GPTQ `283/329`, and TensorFold MLX base `282/329` were pairwise non-significant. Exact McNemar examples: `p=0.48068` for the vLLM HF versus shipped GPTQ comparison and `p=0.80362` for TensorFold MLX base versus MLX plus BiasCorr. Those are not victories; they are controlled ties inside the tested battery.

### 3.5 Serving speed and kernel truth

vLLM serving showed a real split. HF-G16-BC prefill was faster than shipped GPTQ at 8K and 32K prompt lengths (`+27.87%` and `+25.74%` respectively), but decode was slower by roughly 10 to 12 percent. The kernel diagnosis explains why this is not contradictory: at large M, the HF-G16 path can beat oneDNN BF16 and stock `inc_w4a16`; at M=1, stock `inc_w4a16` remains faster, and oneDNN INT8 remains much faster still. The format is useful, but the decode kernel is not yet the best use of Xe2.

Wave 9C added a second kernel warning for MoE. At the M=512 Gemma shape, the implementation packed only about 388 MiB of weights but logically read about 12.1 GiB per call, roughly 32x duplicate weight traffic. Measured bandwidth around 537 GB/s, about 88% of peak, means the kernel was efficient at moving bytes and inefficient at choosing which bytes to move. The proposed Step A fix groups work by expert while preserving exact numerics. A later Step B or DPAS rewrite may change numerics and must be a separate PR with separate evidence.

The Wave 8A GPU0 MoE benchmark also had a contamination caveat: Wave 8A was simultaneously running on GPU1, and mean power exceeded the 230 W cap. Therefore the headline 4.4x to 5.7x-style numbers must not be quoted without explanation or rerun. The raw receipts are still useful, but they are not clean public performance claims.

## 4. The adversarial multi-reviewer ledger

This table is the heart of the chronicle. It separates who found what, because the project only became trustworthy when each reviewer was allowed to break the previous story.

| Reviewer or agent | Role | Exact defects and contributions | Where recorded |
|---|---|---|---|
| Claude Opus 5.5, Wave 1 forensic audit | First adversarial auditor | Rejected the `-0.0f` test as self-verifying because indexing, interleave, and tensor-parallel slicing bugs can cancel or remain invisible. Forced nonzero index-coded host FP64 oracles. Caught the EXL3-SC5 misattribution. Forced attention causality, train/test separation, manifest hashing, and realistic PPL prechecks after the impossible non-causal WT2 result. | `opus55_wave1_audit.md` |
| GPT-6.2 Astra, Wave 2B design | Kernel and systems architect | Designed the vLLM-HF-INT4-G16 architecture before implementation: packed nibble contract, Hadamard sign provenance, native versus Triton split, fused loader requirements, strict versus mixed W4A8 mode, TP row/column semantics, and memory lifetime requirements. | `vllm_plugin_design_astra.md` |
| GPT-6.2 Astra, Wave 3A/3C review | Adversarial code reviewer | Issued `CHANGES REQUESTED` rather than accepting a good WT2 number. Caught fused-loader coverage gaps, missing dtype/shape checks, replicated-KV TP slicing errors, block-I/O alignment gaps, wrong-device and wrong-dtype native entry acceptance, Triton tail over-read, half-precision ACTQ overflow before RNE, Triton `floor(z+0.5)` instead of nearest-even, unsafe rank-0-only correction without proven downstream sum, context key and kernel-cache lifetime races, and strict/prefix/speculative scheduler unsoundness. The follow-up counted 12 native/Triton/admission/RNE/digest-cache/replicated-KV defect classes and ended with 130/130 tests in the hardened tree. | `astra_wave3_code_review.md`, `astra_wave3_rereview.md` |
| Claude Opus 5.5, Wave 6A sign-off | Narrow implementation sign-off | Signed off only the tested dense implementation scope. Refused to generalize to strict serving, tensor parallel, prefix cache, speculative execution, or MoE. Preserved the distinction between a passing test suite and a production engine contract. | `opus55_wave6a_code_signoff.md` |
| Claude Opus 5.5, Wave 7B publication review | Publication auditor | Caught the `+0.08%` versus `+0.0015%` live-versus-offline arm mismatch. Caught that the train-split `f5fce21b` sidecar was not the predecessor `5719cfb4` sidecar used in much TensorFold live evidence. Required Intel-only loader honesty and exact caveats around compressed-weight totals. | `opus55_wave7b_publication_signoff.md` |
| Claude Opus 5.5, Wave 7D final sign-off | Final publication gate | Approved the public V10 study only after the arm mismatch, sidecar mismatch, exact-zero padding, FP8 comparability, and measurement-scope language were corrected. The verdict was approval for publication with documented scope, not blanket endorsement. | `opus55_wave7d_final_signoff.md` |
| Claude Opus 5.5, Wave 9C v0.2.0 review | Release blocker review | Issued `CHANGES REQUESTED` with four blockers: uniform Had64 had not been quality-gated against the Exp17B mixed Had128 plus Had64-tail receipt; v0.1.0 Had128 kernels were rewritten as macros without bit-exact same-output evidence; admission needed a `processed == admitted` proof and negative rename test; and `had_block_rule` had to be validated exactly for `/2` while rejecting contradictory top-level `hadamard_block` metadata. | `opus55_wave9c_v020_review.md` |
| GPT-6.2 Astra, Wave 9C v0.2.0 review | Systems and admission reviewer | Caught Xe2 2D block-I/O 64-byte surface alignment and legal width/height/pitch requirements, the possibility that a missing whole expert bank looks like intentional BF16, standalone Triton MoE accepting `E=0` and reading an empty expert table, Granite's documented BF16 expert opt-out silently quantizing legacy `block_sparse_moe` keys, and int64 route IDs being narrowed before validation with modulo-2^32 aliasing. Designed the Step A grouped-by-expert prefill fix that preserves exact numerics before any numerics-changing DPAS rewrite. | `astra_wave9c_v020_review.md` |
| Claude Sonnet 5.5 | Execution, bug hunting, and gate engineering | Fixed the OLMoE per-expert BF16 accumulation bug by replacing lossy BF16 index accumulation with slot-wise FP32 accumulation; the diagnostic moved the patched path to exact `0.0` max difference against stock grouped matmul on tested layers. Built the three-check KL gate and receipt-verified 1600x900 result-card generation path. Diagnosed Gemma-4-26B XPU matmul nondeterminism and the grouped-matmul batch-rounding floor, leading to deterministic algorithms plus the explicit rounding-floor arm. Added VRAM spill protection after the host reboot. In v0.2.0 bring-up, caught Gemma `k_eq_v` double-`k_proj` behavior and Granite `mixer`/`scale_super` mapping problems before they could be treated as clean model support. | `results/exp15b_olmoe_patch_identity_diag.json`, `results/exp17b_gemma4_26b_a4b.json`, `opus55_wave9c_v020_review.md`, `astra_wave9c_v020_review.md` |
| SWE-2.1 | Multi-family scaling, harness engineering, ecosystem mapping, chronicle maintenance | Built and drove the Exp15B multi-family harness through the completed families, kept negative and null outcomes visible, mapped llama.cpp and MLX adoption surfaces, prepared the Exp18 ultra-low-bit ternary research plan, and maintained the chronicle lineage. The task brief credits SWE-2.1 with harness construction, but the current Exp18 directory is empty of harness files and receipts at compile time, so this chronicle cannot verify a completed harness artifact. | `results/exp15b_multifamily_scores.json`, `exp18_llamacpp_mlx_loader_blueprint.md`, `COMPLETE_RESEARCH_CHRONICLE.md` |
| External GitHub reviewer `@AIFlowML` / Igor Lessio | Independent public verification on TensorFold PR 5028 | Verified host-side `biascorr.zig` on an Apple M5 Max with Zig 0.17.0, including synthetic safetensors validation. Asked for exact build command, `ocloc` version, workaround disclosure, and a reproducible fixture/sidecar recipe. | Public PR `ashhart/TensorFold#5028` |
| External GitHub reviewer `@ashhart` / Synthena | Upstream maintainer review | Caught that the PR was accidentally stacked on local tensor-parallel and prefix-attention commits rather than upstream `main`, producing an 18-file conflict in the trial merge. Clarified TensorFold's target-preservation invariant: nonzero target corrections change target accumulators and cannot be accepted as target-preserving upstream; they must be removed, made explicitly non-target-preserving, or kept draft-only. | Public PR `ashhart/TensorFold#5028` |

Two reviewer-ledger cautions matter.

First, internal model names are labels for the agents used in the workflow. They do not imply independent human authorship. The reviews are valuable because they were adversarial and artifact-based, not because a model name sounds authoritative.

Second, not every reviewer role is a sign-off role. Sonnet 5.5 and SWE-2.1 mostly executed, diagnosed, and built. Opus and Astra often reviewed. The external GitHub reviewers were the only humans in the ledger, and they entered only after the work became public.

## 5. Bugs that changed the direction of the project

Some bugs were small implementation errors. Others changed the scientific conclusion. These are the ones that mattered most.

1. **Non-causal attention oracle.** The original standalone attention path read future information. It produced an impossible WikiText-2 perplexity scale around `2.85`. This was not a quantization result. It was a broken oracle. The causal rewrite made all subsequent claims slower, more expensive, and believable.

2. **Self-verifying zero oracle.** A `-0.0f` check can pass even when channel ordering is wrong because both sides of the comparison inherit the same mistake. Nonzero index-coded host FP64 fixtures were required for interleaved gate/up, fused QKV, GDN ordering, mixer output, and tensor-parallel slices.

3. **Train/test contamination.** Early predecessor calibration overlapped evaluation structure. Wave 7 forced explicit hashes, train-split calibration, zero 6-gram overlap, and disclosure of the predecessor sidecar mismatch.

4. **Triton rounding and ACTQ overflow.** The Triton activation quantizer used `floor(z+0.5)` rather than nearest-even. The ACTQ path could cast finite FP32 values to half before RNE and overflow. Fixed-input tests could pass while real activations quantized incorrectly.

5. **Replicated KV tensor-parallel slicing.** Rank 1 cannot load the "next" single KV head. Replicated heads must be replicated. The original fused loader logic sliced them as if they were unique.

6. **Precision switching at physical batch size.** Switching W4A16 to W4A8 at physical M=4 or 5 can make the same logical token depend on scheduler packing. Strict mode now defaults to uniform precision unless mixed mode is explicitly requested and isolated from prefix/speculative reuse.

7. **OLMoE expert accumulation.** Per-expert BF16 index accumulation did not reproduce stock grouped matmul. Slot-wise FP32 accumulation did, exactly. This is why MoE oracle implementation is part of the measurement, not a detail beneath it.

8. **Gemma nondeterminism and grouped batching.** XPU matmul repeated calls produced multiple distinct results before deterministic mode. Separately, deterministic grouped matmul still depends on which tokens share an expert batch. Both facts had to be represented in the rounding floor.

9. **Gemma raw-text evaluation mismatch.** Raw text on the instruction-tuned checkpoint produced BF16 WikiText-2 PPL around `40,294`, far outside sanity bounds. Chat wrapping was therefore primary, with raw text retained only as secondary evidence.

10. **Uniform Had64 evidence gap.** v0.2.0 wrote uniform Had64, but Exp17B had validated mixed Had128 plus Had64-tail blocks. This is the central open release blocker.

11. **MoE admission edge cases.** Missing whole expert banks, legacy Granite key aliases, zero-expert dispatch, narrowing route IDs, and unvalidated block-I/O surfaces are all loader or admission failures. They can produce plausible output while silently dropping or aliasing real computation.

12. **TensorFold upstream mismatch.** The public PR was accidentally stacked on unrelated local tensor-parallel and prefix-attention commits. Separately, the correction epilogue violates a strict target-preservation invariant even when it improves output. This is an upstream design conflict, not a test failure.

## 6. Public today versus still in flight

| Artifact or claim | Status as of 2026-11-25 | Precise meaning |
|---|---|---|
| Intel Arc Pro B70 Inference Cookbook `v1.23.0` | Public | Cookbook lineage is public; the exact release contents should be read from the public cookbook, not inferred from this chronicle. |
| `vllm-hf-int4-g16-xpu v0.1.0` | Public | Released only with documented scope: dense HF-INT4-G16 XPU plugin evidence, not a general quantization framework and not MoE/Had64 v0.2.0 support. |
| `SergiioB/biascorr-sidecars-v1` | Public | Public sidecar collection. The train-split sidecars and their exact generation/gate provenance are part of the artifact contract. |
| `SergiioB/Qwen3.8-27B-HF-INT4-G16-BC` | Public | Public Qwen3.8 HF-INT4-G16 plus BiasCorr model artifact, again scoped to the tested dense layout. |
| `ashhart/TensorFold#5028` | Public but open and conflicting | Not merged. Head `b30d13861675cd490fef9819b0e42b05088f7a77` at the time of inspection. The PR is public evidence of the fused epilogue approach, but it is blocked by both the accidental stacked base and the upstream target-preservation question. |
| Exp17A Gemma-4-E4B tasks | In flight | Part A causal KL is complete and positive for all four tested formats. Part B task scoring is still incomplete. At compile time, BF16, HF-G16 DF, MLX BC, MLX DF, and SymINT4 DF task arms were present; other arms were not final. |
| Exp17A Qwen3-8B | In flight | Listed as a target, but no completed receipt existed at compile time. |
| Wave 9D v0.2.0 fixes | In flight | Needs the actual v0.2.0 converter rerun through the Exp17B KL gate, bit-exact v0.1.0 kernel A/B proof, admission completeness assertions, metadata validation, and negative tests. |
| Grouped MoE prefill | In flight | Step A exact-numerics grouping is designed but not yet a clean public benchmark claim. The contaminated Wave 8A benchmark must be rerun before quoting speedups. |
| Exp17C `Qwen/Qwen3.6-35B-A3B` | In flight | Planned modern MoE scaling target. No completed receipt in the study at compile time. |
| Exp18 ultra-low-bit `3b/2b/1.58b` ternary plus closed-loop BiasCorr | In flight | Blueprint and ecosystem mapping exist. The local `exp18-ultralowbit` directory contained no harness files or receipts at compile time, so there are no measurements to claim yet. |
| llama.cpp and MLX BiasCorr integration | In flight | Adoption blueprint exists. No merged upstream implementation is claimed. MLX injection after `load_model` is the lowest-cost first proof. |

## 7. What the project established

The strongest positive claim is narrow but real. On the tested Xe2 B70 stack, Had128NoPerm plus HF-INT4-G16 is a credible 4.25-bit-class serving format for dense linears, with causal KL wins across an expanding set of model families and with prefill advantages in vLLM at long prompts. It is not yet the fastest decode path, not yet a general MoE release, and not yet proven for strict speculative or prefix-cache serving.

The strongest correction claim is also narrow. BiasCorr can be substantial on architectures whose quantization error has a stable mean component, including Qwen3.8, Granite-4.0-h-tiny, Gemma-4-E4B, and Gemma-4-26B-A4B. It can also hurt badly on small dense models and on already-calibrated GPTQ-style arms. The gate is therefore not bureaucracy. It is the feature that keeps the method honest.

The most important scientific negative result is that KL does not predict binary task movement in a simple way across families. That result prevents overclaiming. It also means future work must report both distributional and behavioral evidence rather than treating either one as a proxy for the other.

The most important engineering negative result is that a kernel can be bandwidth-efficient and algorithmically wasteful at the same time. The MoE prefill path hit high bandwidth because it reread weights, not because it had found a good schedule. The fix is to reduce duplicate expert traffic before changing numerics.

The most important governance result is that review worked. Every attractive simple story was damaged by evidence: the first oracle was broken, BiasCorr was not universal, Qwen-only skepticism was also wrong, trellis coding won rate-distortion but not serving, v0.2.0 was ahead of its evidence, and the upstream TensorFold PR exposed a design invariant rather than a mere patch bug.

That is the actual chronicle. It is messier than a launch post, but it is the version future agents and humans need if they are going to extend the work without repeating the same mistakes.

## 8. Source index used for this chronicle

Primary chronicle and public-study sources:

- `COMPLETE_RESEARCH_CHRONICLE.md`
- `publish-bundle/PUBLIC_STUDY_V10.md`

Review and sign-off sources:

- `opus55_wave1_audit.md`
- `vllm_plugin_design_astra.md`
- `astra_wave3_code_review.md`
- `astra_wave3_rereview.md`
- `astra_wave5a_code_signoff.md`
- `opus55_wave6a_code_signoff.md`
- `opus55_wave7b_publication_signoff.md`
- `opus55_wave7d_final_signoff.md`
- `opus55_wave9c_v020_review.md`
- `astra_wave9c_v020_review.md`

Receipt sources:

- `results/exp15b_multifamily_scores.json`
- `results/exp15b_olmoe_patch_identity_diag.json`
- `results/exp17a_granite4_htiny.json`
- `results/exp17a_gemma4_e4b.json`
- `results/exp17b_gemma4_26b_a4b.json`
- `results/exp17b_scripts/PROCESS.md`
- `results/exp17_moe_plugin_bench_gpu0.json`
- `results/exp17_moe_engine_smoke_gpu0.json`

Blueprint and public-review sources:

- `exp18_llamacpp_mlx_loader_blueprint.md`
- Public GitHub PR `ashhart/TensorFold#5028`, inspected at head `b30d13861675cd490fef9819b0e42b05088f7a77`
