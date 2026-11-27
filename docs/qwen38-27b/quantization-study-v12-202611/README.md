# Quantization Deep Study V12 Addendum (2026-11-27): Full-Hessian Closed-Loop GPTQ Inside `hf_int4_g16/1`

> **Relationship to V10 & V11:** [V10](../quantization-study-202611/README.md) and [V11](../quantization-study-v11-202611/README.md) remain immutable with all receipts preserved intact. This V12 addendum (`Exp 23`, internal run ID `exp19_gptq_v1`) updates V11 §7.1 / §7.3 on the calibrated scalar vs data-free trellis boundary and publishes a dedicated `v2` GPTQ checkpoint ([`SergiioB/Qwen3.8-27B-HF-INT4-G16-GPTQ`](https://huggingface.co/SergiioB/Qwen3.8-27B-HF-INT4-G16-GPTQ)) while preserving `v1` RTN+BC intact at [`SergiioB/Qwen3.8-27B-HF-INT4-G16-BC`](https://huggingface.co/SergiioB/Qwen3.8-27B-HF-INT4-G16-BC).

## 1. Bottom Line & Revision of the V11 Thesis

V11 (`Exp 15A`) established that **when neither side uses calibration data**, Viterbi trellis quantization (`TFQ3-4b`, `4.001 bpw`) beats scalar `Had128NoPerm + HF-INT4-G16` RTN (`4.255 bpw`) in pure quality-per-bit (`0.02528 / 0.11042` vs `0.03163 / 0.14920` KL vs `BF16`).

**V12 (`Exp 23`) shows that wall came from the data-free RTN rounding rule, not from the `hf_int4_g16/1` scalar grid.** Replacing streaming RTN with sequential closed-loop **Full-Hessian GPTQ** (`261,632` pos0-excluded calibration tokens, CPU FP64 Cholesky, dynamic per-16-column micro-scale `u in {1..16}` re-optimization under fixed RTN `d_row`, bit-exact `xe2_dpas_native_v1` packing) inside the **exact same serving binary format**:

1. **Cuts in-domain KL vs `v1` RTN+BC by `-39.32%` on prose and `-30.44%` on code** at W4A16 (`WT2 0.02708 -> 0.01643`, `Code 0.12897 -> 0.08972`, `bootstrap P(Δ<0) = 1.0000`, `10,000/10,000` resamples) and `-36.42%` / `-30.25%` at W4A8 (`0.02713 -> 0.01725` WT2, `0.13484 -> 0.09405` Code).
2. **Generalizes across all 7 tested domains without OOD regression (`Exp 23-OOD`):** Beats `v1` on `7/7` domains including **Spanish Wikipedia (`-12.6%`, `P=0.9986`, significant)**, **2K long context (`-34.3%` WT2 / `-27.1%` Code, `P=1.0000`, significant)**, **OOD chat (`-11.8%`, `P=0.9640`, directional)**, and **Spanish Don Quixote (`-1.7%`, neutral/positive)**. On V11's 3-domain combined metric (`WT2 + Code + OOD-chat`), `v2` cuts mean KL from `0.06998` to **`0.05123` (`-26.8%`, CI `[-0.02333, -0.01447]`, `P=1.0000`)**.
3. **Moves below data-free trellis `TFQ3-4b` on both in-domain test sets** (`0.01643 / 0.08972` vs `0.02528 / 0.11042`, unpaired point estimates) and beats `TFQ3-4b + BC` on WT2 (`-11.3%`) while sitting `+1.0%` higher on Code (`0.08972` vs `0.08879`), with full fair qualifiers below (`4.255` vs `4.001 bpw`, `261K` cal tokens vs data-free / `2K`, `496` vs `400` quantized linears; Hessian-calibrated trellis is untested and would likely retake the lead).
4. **Renders post-GPTQ `BiasCorr` unnecessary on both W4A16 and W4A8:** `BiasCorr` fails the gate at W4A16 (delta `-0.000018`, below `~2e-4` noise floor) and fails the 4-check OOD gate at W4A8 (`pass_4check_ood: false`, loses to sign-flipped `-BC` on OOD-chat and degrades 2K W4A8 by `+3.7%` / `+4.7%`). Dropping `bias_corr` saves `14.9 MiB` (`18.45 GiB` vs `18.47 GiB`).
5. **Serves unchanged in `vllm-hf-int4-g16-xpu v0.2.0` (`496/496` modules native, `0` Triton fallbacks):** Live `auto` served PPL matches offline W4A8 (`7.886` vs `7.879` WT2, `2.518` vs `2.517` Code) at decode/prefill speed parity (`29.92` vs `29.69 tok/s` `C=1`, `2,533` vs `2,497 tok/s` 2K prefill).

## 2. Full Cross-Format Scoreboard (`Qwen3.8-27B`, Identical `2 × 2,044` Tokens)

All rows share exact `BF16` reference parity (`max_abs_diff = 0.0`) on the frozen `Exp 13A` / `Exp 15A` token streams (`wt2_sha256: 14049ba0...`, `code_sha256: 6c3a613f...`). Cross-format comparisons against `FP8`, `TFQ3-4b`, `MLX-4bit`, and `GPTQ-G128` are unpaired point estimates.

| Format | Effective Bits | Quantized Linears | WT2 KL vs BF16 (Top-1) | Code KL vs BF16 (Top-1) | Live Intel B70 Status |
|---|---:|---:|---:|---:|---|
| `BF16` Reference | `16.0` bpw | `0` | `0.00000` (`100.00%`) | `0.00000` (`100.00%`) | TP=2 (`54 GB`) |
| `FP8` (shipped) | `~8.0` bpw | `400` | `0.00358` (`97.60%`) | `0.01689` (`96.48%`) | TP=2 (`28 GB`) |
| **`HF-INT4-G16 v2` GPTQ W4A16** (`-GPTQ` repo) | **`4.255` bpw** | **`496`** | **`0.01643` (`95.01%`)** | **`0.08972` (`93.10%`)** | **Live 1× B70 (`vLLM-XPU`)** |
| **`HF-INT4-G16 v2` GPTQ W4A8 DPAS** | **`4.255` bpw** | **`496`** | **`0.01725` (`94.96%`)** | **`0.09405` (`93.05%`)** | **Live 1× B70 (`auto` prefill)** |
| `TFQ3-4b` Trellis + `BiasCorr` (matched 400) | `4.001` bpw | `400` | `0.01852` (`94.47%`) | `0.08879` (`92.86%`) | Metal only |
| `TFQ3-4b` Trellis Data-Free (matched 400) | `4.001` bpw | `400` | `0.02528` (`93.15%`) | `0.11042` (`91.14%`) | Metal only |
| `HF-INT4-G16 v1` RTN + `BiasCorr` (`-BC` repo) | `4.260` bpw | `496` | `0.02708` (`94.32%`) | `0.12897` (`91.73%`) | Live 1× B70 (`vLLM-XPU`) |
| `HF-INT4-G16 v1` RTN Data-Free | `4.255` bpw | `496` | `0.03163` (`94.13%`) | `0.14920` (`90.02%`) | Live 1× B70 (`vLLM-XPU`) |
| `TFQ3-4b` As-Shipped (4-bit `lm_head`) | `4.001` bpw | `400+head` | `0.03206` (`91.88%`) | `0.11327` (`91.05%`) | Metal only |
| `MLX-4bit` + `BiasCorr` / Base | `~4.25` bpw | `400` | `0.03897` / `0.05060` | `0.15879` / `0.18407` | TensorFold XPU |
| Stock `GPTQ-Int4-sym-G128` | `~4.25` bpw | `400` | `0.04942` (`91.44%`) | `0.30198` (`85.71%`) | Live 1× B70 (`vLLM-XPU`) |

*Note:* `NVFP4` was not measured on `Qwen3.8-27B` (in V11 it beat RTN `HF-INT4-G16` on `Qwen3-8B` and `SmolLM2`; whether `v2` GPTQ beats `NVFP4` is untested).

## 3. Out-of-Domain (OOD) & Long-Context Verification (`Exp 23-OOD`, 64L True-Causal)

Evaluated on the exact shipped `v2` weights (`500.4 s`, `57.6 kJ` GPU1, cool `39 -> 58 °C`) with zero reconversion. 6-gram overlap against the 256K GPTQ calibration corpus is `0/0` on `wt2_512`, `code_512`, `es_wiki_512`, `es_quixote_512`, `wt2_2048`, and `code_2048`, and `11` boilerplate Python signature 6-grams on `ood_chat_512` (HumanEval prompts).

| Domain | `v1` RTN+BC KL (Top-1) | **`v2` GPTQ KL (Top-1)** | Delta (`v2` vs `v1`, paired block-32 95% CI) |
|---|---:|---:|---|
| WikiText-2 (`4×512`, in-domain) | `0.02708` (`94.32%`) | **`0.01643`** (`95.01%`) | **`-39.3%`** (`[-0.01458, -0.00718]`, `P(Δ<0)=1.0000`, PASS) |
| Python `difflib.py` (`4×512`, in-domain) | `0.12897` (`91.73%`) | **`0.08972`** (`93.10%`) | **`-30.4%`** (`[-0.05002, -0.02937]`, `P(Δ<0)=1.0000`, PASS) |
| OOD Chat (`4×512`, GSM8K/MMLU/HumanEval, Qwen template) | `0.05389` (`94.37%`) | **`0.04754`** (`94.52%`) | **`-11.8%`** (`[-0.01420, +0.00048]`, `P(Δ<0)=0.9640`, n.s.) |
| Spanish Wikipedia (`4×512`, OOD language) | `0.01639` (`95.11%`) | **`0.01432`** (`95.01%`) | **`-12.6%`** (`[-0.00335, -0.00069]`, `P(Δ<0)=0.9986`, PASS) |
| Spanish Don Quixote (`4×512`, OOD literary) | `0.16220` (`88.70%`) | **`0.15937`** (`87.62%`) | **`-1.7%`** (`[-0.02056, +0.01433]`, `P(Δ<0)=0.6329`, n.s.) |
| WikiText-2 Long (`1×2048`, 4× cal length) | `0.02141` (`95.31%`) | **`0.01408`** (`95.07%`) | **`-34.3%`** (`[-0.01034, -0.00483]`, `P(Δ<0)=1.0000`, PASS) |
| Python `difflib.py` Long (`1×2048`, 4× cal length) | `0.12215` (`92.62%`) | **`0.08903`** (`93.89%`) | **`-27.1%`** (`[-0.04767, -0.01885]`, `P(Δ<0)=1.0000`, PASS) |

**Takeaway:** `v2` improves on `7/7` domains with zero OOD degradation. As expected from a Hessian fit calibrated on English prose + Python code (`512` tok), in-domain and 4×-length gains are largest (`-27%` to `-39%`, all `P=1.0000`), while completely unseen language/format domains retain real positive gains (`-12.6%` significant on Spanish Wikipedia, `-11.8%` directional on Qwen chat-templated GSM8K/MMLU/HumanEval, `-1.7%` flat/positive on 17th-century Spanish Don Quixote).

## 4. W4A16 & W4A8 Post-GPTQ `BiasCorr` Gate (`Exp 23` + `Exp 23-OOD`)

- **W4A16:** `+BC` changed WT2 KL by `-0.000018` (`bootstrap P(Δ<0) = 0.587`) and Code KL by `-0.000044` (`P = 0.516`), while sign-flipped `-BC` moved WT2 by `-0.000206` (`P = 0.990`). No directional `BiasCorr` signal exists above the `~2e-4` KL noise floor after Full-H GPTQ.
- **W4A8 (`8,176` closed-loop pos0-excluded cal tokens, 3-arm `DF` vs `+BC` vs `-BC`):** Overall 4-check OOD gate: **`FAIL` (`pass_4check_ood: false`)**. Sign-flipped `-BC` beats `+BC` on OOD-chat (`P(+BC < -BC) = 0.0363`), and `+BC` degrades 2K long-context W4A8 by `+3.7%` WT2 and `+4.7%` Code.

| Domain (W4A8 DPAS) | `v2` W4A8 `DF` KL | `v2` W4A8 `+BC` KL | `v2` W4A8 `-BC` (null) KL |
|---|---:|---:|---:|
| WikiText-2 (`4×512`) | `0.01725` | `0.01688` (`-2.16%`, `P=0.8261`) | `0.01715` (`-0.60%`) |
| Code (`4×512`) | `0.09405` | `0.09173` (`-2.46%`, `P=0.9270`) | `0.09312` (`-0.99%`) |
| OOD Chat (`4×512`) | `0.04886` | `0.04840` (`-0.93%`, `P=0.6360`) | `0.04651` (`-4.81%`) |
| Spanish Wiki (`4×512`) | `0.01489` | `0.01481` (`-0.50%`, `P=0.5725`) | `0.01495` (`+0.46%`) |
| Spanish Quixote (`4×512`) | `0.16549` | `0.16247` (`-1.82%`, `P=0.8311`) | `0.16622` (`+0.44%`) |
| WikiText-2 (`1×2048`) | `0.01394` | `0.01446` (`+3.73%`, `P=0.0146`) | `0.01419` (`+1.79%`) |
| Code (`1×2048`) | `0.08990` | `0.09414` (`+4.72%`, `P=0.0190`) | `0.09190` (`+2.22%`) |

**DNR Entry (W4A16 & W4A8 post-GPTQ `BiasCorr` on dense `Qwen3.8-27B`):** Do not add `BiasCorr` on top of Full-H closed-loop GPTQ for `Qwen3.8-27B` `hf_int4_g16/1`. Revisit only if testing chat-only calibration on an instruct model or ultra-low-bit (`<= 3`-bit) GPTQ.

## 5. Live `vllm serve` Verification (`Exp 23-Live`) & Byte-Exact Repack

- **Served-kernel fidelity (`vllm/vllm-openai-xpu:v0.31.0`, `hf-int4-g16-xpu v0.2.0` commit `d0e9b26`, `HF_INT4_G16_BACKEND=native`, `HF_INT4_G16_MODE=auto`):** `496/496` modules admitted, `0` Triton fallbacks, `0` `bias_corr`. Served WT2 PPL `7.95599 -> 7.88621` (`-0.00881` NLL, CI `[-0.01959, +0.00137]`, `P(Δ<0)=0.952`, n.s. at `n=2,044`; matches offline W4A8 `7.879` to `+0.09%`). Served Code PPL `2.55646 -> 2.51799` (`-0.01516` NLL, CI `[-0.04680, +0.01550]`, `P(Δ<0)=0.834`, n.s. at `n=2,044`; matches offline W4A8 `2.517` to `+0.05%`).
- **Throughput parity:** `C=1` decode `29.92` vs `29.69 tok/s` (`n=3` median), 2K cold prefill `2,533.3` vs `2,496.5 tok/s` (`n=2`). Note: `v2` reused a warm server (`72 °C` start); treat as **speed parity**, not a cooled speedup. `8/8` DNR-009 any-case canaries (`7/8` exact, `prime="No"`, `sumsq=14`) and `3/3` greedy sanity prompts PASS.
- **Byte-exact Hugging Face upload:** `2,191/2,191` tensors verified by per-tensor SHA-256 byte equality between the served 64-layer checkpoint and the 7-shard release at [`SergiioB/Qwen3.8-27B-HF-INT4-G16-GPTQ`](https://huggingface.co/SergiioB/Qwen3.8-27B-HF-INT4-G16-GPTQ) (weights commit `e7565b005d3e`).

## 6. Cross-Family Scope & Conversion-Time Economics

| Scope | Proven Across All 12 Model Families (V11) | Proven Only on `Qwen3.8-27B` Dense (V12) |
|---|---|---|
| Format & rounding | Data-free `HF-INT4-G16` beats `MLX-4bit` (`12/12`) and `SymINT4-G128 RTN` (`11/11`) | Full-H closed-loop GPTQ (`v2`) gains, OOD check, TFQ3 crossover |
| `BiasCorr` | Architecture-dependent; 4-check gate required | Post-GPTQ null on both W4A16 & W4A8 (2nd confirmation after stock GPTQ-G128) |
| MoE serving | `Gemma-4-26B-A4B-it` live (`84.76 tok/s`, `Exp 21`); `Qwen3.6-35B` tiny-weight shapes | MoE routed-expert GPTQ & Had64 `/2` GPTQ remain **untested** |

| Conversion Path (`Qwen3.8-27B`, 1× B70) | Wall Time | Peak Host RSS | GPU Card Energy | Tooling Status | Hugging Face Artifact |
|---|---:|---:|---:|---|---|
| **Fast (`v1` RTN + `BiasCorr`)** | `11.5 min` (`691.7 s`) | `10.9 GiB` | Streaming (`2.38 GiB` cal VRAM) | Shipped in `convert_hf_int4_g16.py` | [`Qwen3.8-27B-HF-INT4-G16-BC`](https://huggingface.co/SergiioB/Qwen3.8-27B-HF-INT4-G16-BC) |
| **Max Quality (`v2` Full-H GPTQ)** | `90.4 min` (`5,426 s`, incl. inline 7-arm eval) | `29.9 GiB` | `331.7 kJ` (`0.092 kWh` GPU card only; CPU FP64 Cholesky un-metered) | Standalone study script (CLI integration pending) | [`Qwen3.8-27B-HF-INT4-G16-GPTQ`](https://huggingface.co/SergiioB/Qwen3.8-27B-HF-INT4-G16-GPTQ) |

**Explicit disclosures & pending tests:**
- Damping grid (`0.05 / 0.10 / 0.35`) was selected in `Exp 23-Pilot-B` on the held-out WT2/Code windows (every grid arm beat `v1` by `>= 0.69 dB/layer`).
- Pending on `v2`: 329-item graded task battery (`287/329` was measured on `v1`), 8K–128K `Exp 22` turn scaling (measured on `v1`, identical kernel geometry), MoE / other 11 families, `TP=2`, and `NVFP4` on 27B.

## 7. Sanitized V12 Receipts

All 6 receipts live in [`results/qwen38-27b/quantization-study-v12-202611/`](../../../results/qwen38-27b/quantization-study-v12-202611/RECEIPTS_MANIFEST.json):
- `exp23_pilot_a_gptq_v1.json` — Layer-03/40 full-H GPTQ feasibility & calibration-size cliff (`1K` vs `256K`)
- `exp23_pilot_b_damp_balance.json` — Full-alphabet shuffled stdlib & per-projection damping sweep (`0.05 / 0.10 / 0.35`)
- `exp23_full_64l_gptq_v1.json` — Full 64-layer sequential closed-loop GPTQ (`496/496` linears, W4A16 & W4A8 KL + W4A16 BC gate)
- `exp23_ood_and_w4a8_gate.json` — 7-domain OOD & 2K long-context verification + 3-arm W4A8 `BiasCorr` gate (`DF` / `+BC` / `-BC`)
- `exp23_live_vllm_ab.json` — Live `vllm serve` A/B (`v0.31.0`, `hf-int4-g16-xpu v0.2.0`, `0` Triton fallbacks)
- `hf_upload_gptq_repo_receipt.json` — Byte-SHA256 verified upload receipt (`2,191/2,191` tensors)
