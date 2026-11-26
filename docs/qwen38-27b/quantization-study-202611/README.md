> Superseding generation: see [Quantization Deep Study V11](../quantization-study-v11-202611/README.md) (12 model families, live multimodal Gemma-4 MoE, 2-bit rescue, rotated KV vs TurboQuant, Bonsai-2, Spanish & 4K confirmations, quiet-host Gate 5).

# Fixing 4-Bit LLMs After Quantization: 15 MiB Bias Sidecars, Trellis vs Scalar Limits, and Live Xe2 W4A8 Serving on Intel Arc Pro B70

**Date:** November 25, 2026
**Hardware tested live:** Dual Intel Arc Pro B70 (`bmg_g31`, Xe2, 32 GiB GDDR6 each, 32 Xe-cores, 256 XMX engines, `608 GB/s` peak pin bandwidth, `540.8 GB/s` peak copy rate measured (`receipts/exp1_roofline.json`), `230 W` board power cap on both cards).
**Code review status:** Both the out-of-tree `vLLM-XPU` plugin (`hf_int4_g16_xpu`, `130/130` tests passing) and the TensorFold fused FP32 bias-epilogue patch (`pr_check.py` on the build tree: `0` errors; `363,264/363,264` independent host FP64 zero-input exact checks) received code sign-off (`APPROVED WITH DOCUMENTED SCOPE`) after multiple adversarial review rounds.

---

## Files in This Folder

This is a research study. It adds no rows to the [public benchmark catalog](../../../data/benchmarks.v1.json). The family hub is [Qwen3.8-27B](../README.md).

| File | What it is |
|---|---|
| [AGENT-GUIDE.md](AGENT-GUIDE.md) | When to use a sidecar, when to refuse, and how to apply and serve |
| [REPRODUCE.md](REPRODUCE.md) | Export and gate commands, input hashes, TensorFold build and checks |
| [export_biascorr.py](export_biascorr.py) | MLX-4bit sidecar exporter with the 3-check KL gate |
| [test_export_biascorr_gate.py](test_export_biascorr_gate.py) | 12 CPU tests for the gate (all pass with `transformers` 5.17.0) |
| [gate_hfg16_sidecar.py](gate_hfg16_sidecar.py) | Extracts and gates the corrections stored in an HF-INT4-G16-BC checkpoint (needs the plugin package) |
| [tensorfold-biascorr-fused-epilogue.patch](tensorfold-biascorr-fused-epilogue.patch) | TensorFold patch ([SHA-256](tensorfold-biascorr-fused-epilogue.sha256)) |
| [Card 1](cards/card1_biascorr_sidecar_1600x900.png), [Card 2](cards/card2_vllm_hfg16_b70_1600x900.png) | 1600x900 result cards |
| [receipts/](receipts/RECEIPTS_MANIFEST.json) | 27 sanitized JSON receipts and 3 plugin test outputs, with raw and public SHA-256 in the manifest |

### Claim Limits

- The live TensorFold latency, `11/11` `drafted == serial` and 329-item results used the predecessor `mixed_wt2_code_2048` sidecar (`5719cfb4…`) on patch binary `62fb3828…`. The `exp13a` receipt records 1 shared 6-gram between its calibration and the code eval text.
- With the published MLX file, only loading (`496/496`) and `drafted == serial` on 3 prompts were checked live in TensorFold ([receipt](receipts/tf_v1_published_sidecar_load_check.json)).
- The TensorFold patch is not merged upstream, and there is no Apple MLX/Metal loader.
- The depth benchmark, the task battery, its grader and the raw generations are not in this folder.
- The two sidecar binaries ([Table 2B](#table-2b-the-two-shipped-sidecar-files-live-3-check-kl-gate-one-b70)), the `HF-INT4-G16-BC` checkpoint and the `hf_int4_g16_xpu` plugin are not in this repository.

---

## 1. Plain-English Summary: What Worked and What Failed

Over 48 hours of live hardware experiments and multi-model code reviews, we tested how far 4-bit LLM quantization can go on Intel Arc Pro B70 (`Xe2`) and what generalizes beyond a single model or GPU. Half of our ideas failed when we ran real baselines, multi-family checks, and live serving comparisons. Both the wins and the failures are reported below.

### What Survived Live Serving and Real Controls

1. **A 15 MiB post-hoc bias sidecar cuts next-token KL drift on `Qwen3.8-27B` without touching the 15 GB weights (`[OFFLINE-CAUSAL + LIVE-B70]`).**
   Two sidecar files passed the live 3-check causal KL gate on one B70 with `0/0` shared 6-grams. The train-split `MLX-4bit` sidecar (`f5fce21b31f477229cab3e62cb42d67094983fd0185e7fc6e363df1ad38ed932`) cuts KL vs BF16 by **23.9% on prose (`0.05060 -> 0.03850`) and 14.2% on code (`0.18407 -> 0.15787`)**. The `HF-INT4-G16` sidecar (`afafdc60de19310990da8d6dea0fd3f24f05e79c518643dd6c9234669034d9a8`) cuts it by **15.3% on prose (`0.03196 -> 0.02708`) and 13.5% on code (`0.14905 -> 0.12897`)**. On the 400 common decoder matrices of the trellis `TFQ3-4b` format, a correction computed in the test script cut `26.7%` on prose and `19.6%` on code (offline matched-400 measurement only; no runtime loader yet, sidecar not shipped).
   The only loader for the `MLX-4bit` sidecar is a patched TensorFold XPU build on Intel Arc (patch in this release, not merged upstream). There is no Apple MLX/Metal loader yet. With the `f5fce21b…` file, the patched server loaded `496/496` tensors and gave `drafted == serial` on `3/3` greedy prompts. The live latency (`+0.00%` to `+0.41%` TTFT, `0` to `128K`), the `11/11` `drafted == serial` runs (GPU0 and `--tp 2`) and the 329-item run were measured with the **predecessor sidecar `5719cfb4…`**, which was calibrated on the WikiText-2 test tail and shares `1` 6-gram with the code eval text.
   **Binary revisions:** `exp12a/b/c` (329 items, `11/11` GPU0 and TP2, TTFT) ran on patch binary `62fb3828…`. The patch v5 build passed the `363,264/363,264` FP64 independent oracle and `6/6` fail-closed negatives (the oracle and five negatives in its `xpu-qwen27` binary `8d079ccd…`, the Gemma 4 negative in its `tensorfold-native` binary `75c15aba…`), and `75c15aba…` ran the `3/3` check with the `f5fce21b…` file. Both v5 binaries were built from the 17-file patch plus the host-local `zig/build/xpu.zig` workaround, which the shipped patch excludes. `exp12a/b/c` were not rerun on v5.
2. **Packing the output head (`lm_head`) into 4-bit carries a steep, isolated cost (`[OFFLINE-CAUSAL]`).**
   Testing `TensorFold/Qwen3.8-27B-TFQ3@24630b3` on exact BF16 hidden states shows that compressing `lm_head` alone costs **`+0.00679` WT2 KL and `-4.45 percentage points` of Top-1 agreement**. With a BF16 `lm_head`, BF16 residual tensors and a bias correction computed in the test script (offline only; no `TFQ3` sidecar is shipped), `TFQ3` WT2 KL is **42%** lower than as shipped (`0.03206 -> 0.01852`).
3. **A live `vLLM-XPU` plugin using Xe2's 2x INT8 matrix units runs with XPU graphs enabled and beats shipped GPTQ in prefill speed and code fidelity (`[LIVE-B70]`).**
   Intel Xe2 has no native FP4 or FP8 matrix instructions, but INT8 XMX runs at twice the FP16/BF16 rate. Our `hf_int4_g16_xpu` plugin (block-128 Hadamard rotation without online permutation scatter + symmetric progressive group-16 INT4 weights that unpack cleanly into exact INT8) converts `Qwen3.8-27B` in `11.5 minutes` (`691.7 s`, `10.9 GiB` peak RSS, `receipts/hf_int4_g16_conversion_report.json`) and runs live inside `vllm/vllm-openai-xpu:v0.31.0` with XPU graphs ON. Against shipped `GPTQ-Int4-sym-G128` under identical flags (`--max-model-len 36864 --max-num-seqs 96`, interleaved `ABABAB`, `n=6`, model-card sampling `qwen38-27b:nothink-p0`), it delivers **`1.28x` faster 8K prefill (`2,364` vs `1,849 tok/s`)**, **`1.26x` faster 32K prefill (`2,054` vs `1,634 tok/s`, `-10.7%` J/token)**, **less than half the code KL (offline, exact shipped checkpoint: `0.1290` W4A16 / `0.1348` W4A8 vs `0.3020`)**, and **`287/329` vs `283/329`** on a 329-item graded task battery (`p = 0.481`, not statistically significant).
4. **Data-free `Had128NoPerm + HF-INT4-G16` (`4.25 bpw`) beat `MLX-asym-G64` (`4.50 bpw`) and `SymINT4-G128` on KL in 7 of 7 tested model families (`[OFFLINE-CAUSAL]`).**
   Across `Qwen3.8-27B`, `OLMo-2-7B`, `Llama-3.2-3B`, `Phi-4-mini`, `Qwen2.5-1.5B`, `SmolLM2-1.7B`, and 64-expert MoE `OLMoE-1B-7B`, it achieved lower KL than `MLX-asym-G64` at fewer bits per weight, and lower KL than `SymINT4-G128`, which uses slightly fewer bits (on `Qwen3.8-27B` the two comparators are the shipped `MLX-4bit` and `GPTQ` checkpoints). On real 160-item graded tasks (`GSM8K + MMLU + HumanEval`), it beat `MLX-4bit` by `+14` items on `OLMoE-1B-7B` (`82/160` vs `68/160`, `p = 0.024`) and by `+12` items on `Phi-4-mini` (`107/160` vs `95/160`, `p = 0.050`, not significant). The order reversed on `OLMo-2-7B`, where `MLX DF` beat `HF-G16 DF` `99/160` vs `89/160` (`p = 0.087`).

### What Failed, Was Falsified, or Carries Clear Trade-Offs

1. **Open-loop `BiasCorr` is NOT universal.** It lowered KL on `Qwen3.8-27B` (`MLX-4bit`, `HF-INT4-G16`, `TFQ3-4b`), but made calibrated `GPTQ` prose KL **`+5.7%` worse** and regressed `SmolLM2-1.7B` (`+18%` to `+51%`), `Llama-3.2-3B` (`+17%` to `+43%`, `-9/160` task items), `Phi-4-mini` (`+3%` to `+24%`, `-5/160` task items), and Post-Norm `OLMo-2-7B` (`+92%` to `+211%` KL). Because of this, we built a mandatory 3-check automatic KL gate into `export_biascorr.py` and refuse to publish ungated sidecars.
2. **Cross-domain Wiener-damped `AffineCorr` (`Exp 16`) did not solve the small-model failure.** Adding closed-form per-row gain and cross-domain bias shrinkage eliminated `NaiveBC`'s prose regression on `Qwen3.8-27B` `GPTQ` (by auto-damping median `gamma` to `0.028`). The bias-shrinkage-only arm `WienerBC_proj` (no gain) edged `NaiveBC` on `HF-INT4-G16` (`-3.9%` WT2 / `-2.0%` Code, 95% CI only, did not survive Bonferroni correction across 154 contrasts). On `SmolLM2-1.7B`, every bias-bearing arm still hurt KL (`+5.7%` to `+56%`; the Wiener arms `+8%` to `+46%`). Only raw clipped `GainOnly` (no bias) helped `NVFP4-4/6` on `SmolLM2-1.7B` (`-9%` WT2 / `-10%` Code). We killed `WienerAffineCorr` rather than shipping extra complexity.
3. **Scalar 4-bit hits a hard compression wall behind trellis coding (`Exp 15A`, `Exp 15C`).** On the 400 common `Qwen3.8-27B` decoder matrices, Ash's trellis `TFQ3-4b` (`4.001 bpw`) beats scalar `HF-INT4-G16` (`4.260 bpw`) in both prose (`0.0253` vs `0.0318` data-free KL) and code (`0.1104` vs `0.1480`). Zero-extra-bit activation-weighted coordinate descent (`ActMSE-CD`) narrowed the as-shipped code gap (`0.1255` vs `0.1133`, 95% CI crosses zero), and even spending `+1.01 GiB` (`4.62 bpw`) keeping top-8 `down_proj` layers and 96 small projections in BF16 stopped at `0.1218`. Trellis coding owns pure 4-bit quality-per-bit; scalar G16's advantage is live Xe2 INT8 DPAS serving today.
4. **Live `vLLM-XPU` decode is slower than stock G128 GPTQ (`Exp 14D`).** At `C=1`, `HF-INT4-G16` decodes `10.1%` to `11.8%` slower than `AutoGPTQ / XPUwNa16` (`29.48` vs `33.42 tok/s` at 0K; `26.26` vs `29.20 tok/s` at 32K). At `C=4` and `C=8` concurrent requests, aggregate decode throughput is `29%` and `22%` lower (`69.1` vs `97.9` and `166.8` vs `213.3 tok/s`), though median TTFT stays faster. At `0K`, energy per token is `+13.7%` higher (`7.86` vs `6.91 J/tok`), flipping to `-10.7%` lower at `32K` (`22.94` vs `25.68 J/tok`).
5. **Large KL drops did not produce statistically significant jumps on our 329-item binary task battery.** Shipped `MLX-4bit` with the predecessor sidecar `5719cfb4…` moved `282 -> 284/329` (`+9/-7`, exact McNemar `p = 0.8036`, inside TensorFold's `+-2 to 3` perturbation floor, with `7` of `16` discordant items hitting the `1,536`-token cap). Live `vLLM` `HF-INT4-G16-BC` scored `287/329` vs `283/329` for shipped GPTQ (`+11/-7`, `p = 0.4807`, not statistically significant).
6. **No universal "K is always more sensitive than V" law (`Exp 13B`, `Exp 15B`).** Isolated K-only vs V-only quantization sensitivity reversed by architecture: K was more sensitive on `Qwen2.5-1.5B`, whereas V was more sensitive on `Qwen3.8-27B`, `SmolLM2-1.7B`, `Llama-3.2-3B`, `Phi-4-mini`, `OLMo-2-7B`, and `OLMoE-1B-7B` (`6` of `7` families).
7. **Structural blockers:** `NemotronH` has `2,944` expert `down_proj` matrices with `K = 1856` (`K % 128 = 64`), incompatible with unpadded block-128 Hadamard rotation. Our local `Qwen3.6-35B` BF16 file had all `768` routed expert tensors per layer equal to zero and was excluded as a corrupt reference.

---

## 2. True Causal 64-Layer `Qwen3.8-27B` Quality Scoreboard (`Exp 13A v2`, `Exp 15A`, `Exp 15C`)

### Critical Evaluator Bug Caught by Live Serving Reconciliation
Early offline PyTorch sweeps (`Exp 7b` and first-pass `Exp 13A`) constructed a 1-layer `tiny_cfg` shell, which set `tiny_cfg._attn_implementation = "sdpa"` while leaving `full_cfg._attn_implementation = None`. Passing `attention_mask = None` into `transformers` eager attention caused all 16 full-attention layers to run **bidirectional attention** (peeking at future tokens), producing an artificially low BF16 WT2 perplexity of `2.85`. When live reloaded `vllm serve` returned `7.95` WT2 perplexity on the exact same `2,044` tokens, we traced and fixed the mask bug. (A separate `256`-token WT2 prefix probe gave `7.65` for HF-BC, `7.60` for HF-DF and `7.81` for shipped GPTQ.)

With `full_cfg._attn_implementation = "sdpa"` pinned, independent streamed BF16 matches the saved causal reference to `max_abs_diff = 0.0` (`WT2 PPL = 7.79822`, `Code PPL = 2.37436`), and WT2 perplexity reconciles at **7.94887 live vs 7.94236 offline for the exact served checkpoint (`exp15a`, +0.08%)** across the same `2,044` scored positions (live ran `HF_INT4_G16_MODE=auto`; the offline arm is W4A8). All numbers below use the verified causal evaluator (`2,044` scored WT2 tokens + `2,044` scored Python stdlib tokens, four non-crossing 512-token windows, `0/0` 6-gram overlap with disjoint calibration, paired `blk32` 95% CIs).

### Table 2A: All Real Checkpoints and Formats Side by Side (`[OFFLINE-CAUSAL]`)

| Checkpoint / Variant | Decoder `bpw` | Endpoint (`lm_head`) | WT2 KL | WT2 PPL | WT2 Top-1 | Code KL | Code PPL | Code Top-1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `BF16` reference | `16.00` | BF16 | `0.00000` | `7.798` | `100.00%` | `0.00000` | `2.374` | `100.00%` |
| `FP8` shipped | `8.00` | BF16 | `0.00358` | `7.813` | `97.60%` | `0.01689` | `2.381` | `96.48%` |
| **`TFQ3-4b` matched-400 + our `BiasCorr`** | `4.006` | BF16 | **`0.01852`** | `8.016` | **`94.47%`** | **`0.08879`** | **`2.426`** | **`92.86%`** |
| `TFQ3-4b` matched-400 (no BC) | `4.001` | BF16 | `0.02529` | `8.011` | `93.15%` | `0.11042` | `2.435` | `91.14%` |
| `HF-INT4-G16` shipped W4A16 + `BiasCorr` | `4.260` | BF16 | `0.02708` | `7.947` | `94.33%` | `0.12897` | `2.541` | `91.73%` |
| `HF-INT4-G16` `ActMSE-CD` W4A16 + `BiasCorr` | `4.260` | BF16 | `0.02739` | `7.919` | `94.62%` | `0.12552` | `2.534` | `91.54%` |
| `HF-INT4-G16` shipped W4A8 + `BiasCorr` | `4.260` | BF16 | `0.02761` | `7.942` | `94.47%` | `0.13480` | `2.555` | `91.68%` |
| `HF-INT4-G16` W4A16 data-free, **in-memory** (`exp13a`) | `4.254` | BF16 | `0.03163` | `7.972` | `94.13%` | `0.14920` | `2.639` | `90.02%` |
| `HF-INT4-G16` shipped W4A16, corrections off (`exp15c`) | `4.254` | BF16 | `0.03196` | `7.967` | `94.47%` | `0.14905` | `2.631` | `89.87%` |
| `TFQ3-4b` **as-shipped** | `4.001` | 4-bit TFQ3 | `0.03206` | `8.029` | `91.88%` | `0.11327` | `2.446` | `91.05%` |
| `MLX-4bit` shipped + strict-mixed `BiasCorr`, **in-memory study arm** (not the shipped sidecar) | `4.505` | 4-bit MLX | `0.03897` | `8.090` | `91.44%` | `0.15879` | `2.606` | `88.89%` |
| `GPTQ-Int4-sym-G128` shipped (base) | `4.164` | BF16/FP16 | `0.04942` | `8.119` | `91.44%` | `0.30198` | `2.835` | `85.71%` |
| `MLX-4bit` shipped (base) | `4.500` | 4-bit MLX | `0.05060` | `8.136` | `90.56%` | `0.18407` | `2.633` | `88.36%` |
| `GPTQ-Int4-sym-G128` shipped + `BiasCorr` | `4.169` | BF16/FP16 | `0.05221` | `8.127` | `91.05%` | `0.28701` | `2.806` | `85.81%` |
| *Head isolation:* `BF16` body + `TFQ3` `lm_head` | `16.00` | 4-bit TFQ3 | `+0.00679` | `7.822` | `95.55%` (`-4.45 pp`) | `+0.00327` | `2.387` | `98.43%` (`-1.57 pp`) |

*Important scope note on `TFQ3-4b` as-shipped:* Our shard inspection showed `TFQ3`'s BF16 embedding and small residual tensors are not bit-identical to the local `Qwen3.8-27B-bf16` base. The `matched-400` rows share identical BF16 embeddings, `lm_head`, residuals, and 96 small GDN layers, giving the clean format comparison on `400` common decoder matrices. Both `TFQ3` correction rows are an offline matched-400 measurement only; no runtime loader yet, sidecar not shipped.

### Table 2B: The Two Shipped Sidecar Files (Live 3-Check KL Gate, One B70)

| Sidecar file | WT2 KL (none -> `+b`) | Code KL (none -> `+b`) | Shared 6-grams | Receipt |
|---|---|---|---:|---|
| `qwen38_27b_mlx4_biascorr_v1.safetensors` (`f5fce21b…`) | `0.05060 -> 0.03850` (`-23.9%`) | `0.18407 -> 0.15787` (`-14.2%`) | `0/0` | `receipts/qwen38_27b_mlx4_biascorr_v1.public_meta.json` |
| `qwen38_27b_hfg16_biascorr_v1.safetensors` (`afafdc60…`) | `0.03196 -> 0.02708` (`-15.3%`) | `0.14905 -> 0.12897` (`-13.5%`) | `0/0` | `receipts/qwen38_27b_hfg16_biascorr_v1.public_meta.json` |

The gate scores next-token KL against BF16 in a PyTorch stream. It does not run TensorFold or vLLM kernels.

---

## 3. Live Serving Benchmarks and 329-Item Graded Battery on Intel Arc Pro B70

### Table 3A: Live `vllm serve` A/B (`Exp 14C`, `Exp 14D`, GPU1, `230 W`, XPU Graphs ON, `[LIVE-B70]`)
Identical flags for both arms (`vllm/vllm-openai-xpu:v0.31.0`, `--max-model-len 36864 --max-num-seqs 96 --gpu-memory-utilization 0.90 --no-enable-prefix-caching --trust-remote-code`). Speed measured across 3 interleaved `ABABAB` rounds (`n=6` depth cells, `n=3` concurrency batches) at model-card sampling `qwen38-27b:nothink-p0` (`T=0.7, top_p=0.8, top_k=20`, thinking off, `256` tokens out). Note: stock `--quantization inc` (`VLLM_XPU_INC_WNA16_BACKEND=w4a8`) rejected the shipped `gptq` checkpoint config at startup; vLLM automatically served GPTQ through `AutoGPTQLinearMethod` / `XPUwNa16LinearKernel`.

| Measurement | `vLLM` `HF-INT4-G16-BC` (`auto`) | `vLLM` Shipped `GPTQ` (`XPUwNa16`) | Delta |
|---|---:|---:|---:|
| **0K Prefill TTFT / tok/s** (`n=6`) | `0.1146 s` (`548.4 tok/s`) | `0.1139 s` (`559.3 tok/s`) | `-1.9%` |
| **8K Prefill TTFT / tok/s** (`n=6`) | **`3.3707 s` (`2,364.4 tok/s`)** | `4.3099 s` (`1,849.1 tok/s`) | **`1.28x` faster prefill** |
| **32K Prefill TTFT / tok/s** (`n=6`) | **`15.8417 s` (`2,054.4 tok/s`)** | `19.9200 s` (`1,633.8 tok/s`) | **`1.26x` faster prefill** |
| **0K / 8K / 32K `C=1` Decode** (`n=6`) | `29.48 / 28.38 / 26.26 tok/s` | **`33.42 / 31.95 / 29.20 tok/s`** | **`-10.1%` to `-11.8%`** |
| **0K / 8K / 32K Energy (`J/tok`)** (`n=6`) | `7.861 / 11.110 / 22.936` | `6.912 / 11.028 / 25.683` | `+13.7%` (0K), **`-10.7%` (32K)** |
| **`C=4` Aggregate Decode / Med TTFT** (`n=3`) | `69.06 tok/s` / `2.807 s` *(reps 1-2: `78.73` / `0.267 s`)* | **`97.93 tok/s`** / `3.070 s` *(reps 1-2: `115.81` / `0.525 s`)* | `-29.5%` agg decode, faster TTFT |
| **`C=8` Aggregate Decode / Med TTFT** (`n=3`) | `166.75 tok/s` / **`0.378 s`** | **`213.30 tok/s`** / `0.489 s` | `-21.8%` agg decode, `-22.8%` TTFT |
| **8-Item Exact Canary Battery** (greedy) | **`7/8` exact case, `8/8` any case** (`sumsq=14`) | `6/8` exact case, `7/8` any case (`sumsq=30`) | Fixes `sumsq` failure; `NegZero == DF` `8/8` |
| **329-Item Graded Battery** (no-net Docker) | **`287 / 329`** (`7` 1536-cap hits) | `283 / 329` (`10` 1536-cap hits) | `+4` (`+11/-7`, `p = 0.4807`, not sig.) |

### Table 3B: Cross-Engine 329-Item Graded Battery and TensorFold Depth Ladder (`[LIVE-B70]`)

| Live Served Route (`Qwen3.8-27B`) | 329-Item Score | Split Breakdown (`tune / test / prof / struct / ifeval`) | Pairwise vs `vLLM HF-INT4-G16-BC` |
|---|---:|---|---|
| **`vLLM` `HF-INT4-G16-BC`** (reloaded, GPU1) | **`287 / 329`** | `79/82, 78/82, 40/55, 55/55, 35/55` | baseline |
| `EXL3-4.00` (historical B70 5-run mean, MTP off) | `~287 / 329` | historical mean | Reference point (not a same-run pair) |
| `TensorFold` `MLX-4bit` + predecessor sidecar `5719cfb4…` (`Exp 12A`) | `284 / 329` | `78/82, 77/82, 41/55, 55/55, 33/55` | `+10/-7` (`p = 0.6291`) |
| `vLLM` Shipped `GPTQ-Int4-sym-G128` (GPU1) | `283 / 329` | `79/82, 76/82, 40/55, 55/55, 33/55` | `+11/-7` (`p = 0.4807`) |
| `TensorFold` `MLX-4bit` base (`Exp 12A`) | `282 / 329` | `78/82, 76/82, 40/55, 55/55, 33/55` | `+9/-4` (`p = 0.2668`) |

In TensorFold (`Exp 12C`, predecessor sidecar `5719cfb4…`, binary `62fb3828…`, `0` to `128K`, `n=3` interleaved `ABABAB` on GPU1 at `230 W`, model-card sampling `qwen38-27b:nothink-p0`, both arms passing `check_publishable.py`), fusing `BiasCorr` into the nGEN/OpenCL epilogue added `+0.00%` to `+0.41%` TTFT (`+0.29 s` at `128K` over `129.2 s`) with no statistically detectable decode slowdown. Note: cell start temperatures in `Exp 12C` were `62` to `67 C` (symmetric across arms, above the `55 C` cool target).

**Three limits on the TensorFold A/B runs:**
- In `Exp 12A`, `202` of `329` outputs changed between arms.
- In `Exp 12A`, the baseline ran on GPU0 while the BiasCorr arm ran at the same time on GPU1, so card is confounded with arm (in `Exp 14C`, both vLLM arms ran one after the other on GPU1).
- In `Exp 12B`, TP1 and TP2 outputs differed on `2/22` responses in both arms.

### Table 3C: Same-Session Single-Matrix GPU0 Kernel Microbenchmarks (`kernel_bench_gpu0_v2.json`, `[MICROBENCH]`)
Four rotated cold-L2 weight copies, 20 back-to-back launches per steady repeat, identical host timer across all 5 backends on GPU0 (`230 W` cap):

| Backend | `M=1` Steady Decode (`attn_q / mlp_up / mlp_down`) | `M=4096` Steady Prefill (`attn_q / mlp_up / mlp_down`) |
|---|---|---|
| **`HF-INT4-G16` native** (SYCL/L0 zebin) | `67.0 / 91.7 / 106.4 us` (`446` to `518 GB/s`) | `3,091 / 4,399 / 5,133 us` (`142` to `167 TOPS`) |
| Stock `vLLM` `inc_w4a16` (`_xpu_C`) | `62.9 / 86.3 / 86.2 us` (`516` to `533 GB/s`) | `5,392 / 8,313 / 5,822 us` (`88` to `125 TOPS`) |
| Stock `vLLM` `inc_w4a8` (`_xpu_C`) | `211.1 / 168.4 / 179.2 us` | `2,812 / 3,757 / 4,278 us` (`171` to `194 TOPS`) |
| `oneDNN` `BF16` (`torch.mm`) | `217.6 / 304.4 / 311.9 us` | `3,793 / 5,696 / 5,912 us` (`124` to `136 TOPS`) |
| `oneDNN` `INT8` (`torch._int_mm`) | `125.4 / 177.6 / 184.2 us` | `1,868 / 2,712 / 2,538 us` (`269` to `288 TOPS`) |

*Honest kernel diagnosis:* Hardened `HF-INT4-G16` beats `oneDNN BF16` (`1.15x` to `1.29x`) and stock `inc_w4a16` (`1.13x` to `1.89x`) at `M=4096`. Stock `inc_w4a8`, which has no online `FWHT-128` step, stays `1.10x` to `1.20x` faster at `M=4096`. `oneDNN` `INT8` (dense 8-bit weights, so not a 4-bit footprint) is `1.62x` to `2.02x` faster than `HF-INT4-G16` native at `M=4096`. At `M=1`, stock `inc_w4a16` takes `6%` to `19%` less time. It reads slightly fewer bytes and does not rotate; we did not separate those two causes. The live `vllm serve` `C=1` decode gap (`-10%` to `-12%`) points the same way.

---

## 4. Multi-Family Reproduction and Real Graded Task Scores (`Exp 13B`, `Exp 15B`, `Exp 16`)

Every model below was tested with causal `sdpa` pinned (`max_abs_diff == 0.0` vs direct `model.forward()`, `0/0` 6-gram calibration overlap). On the new `Exp 15B` models, we also ran a frozen 160-item deterministic graded task suite (`60` GSM8K math, `60` STEM MMLU, `40` HumanEval Python execution).

| Model Family & Architecture | `HF-INT4-G16 DF` KL (`WT2 / Code`) | `MLX-4bit DF` KL | `NVFP4-4/6 DF` KL | `SymINT4-G128` KL | `BiasCorr` Effect on KL | Graded Task Score (`BF16 / HF-G16 DF / HF-G16+BC / MLX DF / MLX+BC`) |
|---|---|---|---|---|---|---|
| **`Qwen3.8-27B`** (Hybrid GDN+Attn, QK-Norm) | **`0.0320 / 0.1490`** *(shipped checkpoint)* | `0.0506 / 0.1841` *(shipped)* | n/a | `0.0494 / 0.3020` *(shipped GPTQ)* | **KL cut (`-13.5%` to `-27%`)** on MLX, TFQ3, HF-G16; hurts GPTQ WT2 (`+5.7%`) | 329-item: `287` HF-G16+BC vs `283` GPTQ; MLX `282` base, `284` with predecessor sidecar |
| **`OLMo-2-7B`** (Post-Norm + QK-Norm) | **`0.0731 / 0.0648`** *(beats NVFP4)* | `0.0989 / 0.0822` | `0.0766 / 0.0699` | `0.1979 / 0.1564` | **Hurts badly (`+92%` to `+211%` KL)** | 160-item: `90 / 89 / 93 / 99 / 92` |
| **`Llama-3.2-3B`** (Meta GQA, no QK-Norm) | `0.0781 / 0.0811` | `0.1062 / 0.0995` | **`0.0583 / 0.0586`** | `0.1363 / 0.1438` | **Hurts all formats (`+17%` to `+43%` KL)** | 160-item: `104 / 93 / 84 / 94 / 92` *(BC dropped HF-G16 `-9`)* |
| **`Phi-4-mini`** (3.8B, partial RoPE) | `0.1936 / 0.1899` | `0.2391 / 0.2205` | **`0.1486 / 0.1595`** | `0.2631 / 0.2455` | **Hurts 3/4 formats (`+3%` to `+24%` KL)** | 160-item: `104 / 107 / 102 / 95 / 91` *(HF-G16 DF beat MLX `+12`)* |
| **`Qwen2.5-1.5B`** (Qwen2 GQA, no QK-Norm) | **`0.1472 / 0.1073`** *(beats NVFP4)* | `0.1773 / 0.1801` | `0.1813 / 0.2335` | `0.3273 / 0.2315` | Mixed (helps MLX/SymINT4/NVFP4, hurts HF-G16 `+18%` WT2 / `+47%` code) | KL only (+ live vLLM smoke) |
| **`SmolLM2-1.7B`** (Llama GQA, no QK-Norm) | `0.1804 / 0.1516` | `0.2011 / 0.1830` | **`0.1269 / 0.1223`** | `0.2472 / 0.2373` | **Hurts all formats (`+18%` to `+51%` KL)** | KL only (`GainOnly` helped NVFP4 `-9% / -10%`) |
| **`OLMoE-1B-7B`** (64-expert MoE, QK-Norm) | `0.1312 / 0.0815` | `0.1636 / 0.1053` | **`0.1212 / 0.0783`** | `0.2345 / 0.1687` | Small KL win on HF-G16 (`-3.3% / -7.8%`) and MLX (`-2.2% / -8.7%`), both with `all_experts`; hurts NVFP4 & SymINT4 WT2 | 160-item: `83 / 82 / 77 / 68 / 78` *(HF-G16 DF beat MLX `+14`, `p = 0.024`)* |

*Note on `Exp 15B` quantizers and methodology:* `Llama-3.2-3B`, `Phi-4-mini`, `OLMo-2-7B`, and `OLMoE-1B-7B` compare controlled in-memory `MLX-asym-G64`, `NVFP4-4/6`, `SymINT4-G128`, and `HF-INT4-G16` quantizers against the exact BF16 model, rather than third-party Hub checkpoints. On `Qwen3.8-27B`, every comparator (`FP8`, `GPTQ`, `MLX-4bit`, `TFQ3-4b`, `HF-INT4-G16`) was loaded from a real serialized checkpoint on disk. Across the `20` quantized `Exp 15B` points (`4` models x `5` arms), KL did not reliably predict binary task-score drop (`r = -0.155` pooled; `0.247` after subtracting each model's mean, both from the `20` points stored in `exp15b_multifamily_scores.json`), and only `2` of `20` pairwise task comparisons reached `p < 0.05` (`Llama HF BC` worse than BF16, `p = 0.0012`; `OLMoE HF DF` beating `MLX DF`, `p = 0.024`). On `OLMoE-1B-7B`, correcting routed experts gave a small but significant KL reduction over attention-only correction (`all-experts` and `threshold >= 64` tokens tied within CI). The KL column uses `all_experts` for both formats. The `MLX+BC` task score (`78/160`) was run with `threshold >= 64`. The `SmolLM2-1.7B` row comes from `Exp 16`; the earlier `Exp 13B` run of that family gives `+16%` for `SymINT4` code.

### Key Architectural Takeaways
- **No universal single 4-bit format:** `HF-INT4-G16` beats `MLX-asym-G64` and `SymINT4-G128` across every architecture tested (`7/7`), and beats `NVFP4-4/6` on `OLMo-2-7B` and `Qwen2.5-1.5B`. On `Llama-3.2-3B`, `Phi-4-mini`, `SmolLM2-1.7B`, and `OLMoE-1B-7B`, `NVFP4-4/6` has lower KL (though Xe2 has no native FP4 hardware).
- **Why `BiasCorr` needs a mandatory gate:** The same open-loop correction lowered KL on `Qwen3.8-27B` and raised it on `OLMo-2-7B`, `Llama-3.2-3B`, `Phi-4-mini` and `SmolLM2-1.7B`. We have not isolated the cause. Always run `export_biascorr.py --kl-gate` before publishing any sidecar.

---

## 5. Engineering Verification and Code Audit Trail

Both production artifacts went through four rounds of adversarial forensic code review (catching real bugs including `fp64_ref` reading back device `w.bias`, FP16 tag collisions in `k_proj`, per-launch `1.48 MB` zebin re-hashing, replicated-KV oversized width admission, and one-way shard header checks):

1. **`vllm-hf-int4-g16` (`APPROVED WITH DOCUMENTED SCOPE`):**
   - `130/130` GPU tests pass (`receipts/hf_int4_g16_plugin_pytest_full.txt`). Reverting each of the final 4 fixes makes tests fail (`receipts/hf_int4_g16_plugin_mutation_check.txt`).
   - Zero-copy Level-Zero zebin imported once per `(device, path)` via `admit_module` (verified by deleting the zebin after admission), submitted through PyTorch's current SYCL queue (`sycl::make_kernel_bundle` + `q.submit`), with `XPUCachingAllocator::recordStream` and `XPUGraph` changed-input replay verified across `M in {1, 2, 3, 4, 8}`.
   - Exact FP32 round-to-nearest-even (`rint`) parity between native OpenCL and Triton; true `NULL` vs `-0.0f` bit-identity on both W4A16 and W4A8; two-way safetensors header admission verified `496/496` on the real 27B checkpoint (`receipts/hf_int4_g16_plugin_header_admission.txt`).
   - Explicit out-of-scope guards: packed MoE experts fail closed at admission; live multi-GPU TP=2 vLLM collectives are unbenchmarked.
2. **TensorFold Fused Bias Epilogue Patch v5 (`APPROVED WITH DOCUMENTED SCOPE`):**
   - The shipped patch has `17` files and passes `git apply --check` on pristine `HEAD`. `zig fmt --check` passes. `python3 tools/pr_check.py --base HEAD` ran on the `18`-file build tree (the patch plus the host-local `zig/build/xpu.zig` workaround): `0 errors`, `lean 221 vs 221, 0 new`. It was not rerun on the `17`-file patch alone.
   - Independent host FP64 zero-input oracle (reads raw logical sidecar keys directly, never copies device `w.bias`): `363,264/363,264` exact outputs on both FP16-unique tags and the real `mixed2048` FP32 sidecar (`xpu-qwen27` binary `8d079ccd…`). Swapping adjacent `self_attn.k_proj` channels `0/1` fails closed (`6` mismatches, `BiasCorrZeroInputMismatch`).
   - Production fused-SwiGLU oracle check passes enforced `5.0e-3` bound (`1.58e-3` L0, `1.60e-3` L3); `1e-6`, `nan`, and `inf` bounds fail closed (`FusedSwigluOracleMismatch` / `BadSwigluRelL2Bound`).
   - Fail-closed startup checks verified for `IncompleteBiasCorrCoverage`, `UnusedBiasCorrTensor`, `NonFiniteBiasCorr`, and `UnsupportedBiasCorrFamily` (Gemma 4, `tensorfold-native` binary `75c15aba…`).
- Live check of the shipped `f5fce21b…` sidecar on `75c15aba…`: `496/496` loaded, `drafted == serial` on `3/3` greedy prompts (`receipts/tf_v1_published_sidecar_load_check.json`). Speed, the 329-item battery and `--tp 2` were not measured on v5.

---

## 6. How to Apply and Reproduce

See the files in this folder:
- **[AGENT-GUIDE.md](AGENT-GUIDE.md):** Step-by-step quickstart and refusal rules for humans and coding agents.
- **[export_biascorr.py](export_biascorr.py):** Standalone exporter with the automatic 3-check KL gate (`12/12` CPU tests pass with `transformers` 5.17.0).
- **[tensorfold-biascorr-fused-epilogue.patch](tensorfold-biascorr-fused-epilogue.patch):** 17-file TensorFold patch.
- **[REPRODUCE.md](REPRODUCE.md):** Exact build commands (including `-Doptimize=ReleaseFast`), dataset hashes, and verification steps.
- **[cards/](cards/card1_biascorr_sidecar_1600x900.png):** [card 1](cards/card1_biascorr_sidecar_1600x900.png) and [card 2](cards/card2_vllm_hfg16_b70_1600x900.png).
- **Cross-hardware scope:** Every live timing and energy number in this study was measured on Intel Arc Pro B70 (`Xe2`). The `BiasCorr` math and `lm_head` findings are expected to transfer to Apple Silicon / CUDA / ROCm, and `W4A8` INT8 DPAS mirrors INT8-only GPUs such as NVIDIA Ampere, but non-B70 hardware was not benchmarked here.
