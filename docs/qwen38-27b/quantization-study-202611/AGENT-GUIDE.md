# Agent and Developer Guide: Post-Hoc 4-Bit Bias Sidecars and `HF-INT4-G16` on Intel Arc Pro B70

This guide is written for both humans and coding agents. Follow the rules below before applying a sidecar, generating a new one, or serving `HF-INT4-G16` in `vLLM-XPU`.

## 1. Quick Decision Tree

Pick the right path before running anything:

1. **You already run `Qwen3.8-27B` `MLX-4bit` on TensorFold:**
   Use the **KL-gate-verified 15 MiB `BiasCorr` sidecar** `qwen38_27b_mlx4_biascorr_v1.safetensors` (`f5fce21b…`): live-gated KL vs BF16 is `-23.9%` on prose (`0.05060 -> 0.03850`) and `-14.2%` on code (`0.18407 -> 0.15787`), with `0/0` shared 6-grams. Do not repack the 15 GB weights. It loads today in patched TensorFold on Intel Arc (XPU), PR open; no Apple MLX/Metal loader yet. `TFQ3-4b`: offline matched-400 measurement only; no runtime loader yet, sidecar not shipped. `export_biascorr.py` reads MLX-4bit weights only. For an `HF-INT4-G16-BC` checkpoint, the plugin already applies the stored corrections; its gated copy (`afafdc60…`, `-15.3%` prose / `-13.5%` code) is for A/B tests only.
2. **You run a calibrated checkpoint (`GPTQ`, `AutoRound`, `AWQ`, `EXL3`) or a small dense model (`<= 7B` such as `SmolLM2-1.7B`, `Llama-3.2-3B`, `Phi-4-mini`, `OLMo-2-7B`):**
   **Stop.** Do not apply open-loop `BiasCorr` blindly. Across our 7-family tests it made calibrated `GPTQ` prose KL `+5.7%` worse and increased KL by `+3%` to `+211%` on `SmolLM2-1.7B`, `Llama-3.2-3B`, `Phi-4-mini`, and `OLMo-2-7B`. Only ship a sidecar if it passes the automatic 3-check KL gate below.
3. **You run `vLLM-XPU` (`vllm/vllm-openai-xpu:v0.31.0`) on an Intel Arc Pro B70 (`bmg_g31`, Xe2) and care about long-prompt prefill speed and code quality:**
   Use **`hf_int4_g16_xpu`** (`Had128NoPerm + HF-INT4-G16`). On `Qwen3.8-27B` live `vllm serve` (XPU graphs on), it runs `1.28x` faster at 8K prefill (`2,364` vs `1,849 tok/s`) and `1.26x` faster at 32K prefill (`2,054` vs `1,634 tok/s`) than shipped `GPTQ-Int4-sym-G128`, with less than half the code KL (offline, exact shipped checkpoint: `0.1290` W4A16 / `0.1348` W4A8 vs `0.3020`). Trade-off: `C=1` decode is `10%` to `12%` slower (`29.5` vs `33.4 tok/s`), and compressed MoE experts are not supported yet.

---

## 2. How to Check If a Checkpoint Is Uncalibrated

Check metadata first (10 seconds), then always trust the automatic KL gate over the label.

| Signal in `config.json` or Model Card | Likely Type | Action |
|---|---|---|
| `"quantization": {"bits": 4, "group_size": 64}` (MLX), plain RTN, BitsAndBytes, non-imatrix GGUF, data-free `HF-INT4-G16`, `TFQ3` | Uncalibrated or codebook without output-mean folding | Good candidate. For MLX-4bit, export a sidecar and run `--kl-gate`. For other formats, feed per-position KL from your own evaluator to `export_biascorr.gate_sidecar()`. |
| `"quant_method": "gptq"`, `"auto-round"`, `"awq"`, `quantize_config.json`, `quant_log.csv`, `imatrix` GGUF, EXL2/EXL3 | Already calibrated on text | Usually small gain or prose regression. Do not ship unless `--kl-gate` passes cleanly on held-out prose and code. |

---

## 3. Apply a Verified 15 MiB `BiasCorr` Sidecar in TensorFold

1. Apply `tensorfold-biascorr-fused-epilogue.patch` to TensorFold and build in `ReleaseFast`:
   ```bash
   git apply tensorfold-biascorr-fused-epilogue.patch
   zig build xpu-native -Doptimize=ReleaseFast -Dngen-include=/path/to/onednn/third_party/ngen
   ```
2. Verify the sidecar SHA-256 against its `.meta.json` receipt before loading. For `qwen38_27b_mlx4_biascorr_v1.safetensors` it is `f5fce21b31f477229cab3e62cb42d67094983fd0185e7fc6e363df1ad38ed932`.
3. Serve with `XPU_BIASCORR_FILE` set:
   ```bash
   ZE_AFFINITY_MASK=0 \
   XPU_BIASCORR_FILE=/path/to/qwen38_27b_mlx4_biascorr_v1.safetensors \
   ./zig-out/xpu/bin/tensorfold-native serve /path/to/Qwen3.8-27B-MLX-4bit \
     --drafter /path/to/Qwen3.8-27B-DFlash2 \
     --context 40960 --parallel 1 --no-thinking --no-update-check
   ```
4. **Fail-closed safety checks built into the loader:**
   - Setting `XPU_BIASCORR_FILE` on an unsupported family (for example Gemma 4) aborts at startup with `UnsupportedBiasCorrFamily`.
   - Any missing projection (`IncompleteBiasCorrCoverage`), orphan key (`UnusedBiasCorrTensor`), or non-finite value (`NonFiniteBiasCorr`) aborts at load time.
   - When `XPU_BIASCORR_FILE` is unset, base checkpoint `.bias_corr` tensors are ignored. In `exp12b`, a `-0.0f` sidecar gave output identical to the same patched binary with no sidecar (`-0.0f` == `NULL`), `22/22` on GPU0 and with `--tp 2`. Output was not compared with an unpatched `main` build.

---

## 4. Generate and Automatically Gate a New `BiasCorr` Sidecar

Never upload a sidecar without running the 3-check KL gate. Calibration text must come from train splits and have **zero 6-gram overlap** with both evaluation windows. The exporter stops when an eval window equals a calibration window, but shared 6-grams only print a `WARNING`. Check that both `cal_6gram_overlap` fields in `.meta.json` are `0`.

```bash
python3 export_biascorr.py \
  --bf16-dir /path/to/model-bf16 \
  --mlx-dir /path/to/model-mlx-4bit \
  --prose-files /path/to/wikitext2-train.parquet --prose-from-tail \
  --code-files /path/to/disjoint_python_stdlib.py \
  --eval-prose /path/to/wikitext2-test.parquet \
  --eval-code /path/to/heldout_code.py \
  --out /path/to/output_biascorr.safetensors
```

### Mandatory Pass/Fail Gate Rules (`export_biascorr.py`)
A sidecar is marked `PUBLISHABLE` (exit code `0`) only when **all three checks pass**:
1. Held-out prose (`WT2`) KL drops with 95% block-bootstrap (`blk32`) CI strictly below `0`.
2. Held-out code KL drops with 95% block-bootstrap (`blk32`) CI strictly below `0`.
3. Negated sign-flip control (`-b`) worsens KL on both domains.

If any check fails, `export_biascorr.py` writes `status: "REJECTED_BY_KL_GATE"` in the `.meta.json` receipt and exits with code `3`. Agents must treat exit code `3` as a hard stop and must not publish the sidecar.

---

## 5. Convert and Serve `HF-INT4-G16` in `vLLM-XPU` (Intel Arc Pro B70)

Requires `vllm/vllm-openai-xpu:v0.31.0` (`torch 2.14.0+xpu`, `triton 3.8.0+xpu`, `intel-sycl-rt 2026.1.1`) on an Intel Arc Pro B70 (`bmg_g31`).

### Step 1: Install the out-of-tree plugin
The plugin (`vllm-hf-int4-g16`) is a separate package and is not part of this bundle. `convert_hf_int4_g16.py` ships with it. `--cal-mode mixed_train` also needs the `datasets` package and the `Salesforce/wikitext` dataset.

```bash
pip install --no-deps --no-build-isolation -e /path/to/vllm-hf-int4-g16
```
This registers the `vllm.general_plugins` entry point (`hf_int4_g16`) so both the API server and EngineCore workers discover `@register_quantization_config("hf_int4_g16")` automatically.

### Step 2: Convert a BF16 checkpoint (streaming, bounded RAM/VRAM)
```bash
python3 convert_hf_int4_g16.py \
  /path/to/Qwen3.8-27B-bf16 \
  /path/to/Qwen3.8-27B-HF-INT4-G16-BC \
  --biascorr --cal-mode mixed_train --biascorr-tokens 1024 \
  --max-shard-gib 3.0
```
On one B70, converting `Qwen3.8-27B` takes about `11.5 min` (`691.7 s`), peaks at `10.9 GiB` host RSS and `2.38 GiB` calibration VRAM, and writes `496/496` quantized decoder linears with `BF16` embeddings and `lm_head` (study receipt `hf_int4_g16_conversion_report.json`, in the cookbook study folder).

### Step 3: Serve live with XPU graphs enabled
```bash
HF_INT4_G16_MODE=auto \
HF_INT4_G16_BIASCORR_MODE=default \
vllm serve /path/to/Qwen3.8-27B-HF-INT4-G16-BC \
  --served-model-name default \
  --host 127.0.0.1 --port 8091 \
  --max-model-len 36864 --max-num-seqs 96 \
  --gpu-memory-utilization 0.90 --trust-remote-code
```

### Runtime Environment Knobs
- `HF_INT4_G16_MODE`:
  - unset / `w4a16` (strict default): uniform W4A16 precision across all batch sizes.
  - `auto`: native Xe2 fused-FWHT W4A16 for physical `M in {1, 2, 3, 4}` and native Xe2 INT8 XMX DPAS W4A8 for `M >= 5`. Use this for maximum prefill throughput (`1.28x` at 8K vs GPTQ).
- `HF_INT4_G16_BIASCORR_MODE`:
  - `default`: load serialized `.bias_corr` tensors.
  - `off`: pass true `NULL` pointer to kernels (data-free ablation on the exact same checkpoint).
  - `neg_zero`: load `-0.0f` tensors for bit-identity path checks.

### Known Scope Boundaries (Do Not Violate)
- **Dense models only:** Packed MoE expert keys are rejected at load time (fail-closed).
- **TP > 1:** Row-parallel layers force W4A16 and apply `bias_corr` on rank 0 only. Live multi-GPU TP=2 vLLM collectives have not been benchmarked yet.
- **Mamba/GDN hybrid models:** Set `--max-num-seqs 96` (or lower at longer context) on 32 GiB cards because vLLM's default `256` exceeds Mamba state blocks.
