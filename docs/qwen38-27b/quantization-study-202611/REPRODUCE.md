# Reproducing Fused-Epilogue Pos0-Excluded BiasCorr on Intel Arc Pro B70

This bundle contains everything required to reproduce our Qwen3.8-27B-MLX-4bit Pos0-excluded Nagel BiasCorr study, build the patched TensorFold native XPU server (`pr_check.py`: 0 errors, 0 new lean violations), export and KL-gate the published ~15 MiB FP32 sidecar, export the IEEE-754 `-0.0f` bit-identity control, and run the verification checks. The depth benchmark and 329-item battery scripts in section 5 are lab tools and are not part of this bundle.

## 1. Prerequisites

- **Hardware:** 1x or 2x Intel Arc Pro B70 (Xe2 Battlemage, 32 GB VRAM, 230 W equal card power cap).
- **Checkpoints (local copies):**
  - BF16 reference: `Qwen3.8-27B` (BF16 safetensors) -> `$BF16_DIR`
  - Shipped 4-bit MLX checkpoint: `Qwen3.8-27B-MLX-4bit` (group_size=64 affine uint4) -> `$MLX_DIR`
  - Optional DFlash2 speculative drafter: `Qwen3.8-27B-DFlash2` -> `$DRAFTER_DIR`
- **Build toolchain:**
  - Zig `0.17.0` (`$ZIG_BIN`)
  - Intel `ocloc` (Level Zero / Compute Runtime)
  - oneDNN nGEN headers (`$NGEN_INCLUDE`, e.g., `onednn/third_party/ngen`)
  - TensorFold source checkout at base commit `2286abc0c45c284dae60a9286ae02612eda3b44c`
- **Python environment (for sidecar export):** PyTorch (`xpu`, `cuda`, or `cpu`), `transformers`, `safetensors`, `numpy`, `pyarrow` (optional, for `.parquet` calibration files), and `triton` (optional on GPU; automatic vectorized PyTorch fallback on CPU).

## 2. Export and Gate the Published Sidecar (`qwen38_27b_mlx4_biascorr_v1`, about 7 min on 1x B70)

This is the command that produced the published file. It ran on one B70 (230 W cap, 2,800 MHz max) inside `vllm/vllm-openai-xpu:v0.31.0` (`torch 2.14.0+xpu`, `transformers 5.17.0`, `triton 3.8.0+xpu`, `pyarrow 25.0.1`, Python 3.12.3).

```bash
python3 export_biascorr.py \
  --bf16-dir "$BF16_DIR" \
  --mlx-dir "$MLX_DIR" \
  --device xpu \
  --out ./qwen38_27b_mlx4_biascorr_v1.safetensors \
  --cal-label wt2train_tail_pydecimal_2048 \
  --seq-len 512 \
  --prose-files "$WT2_TRAIN_PARQUET" --prose-seqs 2 --prose-from-tail --prose-tail-offset 4096 \
  --code-files "$PY312/_pydecimal.py" --code-seqs 2 \
  --eval-prose "$WT2_TEST_PARQUET" \
  --eval-code "$PY312/difflib.py" --eval-seqs 4 \
  --kl-gate
```

`$PY312` is `/usr/lib/python3.12` inside that image. The WikiText-2 files are `Salesforce/wikitext`, config `wikitext-2-raw-v1`.

| Input | Identity |
|---|---|
| BF16 reference | `Qwen/Qwen3.8-27B`, revision `8c0b3d30f3ae96f5c7f774de704bf0ada83e422a` |
| MLX checkpoint | `TensorFold/Qwen3.8-27B-MLX-4bit`, revision `22d8d538154e0e5b6f8dcb1fcc74a60a0798104c` |
| WikiText-2 train parquet SHA-256 | `e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7` |
| WikiText-2 test parquet SHA-256 | `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91` |
| `_pydecimal.py` SHA-256 | `c7108f0dc0beedb245daede78c9be602a64d47d3fb988e166e0474aa5bf317ea` |
| `difflib.py` SHA-256 | `0c6afc23568d55b3e9ac914f9c5361e3033e778aa5b58d3cc82835fc5c638679` |

Result: exit code `0`, `status: "PUBLISHABLE"`, 496 tensors, 15,678,520 bytes.

| Held-out domain | KL no correction | KL with `+b` | 95% blk32 CI of the change | KL with `-b` | Shared 6-grams |
|---|---:|---:|---:|---:|---:|
| WikiText-2 test | 0.05060 | 0.03850 (-23.9%) | [-0.01667, -0.00806] | 0.08791 | 0 |
| `difflib.py` | 0.18407 | 0.15787 (-14.2%) | [-0.03717, -0.01556] | 0.29898 | 0 |

- `sidecar_sha256`: `f5fce21b31f477229cab3e62cb42d67094983fd0185e7fc6e363df1ad38ed932`
- `cal_token_ids_sha256`: `e10f3445897905b71d15829bf8632e43d2f47f8261d5b2026ad5730d6631f269`
- The calibration windows were chosen before any KL run, by 6-gram overlap count only. The head of the train split shares 2 6-grams with the prose eval windows, and the study's three code files share 1 with the code eval windows.
- A re-export gets a different file hash even when every tensor is identical, because `safetensors` writes the header metadata keys in a random order. Compare `cal_token_ids_sha256` and tensor values instead. A second export of this command has not been run.

### HF-INT4-G16 sidecar (stored checkpoint tensors)

`gate_hfg16_sidecar.py` copies the 496 `.bias_corr` tensors out of the `HF-INT4-G16-BC` checkpoint, checks them byte for byte, and applies the same gate. It needs `export_biascorr.py` beside it and the `hf_int4_g16_xpu` package (a separate plugin, not part of this bundle) on `PYTHONPATH`.

```bash
python3 gate_hfg16_sidecar.py \
  --bf16-dir "$BF16_DIR" \
  --ckpt-dir "$HFG16_DIR" \
  --device xpu \
  --out ./qwen38_27b_hfg16_biascorr_v1.safetensors \
  --train-parquet "$WT2_TRAIN_PARQUET" \
  --eval-prose "$WT2_TEST_PARQUET" \
  --eval-code "$PY312/difflib.py"
```

Result on one B70: exit code `0`, `status: "PUBLISHABLE"`, 496 tensors, 15,678,488 bytes, `sidecar_sha256` `afafdc60de19310990da8d6dea0fd3f24f05e79c518643dd6c9234669034d9a8`. The script rebuilt the converter's calibration tokens to its recorded hash (`c14de8789c52ed86...`), matched all 703 copied tensors byte for byte with BF16, and decoded layer 0 `gate_proj` at 22.1292 dB SQNR (the `exp15a` value).

| Held-out domain | KL no correction | KL with `+b` | 95% blk32 CI of the change | KL with `-b` | Shared 6-grams |
|---|---:|---:|---:|---:|---:|
| WikiText-2 test | 0.03196 | 0.02708 (-15.3%) | [-0.00741, -0.00280] | 0.05727 | 0 |
| `difflib.py` | 0.14905 | 0.12897 (-13.5%) | [-0.02962, -0.01151] | 0.25463 | 0 |

The `+b` values equal the `exp15a` as-shipped arm to 8 decimals (0.02707786 and 0.12897424). The no-correction values differ from the `exp13a` data-free arm (0.03163 and 0.14920) by up to 1%. The cause has not been traced.

## 2B. Predecessor Sidecar `5719cfb4…` (Historical, Not Published)

The live TensorFold latency, `11/11` determinism and 329-item results (`exp12a/b/c`) used an earlier sidecar, `mixed_wt2_code_2048`:

- `sidecar_sha256`: `5719cfb4ec2978b8c6e201a587dc81b9a3871e6931168cbfb33e0b5872c74b79`
- `cal_token_ids_sha256`: `1dfe706cdbc11628...`

It is not published. Its prose calibration came from the WikiText-2 **test** split, which breaks the train-split rule, and the `exp13a` receipt records 1 shared 6-gram with the code eval text. Its export command is left out on purpose. Use the section 2 command (WikiText-2 **train** split and `_pydecimal.py`) for every new export.

Sidecar layout, for both files: 496 FP32 1D tensors named `language_model.model.layers.{0..63}.{module}.bias_corr` (MLX file), with calibration provenance in the safetensors header metadata.

### Automatic KL pass/fail gate

With `--eval-prose` and `--eval-code`, the gate runs after the export and `--kl-gate` is on by default. Use `--no-kl-gate` to skip it.

Eval windows come from the head of each text. The WikiText-2 test parquet gives the study's frozen prose eval tokens (`14049ba0a85c3ba6...`). `difflib.py` alone gives the same 2,048 code eval tokens (`6c3a613f281669f2...`) as the study's `difflib.py`, `inspect.py`, `argparse.py` list. The export stops if an eval window equals a calibration window, and `.meta.json` records the shared token 6-grams.

The gate reloads the saved sidecar and runs four streams through all decoder layers: BF16, MLX, MLX with `+b`, and MLX with `-b`. It marks the sidecar `PUBLISHABLE` only if all three conditions hold:

1. Prose KL goes down, and the whole 95% block-32 bootstrap CI is below 0.
2. Code KL goes down, and the whole 95% block-32 bootstrap CI is below 0.
3. The sign-flipped `-b` control raises mean KL on both prose and code.

| `status` in `.meta.json` | Meaning | Exit code |
|---|---|---:|
| `PUBLISHABLE` | All three conditions hold | 0 |
| `REJECTED_BY_KL_GATE` | At least one condition failed. `kl_gate.failed_checks` names it | 3 |
| `GATE_PENDING` | The gate started and did not finish | non-zero |
| `UNGATED` | The gate did not run | 0 |

The sidecar file is written in every case, so check `status` before you ship it. `.meta.json` is written next to `--out` unless you set `--meta-out`. To apply the same rule to another evaluator, call `export_biascorr.gate_sidecar(kl)` with per-position KL arrays of shape `(windows, positions)`.

The gate has run on the 27B model on a B70 (section 2). Its 12 unit and end-to-end tests run on CPU with a tiny synthetic checkpoint. All 12 pass with `transformers` 5.17.0. With 5.15.0, two end-to-end tests fail: the tiny model's code CI upper bound lands just above 0.

```bash
python3 test_export_biascorr_gate.py -v   # CPU only; needs torch, transformers with Qwen3_5, safetensors, tokenizers
```

### Export the IEEE-754 `-0.0f` Bit-Identity Control Sidecar (<2 s)

```bash
python3 export_biascorr.py \
  --bf16-dir "$BF16_DIR" \
  --mlx-dir "$MLX_DIR" \
  --out ./qwen38_mlx4_zerobias_496L.safetensors \
  --zero-bias
```

Why `-0.0f` instead of `+0.0f`? In IEEE-754 round-to-nearest-even, `(-0.0f) + (+0.0f) = +0.0f` (flips sign bit), whereas `x + (-0.0f) == x` bit-for-bit for **every** finite floating-point value including both `+0.0f` and `-0.0f`.

## 3. Apply the Fused-Epilogue Patch and Build TensorFold

```bash
# The base commit is on branches of this fork.
git clone https://github.com/SergiioB/TensorFold.git tensorfold
cd tensorfold
git checkout 2286abc0c45c284dae60a9286ae02612eda3b44c
git apply /path/to/tensorfold-biascorr-fused-epilogue.patch

# Verify repository hygiene gate (0 errors, 0 new lean violations)
python3 tools/pr_check.py --base HEAD

# Build native XPU server for Battlemage (Arc Pro B70)
"$ZIG_BIN" build xpu-native -Doptimize=ReleaseFast -Dngen-include="$NGEN_INCLUDE" -Dxpu-device=bmg
```

*(Note: on hosts with `ocloc 26.05.1` that reject `-cl-intel-128-GRF-per-thread`, apply your host's local `zig/build/xpu.zig` flag adjustment separately; it is intentionally excluded from `tensorfold-biascorr-fused-epilogue.patch`.)*

## 4. Run the Server (Single-Card or Dual-Card `--tp 2`)

The commands below load the published sidecar from section 2. The `exp12a/b/c` live results used the predecessor sidecar (section 2B) on patch binary `62fb3828…` with these flags.

**Binary revisions:** the patch v5 build passed the `363,264/363,264` FP64 oracle and `6/6` fail-closed negatives (oracle and five negatives in `xpu-qwen27` `8d079ccd…`, Gemma 4 negative in `tensorfold-native` `75c15aba…`). Both were built from the patch plus the local `zig/build/xpu.zig` workaround. `exp12a/b/c` were not rerun on v5.

**Published file load check (1 card, 2026-11-25, `receipts/tf_v1_published_sidecar_load_check.json`):** the patch v5 server (`75c15aba…`) loaded `qwen38_27b_mlx4_biascorr_v1.safetensors` with `matched=496/496` and the SHA-256 from section 2. On 3 greedy prompts (`temperature=0, top_p=1, seed=4242`, 256 max tokens, thinking off, context 8192, DFlash2 drafter), drafted and serial text matched by SHA-256 on 3 of 3. Speed, the 329-item battery and `--tp 2` were not measured with this file.


### Single-Card (e.g., GPU0 or GPU1)

```bash
ZE_AFFINITY_MASK=0 \
XPU_ATTN_PREF=256 \
XPU_BIASCORR_FILE=./qwen38_27b_mlx4_biascorr_v1.safetensors \
./zig-out/xpu/bin/tensorfold-native serve "$MLX_DIR" \
  --host 127.0.0.1 --port 8036 \
  --drafter "$DRAFTER_DIR" \
  --context 139264 --parallel 1 \
  --prompt-cache-gib 0 --no-thinking --no-update-check
```

### Dual-Card Tensor Parallel (`--tp 2`)

```bash
XPU_ATTN_PREF=256 \
XPU_BIASCORR_FILE=./qwen38_27b_mlx4_biascorr_v1.safetensors \
./zig-out/xpu/bin/tensorfold-native serve "$MLX_DIR" \
  --tp 2 \
  --host 127.0.0.1 --port 8039 \
  --drafter "$DRAFTER_DIR" \
  --context 32768 --parallel 1 \
  --prompt-cache-gib 0 --no-thinking --no-update-check
```

Startup fail-closed verification line expected on stderr:
```text
biascorr sidecar loaded: ./qwen38_27b_mlx4_biascorr_v1.safetensors (matched=496/496, expected=496, sha256=f5fce21b31f477229cab3e62cb42d67094983fd0185e7fc6e363df1ad38ed932)
```

If any sidecar tensor has an invalid suffix, wrong dtype (`!= F32`), mismatched shape, or goes unused during model load, the server refuses to start (`EmptyBiasCorrSidecar`, `BadBiasCorrSidecar`, `UnexpectedShape`, or `UnusedBiasCorrTensor`).

## 5. Verification Checklist

1. **Bit-identity null control (`-0.0f`):** Run greedy prompts (`temperature=0, top_p=1, seed=4242`) with `XPU_BIASCORR_FILE` unset vs set to `qwen38_mlx4_zerobias_496L.safetensors`. Output SHA-256 digests must match **100%** in both `drafted` and `serial` (`"draft": false`) modes. `exp12b` measured `22/22` on GPU0 and with `--tp 2`. This compares the patched binary with itself, not with an unpatched build.
2. **Speculative exactness:** With the sidecar loaded, `drafted` and `serial` greedy outputs must match **100%** SHA-256. Measured: `3/3` with `qwen38_27b_mlx4_biascorr_v1` (one card), `11/11` with the predecessor on GPU0 and with `--tp 2`.
3. **Arithmetic and primality canary (8 items):** In `exp12b` the baseline scored `7/8` (it answered `"yes"` to `"Is 91 a prime number?"`) and the predecessor sidecar scored `8/8` (`"no"`). The published file has not been scored on this canary.
4. **Publishable depth speed ladder:** Run `depth_bench.py run` with `--sampling qwen38-27b:nothink-p0 --depths 0,8k,32k,64k,128k --reps 3` on a single idle 230 W card, then confirm:
   ```bash
   python3 check_publishable.py --max-context 139264 --min-reps 3 cells.jsonl
   # Expected exit code 0: PASS
   ```
5. **329-item frozen quality battery:** Run `battery.py` against the baseline and sidecar servers (`temperature=0, seed=4242, enable_thinking=False`) and grade both in a `--network none` container using `grade.py`.
