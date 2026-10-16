# Nemotron 3.5 Bolt 74B-A9B: making the AutoRound W4A16 quant on two B70

[Family index](README.md) · [serving recipe (vLLM TP2)](VLLM-TP2-AUTOROUND-W4A16.md) · [scripts](../../tools/autoround-b70/README.md)

This page is the full process behind the vLLM TP2 route: how the int4 weights
were made, what broke, how each break was fixed, and how the result was
checked. Follow it to redo the export, or to port it to another Nemotron-H
style hybrid. The serving commands themselves live in the
[recipe](VLLM-TP2-AUTOROUND-W4A16.md); this page does not repeat them.

Evidence tier E2: self-reported, one host, two cards, not independently
reproduced. All times are from one run on 2026-10-16.

## 1. Result

Intel AutoRound quantized `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16`
(139.5 GiB) to W4A16: int4, group 128, symmetric, GPTQ packing. The output is
41 GB (40.65 GiB) in 9 shards and is published as
[`SergiioB/Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym`](https://huggingface.co/SergiioB/Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym)
(weights commit `31f574676ecc`). Tuning ran on two Arc Pro B70 cards at once
and took 5 h 57 min. One card was not enough. Three things had to be in place
first: a pure-torch `mamba_ssm` shim, an einsum fix in Bolt's own Mamba2 code,
and five loader fixes that a CPU dry run on a tiny checkpoint found before the
real run. A static check against the BF16 source passed for all 27,920
quantized layers. The model then served on vLLM 0.31.0 XPU with tensor
parallel 2 after three serve fixes.

## 2. What you need

### Hardware

- **Two B70 cards (32 GB each).** AutoRound spreads each layer over both cards
  (`device_map 0,1`). On one card the run died with `DEVICE_LOST` at layer 21
  of 88 after 71 min. The cause was a driver out-of-memory (`VM worker error:
  -12`) on a MoE layer with 432 experts x 2,560 rows. Nine MoE layers are that
  big or bigger, so a single card is not an option at these settings.
- Peak VRAM on two cards: 17.1 GB on card 0 and 24.1 GB on card 1. Card 0 also
  drove the desktop.
- Card power during tuning stayed at 96-137 W, well under the caps. Tuning is
  not power-bound.

### Host RAM and swap

- The host has 60 GB of RAM. The export process grows during the run: 33 GB at
  layer 0, 39 GB at layer 37, 44 GB at layer 79, peak 47.5 GB.
- We added a 32 GB swap file at layer 76, when free memory ran low. **Add it
  before you start.** See step 6 below.

### Disk

- 139.5 GiB for the BF16 source and 41 GB for the output. Put both on the
  same fast disk with room to spare.
- Weights stream from disk one layer at a time, so disk speed matters during
  the export and during the vLLM load. Do not run another disk-heavy job on
  that disk at the same time. A GGUF conversion on the same USB disk cut our
  checker to 1-2 MB/s.

### Software

| Item | Version used |
|---|---|
| Source model | `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16`, commit `a6a1e5030819a130a1ea08aadb1fe42f4cc86777` (main at download time; `modeling_nemotron_h.py` SHA-256 `e1cb5fc02e887983f0a445bf4c1a2604453b2cb2db4624c7004dcf663bbb1b6e`, `model.safetensors.index.json` SHA-256 `25bf04a0a33e49af0d861b3b86d7b6e39b461b95d86daa12be865842954cddee`, both match the local copy) |
| AutoRound | main `08d7085e7ef6c9b8ff5969d74dabe8ae4e730c5d` plus the first two commits of [intel/auto-round#2512](https://github.com/intel/auto-round/pull/2512) (`6daaf7ce`, then `a5ea56e6e74385c3250d0d123699c0f55a72a1a7`) plus [`autoround-a5ea56e6-dryrun-fixes.diff`](../../tools/autoround-b70/patches/autoround-a5ea56e6-dryrun-fixes.diff). Editable install (`0.17.0.dev238+g08d7085e`). |
| AutoRound upstream | PR #2512, head `345e0fe1`, approved, not merged. It contains these fixes in reworked form plus later changes. The Bolt export was **not** re-run on the PR head. |
| Python | 3.12.13 |
| torch | 2.12.1+xpu |
| transformers | 5.12.1 (Bolt's remote code was written for 5.8.1; see fix 3) |
| safetensors / accelerate | 0.8.0 / 1.15.0 |
| intel-sycl-rt (in the venv) | 2025.3.2 |
| Host GPU runtime | compute-runtime 26.05.37020.3 (`intel-opencl-icd`), as installed on the host on 2026-10-16 |
| Serving image | `vllm/vllm-openai-xpu@sha256:95ac815038c4b3537b173798bfe3b89bdf525a52bd653cd14d4deb01f1062be7` (vLLM 0.31.0) |

## 3. Timeline

Local time (UTC+2), 2026-10-16. The speed confirmation ran later the same
evening.

| Time | Duration | Step | Outcome |
|---|---:|---|---|
| 06:21-08:01 | 1 h 40 min | BF16 download, 63 shards + 22 small files | 139.5 GiB. Wired link 10.9 MB/s, wifi 25-30 MB/s. |
| ~07:15-07:42 | ~27 min | CPU dry run on a tiny Bolt-shaped checkpoint, during the download | Found 5 crashes and 1 out-of-memory before the real run (fixes 3-7). |
| 08:01-08:15 | 14 min | Attempt 1, one card | **Lost.** Ran on the CPU by mistake (device selector clash, fix 9). Killed. |
| 08:15-08:17 | 2 min | Attempt 2, one card | **Lost.** Out of memory in layer 0 (Mamba2 backward pass). |
| 08:17-08:29 | 12 min | Einsum fix, checked on the GPU | Peak 25.8 GiB (out of memory) down to 3.0 GiB, same output (fix 2). |
| 08:29-09:48 | 79 min | Attempt 3, one card | **Lost.** `DEVICE_LOST` at layer 21 of 88 after 71 min (fix 10). |
| 09:54-15:52 | 5 h 57 min | Attempt 4, **two cards** | `status=ok`, 0 errors, 21,436 s including the save. |
| 15:53-16:02 | 9 min | Static check, first version | False failures: the checker looked up source names (`backbone`), the artifact uses `model` (fix 12). |
| 16:03-16:09 | 6 min | Checker fixed, re-run | 20 of 21 checks pass. The one left: 120 Mamba `A_log`/`D`/`dt_bias` tensors stored as F32 with the exact BF16 values. The checker now accepts that lossless widening (fix 12). |
| 16:11-16:37 | 26 min | vLLM TP2, serve attempts 1-2 | Both hung at the oneCCL rendezvous between the cards. |
| 16:37-16:59 | 22 min | Worker-affinity patch + CCL thresholds, then loopback pinning | CCL init passes once both ranks use loopback (fixes 13, 14). |
| 17:07-17:24 | 17 min | First full weight load | 40.6 GiB in 1,017 s from the USB disk, 20.0 GiB per card. |
| 17:28 | | First forward pass | `RuntimeError: Unsupported TopK value` (fix 15). |
| 17:35-17:44 | 9 min | Top-k pad patch | Serving. Load gate pass. |
| 17:46-18:10 | 24 min | Gate, exact battery, speed, eager then graph mode | 5/6 gate, 8/8 exact; greedy text identical between eager and graph mode on all 6 gate cases; graph mode decodes 6x faster. |
| 21:24-21:44 | 20 min | Speed confirmation (5 reps per shape, 262K server config) | The published speed table. |

About 1 h 50 min of GPU time was lost (08:01-09:54) to attempts 1-3 and the
einsum fix. Every one of those losses is avoidable with this guide: start on
two cards, with the shim, the einsum patch and the device selector set right.

## 4. Step by step

Variables used below. Pick your own paths.

```bash
export TOOLS=$PWD/tools/autoround-b70          # from the root of this repository
export WORK=/fast-disk/bolt-work                # scratch: logs, AutoRound workspace, tiny model
export SRC=/fast-disk/bolt74b-bf16              # BF16 source checkpoint
export OUT=/fast-disk/bolt74b-autoround-w4a16   # export directory
export CALIB_CORPUS=/fast-disk/calib-corpus     # directory of *.txt.gz calibration text
export PY=/fast-disk/arvenv/bin/python          # python of the export venv
mkdir -p "$WORK"
```

### Step 1. Build the export venv

The exact commands we used to create the venv were not recorded. The package
versions are in the table above. A venv with the same versions:

```bash
python3.12 -m venv /fast-disk/arvenv
$PY -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/xpu
$PY -m pip install transformers==5.12.1 safetensors==0.8.0 accelerate==1.15.0

git clone https://github.com/intel/auto-round "$WORK/auto-round"
git -C "$WORK/auto-round" fetch origin pull/2512/head
git -C "$WORK/auto-round" checkout a5ea56e6e74385c3250d0d123699c0f55a72a1a7
git -C "$WORK/auto-round" apply "$TOOLS/patches/autoround-a5ea56e6-dryrun-fixes.diff"
$PY -m pip install -e "$WORK/auto-round"
```

Check that torch sees both cards. Do not set `ZE_AFFINITY_MASK` (step 7):

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0,1 $PY -c "import torch; print(torch.xpu.device_count())"   # must print 2
```

### Step 2. Download the small files first

The dry run needs only the config, remote code, tokenizer and index. Start it
while the 139.5 GiB of shards download.

```bash
hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16 \
  --revision a6a1e5030819a130a1ea08aadb1fe42f4cc86777 \
  --local-dir "$SRC" --exclude "*.safetensors"
# then, in another shell, the weights (1 h 40 min for us)
hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16 \
  --revision a6a1e5030819a130a1ea08aadb1fe42f4cc86777 --local-dir "$SRC"
```

### Step 3. CPU dry run on a tiny checkpoint

This step found five of the fixes (3-7) in 27 minutes. Each run takes 25-35 s
on the CPU. **Always do it first on a new Nemotron variant.**

```bash
$PY "$TOOLS/build_tiny_bolt.py" --src "$SRC" --out "$WORK/tiny"
PYTHONPATH="$TOOLS/shims" AR_WORK_SPACE="$WORK/ar_ws_tiny" \
  timeout 840 $PY "$TOOLS/convert_nemotron_omni.py" --src "$WORK/tiny" --out "$WORK/tiny-out" \
  --corpus "$CALIB_CORPUS" --bits 4 --iters 2 --nsamples 2 --seqlen 64 --batch-size 1 --device cpu \
  > "$WORK/dryrun.log" 2>&1; echo "exit=$?"
```

The tiny model has 6 layers (2 Mamba, 3 MoE with 8-16 experts, 1 attention),
a 2-layer vision tower, and Bolt's real tensor names. `build_tiny_bolt.py`
prints `ONLY_IN_REAL` and `ONLY_IN_TINY`: name patterns that differ from the
real index. Both should be empty.

A passing dry run ends like this (our log):

```text
nemotron_h: restored 9 tensor(s) at high precision
is_model_patched=False disk_stream_index=yes
Summary: quantized 90/109 in the model, unquantized layers: language_model.lm_head, language_model.model.layers.[0,4].mixer.in_proj, ...
Successfully wrote 36 missing tensor(s) to 'model_extra_tensors.safetensors'
quantization+export wall time: 34s
```

`is_model_patched=False` is expected: the disk-stream path is used instead.
The two tiny `in_proj` layers are skipped only because their shape is not a
multiple of 32; on the real model every `in_proj` is quantized. The 36 missing
tensors must be vision and projector tensors only. If language-model tensors
show up in that list, the name alias (fix 6) is missing and the real run would
copy about 137 GiB of BF16 back in.

### Step 4. The `mamba_ssm` shim

Bolt's remote code imports `rmsnorm_fn` from `mamba_ssm`, a CUDA-only package.
The shim in [`tools/autoround-b70/shims/mamba_ssm/`](../../tools/autoround-b70/shims/mamba_ssm/ops/triton/layernorm_gated.py)
gives the same math as `mamba_ssm`'s `rms_norm_ref` in pure torch (we checked
it bitwise against that reference). Its kernel entry points raise
`NotImplementedError`, so the model takes its torch path. Nothing to install:
put the directory on `PYTHONPATH`. The runner does that for you.

### Step 5. The Mamba2 einsum patch

Bolt's torch path for Mamba2 builds 6-D tensors `(b, c, l, s, h, n)` and sums
them. Forward plus backward on 1,536 tokens peaked at 25.8 GiB and ran out of
memory on one card. Four of those lines become `torch.einsum` calls. Peak
memory dropped to 3.0 GiB. Cosine similarity against the original was
1.000000 for the output, the input gradient and the `in_proj` weight gradient
at 512 tokens.

Save NVIDIA's file first. You must ship the original, not the patched copy.

```bash
cp "$SRC/modeling_nemotron_h.py" "$WORK/modeling_nemotron_h.py.orig"
sha256sum "$WORK/modeling_nemotron_h.py.orig"   # e1cb5fc0...b1b6e for commit a6a1e503
patch -p1 -d "$SRC" < "$TOOLS/patches/nemotron_h_mamba2_einsum.diff"
```

The patch is [`nemotron_h_mamba2_einsum.diff`](../../tools/autoround-b70/patches/nemotron_h_mamba2_einsum.diff).
`modeling_nemotron_h.py` is NVIDIA's file under the OpenMDW 1.1 license; only
the diff is published here.

### Step 6. Add swap

```bash
sudo fallocate -l 32G /fast-disk/swap-export.img
sudo chmod 600 /fast-disk/swap-export.img
sudo mkswap /fast-disk/swap-export.img
sudo swapon -p 5 /fast-disk/swap-export.img
# after the export: sudo swapoff /fast-disk/swap-export.img && sudo rm /fast-disk/swap-export.img
```

### Step 7. Device selection: one variable, not two

Use `ONEAPI_DEVICE_SELECTOR=level_zero:0,1` alone. Attempt 1 had
`ZE_AFFINITY_MASK=1` together with `ONEAPI_DEVICE_SELECTOR=level_zero:1`. The
affinity mask renumbered the card to 0, the selector then asked for a card 1
that no longer existed, and torch saw zero XPU devices. AutoRound did not
fail. It ran on the CPU and we lost 14 minutes before noticing. The runner now
unsets `ZE_AFFINITY_MASK` and refuses to start unless `torch.xpu.device_count()`
is 2.

### Step 8. Run the export

```bash
cd "$WORK"
SRC=$SRC OUT=$OUT CALIB_CORPUS=$CALIB_CORPUS PY=$PY WORK=$WORK \
  setsid nohup "$TOOLS/run_bolt_export.sh" > "$WORK/run_bolt_export.log" 2>&1 < /dev/null & disown
```

Start it detached with `setsid nohup ... & disown`. A plain `nohup ... &`
died with the shell that started it. The runner waits until every shard in the
index exists, checks for two XPU devices, takes the `/tmp/b70-gpu0.lock` and
`/tmp/b70-gpu1.lock` locks, and runs:

```bash
PYTHONPATH=$TOOLS/shims ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
$PY $TOOLS/convert_nemotron_omni.py --src $SRC --bits 4 --out $OUT --corpus $CALIB_CORPUS \
  --iters 200 --nsamples 128 --seqlen 1536 --batch-size 1 --device 0,1
```

| Setting | Value | Why |
|---|---|---|
| Scheme | W4A16, int4 | vLLM XPU's int4 MoE kernel (`XPUExpertsWNA16`) loads it. 41 GB fits two 32 GB cards at 20.0 GiB of weights each. |
| Group size, symmetry | 128, symmetric | The GPTQ int4 g128 sym layout that vLLM XPU loads. |
| Export format | `auto_gptq` (GPTQ packing, `desc_act=False`) | Same reason. |
| `iters` | 200 | AutoRound's default tuning length. |
| `nsamples` | 128 | AutoRound's default sample count. |
| `seqlen` | 1,536 | Shorter than AutoRound's default to bound activation memory in the Mamba2 torch path. Longer sequences were not tried. |
| `batch_size` | 1 | Memory. |
| `device_map` | `0,1` | One card hits `DEVICE_LOST` at layer 21 (fix 10). |
| `low_gpu_mem_usage` | True | AutoRound keeps more state off the GPU. |
| `low_cpu_mem_usage` + `AR_DISK_STREAM_MODEL=1` | on | Builds a meta skeleton and reads each block's weights from disk when it is tuned. 139.5 GiB does not fit in 60 GB of RAM. |
| `enable_torch_compile` | off | Not tried on XPU for this run. AutoRound says it can cut tuning time by about 20%. |
| `ignore_layers` | `vision_model,vision_projector,lm_head,embeddings` | Only the language model is quantized. AutoRound also skips the 40 MoE router gates by itself. |
| Seeds | AutoRound 42, calibration shuffle 20260609 | Fixed for repeatability. |

**Calibration data.** The converter reads up to 4,000,000 characters from each
`*.txt.gz` file in `--corpus`, cuts them into 8,000-character pieces, shuffles
them with seed 20260609, keeps 128, and truncates each to 1,536 tokens. Our
corpus was three gzipped text files of our own documentation and code. It is
private and is not published, so a rerun will not be bit-identical. Any
English and code text should work; the effect of a different corpus on
quality was not measured.

### Step 9. Watch the run

A healthy two-card run looks like this:

- At load: `nemotron_h: restored 160 tensor(s) at high precision` (40 Mamba
  layers x `A_log`/`D`/`dt_bias`, plus 40 router biases), then
  `is_model_patched=False disk_stream_index=yes`. 128 vision attention-output
  tensors are left on the meta device. That is fine: vision is never quantized
  and its real weights are copied from the source at the end.
- `Ignored layers:` lists the 40 router gates (`mixer.gate`), embeddings,
  `lm_head` and the vision model.
- Per layer: about 41-44 s for a Mamba layer and 2-6 min for a MoE layer.
  Layers 48, 79 and 85 are the largest. The average was 158-240 s per layer
  and the last layers were the slowest (up to 387 s). Our run was at layer 44
  of 88 after 2 h 46 min.
- Each block logs a loss line, for example
  `quantized 740/740 layers in the block, loss iter 0: 5.318e+01 -> iter 180: 1.777e+01`.
  Loss values grow with depth. Our good run went from about 1e-7 in the first
  layers to about 1e1 in the last ones.
- AutoRound logs `'peak_ram'` and `'peak_vram'`. Ours ended at
  `'peak_ram': 47.45GB, 'peak_vram': {'0': 17.11GB, '1': 24.1GB}`.
- The output directory grows during the run (25 GB at layer 79).

Commands we used to watch it:

```bash
grep -o "Quantizing language_model.model.layers.[0-9]*: *[0-9]*%[^]]*]" "$WORK/export_logs/bolt-w4a16.log" | tail -1
grep -o "'peak_ram'[^}]*}" "$WORK/export_logs/bolt-w4a16.log" | tail -1
free -g; swapon --show
sudo dmesg | grep -iE "xe .*VM worker error|DEVICE_LOST" | tail
```

The end of a good run:

```text
Summary: quantized 27920/28117 in the model, unquantized layers: language_model.lm_head, vision_model...
Found 393 tensor(s) in the source checkpoint that are absent from the saved output ...
Successfully wrote 393 missing tensor(s) to 'model_extra_tensors.safetensors' ...
quantization+export wall time: 21436s
[bolt-runner ...] Bolt W4A16 status=ok size=43GB
```

The artifact lands in a subdirectory, `$OUT/bolt74b-bf16-w4g128/`.

### Step 10. After the export

[`post_export_chain.sh`](../../tools/autoround-b70/post_export_chain.sh) does
these in order and stops at the first failure. You can also run them by hand.

1. **Restore NVIDIA's modeling file** in the artifact and in the source. The
   export copied the einsum-patched file. The published repo must carry
   NVIDIA's original.

   ```bash
   ART=$OUT/bolt74b-bf16-w4g128
   cp "$WORK/modeling_nemotron_h.py.orig" "$ART/modeling_nemotron_h.py"
   cp "$WORK/modeling_nemotron_h.py.orig" "$SRC/modeling_nemotron_h.py"
   ```

2. **Static check** (CPU only, section 6):
   `$PY $TOOLS/check_artifact.py --src $SRC --artifact $ART --json $WORK/check.json`.
3. **Serve** with the [recipe launcher](VLLM-TP2-AUTOROUND-W4A16.md#launch). The
   load gate is `Using XPUExpertsWNA16` on both ranks.
4. **6-case gate, then speed**: `correctness6.py`, then `speed2.py`.

Our chain stopped at step 2 on the first checker version (fix 12). The rest ran
by hand.

## 5. Every fix

Fixes 1-11 are for the export, 12 is for the checker, 13-15 are for serving.

| # | Symptom | Cause | Fix | How verified |
|---:|---|---|---|---|
| 1 | `ImportError` for `mamba_ssm` at model load | Bolt's remote code imports `rmsnorm_fn` from `mamba_ssm`, a CUDA-only package | Pure-torch shim on `PYTHONPATH` (`shims/mamba_ssm`), same math as `rms_norm_ref`; the CUDA fast path stays off | Bitwise equal to the `mamba_ssm` reference; the model loads and the export completes |
| 2 | Out of memory in layer 0, Mamba2 backward pass (attempt 2) | The naive SSD path builds 6-D `(b, c, l, s, h, n)` tensors: 25.8 GiB at 1,536 tokens | Four broadcast-sum lines become `torch.einsum` in the source copy of `modeling_nemotron_h.py`; NVIDIA's file restored in the output | GPU check: peak 3.0 GiB, cosine 1.000000 for output, dx and d in_proj at 512 tokens; attempt 4 passed layer 0 and all 88 layers |
| 3 | Processor rejected, then a crash in the fallback | Bolt's image processor subclasses `TorchvisionBackend`, which the transformers 5.12.1 `ProcessorMixin` class check rejects (the remote code targets 5.8.1) | The converter disables `ProcessorMixin.check_argument_for_proper_class` | Dry run logs `processor: NemotronH_Omni_Reasoning_V3Processor` |
| 4 | `RuntimeError: AutoRound does not support parameters on meta device` | The meta-device guard also fires on the multimodal disk-stream skeleton | Run the guard only when there is no disk-stream index (`context/model.py`) | Dry run passes the load |
| 5 | Every LM tensor left on meta; `no data has been cached` | No checkpoint rename registered for `nemotron_h_puzzle`. Reusing the `nemotron_h` `backbone.`->`model.` rule is wrong: its reverse matches the `model.` inside `language_model.` | Register the full-prefix rename `language_model.backbone.` -> `language_model.model.` in the converter | Dry run quantizes 90/109 tiny layers; real run 27,920 |
| 6 | Would re-copy all LM tensors in BF16 (about 137 GiB) and run out of memory at save | The missing-tensor copy has no alias for the nested trunk | Alias `language_model.backbone.` <-> `language_model.model.` in `utils/missing_tensors.py` | Dry run copies only 36 vision/projector tensors; real run 393, none from the LM |
| 7 | Converter exits with code 3 | Its `is_model_patched` check fails on the multimodal path | Fail only when there is also no disk-stream index | Dry run and real run continue with `is_model_patched=False disk_stream_index=yes` |
| 8 | Would run out of memory or write stray files after the export | AutoRound's `verify_artifact.py` misses the name alias and the output subdirectory | Removed from the runner; `check_artifact.py` checks the artifact instead | Static check runs on the artifact subdirectory (section 6) |
| 9 | Torch sees 0 XPU devices; the export silently runs on the CPU (attempt 1) | `ZE_AFFINITY_MASK=1` renumbers the card before `ONEAPI_DEVICE_SELECTOR=level_zero:1` selects it | Use `ONEAPI_DEVICE_SELECTOR` alone; the runner unsets the mask and asserts 2 devices before starting | Attempt 4 used both cards (peak VRAM 17.1 + 24.1 GB) |
| 10 | `DEVICE_LOST` at layer 21 of 88 after 71 min (attempt 3); xe `VM worker error: -12` | One card's memory is not enough for the largest MoE blocks (432 experts x 2,560 at layer 21, larger at 48, 79, 85) | `--device 0,1`: AutoRound spreads each block over both cards | Attempt 4 finished 88/88 layers with 0 errors |
| 11 | The runner logged `exit=0` on a 5 GB partial output | It trusted the shell exit code | The runner checks the Python return code, errors in the log, the quantization summary, size >= 30 GB and `config.json` | Attempt 4: `status=ok size=43GB` with all five conditions met |
| 12 | Static check reported 11 false failures, then 1 | It looked up tensors under source names (`backbone`) while the artifact uses `model`; then it treated F32-stored `A_log`/`D`/`dt_bias` as changed | Compare under the source names (alias), and accept an F32 tensor when it holds exactly the BF16 source values | 20/21 after the alias fix; the F32 rule covers the last 120 tensors (section 6) |
| 13 | vLLM TP2 hung at the oneCCL rendezvous, both cards idle, no log for 20 min | The serve command lacked the cookbook dual-B70 contract (one card per worker, CCL thresholds). Needed, but not the whole cause (see 14) | [`patch_vllm_worker_affinity.py`](../../patches/patch_vllm_worker_affinity.py) plus the four CCL thresholds from [DUAL-B70-TP2.md](../DUAL-B70-TP2.md) | Each worker sees one card; the hang stayed until fix 14 |
| 14 | Still hung at startup | With `--network host`, oneCCL saw 7 interfaces and the two ranks paired endpoints on different ones (wifi and the docker bridge) | `CCL_KVS_IFACE=lo FI_TCP_IFACE=lo VLLM_HOST_IP=127.0.0.1` | CCL init passes; weights load |
| 15 | `RuntimeError: Unsupported TopK value` on the first forward pass | vllm-xpu-kernels `remap_hidden_states` supports TopK 1, 2, 4, 6, 7, 8, 10, 12, 16; Bolt routes 9 experts per token | [`patch_xpu_moe_topk_pad.py`](../../patches/patch_xpu_moe_topk_pad.py) pads to 10 with a copy of the last expert at router weight 0 (about 11% more routed-expert work) | Serving; gate and exact battery below. Output equality with an unpadded run cannot be tested: no TopK 9 kernel exists. |

Fixes 3-7 came from the CPU dry run. Fixes 4, 6 and 7 are in
`autoround-a5ea56e6-dryrun-fixes.diff`; 3 and 5 are in the converter. The
diff also adds `exc_info=True` to one calibration warning so that a failing
forward pass shows its traceback.

## 6. Verification

### Static artifact check

[`check_artifact.py`](../../tools/autoround-b70/check_artifact.py) compares the
artifact with the BF16 source on the CPU. It never writes to either.

- **Inventory.** It predicts the quantized layers from the source index and
  `llm_config.block_configs`: 27,648 routed-expert projections (13,824 experts
  x up/down), 80 shared-expert, 80 latent, 80 Mamba (`in_proj`/`out_proj`) and
  32 attention (`q`/`k`/`v`/`o`), 27,920 in total. The artifact must have
  exactly that set, layer by layer, and nothing quantized outside it.
- **Scales.** Every quantized layer has a scale tensor; no NaN, Inf or zero.
- **Byte identity.** Router `e_score_correction_bias` (must be F32), Mamba
  `A_log`/`D`/`dt_bias`/`conv1d`, every norm, router gates, the vision tower,
  the projector, embeddings and `lm_head` must be identical to the source,
  hashed in row chunks.
- **Lossless F32 rule.** AutoRound stores the 120 Mamba `A_log`/`D`/`dt_bias`
  tensors in F32. The checker accepts a tensor when the shape matches, the
  artifact dtype is F32, and the BF16 source cast to F32 is exactly equal to
  it. That is a widening, not a change.
- Also: all shards in the artifact index exist, `config.json` has an int4
  `quantization_config`, and every remote-code, tokenizer and template file is
  present.

The published version of the checker, run on the CPU against our local copy
of the uploaded artifact, reports 21 checks and 0 failures in 501 s. Results are in
[`autoround-w4a16-process-20261016-summary.json`](../../results/nemotron35-bolt-74b/autoround-w4a16-process-20261016-summary.json):
27,920/27,920 layers in scope, no NaN, Inf or zero scales, router biases F32
and identical, every other unquantized tensor identical (391 vision, 3
projector, 129 norms, 40 gates, embeddings, `lm_head`), and 200 Mamba core
tensors identical, 120 of them as exact F32 widenings. It also reports, as
information only, 455 artifact-only key names under `vision_model.*` and no
Mamba `out_proj` scale below 6.1e-5.

### Function and quality

All requests are greedy (temperature 0, top_k 1, seed 0) with thinking off.
Unless noted, the server ran the 262K config with prefix caching on.

| Check | Script | Result |
|---|---|---|
| 6-case gate | `correctness6.py` | 5/6. Arithmetic, capital, letter count, code and Spanish pass. Vision with the question text **before** the image answers only "42" (deterministic, 3/3). Image first answers "Red circle with 42". Cause (quantization or vLLM image placement) not separated. |
| Exact-answer battery | `exact_battery.py` | 8/8, including `sum(i*i for i in range(4))` = 14, the case that broke earlier hybrid W4A16 exports. |
| Vision, both orders | `bolt_suite.py ... vision` | 6/6 image first, 6/6 text first (red circle text first: "42 red circle"). Put the image before the text. |
| GSM8K | `bolt_suite.py ... gsm8k --n 200` | 192/200 = 96.0% on the first 200 test questions. Not a full-benchmark claim. |
| Multi-turn recall | `bolt_suite.py ... multiturn` | 3/3. |
| Prefix cache equivalence | `bolt_suite.py ... cacheeq` | 6/6 identical with a 20,000-character shared prefix. |
| Concurrency | `bolt_suite.py ... conc` | 4/4 correct, identical to serial. |
| Needle retrieval | `bolt_suite.py ... needle` | Found at 31,150-31,153 tokens (10%, 50%, 90% depth) and 127,071-127,078 tokens (10%, 50%, 90%). One needle at 50% depth in a 249,212-token prompt was found; that request took 34.6 min. |
| Eager vs graph mode | `correctness6.py` | Greedy text byte-identical on all 6 gate cases at the 16K config. |

Inputs: `bolt_suite.py` draws its own images. GSM8K is `openai/gsm8k`
revision `740312add88f781978c0658806c59bc2815b9866`. The needle and cache
tests use the WikiText-2 raw test file `wiki.test.raw` (1,292,013 bytes,
SHA-256 `aca2f46735043bcfd0a44eca981d04627b9cdf74c4c9a04bf0856d04066f58fc`).

### Speed protocol

[`speed2.py`](../../tools/autoround-b70/speed2.py) builds prompts of an exact
token count with `/tokenize`. Each request starts with a unique random prefix,
so nothing is shared. One request at a time, 128 output tokens with
`ignore_eos`, greedy, thinking off. Per shape: 1 discarded warmup, then 5
reps; the table shows the median.

- Prefill rate = prompt tokens / time to first token (includes client
  overhead). Decode rate = (output tokens - 1) / time after the first token.
- The server ran with `--max-model-len 262144`, prefix caching on, graph mode.
- Prefix-cache hits were 46,464 before and after the run: delta 0, so every
  prompt was cold.
- Both cards capped at 180 W. Average draw over the 1,178 s run, from the
  card energy counters: 170.8 W (card 0) and 172.3 W (card 1).

```bash
REPS=5 python3 tools/autoround-b70/speed2.py "$WORK/speed-confirm" 886 8192 32000 64000
```

## 7. Results and claim boundary

| Item | Value |
|---|---|
| Tuning time | 5 h 57 min on two cards, 88 layers |
| Peaks during tuning | VRAM 17.1 GB (card 0) + 24.1 GB (card 1); host RAM 47.5 GB |
| Artifact | 41 GB (40.65 GiB), 9 shards, 112,940 tensors; 27,920 quantized layers; 393 unquantized tensors copied from the source |
| Static check | 27,920/27,920 quantized layers in scope; no NaN, Inf or zero scales; every unquantized tensor identical (Mamba core exact in F32) |
| vLLM load | 20.0 GiB weights per card; `XPUExpertsWNA16` on both ranks; about 12-15 min to ready on our USB SSD |

C1 speed, two B70, TP2, graph mode, both cards at 180 W, median of 5
(the same numbers as the [recipe](VLLM-TP2-AUTOROUND-W4A16.md#speed-c1)):

| Prompt tokens | Prefill tok/s (prompt/TTFT), median [min-max] | TTFT s | Decode tok/s, median [min-max] |
|---:|---:|---:|---:|
| 886 | 2,925 [2,895-2,943] | 0.30 | 60.5 [60.2-60.7] |
| 8,192 | 2,014 [1,993-2,026] | 4.07 | 50.7 [49.9-50.7] |
| 32,000 | 871 [835-873] | 36.7 | 33.5 [32.0-33.5] |
| 64,000 | 486 [463-487] | 131.7 | 23.0 [23.0-23.0] |

Claim boundary:

- One artifact: the repo above at weights commit `31f574676ecc`. A rerun with
  another calibration corpus or another AutoRound commit is a different
  artifact.
- One host, two B70, one engine image with the patches in the recipe.
  Evidence tier E2.
- Quality evidence is the checks in section 6. That is not a general quality
  claim, and there is no comparison against the BF16 model on vLLM.

Not tested:

- The export on intel/auto-round#2512's head `345e0fe1` (or on AutoRound main
  once it merges).
- A different calibration corpus, `seqlen`, `iters` or `torch.compile` during
  tuning.
- The export on one card plus CPU offload.
- Task benchmarks beyond the first 200 GSM8K questions; tool calling; video;
  concurrency beyond 4; the LiLiCorr DFlash drafter.

## 8. If you do this again

1. Download the small files first and run the CPU dry run while the weights
   download.
2. Save NVIDIA's `modeling_nemotron_h.py`, then apply the einsum patch.
3. Add 32 GB of swap on a 60 GB host before you start.
4. Set `ONEAPI_DEVICE_SELECTOR=level_zero:0,1` and nothing else. Check that
   torch sees 2 devices.
5. Start on two cards from the beginning. Start detached with `setsid nohup`.
6. Budget about 6 h of GPU time, 1 h 40 min of download at 25-30 MB/s, and
   30 min of setup.
7. Judge the run from the log and the output, not from one exit code.
8. Restore NVIDIA's modeling file in the output before you upload.
9. Run the static check with the name alias and the F32 rule.
10. Keep other disk-heavy jobs off the model disk during the check and the
    vLLM load.

## 9. Adapting it to another Nemotron-H or Mamba hybrid

| Part | Bolt-specific or reusable | What to change |
|---|---|---|
| CPU dry run on a tiny checkpoint | Reusable method | Edit the config overrides in `build_tiny_bolt.py` to the target's config keys. Keep the real remote code and tensor names. |
| `mamba_ssm` shim | Reusable for any remote code that imports `mamba_ssm`'s `rmsnorm_fn` | Nothing, unless the code imports other `mamba_ssm` symbols. |
| Mamba2 einsum patch | Reusable where the modeling file has the same naive SSD code | Line numbers differ: look for `G_intermediate` and the three other broadcast-sum lines, and re-check memory and output on the GPU. |
| `language_model.backbone.` -> `language_model.model.` rename and alias | Specific to `nemotron_h_omni` wrappers | Check the on-disk prefix of the language model in the index. |
| `ProcessorMixin` check bypass | Specific to Bolt's image processor on transformers 5.12.1 | Drop it if the processor loads. |
| Two cards (`device_map 0,1`) | Depends on the largest MoE layer | Compare the target's largest layer (experts x rows) with Bolt's layer 21 (432 x 2,560). |
| Swap and RAM | Depends on model size | Expect RSS to grow during the run. |
| `check_artifact.py` | Mostly reusable | The quantized-layer regex and the `block_configs` counting follow Bolt's layout. The 27,920 total is Bolt's only. |
| Top-k pad for vLLM XPU | Needed only when experts per token is not 1, 2, 4, 6, 7, 8, 10, 12 or 16 | Check `num_experts_per_tok`. |
| Loopback pinning for TP2 | Host-specific | Needed on any host with several network interfaces and `--network host`. |
