# AutoRound W4A16 export and checks for Nemotron 3.5 Bolt on two B70

Scripts used to make and check
[`SergiioB/Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym`](https://huggingface.co/SergiioB/Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym).
The guide that ties them together is
[docs/nemotron35-bolt-74b/AUTOROUND-W4A16-PROCESS.md](../../docs/nemotron35-bolt-74b/AUTOROUND-W4A16-PROCESS.md).

These scripts are not part of the stdlib-only repository runtime. The export
scripts need the AutoRound venv (torch XPU, transformers 5.12.1, auto-round).
The checks need `torch` and `safetensors`. The client scripts use the stdlib,
except `bolt_suite.py` (Pillow for `vision`, pyarrow for `gsm8k`).
No path is hard-coded: pass paths as arguments or environment variables.

## Export

| File | What it does | Usage |
|---|---|---|
| `build_tiny_bolt.py` | Builds a tiny random checkpoint with Bolt's real config, remote code and tensor names. Needs only the small files of the BF16 repo. | `python build_tiny_bolt.py --src $SRC --out $WORK/tiny` |
| `convert_nemotron_omni.py` | The converter. AutoRound W4A16 (or W3A16), group 128, symmetric, GPTQ packing, disk-stream load. Quantizes only the language-model Linear layers. | `python convert_nemotron_omni.py --src $SRC --out $OUT --corpus $CALIB_CORPUS --device 0,1` |
| `run_bolt_export.sh` | Waits for every shard, checks that torch sees two XPU devices, runs the converter on both cards, then judges the run from the log and the output size. | `SRC=... OUT=... CALIB_CORPUS=... PY=... ./run_bolt_export.sh` |
| `shims/mamba_ssm/` | Pure-torch stand-in for the CUDA-only `mamba_ssm` package. `rmsnorm_fn` has the same math as `mamba_ssm`'s `rms_norm_ref`. The kernel entry points raise `NotImplementedError`, so the remote code takes its torch path. | `PYTHONPATH=tools/autoround-b70/shims` |
| `patches/nemotron_h_mamba2_einsum.diff` | Replaces four broadcast-and-sum lines in Bolt's Mamba2 torch path with `torch.einsum`. Forward plus backward on 1,536 tokens peaked at 25.8 GiB (out of memory); with the patch, 3.0 GiB. Apply to the source copy only, never ship it. | `patch -p1 -d $SRC < patches/nemotron_h_mamba2_einsum.diff` |
| `patches/autoround-a5ea56e6-dryrun-fixes.diff` | The three AutoRound fixes found by the CPU dry run, on top of auto-round commit `a5ea56e6` (from intel/auto-round#2512). | `git -C auto-round apply patches/autoround-a5ea56e6-dryrun-fixes.diff` |

`patches/nemotron_h_mamba2_einsum.diff` modifies NVIDIA's `modeling_nemotron_h.py`,
which is licensed under the OpenMDW 1.1 license with the model. Only the diff is
published here, not the file.

## Checks

| File | What it does | Usage |
|---|---|---|
| `post_export_chain.sh` | After the export: restores NVIDIA's modeling file, runs the static check, starts the cookbook vLLM TP2 launcher, waits for the load gate, runs the 6-case gate, then speed. Stops at the first failure. | `SRC=... OUT=... ORIG=... PY=... ./post_export_chain.sh` |
| `check_artifact.py` | Static CPU check of the export against the BF16 source: quantized-module inventory, scale sanity, byte identity of every unquantized tensor. Exit code = number of failed checks. | `python check_artifact.py --src $SRC --artifact $ART --json check.json` |
| `correctness6.py` | 6-case greedy gate (5 text cases + 1 vision case on `red-circle-42.png`). | `python3 correctness6.py OUTDIR --url http://127.0.0.1:8000/v1` |
| `exact_battery.py` | 8 exact-answer questions, thinking off, including `sum(i*i for i in range(4))` = 14. | `URL=http://127.0.0.1:8000 python3 exact_battery.py` |
| `bolt_suite.py` | vision (6 images x 2 orders), gsm8k (first N), multiturn, cacheeq, needle, conc. | `GSM8K_PARQUET=... WIKITEXT=... python3 bolt_suite.py OUTDIR gsm8k --n 200` |
| `speed2.py` | C1 speed: exact prompt sizes via `/tokenize`, unique prefix per request, 1 warmup + `REPS` reps, median. | `REPS=5 python3 speed2.py OUTDIR 886 8192 32000 64000` |
| `red-circle-42.png` | The 512x512 test image used by `correctness6.py`. | default `--image` |

`post_export_chain.sh` calls the cookbook launcher
[`launch-nemotron-bolt-vllm-tp2.sh`](../../benchmarks/nemotron35-bolt-74b/launch-nemotron-bolt-vllm-tp2.sh)
(graph mode, 65,536 context). Our own run used an eager-mode wrapper first;
the gate, exact battery and speed numbers in the guide come from graph mode.
