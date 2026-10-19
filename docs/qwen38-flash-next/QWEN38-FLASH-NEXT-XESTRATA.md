# Flash-Next: IQ3_S versus IQ3_XXS on two B70s

[Family index](README.md) · [Dual-B70 topology](../DUAL-B70-TP2.md) · [Raw evidence](../../results/qwen38-strata-quant-20261019/summary.json)

## 1. Result

IQ3_S runs on two Arc Pro B70s with XeStrata `xe0.1.45-1`, a 262,144-token configured context, int8 attention cache and pipelined MTP4. In matched C1, thinking-off tests, its median code decode was **86.7 tok/s**, versus **86.8 tok/s** for IQ3_XXS: 34 prompt tokens, 512 output tokens, three measured runs after one warm-up. At 32,429 prompt tokens, median prefill was **1,879.5 versus 1,894.2 tok/s**. IQ3_S is the largest quant in the pinned ISTA repository listing, and fits this setup, but GPU 1 is nearly full. Both quants passed 5/6 small answer checks without thinking and 6/6 with high effort. These checks establish no general quality advantage for IQ3_S.

## 2. What you need

| Item | Measured setup / requirement |
|---|---|
| GPUs | Two Intel Arc Pro B70s, nominal 32 GB each; first stage layers 0–23, second stage layers 24–47 |
| Power | As-found caps: GPU 0 **180 W**, GPU 1 **230 W**. The launcher did not change them. This is not a matched-180-W-per-card test. |
| Host RAM | 64 GB class host; engine memory guard reported 60.7 GiB total. CPU expert kernels used AVX2. |
| Disk | IQ3_S first shard: 54,817,524,224 bytes (51.05 GiB). Shared lookup shard: 28,800,138,432 bytes (26.82 GiB). Native pack about 1.43 GiB, MTP files about 787 MiB, CPU vision projector about 866 MiB. Allow roughly 81 GiB for a fresh serving set. |
| Model source | [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09), revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09` |
| Runtime | [XeStrata](https://github.com/MistvVK/XeStrata), packaged SYCL engine reporting `xe0.1.45-1`; host oneAPI compiler and MKL 2026.1 library paths; no Docker image in this run |
| Launcher | B70 Launcher 0.6.19; separate `qwen38-flashnext-iq3s` recipe and temporary per-launch config |

Exact engine, server, expert-profile and model SHA256 values are in [`artifacts.json`](../../results/qwen38-strata-quant-20261019/artifacts.json). The measured binary is identified by its digest, not by an assumed upstream commit. Its engine SHA256 is `d633e321ba22179c9b1515840b922a49d0d426caea4819237fa2350f4c92e450`.

The existing IQ3_XXS lookup shard and the IQ3_S repository's second shard have the same SHA256, `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`. Reuse those bytes through an IQ3_S-named symlink. Do not copy or download another 26.82 GiB unnecessarily.

## 3. Timeline

Clock times below are local UTC+02:00 on 2026-10-19. Missing clock measurements are labeled rather than reconstructed from estimates.

| Time / elapsed | Step | Outcome |
|---|---|---|
| Before the timed native comparison; clocks not captured | Two research agents failed to start with `No model selected`; continued inline | No agent findings used |
| Before download; clock not captured | Found insufficient free space; removed the stale Signal AP-Q4_K_XL download with operator approval | Preserved the running IQ3_XXS artifacts |
| Download elapsed **32m20.62s**; start/end clock not captured | Downloaded the pinned 51.05 GiB IQ3_S first shard over Wi-Fi | Completed; SHA256 matched the repository LFS digest |
| 09:30:27 | Submitted nine previously catalog-blocked results | IDs 265–273, read back through the API |
| 09:33:14 | Opened [IntelInside catalog PR #116](https://github.com/labscommunity/intelinside/pull/116) | Correct IQ3 boards and Strata runtime requested |
| 09:34:40 | Submitted two historical llama.cpp Flash-Next results with separate workload coordinates | IDs 274–275, read back through the API |
| Catalog validation, clock not captured | First validator run could not import `vite` | Reused the installed dependency directory; catalog validation then passed |
| 09:39:00 | Started native-context IQ3_XXS through the new launch wrapper | Source config remained separate from launch overrides |
| Before 09:40:50 | Started benchmark before the HTTP endpoint was ready | `ConnectionRefusedError`; waited for readiness before retrying |
| 09:40:50–09:42:43 | IQ3_XXS native speed suite, including warm-ups | Twelve saved responses |
| After native speed suite | IQ3_XXS answer checks, none and high effort | 5/6 and 6/6 |
| 10:03:37; failed launch elapsed **9.91s** | First IQ3_S launch pointed `--ple-gguf` at the old alias | `native dense: duplicate split shard number` |
| **10:04:49; successful launch elapsed 2m37s** | **Used the IQ3_S-named second-shard symlink for `--ple-gguf`** | **Ready; launcher smoke answered in 0.718s** |
| 10:07:44–10:09:40 | IQ3_S native measured speed rows | Three measured rows per case after warm-up |
| After native speed suite | IQ3_S answer checks, none and high effort; generic `loaded` alias | 5/6, 6/6, and `4` for 2+2 |
| During allocation/client checks; clocks not captured | `ps -C` was unsupported; a remote config write was refused because the read was truncated; unauthenticated UI GET returned 403 | Used the known PID, reconstructed the complete config before writing, and used the documented nonce bootstrap |
| During XeCores publication; clock not captured | Same-commit cookbook sync reported missing pin constants even though both existed | Fixed presence validation; same-pin sync, pin gate, build and nine-page navigation gate passed |

Native baseline engine-start to final measured IQ3_S speed row: **30m40.51s**. This interval overlaps the weight download and other work; it is not pure benchmark compute time. Exactly timed failed IQ3_S load: **9.91s**. The premature benchmark and dependency failure were not separately clocked, so total time lost to failures and full-session wall time are **not measured**.

## 4. Step by step

### 1. Download and verify the first shard

Set paths for your checkout and storage. The examples reuse a working IQ3_XXS installation and its MTP, expert profile and CPU vision files.

```bash
export XE=/path/to/XeStrata
export MODEL_DIR=/your/disk/qwen38-flashnext-iq3s/IQ3_S
export PACK=/your/disk/strata-packs/iq3s
export OLD_LOOKUP=/your/disk/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
export FIRST="$MODEL_DIR/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
export SECOND="$MODEL_DIR/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"
mkdir -p "$MODEL_DIR"
curl --fail --location --retry 3 --continue-at - --output "$FIRST" \
  https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
printf '%s  %s\n' 4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3 "$FIRST" | sha256sum -c -
printf '%s  %s\n' 316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113 "$OLD_LOOKUP" | sha256sum -c -
ln -s "$OLD_LOOKUP" "$SECOND"
```

On this workstation, adding `--interface wlx7419f81751a9` selected the operator-requested Wi-Fi connection. Use your own interface name if needed. The symlink command assumes the new link does not already exist.

### 2. Build a separate native pack

```bash
"$XE/.venv/bin/python" "$XE/tools/iq_pack.py" --gguf "$FIRST" --out "$PACK"
```

Observed output: 1,079 tensors, 302 served natively, zero tensors in `extra.bin`, 1.43 GiB arena. The tokenizer builder wrote vocabulary and chat-template files. Do not reuse the IQ3_XXS pack or generate a second full `experts.bin` for this mmap route.

### 3. Derive the IQ3_S config without changing IQ3_XXS

```bash
export BASE_CONFIG="$HOME/.config/xestrata/xestrata-iq3_xxs.json"
export NEW_CONFIG="$HOME/.config/xestrata/xestrata-iq3_s.json"
python3 - <<'PY'
import json, os
from pathlib import Path
cfg = json.loads(Path(os.environ['BASE_CONFIG']).read_text())
args = cfg['args']
for flag, value in [('--native', os.environ['FIRST']), ('--ple-gguf', os.environ['SECOND']),
                    ('--pack', os.environ['PACK']), ('--max-context', '262144')]:
    args[args.index(flag) + 1] = value
for flag in ('--rope-scaling', '--rope-scale', '--yarn-orig-ctx'):
    if flag in args:
        i = args.index(flag)
        del args[i:i + 2]
cfg['tokenizer'] = os.environ['PACK'] + '/tokenizer'
cfg['model_name'] = 'qwen3.8-flash-next-iq3_s'
cfg['log'] = str(Path.home() / '.local/state/xestrata/xestrata-iq3_s.log')
if cfg.get('vision'):
    cfg['vision']['model'] = os.environ['FIRST']
Path(os.environ['NEW_CONFIG']).write_text(json.dumps(cfg, indent=2) + '\n')
PY
```

Keep `--layer-split 24`, `--pipeline-windows 2`, `--spec 4`, `--kv int8`, the MTP directory and expert profile matched across arms. The explicit lookup path must be the same spelling as the shard discovered beside `--native`; the older path alias causes the duplicate-shard failure even when its bytes are identical.

### 4. Select and launch the recipe

The shipped recipe contains the measured workstation's local paths. For another disk layout, set `model_path`, `config_file`, `py_exe`, `server_script`, `vision_exe` and `mmproj` in local `recipe_overrides` before launching; see the launcher's recipe-format documentation.

```bash
b70 launch qwen38-flashnext-iq3s -e xestrata --ctx 262144 --vision --effort none
# Text-only launch:
# b70 launch qwen38-flashnext-iq3s -e xestrata --ctx 262144 --no-vision
```

Wait for `serving` before measuring. `--effort` sets the launch default through the temporary config. API requests can select `reasoning_effort: none|low|medium|high|xhigh` without reloading weights. `--vision` enables the CPU encoder; accepting an image request does not establish image-answer accuracy.

### 5. Run both arms with the same protocol

From the cookbook root, with IQ3_S serving:

```bash
python3 benchmarks/qwen38-strata-quant-compare.py --label iq3_s-native262k-vision-cpu --output results/qwen38-strata-quant-20261019/iq3_s-native
python3 benchmarks/qwen38-strata-quant-compare.py --suite quality --repeats 1 --label iq3_s-native262k-vision-cpu --output results/qwen38-strata-quant-20261019/iq3_s-native-quality
python3 benchmarks/qwen38-strata-quant-compare.py --suite quality --effort high --repeats 1 --label iq3_s-native262k-high --output results/qwen38-strata-quant-20261019/iq3_s-native-quality-high
```

Stop this recipe, launch `qwen38-flashnext -e xestrata --ctx 262144 --vision --effort none`, and repeat with `iq3_xxs` labels and output folders. Use fresh folders for a new campaign rather than overwriting the committed evidence.

## 5. Every fix

| # | Literal symptom / observed problem | Cause | Fix | Verification |
|---|---|---|---|---|
| 1 | `No model selected` from research agents | Agent configuration unavailable | Continued the research inline | Source paths, catalog responses and saved run evidence used directly |
| 2 | Insufficient space for a 51.05 GiB new shard | Storage occupied by a stale Signal download | Removed that download with operator approval; reused the lookup shard | Full pinned download completed and SHA256 matched |
| 3 | `Cannot find package 'vite'` | New catalog worktree lacked dependencies | Linked the existing installed `node_modules` | Catalog validator passed: 52 hardware, 51 models, 26 quants, 9 runtimes |
| 4 | `ConnectionRefusedError: [Errno 111] Connection refused` | Benchmark began while weights were loading | Waited for the HTTP models endpoint | Twelve native IQ3_XXS speed responses saved |
| 5 | `native dense: duplicate split shard number` | Explicit PLE path used an old alias, so the same split shard was appended twice | Pointed PLE at the IQ3_S-named symlink beside shard 1 | IQ3_S loaded and completed all 36 comparison requests |
| 6 | Earlier dry-build implementation changed vision in the source config; effort was passed as an unsupported server flag | Launch controls modified shared state / targeted the wrong interface | Temporary per-launch config; effort, context and vision applied there | 348 launcher tests passed, including source-config preservation; native launches and generic-alias request succeeded |
| 7 | Result 240 classified IQ3_XXS as `q2_0` under llama.cpp | Missing catalog quant/runtime entries had been replaced with incorrect labels | Withdrew result 240; requested proper entries in PR #116 | DELETE returned 200; new submission command refuses to post until catalog sync |
| 8 | `official-results.js pin constants not found — G-PIN would fail` | Vendor script treated an unchanged replacement as a missing constant | Validate both constant patterns before replacement, not whether the text changes | Same-commit `sync:cookbook` passed; 30 docs and 50 benchmark records vendored; G-PIN and G-NAV passed |
| 9 | `Refusing to overwrite ... with an incomplete read projection` | Remote config exceeded the read output cap | Read the omitted tail, reconstructed all 52,696 bytes, then changed three context values | Remote YAML parsed; loaded-model context asserted at 262,144; remote models endpoint returned IQ3_S |
| 10 | `ps: unsupported option '-C'` | The available command wrapper does not support process-name selection | Queried the known engine PID and its `smaps_rollup` | Main-engine resident/file-backed/anonymous/swap figures recorded |
| 11 | `unauthorized request` on the launcher root URL | Browser had no launcher authentication cookie | Used authenticated `POST /api/window` and its one-use nonce URL | Chromium displayed the new on-disk IQ3_S recipe and its context controls |

## 6. Verification

Each native quant arm has **36 saved responses**: 12 speed responses (three cases, one warm-up plus three measured runs), 12 thinking-off answer checks, and 12 high-effort answer checks. Answer scores use six measured checks per mode, excluding each warm-up. All measured speed responses report zero prompt-cache hits.

| Evidence | Location |
|---|---|
| Raw requests, full responses, token counts and engine timings | [`iq3_xxs-native/`](../../results/qwen38-strata-quant-20261019/iq3_xxs-native/) and [`iq3_s-native/`](../../results/qwen38-strata-quant-20261019/iq3_s-native/) |
| Answer checks | Corresponding `*-native-quality/` and `*-native-quality-high/` folders in the same evidence directory |
| Compiled values and allocation snapshots | [`summary.json`](../../results/qwen38-strata-quant-20261019/summary.json), [`allocation.json`](../../results/qwen38-strata-quant-20261019/allocation.json) |
| Generic loaded-model request | [`loaded-smoke.json`](../../results/qwen38-strata-quant-20261019/loaded-smoke.json): model resolved to IQ3_S and answered `4` |
| Historical upload receipts | [`intelinside-receipts.json`](../../results/qwen38-strata-quant-20261019/intelinside-receipts.json): 11 new results, each read back |

The launcher UI was exercised in Chromium: IQ3_S appeared as a selectable on-disk model with its XeStrata route, 262,144-token default and 350,000-token experimental ceiling. No UI unit tests were added. Launcher release verification checked 76 packaged files.

Publication checks also passed the four existing benchmark-catalog tests and the renderer's 50-record current-output check. XeCores preview rendered the quant guide and linked the 0.6.19 tarball, checksum and manifest. The cookbook document allowlist now includes the Flash-Next family index, this guide and the llama.cpp guide. [Launcher release 0.6.19](https://github.com/SergiioB/b70-launcher/releases/tag/v0.6.19) has SHA256 `fb46ef1e553b9c7a62a2643f562ce54ea246be3c17e04aa8b20f545d859e6594`.

## 7. Results and claim boundary

### Matched native-context speed

C1, temperature 0, `reasoning_effort: none`, MTP4, K=24, two pipeline windows, int8 attention cache, 262,144-token configured capacity. Medians of three measured rows after warm-up. Rates use the engine's `timings`, not client time-to-first-token or wall-clock throughput. Power caps were 180/230 W as found; this was a sequential A-then-B comparison, not an interleaved or iso-power study.

| Workload / metric | IQ3_XXS | IQ3_S |
|---|---:|---:|
| Code decode, P34/G512 | 86.8 tok/s | 86.7 tok/s |
| Train explanation decode, P59/G512, thinking disabled | 93.3 tok/s | 93.2 tok/s |
| Prefill, P32,429, output cap 128 | 1,894.2 tok/s | 1,879.5 tok/s |
| Small answer checks, thinking off | 5/6 | 5/6 |
| Small answer checks, high effort | 6/6 | 6/6 |

The long-prefill outputs ended naturally: IQ3_XXS produced 13–17 tokens and IQ3_S 36–54 tokens. Their decode speeds are therefore not a matched generated-token-length comparison and are not used as headline decode results. The code outputs reached their 512-token cap; this measures generation speed, not a completed or executed LRU-cache implementation.

Both thinking-off arms answered `50` instead of `48` on the exact-answer average-speed question. Both high-effort arms answered correctly. Changing quant did not fix that failure in this small set.

### Disk, RAM and VRAM

| Allocation | IQ3_XXS | IQ3_S |
|---|---:|---:|
| GPU 0 dashboard usage | 23.1 GiB | 27.7 GiB |
| GPU 1 dashboard usage | 29.2 GiB | 31.7 GiB |
| Expert cache, GPU 0 / GPU 1 | 17.72 / 22.24 GiB | 22.14 / 24.51 GiB |
| Resident expert slots | 24,576 / 24,576 | 24,488 / 24,576 |
| Main engine resident RAM snapshot | 39.7 GiB | 43.1 GiB |
| Main engine swap snapshot | 439 MiB | 0 MiB |
| First shard | 43.81 GiB | 51.05 GiB |

These are snapshots, not peak-memory measurements. Whole-card usage includes buffers and other allocations. Most engine RAM is clean, file-backed mapped weight data, not private heap; Linux can reclaim it. Disk, RAM and VRAM contain overlapping copies and must not be added as unique model size. The shared lookup stays on disk and is read through the PLE path. A separate CPU vision encoder is not included in the main-engine RAM snapshot.

IQ3_S fits, with 88 expert slots outside its resident cache and little GPU 1 headroom. IQ3_XXS remains a reasonable choice when extra memory margin matters. The measured speed difference is below 1%; this run does not establish a useful speed advantage for either quant.

### IntelInside publication

| Results | Status |
|---|---|
| Bolt SYCL / IQ2_XXS | [265](https://www.intelinside.ai/results/265), [266](https://www.intelinside.ai/results/266), [267](https://www.intelinside.ai/results/267) published |
| Bolt AutoRound TP2 | [268](https://www.intelinside.ai/results/268)–[271](https://www.intelinside.ai/results/271) published |
| Qwen3.8-27B FP8 / EXL3 | [272](https://www.intelinside.ai/results/272), [273](https://www.intelinside.ai/results/273) published; FP8 notes identify the synthetic diagnostic workload |
| Historical Flash-Next llama.cpp | [274](https://www.intelinside.ai/results/274), [275](https://www.intelinside.ai/results/275) published under custom runtime 16; prompt/decode coordinates kept separate |
| New Strata IQ3 results | Waiting for [catalog PR #116](https://github.com/labscommunity/intelinside/pull/116) and database sync; no Q2_0 or llama.cpp substitution |

`python3 benchmarks/submit-qwen38-strata-quants.py` prints the prepared native code payload. Add `--submit` only after the catalog sync. The command registers XeStrata under Strata, uses fixed idempotency keys, and submits prompt/decode medians from the same P34/G512 workload. It does not pair 32K prefill with short-prompt decode. The blocked `--submit` path was exercised and refused to post. Correct-runtime live submission remains untested until the missing catalog entries exist.

PR #116's four completed check runs passed; its Vercel preview requires Community Labs team authorization. No Greptile review was attached when checked. Prior launcher PR #4 had no inline review comments; catalog PR #115 had no reviews returned.

**Not tested:**

- Long-context retrieval or perplexity at the configured 262K capacity; the longest measured prompt here was 32,429 tokens.
- IQ3_S at 350K YaRN, its long-context accuracy, or a claim that extrapolation degradation is negligible.
- Broad coding-task accuracy, KL divergence, BF16 parity, sustained thermal behavior or concurrent requests.
- Correct visual understanding. Earlier small color-image replies were wrong; image-input metadata and HTTP success are not correctness proof.
- Equal power caps, energy per token, a quant-by-quant autotuning search, or a larger quant from another repository.

This is self-reported workstation evidence with saved requests and timings. It supports the stated serving configuration and workloads, not a model-wide quality ranking. Earlier external 122.2 tok/s claims are not reproduced by this campaign and are not used as local measurements. The retained `iq3_xxs/` folder is an earlier 350K configuration run and is not the native-context comparison arm.

## 8. If you do this again

Reuse the verified lookup shard and MTP files, download only the new first shard, make a separate native pack, use the matching new lookup filename, and wait for `serving`. On this Wi-Fi download, weights took 32m20s; hashing took 1m46s, packing 5.23s, successful IQ3_S launch 2m37s, and its speed-plus-answer campaign 2m30s. Allow about 40 minutes on the same connection; disk/cache state and network speed can change that substantially.

## 9. Adapting it

For another quant, regenerate its pack and change both the native shard and tokenizer. Share the lookup only after comparing its complete digest; matching filenames or sizes are insufficient. Preserve prompt text, token caps, effort, cache behavior, GPU selection and power settings before comparing rates. Keep native context as the measured default; treat YaRN capacity and vision accuracy as separate experiments. Publish under the actual quant and runtime, and retain failed attempts with their literal errors.

**Keeper decision (2026-10-19):** the workstation now keeps **IQ3_XXS as its
only Flash-Next quant**; the IQ3_S first shard and pack were deleted (about
53 GiB freed). Equal speed, equal check scores, more disk, more VRAM and a
nearly full second GPU bought nothing measurable. Section 8 restores IQ3_S in
about 40 minutes if a denser quant is ever wanted again.
