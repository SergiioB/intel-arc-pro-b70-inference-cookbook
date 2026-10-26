# Nanbeige4.3-33B on one Arc Pro B70 — llama.cpp SYCL recipe

> **Status: measured on one card, single stream, provisional.** Every number in
> this page was measured on 2026-10-25 on one Arc Pro B70, `Q4_K_M` GGUF,
> llama.cpp SYCL build 361 (`1692f9e50`), flash attention on, KV `q8_0` K /
> `q4_1` V, no speculation, 230 W configured cap (measured draw 230 W).
> Concurrency, context above 32K, and task-level quality are **not** measured.
> This page is not in `data/benchmarks.v1.json` yet.

Nanbeige4.3-33B is a dense `LlamaForCausalLM`: 52 layers, hidden 7168, 56 query and
8 KV heads at head_dim 128, FFN 22016, vocab 166,912, rope theta 1.2e8, 512K
position training. All 52 layers use full attention. That is the whole story on
this hardware: about 20 GB of weights are read for every token, and every extra
context token makes the attention kernel work longer.

## 1. What this recipe gives you

| Cell | Measured |
|---|---:|
| Decode, empty context (tg128, llama-bench) | **23.14 tok/s** |
| Decode at 8K occupied context | 14.63 tok/s |
| Decode at 32K occupied context | 6.67 tok/s |
| Prefill pp2048 | **1,175 tok/s** |
| Prefill pp8192 | **1,082 tok/s** |
| Server C1, 777-token prompt, 128 output tokens | 21.80 tok/s, 0.095 tokens/J |
| Server C1, 12,844-token prompt, 128 output tokens | 12.16 tok/s, 0.053 tokens/J |

Two settings do most of the work: `-ub 2048` for prefill (+43 % at pp8192 over the
old `-ub 512` default), and a correctly sized n-gram draft for repetitive work
(**+73 %** decode on repeated code — but only with `size_m 7`; `size_m 8` is 31 %
slower, because it makes the verify batch 9 and leaves the 8-wide kernel window).

## 2. Prerequisites

- One Intel Arc Pro B70 (32 GB). A second card is not needed; if you have one, run
  an independent service rather than splitting this model.
- Linux with the `xe` driver, oneAPI 2026.0 (or the version your llama.cpp SYCL
  build was compiled with), `icpx`, and Level Zero.
- 62 GiB host RAM is comfortable; the model is device-resident, but mmap needs page
  cache during load.
- Disk: 20.8 GB for the artifact, plus run outputs.

Map your card first. Never identify a card by a remembered `hwmon` number.

```bash
for d in /sys/class/drm/card*/device; do
  echo "$d -> $(readlink -f "$d")"
done
for h in /sys/class/hwmon/hwmon*; do
  [ "$(cat "$h/name")" = xe ] && echo "$h $(readlink -f "$h/device") cap=$(cat "$h/power1_cap")"
done
# the card with a connected display is the desktop card; use the other one
grep -l connected /sys/class/drm/card*-*/status
```

Record the as-found `power1_cap` of your card and restore it at the end. Do not
raise the cap to make a number look better.

## 3. Get the model

One file, pinned revision, hash checked. Do not download the whole repository.

```bash
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0      # keep the system disk free
mkdir -p /data/models/nanbeige43-33b
hf download bartowski/Nanbeige_Nanbeige4.3-33B-GGUF \
  Nanbeige_Nanbeige4.3-33B-Q4_K_M.gguf \
  --revision 9beb68421fdf7be095093a6888325fa2c83ddd38 \
  --local-dir /data/models/nanbeige43-33b

cd /data/models/nanbeige43-33b
ls -l Nanbeige_Nanbeige4.3-33B-Q4_K_M.gguf      # expect 20788825216 bytes
sha256sum Nanbeige_Nanbeige4.3-33B-Q4_K_M.gguf  # expect 2cef79855ca47ee6571e166db32a07089e34f3cf78e0397b262d124ddb814c37
```

Why this file: it is the only day-0 quant with a model-specific quality number
(producer-reported KLD 0.0600 ± 0.0020 against the unquantized model) **and**
every tensor type lands on the Intel reorder MMVQ path (Q4_K, Q5_K, Q6_K, Q8_0,
including the Q6_K output head). FP8 (35.5 GB) does not fit one card; the public
EXL3 pack is 2.5 bpw with a 2 GB FP16 head read every token.

## 4. Build llama.cpp for SYCL

`GGML_SYCL_F16=ON` is not optional: with it off, dense prefill has measured 3.4×
slower on this hardware. Verify the flag in the build cache rather than trusting a
directory name.

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp
git checkout 1692f9e50        # or any build >= b11845 that opens this GGUF
source /opt/intel/oneapi/setvars.sh
cmake -B build-sycl \
  -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_TARGET=INTEL \
  -DGGML_SYCL_DNN=ON -DGGML_SYCL_GRAPH=ON \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build-sycl -j"$(nproc)"
grep -E 'GGML_SYCL_F16|GGML_SYCL_DNN' build-sycl/CMakeCache.txt   # F16:BOOL=ON
```

## 5. Environment for the campaign card

```bash
source /opt/intel/oneapi/setvars.sh
export ZE_AFFINITY_MASK=1                 # the card WITHOUT the display
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export GGML_SYCL_ENABLE_OPT=1             # default; keep it
export GGML_SYCL_PRIORITIZE_DMMV=0        # forcing DMMV is an unmeasured arm
export GGML_SYCL_ENABLE_ESIMD=1
export GGML_SYCL_ENABLE_GRAPH=0           # SYCL graphs measured ~±1 % on dense
unset SYCL_CACHE_PERSISTENT               # startup-only; stale JIT mixes can crash
```

Check the card is free and empty before every run:

```bash
pgrep -af 'llama-server|llama-bench|vllm'    # must be empty
docker ps                                    # must show no inference container
llama-bench --list-devices                   # expect ~31,000 MiB free on one device
```

## 6. Measure decode and prefill

```bash
M=/data/models/nanbeige43-33b/Nanbeige_Nanbeige4.3-33B-Q4_K_M.gguf
B=~/llama.cpp/build-sycl/bin/llama-bench

# decode at three occupied-context depths + prefill. 3 reps after load.
$B -m $M -p 512,8192 -n 128 -r 3 -ngl 99 -fa on \
   -ctk q8_0 -ctv q4_1 -t 8 -b 2048 -ub 2048 -o json > pp-tg.json
$B -m $M -p 512 -n 128 -d 0,8192,32768 -r 3 -ngl 99 -fa on \
   -ctk q8_0 -ctv q4_1 -t 8 -b 2048 -ub 2048 -o json > depth.json
```

Do **not** add `-dev SYCL0` while `ZE_AFFINITY_MASK` is set; on build 361 that
combination exits with code 3 and no message.

In a shell script, source `setvars.sh` with `set +u` around it: it is not
`set -u` safe and returns non-zero when oneAPI is already initialised, which under
`set -e` kills the launch before llama.cpp prints anything.

`-ub 2048` is the prefill lever. Measured, same artifact and build:

| Setting | pp2048 | pp8192 | decode d0 / 8K / 32K |
|---|---:|---:|---|
| `-ub 512` | 783 tok/s | 754 tok/s | 23.14 / 14.63 / 6.67 |
| `-ub 2048` | **1,175 tok/s** | **1,082 tok/s** | 23.13 / 14.63 / 6.66 |
| `-ub 2048 -b 4096` | 1,175 tok/s | 1,085 tok/s | not separately measured |

Prefill is bound by the per-ubatch dequant-to-F16 round trip, so a larger ubatch
halves the number of round trips. The compute buffer grows with it; at 32K context
the server still left 8.6 GiB free.

## 6b. KV cache: pick the pair by kernel path, not by bytes

Measured on this path (one variable per row, 2 reps each, decode at 32K occupied context):

| KV K / V | KV B/token | Max context at 2 GiB reserve | tg d0 / 8K / 32K |
|---|---:|---:|---|
| **q4_1 / q4_1** | 66,560 | **143,360** | **23.48 / 18.76 / 11.03** |
| q5_1 / q5_1 | 79,872 | 125,952 | — / — / 10.36 |
| q8_0 / q4_1 | 89,856 | 106,496 | 23.14 / 14.63 / 6.67 |
| q4_1 / q4_0 | 63,232 | 150,528 | — / — / 6.94 |
| f16 / f16 | 212,992 | 33,280 | — / — / 6.69 |

The 8-bit K cache was throttling depth decode: going to `q4_1` K makes the cache 25 %
smaller **and** 65 % faster at 32K. `f16` K, twice the bytes, ties `q8_0`; `q4_0` V is
the smallest pair and the slowest of the small ones. Capacity is proven end to end —
`-c 131072` with `q4_1/q4_1` loaded and processed a **93,748-token prompt** (prefill
438 tok/s, decode 5.53 tok/s, needle at that depth recalled), and the prefill rate by
depth on that run was 1,413 tok/s at 6K, 929 at 25K, 689 at 49K, 477 at 86K.

Divergence from the `q8_0/q4_1` pair (greedy, 128 tokens per cell): a code cell stayed
**128/128 identical** (mean |Δlog p| 0.0098); prose diverged at token 12 and reasoning at
token 2 (mean 0.297 / 0.123), with the top-5 candidate set identical in every cell. No
KL/perplexity number is claimed: `llama-perplexity` in this build segfaults (it pins a
compiler version that is no longer installed). If you need a KL guarantee, stay on
`q8_0` K / `q4_1` V and accept 106K context and 6.67 tok/s at 32K.

## 7. Serve it

```bash
~/llama.cpp/build-sycl/bin/llama-server -m $M \
  --device SYCL0 --split-mode none -ngl 99 \
  -fa on -ctk q4_1 -ctv q4_1 \
  -c 32768 -b 2048 -ub 2048 -t 8 -tb 8 -np 1 \
  --mmap --jinja --reasoning-format deepseek \
  --host 127.0.0.1 --port 8084 --no-webui
```

- **Thinking on with a budget, never "off".** `--reasoning off` does not cleanly
  disable thinking on this template: the model writes its plan into `content` and
  derails on short exact answers (3/5 on a five-prompt battery), while
  `--reasoning-budget 800` scores 5/5. Use a budget, and reserve `--reasoning off` for
  bulk code emission.
- **Sampling, exactly as the model card asks.** Agentic and tool use: `temperature 1.0`,
  `top_p 0.95`, up to 65,536 new tokens. Reasoning and chat: `temperature 0.6`, `top_p 0.95`.
  The launcher defaults to the agentic profile; override `TEMP=`/`TOPP=` for chat. Every
  number in section 1 was measured with **greedy** decoding instead, which is repeatable for
  comparison; sampling does not change decode speed.
- `--reasoning-preserve` keeps previous turns' reasoning, the vendor's
  `preserve_thinking=true`, recommended for multi-turn tool use, office tasks and code-agent
  workflows.
- `--jinja` uses the template inside the GGUF. Its marker triple
  (`<tool_call>`, `<function=`, `<parameter=`) makes llama.cpp select its
  Qwen3-Coder XML parser automatically; there is no `--tool-call-parser` flag here.
- Thinking is **on** by default and consumes a lot of the output budget: one
  arithmetic question used 205 of 256 tokens thinking. Use `--reasoning off` or
  `--reasoning-budget N` for short answers, and `--reasoning-preserve` for
  multi-turn tool flows (the vendor recommends keeping prior reasoning in agent
  work).
- `--reasoning-format deepseek` returns thoughts in `reasoning_content` instead of
  `content`.
- Capacity: at 32K context the server reported 19,180 MiB of model, 2,808 MiB of KV
  and 378 MiB of compute. Arithmetic ceiling with a 2 GiB reserve is 95,232 tokens;
  above that, KV must shrink first.

Verify the template and tool path before trusting any agent workload:

```bash
curl -s localhost:8084/props | python3 -c 'import json,sys;d=json.load(sys.stdin);t=d["chat_template"];print(len(t), "enable_thinking" in t, "<tool_call>" in t)'
```

Expected: a ~8,000-character template with both markers present, and a tool-call
request returning `tool_calls[0].function.name == "get_weather"` with parsed
arguments. Measured: that works.

## 8. Measure a served cell (C1, with energy)

The client sends `cache_prompt: false`, one discarded same-shape warmup, then five
samples; it records endpoint token counts, `finish_reason`, `timings` and the card's
`energy1_input` delta. Use `--reasoning-budget 0` if you want the decode rate to
describe content rather than thinking.

```bash
# two cards installed? name the one llama.cpp is pinned to, or the client refuses to guess
python3 client.py --port 8084 --out cells.json --n 5 --gen 128 --pci 0000:0f:00.0
```

With a single Arc card the `--pci` argument can be dropped; with two, a first-match
`hwmon` search would report the display card's power (this happened during the
measurement session and produced a false 57 W), so the client errors instead of
guessing.

Report the median of the five, the range, the actual completion tokens, the cache
state, the configured cap, the measured draw, and the timing source. Never mix a
`llama-bench` rate with a client-observed rate in one column.

## 9. Context ladder (go up slowly)

Each rung needs a real prompt of that size plus a real output, and the reserve
logged after load:

```bash
for C in 16384 32768 65536; do
  ~/llama.cpp/build-sycl/bin/llama-server -m $M --device SYCL0 --split-mode none \
    -ngl 99 -fa on -ctk q8_0 -ctv q4_1 -c $C -b 2048 -ub 2048 -t 8 -tb 8 \
    --mmap --jinja --host 127.0.0.1 --port 8084 --no-webui &
  sleep 45   # load; then read the memory-breakdown lines in the log
  # ... run your 1-request cell at this context, then stop the server ...
  kill %1; wait
done
```

Stop rules: below 500 MiB free abort; 500–1,023 MiB is a capacity observation only;
1,024–2,047 MiB is staged research; 2,048 MiB or more is required for a performance
cell. Above 32K this recipe has **no** measurement — expect decode to keep falling
(measured slope: −3.26 µs of token time per extra context token, i.e. 6.67 tok/s at
32K).

## 10. Draft-free speculation — the one big decode lever that works

Add these flags and decode on repetitive text jumps; the verify batch must stay at 8:

```bash
  --spec-type ngram-simple \
  --spec-ngram-simple-size-n 4 \
  --spec-ngram-simple-size-m 7      # batch 8. size_m 8 => batch 9 => slower than plain
```

Matched pair, same prompt and flags, fresh processes, 128 output tokens:

| Prompt | plain | with `size_m 7` |
|---|---:|---:|
| 777-token repeated code, acceptance 99/127 | 21.80 tok/s | **37.80 tok/s (+73.3 %)** |
| 12,844-token log, acceptance varies 0/26–51/104 | 12.16 tok/s | 11.7–15.8 (median ~13.4) |

Scope it honestly: the win comes from text that repeats itself (code, logs, agent
transcripts pasted back). Novel prose is expected neutral and has not been measured
with `size_m 7`. Keep the no-spec arm in every comparison, and check that greedy
output matches the no-spec run before making it a default for a workload.

## 11. What does not help (measured or closed)

| Attempt | Result |
|---|---|
| N-gram / draft-free speculation with `size_m 8` | 21.80 → **15.12 tok/s**. Negative control: `--spec-draft-n-max` does *not* cap the draft, `--spec-ngram-simple-size-m` does; a 9-wide batch leaves the 8-wide reorder-MMVQ window |
| F16 KV instead of `q8_0`/`q4_1` at 32K | 6.69 vs 6.66 tok/s — depth cost is attention latency, not KV bytes |
| `q8_0` K + `q8_0` V, and `q4_0`/`q4_0` at 32K | 4.61 and 4.84 tok/s — worse than the fleet pair |
| `-b 4096` with `-ub 2048` | no change |
| Thinking left unbounded | correct but slow: 205 of 256 output tokens were reasoning on a one-line arithmetic question |
| Verify batches above 8 | every weight matrix is dequantized to f16 and run through oneDNN, so a 9-wide batch loses more than the extra tokens win |
| `llama-bench` for the dispatch log | it suppresses INFO; use `llama-server -lv 4` with `GGML_SYCL_MKL_FA_DEBUG=1` |
| A draft model for speculation | none exists: the 3B student is unreleased and older Nanbeige 3B vocabularies (166,144) do not match this one (166,912) |

## 12. Cleanup

```bash
pkill -f 'llama-server -m /data/models/nanbeige43-33b'   # only your own process
pgrep -af 'llama-server|llama-bench'                     # must be empty
cat /sys/class/hwmon/hwmon6/power1_cap                   # confirm the as-found value
```

Delete the run directory only after copying its JSON and logs somewhere you keep.

## 13. Coding quality and the reasoning budget

Greedy, single-file HTML tasks, each verified in Chromium through the app's own handlers
(not by reading the source):

| Task | Output tokens | Result |
|---|---:|---|
| Todo list, `localStorage`, clear-done | 3,239 | **6/6** checks pass |
| 15×15 BFS visualiser, animation, unreachable case | 7,722 | **6/6** checks pass |
| Paint: undo, clear, PNG export, pointer-leave (greedy) | 1,375 | **5/6**: the first `undo` is a no-op (snapshots pushed after the action) |
| Paint, same prompt at the card's agentic sampling (T=1.0, top_p=0.95) | 1,584 | **5/6, different defect**: stroke undo now works (2111 → 1214 px), but undo-after-clear still does not restore |
| Voxel-art pagoda garden, single file, "use any library" | 16,496 | **runs in Chrome**: 39.5 KB, three.js from unpkg, 137,775 voxels, 25 material groups, no elisions. Defects: `pick()` given scalar palette entries leaves three element families in the default colour (3 unique `THREE.Color: Unknown color` warnings), and the default camera starts inside the cherry canopy |

**Thinking must be budgeted for code emission.** Thinking is on by default and can consume
the whole completion budget: two of three tasks returned **empty content** after 12,000
reasoning tokens, and with `--reasoning-budget 1500` the paint task truncated and inserted
a literal `...` inside a function. Turning thinking off produced the only complete paint
app, 5–8× faster per finished file:

```bash
  --reasoning off                 # or --reasoning-budget 1500 for tasks that need planning
```

## 14. Evidence

Measured 2026-10-25, one B70, `0000:0f:00.0`, build 361 (`1692f9e50`), artifact
sha256 `2cef7985…4c37`, 230 W cap, no container. Raw llama-bench JSON, server logs,
quality probe, n-gram A/B samples and energy cells are kept with the run record
(private lab). Verified in a real browser: the todo app passed 6/6 acceptance
checks including a `localStorage` round-trip; the BFS visualiser passed 6/6 including the
unreachable case; the paint app passed 5/6 with one real state-management defect
(off-by-one undo). Decode attention is confirmed **VEC** on this path
(`GGML_SYCL_MKL_FA_DEBUG=1` at `-lv 4`: 104/104 `[FA-DISP] VEC D=128 n_kv=256`, zero
TILE/MKL/ONEDNN), which is why context costs so much at M=1.
Validation tier: measured on one host, one artifact, one build, no
independent reproduction. Quality was smoke-tested only (5/5 short prompts
including one tool call); no benchmark suite was run, so this page makes **no**
agentic-quality claim.
