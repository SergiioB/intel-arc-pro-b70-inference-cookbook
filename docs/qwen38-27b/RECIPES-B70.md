# Qwen3.8-27B on Arc Pro B70: recipes

Updated 2026-11-19. Copy-paste recipes for one and two Intel Arc Pro B70 cards, each with its build, flags, serve
command, what to expect and the cap setting it was measured at. Status: **screen** (one host). Raw files and
`summarize.py`: [`results/qwen38-27b/recipes-equal-caps-20261119/`](../../results/qwen38-27b/recipes-equal-caps-20261119/README.md).

The first-run route stays [EXL3 + MTP3](EXL3-XPU.md) ([START](../../START.md)): the model card's sampling
comparison that would decide between it and TensorFold is still pending (2026-11-19). The numbers below are greedy.

## Pick a recipe

| You have | Recipe | Expect (code prompt, decode tok/s after the first token) |
|---|---|---|
| Any B70 host | [0. Set both cards to 230 W and 2,800 MHz](#0-host-both-cards-at-230-w-and-2800-mhz) | two cards 5–6 % faster than with one card capped lower |
| One user, one card | [2. TensorFold, one card](#2-one-user-one-card) | 127.8 at 68 tokens, 114.6 at 8K, 90.0 at 32K, 50.7 at 131K |
| One user, two cards | [3. TensorFold `--tp 2`](#3-one-user-two-cards) | 173.3 at 68 tokens, 151.4 at 8K, 123.8 at 32K, 81.6 at 131K |
| Prompts up to 250K | [4. 250K on two cards](#4-250k-context-on-two-cards) | 62.3 at 249,784 tokens (bf16 K/V) |
| Chats that come back to a long context | [5. Warm multi-turn sessions](#5-warm-multi-turn-sessions) | follow-up turns start in 0.3–1.6 s instead of re-reading the prompt |
| Several users, one card | [6. Many users](#6-many-users-on-one-card) | TensorFold up to 4 streams, exl3xpu from 8 streams |
| GGUF on two cards | [7. llama.cpp `-sm tensor`, local patch](#7-llamacpp-gguf-on-two-cards-local-patch) | 32.2 tg128 against 28.3 on one card |

## 0. Host: both cards at 230 W and 2,800 MHz

Both cards run the same power cap and clock limit. Until 2026-11-19 09:45 CET this host ran card 0 at 180 W with a
2,400 MHz clock cap (a cooler profile that a tuning script applied to the first xe card only) and card 1 at 230 W.
On two cards every exchange waits for the slower card.

Set and check it through sysfs (xe driver; resets at reboot, so run it from a boot-time service):

```bash
for c in /sys/class/drm/card[0-9]*; do
  case $c in *-*) continue ;; esac                                   # skip connector entries
  [ "$(basename "$(readlink -f "$c/device/driver")")" = xe ] || continue
  hw=$(echo "$c"/device/hwmon/hwmon*)
  echo 230000000 | sudo tee "$hw/power1_cap" >/dev/null              # microwatts
  echo 2800 | sudo tee "$c/device/tile0/gt0/freq0/max_freq" >/dev/null   # MHz
  printf '%s %s cap %s uW max_freq %s MHz\n' "$(basename "$c")" "$(basename "$(readlink -f "$c/device")")" \
    "$(cat "$hw/power1_cap")" "$(cat "$c/device/tile0/gt0/freq0/max_freq")"
done
# expect, on every card:  cardN 0000:xx:00.0 cap 230000000 uW max_freq 2800 MHz
```

Measured effect, served, same binary (`2286abc0c`), two cards, greedy, 512 tokens out, card 0 moved from
180 W / 2,400 MHz to 230 W / 2,800 MHz, card 1 at 230 W / 2,800 MHz both times:

| Prompt tokens | 180/230 W, drafted | 230/230 W, drafted (two runs) | Serial, before → after | First token, before → after |
|---|---|---|---|---|
| 68 | 165.2 | 173.3 / 173.7 | 48.4 → 50.5 | 0.09 → 0.09 s |
| 7,976 | 142.6 | 151.4 / 151.5 | 44.5 → 46.8 | 3.54 → 3.50 s |
| 32,552 | 117.3 | 123.8 / 123.7 | 42.7 → 44.6 | 15.88 → 15.58 s |
| 130,856 | 76.8 | 81.6 / 81.3 | 31.2 → 32.1 | 86.10 → 83.11 s |

- Same text in 24 of 24 cells (drafted, drafted repeat and serial; code and prose).
- Dev runner, 68 tokens: a two-card drafted round 33.1 → 31.65 ms (151.3 → 158.6 tok/s); the "card 0 slower" part
  of the round goes from 1.5 ms to 0. One card unchanged (123.4 → 123.6 tok/s).

**Host cleanup on 2026-11-19.** One boot service now applies 230 W / 2,800 MHz to every xe GPU (it had applied the
180 W / 2,400 MHz cooler profile to the first card only). A user-level unit that wrote 180 W to both cards at login
and two user-level units that restarted an old overnight sweep at login were disabled the same day (all three show
`disabled`; the cap unit stays loaded for the login it already ran in, and the live caps read 230 W). If you copied an
older setup from this host, check for login-time units that write `power1_cap` and keep one authority:

```bash
systemctl --user list-unit-files --state=enabled | grep -i -E 'power|cap|b70'
grep -l power1_cap ~/.config/systemd/user/*.service /etc/systemd/system/*.service 2>/dev/null
```

**Which numbers were measured at which setting.** One-card numbers on these pages ran on card 1 at 230 W and are
not affected. Two-card numbers measured **before 2026-11-19 09:45 CET ran at 180/230 W**: the first two-card record
([TENSORFOLD-B70](TENSORFOLD-B70.md#two-cards-for-one-stream-2026-11-18-13-to-32k-15-at-131k-250k-prompt-with-bf16-kv)),
everything two-card on [the 250K page](LONG-CONTEXT-250K-20261119.md), the prompt-length sweep and its
[intelinside.ai submission](https://intelinside.ai/runtimes/tensorfold/custom/29), the several-stream two-card screens
and the 33.1 ms budget. Measured **at 230/230 W**: the table above, the 31.65 ms budget, the speed-lever screens and
the llama.cpp two-card runs below.

## 1. Build TensorFold (one build for every TensorFold recipe)

```bash
git clone https://github.com/SergiioB/TensorFold && cd TensorFold
git checkout xpu-tp2-fixes-20261119        # ac346900b: PR #3586 (--tp 2) + prefix attention kernel + two fixes
# ocloc 26.05 rejects one 128-GRF flag used only by Gemma 4 kernels; skip if your ocloc accepts it:
git apply /path/to/results/qwen38-27b/recipes-equal-caps-20261119/build/ocloc-26.05-build.diff
zig build xpu-native -Doptimize=ReleaseFast -Dngen-include=/path/to/onednn/third_party/ngen   # Zig 0.17.0
# weights (the measured revision of the target is 22d8d538):
hf download TensorFold/Qwen3.8-27B-MLX-4bit --local-dir Qwen3.8-27B-MLX-4bit
hf download z-lab/Qwen3.8-27B-DFlash2 --local-dir Qwen3.8-27B-DFlash2
```

- Host stack measured: Linux 7.0.0-28, Intel compute-runtime 26.05.37020.3, Level Zero loader 1.28.2.
- The served numbers were taken on `2286abc0c` (the same tree without the two fixes). Builds on top of `ac346900b`
  ran at the same speed: 173.6 / 151.5 / 123.8 / 81.6 tok/s at 68 / 8K / 32K / 128K in the flag-off arms of the
  lever screens below, against 173.3 / 151.4 / 123.8 / 81.6 on `2286abc0c`. The fixes matter for several streams on
  two cards (fix B) and warm sessions past about 131K with int8 K/V (fix A); see
  [TENSORFOLD-TWO-CARDS](TENSORFOLD-TWO-CARDS-20261119.md#fix-b-a-shared-draft-tree-took-another-attention-path).

## 2. One user, one card

```bash
XPU_ATTN_PREF=256 ZE_AFFINITY_MASK=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE zig-out/xpu/bin/tensorfold-native serve \
  Qwen3.8-27B-MLX-4bit --drafter Qwen3.8-27B-DFlash2 --tp 1 --parallel 1 --context 139264 \
  --prompt-cache-gib 0 --no-thinking --no-update-check --host 127.0.0.1 --port 8046
# ZE_AFFINITY_MASK picks the card (0 or 1). For long chats replace --prompt-cache-gib 0, see recipe 5.
```

Expect (card 1, 230 W, greedy, 512 tokens out; drafted output equals `"draft": false` output):

| Prompt tokens | Drafted, code / prose | Serial | First token |
|---|---|---|---|
| 68 / 52 | 127.8 / 102.2 | 32.4 | 0.12 s |
| 7,976 / 7,963 | 114.6 / 91.1 | 30.7 | 4.63 s |
| 32,552 / 32,540 | 90.0 / 66.3 | 27.6 | 21.5 s |
| 130,856 / 130,846 | 50.7 / 45.6 | 18.9 | 128.9 s |

Up to 131,072 tokens fit with bf16 K/V. For a single cold prompt of 32K or more, exl3xpu reaches the first token
1.5–1.6× sooner, and at 250K on one card it decodes 1.8× faster
([greedy comparison](ROUTE-GREEDY-SAME-SESSION-20261119.md)).

## 3. One user, two cards

```bash
XPU_ATTN_PREF=256 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE zig-out/xpu/bin/tensorfold-native serve \
  Qwen3.8-27B-MLX-4bit --drafter Qwen3.8-27B-DFlash2 --tp 2 --parallel 1 --context 139264 \
  --prompt-cache-gib 0 --no-thinking --no-update-check --host 127.0.0.1 --port 8046
# do not set ZE_AFFINITY_MASK: both cards must be visible
```

Expect (230/230 W, two drafted runs a cell, drafted == serial in every cell):

| Prompt tokens | Drafted, code | Drafted, prose | Serial | First token |
|---|---|---|---|---|
| 68 / 52 | 173.3 / 173.7 | 133.9 / 134.4 | 50.5 | 0.09 s |
| 7,976 / 7,963 | 151.4 / 151.5 | 113.6 / 113.9 | 46.8 | 3.50 s |
| 32,552 / 32,540 | 123.8 / 123.7 | 90.8 / 91.1 | 44.6 | 15.6 s |
| 130,856 / 130,846 | 81.6 / 81.3 | 71.5 / 72.0 | 32.1 | 83.1 s |

Two cards give 1.36× one card at 68 tokens and 1.61× at 131K on code (one-card cells from 2026-11-18, same
binary). Text differs from the one-card text from
about token 26 (each setup is exact within itself).

**Faster drafter attention, screen (not yet on the branch).** `XPU_DATTN_SPLIT=1` splits the drafter's block attention
over key chunks (commit `f352f8c89` on `ac346900b`, not pushed; patch
`tptf-speed/0001-xpu-qwen27-XPU_DATTN_SPLIT-splits-the-drafter-s-bloc.patch`). Same binary, flag off against on, 3
interleaved pairs, 230/230 W, median:

| Prompt tokens | Code off → on | Prose off → on | Gain (code) |
|---|---|---|---|
| 68 / 52 | 173.6 → 176.9 | 134.4 → 136.9 | +1.9 % |
| 7,976 / 7,963 | 151.5 → 166.1 | 113.9 → 125.0 | +9.6 % |
| 32,552 / 32,540 | 123.8 → 134.8 | 91.5 → 99.8 | +9.0 % |
| 130,856 / 130,846 | 81.6 → 85.6 | 71.7 → 76.5 | +4.9 % |

Same token hash in 8 of 8 cells and drafted == serial with the flag on. Status: **screen**; the five-pair confirmation
and the one-card run are **pending** (2026-11-19). Not part of the recipe yet.

## 4. 250K context on two cards

```bash
# fastest, no prompt cache: bf16 K/V
XPU_ATTN_PREF=256 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE zig-out/xpu/bin/tensorfold-native serve \
  Qwen3.8-27B-MLX-4bit --drafter Qwen3.8-27B-DFlash2 --tp 2 --parallel 1 --context 262144 \
  --prompt-cache-gib 0 --no-thinking --no-update-check --host 127.0.0.1 --port 8046
# warm follow-ups at 250K (needs the xpu-tp2-fixes-20261119 build): int8 K/V and a 5 GiB cache
XPU_ATTN_PREF=256 XPU_KV_INT8=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE zig-out/xpu/bin/tensorfold-native serve \
  Qwen3.8-27B-MLX-4bit --drafter Qwen3.8-27B-DFlash2 --tp 2 --parallel 1 --context 262144 \
  --prompt-cache-gib 5 --no-thinking --no-update-check --host 127.0.0.1 --port 8046
```

Expect at a 249,784-token prompt, **measured at 180/230 W** (the 230/230 W rerun is pending, 2026-11-19):

| Setup | Drafted | Serial | First token | Follow-up first token |
|---|---|---|---|---|
| bf16 K/V, no cache | 62.31 | 23.07 | 218.8 s | – |
| int8 K/V, no cache | 32.46 | 22.84 | 206.9 s | – |
| int8 K/V, 5 GiB cache, fix branch | – | – | 205.0 s | 1.63 / 1.07 / 1.11 s, resumed == fresh 4 of 4 |

bf16 K/V with a 16 GiB cache does not load on two cards; without the fix, resumed int8 turns past about 131K fail
`InvalidQwenToken`. Details: [250K page](LONG-CONTEXT-250K-20261119.md).

## 5. Warm multi-turn sessions

TensorFold keeps the prompt state of finished turns in a VRAM budget, `--prompt-cache-gib`. Size it to hold at least
one state of the longest session; the 2 GiB default re-reads any context past about 29K on every turn.

| K/V | One card, a state | Two cards (`--tp 2`), a state as the server logs it |
|---|---|---|
| bf16 | 64 KiB a token + about 0.2 GiB (2,219 MiB at 32,552 tokens) | about half: 8,340 MiB for 2 states at 130,856 tokens |
| int8 | about 32 KiB a token + 0.25 GiB (4,339 MiB at 130,856 tokens) | about half: 2,201 MiB at 131,549 tokens, 4,077 MiB at 249,784 |

Measured settings (the server line `prompt cache: … N states, M MiB` shows what is kept):

| Session | Setting | Follow-up first token | Caps |
|---|---|---|---|
| 32K, one card, bf16 | `--context 40960 --prompt-cache-gib 3` | 0.31–0.57 s | card 1 230 W |
| 64K, one card, bf16 | `--context 73728 --prompt-cache-gib 5.5` | 0.42–0.75 s | card 1 230 W |
| 131K, one card, int8 | `XPU_KV_INT8=1 --context 139264 --prompt-cache-gib 5.5` | 0.65–1.06 s | card 1 230 W |
| 32K, two cards, bf16 | `--tp 2 --context 40960 --prompt-cache-gib 3` | 0.30–0.49 s, resumed == fresh 5 of 5 | 180/230 W |
| 131K, two cards, bf16 | `--tp 2 --context 139264 --prompt-cache-gib 9.5` | 0.65–0.83 s, resumed == fresh 5 of 5 | 180/230 W |
| 250K, two cards, int8, fix branch | `XPU_KV_INT8=1 --tp 2 --context 262144 --prompt-cache-gib 5` | 1.07–1.63 s, resumed == fresh 4 of 4 | 180/230 W |

At 250K only int8 on two cards fits a state beside the stream; one card refuses a 262,144-token int8 stream with any
cache of 4.1 GiB or more.

## 6. Many users on one card

Short prompts, 512 tokens a stream, all streams at once, three passes a level, card 1 at 230 W. Median aggregate tok/s
(range):

| Streams | TensorFold `--parallel 32 --context 4096` | exl3xpu MTP3, `max_num_seqs` 32 | Text equal to the solo run, TF / exl3xpu |
|---|---|---|---|
| 1 | 130.1 (130.0–130.1) | 82.5 (81.5–84.8) | 3 of 3 / 2 of 3 (the third pass differed) |
| 4 | **242.4** (240.7–242.5) | 220.0 (118.4–220.4) | 12 of 12 / 5 of 12 |
| 8 | 270.9 (268.5–272.7) | **342.2** (334.7–350.7) | 24 of 24 / 8 of 24 |
| 16 | 297.7 (295.0–297.8) | **333.3** (318.6–334.3) | 48 of 48 / 7 of 48 |
| 32 | 297.5 (297.4–297.5) | **374.8** (373.1–375.9) | 96 of 96 / 27 of 96 |

Use TensorFold up to 4 streams or when every user must get the same answer for the same request; exl3xpu from 8
streams when aggregate throughput matters more. With both cards free and several users, run one server a card: at 4
streams two one-card TensorFold servers gave 159–161 tok/s against 129–130 for one two-card server (180/230 W).

```bash
# TensorFold, one card, up to 32 streams
XPU_ATTN_PREF=256 ZE_AFFINITY_MASK=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE zig-out/xpu/bin/tensorfold-native serve \
  Qwen3.8-27B-MLX-4bit --drafter Qwen3.8-27B-DFlash2 --tp 1 --parallel 32 --context 4096 \
  --prompt-cache-gib 0 --no-thinking --no-update-check --host 127.0.0.1 --port 8051
# exl3xpu, one card, MTP3
docker run -d --network=host --device=/dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro --shm-size=8g \
  -v /path/to/exl3-4.00bpw:/mnt/exl3model:ro -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e UR_L0_USE_COPY_ENGINE=0 -e SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0 \
  ghcr.io/sergiiob/exl3xpu-mixedk@sha256:933eda54bb51a2e8a6669a70f5a2d7696fabd12a5f8b7c469df02fdfecc79d3a \
  models/qwen3.8-27b-exl3-4.00bpw --gpu 1 --port 8051 --model-path /mnt/exl3model \
  --set vllm.gpu_memory_utilization=0.92 --set vllm.max_model_len=16384 --set vllm.max_num_seqs=32 \
  --set vllm.enable_prefix_caching=false --set vllm.reasoning_parser=qwen3
```

Two-card and two-server arms with three passes: **pending** (2026-11-19).

## 7. llama.cpp GGUF on two cards (local patch)

Stock llama.cpp `-sm tensor` is slower on two B70s than one card. Our local patch adds a peer-store allreduce
(`GGML_SYCL_TP_P2P=2`): each card stores its F32 partial into the other card's memory and the add kernel waits on a
flag, with no host wait. Patches and checks: [`patches/llamacpp-sycl/`](../../patches/llamacpp-sycl/) and the
[image and patch matrix](../IMAGE-AND-PATCH-MATRIX.md). Not proposed to llama.cpp.

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 78e05298f0d68abff41d08d79fa7469264b2aff0
git apply /path/to/cookbook/patches/llamacpp-sycl/sycl-tp-p2p-0001-peer-copy-allreduce.patch
git apply /path/to/cookbook/patches/llamacpp-sycl/sycl-tp-p2p-0002-peer-store-mode-2.patch
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DGGML_SYCL=ON -DGGML_SYCL_F16=OFF -DGGML_SYCL_DNN=ON -DGGML_SYCL_GRAPH=ON \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_MTMD=OFF
cmake --build build --parallel 10 --target llama-bench llama-server
export SYCL_ENABLE_DEFAULT_CONTEXTS=1 UR_LOADER_USE_LEVEL_ZERO_V2=0 ONEAPI_DEVICE_SELECTOR='level_zero:*'
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 SYCL_CACHE_PERSISTENT=0 GGML_SYCL_ENABLE_GRAPH=0 GGML_SYCL_TP_P2P=2
./build/bin/llama-server -m qwen38-27b-Q4_K_M.gguf -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -b 512 -ub 512 -t 8 \
  -lm none -dev SYCL0,SYCL1 -sm tensor -ts 1,1 --fit off -c 40960 -np 1 --reasoning off --host 127.0.0.1 --port 8046
```

| Patch (verbatim, apply in this order on `78e05298f`) | sha256 |
|---|---|
| `sycl-tp-p2p-0001-peer-copy-allreduce.patch` (adds `GGML_SYCL_TP_P2P`, mode 1) | `5f25dfed9bb7a1f30f4a6a1cee2814a6c0bf698870a75da121dc9ac142ffc71b` |
| `sycl-tp-p2p-0002-peer-store-mode-2.patch` (adds mode 2, the measured one) | `9a66ca4949e61b3aa344c1d2ee600c72b33639fe671a125277e4f40daca9da2a` |

GPU-free check: both apply cleanly in that order on `78e05298f` (2 files, +171/−5; `git apply`, 2026-11-19).
Measured binary: that tree plus a measurement-only switch that skips every allreduce when set (unset in every arm
below except the bound). Unset `GGML_SYCL_TP_P2P` to get the stock path back.

Expect (Qwen3.8-27B Q4_K_M, q8_0 K/V, llama-bench `-r 3`, 230/230 W):

| Setup | tg128 tok/s | pp512 tok/s | Served decode, 76–78-token prompt |
|---|---|---|---|
| One card (`-sm none`) | 28.28 ± 0.02 | 319.07 ± 1.16 | 28.11 |
| Two cards, stock `-sm tensor` | 23.28 ± 0.25 | 558.11 ± 2.88 | 23.60 |
| Two cards, `GGML_SYCL_TP_P2P=2` | **32.18 ± 0.03** | 544.37 ± 1.73 | 31.89 |

- Mode 2 is 1.14× one card and 1.38× stock on tg128. The served reply is the same text on all three setups.
- Skipping every allreduce (wrong output, a bound for this split) gives 38.09 ± 0.04 tg128.
- Per allreduce at decode size: stock 49–52 µs, mode 2 25.8–26.5 µs. At prefill size (2,621,440 values) mode 2 is
  slower: 1,175 µs against 789 µs, which is why pp512 drops 2.5 %.
- Sums are exact F32 on both cards: 210,370,432 values at 6 sizes.
- Host memory still grows 1.02 KiB a call in mode 2 (stock 2.58 KiB; stock also leaves 224,800 Level Zero events
  undestroyed in the same test, about 2 an allreduce). Unexplained.
- Mode 1 (`GGML_SYCL_TP_P2P=1`, peer copy with cross-queue events) is **not for use**: it grows 5.16 KiB a call and
  segfaulted in both check runs.
- One served cell a setup at a short prompt; depth runs, repeated screens and a confirmation are **pending**
  (2026-11-19).

## Do not

| Setting | Measured result |
|---|---|
| `NEOReadDebugKeys=1 EnableDirectSubmission=0` (Intel direct submission off) | TensorFold −17 to −23 % on two cards, −6 % on one ([DNR-020](../DO-NOT-REPEAT.md)) |
| `SYCL_CACHE_PERSISTENT=1` with DPC++ 2026.1 for llama.cpp | llama-bench segfaults in `PersistentDeviceCodeCache::getItemFromDisc` on its first kernel load; `SYCL_CACHE_PERSISTENT=0` runs ([DNR-031](../DO-NOT-REPEAT.md)) |
| Cards at different caps for `--tp 2` | the slower card sets the pace; 1.5 ms of a 33.1 ms round ([recipe 0](#0-host-both-cards-at-230-w-and-2800-mhz)) |
| `XPU_GU_WIDE64=1` or `XPU_GU_KERNEL=d16/v32/w8` (other gate/up kernels on two cards) | slower in every cell: −1.1 to −1.9 % (3 pairs), −0.6 to −6.1 % (1 pair each) ([DNR-030](../DO-NOT-REPEAT.md)) |
| unitrace `--device-timing` on `--tp 2` | segfault, and once an xe engine fault on both cards ([DNR-022](../DO-NOT-REPEAT.md), [DNR-028](../DO-NOT-REPEAT.md)) |
| vLLM XPU v0.31.0 AutoRound int4 on an AMD Zen 3 CPU | `invalid opcode` (AVX-512) in its kernel library ([DNR-027](../DO-NOT-REPEAT.md)) |
| llama.cpp `GGML_SYCL_TP_P2P=1` | leaks and segfaults ([DNR-032](../DO-NOT-REPEAT.md)) |

## Pending (as of 2026-11-19 12:25 CET; queued or running)

The model card's sampling comparison; 250K and warm-session reruns at 230/230 W; `XPU_DATTN_SPLIT` confirmation and
one-card run; two-card many-user arms; llama.cpp depth runs and confirmation; one-card and exl3xpu prompt-length
sweeps; a 2-hour stability run.
