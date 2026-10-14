---
title: "Night prep 2026-09-25 — flash-next C1 screen: artifact verification, offline exact-token calibration, runtime gate"
type: worklog
date: 2026-09-25
status: final
---

# Night prep 2026-09-25 — flash-next C1 screen (CPU-only session)

Prepared for the 22:00 research night. Twelve prior attempts failed on thermal
and live-calibration grounds (RL-057); this prep removes the live-calibration
failure mode entirely. No GPU work was performed: no server launch, no model
load, no GPU workload, nothing modified on the model tree or artifacts.

## 1. Artifact verification table (all checked 2026-09-25, this session)

| Artifact | Path | Size / identity | Status |
|---|---|---|---|
| Main GGUF (AP-Q4_K_XL) | `/mnt/<nvme-disk>/models/signal-3.8-flash-next/hf-dl/AP-Q4_K_XL/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf` | 101,142,769,984 bytes (~94.2 GiB), sha256 `427a7a10dd5a98d77c71be62a89d14e7408fc61a84c6efe889393852282cd3c6` | VERIFIED (full-file sha256 recomputed; matches both RL-055 manifests) |
| MTP GGUF (Q8_0) | `/mnt/<nvme-disk>/models/signal-3.8-flash-next/hf-dl/Qwen3.8-Flash-Next-MTP-Q8_0.gguf` | 4,135,893,376 bytes | present, untouched (not part of tonight's cells) |
| mmproj (F16) | `/mnt/<nvme-disk>/models/signal-3.8-flash-next/hf-dl/mmproj-F16.gguf` | 904,004,128 bytes | present, untouched |
| Serving binary | `/mnt/<sata-disk>/qwenflash-llama-pinned-9723942/build-qwen4exp/bin/llama-server` | sha256 `7c2d6adc4ac5dfa6f1deb8db8d545d8e4cae390ed9c1cd255d1e7f71a83e84a1` | VERIFIED (matches RL-055 manifests; icx/icpx 2026.1.1 matched build) |
| Pinned tree | `/mnt/<sata-disk>/qwenflash-llama-pinned-9723942` | `9723942` + uncommitted qwen4exp/SYCL patches (see tree `git status`) | unchanged; tokenizer path read from source this session |

GGUF tokenizer metadata (read from the header this session): model `gpt2`,
pre `qwen35`, 248,320 tokens, 247,587 merges, `add_bos_token=0`, no
`add_eos_token` key, no `ignore_merges` key, bos id 248044 (`<|endoftext|>`),
eos id 248046 (`<|im_end|>`), arch `qwen4exp`. Consequence: a raw
`/completion` prompt is tokenized with NO added special tokens, and the
server prompt count must equal the plain BPE token count of the text.

## 2. Ready-to-run launch checklist (RL-055 corrected placement, verbatim)

Reproduced from `results/2026/09/flashnext-ap-placement-smoke-20260923T113433Z/manifest.txt`
and `.../flashnext-ap-placement-correctness-20260923T114633Z/manifest.txt`
(scripts `scripts/tmp/flashnext-ap-placement-smoke-20260923.sh` and
`scripts/tmp/flashnext-ap-placement-correctness-20260923.sh`; server launch
lines are identical in both). NOT executed by this prep session.

Preconditions (RL-055 discipline, all gates mandatory):

1. `pgrep -af 'llama-server|llama-bench|vllm|vllm-xpu'` empty; `docker ps` empty.
2. Both cards ≥ ~31,000 MiB `visible_avail` (`sudo cat /sys/kernel/debug/dri/0000:0b:00.0/tile0/vram_mm`, same for `0000:0f:00.0`).
3. Both named-`xe` package sensors ≤ 55 C (hwmon resolved dynamically; never hard-coded).
4. Port 8010 free.
5. Caps as found (do not change caps mid-run; restore standing values after).

Environment + launch (RL-055 exact, with the screen-harness `-c 9216`
variant for the p8192 cell — `flashnext-ap-c1-screen-20260923.sh` used
`-c 9216`; RL-055 correctness used `-c 8192`):

```bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_CACHE_PERSISTENT=0
export SYCL_DEVICE_FILTER=level_zero ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE

BIN=/mnt/<sata-disk>/qwenflash-llama-pinned-9723942/build-qwen4exp/bin/llama-server
MODEL=/mnt/<nvme-disk>/models/signal-3.8-flash-next/hf-dl/AP-Q4_K_XL/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf

setsid "$BIN" -m "$MODEL" \
  --device SYCL0,SYCL1 --tensor-split 49,51 --split-mode layer -ngl 99 \
  -ot 'per_layer_token_embd=CPU,blk\.([0-5]|4[2-7])\.ffn_(up|gate|down)_exps\.weight=CPU' \
  -c 9216 -fa on -t 6 -tb 14 -np 1 -b 512 -ub 512 \
  --cache-type-k q8_0 --cache-type-v q4_1 \
  --no-warmup --no-cache-prompt --metrics --slots --jinja \
  --host 127.0.0.1 --port 8010
```

Expected load behavior (RL-055, same artifact + binary):

- /health reached in ~75–90 s after launch.
- Post-load free VRAM: `-c 8192` run left GPU0 3,372 MiB / GPU1 7,050 MiB
  free; `-c 9216` sessions left GPU0 3,515 / GPU1 7,036 MiB free — both
  above the 2,048 MiB post-load gate. If tonight's numbers land materially
  lower, stop and triage before any request.
- No DEVICE_LOST/OOM lines in the server log.

Expected smoke answers (RL-055 correctness trio, chat endpoint, temp 0):

- "Reply with exactly B70-FLASH-OK and nothing else." → contains `B70-FLASH-OK`
- "Calculate 37 * 19. Give only the integer." → contains `703`
- "Name the capital of France. Give only the city name." → contains `Paris`

## 3. Offline exact-token calibration (the RL-057 repair)

Method. The tokenizer was extracted from the GGUF's own `tokenizer.ggml.*`
keys (248,320 tokens / 247,587 merges) and the BPE was reimplemented in
python as a line-by-line port of the code the pinned binary runs:
`src/unicode.cpp::unicode_regex_split_custom_qwen35` (the `pre=qwen35`
dispatch bypasses std::wregex entirely), `unicode_byte_encoding_process`
(GPT-2 byte map), and `src/llama-vocab.cpp::llm_tokenizer_bpe_session::tokenize`
(min-rank heap, ties by smaller left index, per-byte fallback).
`add_bos`/`add_eos` are false in this GGUF, so no special tokens are added.
Port restriction: ASCII-printable + `\n` input only (asserted); all prompt
files are pure ASCII.

Files (all under `~/B70-DOCS/scripts/tmp/flashnext-calib-20260925/`):

| File | Content |
|---|---|
| `flashnext_bpe.py` | exact tokenizer port; `--extract` builds the cache, `--text-file` encodes/verifies |
| `tokenizer_cache.json` | vocab + merges + types extracted from the GGUF header |
| `build_prompts.py` | constructs and calibrates the prompt files |
| `p512.txt` | decode-screen prompt: exactly **512** tokens, 2,592 bytes, sha256 `b82744d1bc2a3c2491c5cbc325c8db8fa7466aba70a0e150eb23c94192bd384b` |
| `p512.meta.json` / `p512.ids.json` / `p512.verify.json` | metadata, exact `input_ids`, independent re-verify from file bytes |
| `p8192.txt` | cold-input prompt: exactly **8192** tokens, 45,486 bytes, sha256 `4036cdffeb848e76faa2335a8d84e081708f3227b0a3e5ec08579e748746236e` |
| `p8192.meta.json` / `p8192.ids.json` / `p8192.verify.json` | as above |
| `calibration-summary.json` | both cells' counts + hashes |
| `verify_prompt_n.sh` | runtime gate (section 4) |

Structure: `ENTROPY-<64 decimal digits>\n` + natural-language body. Token
accounting: `ENTROPY-` = 4 tokens, the 64-digit block = exactly 64 tokens
(every digit is its own pre-split word under the qwen35 splitter and each
single digit is exactly one token regardless of value), `\n` = 1 — a fixed
69-token header; the body is calibrated to the remaining target. Because the
digit block's token count is value-invariant, tonight's harness may
substitute fresh 64-digit entropy per request for the cache guard WITHOUT
recalibrating. Files contain the exact prompt bytes with NO trailing
newline; do not "fix" them with editors. Round-trip decode of the id
sequence reproduces the file text exactly; zero special/control tokens in
both id sequences.

Known limitation (honest boundary): the python port was validated by
construction against the C++ source plus invariant tests and a mock-server
plumbing test; the C++↔python equivalence gets its definitive proof only
from the live `/tokenize` step below. No offline ground-truth full prompt
text survives from earlier sessions (only 73-char prefixes were retained),
so no offline cross-check against a recorded server count was possible.
If the live `/tokenize` step disagrees, STOP — re-derive offline; never
live-patch the count.

## 4. Runtime verification step (tonight, after load, before ANY measured request)

```bash
bash ~/B70-DOCS/scripts/tmp/flashnext-calib-20260925/verify_prompt_n.sh \
     ~/B70-DOCS/scripts/tmp/flashnext-calib-20260925/p512.txt 8010 <server.log>
```

The script (tested this session against a mock tokenizer server; both the
PASS path and each FAIL path exit correctly): (1) sha256-checks the file
against its metadata; (2) `POST /tokenize` with `add_special=true,
parse_special=true` and requires the returned token id sequence to EQUAL the
offline `input_ids` (stronger than count-only); (3) sends exactly ONE
unmeasured `/completion` request (`n_predict=1`, temp 0, `cache_prompt:false`)
and requires `timings.prompt_n` == predicted and cached == 0; (4) greps the
final `prompt eval time` line of the server log for the same count. Run it
once for `p512.txt` and once for `p8192.txt` (each run sends one unmeasured
request; the `/tokenize` calls are CPU-side only). Exit 0 = calibrated;
anything else = do not measure.

## 5. Thermal-block history and substitution discipline

Twelve prior attempts (RL-057): p512 partials at 59–63 C are invalid
(`...T152049Z`, `...T155012Z`, `...T164522Z` THERMAL-INVALID); the p8192
exploratory cell (`...T232536Z`, 344.96 tok/s median) ran at 56–63 C and is
operator-authorized thermal-deviation diagnostic only; exact-token
calibration failed live twice (last: `...T222237Z`, warmup discarded, no
measured request); the two-arm placement run (`...ab-thermaldev-20260924T000236Z`)
is exploratory-only. Repeated warmups push package sensors above 55 C and
unload does not cool below the gate.

Discipline for tonight:

- Fresh server per cell; both cards ≤ 55 C before launch; cooldown between
  cells to the declared verified threshold; never modify caps or close
  desktop services to force cooldown.
- ONE same-shape warmup per cell, discarded, then the verify step above,
  then measured samples. The offline files replace the live calibration
  loop that failed — do not reintroduce live filler-word calibration.
- Every sample keeps the ≥64-char unique leading entropy (substitute the
  64-digit block per request — count-invariant by construction) and the
  prefix-cache counter snapshots.
- p512/g128 decode cell: `-c 8192` suffices; p8192/g1 cold-input cell: use
  `-c 9216` (8192 prompt + continuation must fit; prior sessions did this).
- Unload → verify empty → cooldown before the second cell.
- If the verify step fails: STOP the session, retain evidence, re-derive
  offline. No live-fixing, no cap changes, no exceptions — that is how the
  last twelve attempts died.
- Cleanup as always: stop the exact server, restore standing caps, re-run
  empty gates, retain evidence.

## 6. Session record (this prep)

Changed/created (all inside the repo, nothing touched on the model tree or
build tree): `scripts/tmp/flashnext-calib-20260925/` (new:
`flashnext_bpe.py`, `build_prompts.py`, `tokenizer_cache.json`,
`p512.txt/.meta.json/.ids.json/.verify.json`,
`p8192.txt/.meta.json/.ids.json/.verify.json`,
`calibration-summary.json`, `verify_prompt_n.sh`) and this worklog.
Checks run: binary + GGUF sha256 (both match RL-055); tokenizer metadata
extraction; single-digit/64-digit invariance, round-trip, and special-token
assertions; prompt files re-verified from disk bytes; verify script PASS
and FAIL paths exercised against a local mock (mock stopped and its
artifacts removed). Unverified until tonight: C++↔python tokenizer
equivalence on the live server (that is precisely what section 4 gates).
