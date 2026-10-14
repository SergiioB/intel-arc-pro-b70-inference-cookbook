---
title: "MiMo-V2.6-Flash-RL acquisition + conversion-route verdict (dual-B70 host)"
type: plan
date: 2026-09-23
status: acquisition-preserved-blocked-on-storage-route-analyzed
---

# MiMo-V2.6-Flash-RL — acquisition, inventory, and conversion-route verdict

**PROVISIONAL — no GPU work was done (GPUs owned by another campaign), no benchmark numbers,
no public claims. CPU/network acquisition + header-only inventory + converter/runtime analysis.**
Target: `XiaomiMiMo/MiMo-V2.6-Flash-RL` (NOT 2.5 — operator-confirmed). HF revision pinned at
acquisition time: `5711b268169967567844e1e560e8a3966da959b1`.

> **Second-pass verification (2026-09-23, this session):** every load-bearing claim below was
> independently re-verified after a directive reversal froze the acquisition. Full re-check
> results are inlined as "[re-verified]" and summarized in §e.2. Final operator state:
> **partials preserved with zero deletion; no further downloads/moves; full download and any
> conversion remain blocked on storage.**

## a. Download manifest (PAUSED at 90.01 / 172.96 GB)

Destination: `/mnt/<sata-disk>/models/MiMo-V2.6-Flash-RL/` (never `/`, never `/tmp`).
Method: `hf download` (huggingface_hub 1.27.0, Xet backend). No curl/urllib, no credentials written.

| Item | Count | Bytes |
|---|---|---|
| Repo total (90 files) | 90 | 177.77 GB |
| Selected set (main shards + MTP + tokenizer/config) | 79 | 172.958 GB |
| **Complete + size-verified at pause** | **46** | **90.010 GB** (incl. all tokenizer/config/code + `model_mtp.safetensors` + 31/64 main shards) |
| Missing at pause | 33 | 82.948 GB (all `model_pp0_ep*_shard0.safetensors`) |
| Resumable partial blobs in `.cache/huggingface/download/` | — | 2.682 GB |
| Skipped by headroom policy (recorded, not deleted) | 11 | 4.809 GB |

- **Skipped by policy** (repo 177.77 GB exceeded the mount's safe headroom; task's fallback branch):
  `dflash/` (2.936 GB draft + code/index — DFlash drafter), `audio_tokenizer/` (1.873 GB), `assets/`.
  These are auxiliary; text-serving needs none of them.
- **Pause**: coordinator-ordered 2026-09-23 when remaining (82.95 GB) approached mount free space
  (~87 GB). Download killed mid-flight; partials preserved; **no files deleted**.
- Exact per-file state: `DOWNLOAD-PAUSED-MANIFEST.json` in the model dir (complete/missing lists,
  bytes, resume command, pinned revision). Hashes: `SHA256SUMS` (46 entries, sha256 of every
  completed file). Key anchors: `config.json`
  `61bea4a0f7a0dd8969f8cae528761e26b697dd12ff63e98804c3f0945492e621`, `model.safetensors.index.json`
  `09d9b96a77ed9765fa4e02a6a45f92797702eef432da426efc6b45c1131b1812`, `model_mtp.safetensors`
  `f8f78a78a79e4ec9e42e02ff94790d2a0183995ae278ff521416baa25ae2f38c`.
- **Resume** (needs a volume decision first — see §d space math; currently FROZEN by operator
  directive, do not run without authorization):
  `hf download XiaomiMiMo/MiMo-V2.6-Flash-RL --revision 5711b268169967567844e1e560e8a3966da959b1
  --local-dir /mnt/<sata-disk>/models/MiMo-V2.6-Flash-RL --exclude "dflash/*" --exclude
  "audio_tokenizer/*" --exclude "assets/*" --max-workers 4` (resumes from partials; same command as
  recorded in the manifest JSON; revision pin added at verification pass).

### §a.2 Final operator directives and second-pass verification (2026-09-23)

Directive log after the pause: the operator conditionally authorized a three-mount split
continuation (nvme ep3–9 + ep38–47 ≈ 42.8 G, sata-wd ep48–63 ≈ 40.2 G, symlinks into this
canonical dir; per-shard 2x headroom trivially satisfied), then **revoked it and froze the
acquisition**: preserve the partial set unchanged, no further downloads, no moves, no
deletions (only DeepSeek was ever authorized for deletion; DeepSeek is retired and was not
touched). The split layout is recorded here for a future relaxed decision only — it was
never executed. Blocked-on-storage verdict re-affirmed under the strict single-mount rule:
remaining shard set 82.95 GB needs ~166 GB free on ONE mount at 2x; best mount has 105 GB
(nvme-wd-sn770). Current free at freeze: tforce 87 GB, nvme 105 GB, sata-wd 84 GB,
external 72 GB.

Second-pass verification (all PASS):

- `sha256sum -c SHA256SUMS` re-run independently: **45/45 OK** (exit 0).
- HF LFS cross-check at the pinned revision: local sha256 == remote LFS oid for
  `tokenizer.json` (`ff15eb92…`) and `model_mtp.safetensors` (`f8f78a78…`); remaining small
  files are git blobs (no LFS oid) — integrity rests on size match + local hashes above.
- Byte sizes of all 46 completed files == HF API `files_metadata` (zero mismatches).
- `ep25` vs `ep0` expert headers: zero shape/dtype mismatches across all 6 tensor roles —
  all 64 EP ranks structurally identical, only expert-id ranges differ (ep25 carries 100–103).
- vLLM and converter claims in §c re-verified inside the champion image / both trees
  (see the "[re-verified]" tags there).

Checkpoint layout (from `model.safetensors.index.json`, 73,081 tensors): 64 EP shards
(`model_pp0_ep{0..63}_shard0`), 4 distinct experts per rank (each ep>0 shard holds exactly
1128 = 4 experts × 47 layers × 6 tensors; ep1 verified by header to carry expert IDs 4–7),
all dense + vision/audio tensors in `ep0`, MTP in `model_mtp.safetensors`. **All 64 shards are
mandatory** — partial expert sets cannot be reassembled. Index metadata: `{'save_format': 'mxfp4',
'total_size': 172,923,364,096, 'tp_size': 4}` (byte-total reconciles with observed file bytes 172.93 GB).

## b. Header inventory (safe_open metadata only — no weights loaded)

**MoE shape**: `MiMoV2ForCausalLM`, `model_type mimo_v2`; 48 layers (47 MoE; layer 0 dense),
hidden 4096, **256 routed experts**, `moe_intermediate_size` 2048, 8 experts/token, sigmoid +
noaux_tc routing with F32 `e_score_correction_bias`, no shared experts, untied embeddings
(`lm_head` + `embed_tokens`, both BF16 [152576, 4096]), vocab 152,576. Attention: 64 heads,
head_dim 192, v_head_dim 128; hybrid 9 full-attention layers {0,5,11,17,23,29,35,41,47} + 39 SWA
layers (window 128, SWA kv 8, sink bias BF16 [64] on SWA layers only), partial rope 0.334,
1M max positions. Dense tensors: qkv FP8 e4m3 block [128,128] with F32 `weight_scale_inv`
(full layers [13568,4096]/[108,32]; SWA layers [14848,4096]/[116,32]); **all `o_proj` BF16**
(quantization_config `ignored_layers`); routers BF16; layer-0 dense MLP FP8 (intermediate 16384).

**Total parameters: 309.77 B** (experts 302.795 B + attention 4.483 B + embeddings/head 1.250 B +
MTP 0.988 B + dense-MLP 0.201 B + routers/norms 0.050 B); **~15.4 B active** per token (8/256 experts).
Independent cross-check: expert bytes 302.795 B × 0.53125 (MXFP4 + e8m0/32) = 160.86 GB ≈ 160.87 GB
observed expert-shard bytes. Routed experts stored as **packed U8 MXFP4** (verified ep1 headers:
gate/up `U8 (2048, 2048)`, down `U8 (4096, 1024)` + `U8` e8m0 `weight_scale`), i.e. the model is
**natively MXFP4 QAT** (independently reported for this family).

**MTP module** (`model_mtp.safetensors`, 48 tensors, **987,868,224 params = 987.9 M**, 3 layers ×
329.3 M; per layer: qkv FP8 [14848,4096] + scale [116,32]; o_proj BF16 [4096,8192]; dense MLP gate/
up/down FP8 [16384,4096]/[4096,16384] + `weight_scale_inv`; `enorm`/`hnorm`/`eh_proj` [4096,8192]/
`final_layernorm`/`input_layernorm`/`pre_mlp_layernorm`; `attention_sink_bias` [64]).
**The MTP module owns NO LM head** — no `head`/`shared_head` tensor exists in the MTP file; the draft
reuses the verifier's `lm_head.weight` (ep0).

### MTP-head comparison vs Qwen3.8 (D11 reference)

| | Qwen3.8-27B (D11) | MiMo-V2.6-Flash-RL (this repo) |
|---|---|---|
| MTP tensors / params | 15 `mtp.*` / 424.70 M (BF16) | 48 `model.mtp.layers.{0,1,2}.*` / 987.87 M (FP8 + BF16) |
| Draft depth | 1 layer | 3 layers |
| Draft LM head | none — **shared** verifier head (`mtp.norm` → shared LM head) | none — **shared** verifier head (`lm_head.weight` in ep0) |
| Structure | enorm/hnorm/fc + attn + dense MLP | identical DeepSeek-style pattern × 3 (enorm/hnorm/eh_proj + attn + dense MLP) |

**Draft-side INT4 lever transfer: YES — structurally, with the same lossless boundary.** The
champion lever's losslessness rests on quantizing only the draft-side subtree while `lm_head.weight`
and all verifier weights stay untouched (D11 §f: safetensors diff shows only `mtp.*` changed). MiMo
has the same property — its draft head IS the verifier head, so draft-side quantization must exclude
it exactly like D11 does. The quantizable draft payload is 987.9 M params (3 layers), so the draft
cost reduction is proportionally larger than Qwen3.8's single 424.7 M-layer case. Note the draft
ships FP8 (not BF16), and ggml-org already publishes `mtp-MiMo-V2.6-Flash-RL-{MXFP4,Q8_0,BF16}.gguf`
(2.38/2.38/4.48 GB) — the MXFP4/Q8 draft GGUFs are the drop-in expression of this lever; a GPTQ-INT4
draft re-quant remains an optional delta on top.

## c. Conversion-route verdict

### llama.cpp — local trees BLOCKED at one exact point; upstream already fixed it

Verified working in BOTH local trees (`/mnt/<sata-disk>/qwenflash-llama.cpp` and
`~/llama.cpp`; the two `conversion/mimo.py` are identical except example decorators
**[re-verified: `diff` shows only the two `@ModelBase.example` lines]**):

- Registration + arch: `conversion/mimo.py:16` registers `MiMoV2FlashForCausalLM`/`MiMoV2ForCausalLM`
  → `MODEL_ARCH.MIMO2`; `gguf-py/gguf/constants.py:617/1366` define MIMO2/"mimo2"; C side knows it
  (`src/llama-arch.cpp:146`) with a full implementation `src/models/mimo2.cpp` (SWA standard + window
  pattern, NEXTN `eh_proj/enorm/hnorm` + optional shared-head with graceful trunk-only probe, attn sinks).
- File discovery handles the unusual pp/ep sharding: index-driven `weight_map` → file list
  (`conversion/base.py:233-247`), with a hard "Missing or incomplete model files" guard (`:282-291`) —
  confirming all 64 shards are required.
- TP=4 fused-qkv regroup for the FP8 `qkv_proj` is implemented (`mimo.py:31-95`) and the checkpoint
  matches its detector exactly (full 108 = 4×ceil(3392/128) scale rows; SWA/MTP 116 = 4×ceil(3712/128)).
- MTP remap `model.mtp.layers.{i}.*` → extra blocks (`mimo.py:181-192`) matches the checkpoint's
  tensor names and `_n_nextn = 3` matches `num_nextn_predict_layers: 3`.

**The gap (file/line evidence; [re-verified] this session): routed-expert MXFP4 is NOT handled.**
The checkpoint is a mixed store: `quantization_config.quant_method = "fp8"` **with `store_dtype = "mxfp4"`**
(experts packed U8 + U8 e8m0 `weight_scale`; dense FP8; o_proj/router BF16). The local trees' `base.py`
(`:870` qwenflash / `:833` pinned) sets `_is_mxfp4` only when `quant_method == "mxfp4"`, and the `fp8`
branch (`base.py:437-462`) dequants only `*_scale_inv` — packed expert weights + `weight_scale` pass
through untouched. `repack_mxfp4_blocks` (`base.py:662`) is called from `deepseek.py:732` and
`kimi_k3.py:123` only — **not from `mimo.py`** (re-verified by grep in both trees; neither local
`mimo.py` mentions mxfp4/store_dtype at all). Failure mode is deterministic: `mimo.py`'s expert staging
(`modify_tensors`, `mimo.py:195-221`) stages `weight_scale` tensors alongside weights, so the
`n_experts*3` merge trigger misfires → `KeyError` on a missing `...weight` entry, or the leftover
scales trip the "Unprocessed experts" check (`mimo.py:223-230`); even with scales stripped, packed U8
would be silently cast to f32 in the `prepare_tensors` loop — poison output. **No conversion was run
(local gap + incomplete download + space, below); this is a static-analysis gap record.**

**Upstream already solves this**: master `conversion/mimo.py` has `_is_mxfp4_packed()`
(`store_dtype == "mxfp4"`, enforces QK_MXFP4 = 32) and `_write_mxfp4_experts()` writing
`raw_dtype = MXFP4` + `MOSTLY_MXFP4_MOE` — proven by **ggml-org's official conversion**
`ggml-org/MiMo-V2.6-Flash-RL-GGUF` (revision `9bf2e45b30eb518326536a7f9703bc1bc9b49d06`):
`MXFP4` GGUF 167.36 GB (native expert precision, QAT-lossless), `Q2_K` 125.71 GB,
`mmproj` BF16 2.75 / Q8_0 1.56 GB, **`mtp` draft GGUFs MXFP4/Q8_0 2.38 GB / BF16 4.48 GB**.
Shortest llama.cpp path is therefore: sync `conversion/` + `gguf-py/` from upstream (bounded update,
not a from-scratch port) — or skip local conversion entirely and fetch the official GGUF.

**Space blocker (either path), recorded with math (refreshed at the 2026-09-23 verification pass)**:
a local MXFP4-passthrough GGUF (~167 GB) cannot coexist with the 172 GB safetensors sources on this
458 GB mount (185 GB free at task start, 87 GB now — the partial itself consumed the difference):
≥355 GB free is required with sources preserved (repo rule: preserve source artifacts). Current free:
tforce 87 GB, nvme-wd 105 GB, sata-wd 84 GB, external 72 GB — combined 348 GB is below the ~348–355 GB
input+output working set even before the 2x rule, and the 2x-headroom rule alone (needs ~356 GB on ONE
mount; best 105 GB) already blocks completing the safetensors set. A Q2_K derivative (125.7 GB) has
the same concurrency problem and fails the 2x rule too (needs ~251 GB on one mount). Downloading the
official MXFP4 GGUF directly (167.36 GB) instead of completing the safetensors set fails the same rule
(needs ~335 GB on one mount). Decision needed from operator: reclaim/external volume ≥360 GB, or park
the model until such a volume exists; the two acquisition forms (safetensors vs official GGUF) are
redundant for serving, so at most one should ever be stored.

### Cascadia 0.2.3 CLI — BLOCKED at engine registration (no MiMo support as installed)

`~/cascadia/cascadia-0.2.3-linux-x86_64/cascadia` (built 2026-09-09), CPU-only
inspection:

- `cascadia engines` registers: mock, ov-genai, ov-runtime, ov-dist-spec, gemma4,
  sparse-moe (Kimi K2.6 / MiniMax-M2), qwen35 (Qwen3.5/3.6 MoE, 3.8 dense). **No MiMo
  engine or exporter registration** — a `strings` scan of the binary finds zero `MiMoV2`
  references.
- The `conversion/mimo.py` assessed above lives in the llama.cpp trees and exports **GGUF**,
  not the OpenVINO IR shards `cascadia shard` consumes — it is not the Cascadia exporter.
- The "native PP/EP shards" framing also resolves here: the checkpoint is a **single
  pipeline stage** (`model_pp0_*` = pp stage 0 of 1) sharded 64-way over experts. There is
  no multi-stage PP layout in the repo to exploit; pipeline parallelism for this model on
  this host would have to be manufactured by `cascadia shard` stage-splitting (unsupported
  for mimo in 0.2.3) or by llama.cpp `-ts` row split across the two B70s (ADR-0008
  independent-stacks territory).

**Verdict: Cascadia route requires upstream cascadia work (a mimo engine/exporter
registration on the ov-runtime/ov-dist-spec path) before it can even be tested; blocked.**
The nearest usable PP-flavored fallback remains llama.cpp dual-card independent stacks.

### vLLM XPU — arch + kernel READY in the champion image, capacity-BLOCKED on this host

Verified inside the champion image `b70-vllm-kernels-0.1.15.4` (= `vllm/vllm-openai-xpu` line,
tag/digest `f01e24f6`/`sha256:3ae356e0…`), vLLM `0.27.2rc1.dev77+gac7509e2b`, CPU-only throwaway
container:

- `model_executor/models/registry.py:180-181`: `MiMoV2FlashForCausalLM`/`MiMoV2ForCausalLM` →
  `mimo_v2.py`; `:621` `MiMoV2MTPModel` → `mimo_v2_mtp.py` (draft model class present).
- The mixed store dispatches: `layers/quantization/fp8.py:206` `if self.store_dtype == "mxfp4":`
  → `Mxfp4MoEMethod`; MXFP4 kernels include `kernels/linear/mxfp4/xpu.py` — `XPUMxFp4LinearKernel`,
  "MXFP4 W4A4 GEMM on XPU" (`torch.ops._xpu_C.fp4_gemm`), gated on `current_platform.is_xpu()`.

So the champion image can in principle load this checkpoint as-shipped **[re-verified:
`fp8.py:206` `if self.store_dtype == "mxfp4":` and `XPUMxFp4LinearKernel` gated on
`current_platform.is_xpu()` both present in the image]**. **The blocker is capacity:
~172 GB of weights vs 65.3 GB combined B70 VRAM, and the vLLM XPU backend has no MoE CPU-offload
path** (no `--cpu-offload-gb` equivalent for XPU; expert-offload unproven). vLLM on this host would
need a sub-65 GB derivative (a ~3.3-bpw requant of a QAT MXFP4 model — a major quality-gated project,
this fleet's K216-IQ3 situation repeated), or a future XPU offload feature.

### Shortest path to serving MiMo-V2.6-Flash on this host

1. **llama.cpp SYCL** (recommended): obtain a GGUF — either official `ggml-org/MiMo-V2.6-Flash-RL-GGUF`
   (MXFP4 167.4 GB or Q2_K 125.7 GB) or local conversion after syncing `conversion/` from upstream —
   onto a volume that actually fits it, then serve with the proven DSV4-K216 playbook (partial CPU
   expert residency `-ncmoe`, 8 threads, `-fa on`, q8_0 K / q4_1 V KV, 150 W cap). Serving-time
   verification gates: SYCL MMVQ coverage for MXFP4 experts (may fall back/dequant — measure, do not
   assume; the DSV4 lesson: check the fused `MUL_MAT_ID` path before headline numbers; note the
   qwenflash tree's `ggml-sycl` already carries MXFP4 in `convert.cpp`/`vecdotq.hpp`/`mmvq.cpp` while
   the pinned tree has it only in data-movement paths — target the qwenflash/post-upstream SYCL
   state), and mimo2 MTP-draft speculation support in the pinned build.
2. **vLLM XPU**: ready in software, blocked by VRAM capacity on 2×32 GB — revisit only with an
   offload feature or a validated low-bpw derivative.

## d. Ready-to-run benchmark plan (gated — execute only after route + space are resolved)

PP nomenclature note (added at verification pass): the repo's `model_pp0_*` prefix means
pipeline stage 0 **of one** — the checkpoint carries no multi-stage PP layout, so a "PP
benchmark" here means either (a) llama.cpp dual-card independent stacks via `-ts`
(ADR-0008 default; the `-ts` screens below), or (b) true staged PP via Cascadia, which is
blocked until cascadia gains a mimo engine (§c). Plan below assumes (a).

Preconditions: (1) safetensors set completed or official GGUF obtained (volume decision per §c);
(2) empty-GPU gate per AGENTS.md §4 (no inference processes/containers, ≥31,000 MiB visible_avail per
card in use, one declared engine); (3) hwmon discovered dynamically, 150 W restored after; (4) engine
version + artifact sha256 recorded (GGUF hash + conversion log if local).

Cells (fleet Lane-1 standard, single card first):

- p2048/g1 cold input; p8192/g1 cold input (HTTP cache guards §8: ≥64-char unique entropy prefix,
  prefix-cache zero-hit-delta snapshot, endpoint token verification).
- p512/g128 decode; p8192/g128 decode.
- One representative real workload (natural text, declared sampling: non-thinking temp 0.7 / top_p 0.8 /
  top_k 20 / presence 1.5 class per model-card policy; greedy cells labeled diagnostic only).
- Screens n=3 for config variants (`-ncmoe` sweep 0/…/K to map CPU-expert residency; `-ts` split if
  dual-card independent stacks per ADR-0008), then fresh n=5 finalist confirmation.
- C1 single-client gate enforced (one client at a time; server counter snapshots bracketing).
- Cooldown to ≤52–55 °C between cells; progress milestones; declared timeout; cleanup + 150 W + re-run
  gates (AGENTS.md §9, §13).
- MTP/spec-decode add-on (only after base route passes): mtp GGUF draft vs no-draft ABBA, per-step
  acceptance from counter diffs, prompt-content class labeled; draft-side quant delta (MXFP4/Q8/INT4
  draft) compared under the same protocol.
- All numbers land results → campaign → cookbook per the 2026-09-15 chain; PROVISIONAL until the
  claims audit (`b70-benchmark-claims-standard`).

## e. What was and was not done

Done: partial acquisition (46 files verified + hashed), full header inventory of downloaded artifacts,
sharding/MoE/MTP analysis, converter + runtime route verification (local trees, upstream, ggml-org
artifacts, champion vLLM image internals), gap records with file/line evidence, this worklog +
machine-readable manifest/hash files. NOT done: remaining 33 shards (paused — coordinator decision
pending on volume), dflash/audio_tokenizer download (skipped by headroom policy), any GGUF conversion
(blocked: local converter gap + download incomplete + space), any GPU execution (out of scope by
mandate). Sibling models (MiMo-V2.5*, MiMo-V2-Flash, MiMo-V2.6-Pro-RL) NOT downloaded.

### e.2 Second-pass verification record (2026-09-23, after the acquisition freeze)

This session re-derived the report from scratch after the operator froze the acquisition;
nothing on disk was modified (zero deletions, zero moves, zero new downloads; the 2.68 GB
`.cache` partials and the four `.incomplete` temps preserved). Independently re-verified:

1. Hashes: `sha256sum -c SHA256SUMS` → 45/45 OK (exit 0); HF LFS oid match for
   `tokenizer.json` + `model_mtp.safetensors`; byte-size equality with HF API for all 46
   completed files (zero mismatches).
2. Inventory: re-read ep0/ep25/model_mtp headers via `safe_open`/`get_slice` (no tensor
   loads). Reproduced 308.78 B main-model logical params (experts 302.79 B + text core
   5.98 B) + 987.87 M MTP; active ≈ 15.45 B/token; stored bytes reconcile to the index
   `total_size` 172,923,364,096 exactly (experts 160.86 GB + non-expert 10.87 GB + MTP
   1.19 GB). ep25-vs-ep0: zero structural mismatches across EP ranks.
3. Converters/runtime: both local `conversion/mimo.py` identical modulo example decorators
   (`diff`); MXFP4 gap confirmed in both (`_is_mxfp4` keyed on `quant_method` only —
   `base.py:870` qwenflash / `:833` pinned; `repack_mxfp4_blocks` called only from
   `deepseek.py:732` / `kimi_k3.py:123`); champion image contains `fp8.py:206`
   `store_dtype == "mxfp4"` dispatch, `Mxfp4MoEMethod`, and
   `kernels/linear/mxfp4/xpu.py::XPUMxFp4LinearKernel` (XPU-gated); MiMoV2 archs incl.
   `MiMoV2MTPModel` present in the image's ModelRegistry (vllm 0.27.2rc1.dev77).
4. Cascadia 0.2.3: `cascadia engines` has no MiMo entry; zero `MiMoV2` strings in the
   binary; `conversion/mimo.py` is the llama.cpp GGUF exporter, not a Cascadia engine →
   Cascadia route blocked at registration (new §c subsection).
5. GPU hygiene: the only container run was CPU-only (no `--gpus`/`--device` flags,
   registry/source imports only); no inference process, server, or benchmark was started
   at any point in either pass; Qwen3.8-27B PP smoke untouched.
