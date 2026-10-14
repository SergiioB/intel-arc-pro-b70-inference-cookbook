---
type: research
date: 2026-09-29
status: provisional
author: b70-upstream-researcher
topic: upstream dig — llm-scaler sym_int4 fix vs the 30-defect; vLLM #53912; exl3xpu/bpw variants; llama.cpp #28796/#26689 follow-ups
PROVISIONAL: NOT FOR PUBLIC HEADLINE
---

# Upstream dig: sym_int4 "output fix" vs the Qwen3.8-27B 30-defect (2026-09-29)

Scope: zero-GPU, primary-source only. Two fleet defects, both possibly already
fixed upstream:

- **Defect 1 (30-defect):** every symmetric-INT4/W4A16 export of Qwen3.8-27B
  (GPTQModel, AutoRound→compressed-tensors, AutoRound→auto_gptq) answers 30
  instead of 14 on `sum(i*i for i in range(4))` (greedy, thinking off). BF16
  base, EXL3, OV-int4, Q4_K_M answer 14 (RL-084, RL-087, RL-088). CPU
  discriminator (probe 2A) in flight to split kernel-vs-weights.
- **Defect 2 (W4A16+MTP4 prefix-cache corruption):** cache ON + MTP4 → 5/6
  outputs degenerate, including zero-cache-hit first requests (2026-09-28,
  `results/2026/09/b70-novelty-20260928/RESULTS.md`; vLLM issue #53912).

Hot lead: [intel/llm-scaler `vllm-0.26.0-b2`](https://github.com/intel/llm-scaler/releases/tag/vllm-0.26.0-b2)
(2026-09-09) release notes advertise a "sym_int4 output fix", an MTP
prefix-cache fix, and a block-FP8 concurrent-decode correctness fix.

## Findings

(filled in as dug — see sections below)

### 1. llm-scaler vllm-0.26.0-b2 — sym_int4 output fix

Release identity (verified live 2026-09-29):
- Tag `vllm-0.26.0-b2` = commit `77c547307e0f0289044c8a610bd9603c19c2c483` (2026-09-08).
  <https://github.com/intel/llm-scaler/releases/tag/vllm-0.26.0-b2>
- Image: `intel/llm-scaler-vllm:0.26.0-b2`
  digest `sha256:52218ad85513ab6686d4c090c83c2bd8c5b02423c63aa4dabd41837fe641fe3b`.
- **No newer llm-scaler-vllm release exists after 0.26.0-b2** (checked 2026-09-29;
  release list tops out at b2, 2026-09-09). Repo `pushed_at` 2026-09-28 — main has
  moved but no new release tag.

All four vLLM release items itemized to commits (b1 tag = `25b645ae`, 2026-09-02):

| Release item | Commit / PR | Files | What it actually is |
|---|---|---|---|
| Fix incorrect output for `sym_int4` models | `ede4320a24a67f664fb53081d2623f9efe9a75b7` = PR [intel/llm-scaler#675](https://github.com/intel/llm-scaler/pull/675) "esimd: fix INT4 router residual race for all shapes", merged 2026-09-04, fixes [issue #671](https://github.com/intel/llm-scaler/issues/671) | `vllm/custom-esimd-kernels-vllm/csrc/xpu/esimd_kernels/resadd_norm_gemv_int4.h`, `int4_GEMV.h` | see below |
| Fix prefix caching with MTP | PR [intel/llm-scaler#661](https://github.com/intel/llm-scaler/pull/661) "Upgrade vLLM to v0.26.0", commit `16bfc56af65d914725b310a6a03eb051679aea9a`, 2026-09-02 | `vllm/patches/vllm_for_multi_arc.patch`, `vllm/patches/vllm_xpu_kernels.patch` | The b1→b2 jump replaces the whole vLLM base (0.21.x-era → v0.26.0). The old llm-scaler-local `requires_eagle_cache_drop` patch logic ("prefix cache hits must drop one block for hidden states") is **removed** in #661 because upstream vLLM by 0.26.0 carries the `drop_eagle_block` machinery natively (`drop_eagle_block` appears 13x in `vllm/v1/core/single_type_kv_cache_manager.py` at upstream `568afb3a1`, 2026-07-26, verified). Plus GDN XPU spec-decode enablement (removes the "xpu does not support speculative decoding yet" assert; wires `num_spec_decodes`/spec state tensors into `torch.ops._xpu_C.gdn_attention`). |
| Improve MTP decoding performance | PR [intel/llm-scaler#683](https://github.com/intel/llm-scaler/pull/683) = b2 tag commit `77c547307` | `vllm/patches/vllm_xpu_kernels.patch` | FA2 speculative-decode fast path: uniform small causal q_len batches (q_len = num_spec_tokens+1, threshold 16, env `VLLM_XPU_SPEC_DECODE_MAX_QLEN`) rerouted from chunk-prefill kernel to the split-K **decode** kernel (`_spec_decode_varlen_fwd` in `flash_attn_interface.py`), per-position `seqused_k` adjustment; regression test asserts the fast path is taken and matches causal reference. |
| Fix block-FP8 output corruption during concurrent decoding | PR [intel/llm-scaler#677](https://github.com/intel/llm-scaler/pull/677), commit `2ac6a6f68f58a7789e6f3652fe5d083047457339`, 2026-09-07, fixes [issue #676](https://github.com/intel/llm-scaler/issues/676) | `vllm/custom-esimd-kernels-vllm/csrc/xpu/esimd_kernels/gdn_conv_fused.h`, `gdn_conv_fused_seq.h` | GDN conv-state race: with `--mamba-ssm-cache-dtype float16` the fused GDN kernel let the `hv==0` work-group update shared `conv_state` inline while other work-groups still read it (no cross-WG barrier) — state corruption accumulated across decode steps → repeated/garbled output under concurrency. Fix: always shift conv state in a separate kernel after the main GDN kernel. Issue #676 is on **Qwen3.8-27B-FP8** itself. Default FP32 SSM cache never selects the broken path. |

#### The sym_int4 fix itself (commit `ede4320a`, PR #675, fixes #671)

Symptom in #671 (zhm-algo, 2026-09-03): Qwen3.6-35B-A3B, `--quantization sym_int4`
(llm-scaler's own runtime INT4 quant, `SymInt4LinearMethod` from their patch set —
NOT a GPTQ/AutoRound checkpoint), TP=2 on two B70, image 0.26.0-b1, sampled decoding
(temp 0.7 / top_k 40 / presence_penalty 2.0) → degenerates into repeated `!` runs /
NaN. Deterministic at kernel level: the PR's regression measured **2,036 / 4,096
router rows mismatched, max abs error 48.4** before the fix.

Failing code (old `resadd_norm_gemv_int4.h`, read in the PR diff): the fused
kernel launches one work-group per router output row `n`; every work-group reads
`residual`, but only row 0 writes the in-place residual update:

```cpp
simd<float, VL> added = h + r;
if (n == 0) {   // <-- only WG 0 updates residual in place
    block_store<fp16, VL>(residual_ptr + offset, simd<fp16, VL>(added));
}
```

No global barrier exists across work-groups, so later-scheduled groups can read the
*updated* residual and add `hidden` twice — perturbing every router logit computed
after the race. PR #399 had fixed this only for `N > 512`; #671 reproduced at a
256-expert router (N=256... exact shape per issue: Qwen3.6-35B-A3B TP2).

Repair (read in the same diff): delete the fused kernel; run residual-add+RMSNorm
once in a single-work-item kernel (`ResAddRMSNorm_int4_kernel`), then reuse the
existing ordered `GEMV_int4_host` dispatcher — "every GEMV work-group sees the same
normalized vector". Cost: +1 device submission, ~+1.3–1.8 us on a 1000-iter
microbenchmark, <1% of decode.

Header comment of the fixed file states the use case explicitly: *"post_attention_layernorm
+ MoE router GEMV (INT4 quantized)"* — this is the **MoE router** path of llm-scaler's
sym_int4 runtime quant, in llm-scaler's own `custom_esimd_kernels_vllm` wheel.

#### Verdict: does it cover our 30-defect class? **NO** (high confidence)

1. **Wrong model architecture for the path.** Qwen3.8-27B is `model_type qwen3_5`,
   **dense** — "no MoE involvement anywhere" (`models/qwen/qwen3.8-27b/serving.md:470`).
   The fixed kernel exists only for an MoE router GEMV; a dense model never executes it.
2. **Wrong kernel package.** `custom-esimd-kernels-vllm` ships only inside llm-scaler
   images. Our vehicles are the champion image (vLLM `0.27.2rc1.dev77+gac7509e2b`,
   XPU kernels 0.1.12.3) and the exl3xpu image (vLLM `0.26.1.dev0+g568afb3a1.d20260907`,
   kernels `0.1.13-5-g4043e76`); their W4A16 GEMM is `torch.ops._xpu_C.int4_gemm_w4a16`
   implemented in `vllm-xpu-kernels` via **oneDNN** (`csrc/xpu/onednn/onednn_matmul.cpp`,
   `tests/test_int4_gemm_onednn.py`) — a completely different code path from
   `resadd_norm_gemv_int4.h`.
3. **Wrong quant mechanism.** `sym_int4` is llm-scaler's runtime weight-only quant of a
   BF16/FP16 checkpoint (router included). Our defect reproduces on **pre-quantized
   checkpoints** (GPTQModel, AutoRound→compressed-tensors, AutoRound→auto_gptq;
   RL-084/087/088).
4. **Wrong symptom class.** #671 is hard corruption (repeated-token walls, NaN,
   sampled, nondeterministic-looking). Ours is a deterministic greedy arithmetic slip
   (14→30) with otherwise coherent output, stable across three exporters.

Residual caveat: probe 2A (CPU discriminator) is still the decider for kernel-vs-weights
on OUR side. If 2A says "CPU also answers 30", the whole XPU-kernel hypothesis class
(including this one) is dead and the defect is in the exported weights/architecture
interaction.

### 2. vLLM #53912 (prefix-cache + MTP corruption) — current state

<https://github.com/vllm-project/vllm/issues/53912> — "[Bug]: prefix caching + MTP
still corrupts output on hybrid Mamba/GDN models in v0.28.0 (#43559 closed but
unfixed)". **OPEN**, 20 comments, label `kv-cache-manager`. Original reporter:
ionut-anghelina (H100, vLLM 0.28.0, Qwen3.5-class hybrid) — "same failure as #43559"
([#43559](https://github.com/vllm-project/vllm/issues/43559), CLOSED 2026-05-25:
~20% tool-eval accuracy drop with prefix caching + MTP on Qwen3.6-35B-A3B — closed
but per this thread not actually fixed for all mechanisms).

The thread has decomposed the corruption into **multiple distinct upstream bugs**
(verified states as of 2026-09-29):

| # | Mechanism | Evidence / fix | State | In our exl3xpu image (base `568afb3a1`, 2026-07-26)? |
|---|---|---|---|---|
| A | **Mamba/GDN conv-state overlapping copy race** — speculative-decode conv-state left-shift within the *same physical block* uses a flat parallel memcpy; overlapping src/dst has no memmove semantics → shifted window reads partially-overwritten state. Cache-hit-independent (runs every MTP step that accepts tokens). | PR [#50729](https://github.com/vllm-project/vllm/pull/50729) "[Bugfix][Mamba] Fix overlapping state copy race", MERGED 2026-08-17, merge commit `a02cfccbc6187344325e364f09f6d8c33c4b253b`; OP re-tested and **confirmed it solves their repro**. Missed the v0.28.0 cut; ships from **v0.28.1rc0 / v0.29.0 / v0.30.0**. | Merged | **NO — image predates it.** Verified the *buggy* flat-copy code (`per_row_bytes = (conv_width - token_bias)...; src_offset = token_bias...`) is present in `vllm/v1/worker/mamba_utils.py` at `568afb3a1` (fetched and read). |
| B | **Async-scheduling accepted-token count race** — counts copied D2H in step N's row order; `_update_states` permutes the same pinned buffer in step N+1 before the event is waited; `prev_positions` gather permutes again → a request reads another's accepted count. | Issue [#51571](https://github.com/vllm-project/vllm/issues/51571) (OPEN, 2026-08-09); patch PR [#53919](https://github.com/vllm-project/vllm/pull/53919) "**OPEN, NOT MERGED**". Discriminator: `--no-async-scheduling` → 0/288 (johny-jose). | Open | Async-sched path; our probe ran serial C1 `max-num-seqs=1` (permutation needs >1 in-flight row) — unlikely to be our zero-hit mechanism, untested on XPU. |
| C | **`MambaManager.find_longest_cache_hit` ignores `drop_eagle_block`** — accepts the flag, never acts on it; cache hit restores recurrent state written over draft positions (incl. rejected MTP drafts) → corrupted state is hash-reachable and spreads to every later request sharing the prefix. | PRs [#48375](https://github.com/vllm-project/vllm/pull/48375) / [#57128](https://github.com/vllm-project/vllm/pull/57128) — **both OPEN, NOT MERGED**. cch-zuzuche implemented + validated the fine-grained half against 0.29.0 in-thread (2026-09-29, today). | Open | Mechanism requires cache **hits**; `drop_eagle_block` machinery exists at `568afb3a1` (13 refs). Not our zero-hit cells; could explain later same-prefix requests. |
| D | **Strided `state_indices_tensor_p[:, 0]` view read with unit stride** by a state-carrying kernel — with a (1+num_spec_blocks)-wide Mamba block table, every prefill batch row >0 writes its state into request 0's checkpoint blocks. Tell: needs ≥2 prompts prefilled in the same step, cache-independent. | Issue [#55357](https://github.com/vllm-project/vllm/issues/55357) (OPEN); fix PR [#55375](https://github.com/vllm-project/vllm/pull/55375) MERGED 2026-09-05 (`28e605fb`, Qwen4Exp/Flash-Next PLE conv strides). | Partially merged | Our probe: serial requests, `max-num-seqs=1` — no multi-row prefill steps. Not our mechanism as-run. |
| E | **`disable_eagle_block_drop=true` opt-out** (#53388) causes rejected-draft contamination of cached state under dflash. | PR [#53388](https://github.com/vllm-project/vllm/pull/53388) MERGED 2026-09-01 (`481839ad`); Suppressor72 reproduced + reverted the opt-out (Qwen3.8-27B-FP8, dflash K=7). | Merged (feature) | Requires the opt-out flag — we never passed it. N/A. |
| F | GLM-5.3-Flash kpool corruption with spec decode; RecoverSSM block-boundary column-selection bug (Kimi-K3 path). | PR [#58454](https://github.com/vllm-project/vllm/pull/58454) MERGED 2026-09-25; recoverssm issues tracked in #58863. | Merged / tracking | Not our model. N/A. |

Key structural fact: **our exl3xpu image's vLLM base (`568afb3a1`, 2026-07-26, one day
before the v0.26.0 tag) predates every one of the merged fixes (A: Aug 17, D-fix: Sep 5,
E: Sep 1, F: Sep 25)**, while the two still-open bugs (B, C) are unfixed in any release
up to and including v0.30.0 (2026-09-22). The `.d20260907` suffix on the image's version
is the fork's patch date, not the base.

Corroboration in-thread for our exact model class: karls0r (2026-08-31, vLLM 0.27.1,
**Qwen3.8-27B** INT8 W8A16 compressed-tensors, 2x RTX 3090 TP2, MTP head baked in,
prefix caching on, mamba-cache-mode align → corruption); rdlh (0.27.1, Qwen3.6-35B-A3B-FP8,
MTP1, prefix caching, async default-on → 12/4800 garbage, nightly with A+B fixes → 0/4800);
cch-zuzuche (0.29.0, RTX 5090, dflash K=7, `!`-wall with #55357 fingerprint). This is a
**general V1 hybrid-GDN bug family**, not XPU-specific — but the *specific mechanism that
best matches our zero-hit cold-request corruption is A (#50729)*, which is merged and
simply absent from our pinned base.

Our SergiioB comment (2026-09-28) is the only XPU datapoint in the thread.

### 3. 30-defect corroboration hunt (vLLM / vllm-xpu-kernels / Intel / AutoRound / GPTQModel)

**No upstream report matches our exact defect** (deterministic greedy 14→30 arithmetic
slip, GPTQ/W4A16 checkpoint, XPU, coherent output otherwise). Searched: vLLM issues
(`gptq wrong output xpu`, `w4a16 accuracy intel`, `int4 wrong answer arc`, `qwen3 gdn
quantization wrong answer`), vllm-xpu-kernels issues (full list), AutoRound, GPTQModel,
llm-scaler issues. Nearest neighbors, by relevance:

1. **AutoRound [#2142](https://github.com/intel/auto-round/issues/2142)** (OPEN,
   2026-08-11) + [HF discussion Intel/Qwen3.6-27B-int4-AutoRound #3](https://huggingface.co/Intel/Qwen3.6-27B-int4-AutoRound/discussions/3):
   accuracy complaint against **Intel's own published int4 AutoRound export of the
   same 27B dense GDN family** — CoT leaking into code comments + stuck self-loops,
   seen "only with Intel's quantized version". Intel maintainer (wenhuach) could NOT
   reproduce on quicksort/quantization-explain tasks with or without fp8 KV
   (vLLM nightly + FlashInfer; hardware not stated, CUDA-flavored stack). Unresolved;
   reporter says they moved to Qwen3.8. This is the closest public sibling of our
   defect and it is NOT XPU-specific — consistent with the "exported weights +
   GDN-architecture interaction" half of probe 2A's hypothesis space.
2. **vllm-xpu-kernels [#548](https://github.com/vllm-project/vllm-xpu-kernels/issues/548)**
   (CLOSED 2026-08-26): "Deterministic all-NaN GDN output for single-sequence prefill
   lengths T % 64 == 5 on Xe2/Battlemage" — proof that **deterministic,
   shape-dependent** GDN kernel correctness bugs exist on our exact silicon. Fix is
   after our champion's kernels 0.1.12.3 (v0.1.12 tracker closed 2026-07-28); the
   exl3xpu image kernels `0.1.13-5-g4043e76` also predate the 2026-08-26 close —
   whether the fix is in either image is checkable locally by tag arithmetic, but
   all-NaN ≠ subtle arithmetic slip.
3. **vllm-xpu-kernels [#454](https://github.com/vllm-project/vllm-xpu-kernels/issues/454)**
   (CLOSED 2026-07-09): "XPU silu_and_mul leaves output channels >= 257 unwritten at
   width 768" — a *silent partial-write* kernel bug class (not NaN, not crash). Fixed
   before 0.1.12.3. Cited as the precedent that subtle channel-dependent silent
   corruption has already happened once in this kernel package.
4. **llm-scaler #671** (the sym_int4 item itself) — hard corruption class, MoE router
   only; see §1.
5. GPTQModel trackers: nothing matching ([#2679] "Problems with Qwen3 Quantization"
   closed-generic; [#2700] Qwen3.5-MoE stopped). No Qwen3.5/3.8-GDN-specific numeric
   divergence report found in GPTQModel or AutoRound beyond #2142.

Also recorded while in the tracker (GDN+MTP XPU context for Defect 2):
vllm-xpu-kernels [#510](https://github.com/vllm-project/vllm-xpu-kernels/issues/510)
(GDN causal_conv1d rejects mixed spec/non-spec batches — MTP unusable under
concurrency; CLOSED 2026-09-02) and [#593](https://github.com/vllm-project/vllm-xpu-kernels/issues/593)
(OPEN: reduced active speculative width rejected when state-index cache retains
configured width) — both show the GDN+spec state machinery on XPU is young.

`utorch`: no `intel/utorch` repository exists (gh: could not resolve). The llm-scaler
vLLM image components that actually carry the fixes are the patched vLLM tree, patched
`vllm-xpu-kernels`, and `custom-esimd-kernels-vllm` (all inside the llm-scaler repo).

### 4. exl3xpu upstream activity + EXL3 bpw variant inventory

#### Repo activity after our pinned image (pulled 2026-09-26)

Repo: <https://github.com/0xSero/exl3xpu> ("EXL3 (exllamav3 trellis) inference on
Intel Arc Battlemage: bit-exact ESIMD kernels as a vLLM XPU plugin"). Last push
2026-09-26T12:01Z (HEAD `15ded2f3ad`). Commits since our image era are **recipes,
README numbers, and PROGRESS log entries**, plus exactly two functional items —
both **prefill-side**, nothing touching decode speed, trellis kernels, MTP, or
memory layout:

- `9db051daa2` / `2de46a8824` (2026-09-25): **oneDNN Graph fused SDPA for fp8-KV
  prefill** (`EXL3_ONEDNN_ATTN`): 128K prefill 1234 → 1415 tok/s, 254K 769 → 911.
- `e5adb9f109`/`3947549e90`/`061268ffd6` (2026-09-25): **int8 prefill on by
  default** (`EXL3_INT8_PREFILL=1`), oneDNN scratchpad via torch allocator, rows
  padded to 256.

**Image lineage (important):** `docker/Dockerfile` line 1 is
`FROM intel/llm-scaler-vllm:0.26.0-b2@sha256:52218ad8…` — the exl3xpu image is
built **on top of the llm-scaler 0.26.0-b2 image** analyzed in section 1. The
in-container vLLM `0.26.1.dev0+g568afb3a1.d20260907` is therefore llm-scaler's
patched vLLM: upstream main at `568afb3a1` (2026-07-26) + llm-scaler patch set,
built 2026-09-07. So the Defect-2 vehicle inherits b2's fixes (GDN concurrent
race #677, the FA2 spec fast path) **but still predates upstream #50729** — see
section 2 row A.

#### Turboderp EXL3 Qwen3.8-27B variant inventory (no downloads made)

Everything lives in ONE repo: `turboderp/Qwen3.8-27B-exl3` — `main` carries only
recipes/calibration artifacts; **each bpw variant is a branch**. All variants
`quant_method exl3`, **format `version 1.4.2`**, `codebook mul1` (cb=2). Sizes are
safetensors bytes; widths are the trellis tensor's last dimension (K = width/16
bits per weight), counted from `quantization_config.json:tensor_storage`:

| Branch (revision) | safetensors | bpw / head_bits / mtp_bits | trellis widths (count) | ESIMD K needed vs pinned image |
|---|---|---|---|---|
| `4.00bpw` (`113cf7ab`, our current) | 16.861 GB | 4.0 / 6 / 4 | 64 (400), 96 (1) | {4,6} — supported (what we run) |
| `3.50bpw` | 15.338 GB | 3.5 / 6 / 4 | 64 (262), 48 (137), 80, 96 | needs 3 and 5 — unsupported |
| `3.00bpw` (`6fe61ad6`) | **13.820 GB** | 3.0 / 6 / 4 | **48 (400)**, 96 (1) | **needs 3 — unsupported** |
| `2.50bpw` | 12.298 GB | 2.5 / 6 / 4 | 48 (262), 32 (137), 64, 96 | needs 2,3 — unsupported |
| `2.00bpw` (`5fce94d8`) | 10.779 GB | 2.0 / 6 / 4 | (not surveyed) | needs <4 — unsupported |
| `SC_4.00bpw_H5_V6` | 16.354 GB | 4.0 / 5 / ? | (not surveyed) | mixed |
| `SC_3.00bpw_H4` (`86b95530`) | **13.452 GB** | 3.0 / 4 / 3 | 48 (297), 64 (66), 32 (34), 80 (4) | needs 2,3,5 — unsupported |
| `SC_3.00bpw_H4_V4` (`004a8871`) | 12.986 GB | 3.0 / 4 / 3 | 48 (297), 64 (66), 32 (34), 80 (4) | needs 2,3,5 — unsupported |
| `SC_2.20bpw_H3_V3` | 10.285 GB | 2.2 / 3 / 2 | 32 (278), 48 (104), 64, 16 | unsupported |
| `SC_1.80bpw_H3_V3` | 9.068 GB | 1.8 / 3 / 2 | (not surveyed) | unsupported |
| `SC_1.60bpw_H3_V3` | 8.460 GB | 1.6 / 3 / 2 | (not surveyed) | unsupported |
| `SC_1.40bpw_H3_V3` | 7.852 GB | 1.4 / 3 / 2 | 16 (210), 32 (157), 48 (34) | unsupported |

Lower than 3bpw exists: `SC_2.20` (10.285 GB), `SC_1.80` (9.068 GB), `SC_1.60`
(8.460 GB), `SC_1.40` (7.852 GB); plain non-SC below 2.00 does not exist. Older
generation branches without `_V` (e.g. `SC_3.00bpw_H4`) and a separate
`turboderp/Qwen3.8-Flash-Next-exl3` + `turboderp/Huihui-Qwen3.8-27B-abliterated-exl3`
also exist. Sibling repo notes: the exl3xpu repo's own
`models/qwen3.8-27b-exl3-4.00bpw/model.yaml` pins our exact revision `113cf7ab`
(4bpw linears, 6bpw lm_head = the one width-96 module, 4bpw MTP head) and —
Defect-2-relevant — sets `enable_prefix_caching: true` with the comment "must be
explicit: vLLM 0.26 leaves it off for hybrid (GDN) models".

#### A3 tile-width kernel gate (coordinator follow-up) — verdict: width 48 UNSUPPORTED in the pinned image

Claim tested: "fork kernel dispatch covers trellis widths {64, 96} only; plain
3.00bpw needs 48; SC needs {32, 48, 80}". **Confirmed at source AND binary level.**

Source (upstream `main`, read 2026-09-29), `csrc/exl3_ops.sycl`:

```cpp
// :760
bool exl3_supported(int64_t K, int64_t cb) {
    if (cb == 2 && (K == 4 || K == 6)) return true;          // widths 64, 96
#ifdef EXL3_ALL_CODEBOOKS
    if (cb == 2 && (K == 2 || K == 3 || K == 5)) return true; // widths 32, 48, 80
    ...
#endif
    return false;
}
```

- Fast-path dispatch sites instantiate only `<4,2>` and `<6,2>`: `exl3_gemm_small`
  (`csrc/exl3_ops.sycl:264-265`) and `exl3_gemm_raw` (`:585-588`, ends
  `TORCH_CHECK(ok)`); templates `GemvKernel<K,CB,MR,NT>` / `DpasKernel<K,CB,…>`.
- Build: `scripts/build_ext.sh` compiles with `${EXL3_FLAGS:-}` only;
  `docker/Dockerfile` sets no `EXL3_FLAGS` → the shipped image is a **default
  build without `-DEXL3_ALL_CODEBOOKS`**.
- Binary proof (extracted `_C.so` from the pinned image
  `ghcr.io/0xsero/exl3xpu@sha256:21412bdd…` via `docker create` + `docker cp`
  — container never started): mangled template instantiations present are ONLY
  `ILi4ELi2E` (470) and `ILi6ELi2E` (478). **Zero `ILi3ELi2E` / `ILi2ELi2E` /
  `ILi5ELi2E`.** (`/tmp/exl3_C_pinned.so`, 2,936,272 bytes.)

Consequences and what a width-48 extension would touch:

1. A `3.00bpw` artifact would **load and run today** but decode would silently
   route through the Triton fallback (`exl3xpu/ops.py:78-84`: if
   `exl3_supported(K, cb)` is false → `tk.gemm_small`, generic `tl.constexpr K`)
   plus Hadamard wraps — functional, but not the bit-exact fused ESIMD path; the
   sub-4bpw speed case is gone without kernel work. (Same for SC variants, worse.)
2. Extension path: rebuild with `EXL3_FLAGS="-DEXL3_ALL_CODEBOOKS"` — K∈{2,3,5}
   instantiations already exist in source behind the ifdef. Also needs: a
   bench-swept `nt_of` column-strip entry for K=3 (`exl3_ops.sycl:181-183`
   currently special-cases K=4/K=6 from B70 sweeps; K<=4 branch would give K=3
   M=1→8 / else 4 — plausible but unswept), and acceptance that the **int8-prefill
   oneDNN fast path stays K∈{4,6}-only** (`exl3_ops.sycl:658`) — 3bpw prefill
   falls back to plain reconstruct+oneDNN fp16 GEMM. Correctness of the
   ifdef-guarded K=3 kernel bodies is unverified upstream (no test in-repo
   exercises K=3; `exl3_reconstruct` covers K=3 at `:545`, used by the slow path).
3. Recommendation shape: before any 13.8 GB download, decide between (a) rebuild
   image with `EXL3_ALL_CODEBOOKS` + a K=3 bit-exactness test vs the Triton
   reference, or (b) accept the Triton decode path and measure whether it even
   beats the current 4bpw champion-class numbers (unlikely per the author's own
   kernel-perf notes). The SC_3.00bpw_H4 (13.452 GB) needs K∈{2,3,5} — strictly
   more work than plain 3.00bpw for 0.37 GB less.

### 5. Bonus: llama.cpp #28796 (sparse QSA/FA) and #26689 (quantized-KV TILE) follow-ups

- **#26689 "SYCL: TILE for quantized KV decode"** — MERGED 2026-08-28, merge
  commit `d077b4c214`, touches `ggml/src/ggml-sycl/fattn.cpp`. Post-merge commits
  on that file: `cc83d7b482` (2026-08-29, `--fit` respect, unrelated), `f9f09f02cc`
  (2026-09-04, MKL-FA flag refactor #26863, mechanical), then #28796. **No
  regression reports and no follow-up fixes** touching the TILE logic found
  (issue search + file history, through 2026-09-29).
- **#28796 "[SYCL] support sparse FA"** — MERGED **2026-09-25** (only 4 days old),
  merge commit `cd74ef6274`, adds `fattn-sparse.cpp/.hpp`. **Zero follow-up
  commits and zero issue reports since merge** — no soak time at all.
- Caution signals for the flash-next lane (not SYCL regressions): CUDA-side
  sparse-FA had its own quantized-KV gap ([#28454](https://github.com/ggml-org/llama.cpp/issues/28454),
  closed: sparse-fa had no quantized-KV path, ~2.8x decode slowdown) — verify the
  SYCL sparse path's KV-type coverage on the actual build before pairing it with
  q8_0/q4_1 KV; and the model class itself has OPEN long-context issues on other
  backends ([#28805](https://github.com/ggml-org/llama.cpp/issues/28805) Metal:
  1-token-then-EOS at long context, threshold moves with quant/KV/n_ctx;
  [#28734](https://github.com/ggml-org/llama.cpp/issues/28734) CUDA: decode slows
  linearly with context).
- **Pinned-tree fact:** `~/llama.cpp` (HEAD `1692f9e50`, 2026-08-14,
  24 dirty files) contains **neither** merge — both post-date the pin. Using them
  means re-pinning/rebasing the local patch set, not flipping flags.

## Decision table

| Defect | Is a fix already upstream? | Which image/tag carries it | Local A/B that would prove it | Expected discriminator outcome |
|---|---|---|---|---|
| **1: 30-defect** (W4A16 sym-INT4 answers 30 not 14, greedy) | **The llm-scaler "sym_int4 output fix" is NOT our fix** (commit `ede4320a`/PR #675 — MoE-router ESIMD kernel, llm-scaler-only package, runtime sym_int4 quant, hard-corruption symptom; Qwen3.8-27B is dense `qwen3_5` — the fixed kernel cannot execute). No upstream report of our exact defect exists anywhere (closest: AutoRound #2142, unreproduced, not XPU-specific). | None applicable. llm-scaler `intel/llm-scaler-vllm:0.26.0-b2` (digest `sha256:52218ad8…`) carries the fix but for a path we do not run. | Probe 2A (running): same W4A16 checkpoints on CPU. If CPU answers **14** → XPU serving path (oneDNN `int4_gemm_w4a16` via vllm-xpu-kernels, or GDN kernels) is implicated → then a targeted kernel-tier dig (oneDNN int4 on BMG, vllm-xpu-kernels 0.1.12.3-vs-0.1.15 numerics). If CPU answers **30** → weights/export class is implicated (probe 2A verdict "file a bug" on the exporter/model side, with AutoRound #2142 as prior art) — no image upgrade helps. | 2A decides kernel-vs-weights. Do NOT spend a GPU window on llm-scaler-b2 for this defect. |
| **2: W4A16+MTP4 prefix-cache corruption (incl. zero-hit cold requests)** | **Yes — partially.** Best zero-hit-match mechanism is vLLM **PR #50729** (Mamba conv-state overlapping copy race), merged 2026-08-17 (`a02cfccb`), confirmed by #53912 OP; ships in vLLM **v0.29.0 / v0.30.0** (and v0.28.1rc+). Cache-hit-path bugs are NOT fixed anywhere: `MambaManager` `drop_eagle_block` ignoring (#48375/#57128) and the async accepted-count race (#53919/#51571) remain OPEN at v0.30.0. Our exl3xpu image (vLLM base `568afb3a1`, 2026-07-26, inside llm-scaler-b2) **predates #50729** — buggy code verified present at its base. | Any vLLM ≥ v0.28.1rc0 / v0.29.0 / v0.30.0 carries #50729. No XPU image we have carries it. llm-scaler has NOT rebased past it either (their vLLM base is the same 2026-07-26 commit). | On a #50729-containing build with the SAME AutoRound W4A16 artifact + MTP4 + cache ON: rerun the 2026-09-28 novelty cells (`results/2026/09/b70-novelty-20260928/RESULTS.md`): 3 fresh cold zero-hit cells + 3 same-prefix repeats, temp 0, 3072 tokens. Discriminators: (a) zero-hit first-request degeneration should go 5/6 → 0/6 if #50729 is our mechanism; (b) later same-prefix requests may STILL corrupt (open MambaManager bug) — that split would itself confirm mechanism C vs A. Cheap CPU-side pre-check: none (needs the GDN spec path). | If (a) fixes cold but not repeats → two-bug picture, matches thread; track #48375/#57128 upstream. If nothing changes → our corruption is in the fork/XPU-specific GDN path (then compare against vllm-xpu-kernels #510/#593 class). |
| **Sub-4bpw EXL3 route (A3 gate)** | Kernel support for width 48 exists in source but is **compile-time disabled**; pinned image binary lacks it. | Rebuilt exl3xpu image with `EXL3_FLAGS=-DEXL3_ALL_CODEBOOKS` (DIY — none published). | Zero-GPU first: rebuild `_C.so`, check `strings` for `ILi3ELi2E`, run `tests/test_bitexact_xpu.py` extended to K=3 vs Triton reference on one idle GPU window. Only then consider the 13.820 GB `3.00bpw` download. | Without the rebuild, a 3bpw download decodes on the Triton fallback — measure that first on the existing 4bpw artifact if curious (same code path), before any download. |
