# Wave 9G final sign-off: `vllm-hf-int4-g16-xpu` v0.2.0

Reviewer: Opus 5.5. Read-only review: no edits, no GPU or Docker runs. Date 2026-11-25.
Door: **one-way** for the `hf_int4_g16/2` on-disk contract. **Two-way** for the Wave 9F code delta and the README.

## Verdict: APPROVED FOR v0.2.0 RELEASE (WITH DOCUMENTED SCOPE)

1. **`hf_int4_g16/2` uniform-Had64 format contract: CLEARED TO FREEZE.** Blocker #1 from Wave 9C is closed under the preregistered rule 9.
2. **Wave 9F code delta: approved.** It adds no new code blockers.
3. **Release condition (doc-only):** cut the tag only after the three README edits in section 4. Each is a two-way door. I only need to read their diff. Nothing needs to rerun.
4. **No Gemma-4 `/2` checkpoint may ship with the converter's default raw-text BiasCorr.** That bias failed the 4-check gate (section 2).

## 1. Blocker #1: uniform Had64 on Gemma-4-26B-A4B-it (closed)

Provenance: the frozen hashes match for 5 of 5 files (`exp17d_frozen_before_main.sha256`). `PREREGISTRATION.md` (20:47) is older than every Exp17D score. Run A (21:50) and run B (22:04) are bit-identical: 207 arrays and 423,108 values, 0 differing. The patched-zero-bias control is exactly 0. The "uniform" arms read the bytes that the unmodified v0.2.0 converter wrote: 48 of 48 tensors are bit-equal to a re-quantization.

| Check (rule 9) | Value (KL nats/token, 2,044 targets/domain) | Pass |
|---|---|---|
| U64 + shared BiasCorr, 3-check (harness) | WT2 0.138 → 0.097, CI [-0.055, -0.028]; Code 0.193 → 0.138, CI [-0.077, -0.033] | yes |
| U64 + shared BiasCorr, 4-check OOD (harness) | 1.167 → 0.982, CI [-0.263, -0.111]; `-b` worsens it by +0.97 | yes |
| U64 + shared BiasCorr, 4-check (live plugin kernels) | OOD 1.215 → 1.013, CI [-0.304, -0.108] | yes |
| Uniform − mixed, BC_shared, paired | WT2 +0.001 [-0.008, +0.011]; Code +0.010 [-0.007, +0.027]; OOD **-0.113 [-0.183, -0.048]** | not worse on any domain |
| Uniform − mixed, DF, paired | WT2 -0.001; Code +0.017 [-0.003, +0.036]; OOD +0.003 | not separable |
| Had × BiasCorr interaction (WT2+Code) | Had alone -0.022, BC alone -0.058, both -0.070; interaction +0.010 [-0.008, +0.027] | the two effects add; they do not cancel |

Mixed + shared fails the 4-check gate: OOD CI [-0.175, +0.028]. Uniform passes, so the format change costs nothing measurable.

Claim boundary (keep it in the release notes):
1. The uniform-better-on-OOD result (-0.113) comes from one model, one calibration draw and one noisy domain: BF16 PPL is 143, the rounding floor is 0.081, and 1,451 of 2,044 targets fall outside model turns. Report it as "not worse". Do not headline it as "better".
2. Code under DF (+0.017, upper CI bound +0.036) comes close to separable. It sits near the code rounding floor of 0.011. Watch this on the next MoE.
3. Live minus harness on OOD is separable (+0.048 DF, +0.031 BC) but stays below the OOD floor. The kernels are not the gap.
4. The gate covers E=128 only. Uniform Had64 has no KL measurement on Qwen3.6-35B (E=256).
5. The attempt-1 control broke at layer 4 (0.25), and the root cause is **not established**. It touched the stock BF16 HF path, not the plugin, and two later full runs are bit-identical. It stays disclosed in the receipt.

## 2. Served checkpoint and live `vllm serve` (Gate 3B)

1. The server booted under `Gemma4ForConditionalGeneration`, full multimodal, with no `--language-model-only`. Coverage was verified before the first forward (205 linear, 30 expert modules). The log has 0 error lines.
2. Single-stream decode: 83.2 / 70.6 / 55.3 tok/s at 51 / 1,988 / 8,133 prompt tokens. This was **not on a quiet host** (GPU1 busy, desktop session on GPU0). Keep these numbers inside the lab.
3. 8 of 8 canaries pass, greedy repeats are identical, and all logprobs are finite. The code canaries execute 512 of 512 tests (8 palindrome + 504 intervals). The brief said "504/504"; that is one canary only. The vision canary passes (red circle, blue square, "472"), and the repeat is identical.
4. **That server ran the converter's raw-text BiasCorr.** That bias fails the 4-check gate: OOD +0.044, CI [-0.074, +0.165] (harness), and the live kernels fail it as well. Holding back the raw-cal sidecar was the right call.
5. Not tested: a `vllm serve` boot of the chat-cal bias (`gate2b_gemma4_shared_only_biascorr_chat_cal.safetensors`, sha `a374418f…`). That gate belongs to the checkpoint release, not to the plugin. Before any Gemma `/2` upload, boot that exact checkpoint and rerun the 8+1 canaries.

## 3. Wave 9F delta (approved)

Tree integrity: all 8 file sha256 values in `exp17_moe_plugin_pytest_wave9f_gpu0.log:3-11` match the current tree. The zebin `hf_int4_kernels_bmg.bin` = `5ce45179…`, and `hf_int4_kernels.cl` is byte-identical to the 9D snapshot, so the GPU kernels did not change. Only the host-side `_l0_native.cpp` changed. Suite: 699 passed, 0 failed (683 + 16 new). Mutations: 8 of 8 caught, and the unedited control passes. The new file passed 12 of 12 fresh-process repeats.

| Change | Assessment |
|---|---|
| Opaque `GroupedBucketDescriptor` capsule (`_l0_native.cpp` patch ~407-432, 771-776, 825-838) | Correct. Python can no longer forge or swap the buffers that the unchecked tile kernel uses as addresses. `moe_tile` checks pairs, E and device from metadata only, so it adds no host sync and stays capture-safe. The `P ∈ (0, 2^29]` bound moved to `moe_bucket` (`_l0_native.cpp:742`). The capsule's `unique_ptr`/`release` ordering does not leak on a throw. The M=16 vs M=17 test proves that capacity alone cannot tell the two batches apart, so the pair-count check is required. |
| Granite member names scoped to raw `block_sparse_moe` (`moe_admission.py` patch ~263-305) | Closes the alias hole. Two tests plus the Qwen-layer negative. |
| `moe_grouped_gemm` rejects other output dtypes before rotation (`moe.py:179-180`) | Correct. The rejection fails before any kernel runs, and it no longer coerces silently. |
| `--cal-chat-template`, off by default, fails closed (`calibration.py`, `converter.py`) | Fine as an opt-in. The tokenizer check runs before staging, and the test asserts that nothing gets written. Content-only overlap invariant, rectangular trim ≤ 64 from the content end, never from the template. The config and report record it. The real-tokenizer smoke gives Gemma 530 tokens with one BOS at position 0. |
| Segfault control removed from the capture test | Acceptable. The bisect shows the segfault comes from a host sync (`.item()`) that the test itself made inside capture (variant C crashed 4-7 of 15 times; B passed 60 of 60). That is an illegal operation, not a plugin path. Follow-up: file it as a torch-xpu/Level Zero bug, so that a future stray sync errors out instead of crashing. |

Non-blocking: the README states that `--cal-chat-template` is unmeasured. That is correct. Its rows (prefix and template tokens, 2×~530) differ from the harness chat calibration that passed (Exp17B chat prefix, 8×511). The sidecar that passed is therefore **not** reproducible with the converter today. Measure the flag before it becomes the default for `-it` models.

## 4. Release condition: three doc-only README edits

| # | file:line | Problem | Required text |
|---|---|---|---|
| 1 | `vllm-hf-int4-g16/README.md:129` | It says no real MoE checkpoint has been converted or served, and that `vllm serve` and the multimodal wrappers have not run with MoE. Exp17D did both. | Replace it with what Exp17D did: Gemma-4-26B-A4B-it converted and served under `Gemma4ForConditionalGeneration`, single stream, not quiet-host. List what is still untested: concurrency above 1, context above 16K, `Qwen3_5MoeForConditionalGeneration`. |
| 2 | `vllm-hf-int4-g16/README.md:131` | It says no quality numbers describe the uniform Had64 layout. Exp17D measured that layout. | State the Exp17D result: on Gemma-4-26B, uniform minus mixed is not separable on WT2 or Code, and uniform passes the 4-check gate with chat-calibrated BiasCorr. Include the claim boundary from section 1. |
| 3 | `vllm-hf-int4-g16/README.md:31` (or 41) | There is no warning that the default `mixed_train` raw-text BiasCorr failed the 4-check gate on an `-it` model. | One sentence: on Gemma-4-26B-A4B-it, raw-text BiasCorr passed WT2/Code and failed OOD chat. Gate the bias of an instruct model on chat-format eval before shipping it. |

## How to revert

- Code: `git revert` the 9F commit, or pin the wheel to 0.1.0. The `/1` bytes and outputs stay identical (Wave 9E, 308/308).
- Format: no `/2` checkpoint is published yet. Freezing the layout rule changes no bytes until the first upload. After that upload, a revert means a `/3` version bump.

## Evidence read

`opus55_wave9e_v020_signoff.md`; `results/exp17d_gemma4_26b_had64_ablation_and_live_vllm.json` (verdicts, kl_gate, paired_blk32_95ci, ablation, sidecars, part2, not_tested, failed attempt); `results/exp17d_scripts/{PREREGISTRATION.md, PROCESS.md, exp17d_validation.json, exp17d_ood_split_posthoc.txt, exp17d_frozen_before_main.sha256, exp17d_canary_code_exec.json}`; `results/gate2b_gemma4_shared_only_meta.json`; `results/exp17_hfg16_wave9f_delta.patch` (782 lines, read in full); `results/exp17_moe_plugin_pytest_wave9f_gpu0.log`; `results/exp17_wave9f_{mutation,repeat,segfault_bisect,real_tokenizer_smoke}.log`; the current tree's sha256 values and `README.md`.
