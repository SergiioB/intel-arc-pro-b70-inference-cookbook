# RQ-45. W4A8 prompt GEMMs for TensorFold MLX g64

## 1. Result

Keep `XPU_PREFILL_INT8=1` opt-in. The Wave 2 same-binary observation on card 1 took 115.18 s to first token on the 130,856-token code prompt, down from 128.92 s. It missed the operator's approximately 95 s goal. Across eight standard code and prose prompts, the flag changed seven outputs. Drafted and serial output matched on all eight with the flag on. The short-prompt battery moved from 282/329 to 283/329 with four gains and three losses. W4A8 is an output-changing option, like the prefix kernel. It changed greedy text on seven of eight standard prompts. The 131K observation missed the latency target by 20.18 s. At the initial 08:00 CET cutoff, the build and confirmation arms were still queued. The operator extended the window to 23:30 CET and the timing jobs returned to the queue. Keep the fork option for latency-sensitive cold prompts where changed text is acceptable. The evidence does not support making it the default.

## 2. What you need

- Intel Arc Pro B70 card 1 (`0000:0f:00.0`, 230 W cap, 2,800 MHz limit) for single-card timing. Add card 0 (`0000:0b:00.0`, 180 W cap, 2,400 MHz limit) for tensor parallelism. Linux `7.0.0-28-generic`, 60 GiB system RAM and 15 GiB swap.
- TensorFold base `2286abc0c45c284dae60a9286ae02612eda3b44c`. W4A8 commit `4b21412ba2dd28c5cbabea07d8f9004e6dfd9a3e`. Both produce the original committed source tree at their corresponding pre-rebase commits.
- Zig 0.17.0, `/usr/bin/ocloc-26.05.1` from the lab toolchain and nGEN headers under `$TF/onednn/third_party/ngen`. The worktree carries a one-line local ocloc flag change. That line is outside the patch.
- Target `$TF/models/Qwen3.8-27B-MLX-4bit`, MLX 4-bit affine weights in groups of 64, about 15 GiB. Drafter `$TF/models/Qwen3.8-27B-DFlash2`, about 3.6 GiB. Use bf16 K/V. Raw timing artifacts go to `$WORK/prefill8/`.
- `$TF=$TF` in the commands below. At intake `$DATA/sata1` had 35 GiB free and `$DATA/samsung980` had 372 GiB free.

## 3. Timeline

All clock times are CET on November 18–19, 2026.

| Time | Step | Outcome |
|---|---|---|
| Nov 18 08:55–08:58 | Wave 2 build | Build exited zero. Zig printed `failed command` for a fake Level Zero launch error in a passing test. |
| 10:22–10:39 | Wave 2 device checks | Prompt seam, verify and drafted/serial checks passed. |
| 13:19–15:17 | First battery attempts and prompt checks | The early battery did not complete. The eight-prompt run completed for both settings, drafted and serial. |
| 23:24–23:41 | Lead's flag-on battery | 329 outputs, 283 passes. |
| 23:35–23:37 | First post-reboot flag-off attempt | The server tried port 8046, which the flag-on server still held. The lead preserved the partial log. |
| 23:37–23:55 | Flag-off battery on 8050 | 329 outputs and 282 passes. The job wrapper returned 1 after grading. |
| 23:40 | Rebase | Rebased onto the published base with the same tree. Preserved the worktree's local compiler line and generated the patch. |
| 23:41–23:43 | Submitted build and confirmation | CPU build and three exclusive timing jobs entered the shared queue. |
| 23:55 | Pushed the fork branch | Remote ref verified at `4b21412ba`. |
| 23:58 | Lead restarted scheduler | Build and timing remained pending behind tensor-parallel jobs. |
| Nov 19 00:00–00:01 | Verified queue dependencies and memory limits | Used the full build filename stem and verified a system-slice memory cap before GPU work. |
| 04:12–04:13 | Updated pending jobs after queue changes | Set the 08:00 cutoff and 60 °C / 68 °C cooling limits. Checked that the freshness guard rejects the old executable. |
| 05:35:54–06:28:38 | Earlier two-card sweep | Sweep completed before the next higher-priority jobs started. |
| 08:00 | GPU cutoff | All four Wave 3 jobs still pending. No prefill8 build or timed arm started. Archived the three expired measurement job descriptions. |

Elapsed from the 23:39 intake to cutoff was 8 h 21 min. The earlier port-clash attempts occupied about 2 min of wall time before the replacement. The 18-minute replacement generated and graded the complete flag-off battery. It lost only its wrapper cleanup receipt. Wave 3 prefill8 used zero minutes of GPU time. Queue evidence (lab file `queue-at-cutoff/events.log`, not published).


## 4. Step by step

1. Rebase the original commit. Set both Git identities to the noreply identity on the base before the rebase and amend.

   ```bash
   cd $TF/src-prefill8
   git rev-parse d073d666f^{tree} 2286abc0c^{tree}
   git rebase --autostash --onto 2286abc0c d073d666f
   git commit --amend --no-edit --reset-author
   git diff --exit-code 942fff53d HEAD
   git diff --check HEAD^ HEAD
   ```

2. Submit `build_wave3.sh` through the shared queue on `"cards": "none"` with `"exclusive": true`. Run the build through `queue_scope.sh 24G`. The build script runs this command in the TensorFold worktree and verifies the host test count.

   ```bash
   zig build xpu-qwen27 xpu-native xpu-qwen27-tests -Doptimize=ReleaseFast \
       -Dngen-include=$TF/onednn/third_party/ngen --summary all
   ```

3. Submit the following commands as exclusive jobs with `after: ["300-prefill8-build"]`. Run each through `queue_scope.sh 28G`. Set `D=~/B70-DOCS/scripts/tmp/breakthrough_20261118/prefill8` and `W=$WORK/prefill8`. The cutoff argument is the epoch time for November 19 23:30 CET.

   ```bash
   bash "$D/confirm_wave3.sh" "$W/prefill8-tp1-long" 1 5 1795127400 128k
   bash "$D/confirm_wave3.sh" "$W/prefill8-tp1-short" 1 5 1795127400 8k 32k
   bash "$D/confirm_wave3.sh" "$W/prefill8-tp2" 2 3 1795127400 32k 128k
   ```

4. Compare the lead's graded batteries and the eight standard prompt outputs from the B70-DOCS root.

   ```bash
   R=results/2026/11/b70-breakthrough-20261118
   python3 "$D/audit_quality.py" "$R/wave2/prefill8" \
       results/2026/11/tf-prefix-attn-20261113/grade-tmp/graded.json \
       "$R/wave3/prefill8" > "$R/wave3/prefill8/quality.md"
   ```

5. Generate the patch from the committed tree and push the approved branch.

   ```bash
   git -C "$TF/src-prefill8" format-patch --no-signature -1 \
       --output-directory ~/B70-DOCS/results/2026/11/b70-breakthrough-20261118/wave3/prefill8
   git -C "$TF/src-prefill8" push fork HEAD:refs/heads/xpu-qwen27-w4a8-g64-prefill
   ```

## 5. Every fix

| # | Symptom | Cause | Fix | Verification |
|---:|---|---|---|---|
| 1 | `Level Zero : 0x1` and `failed command: .../test ... --listen=-` | The fifth host test makes a fake Level Zero kernel launch return 1. `ze.check` prints the code to stderr. Zig prints the command with its warning. | Run the test executable directly and count its assertions. | Wave 2 `dev1/hosttest.txt` ends with `All 20 tests passed.`. |
| 2 | `tensorfold: cannot listen on 127.0.0.1:8046 (AddressInUse)` | The initial flag-off battery used the flag-on server's port. | Use scheduler-assigned card and port. | The replacement flag-off server listened on 8050. The discarded run is in `battery-off-portclash-discard/`. |
| 3 | `ValueError: nonce could not preserve the prompt's token length` | Decoding a sliced token array can change the text. Slicing raw text at token offsets can still merge tokens across the nonce boundary. | Keep the original text by offset and pad the nonce where a boundary merge skips the exact length. | CPU dry run produced 30 unique length-matched prompts at each of three depths. |
| 4 | `rc=1` after the completed flag-off summary | The EXIT trap tried to kill the same PID after the script had already terminated it for grading. With `set -e`, the trap exited before writing its receipt. | Make `battery_run2.sh` shutdown idempotent by checking the PID first. Verify the earlier shutdown from its log, process table and container list. | PID gone, no owned grader, both caps match. The patched stop function passed a two-call test with an owned dummy process. Cleanup proof (lab file `battery-off.cleanup-observed.txt`, not published), test log (lab file `battery-cleanup-test.txt`, not published). |
| 5 | The timing jobs referenced `prefill8-build` in `after`. | The live scheduler compares full filename stems. | Use `300-prefill8-build` in all three timing job files. | All three GPU job descriptions name the exact build job stem. |
| 6 | `Missing or newer TensorFold build input: zig/build/xpu.zig` | The Wave 2 executable predates the rebased worktree state. A scheduler dependency can finish with a failure. | Require a successful build receipt with matching input and executable hashes. Reject a missing executable or a newer build input before GPU checks. | The guard rejected the existing Wave 2 binary at 04:13. |

## 6. Verification

- The [format patch](0001-xpu-qwen27-W4A8-prompt-GEMMs-for-groups-of-64-MLX-4-.patch) is the single commit on top of `2286abc0c`. The verified [fork branch](https://github.com/SergiioB/TensorFold/tree/xpu-qwen27-w4a8-g64-prefill) points to it. The rebased commit has the same tree as `942fff53d`. The only worktree modification is the existing ocloc line. `git diff --check HEAD^ HEAD` passed.
- Eight standard prompts were run with the flag on and off, drafted and serial. SHA-256 recomputation matched all 32 nonempty outputs. The only common FP16/int8 text was the 130,856-token code prompt. The exact hashes are in [text-parity.tsv](text-parity.tsv).
- Wave 2 device checks in dev1 (lab file `../../wave2/prefill8/dev1`, not published) include two seam cuts per flag. All four matched all 248,320 logits. The flag-on perfect and wrong draft test passed both comparisons. The flag-on DFlash2 dev test matched serial on 256 generated tokens.
- At cutoff the queue had launched 0 of 4 Wave 3 jobs. The fresh confirmation completed 0 of 21 pairs. The rebased executable and a fresh host-test pass remain unverified. Queue snapshot (lab file `queue-at-cutoff/status.json`, not published), planned pair coverage (lab file `timing-coverage.md`, not published).
- Queue records live under `$WORK/queue/`. The expired job files are in queue-at-cutoff (lab file `queue-at-cutoff`, not published). The CPU-only build remains queued.

## 7. Results and claim boundary

### Battery and text

| Set | Items | ADR stock | ADR prefix | Flag off | Flag on |
|---|---:|---:|---:|---:|---:|
| HumanEval tune | 82 | 78 | 78 | 78 | 78 |
| HumanEval test | 82 | 76 | 76 | 76 | 76 |
| Professional knowledge | 55 | 40 | 42 | 40 | 41 |
| Structured output | 55 | 55 | 55 | 55 | 55 |
| Instruction following | 55 | 33 | 33 | 33 | 33 |
| Total | 329 | 282 | 284 | 282 | 283 |

Each current arm graded all 329 items once. Both used the same pre-rebase binary, greedy DFlash2, one stream, context 16,384, and cache disabled. The flag-on arm ran on card 1 at 230 W and the flag-off arm ran on card 0 at 180 W. Both set `XPU_ATTN_PREF=256`. Preserve the difference in hardware when interpreting individual flips. The historical controls are from a different build and session. Same pass totals do not establish quality equivalence.

| From | To | Fail → pass | Pass → fail | Equal output hashes |
|---|---|---|---|---:|
| Flag off | Flag on | he-032, he-093, prof-biz-401, prof-law-1935 | he-116, he-127, prof-biz-484 | 201/329 |
| ADR stock | Flag off | 0 | 0 | 305/329 |
| ADR stock | Flag on | he-032, he-093, prof-biz-401, prof-law-1935 | he-116, he-127, prof-biz-484 | 201/329 |
| ADR prefix | Flag off | 0 | prof-biz-401, prof-law-1935 | 261/329 |
| ADR prefix | Flag on | he-032, he-093 | he-116, he-127, prof-biz-484 | 201/329 |

The run receipts are flag on (lab file `../../wave2/prefill8/battery-on/grade/graded.json`, not published), flag off (lab file `../../wave2/prefill8/battery-off/grade/graded.json`, not published), and ADR controls (lab file `../../../tf-prefix-attn-20261113/grade-tmp/graded.json`, not published). [Per-item grades and hashes](battery-flips.tsv).

### Timing

The Wave 2 results use Qwen3.8-27B MLX g64, DFlash2, one greedy stream, thinking off, 512 requested output tokens, and cache disabled. The host ran card 1 at 230 W for one-card timing and both cards at their as-found caps for tensor parallel timing. Time to first token is client-observed. Divide actual prompt tokens by client TTFT for the cold input rate. This rate includes first-token work.

| Setup | Actual prompt | Pairs | TTFT off → on | Client decode off → on |
|---|---:|---:|---:|---:|
| Card 1 | 7,976 | 3 | 4.653 → 3.827 s | 114.47 → 111.01 tok/s |
| Card 1 | 32,552 | 3 | 21.567 → 18.145 s | 89.88 → 88.93 tok/s |
| Card 1 | 130,856 | 1 diagnostic | 128.918 → 115.176 s | 50.7 → 50.7 tok/s |
| Both cards | 7,976 | 3 | 3.605 → 3.156 s | 140.66 → 142.67 tok/s |
| Both cards | 32,552 | 3 | 16.171 → 14.200 s | 115.89 → 116.85 tok/s |

The Wave 2 code screen reached 1.216× at 8K on one card and 1.142× on two cards. Both are below the original 1.25× RQ-45 gate. Flag-on 131K decode held at 50.7 tok/s for the code prompt. Decode rates varied slightly on other prompts where the text and draft acceptance changed. Provisional Wave 2 screen. Sources are ab.console (lab file `../../wave2/prefill8/ab.console`, not published) and text outputs (lab file `../../wave2/prefill8/text/summary.md`, not published). Timing observations at 131K target the same context regime as the historical 85 s EXL3 result. Engine and weight format differ.

The 7,976-token dev profile showed prompt GEMMs using 3.62 s of summed GPU time with the flag off. Int8 GEMMs and their quantizer used 2.86 s with the flag on. That accounted for about 0.75 s of the 0.84 s reduction in summed GPU time. The two diagnostic traces started at 63 °C and 47 °C respectively. Flag-off profile (lab file `../../wave2/prefill8/dev1/prof-off.txt`, not published), flag-on profile (lab file `../../wave2/prefill8/dev1/prof-on.txt`, not published). The longer prompt has more attention work, which this kernel does not accelerate. The 131K observations saved about 14 s, while the 95 s target requires saving about 34 s from the existing one-card baseline.

### Not tested

- Sampled decoding, thinking on, more than one stream, cache reuse with W4A8, or long-context recall quality.
- W4A8 battery repetition, g128 endpoint regression, or other devices and power limits.
- The requested Wave 3 matrix, fresh host tests on the rebased executable and a new text run on that binary.
- A matched latency ranking against EXL3.

## 8. If you do this again

Allow about five minutes for the queued build. Keep the build inside the CPU queue. Resubmit the archived GPU jobs with a fresh authorized deadline. Use one shape-matched warm-up per arm, retain the first-token and prompt-token receipts, and test the 131K card-1 cell first. Allow about 45–65 minutes for five long pairs, depending on cooldown. Reuse the eight-prompt check when the binary hash and all inputs match.

## 9. Adapting it

Check the weight group, affine bias layout, and activation quantizer scale layout before reusing the kernels. A new group size changes both accumulation and quantization boundaries. Repeat the chunk-boundary and drafted/serial checks. Use a fresh battery for each new checkpoint.
