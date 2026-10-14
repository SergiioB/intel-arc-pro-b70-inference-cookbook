# FlashNext EXL3 port — sanitized evidence bundle (2026-10-14)

Source of every file here: `/data/nvme/work/flashnext-kl` and the port tree
`/data/nvme/engines/exl3xpu-mixedk/flashnext`. Nothing here contains credentials, tokens or keys;
the large raw tensors are deliberately NOT copied (see "Not included").

## Exact inputs and pins
| item | value |
|---|---|
| artifact | `turboderp/Qwen3.8-Flash-Next-exl3`, branch `2.05bpw_h4_ng4`, commit `65c895314393431c09050b2e04e250836b3a6eb4` |
| BF16 reference table | `Qwen/Qwen3.8-Flash-Next` rev `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| pinned ref decoder | sha256 `3f614b7c91510d8520202e85dd0a08255370349a4d5ae4e7601edfa6897d1780` |
| tokenizer (panels) | sha256 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| engine ref | exllamav3 XPU, torch 2.14.1+xpu, driver 1.14.37020, card UUID `…0f00…` |
| tools | `fn_kl_eval.py` sha256 `182ae76d…`, `fn_sources.py` sha256 `a4f9b00d…` (copies in `source/`) |
| exact forward input | panels row, **20 token ids** (`parity20.json -> ids`), batch 1, no padding, full model, reference FP32 |

## Measured results
1. **Native CLI exact-answer smoke checks**: `The capital of France?` -> `Paris`,
   `Germany?` -> `Berlin`, `17+25` -> `42`, all repeat-identical, multiturn France -> Germany
   answers `Paris` then `Berlin`. These are qualitative correctness checks; the model is
   **numerically unvalidated**.
2. **Port gate at the head (declared before the numbers)**: argmax PASS (`248046` = `248046`),
   **top-5 id sets FAIL**, **cosine 0.98681 FAIL**, **relative Frobenius 0.16250 FAIL**.
   Verdict: **NOT parity-validated**. The README records experimental scope only; no validated serving recipe is claimed.
3. **Reference pass (EXL3)**: full forward **COMPLETED**, `forward_seconds 18247.24` (5 h 4 m),
   48/48 layers, `2026-10-14T10:41:23Z`, inside its fixed 11:04:20Z deadline. It then died in
   `mode_parity` with `NameError: name 'save_file' is not defined` (function-local safetensors
   import vs module-scope writer). **Serialization failure of the tool chain, not a numerical
   failure of model or port.** The per-layer raw tensors were memory-only and are lost; no core
   dump, no partial file. See `logs/save_file_failure_proof.txt`.
4. **Surviving per-layer comparison (aggregate statistics only)**: `layer_statistics.json`.
   Layer 0 agrees closely (std ratio 0.99952, absmax ratio 0.99779), layer 2 within ~2-19 %, and
   **layer 1 shows the first strong deviation (std ratio 0.65878, absmax ratio 1.75740)** - the
   region containing the PLE. This is a **weak region hypothesis only**: it compares the same
   quantization's native XPU path against its FP32 reference path. It **cannot** be attributed
   to differences between the artifact's n-gram rows and the official BF16 rows; aggregate
   statistics are not the port verdict. Element-wise localization needs the raw capture that was lost.
5. **Capture defect**: native `m_49` and `m_50` (reference layers 46/47) are bitwise identical;
   flagged (pair + "verify capture reuse"), analysis continues - legitimate identical tensors are
   possible, so it is a review flag, never a hidden special case.
6. **BF16 KL input**: arm ended at its fixed deadline (`SIGINT 11:26:05Z`, `SIGTERM 11:27:05Z`,
   gone `11:28:05Z`) at 20/48 layers, zero errors (`logs/bf16_deadline_end.txt`). No BF16 logits
   exist, therefore **the artifact/reference-weight KL is absent - not estimated**.

## Fixes landed after the loss (all CPU-validated)
* module-scope `from safetensors.torch import save_file, load_file` in `fn_kl_eval.py` (root cause);
* per-layer disk checkpoints written **during** the forward, into `raw_layers_partial/`;
* `selftest-write` mode exercising the real writer helpers: 4/4 checkpoints on disk while the
  capture loop is still running, consolidated round-trip bitwise identical, and a simulated crash
  before the consolidated write still leaves every layer on disk (`logs/save_selftest_report.txt`);
* duplicate-tensor **flag** (not a refusal) and the attribution tool `fn_divergence_attrib.py`.

## Not included on purpose
`logits_exl3.safetensors` (3.1 GB) and `ref_row0_pos76.pt` (3.1 GB) stay in
`/data/nvme/work/flashnext-kl/runs/main-20261014T032613Z/` (outside sync). This bundle is 5.2 MB.

## Owned-process cleanup (actual, at hand-off)
No process of mine is alive: both arms gone (`arm_deadlines.txt` ends with "both arms no longer
present" for the first enforcer batch, then "bf16 deadline reached -> SIGINT / SIGTERM" and
"both arms no longer present" at 11:28:05Z); no GPU/VRAM held; nothing committed in the port repo
(HEAD still `03d106c`, `flashnext/` untracked); no foreign process touched.
