# FlashNext EXL3 port — status, evidence and open blockers

Operator-directed campaign (parent/Main directed, user authorised). Standalone work on the
Flash-Next EXL3 artifact: **not** a reopening of any held kernel, and no adoption status or
registry entry is claimed here. Everything below is either a measured number or an explicitly
named blocker.

## Pins
| item | value |
|---|---|
| artifact | `turboderp/Qwen3.8-Flash-Next-exl3`, branch `2.05bpw_h4_ng4` -> commit `65c895314393431c09050b2e04e250836b3a6eb4` (local files size+hash identical; `qbench_prompts*` deliberately not downloaded) |
| BF16 source table | `Qwen/Qwen3.8-Flash-Next` revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| pinned ref decoder | `refsrc/ref.py` sha256 `3f614b7c91510d8520202e85dd0a08255370349a4d5ae4e7601edfa6897d1780` |
| tokenizer (panels) | `tokenizer.json` sha256 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| engine | vendored exllamav3 XPU build, ext sha256 `a13ac880828e1963…`, torch 2.14.1+xpu, driver 1.14.37020 |
| tools | `fn_kl_eval.py` sha256 `182ae76d0e9de1012911e654bb9d617f12d55675c58046826ca6610fde71fecb`, `fn_sources.py` sha256 `a4f9b00d1b16e35e9d3cfc85c6361942898c2388dbd7eea9e4b974a006e54690` |

## Fixes landed (each CPU-verifiable)
1. **QSA plane write** (`exllamav3/modules/qsa_indexer.py::update_planes_ref`) — the XPU port
   raises `numel: integer multiplication overflow` for the bool-mask select `pflat.flatten()[vm]`
   and for `torch.nonzero(vm)` on a sane `(bsz*nb_max,)` mask; selection indices are now derived
   on the CPU and the write is `index_select` + `index_copy_`. Cost for the actual chunk this port
   ran (plen 77, layer 3, original x `(1, 76, 2560)`, `nb_max = 19` selected compressed blocks):
   a `(bsz*nb_max,)` = 19-element bool mask plus its index vector, ~170 B in total, plus one
   host<->device sync for that chunk's selected positions. This is not a per-token figure. The
   `nb_max == 0` path performs none of this work, which is known from the code (the
   `if nb_max > 0:` guard), not from a benchmark.
   Evidence: `runs/engine1-20261014T010425Z/qsa_dbg*.log`, `semantic_clean.log`.
2. **Greedy sampling** (`generator/sampler/custom.py`) — the greedy mode no longer touches the
   extension's missing `FUSED_SAMPLER_*` symbols; it takes the real `torch.argmax` path
   (one clean path; masked greedy and the sample/filter modes still fail loudly). No hand-copied
   constant was exported.
3. **Job errors surfaced** (`flashnext/serve/fn_generate.py::run_generate`) —
   `generator.reap_failed_job()` results (`stage="error"`) were silently returned as an empty
   success; now printed, recorded and raised (`FN_TOLERATE_JOB_ERROR=1` keeps the record only).
4. **Coverage** (`tools/fn_kl_eval.py`) — quadratic `any(n.startswith(...))` over 77,112 artifact
   names replaced by a prefix set (coverage now ~4 s, was >hours); EXL3 keys resolve through
   quantized stems (`.trellis/.suh/.svh/.mul1`); `ngram_tables()` reads the artifact's own
   n-gram metadata and returns the real stem keys. Final: bf16 consumed 1294 / leftover 0 /
   problems 0; exl3 consumed 148,963+ / leftover 4 (named n-gram-embedding buffers) / problems 0.
5. **N-gram row decode** (`tools/fn_sources.py::EXL3Source._ngram_rows_upstream`) — default and
   only path for `exl3_ngram_trellis`: 41 int16/row, word 0 fp16 scale, mul1 ring, `+ head_bias[head]`.
   Decode verified **bit-exact** against the extension's own `ngram_dequant`
   (`logs/ngram_dequant_oracle.json`, `logs/ngram_row_gate.json`).
6. **Signed CDN URL refresh** (`tools/fn_sources.py::_HTTPFile`) — a fresh resolve at the pinned
   revision for the exact failing range returns 302 -> signed URL -> 206, so a long run outliving
   `Expires` no longer fails; 401/403/410 re-resolve once and retry the same range (counter
   `url_refreshes`).
7. **Gather coalescing** (`tools/fn_kl_eval.py::_gather`) — `max_gap_bytes=1<<20` merges
   near-adjacent n-gram rows into few spans instead of one request per contiguous run.
   **Byte-identical**: `torch.equal` True and identical sha256 of the returned bytes
   (`9c1966b6…c98fd`) with coalescing off vs on, on real BF16 ids.
8. **Per-layer disk checkpoints** (`mode_parity`) — **added for FUTURE runs only; it is not in
   effect for the currently running capture, which still holds its per-layer tensors in memory
   and writes them at the end.** With this change every captured reference layer is written to
   `raw_layers_partial/layer_<i>.safetensors` as it is produced, so a run cut short by a deadline
   preserves all completed layers.

## Measured behaviour of native inference through the engine CLI (card ordinal 1, UUID `…0f00…`)
* Scope note: these are direct `fn_generate.py` CLI generations on the port, **not** an OpenAI-compatible
  API server and not a served-endpoint measurement; no HTTP server was started for these numbers.
* `The capital of France?` -> `Paris`; `Germany?` -> `Berlin`; `17+25` -> `42`; all repeat-identical;
  multiturn France -> Germany answers `Paris` then `Berlin`.
* Frozen panel prefixes: plen 48 (formerly SIGABRT) and plen 77 (formerly error/zero) both generate.
* Native first-step logits on a 20-id chat prompt: argmax `248068`, top-10 recorded
  (`runs/engine1-20261014T010425Z/native_first_logits.pt`).

## Declared port gate and current verdict
Gate (declared before the numbers): same-quant parity at the first step - argmax equal, top-5 id
sets equal, cosine >= 0.99, relative Frobenius <= 0.02.
Measured on the frozen panel prefix (heldout row 0, prompt 77, reference slice = panels row 2):
argmax `248046` = `248046` PASS; top-5 sets differ (4/5 overlap) FAIL; cosine 0.98681 FAIL;
rfn 0.16250 FAIL. A position sweep (six positions of the same row) shows the same picture:
argmax equal everywhere, top-5 unequal everywhere, rfn 0.18-0.34, native std 5-13 % below the
reference at every position. These are **position-wise observations only: they do not localize,
and must not be read as evidence about, any layer's cause or about whether the deviation
accumulates with depth.**
**The port is therefore NOT parity-validated, and no support table or README claim is made.**

## Outcome of the 2026-10-14 reference campaign (measured, with limits)
* **Reference parity pass (EXL3): the full forward COMPLETED** - `forward_seconds 18247.24`
  (5 h 4 m) over all **48/48** layers, batch 1, exact 20 ids, timestamp `2026-10-14T10:41:23Z`,
  inside its fixed deadline. It then died in `mode_parity` with
  `NameError: name 'save_file' is not defined` because the safetensors import was function-local
  while the raw writer runs at module scope. **That is a serialization/write defect of this tool
  chain, not a numerical failure of the model or of the port.** Consequence: the per-layer raw
  tensors were held in memory only and **were lost**; there is no core dump (core_pattern is
  apport; no python crash entry) and no partial file (per-layer checkpoints did not exist yet in
  that build).
* **What survived**: `parity20/parity.json` (5.1 MB) with the exact ids, `logits_last` (the
  20x248320 reference head logits), per-layer aggregate statistics for all 48 layers, the n-gram
  shim table and the head metadata; plus the native side
  (`parity20/native_raw_modules.safetensors`, 53 modules x 20 x 10240).
* **Statistics-only localization (weak, stated as such)**: comparing the surviving reference
  per-layer `mean/std/absmax` with the identity-mapped native module statistics gives: layer 0
  agrees closely (std ratio 0.99952, absmax ratio 0.99779), layer 2 within ~2-19 %, while
  **layer 1 is the first strong deviation (std ratio 0.65878, absmax ratio 1.75740)**, the region
  that contains the PLE. This is a **weak region hypothesis only**: the comparison is against the
  same artifact's reference path, so it **cannot** be attributed to the artifact's n-gram rows
  differing from the official BF16 rows, and any such linkage is explicitly not claimed. Aggregate
  statistics cannot prove element-wise agreement and are not the port verdict - element-wise
  localization needs the raw capture, which this day's run lost.
* **Native capture defect found**: native modules `m_49` and `m_50` (reference layers 46 and 47)
  are **bitwise identical**, so at least one of them is a duplicated tensor; the native side of
  layers 46/47 is therefore unreliable in this capture. `fn_divergence_attrib.py` now **FLAGS**
  duplicate native tensors (pair, shape, "verify capture reuse") and continues the analysis rather
  than refusing outright - identical tensors can be legitimate (e.g. an exactly-zero residual),
  so the flag is a review prompt, never a special case that hides a bug.
* **BF16 arm**: ended at its fixed deadline (`SIGINT 11:26:05Z`, `SIGTERM 11:27:05Z`, gone
  `11:28:05Z`) at **20/48** layers, so **no BF16 logits exist and no KL number was produced** -
  reported as absent, not estimated.
* **Fixes landed after the loss, all CPU-validated**: module-scope `save_file`/`load_file`
  import (root cause of the loss), per-layer disk checkpoints for future runs, the duplicate-tensor
  guard, and a throwaway contract smoke test that proves a completed capture can no longer be
  lost at the write step (48-layer round trip bitwise identical; checkpoint write/read; a view row
  serialized at ~1 MB - note the 3.1 GB case was `torch.save` on a view, while `save_file` handles
  a view correctly, so the clones are a guard, not the cause of that bloat).
* **Head-level numerical verdict is unchanged and stays**: argmax PASS, top-5 sets FAIL,
  cosine 0.98681, rfn 0.16250 - the port is **NOT parity-validated**. No further full reference
  pass will be started without an explicit budget/mechanism authority.

## Open blockers (honest)
* **First divergent module not yet named** — needs the reference raw per-layer capture
  (`runs/main-20261014T032613Z/parity20/raw_layers.safetensors`), batch 1, exact 20 ids, full 48
  layers; the comparator (`tools/fn_layer_parity.py`) maps by the *enumerated* module identity
  (decoder 0 -> native 2, decoders 1..47 -> 4..50; there is no final-norm module) and reports
  byte-level RFN + max|Δ|. A targeted FP16-compute CPU reference for the named module follows
  (the needed inputs are the saved per-layer outputs, so no extra streaming pass).
* **Artifact/reference-weight KL not produced** — needs the BF16 arm's raw logits; that arm is
  throughput-bound (~7 MB/s observed, ~20 min/layer; the full ~237 GB stream projects past its
  fixed deadline). It is left to end at its deadline; no restart, no invented number. The KL is
  a reference-weight result (full-model exact reference), explicitly **not** a served-port KL,
  and the n-gram deviation of the artifact from the official BF16 rows (rfn 0.9927-0.9932,
  std ratio ≈ 2^7) is reported as a diagnostic of the artifact, not as a port claim.
* **GPU blocked** — a foreign `vllm serve` job (pid 1845042) holds GPU1 since 08:17:36Z; no GPU
  work of ours runs, and any port step needing a GPU waits for that job to clear plus an explicit
  allocation. All correlation/parity work here is CPU-only.
* **Retry policy** (recorded in `runs/main-20261014T032613Z/RETRY_POLICY.txt`): standing rule is
  at most 2 attempts, same mechanism, no unbounded retries; exactly one operator/parent-approved
  exception exists (the coalescing fix above), granted on documented root cause + byte-identical
  mechanism change + preserved previous log.

## Reproduce
```bash
# reference raw per-layer capture (batch 1, exact ids, full 48 layers, checkpoints per layer)
refenv/bin/python3 tools/fn_kl_eval.py parity --arm exl3 \
  --prompt-ids-file panels/parity20_ids.json --out runs/<run>/parity20
# layer comparison (refuses to guess alignment)
refenv/bin/python3 tools/fn_layer_parity.py \
  --native-map runs/<run>/parity20/native_module_map.json \
  --native runs/<run>/parity20/native_raw_modules.safetensors \
  --reference runs/<run>/parity20/raw_layers.safetensors \
  --native-head runs/engine1-20261014T010425Z/native_first_logits.pt \
  --ref-head runs/<run>/parity20/raw_hidden_logits.safetensors \
  --ref-parity-json runs/<run>/parity20/parity.json --out runs/<run>/parity20/layer_parity.json
# campaign (identity -> coverage -> arms -> KL)
FN_RUN_DIR=runs/main-<stamp> bash tools/fn_run_campaign.sh
```
