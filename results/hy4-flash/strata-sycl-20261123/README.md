# Hy4-Flash, Strata SYCL, two Arc Pro B70s: round-2 opt evidence (2026-11-23)

Sanitized summary behind the [Hy4-Flash Strata SYCL recipe](../../../docs/hy4-flash/HY4-FLASH-STRATA-SYCL.md).
`strata-sycl-opt-r2-20261123.json` is the machine-readable record of the verified round-2 arms
(`opt-17-17`, `opt-17-25` asymmetric peer placement, `+ --draft dfly2`, step-timing and syncall diagnostics).

## Provenance

Summarized on 2026-11-23 from a private lab campaign directory (not published). Raw per-arm JSON logs,
engine logs, live VRAM series and byte-diffed `dmesg` snapshots exist in the lab record only; local paths,
mount points and PCI addresses were removed from this public summary. Model artifacts are named by file
name and SHA-256, not by where they were mounted.

## Common shape

- Hardware: two Intel Arc Pro B70 (BMG-G31, 32 GiB each), both at 230 W caps; primary = compute card,
  peer = display card (`--gpus 1,0` in PCI-id order). Driver `xe`, Level Zero 20.2.0, UR 1.14.37020.
- Engine: `strata-hy4-sycl` 0.1.66, `Niko1221/Strata` v0.2.20 `a34979fc94f3d126b9bb8210f4ad311345688b52`
  plus [`strata-hy4-b70-opt.patch`](../../../patches/hy4-flash-strata-sycl/APPLY.md) (binary sha256
  `6d85f1adaf7174b8e4bfcb0bb8c98fdf5c95723d8df4cde4cbaa1035e8c05e1c`).
- Model: `Hy4-Flash-SQ-64.strata` + `Hy4-Flash-Engram-Q8.strata` from
  `strata-engine/Hy4-Flash-Strata` @ `995bac22` (SHA-256s in the JSON).
- Launch shape: `--max-context 32768 --threads 8 --gpus 1,0 --vram-experts-gb 17
  --ram-experts-gb 36 --kv-host-above 131072` (+ `--peer-vram-experts-gb 25`, `--draft dfly2` per arm).
- Sampling: quality gate and parity at `seed=4242 temperature=0`; depth ladder at `temperature=0.6 top_p=0.95`.
- Timing: engine protocol `prompt_ms`/`decode_ms` fields (not wall/SSE estimates); cold launches; C1.
- Controls: `gpu.lock` flock, 52 GiB host-memory scope, READY watchdog, per-arm `dmesg` snapshots diffed
  bytewise against the previous snapshot.

## What is in the JSON

| Block | Contents |
|---|---|
| `decode_tok_s`, `prefill_tok_s`, `ttft_28k_s` | Full 0K / 8K / 28K ladder: llama.cpp-hy4pr baseline, Strata v0.2.20 stock, `opt-17-17`, `opt-17-25`, `opt-17-25-dfly2` |
| `arms` | Per-arm config, gate result, key number, and new GPU `dmesg` lines (`+0` on every opt arm) |
| `phase_timing_step_timing_8k_64tok` | `HY4S_STEP_TIMING=1` phase decomposition: `peer_pf_ms` 20,607.5 → 22.7 ms; `cpu_exp_ms` 491.0 → 125.6 ms |
| `quality_gate` | 5/5 deterministic gate on all three ladder arms (incl. under dfly2) |
| `determinism` | Serial-greedy nondeterminism receipt (SYNCALL does not fix it; dfly2 parity unverifiable) |
| `vram_floors_mib`, `teardown_receipt` | Peer floor 4,618 MiB at 25 GiB; 5/5 clean exits; `dmesg` delta 2 benign lines, +0 GPU faults |
| `dead_levers` | `HY4S_XMX_I8=1`, `STRATA_CPUEXP_ACT=q8` — closed arms (DNR-043, DNR-044) |
| `claim_boundaries`, `pending` | Explicit untested scope |
| `strata-sycl-maxctx-20261123.json` | Localized max-context ladder (`32K / 128K / 256K / 512K / 1M` load+smoke), `64K` & `124K` all-VRAM speed (`726.9` / `702.7` pf, `18.8` / `17.0` serial, `38.3` / `39.5` dfly2), `125,999`-tok exact needle PASS, and `64K` host-KV cliff (`98.2 tok/s`) |

## Not in this folder

Raw engine logs, per-request token streams, live VRAM time series, `dmesg` snapshots and the harness
scripts themselves (private lab paths throughout; sanitized on request for a specific review). Round-1
(`R1`) numbers — including the wedged grouped-prefill arms the `pf_cap_p` fix resolved — are described in
the recipe's optimization section and are not re-published here: the R2 patched build supersedes them.

## Regeneration rule

Any changed engine commit, patch stack, model pack, driver, or launch shape starts a **new** results
generation under a new dated directory. Do not edit this summary in place.
