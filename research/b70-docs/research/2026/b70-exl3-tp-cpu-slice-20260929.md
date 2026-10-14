---
title: "T1 executed — EXL3 4bpw trellis slice-exactness on CPU (representation-only TP gate)"
type: research
date: 2026-09-29
status: provisional
author: "zcode-subagent (correctness-provider substitute)"
scope: "CPU-only numerical test. No GPU, no serving, no build, no pulls, no artifact modification, no ledger/status edits."
publication_gate: PROVISIONAL_NOT_FOR_PUBLIC_HEADLINE
---

# T1 — EXL3 packed-tensor slice exactness (CPU, executed)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.**

This is the executed T1 cell from
`research/2026/b70-exl3-tp-feasibility-upstream-20260929.md` §7. It answers
exactly one question: **does the EXL3 packed representation permit TP-style
N/K slicing at 128-Hadamard-block boundaries without requantization?**

## Verdict (representation-only)

**PASS — 38/38 cells.** On two real 4bpw tensor quads from the incumbent
artifact, in both orientations (k<n and k>n):

- **Decoded-submatrix identity is bitwise-exact** under N-slice and K-slice
  at 128-block boundaries — both for `reconstruct_inner` (Hadamard-domain
  fp16) and `weight_orig` (original-basis fp32).
- **Column (N) split forward is bitwise-exact**: per-shard forwards and
  their concatenation are `torch.equal` to the full forward, at half (3072),
  quarter (1536), and minimal 256-column (2-block) shard widths.
- **Row (K) split forward is exact up to fp32 reassociation**: summed
  partials differ from the full forward by max_abs 2.9e-6 (in_proj_z) /
  9.5e-6 (out_proj) — pure accumulation-order noise, inside the
  pre-declared allclose gate (atol 1e-3, rtol 1e-4). This is exactly the
  TP-allreduce contract.
- **The 128-boundary constraint is real, not cosmetic**: a trellis-legal
  64-wide shard (4 tiles) produces materially wrong values (max_abs 1.4–10.6
  vs truth) — and would additionally be rejected at load by the plugin's own
  `% 128` assert (`vllm_plugin.py:163-164`). A wrong-half sensitivity check
  confirms the harness detects mismatches (max_abs 6.2 / 32.5).

**Scope of the verdict:** the EXL3 format is an *alignment* constraint
(slices must fall on 128-block multiples — already enforced by the plugin),
not a *coupling* constraint. TP2 sharding of this checkpoint needs no
requantization and no format change; the blocker remains loader + transport
(the `vllm_plugin.py:157-162` `NotImplementedError` and the xccl/copy-collective
questions), exactly as the feasibility audit concluded. This test says
nothing about kernel efficiency at half-N, collective latency, graph
capture, or any speed number.

## Identities

| Item | Value |
|---|---|
| Artifact | `/mnt/<nvme-disk>/b70-exl3-qwen38-4bpw-20260926`, shard `model-00001-of-00002.safetensors` |
| Reference | `/tmp/b70-k3-exl3-source-20260929/exl3xpu/ref.py`, sha256 `3f614b7c…d1780`; clone clean @ `15ded2f3add148c4db3c900cba7de878238f53bc` |
| Runtime | torch 2.13.0+cpu, numpy 2.3.5, python 3.14.4, `xpu_available=False` (asserted CPU-only) |
| Codebook | `CB_MUL1` (artifact `quantization_config.json: codebook=mul1`; mul1 scalar present, val −2082680531) |
| K | 4 (trellis last dim 64 = 16·K → 4 bpw) |
| Seed / batch | 2928 (`0xB70`), x = randn[8, k] fp32 |

## Tensors tested

| Tensor | trellis | k | n | 128-blocks |
|---|---|---|---|---|
| `layers.0.linear_attn.in_proj_z` | [320,384,64] i16 | 5120 | 6144 | 40×48 |
| `layers.0.linear_attn.out_proj` | [384,320,64] i16 | 6144 | 5120 | 48×40 |

Payload sha256s recorded in `results.json` (`identity.sha256`).

## Key measured numbers

| Cell | in_proj_z | out_proj |
|---|---|---|
| col half/quarter/256 decode + forward | bitwise (0.0) | bitwise (0.0) |
| row halves decode + weight_orig | bitwise | bitwise |
| row summed forward max_abs | 2.86e-6 | 9.54e-6 |
| neg col-64 mismatch | 5.64 | 5.53 |
| neg row-64 mismatch vs true partial | 1.44 | 10.60 |
| sensitivity (wrong half) | 6.20 | 32.47 |

`max_rel` on the row-sum cell (7.8e-3 / 1.1e-1) is on near-zero elements
under the allclose `atol + rtol·|b|` form; the abs figures are the
meaningful reassociation bound.

## Notes

- `ref.py::had_rows` does not raise on sub-128 widths — `reshape(-1,128)`
  silently regroups across batch rows and returns wrong numbers. The
  plugin's `% 128` assert is therefore a *necessary* guard, not redundant.
- For K-shards finer than one 128-block, a rank cannot even form its input
  Hadamard share: `xh` columns within a block mix all 128 inputs. TP
  partitioning must slice both dims at 128 multiples — consistent with the
  plugin assert and the audit's §4 math.
- Extraction hygiene: artifact opened read-only; no writes outside the
  results dir; nothing started, pulled, or built.

## Evidence

- Script: `results/2026/09/b70-exl3-tp-cpu-slice-20260929/cpu_slice_test.py`
- Log: `results/2026/09/b70-exl3-tp-cpu-slice-20260929/run.log`
- Machine-readable cells: `results/2026/09/b70-exl3-tp-cpu-slice-20260929/results.json`

## What remains open (unchanged by T1)

Loader/plugin sharding (the `NotImplementedError` guard), per-rank
`shard_of_nb` rebuild, flag-slot copy-only collectives and their
reader-side visibility + capture safety (B11, 60 µs kill threshold), kernel
efficiency at half-N, graphs-off penalty, and the ADR-0008 matched-control
campaign — per the feasibility audit §8.
