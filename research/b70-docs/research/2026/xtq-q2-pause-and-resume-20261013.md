---
title: "XTQ full-model quant (Q2): paused after layer 51, resume options (2026-10-13)"
type: research
date: 2026-10-13
status: provisional
---

# XTQ Q2 pause and resume — 2026-10-13

Operator order 2026-10-13 ~08:50 local: stop all GPU work, save the exact state,
and give a resume path on a rented GPU. Nothing here is a quality result.

## 1. State at the stop

| Item | Value |
|---|---|
| Job | our EXL3-mul1 K=4 trellis quantizer, per-channel scales, LDLQ, turboderp default calibration (250 × 2048 tokens) |
| Stop | graceful `PAUSE_AFTER_LAYER`, exit 0 at 2026-10-13T06:51:15Z, after layer 51 committed |
| Done / remaining | layers 0–51 done (325 tensors), layers 52–63 remain (12 layers) |
| Root (read-only now) | `/mnt/<nvme-disk>/b70-run4-20261011/harness-fmt/xtq-full-chscale-cal512k` |
| Checkpoint | `checkpoint.pt`, 5,242,897,893 B, sha256 `058f730f74b09ba14cb6f5088ba17b68bb86bce563e511296568b46ad5a2c4d9` (CPU bf16 hidden states [250, 2048, 5120] after L51, RNG states, identity) |
| Exact state files | `RESUME_STATE.json`, `RESUME.md`, `MIGRATION_CONTAINER_INSPECT.json` in the root |
| Supervisor | `b70-xtq-q2-supervisor-v3` stopped and runtime-masked; container `xtq-full-chscale-Q` kept (exited 0) |
| GPUs | both idle after 09:30 (no inference/quant/SYCL workers); caps as found: GPU0 180 W, GPU1 230 W |
| Quality so far | per-tensor Hessian-weighted error, in-sample: median 1.027× turboderp, 0.93–1.06× (`results/2026/10/rq30-xtq-q2-cal512k-progress-20261013-UNSEALED/`). No full-model KL yet. |

The beat-turboderp variant study on GPU0 was also stopped: its agent restarted
GPU sweeps twice after the stop order, so the agent was cancelled and its
container stopped (~10:25 local). Partial results: RL-195,
`results/2026/10/rq30-xtq-beat-turboderp-partial-20261013-UNSEALED/` —
turboderp's method in our code reaches ~0.99× his held-out error; the best
variant ~0.97×; nothing meets the 10 % win rule yet.

## 2. Resume options

| Option | What runs | Time for 12 layers | Cost | Proof status |
|---|---|---|---|---|
| **B70 (same host)** | unchanged ESIMD path | ~2.1–2.4 h (measured 10.3–12 min/layer) | 0 | proven path; command in root `RESUME.md` |
| **Vast.ai RTX 3090, CUDA ≥ 13.0** | rental kit, torch Viterbi on CUDA | [INFERENCE] ~6–9 h | [INFERENCE] ~$1.5–2.5 at $0.189–0.268/h | CUDA path untested; the kit gate decides first |
| **Vast.ai A100 SXM4** | same kit | [INFERENCE] ~3–5 h | [INFERENCE] ~$2–3 at $0.523/h | same |
| **Colab Pro A100 (High-RAM)** | same kit in a notebook | [INFERENCE] ~3–5 h | ~$0.54/h (Colab units, March 2026 secondary source) | free T4 fails the RAM gate; Colab driver vs the cu130 wheel unverified |

Live Vast.ai on-demand offers, 2026-10-13 ~09:05 local, filter ≥ 48 GiB RAM,
≥ 120 GB disk, ≥ 500 Mbit/s down, reliability ≥ 0.98, CUDA ≥ 13.0 (the
kit's torch 2.13.0+cu130 wheel needs it): RTX 3090 $0.189/h (Croatia),
$0.268/h (Virginia); RTX 4090 $0.349/h; A100 SXM4 $0.523/h. The cheapest 3090
($0.122/h) reports CUDA 12.8 and fails the kit's wheel pin. Source:
`console.vast.ai/api/v0/bundles` query; prices move hourly.

Intel GPU rental (Intel Tiber AI Cloud, Data Center GPU Max) would run the
ESIMD path, but the kernel is untested on Xe-HPC and the pricing page did not
resolve on 2026-10-13.

## 3. Rental kit

`/mnt/<nvme-disk>/b70-run4-20261011/q2-rental-kit-20261013/`
(6,027,614,035 B, 168 files in `SHA256SUMS`; code copy in
`scripts/tmp/xtq_q2_rental_kit_20261013/`). Step-by-step guide:
`RESUME-ON-RENTAL.md` in the kit.

- `code/`: exact module copies (quantizer `xtq_quant_20261012b.py` sha
  `fdeafd7a…` unchanged) plus `xtq_full_quant_rental.py` (`--device cuda`,
  torch backend, `--torch-batch 32`, deterministic CUDA settings, keeps the
  checkpoint at DONE). `--device xpu` keeps the B70 arithmetic.
- `data/`: checkpoint copy, calibration, identities, states farm, ref.py,
  corpus, model manifests, known-answer gate inputs (L0 `linear_attn.out_proj`).
- `bootstrap.sh`: pinned torch 2.13.0+cu130 / transformers 5.16.1 /
  safetensors 0.8.0 / numpy 2.2.6, downloads `Qwen/Qwen3.8-27B@1d4bf0f2` and
  `turboderp/Qwen3.8-27B-exl3@113cf7ab`, verifies manifests, gates RAM ≥ 48 GiB,
  VRAM ≥ 22 GiB, disk ≥ 110 GiB.
- `resume.sh`: runs the CUDA known-answer gate first (Hessian-weighted error
  ratio ≤ 1.005 vs the B70 ESIMD result, prints s per 1M weights and the
  12-layer projection); the driver starts only on PASS.
- `pack_results.sh`: returns L52–L63 tensors, logs, markers and checkpoint.

Verified on CPU only: compile, shell syntax, 2-layer CPU smoke, gate dry run,
toy return pack. No CUDA run has happened. Layers 52–63 made on CUDA would come
from the torch Viterbi, not the ESIMD kernel; on XPU the two are bit-exact
(RL-189), on CUDA tie-breaks may differ. Declare mixed provenance if used.

## 4. After the rental

1. Verify the returned archive (`sha256sum -c`), extract into a new staging
   directory, never into the paused root.
2. Merge L52–L63 into a copy of the root, run the EXL3 writer, full-64 KL vs
   plain 4.0, then canaries and the recipe A/B on a B70.
3. Upload/transfer needs either Vast SSH (rsync from the desktop, no token) or
   a Hugging Face **write** token. The only valid HF token found on the Radxa
   (2026-10-13) is fine-grained **read-only**; it cannot create or upload repos.
