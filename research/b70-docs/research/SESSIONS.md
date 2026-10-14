---
title: "Session log"
type: research
status: living
date: 2026-10-13
---

# Session log

One entry per working session, newest first. Each entry says what changed, what
closed, and where the evidence, scripts and data are. Details live in the
linked notes and ledger entries; this file is the index. Rule: AGENTS.md
section 23.

## 2026-10-14 afternoon - int8 prefill: determinism probe, TTFT pricing, launcher opt-in

- **Ledger:** RL-214, RL-215.
- **RL-214 (determinism probe):** every prefill op is bitwise deterministic in isolation:
  - the exl3 int8 and fp16 linears;
  - `torch.matmul`;
  - oneDNN SDPA and FA2;
  - the FLA GDN chunk kernel.

  int8 amplifies a one-ulp input change about 2x relative to fp16. A deterministic int8 path would not
  remove its ~0.001 KL, so that lever is withdrawn.
- **RL-215 (TTFT pricing):** fp16 prefill costs R = 1.63 / 1.54 / 1.34 / 1.28 at 8K / 32K / 131K / 190K
  on the launcher SC5 profile. 190K completes and canaries pass. By the predeclared rule it is opt-in.
- **Launcher:** b70-launcher commit `b5e7332`, local and **not released**. It adds `--prefill int8|fp16`
  (CLI, API, UI) and fixes extra_env entries being silently overridden by model.yaml env. 313 tests pass.
  Live-verified on GPU1.
- **Release:** shipped as **launcher 0.6.10** after operator approval.
  - Tarball sha256 `c72b199e…fcb6b4c1`.
  - Commits: launcher `7686fec` (pushed to GitHub and Radxa), xecores `0ac5446`.
  - Published on GitHub release v0.6.10 and on xecores.com.
- **Release checks:**
  - The checksum is identical in live version.json, the live tarball and the GitHub asset.
  - A fresh `get.sh` install in a sandbox reports 0.6.10.
  - The local 0.6.9 install updated in-app to 0.6.10.
  - Real SC5 launches through the updated launcher, ready in 6m21s each:
    - `--prefill fp16`: PID1 env shows `EXL3_INT8_PREFILL=0`;
    - default: PID1 env shows `EXL3_INT8_PREFILL=1`, and the command is unchanged;
    - both smoke tests answered 56.
  - Evidence: `results/2026/10/sc5-int8-prefill-ttft-20261014/release_0.6.10/`.
- **Evidence:** `results/2026/10/int8-determinism-20261014/`, `results/2026/10/sc5-int8-prefill-ttft-20261014/`.
- **Scripts:** `scripts/tmp/int8_determinism_20261014/`.
- **Work data:** removed at close: `/data/nvme/work/int8-determinism-20261014/` (extracted image sources) and
  `/data/nvme/work/int8-ttft-20261014/` (corpus ids, regenerable from the wikitext parquet).

## 2026-10-14 - EAQ/TensorFold assessed, as-served int8-prefill quality on exl3xpu

- **Questions:** RQ-30 (reopened for one measurement, no route change).
- **Ledger:** RL-211 to RL-213.
- **Closed (RL-211):**
  - EAQ (MiaAI): unpublished, inconsistent numbers, and it targets TensorFold's load-time q4 dense
    requantization, which exl3xpu does not do. Watch only.
  - TensorFold port to XPU: rejected. It has no SYCL backend, the 27B is qualified on Metal only, and
    SC5 is already at its byte floor.
- **Measured (RL-212, RL-213):**
  - v1 (4 arms) is invalid by its own determinism rule. It found that repeated passes drift after a
    multiple-of-16 token index, and that int8 prefill drifts 30-50x more than fp16.
  - v2 (SC5 chat, 44,473 positions, two passes): int8 prefill moves SC5's served distribution by KL
    0.00099, 70 % of SC5's own weight-quant KL to BF16. fp16 noise floor 0.00004. Verdict: material.
  - Most of the int8 divergence is its own run-to-run non-determinism.
- **Not adopted.** Candidates: `EXL3_INT8_PREFILL=0` for the SC5 tier (TTFT about +23 % per the engine
  README; needs pricing), or a deterministic int8 path first.
- **Ranked ideas** (int8 determinism, SC5 to plain attention-K/V graft, in-graph MTP draft, rotated int4
  KV for 196K, fp8-KV error measurement): `research/2026/asserved-kl-and-eaq-tensorfold-20261014.md`.
- **Scripts:** `scripts/tmp/asserved_kl_20261014/`.
- **Evidence:** `results/2026/10/asserved-kl-int8-prefill-20261014/`.
- **Raw data:** `/data/nvme/archive/asserved-kl-20261014/`. The `work/` folder was removed.
- **Cleanup:** no containers left, caps 180/230 W as found.

## 2026-10-13 night - launcher 0.6.1, SC5 V6 check, X post

- Launcher 0.6.1 released (GitHub v0.6.1, xecores.com, tarball `daedcbbe...fe81a87f2`): Install update button (SHA256-checked, restart with same args), external links open in the browser (the old Download update link did nothing inside the WebKitGTK window), SC5 pinned to `exl3xpu-mixedk:20261013` (RL-204). Local install updated to 0.6.1; the open 0.5.1 window needs one restart.
- V6 check (RL-205): `SC_5.00bpw_H6_V6` has byte-identical text weights (2,093 tensors); only the vision tower is quantized; it fails to load on our image (vision keys). Not adopted; files at `/data/sata1/models/qwen38-27b-exl3-SC_5.00bpw_H6_V6` (18.2 GiB).
- X post draft (thread + single): `social-assets/x-post-exl3-mixedk-sc5-20261013.md`.
- Launcher 0.6.2 (tarball `f897428c...aa6a3514`): custom EXL3 folders with 2/3/5/7-bit tensors use the mixed-K image. The installed 0.6.1 updated itself to 0.6.2 from xecores.com through the new Install update API, then restarted on its own port.

## 2026-10-12 to 2026-10-13: EXL3 quality tier, own quantizer closed, launcher 0.6.0, storage reset

**Questions:** RQ-29 (closed earlier), RQ-30 (closed rejected), RQ-14 (claim audit).
**Ledger:** RL-175 to RL-203.

### Adopted or kept

1. **SC_5.00bpw_H6 is the EXL3 quality tier.** KL vs BF16 0.42x (prose) and
   0.16x (chat) of plain 4.00bpw (RL-179); serves correctly on
   `exl3xpu-mixedk:20261012c` (RL-182, RL-185); 0.89x plain speed under recipe
   sampling, 56.6 vs 63.6 tok/s (RL-186). Plain EXL3 4.00bpw stays the speed route.
2. **EXL3 containers copy weights without the copy engine**
   (`UR_L0_USE_COPY_ENGINE=0`): 2 of 4 loads hung on a BCS page fault before,
   4 of 4 clean after, decode unchanged (RL-198).
3. **b70-launcher 0.6.0** released (package split, one card per container,
   GPU-client detection, EXL3 parser and name fixes); four recipes verified end to
   end on GPU 1 (RL-199). xecores.com deployed from the desktop and from the Radxa.

### Closed (do not reopen without the re-entry condition)

| Line | Result | Re-entry |
|---|---|---|
| Own EXL3 quantizer (XTQ) | Three failures: RL-191 1.16x/1.41x prefix KL, RL-195 per-tensor gate, RL-197 full model 0.888x wiki / 1.085x chat. Best case is parity with turboderp plain, still below SC5. | A new mechanism with a written prediction to beat SC5 quality per bit (RL-203). |
| SC4 serving | Speed gate failed: 1.005x at p512, 0.915x at p8192 (RL-193). | none |
| SC5 speed work | Already at the byte floor; all fixes bound at ~0.93x plain (RL-202). Draft graph capture failed (RL-187, RL-188). | A kernel change predicting >= 10 % end to end. |
| MTP5 instead of MTP3 | +20 % on greedy synthetic, 0 % on thinking prose (RL-200). | none |
| "100 tok/s on 2x B70" (0xSero) | 140.8 = C4 aggregate on AutoRound INT4 TP2; trellis-serve 91.2 prose not reproduced by protocol (RL-200). | Their exact stack (kernel 7.1.8, runtime 26.31) on our host. |
| "100 tok/s Flash-Next" | No such claim; best is 76.3 tok/s C4 aggregate with approximate output (RL-201). | >= 92 GB free NVMe and an exact-output arm. |

### Findings worth remembering

- KL divergence numbers (all vs BF16, native heads): plain 4.0 0.01391/0.00876,
  SC4 0.01602/0.00653, SC5 0.00582/0.00141, XTQ Q2 0.01235/0.00950 (wiki/chat).
- Our calibration rows were not the native EXL3 rows: BOS 248044 missing,
  200,640 of 512,000 positions differ (rank0, not pursued).
- Thinking prose at T0.7 decodes slower than greedy p512/g128 on the same route
  (54.9 vs 78.7 tok/s, MTP3).
- Every EXL3 load spends ~180 s compiling kernels inside a `--rm` container.

### Where things are

- Notes: `research/2026/findings-divergence-quality-speed-20261013.md`,
  `research/2026/stop-decision-low-impact-work-20261013.md`,
  `research/2026/sybil-100tps-claim-audit-20261013.md`,
  `research/2026/flashnext-100tps-claim-audit-20261013.md`,
  `research/2026/xtq-vs-turboderp-gap-20261013.md`,
  `research/2026/sc5-speed-gap-20261013.md`.
- Results: `results/2026/10/rq30-*`, `results/2026/10/sybil-protocol-ab-20261013/`,
  `results/2026/10/launcher-06-verification-20261013/`.
- Scripts: `scripts/INDEX.md` (search by RL ID); this session's clients are
  `scripts/tmp/protocol_ab_20261013.py`,
  `scripts/tmp/exl3xpu_mixedk_20261012/tests/client_cells.py`,
  `scripts/tmp/gpu_guard_20261011.sh`, quantizer scripts `scripts/tmp/xtq_*`.
- Storage reset: 413 GiB deleted (170 closed-line items), disks remounted at
  `/data/nvme`, `/data/sata1`, `/data/sata2`, `/mnt` aliases removed, 5.3 GiB of
  bulky files moved out of B70-DOCS, old home folders archived. Record:
  `results/2026/10/storage-cleanup-20261013/`; layout:
  `docs/operations/storage-layout.md`.
- **Evidence lost in the cleanup:** two deleted NVMe folders were targets of
  B70-DOCS symlinks: the RL-203 prefix-13 attempt (reconstructed record in
  `results/2026/10/rq30-xtq-c-exl3-prefix13-20261013-UNSEALED/RECONSTRUCTED-RECORD.md`)
  and the raw run logs of `results/2026/08/qwen38-flash-next-dual-b70-20260831T130711Z/`
  (`LOST-LINKS-20261013.md`). The inventory tool now treats repo symlinks as LIVE.

### Process lessons (now in AGENTS.md sections 4, 21-23)

1. Price the best case before GPU time; most of this session's GPU hours went to
   a quantizer whose best case could not beat a public artifact.
2. An operator stop must kill retry wrappers and lock holders, not only the
   container (the GPU 0 study restarted three times).
3. Never `git checkout` a registry file to undo an edit (uncommitted
   research-program changes were lost and rebuilt from the Radxa copy).
