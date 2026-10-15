---
title: "Stop decision 2026-10-13: low-impact GPU work quit"
type: research
date: 2026-10-13
status: decision
---

# Stop decision 2026-10-13: low-impact GPU work quit

Operator order (2026-10-13 evening): stop any work that holds the GPUs or the PC
without a realistic chance of an important result. Each open line was ranked
by best-case impact and by evidence so far. Everything below is stopped and
documented. Both GPUs were idle at cleanup (guard `clean`, caps 180 W / 230 W
as found, MemAvailable 52 GiB).

## Ranking

| Rank | Line | Best case if it works | Evidence so far | Decision |
|---|---|---|---|---|
| 1 | Own EXL3 quantizer (XTQ), C_exl3 prefix-13 screen (rank1) and native-token rows (rank2) | Matches or slightly beats plain turboderp 4.00bpw on KL. Same speed. Still far worse than turboderp's own SC5 (KL 0.42x / 0.16x of plain, RL-179). Needs ~2.5 GPU-h per prefix screen, then a full 64-layer rerun and new held-out data before any claim. | Three failed attempts: RL-191 (1.16x / 1.41x), RL-195 (per-tensor gate failed), RL-197 (0.888x wiki / 1.085x chat). AGENTS.md section 20 allows two attempts per unchanged mechanism. | **Quit.** RQ-30 own-quantizer line closed. |
| 2 | Publish our quant to Hugging Face | Public artifact. | No artifact passes its quality gate. SC5 is turboderp's and already public. | **Quit.** Nothing to publish. |
| 3 | SC5 speed gap | +2-3 % decode (co-merged K/V call); all fixes together reach ~0.92-0.93x plain (RL-202). | No-spec SC5/plain speed ratio is 99.5 % of the checkpoint-size ratio (a relative ratio, not bandwidth saturation; correction RL-216). | **Quit.** Below any useful gate. |
| 4 | MTP5 instead of MTP3 | +20 % on greedy synthetic prompts. | 0 % (-4 %, inside spread) on real thinking prose (RL-200). | **Quit.** No gain on real workloads. |
| 5 | Reproduce a Flash-Next "100 tok/s" | None: no such claim exists. Best published is 76.3 tok/s C4 aggregate with approximate output (RL-201). | Needs ~88 GB download and >=92 GB free NVMe. | **Quit.** Not comparable, not feasible. |
| 6 | GPU 0 per-tensor "beat turboderp" sweep (xtq_beat_g) | Same as rank 1, smaller scope. | Same mechanism as rank 1. | **Stopped** on operator choice; partial `beat_g_report.json` kept. |

## What stays

- Adopted routes are unchanged: plain EXL3 4.00bpw (Qwen3.8-27B), SC5 as the
  quality tier at 0.89x speed, Strata IQ3_XXS for Flash-Next.
- RQ-17 (Flash-Next secondary quality gates) stays ready but nothing is running.
- Launcher 0.6.0 and xecores.com are released; the EXL3 copy-engine default
  (RL-198) is kept because it removed real load hangs.

## Evidence kept

- XTQ: `results/2026/10/rq30-xtq-c-exl3-prefix13-20261013-UNSEALED/` (rank0
  tokenizer comparison: native BOS 248044 vs HF none, 200,640 of 512,000
  positions differ; rank1 failure before the first saved tensor; budget
  refusals). Partial checkpoint moved to
  `/mnt/<nvme-disk>/rq30-xtq-c-exl3-prefix13-20261013-qroot/`
  (sha256 `9f6b4488...d03d298`).
- Full XTQ Q2 artifact (16 GB, failed gate):
  `/mnt/<nvme-disk>/b70-xtq-qwen38-4bpw-20261013`. Can be deleted
  to free NVMe space; not deleted without operator approval.
- Notes: `research/2026/xtq-vs-turboderp-gap-20261013.md`,
  `research/2026/sc5-speed-gap-20261013.md`,
  `research/2026/flashnext-100tps-claim-audit-20261013.md`,
  `research/2026/findings-divergence-quality-speed-20261013.md`.

## Re-entry conditions

1. Own quantizer: only with a new mechanism that is not scale/damping/token
   parity with upstream (that path at best equals turboderp), and a written
   prediction that it beats SC5 quality per bit.
2. SC5 speed: only if a kernel change elsewhere predicts >= 10 % end-to-end.
3. Flash-Next: only with >= 92 GB free NVMe and a published exact-output arm.
