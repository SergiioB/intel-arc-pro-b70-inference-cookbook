---
title: "External Flash-Next routes assessed — Strata v0.1.40.2 and the sybil B70 tier build (2026-10-07)"
type: research
date: 2026-10-07
status: provisional
---

# External Flash-Next routes assessed — Strata and the sybil B70 tier build

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Static assessment of two public
repositories against our measured record. External evidence class **E0** (their
numbers, their hosts). No GPU run, no download, no model load. Ledger: RL-121.

Sources (fetched 2026-10-07):

- <https://github.com/Niko1221/Strata/releases/tag/v0.1.40.2> (released 2026-10-07) + repo README
- <https://github.com/sybil-solutions/qwen38-flash-next-b70-offload> (measured 2026-10-06)

## 1. What each project is

**Strata** (16.5k stars, MIT, C++ engine) runs Qwen3.8-Flash-Next (125B-A6B MoE,
512 routed experts top-10) on consumer hardware by keeping hot experts in VRAM
and streaming the rest from host RAM/SSD. v0.1.40.2 adds **official Intel Arc
support** (SYCL port): tested on an Arc Pro B70 32 GB (xe driver) and an A750.

Their B70 row (IQ3_S, E0 external, single-run class, "story/code" content):
4K prompt 980–1,000 tok/s, decode 31 / 41 tok/s. Their NVIDIA/AMD tables show
Q2_0–IQ3_S decode 44–94 tok/s on 12–16 GB consumer cards.

**sybil-solutions tier build** (15 stars, research grade) serves
Qwen3.8-Flash-Next EXL3 3.05 bpw on **one** Arc Pro B70 with **32 GB host RAM**
and an NVMe expert store: 8,000 hot experts in VRAM, 7 GB RAM tier, rest from
NVMe, staged prefill (NVMe → THP buffer → copy engine → VRAM, 4 layers ahead),
async decode step, fused hyper-connection kernels, SYCL sparse attention ported
from Strata. Measured 2026-10-06 on AMD EPYC + 4× NVMe RAID0 (26 GB/s):

| cell | decode tok/s | prefill tok/s |
|---|---:|---:|
| 8K C1 (default, exact path) | 24.0 | 1,056 |
| 8K C2 / C4 | 28.7 / 26.4 | 1,092 / 1,233 |
| 32K C1 / C2 | 23.9 / 25.4 | 1,106 / 1,260 |
| 8K C1–C4, **no-write-back (approximate output)** | 34.5 / 54.8 / 76.3 | ~1,020–1,123 |

## 2. Flash-Next: both beat our placement

Our record (RL-070, two B70s, llama.cpp SYCL `qwen4exp` dual-split AP-Q4_K_XL,
PLE + 12 expert blocks on CPU): **21.6 tok/s median decode** (p512/g128, engine
tg), cold input **185–195 tok/s**. Verdict recorded: "capacity route, not a
speed route as placed".

That verdict binds the **llama.cpp dual-split placement**, not the model. Against
it, both external routes are ahead on one card:

| | ours (2 cards) | Strata B70 (1 card, E0) | sybil tier (1 card, E0) |
|---|---:|---:|---:|
| decode | 21.6 | 31–41 (story/code) | 24.0 exact / 34.5 approx |
| prefill | 185–195 cold input | ~1,000 (4K) | 1,056–1,260 (8K–32K) |

Metric classes differ (our engine tg vs their unknown engine/client split;
"story/code" vs our prompt classes) — matched cells required before any claim.

## 3. Known hazards in the external builds

1. **sybil default path has a documented write-back race**: 50–52 mismatched
   output rows and 8–10 bad RAM pages in a stress simulation; one runaway
   generation observed in the exact config. Their exact-output fix (victim ring,
   bit-exact 72/72 standalone) is **not yet in the server**. The fast
   no-write-back arm masks ~7% of expert picks = approximate output.
2. **Strata's xe-driver landmine**: large GPU allocations came back with
   **aliased pages** (a write at one offset visible 1 GiB further) → cached
   experts held other experts' bytes = wrong answers. Fix: start-time
   write/read self-check on every big allocation, re-allocate until clean.
3. Strata's Intel support shipped **today**; their B70 numbers are
   minimal-scope (10-prompt MTP check 10/10 twice). Their box classes (EPYC +
   26 GB/s RAID0; consumer NVIDIA/AMD) are not this host.

## 4. What transfers to the dense Qwen3.8-27B champion

**Transfers (cheap, high value):**

1. **xe large-allocation alias self-check** in our isolated-op and serving
   harnesses (item 3.2 above). Same card family, same driver. Prevents silent
   wrong bytes in any future kernel work.
2. **MTP draft-token visibility audit.** Strata's MTP acceptance was 4% until a
   wait read a draft token before the GPU wrote it (→ ~60%). Our yield is
   healthy (2.93–3.16 of 4) but the per-position survival profile is bimodal
   and unexplained; a 1-hour static read of the plugin's draft handoff is cheap.
3. **Pinned shared prefix (`pin=N`)** for many-questions-one-document work:
   first answer 0.35–0.55 s instead of 48–149 s. Maps directly to our Pi/RAG
   workload matrix (RQ-10) and our measured 5.4× prefix-cache win.

**Does not transfer (checked, with reasons):**

1. **Strata's 2–4 token verify-window weight sharing** ("read each weight block
   once, apply to interleaved activations"): our M=4 verify pass already reads
   each block once, and our MTP draft chain is **sequentially dependent**
   (step i+1 conditions on step i), so its k weight reads cannot be merged —
   consistent with our recorded "4×-read bytes floor, physically unavoidable".
   Only relevant if drafting becomes parallel-window (not the current route).
2. **sybil's sparse attention and hyper-connection kernels**: flash-next
   architecture only (qwen4exp hyper-connections; per-query top-2048 sparse
   attention is an output-changing approximation). No dispatch on `qwen3_5`.

## 5. Feasibility on this host

- **Storage (live 2026-10-07, `df`):** root 32 GB free, T-FORCE (SATA) **281 GB
  free**, NVMe SN770 **74 GB free**. Strata needs ~80 GB (model ~70 GB) → must
  live on T-FORCE (2× rule: 160 ≤ 281 ✓; the model page-cache trick may lose
  some value on SATA). sybil needs the EXL3 3.05 bpw artifact (~78 GB) plus the
  packed expert store (~46 GB) **on the NVMe it streams from**: 2× rule needs
  ≥92 GB free there, SN770 has 74 GB → **the tier fast path is storage-class
  blocked** until another NVMe or ≥92 GB free appears. Their prefill also
  streams at 26 GB/s (4× NVMe RAID0) vs our single-drive class (≈3–5 GB/s).
- **Quality bar:** our output-preservation rule (§20.5) excludes the
  approximate no-write-back arm; the victim-ring exact path is the only
  admissible tier decode.
- **MiMo (RQ-12):** still **blocked** — the 177.77 GB repo needs ≥355.5 GB free
  under the 2× rule and T-FORCE shows 281 GB (an older "358 GB free" snapshot
  was stale; the earlier "best free 105 GB" was pre-cleanup).

## 6. Verdict and priority

**Better than these?** For Flash-Next, **no** — our current placement is beaten
on both axes by both projects; Strata is the faster adoption (official B70
support, MIT, active, one-click), sybil is the more ambitious design (one card,
32 GB RAM) but research-grade with a corruption race and 5× our storage class.
Note sybil already builds on Strata's kernels: Strata is upstream of this
class. For the dense 27B champion, neither is a product candidate (different
model); only the three transfers above apply.

Run order: RQ-16 (cheap audits) → RQ-02 (champion sweep) → RQ-14 (Strata
screen) → RQ-01/RQ-03 (champion kernel program) → RQ-15 (tier build, gated on
fio pricing and RQ-14). Full definitions: `registry/research-program.json`.
