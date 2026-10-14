---
title: "Flash-Next >=70 tok/s claims by 0xSero / sybil-solutions — audit vs our Strata route"
type: research
date: 2026-10-13
status: provisional
---

# Flash-Next >=70 tok/s claims by 0xSero / sybil-solutions (audit)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Read-only audit answering: does 0xSero
(github.com/0xSero, github.com/sybil-solutions, x.com/0xSero) claim >=70 tok/s,
or "100 tok/s", on Qwen3.8-Flash-Next on one or two B70s? External evidence
class **E0** (their numbers, their hosts). No GPU work, no pulls, no downloads,
no ledger registration. Community claims are leads; unverified items marked
[UNVERIFIED].

Model identity (primary source, HF card `Qwen/Qwen3.8-Flash-Next@main`, "Number
of Parameters: 125B with 6B activated, plus 51B n-gram embedding and 4B MTP";
48 MoE layers x 512 routed experts, top-10 + 1 shared): total ~180B tensors, so
the owner's "180B MoE" matches. The owner's "37B active" is **not** in the
primary sources we can find (HF card, Qwen blog, 0xSero recipe kit all say 6B
activated per token) [UNVERIFIED provenance of "37B"].

Sources scanned (fetched 2026-10-13): sybil-solutions/qwen38-flash-next-b70-offload
README + docs/results.md at HEAD `db27d3c5f922` and at `30d1b5b94d15`
(2026-10-06, the release commit; repo archived, folded into
0xSero/qwen38-3090-sglang `13f8f426e6` as `qwen3.8-flash-next/b70-offload/`);
sybil-solutions/qwen38-flash-next-3090-usb-offload (2026-10-12);
0xSero/local-ai-recipe-kit `targets/qwen3.8-flash-next-offload.md` @
`23e6039d4f25` (2026-09-29); 0xSero/local-ai-registry recipe trees;
0xSero/trellis-serve xpu/README @ `a1f72be2`; x.com/0xSero posts
2104865431659892934, 2104866547768373606, 2104894881781133399 (2026-09-29),
2105932359472157154 (2026-10-02), 2107589223666930132 (2026-10-06),
2108834357662118195 (2026-10-10), 2109534883173281919 / 2109600712879731070
(2026-10-12), 2109967060403175670 + replies (2026-10-13). Local: our Strata
route RL-125 / ADR-0011 (results/2026/10/rq14-strata-matched-screen-20261009/),
RL-121 (research/2026/flashnext-external-routes-assessment-20261007.md), live
`df`/`free` and a names-only artifact census of the three mounts.

## 1. Answer to the question first

**There is no published 0xSero/sybil claim of 100 tok/s on Qwen3.8-Flash-Next on
B70, in one or two cards, anywhere we can find (fetched 2026-10-13).** The
"100 tok/s" that circulates is (a) the 2026-10-13 post about the **new 2x B70
machine running GLM-5.3-Flash** ("can run GLM-5.3-Flash and some 100 tok/s" —
different model), (b) a 2026-10-12 post "Deepseek on 2x sparks 100 tok/s"
(DGX Sparks, not B70), and (c) a 2026-10-02 DeepSeek-V4.1-Flash prefill figure.
For Flash-Next on a B70 the highest number anywhere in their material is
**76.3 tok/s — C4 aggregate, approximate output** (their experimental
no-write-back arm). The highest *exact-path* B70 decode they have published is
29.2 tok/s (C1) and their own MTP projection is 32-43 tok/s exact at C1.

## 2. Evidence table — every Flash-Next >=70 tok/s claim, 2026-10-01 onward

### F1 — sybil repo, no-write-back arm: 76.3 tok/s at 4 streams (the headline number)

- Quote: "Experimental | Decode without the VRAM→RAM victim write-back:
  **34.5 / 54.8 / 76.3 tok/s** at 1 / 2 / 4 streams. More expert picks are
  masked; a quality gate is running" — README "At a glance" @
  `30d1b5b94d15` (2026-10-06),
  https://github.com/sybil-solutions/qwen38-flash-next-b70-offload/blob/30d1b5b94d15/README.md;
  full table survives at HEAD `db27d3c5f922` in docs/results.md ("Experimental:
  no-write-back decode", srv26 = srv23 + `NOVICT=1`, measured 2026-10-06):
  8k C1 34.5 / C2 54.8 / **C4 76.3** / 32k C1 34.1 / 32k C2 42.9 tok/s decode;
  prefill 1,018-1,136 tok/s; token latency p50 27-44 ms.
- Quant: EXL3 3.05 bpw, `turboderp/Qwen3.8-Flash-Next-exl3` branch
  `3.05bpw_h5_ng5` @ revision `69e33439ae950f17bcbe95c98f117d80f759ab6d`
  (48 MoE layers x 512 experts, 45.8 GB experts + ~32.6 GB 5-bit n-gram table).
- Engine: SGLang 0.5.20-xpu + exl3xpu plugin (trellis-serve lineage), container
  image `ghcr.io/sybil-solutions/qwen38-flash-next-b70-offload@sha256:d30df691…`
  (built from `2f108fb`); release build measured 2026-10-06, srv20→srv23 change
  ladder (async decode step, fused Triton hyper-connection kernels, SYCL sparse
  attention + selection ported from Strata, 20.9 → 24.0 tok/s C1).
- Cards: **1x Intel Arc Pro B70 32 GB** (PCIe 4.0 x16), AMD EPYC 7443P, 4x
  Samsung 9100 PRO RAID0 (fio 26 GB/s), 8 container CPUs. Never two cards.
- Offload: 8,000-slot VRAM expert LRU (14.9 GB) + 7 GB RAM tier (memfd 2 MiB
  pages, xe SVM) + NVMe store of all 24,576 experts (1.86 MB O_DIRECT records);
  staged NVMe→RAM→copy-engine→VRAM prefill.
- Speculation: **none** (MTP only planned: `experimental/n113-mtp`, "projected
  32-43 tok/s exact at 1 stream. Not run yet").
- Metric: decode tok/s, "each stream gets its own prompt (8,181-8,189 or
  32,928-32,932 tokens); completions run to their natural end"; at C2/C4 the
  number is the **total across streams**. Exact timing boundary (engine decode
  vs client post-first) not stated in the repo [UNVERIFIED].
- Concurrency: **C4 aggregate** for 76.3 (≈19 tok/s/stream). C1 is 34.5.
- Output quality evidence: **approximate by construction** — the no-write-back
  kernel masks expert picks found in neither VRAM nor RAM ("weight 0");
  masked picks rise from 4-12 to ~32-38 per decode step, "~7% of picks at
  1 stream" (more at C4). The exact-output fix (victim ring) is bit-exact
  72/72 standalone but "not in the server yet". Repo's own words: "That is why
  decode is approximate". Fails our §20.5 output-preservation rule.

### F2 — X post 2026-10-06: "50 tok/s c4" on one B70 (sub-threshold, same stack)

- Quote: "Qwen3.8-next-flash / As promised, 1 GPU, 32GB of DDR4 and 50GB of
  NVMe. Getting: - 24 tok/s decode - 1106 tok/s prefill - **50 tok/s c4**
  https://github.com/0xSero/qwen38-flash-next-b70-offload/" —
  https://x.com/0xSero/status/2107589223666930132 (2026-10-06 21:50 UTC, 605
  likes). Matches the N130 `nvme16` row (29.2 C1 / 52.6 C4 total measured
  2026-10-08). C4 aggregate, approximate output (decode KL 0.055 vs prefill
  path), same stack as F1. Note the post's own numbers pair "24 tok/s decode"
  (srv23 exact path) with "50 tok/s c4" (N130 16 GB approximate) — two
  different arms in one bullet list.

### F3 — X post 2026-10-10: "~80 tok/s decode" Mac mini M4 + RTX 3090 (not B70)

- Quote: "Qwen3.8-next-flash on a 16 GB M4 Mac Mini + RTX 3090! … 2,800$
  200K kv cache **80~ tok/s decode** 2 bit" —
  https://x.com/0xSero/status/2108834357662118195 (2026-10-10 08:17 UTC, 341
  likes). Hardware: 3090 + Mac mini over tinygrad "chestnut" USB, Strata
  framework. **Not B70.** [UNVERIFIED] and internally inconsistent: their own
  repo for this exact setup (sybil-solutions/qwen38-flash-next-3090-usb-offload,
  2026-10-12, Strata v0.1.41 + patches, EXL3 3.05 bpw "2 bit" claim vs 3.05 in
  the repo) measures **13.9-21.4 tok/s decode C1** (three prompts, MTP spec 4;
  reproduced twice), and the 2026-10-12 follow-up posts quote "17.5 tok/s
  decode / 422 prefill" and "22 tok/s" (game generation). The "80~ tok/s" has
  no artifact and no metric definition; treat strictly as a lead.

### Below threshold in the window (recorded so the ladder is complete)

| date | source | claim | hardware | class |
|---|---|---|---|---|
| 2026-10-08 | repo docs/results.md N130 `nvme16` | 29.2 C1 / 52.6 C4 total decode; prefill 1,139-1,301 | 1x B70 | aggregate at C4, approximate (decode KL 0.055) |
| 2026-10-08 | repo README (image digest `d30df691`) | 8k C1 29.3 / C2 39.3 / C4 49.2; 32k C1 29.7 / C2 40.1 | 1x B70 | same |
| 2026-10-08 | repo docs/results.md N130 `nvme32` | 25.6 C1 / 32.4 C4 | 1x B70 | approximate (decode KL 0.021) |
| 2026-10-06 | repo docs/results.md srv23 (default, exact path) | 24.0 / 28.7 / 26.4 (8k C1/C2/C4), 23.9 / 25.4 (32k) | 1x B70 | natural-end decode, exact path but known write-back race |
| 2026-09-29 | recipe-kit target (23e6039d4f25) | B70 reference: decode 1 user 32.5 tok/s, 2-4 users ~32.6 aggregate; prefill 1,027 (8k) | 1x B70, image `ghcr.io/0xsero/sglang-exl3-xpu-flashnext@bce2f9ad` (trellis-serve 72fd2d4) | single-request; kit quality band top-1 >= 0.987 / KL <= 0.0013 (their reference runs 0.988-0.993 / 0.0009-0.00124) |

### Pre-window context (before 2026-10-01) — where "100-ish" impressions come from

| date | source | claim | reading |
|---|---|---|---|
| 2026-09-29 09:26 | X 2104865431659892934 | "RTX 3090: 64 tok/s decode c1 … Intel Arc B70: 800/1200 prefill, **35 decode tok/s**, 270k fp8 cache" | the only stated B70 C1 figure in that post is 35 tok/s |
| 2026-09-29 09:31 | X 2104866547768373606 | "**65 decode tok/s** … 1. RTX 3090 / 4090 / 5090 / Intel Arc B70 / AMD" | GPU list contradicts the 35 tok/s B70 row 5 minutes earlier; [UNVERIFIED] which GPU |
| 2026-09-29 11:23 | X 2104894881781133399 | "you'll need a total of 96gb system ram … ends up at **85.5 tok/s at c3**" | links both the NVIDIA and XPU images; matches the recipe-kit **RTX 3090** row ("3 users 84.8 (max)" aggregate, 75 GB free RAM) better than any B70 number; hardware attribution [UNVERIFIED], metric = C3 aggregate |
| current | x.com/0xSero profile | "able to run Qwen Next at 50-70 tok/s" | no artifact, no hardware, no metric [UNVERIFIED] |

### The "100 tok/s on 2x B70" itself is GLM-5.3-Flash, not Flash-Next

- "New machine new me. This one is about 12,000$ and can run **GLM-5.3-Flash
  and some 100 tok/s**" — https://x.com/0xSero/status/2109967060403175670
  (2026-10-13 11:18 UTC); hardware "2x Intel Arc b70" confirmed in reply
  2109976467505594555. Quant reply (12:41, 2109987856680878259): "it fits
  quite well in exl3-3bpw". No model breakdown, metric class, concurrency,
  engine, or artifact: [UNVERIFIED] beyond the post. Model is GLM-5.3-Flash —
  the earlier audit (research/2026/sybil-100tps-claim-audit-20261013.md, C1)
  already recorded this post; nothing in the thread mentions Qwen3.8-Flash-Next
  or a Flash-Next rate. No sybil repo carries a Flash-Next number on two cards
  at all: their B70 Flash-Next work is strictly 1-card, and the 2-card mode
  story for other models (qwen38-b70 `--dp`, Qwen3.6 TP2) is separate.

## 3. Diff vs our Strata route and vs their earlier repo

**Ours (RL-125, ADR-0011 adopted route):** Strata v0.1.40.3 (logs "engine
0.1.41", rev class `d5ea713374`), Qwen3.8-Flash-Next-GSQ-RCO-**IQ3_XXS**
(00001 47,039,860,096 B + 00002 28,800,138,432 B ≈ 75.8 GB), pack/iq3xxs +
mtp-rt (experts.bin 707,788,800 B, draft_vocab.bin 425,196 B), **one B70**
(level_zero:0, arm-a), `--stream-experts --expert-cache auto --spec 4
--spec-min-p 0.5 --kv int8 --prefill 4096 --max-context 10240`, 10,032 experts
mirrored in 16.37 GiB pinned host RAM; measured **48.03 tok/s engine decode**
(p512/g128, greedy, n=5, engine-side `decode N tokens in X ms` wall
accounting), prefill 989 tok/s at 8K, battery 3/3 (703 / Paris / B70-FLASH-OK)
at gen=64. Two-card arm-b was slower (33.44 / 111.86) — one card is our route.

| axis | ours (Strata IQ3_XXS, 1 B70) | sybil tier build (EXL3 3.05bpw, 1 B70) | what changed in their repo 2026-10-06 → 2026-10-08 |
|---|---|---|---|
| engine | Strata C++ SYCL, self-contained | SGLang 0.5.20-xpu + exl3xpu plugin + nvtier.py | N130 16 GB mode: admission-controlled decode tier (evict only RAM copies of VRAM-resident experts or >4-step-old entries; 8 decode-fill queues), vs the stock tier that collapsed (~470/480 picks masked, garbage output) |
| quant | GSQ-RCO IQ3_XXS (75.8 GB) | EXL3 3.05bpw_h5_ng5 (45.8 GB experts + 32.6 GB n-gram 5-bit) | unchanged since RL-121; a lighter 2.05bpw_h4_ng4 asset exists in their registry (tabbyapi configs) but was not measured on B70 |
| expert caching | 10,032 experts pinned in host RAM, `--expert-cache auto` | VRAM LRU 8,000 slots (14.9 GB) + 2 GB (16 GB mode) or 7 GB RAM tier + NVMe store; misses masked at decode | + `EXL3_NVTIER_VRAM_WARM` warm start (~4 s, on by default in image) to stop cold-start answer loops |
| kernels | Strata native sparse attention + GDN | SYCL sparse attention/selection ported **from Strata**, Triton hyper-connection fusion (945→366 ms per 8K forward), exl3xpu MoE tier patches | new in `experimental/` only: victim ring (bit-exact 72/72 standalone, kills the write-back race, NOT in server), SYCL GDN prefill (CPU-validated, GPU pending), fused decode kernels (builds, GPU pending) |
| MTP | `--spec 4` active, measured | none measured; NEXTN plan, "projected 32-43 tok/s exact at 1 stream. Not run yet" | unchanged — still a projection |
| best published decode | **48.03 engine-decode C1** exact | 29.2 C1 exact-path-class / 34.5 C1 and 76.3 C4 **approximate** | 24.0 → 29.2 C1 via mode change (more masking), 76.3 via removed write-back |

Their earlier repo numbers (RL-121, measured 2026-10-06): 24.0 C1 exact default
path (write-back race known: 50-52 mismatched rows in stress simulation, one
runaway generation observed) and 34.5-76.3 approximate. The 2026-10-08 N130
session added the 16 GB mode (29.2 C1, faster *because* it masks more experts:
decode KL 0.055 vs 0.021 at 32 GB) and honest quality accounting (prefill-path
top-1 0.985 / KL 0.003 — misses their own kit band; greedy panel **0/8** equal
to exllamav3 text in both modes; lab chat gate fails — thinking loop to 64k).
No ≥70 number exists on any exact-output path in either state of the repo.

Metric-class note (AGENTS §8): their decode counts completions to natural end
over 8k/32k prompts, timing boundary unstated [UNVERIFIED]; ours is engine-side
wall accounting at fixed p512/g128. Their 76.3 is C4 aggregate; ours is C1.
Neither side's number is directly the other's; both must be re-quoted under
both protocols before any comparison (see §4).

## 4. Reproduction on our host

**Exact artifact and command (their published path, README @ `db27d3c5f922`):**

```bash
IMG=ghcr.io/sybil-solutions/qwen38-flash-next-b70-offload@sha256:d30df691d6ab890b95f7c6179e4b677a0d759018e39b20e93bf0efdb8780450c
M=turboderp-Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5
hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 69e33439ae950f17bcbe95c98f117d80f759ab6d --local-dir /data/$M
docker run --rm -v /data/$M:/models/$M:ro -v <nvme>:/nvx "$IMG" pack-store   # one-time: 45.8 GB store
docker run -d --name qwen-b70 --device <B70 render node> --memory 16g --memory-swap 16g --shm-size 8g --ulimit memlock=-1 \
  -e QWEN_B70_MODE=nvme16 -p 127.0.0.1:30000:30000 -v /data/$M:/models/$M:ro -v <nvme>:/nvx:ro "$IMG" \
  python3 -m sglang.launch_server --model-path /models/$M --quantization exl3 --trust-remote-code --device xpu \
  --host 0.0.0.0 --port 30000 --served-model-name flashnext --disable-shared-experts-fusion --kv-cache-dtype fp8_e4m3 \
  --context-length 65536 --mem-fraction-static 0.85 --chunked-prefill-size 8192 --max-total-tokens 131072 \
  --max-running-requests 4 --dtype bfloat16 --max-mamba-cache-size 32 --cuda-graph-backend-decode full \
  --cuda-graph-bs-decode 1 2 4 --reasoning-parser qwen3 --tool-call-parser qwen3_coder
```
The 76.3 arm additionally sets `NOVICT=1` in the srv26 container (documented in
docs/results.md; not in the README command above).

**Budget (live, 2026-10-13):**

- Downloads: ~85 GB (checkpoint 85.14 GB: 7 shards 52.3 GB + n-gram
  32.64 GB) + ~3 GB image; `pack-store` builds the 45.8 GB expert store locally
  (no extra download). Total new bytes ≈ **88 GB download, ~135 GB installed**.
- Disk (live `df`): root 2.6 G free, NVMe SN770 **36 G**, T-FORCE 4.2 G,
  sata-wd 42 G. The store MUST sit on one local NVMe/xfs/ext4 with `O_DIRECT`
  and be read at 4-5 GB/s during prefill — our only NVMe has 36 G < 45.8 G, and
  no single mount holds the 85 GB checkpoint. Under our 2× rule the NVMe would
  need ≥92 G free. **Storage-blocked today.** (Our own Flash-Next artifacts are
  Strata-class only: T-FORCE `strata/models-iq3xxs/IQ3_XXS` 47,039,860,096 +
  28,800,138,432 B GGUFs, `strata/pack/iq3xxs`, `strata/mtp-rt` (experts.bin
  707,788,800 B, draft_vocab.bin 425,196 B); no EXL3 3.05bpw checkpoint and no
  packed expert store anywhere on the three mounts.)
- RAM (live `free`): 60 GiB total, 49 available. `nvme16` caps the container at
  16 GiB (their measured peak 14.3 GiB anon+shmem, no swap) → **fits**. The
  recipe-kit pinned-RAM variant (68 GB free needed) → does **not** fit.
- Cards: the tier route is **one B70** — GPU1 (free) is enough; **GPU0 not
  needed**. Two cards play no role in any of their Flash-Next numbers (and our
  own Strata arm-b two-card path was slower).
- GPU-hour estimate (GPU1, after storage exists): pack-store ~1 h (CPU/NVMe),
  server smoke + N130 matrix (2 modes x 8k/32k x C1/C2/C4, natural-end
  completions) ~2-3 h, quality gate ~2 h, matched-metric cells ~1-2 h →
  **≈6-8 GPU-hours**, no kernel builds.

**Quality gate to run (predeclared):**

1. Exact-answer canaries — our batt-{arith,capital,token} battery at gen>=64
   (703 / Paris / B70-FLASH-OK) on both the exact and the no-write-back arm;
   the approximate arm must not be adopted regardless of speed (§20.5).
2. Their own reference panel (`kit/tools/score_ref_panel.py`): top-1 >= 0.987
   and mean KL <= 0.0013 vs the exllamav3 reference; plus their decode-vs-prefill
   KL check (`serve/n130/decode_kl_n130.py`). Their published values already
   fail the kit band (0.985 / 0.003) and show 0/8 greedy panel equality — a
   reproduction should expect to confirm, not fix, that.
3. Thinking-loop / chat-gate check (their lab chat gate fails: thinking until
   the 64k context runs out) and one cold-start answer check (empty VRAM cache
   produced a 4-token answer).

**Matched metric vs our 48.03:** dual-quote both units (AGENTS §8): (a) our
protocol on their stack — p512/g128 greedy, engine decode wall accounting,
C1, 230 W-equivalent declared caps; (b) their protocol on our Strata serve —
8,181-token prompts, completions to natural end, total tok/s at C1/C2/C4.
Only (a) vs our 48.03 is a same-class comparison; their 76.3 is C4 aggregate
approximate and is not comparable to it as published.

## 5. Verdict (provisional)

The circulated "100 tok/s" belongs to **GLM-5.3-Flash on a new 2x B70 machine**
(2026-10-13, no artifact) and to DeepSeek-on-Sparks posts — not to
Qwen3.8-Flash-Next on B70. Every >=70 tok/s Flash-Next claim by 0xSero/sybil in
the window is one number: **76.3 tok/s = C4 aggregate decode, approximate
output** (expert-masking no-write-back arm, 1x B70, EXL3 3.05bpw,
SGLang 0.5.20-xpu + exl3xpu, no speculation). Its C1 sibling is 34.5 tok/s; the
best exact-path-class figure they publish is 29.2 tok/s C1. Compared with our
48.03 tok/s (C1 engine decode, exact output, Strata IQ3_XXS, one B70): **not
comparable as published** (concurrency, approximate output, timing boundary all
differ), and their own exact-path projection (32-43 tok/s with MTP) sits below
our measured route. Reproducing their rows on this host is **not feasible now**:
needs ~88 GB of downloads plus ~135 GB installed, with the 45.8 GB expert store
on an NVMe that must have ≥92 G free (live: 36 G) — GPU1 alone suffices
(one-card route, GPU0 untouched), and once storage exists the run is ≈6-8
GPU-hours with the gates above.
