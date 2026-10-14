---
title: "Sybil Solutions / 0xSero >100 tok/s on 2x B70 claim audit"
type: research
date: 2026-10-13
status: provisional
---

# sybil-solutions (>100 tok/s on 2x B70) claim audit

Read-only audit of every published >100 tok/s decode claim attributed to
github.com/sybil-solutions / 0xSero on 2x Intel Arc Pro B70, checked against our
incumbent (Qwen3.8-27B EXL3 4.00bpw @113cf7ab, exl3xpu, one B70, MTP3, fp8 KV,
63.25 tok/s client post-first p512/g128 n=5 @230 W — ledger RL-090; RL-186 pairs
63.56/64.34 at T0.7). Community claims are leads. All unverified items marked
[UNVERIFIED]. No GPU work, no pulls, no ledger registration performed.

Sources scanned: all 25 sybil-solutions repos (trees + READMEs), exl3xpu and
trellis-serve commits since 2026-09, local-ai-registry `registry/` + `data/`
recipe trees and lab-run evidence, qwen38-b70 / qwen-serve bench scripts and raw
results, GitHub releases (none exist; images ship via release-image workflows),
exl3xpu issues #1-#2, x.com/0xSero posts. Local: ~/B70-DOCS grep
(sybil/0xSero/trellis-serve/exl3xpu), ~/b70-launcher/recipes.json,
/mnt/<nvme-disk> + /mnt/<sata-disk> artifact census.

## 1. Evidence table — every >100 tok/s 2x B70 claim

### C1 — X post 2026-10-13 (the claim as surfaced): "GLM-5.3-Flash and some 100 tok/s", 2x B70

- Quote: "New machine new me. This one is about 12,000$ and can run GLM-5.3-Flash
  and some 100 tok/s" — https://x.com/0xSero/status/2109967060403175670
  (2026-10-13); hardware confirmed "2x Intel Arc b70" in reply
  https://x.com/0xSero/status/2109976467505594555. Build: ASRock ROMED8-2T,
  EPYC 7443P, 512 GB DDR4, 4x1 TB NVMe (inference), 200 GiB/s PCIe NIC.
- Model: GLM-5.3-Flash (version/quant unstated; sybil publishes only
  GLM-5.3-Flash EXL3 3.05bpw ≈ 125 GB, turboderp/GLM-5.3-Flash-exl3@332ab457).
- Engine/image/commit, TP vs replicas, concurrency (C1 vs aggregate), spec,
  prompt/output lengths, metric class, power cap, driver: **all unstated
  [UNVERIFIED]** — no benchmark artifact, recipe, or repo commit backs this post.
- Sanity: the best GLM-5.3-Flash decode anywhere in their own repos is
  28.2 tok/s (1x RTX 3090 + all experts in RAM) / 26.5 tok/s with a 1x B70
  expert-tier companion (glm-flash-lite README.md:8-12,
  https://github.com/sybil-solutions/glm-flash-lite @ main). 100 tok/s on
  2x B70 is ~4x their published ladder → treat strictly as a lead.

### C2 — qwen38-b70: "vLLM TP2 — 140.8 tok/s on 2x B70 (2026-08-20)"

- Quote: "Real tensor parallelism (TP2) with MTP5 speculative decoding on the
  lab vLLM image (`intel/llm-scaler-vllm:0.21.0-b3`) achieves **140.8 tok/s
  aggregate** with 4 concurrent request streams, exceeding the 120 tok/s
  target." — README.md:124-135 @ qwen38-b70 commit
  `0f9950b741185d209a70747db304986ed3b7946b`
  (https://github.com/sybil-solutions/qwen38-b70/blob/0f9950b/README.md), mirrored
  verbatim at qwen-serve/qwen3.8/b70/README.md @ `13f8f426e6c7e7de7b141a7e7b060828a660087d`.
- Model/quant: Qwen3.8-27B **AutoRound INT4 W4A16** ("17.7 GB, ALL layers
  quantized including GDN"), artifact named only in
  docs/DECODE_ANALYSIS.md:8 as `devan-carlin/Qwen3.8-27B-int4-AutoRound`;
  exact revision [UNVERIFIED].
- Engine: vLLM XPU 0.21.1 on lab image `intel/llm-scaler-vllm:0.21.0-b3`
  (no digest recorded [UNVERIFIED]) + 4 in-container patches applied at launch
  by `lab_tp2_batch_mtp.sh` @0f9950b: (1) mamba_utils.py ctypes ptr fix
  (state/block-table data_ptr through c_int64), (2) gdn_linear_attn.py GDN
  ESIMD eligibility try/except for quantized RowParallelLinear, (3) v1/utils.py
  CpuGpuBuffer.copy_to_gpu non_blocking=False, (4) xpu_communicator.py XCCL
  all_reduce wrapped with CPU-gloo fallback on OUT_OF_RESOURCES.
- Parallel: real **TP2** (`--tensor-parallel-size 2`), `--enforce-eager`
  (**graph capture OFF**), `--skip-mm-profiling --max-num-seqs 4
  --gpu-memory-utilization 0.45 --max-num-batched-tokens 2048 --dtype float16`
  (HILLCLIMB.md:25 @0f9950b; launch `lab_tp2_batch_mtp.sh`).
- Speculative: `{"method":"qwen3_5_mtp","num_speculative_tokens":5}` (MTP5).
  Acceptance for the 2x run not recorded; the 1x sibling (HILLCLIMB.md:16,34)
  reports 95.2% acceptance, mean accepted length 5.76 tokens/step.
- Concurrency: **C4 aggregate** (=> ~35 tok/s per stream). Single-user C1 number
  for the same 2x config: not published [UNVERIFIED none exists].
- Prompt/output: prompt = 3x repeated GDN paragraph (~1k tokens), max_tokens
  256, temperature 0; 5 rounds, median (`bench_concurrent_stable.sh` @0f9950b).
- Metric class: **client wall-clock aggregate output rate** =
  sum(usage.completion_tokens) / wave wall time, window **includes prefill**
  (bench_concurrent_stable.sh: T0 before firing the 4 curls, T1 after `wait`,
  TPS=TOTAL_TOKENS/WALL). Not engine decode; not client post-first. Their own
  qwen3.6 docs label this quantity "end-to-end output rate" and report it
  separately from decode.
- Power cap: not stated [UNVERIFIED]. Driver: Arch kernel 7.1.8 (xe) per the
  repo's tested-on note; the 2026-08 box ("omarchy") is not the EPYC box of C1.
- Quality gate: two shallow questions (17x3, capital of France) + `!!!`/repeat
  regex (bench_concurrent_stable.sh). Per our RL-087/ADR-0010 this gate class
  passes fluent wrong answers; the symmetric-int4 weight class is closed on
  Qwen3.8-27B by the 30-defect (weights, RL-092). Coherence of the 140.8 run
  under our battery: [UNVERIFIED].

### C3 — community ceiling cited by qwen38-b70: 101.92 tok/s, 2x B70

- Quote: "| **vLLM AutoRound INT4 TP2 MTP5** | devan-carlin/Qwen3.8-27B-int4-AutoRound
  | **101.92** | INT4 weights (~half the bytes) + MTP. Different stack" —
  docs/DECODE_ANALYSIS.md:8 @ `0f9950b`, attributed to the external
  steveseguin/b70-optimization-lab ("lab results", Aug 2026).
- Metric, concurrency, prompt/output, acceptance, power, engine image digest:
  **[UNVERIFIED]** — community lead, no primary artifact located in the org.
- Their own physics note (DECODE_ANALYSIS.md:26-34): B70 608 GB/s peak,
  Q4_K_M-class weight read 8.5 GB/card => hard target-only ceiling
  608/8.5 ≈ **71.5 tok/s per card**; >71.5 single-stream needs speculative
  yield or lighter weights.

### C4 — qwen-serve: Qwen3.6-35B-A3B "114–116 tok/s" on 2x B70 (TP2, C1)

- Quote: "| sglang XPU online FP8 | 2x B70 (TP=2) | Validated | 114–116 tok/s |
  fp32 SSM state, native GDN fast path, XPU graphs" —
  qwen-serve/qwen3.6/b70/README.md:26,50-57 @ `13f8f426`
  (https://github.com/sybil-solutions/qwen-serve/blob/13f8f426/qwen3.6/b70/README.md;
  content inherited from archived qwen36-b70).
- Model/quant: Qwen3.6-35B-A3B (35B MoE, 3B active, hybrid GDN),
  `Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0`, loaded bf16
  to host RAM and quantized **online to FP8** (`--load-format layered_fp8`).
- Engine: SGLang 0.5.13 XPU, image
  `llm-scaler-sgl:bmg@sha256:d027487a07e4391562b4460da72bc78008be129d44af540148787e02e3045c08`
  (benchmarks/results/20260825-195430-b70-sglang.json header), TP=2, XPU graphs
  on, GDN fp32 runtime contract (SGLANG_MAMBA_SSM_DTYPE=float32,
  SGLANG_XPU_GDN_FAST_PATH=1, conv fp16, SGL_XPU_ENABLE_GRAPH=1 —
  README.md:75-80). Speculative: none (MTP blocked: sgl_kernel build_tree/
  verify_tree not compiled for XPU — README.md:31).
- Concurrency: **C1** (per-stream). Prompt 128 / 512 / 2048 / 8192, output 128
  tok, sampling [UNVERIFIED]; 2026-08-25 sweep.
- Metric class: **client post-first decode** == server decode (raw rows carry
  both: client_decode_tok_s 114.9/116.2/113.9/116.3 vs server_decode_tok_s
  115.3/116.5/…). Genuinely a decode rate — but TTFT is terrible (2.19 s @128,
  8.69 s @512, 130.9 s @8192; end-to-end output rate 1.0 tok/s at 8192 in).
- Registry: `registry/recipes/intel/intel-arc-pro-b70-32gb/qwen3.6-35b-a3b.sglang.8k.2x.json`
  proof `tps: 115.0`, **legacy** (gates "load chat" only). Quality caveat: raw
  128-token outputs are visibly degenerate (repeated "Omarchy serves…",
  "DONE DONE DON") yet `valid: true` — their check is non-empty / no `!!!` /
  no token 0. Repetition-collapse at 115 tok/s is [UNVERIFIED] against any real
  quality gate.
- Power cap: not stated [UNVERIFIED].

### C5 — reference, NOT a 2-card claim: trellis-serve/exl3xpu single-card aggregates

- Quote: "C | prose … 1 | **91.2** … 2 | **151.4** … 4 | **262.6** … 8 |
  **357.1** … 16 | **365.2**" (aggregate tok/s, thinking on, t=0.7, realistic
  corpus) — trellis-serve xpu/README.md:36-40 @ `a1f72be2`
  (https://github.com/sybil-solutions/trellis-serve/blob/a1f72be2/xpu/README.md);
  source tuning log exl3xpu commits `da9014b0`/`05a5bfb4`/`e6d4f745`
  (2026-09-23/24). **One card** (banner: "on one Intel Arc Pro B70"); C>=2
  cells exceed 100 tok/s on one card.
- Same artifact as ours: turboderp/Qwen3.8-27B-exl3 @113cf7ab (4.00bpw), MTP3
  with **pruned draft head**, fp8 KV in 1600-token blocks, 262,144 ctx,
  image `ghcr.io/0xsero/exl3xpu@sha256:21412bdd7535e9c653eeb3d099dce3bc79a83d440def2cd9556111c79c870fa8`
  (built from source commit `2de46a88249235de75a4d90a59a026eea33291a8`).
- Metric: token-arrival rate in a steady window — `bench/sweep.py:5-6`
  ("aggregate decode = output tokens whose arrival falls inside the steady
  window / window; per-stream = (n_chunks-1)/(t_last-t_first)"), 60-90 s
  sustained, cold unique prompts; raw rows bench/results/2026-09-23.jsonl
  @a1f72be2. Registry acceptance run separately recorded 81.6 tps single
  request ("counted: server", 15197 tok / 186.33 s;
  data/lab-runs/intel-arc-pro-b70-32gb.qwen3.8-27b.registry.256k.20260925T161413.json).
- 2-card mode exists but is **unmeasured**: `… --dp` (two replicas behind
  scripts/lb.py; model.yaml `parallel: {mode: data, gpus: [0,1]}` —
  xpu/README.md:125, model.yaml:48-52 @a1f72be2). No 2x B70 number for
  Qwen3.8-27B EXL3 is published anywhere in the org [UNVERIFIED none exists].
- xpu/README.md also states "Tensor parallelism is not supported yet; use data
  parallel" — no TP2 path in the EXL3 engine at all.

### Commits since 2026-09 (what changed under these claims)

- exl3xpu (archived 2026-10-12): last commits 2026-09-26 (`15ded2f3`); the
  2026-09-23/25 window added exact 1600-token KV blocks (`cc575a09`), DPAS
  MB=24/40/48 dispatch (`5de2b911`, `0636d408`), int8 prefill linears
  (`7e15329a`/`170cf83a`), oneDNN Graph SDPA (`9db051da`) — all prefill/C16
  levers except the KV-block patch. Image 21412bdd is the final published build.
- trellis-serve (current `a1f72be2`, 2026-10-12): folds exl3xpu in; adds SGLang
  plugins and launch scripts; **no new decode-speed claim**. GitHub releases:
  none on any repo. exl3xpu issue #2 (TTFT/prefix-cache on vLLM 0.26, closed,
  "update to vllm 0.30") is the only open performance thread besides the AOT
  offer (#1).

## 2. Their launch/config/patches vs our image and recipes

Our side: image `ghcr.io/sergiiob/exl3xpu-mixedk@sha256:7bb0c5cfed8a35da8f4db0034338d8dbe5f53731253d96ea8f1db2b5bddd4a40`
(tag 20261012c; fork SergiioB/exllamav3 branch b70 @`244f57e` "mixed-K EXL3
serving patches for exl3xpu and ESIMD trellis quantizer", verified byte-identical
_C.so), launcher ~/b70-launcher/recipes.json `exl3` entry (image
ghcr.io/0xsero/exl3xpu:0.1.0@21412bdd, isolated dockerd
unix:///run/<exl3-docker>.sock, ctx 65536, power 230 W, `qwen38-27b-sc5`
entry adds the mixedk image). Both `ghcr.io/0xsero/exl3xpu@21412bdd` (23.6 GB)
and our mixedk image are already present on the isolated daemon (read-only
`docker images` check, 2026-10-13).

Absent from our image/recipes (same-artifact EXL3 lane — C5's config):

1. `EXL3_DRAFT_VOCAB: {model_dir}/draft_vocab.json` — pruned MTP draft head
   (model.yaml:42 @a1f72be2; draft_vocab.json ships in their repo). Our
   RL-090/launcher run does not set it (RL-106: "RL-090 launch did not set it.
   No measurement"). The lever itself lives in the pinned plugin
   (vllm_plugin.py:237,271-319) which our image lineage carries — the env var
   and the vocab file are the only missing pieces.
2. `EXL3_KV_BLOCK_EXACT: "1"` (model.yaml:43) — exact 1600-token KV blocks,
   their +17% aggregate at C16 (14 of 16 streams resident). Not set in our
   recipes [UNVERIFIED it was off in RL-090; our pool 287,040 tok @ gmu 0.94 is
   consistent with default blocking].
3. `EXL3_INT8_PREFILL: "1"` + `EXL3_ONEDNN_ATTN: "1"` (model.yaml:45-46) —
   prefill-only; decode unchanged and bit-exact per their gates. Not
   decode-relevant.
4. `mamba_cache_mode: align`, `gpu_memory_utilization: 0.965` (vs our 0.90-0.94),
   `max_num_batched_tokens: 4096`, explicit `--enable-prefix-caching` —
   config deltas, [INFERENCE] < few % at C1 decode.

Absent and NOT portable (the 140.8 lane — C2's stack): the entire
`intel/llm-scaler-vllm:0.21.0-b3` vLLM 0.21.1 engine with its four launch-time
patches (ctypes ptr fix, GDN ESIMD eligibility, non_blocking=False, XCCL→CPU
gloo all_reduce fallback), real TP2, `--enforce-eager` (graphs OFF),
`--dtype float16` KV, AutoRound INT4 W4A16 weights, MTP5 via `qwen3_5_mtp`.
None of this exists in our image (different vLLM generation); the patches
target 0.21 file layouts [INFERENCE: not applicable to 0.26.1/0.30]. The
AutoRound artifact (devan-carlin/Qwen3.8-27B-int4-AutoRound) is not on our
disks. The Qwen3.6 lane (C4) shares nothing with our stack beyond the card
(llm-scaler-sgl:bmg image, layered_fp8 online quant, SGLang GDN contract).

## 3. Differences vs our 63.25 tok/s, ranked by likely tok/s impact

| # | Difference | Est. impact | same-metric? | same-concurrency? | reproducible-on-our-host? |
|---|---|---|---|---|---|
| 1 | **Metric + concurrency inflation of the headline**: 140.8 = C4 aggregate, client wall incl. prefill, ~35 tok/s/stream; ours = C1 client post-first decode | explains the entire ">100" headline | NO | NO | yes mechanically, not comparable (fails AGENTS §8 metric naming) |
| 2 | **Spec depth + acceptance**: their MTP5 @95.2% acc (5.76 tok/step) vs our MTP3; their own 1x sweep MTP3 78.3 → MTP5 90.1 on their bench | +10-15% within one lane | NO (different bench) | YES (C1) | partial — k=5 on the EXL3 MTP head is [UNVERIFIED]; AutoRound sweep only |
| 3 | **Pruned draft head (EXL3_DRAFT_VOCAB)** — same artifact, one env var we never set; removes up to 2.86 GB/round draft lm_head re-read (RL-106) | [INFERENCE] +5-12% at C1 (4.8 ms of a ~45 ms round at ~600 GB/s) | YES | YES | **YES — top actionable lever; image + artifact already local** |
| 4 | **Content/sampling**: thinking-on t=0.7 prose 91.2 vs their own synthetic greedy 76.6 vs ours p512/g128 greedy 63.25 | up to +19% (their own data) | ~YES (both client rates) | YES | YES (bench/sweep.py published) |
| 5 | Power cap: theirs never stated (default card cap) vs our 230 W measured; our own 150→230 W swing is +33-42% chunk-rate | confound, potentially tens of % | n/a | n/a | YES (we can hold/lift cap per §10) |
| 6 | Quant/engine class of the 140.8 lane: AutoRound INT4 + vLLM 0.21 + fp16 KV + TP2-eager vs EXL3 4bpw + vLLM 0.26.1 + fp8 KV + graphs | large but one-way: their DECODE_ANALYSIS itself prices target-only at ≤71.5 tok/s/card | NO | NO | NO — policy: TP2 unauthorized (AGENTS §7), capture-off diagnostic-only (rule 1), W4A16 quality class closed (ADR-0010/RL-092), artifact absent |
| 7 | Card count: 2x TP2 / 2x TP (C2, C4) vs our 1 card, dual-card default = two independent services (ADR-0008); host has no peer atomics, P2P 11-12 GB/s, PCIe x8/x8 | 2x cards ≠ 2x tok/s at C1 here | NO | NO | partial — exl3xpu TP2 was blocked at load (RL-103); `--dp` replicas are the authorized path |
| 8 | Driver/runtime: Arch 7.1.8-arch1-3 + compute-runtime 26.31.39395.13 / IGC 2.40.13 (theirs) vs our kernel 7.0.0-28-generic host | unknown [UNVERIFIED] | n/a | n/a | partial (kernel swap is a host change) |

## 4. Reproduction plan (AGENTS §9 paired design)

**Question**: how much of the same-artifact gap (their 91.2 C1 vs our 63.25-64.34)
is levers (draft head, KV blocks, spec depth) vs protocol (content, window,
metric)?

- **Artifacts (all present, zero downloads)**: model
  `/mnt/<nvme-disk>/b70-exl3-qwen38-4bpw-20260926` (16 G,
  turboderp @113cf7ab — byte-same artifact as their recipe);
  images `ghcr.io/0xsero/exl3xpu@21412bdd` and
  `ghcr.io/sergiiob/exl3xpu-mixedk@sha256:7bb0c5cf…` on the isolated daemon
  (unix:///run/<exl3-docker>.sock); fork
  `/mnt/<nvme-disk>/src/exllamav3-b70` @244f57e;
  `draft_vocab.json` fetched once from
  sybil-solutions/trellis-serve xpu/models/qwen3.8-27b-exl3-4.00bpw/ (small).
- **Command**: existing launcher/recipe launch (models/qwen3.8-27b-exl3-4.00bpw,
  `--gpu 0 --port … -- --enable-prefix-caching …`), plus model.yaml-equivalent
  env `EXL3_DRAFT_VOCAB=/models/draft_vocab.json` for the treatment arm;
  optionally `EXL3_KV_BLOCK_EXACT=1`. GPU0 at 230 W (§10 counters).
- **Metrics — ours matched**: primary = (usage.completion_tokens-1)/(last-first
  content chunk), p512/g128, n=5, client post-first (RL-090 protocol) at 230 W;
  secondary = their protocol verbatim (`bench/sweep.py --corpus real --thinking
  --temperature 0.7`, windowed token arrivals) so both sides quote both units
  (§8 metric naming).
- **Pairs (§9: one discarded warmup; interleaved ABBA; 5 pairs confirmation)**:
  - Pair A (top): EXL3_DRAFT_VOCAB on/off, artifact + image fixed. Gate: greedy
    output byte-identical (target verifies full vocab; pruned head is
    draft-side — expected output-preserving, prove it).
  - Pair B: EXL3_KV_BLOCK_EXACT on/off at C16 (their +17% is C16-bound;
    [INFERENCE] flat at C1).
  - Pair C: protocol content A/B on one server — p512/g128 greedy vs realistic
    thinking t=0.7 window — attributes the 91.2 / 76.6 / 63.25 spread.
  - Pair D (optional, both cards): their `--dp` two-replica + lb.py vs
    ADR-0008 two independent services at C1/C2; declare aggregate and tail
    latency separately (§9).
  - Single-cell probe (not paired): `num_speculative_tokens: 5` on the EXL3 MTP
    head — [UNVERIFIED] supported; canary + battery before any speed cell.
- **GPU time estimate**: Pair A ≈ 1.5 h GPU0; Pair B ≈ 0.5 h; Pair C ≈ 1 h;
  Pair D ≈ 1 h (both cards); MTP5 probe ≈ 0.5 h. Total ≈ **4-5 GPU-hours**,
  splittable around RecipeSweepGpu1; no new kernel builds.
- **Blockers / what we will NOT chase here**: root 2.8 G free (never pull/build
  there; isolated data-root is on NVMe), NVMe 36 G free (enough for Pair A-D;
  nothing to download), sata-tforce 4.2 G free and its
  `b70-autoround-local-qwen38-20260926` dir is **empty**. The 140.8/101.92 lane
  needs the devan-carlin AutoRound INT4 artifact (~18 G, absent) + the
  unpublished lab image `intel/llm-scaler-vllm:0.21.0-b3` + TP2 with gloo CPU
  all_reduce on a no-peer-atomics host — policy-blocked (AGENTS §7, rule 1,
  ADR-0010) and of no decision value. The C1 X-post claim (GLM-5.3-Flash ~100
  tok/s) is unreproducible as stated: no stack published, no GLM-5.3-Flash
  artifact on our disks (125 G vs 36 G free NVMe), and 4x above their own
  published GLM ladder.

## 5. Verdict (provisional)

The ">100 tok/s on 2x B70" headline is **not a decode-rate claim comparable to
our 63.25 tok/s**: the only org-published 2x numbers are (a) 140.8 tok/s =
C4-aggregate client wall rate including prefill on a TP2 AutoRound-INT4 vLLM
0.21 stack (≈35 tok/s/stream, graphs off, shallow quality gate on a weight class
we closed for the 30-defect), (b) 101.92 tok/s community ceiling with no
metric, (c) 114-116 tok/s C1 decode on a different model (Qwen3.6-35B-A3B MoE,
online FP8, TP2) whose raw outputs show repetition collapse, and (d) an
unbacked social post (~100 tok/s GLM-5.3-Flash). The decision-relevant finding
is C5: the **same EXL3 artifact on one card at 91.2 tok/s C1** (thinking t=0.7)
against our 63.25-64.34, with named env-level levers (pruned draft head, exact
KV blocks) that our route never enabled. That gap is reproducible on this host
in ~4-5 GPU-hours with zero downloads and is where the next paired run should
go. [UNVERIFIED until paired confirmation] the split between levers and
protocol.

### Correction (2026-10-13, parent check)

Rows 1-2 of section 2 and row 3 of section 3 are wrong for the current route.
Both levers are already active in every exl3xpu launch: the image's
`models/qwen3.8-27b-exl3-4.00bpw/model.yaml` sets `EXL3_DRAFT_VOCAB` (with
`draft_vocab.json` shipped beside it) and `EXL3_KV_BLOCK_EXACT=1`, and
`scripts/serve.py env_for()` applies them. Today's launcher logs show
"MTP draft head uses 512 of 1940 vocab blocks (26.4% of lm_head)" and
"XPU block size keeps the hybrid negotiation (EXL3_KV_BLOCK_EXACT=1)"
(`/tmp/b70-launcher-06-smoke/b70-launcher/logs/qwen38-27b-exl3-18103-*.log`).
Pair A and Pair B therefore have no untested arm. The remaining same-artifact
gap (91.2 vs 63.25-64.34) is protocol (thinking prose at t=0.7 vs p512/g128),
speculation depth (MTP3 vs MTP5, unverified on the EXL3 head) and power cap.
Pair C plus a single-cell MTP5 probe are the only open measurements.
