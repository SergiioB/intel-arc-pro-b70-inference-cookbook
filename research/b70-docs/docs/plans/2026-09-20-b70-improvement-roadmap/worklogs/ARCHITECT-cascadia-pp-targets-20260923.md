---
title: "Cascadia pipeline-parallelism targets on dual B70 — architecture assessment"
type: worklog
date: 2026-09-23
author: b70-hw-architect
intent: research (read-only; no GPU work, no state changes)
status: final
---

# Cascadia PP targets — qwen3.8-flash-next / qwen3.8-27b / MiMo-V2.6-Flash-RL

> PROVISIONAL — NOT FOR PUBLIC HEADLINE. Assessment only. No launch, no GPU
> workload, no git state change was performed for this note.

Situation compiler returned `halt: true` on RL-016 (`PP2.*host.*RAM|pipeline.*parallel.*2.*B70`).
That block covers the **vLLM XCCL PP2** launch path (root cause:
`research/2026/qw38-fp8-singlecard-unblock-and-pp2-rootcause-20260827.md`). The
Cascadia-engine PP question is a genuinely different mechanism (no XCCL, no
process ever sees both GPUs) and a **live campaign already started today** in a
parallel session (`results/2026/09/cascadia-pp-qwen38-20260923/logs/`, started
2026-09-23T07:24Z). This note is the architecture frame for that campaign and
the two targets behind it.

## 1. Why PP is the right shape for this topology (and TP is not)

- X570-F: dual CPU-attached x8/x8, **no GPU P2P**, measured H2D floors
  8.85 / 7.89 GB/s (`research/2026/x570f-dual-b70-pcie-topology-20260830.md`).
  TP2 needs per-layer host-staged collectives — gated research per
  `decisions/adr-0008-dual-b70-independent-default.md`.
- The vLLM PP2 `blocked_host_ram` failure was isolated to a **single process
  seeing both B70s** (`ZE_AFFINITY_MASK=0,1`): ~1 GiB host RAM per GiB VRAM,
  invisible to normal accounting (probe P1/P3: 0.98–1.00:1). Two processes with
  disjoint masks cost 0.08:1 (probe P5) — the fix, not the model, was the blocker.
- Cascadia's PP model is structurally the safe shape: `cascadia run-placement`
  spawns **one `cascadia worker` process per stage, each pinned to its own
  device** (`--device GPU.0` / `GPU.1`), wired as a ring — rank 0 holds the API
  and dials `--next`, rank r≥1 binds an `ActivationServer` TCP relay
  (default :9100). Source: `cascadia-pr132-wt/crates/cascadia-cli/src/run_placement.rs`
  (`worker_argv`, `plan_workers`), `cascadia-types/src/peer.rs`. No process sees
  both GPUs, no collectives, no P2P requirement.
- Boundary traffic is negligible *in bandwidth* and real *in latency*:
  decode moves one hidden state forward + one token id back per token
  (27B: 5120 × 2 B = 10 KB). Prefill at ubatch 512 moves ~5.2 MB per boundary
  step. Loopback TCP bandwidth is orders of magnitude above need.

## 2. The prior measurement that prices everything

`results/2026/08/cascadia-dualgpu-20260824T130000Z/` (Cascadia 0.2.0, OV 2026.3,
Qwen2.5-1.5B 2-stage INT4, GPU.0 + GPU.1, relay :9100):

- **Mechanics worked end-to-end** on this exact host: both workers reached
  `runner ready`, ActivationServer accepted from 127.0.0.1, text out.
- **Cost**: pipeline median **70.0 vs 85.8 e2e tok/s** 1-stage control = −18 %
  small-model tax. Per-task log fields: `wire_ms` 520–556 per 81 tokens ≈
  **6.5–6.9 ms/token of relay latency** (`alpha_ms` is engine, `wire_ms` is
  relay; 1-stage showed `wire_ms=0`).
- Scaling law this implies: **PP2 nets positive iff per-token stage compute ≫
  ~7 ms.** Qwen2.5-1.5B stage ≈ 5–8 ms → negative. Qwen3.8-27B on Cascadia is
  202–219 ms/token (4.56 tok/s INT8 qwen35 1-stage, bakeoff; 4.95 tok/s
  INT4-GDN8 P512/G128 @ ~102 W per `engines/cascadia/README.md`) →
  **expected PP2 ≈ 0.5 × 210 + 7 ≈ 112 ms ≈ 8.5–9 tok/s ≈ 1.75–1.9×**,
  *if* the 200 ms is per-layer execution (halves with half the layers) rather
  than fixed per-token overhead. That dichotomy is exactly what the A/B decides.
- Absolute-class honesty: even a perfect 2× Cascadia ≈ 9–10 tok/s vs the
  incumbent champion **102.61 tok/s** vLLM GPTQ-INT4+MTP4 p512/g128
  (`models/qwen/qwen3.8-27b/serving.md`). The 27B PP A/B validates **mechanics
  and the wire constant**; it is not a serving-route win. Cascadia remains
  "not a B70 speed path today" per its own dossier.

## 3. Q1 — immediate lowest-risk PP A/B on the 27B (already running)

**Artifacts (all on `/mnt/<nvme-disk>/`):**

| Dir | Shape | Size | Role |
|---|---|---|---|
| `shard-qwen38-1s-64` | 1 stage, 64 layers, embed+head | 15 GB | **A-side control** (same source, same engine) |
| `shard-qwen38-27b-2s` | 32/32, `stage_hidden` boundary | 5.9 + 7.7 + 1.3 GB | **B-side** (live campaign model) |
| `shard-qwen38-2s-60-4 / -56-8 / -63-1`, `3s-48-8-8`, `4s-*` | imbalance variants | 15–16 GB | post-hoc stage-balance tuning only |

**Launch shape (from the live campaign log, `worker-rank1-gpu1.log`):**
rank1 = `cascadia worker --engine Qwen36Moe --rank 1 --total 2 --device GPU.1
--model .../shard-qwen38-27b-2s` (+ `--listen 127.0.0.1:9100`), rank0 = API head
on GPU.0 with `--next 127.0.0.1:9100`; qwen35 prefix-cache budget 15 GiB/worker.
Note: engine id prints as `Qwen36Moe` — label results `cascadia (qwen35)` per
the dossier taxonomy rule.

**Baselines:**

1. **Matched A/B (the only comparison that decides)**: 1-stage
   `shard-qwen38-1s-64` on GPU.0, same binary, same caps, same protocol as the
   2-stage run — bakeoff protocol verbatim (greedy n=128, 1 warmup + 3 reps,
   declared W, `research/2026/ov-cascadia-qwen38-bakeoff-20260911.md`).
2. Context only, never the same table: bakeoff 1-stage 4.56 tok/s (INT8) /
   4.95 (INT4-GDN8); OV GenAI Python 13.09–14.65; llama.cpp 18.58 tg128;
   vLLM champion 102.61. Engine taxonomy per `engines/cascadia/README.md`.

**Pre-declared verdict arithmetic:** PP2 decode ≥ 1.5× matched 1-stage = PP
scaling real (proceed to price MiMo); 1.0–1.5× = partial — split `wire_ms` vs
`alpha_ms` from worker logs before any retry; < 1.0× = PP dead at this layer
count → record ledger closure.

**Correctness gate before any number counts:** both shard manifests say
`source: Qwen3.8-27B-int4-ov`. The *official* int4-ov IR emits collapsed logits
(garbage strings / token id 0) — the known-good artifact is the local
INT4-GDN8 hybrid (`research/2026/qwen38-ov-int4-gdn8-convert-20260912.md`,
public IR `SergiioB/Qwen3.8-27B-int4-gdn8-ov`, language ≈ 15–21 GB — sizes are
consistent with these trees, but the source string is ambiguous). First smoke
must verify real English text on a boundary-crossing prompt plus deterministic
replay n=2; garbage/`!`-only output = abort, wrong artifact.

**Capture:** worker per-task `prefill_ms / decode_tok_s / alpha_ms / wire_ms /
other_ms` (already logged), hwmon energy per contract §10 (dynamic node
resolution — the live campaign found **hwmon5/hwmon6**, indices moved again),
debugfs VRAM both cards.

## 4. Q2 — can Cascadia consume qwen4exp GGUF or MiMo PP/EP shards? **No — both need conversion.**

Cascadia serves **OpenVINO IR shard trees only**. `cascadia shard` "slices a
HuggingFace causal-LM into per-stage OpenVINO IRs" (`cascadia-pr132-wt/docs/SHARDING.md`);
runtime engines are `ov-runtime` / `ov-genai` / family-surgery runtimes
(`qwen35` for qwen3_5 GDN; `cascadia-engine-sparse-moe` targets DSv4 — retired
from scope). Nothing in the crate list reads GGUF; QUICKSTART: "it does not
download or convert at run time."

- **qwen4exp / flash-next**: the 95 GB `Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf`
  is llama.cpp-format — dead on arrival for Cascadia. The Cascadia route needs
  (a) optimum export of the ~180B-tensor checkpoint to OV IR — a staging job
  larger than the 27B GDN8 export that already required ~52 GB BF16 + 48 GB
  FP16 peak and a second disk, (b) exporter support for the qwen4exp arch —
  the exact class of gap that produced the GDN INT4 collapse and the
  `Tokenizer::setup_tokenizer` SIGSEGV, and (c) a qwen4exp family runtime that
  does not exist. Highest-risk path of the three.
- **MiMo-V2.6-Flash-RL**: `model_pp0_ep*_shard0.safetensors` is a
  *distributed-load packaging* (PP rank × EP shard), not an input format.
  Cascadia needs consolidation (or a sharding-aware loader — none exists), then
  optimum export (MiMoV2 arch support unverified; llama.cpp's
  `conversion/mimo.py` registers `MiMoV2FlashForCausalLM`/`MiMoV2ForCausalLM`
  covering V2/V2.5/V2.5-Pro — **V2.6's config arch string must be checked
  against that registration before even the GGUF path**), then a family
  runtime (none exists). `model_mtp.safetensors` additionally has no Cascadia
  consumer (no MTP/speculation story measured on this engine).
- Cheapest Cascadia entry for MiMo, if ever pursued: `cascadia shard --model
  XiaomiMiMo/MiMo-V2.6-Flash-RL` (it performs the HF→OV export itself) — gated
  on optimum supporting the arch; else it closes as
  `blocked_missing_architecture_support`, the same class as flash-next's
  AutoRound stop (`models/qwen/qwen3.8-flash-next/README.md`).

## 5. Q3 — target order and the communication/computation tradeoff

**Order: 27B (validate, today) → MiMo (the real PP product candidate) →
flash-next (not a PP case).**

1. **qwen3.8-27b Cascadia PP2 — now.** Only target with shards, control, and
   protocol on disk; 15 GB VRAM trivial; live campaign running. Deliverable =
   the measured wire ms/token at hidden 5120 and the PP scaling verdict that
   prices the next two.
2. **MiMo-V2.6-Flash-RL — second.** Not on host (scope table in
   `docs/plans/2026-09-20-b70-improvement-roadmap/OUTCOMES.md`: "acquire +
   convert + inventory"). It is the only target whose PP motivation is
   **capacity/speed, not validation**: if its INT4 size lands in the ~33–60 GB
   band (single 32 GB card too small, dual card fits), PP2 is the only whole-model
   serving shape on this host — TP2 stays gated (ADR-0008) and llama.cpp
   `-sm layer` pays host-bounced per-layer staging priced by the K216 record
   (AGENTS §16). Gates before any PP work: config-derived total/active params
   and quantized size (model-can-it-fit arithmetic), arch support in
   optimum/llama.cpp, MTP-head inventory (the scope objective: does the
   draft-side INT4 lever transfer — `research/2026/qwen38-draft-side-int4-upstream-20260921.md`).
3. **qwen3.8-flash-next — last; Cascadia is not on its critical path.** 95 GB
   does not fit 2×32 GB GPU-resident; heterogeneous GPU+CPU stages would put a
   ~180B MoE's CPU stage on DDR4 — the retired K216 experience prices that at
   the 1–2 tok/s class. Its actual blocker is the llama.cpp load: mainline
   1692f9e50 has no `qwen4exp` (verified by grep), the pinned
   `/mnt/<models-disk>/qwenflash-llama-pinned-9723942` tree **does** (qwen4exp present
   in `llama-arch.cpp`/`llama-model.cpp`), and today's four load attempts on
   the 95 GB artifact died at `UR_RESULT_ERROR_DEVICE_LOST` in
   `ggml_backend_sycl_buffer_init_tensor` memset — a VRAM/loading-path
   triage item (AGENTS §5 partial-offload budget), not an arch gap. The scope
   objective there is the P-3b fused-gemm kernel transfer, not PP.

**Tradeoff summary (2 cards, no P2P):** TP2 = per-layer activation exchange
every token over host-staged links (gated, plus the mask=0,1 RAM tax history).
Cascadia PP2 = one 10 KB hidden + one token id per token over loopback TCP —
measured ~6.5–6.9 ms/token fixed on q25; bandwidth irrelevant. Decode per-card
weight reads halve (INT4-GDN8 ~15 GB → ~7.5 GB/card ≈ 12.4 ms floor at
608 GB/s); whether the observed 210 ms/token follows depends on the
per-layer-vs-fixed dichotomy the A/B measures. Prefill relay cost is one
~5 MB message per ubatch — negligible.

## 6. Q4 — go/no-go gates

| Gate | Criterion | On fail |
|---|---|---|
| G0 preflight | Empty-GPU gate both cards (contract §4); dynamic hwmon resolve (now hwmon5/6); **one** declared campaign — the live `cascadia-pp-qwen38-20260923` owns the GPUs; single-client rule §8 for any C1 number | wait / re-schedule; never overlap |
| G1 startup | Both workers `runner ready`; rank1 `ActivationServer listening`; rank0 `/v1` health; record cold vs kernel-cache-warm startup | abort, capture logs |
| G2 correctness | Real text (not garbage/`!`/token-0) on boundary-crossing prompt; deterministic replay n=2 identical; greedy diagnostic labels only | **abort — wrong artifact (collapsed-logits class)**; re-slice from INT4-GDN8 IR |
| G3 memory | Post-load debugfs VRAM both cards (expect ≤ ~12 GB/card); host MemAvailable ≥ 3072 MiB through load — Cascadia's per-worker single-GPU processes are the P5 pattern (0.08:1), so host-RAM death here would be a *new* mechanism and closes the branch | abort per contract §5 floor rules |
| G4 performance | Matched 1-stage vs 2-stage decode: ≥1.5× = real; 1.0–1.5× = one diagnostic iteration (wire/alpha split); <1.0× = no-go + ledger closure | record, no retries beyond one |
| G5 power | **Same declared cap on both cards** — live campaign started 180 W / 230 W asymmetric (hwmon5=180, hwmon6=230): normalize or declare; energy per §10 over exact intervals; restore 150 W at teardown (§13) | not a result until normalized |
| G6 claims | `cascadia (qwen35)` engine id only; never in tables with vLLM/OV/llama.cpp rows; PROVISIONAL banner; no public headline; `vllm bench serve`-class naming rules (cold-input rate vs decode) | — |

Stop rules: any `DEVICE_LOST`/SIGSEGV → stop, capture, teardown, restore
150 W (§13, `docs/operations/incident-and-cleanup.md`).

## 7. What this does not change

- **ADR-0008 stands**: independent per-card stacks remain the serving default;
  a Cascadia PP2 campaign is "one declared engine" occupying both cards —
  allowed, but it is not a serving product and makes no 2× claim.
- **RL-016** should get its outcome recorded against the Cascadia branch
  (superseding note or new entry) once G2/G4 return — the block was earned on
  vLLM/XCCL mechanics, and the live run is the new-hypothesis validation.
- Architect made no state changes; the live campaign belongs to its operating
  session.

## 8. Evidence and code index

- Live campaign: `~/B70-DOCS/results/2026/09/cascadia-pp-qwen38-20260923/logs/`
- PP prior art: `~/B70-DOCS/results/2026/08/cascadia-dualgpu-20260824T130000Z/` (MANIFEST + 3 logs)
- Bakeoff: `~/B70-DOCS/research/2026/ov-cascadia-qwen38-bakeoff-20260911.md`
- GDN8 artifact: `~/B70-DOCS/research/2026/qwen38-ov-int4-gdn8-convert-20260912.md`
- PP2 root cause: `~/B70-DOCS/research/2026/qw38-fp8-singlecard-unblock-and-pp2-rootcause-20260827.md`
- Topology: `~/B70-DOCS/research/2026/x570f-dual-b70-pcie-topology-20260830.md`
- Cascadia source: `/mnt/<nvme-disk>/cascadia-pr132-wt/crates/cascadia-cli/src/{run_placement,placement}.rs`, `cascadia-types/src/peer.rs`, `labscommunity/cascadia: docs/SHARDING.md`, `QUICKSTART.md`; binaries `~/cascadia/cascadia-{0.2.0,0.2.3}-linux-x86_64`, `~/cascadia/cascadia-ov2631`
- Shard trees: `/mnt/<nvme-disk>/shard-qwen38-{1s-64,27b-2s,2s-*,3s-*,4s-*}`
- Flash-next state: `/mnt/<nvme-disk>/models/signal-3.8-flash-next/{status.json,loadtest-20260923/}`, pinned tree `/mnt/<models-disk>/qwenflash-llama-pinned-9723942`
- Scope: `~/B70-DOCS/docs/plans/2026-09-20-b70-improvement-roadmap/OUTCOMES.md` (SCOPE CHANGE 2026-09-23)
