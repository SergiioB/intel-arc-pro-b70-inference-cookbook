---
title: "Cascadia PP three-model audit — qwen38-27b shard validity, PP evidence, MiMo-V2.6 / qwen4exp support"
type: worklog
date: 2026-09-23
status: final
intent: research
scope: read-only audit + this worklog only (no source writes, no git changes, no GPU runs)
models: [Qwen3.8-27B, qwen3.8-flash-next, MiMo-V2.6-Flash-RL]
engine: cascadia 0.2.3
deepseek: retired — deliberately not touched
---

# Cascadia PP three-model audit — 2026-09-23

Question: with TP2 blocked on the no-P2P dual-B70 X570-F host (ADR-0008),
can Cascadia pipeline parallelism (PP2, one stage per card) carry
Qwen3.8-27B, and does it retain speed? Plus: is the existing 2-stage
shard valid for the installed 0.2.3 binary, and does 0.2.3 support
MiMo-V2.6-Flash-RL / qwen3.8-flash-next at all?

## 1. Installed binary and what it supports

- Binary: `~/cascadia/cascadia-0.2.3-linux-x86_64/cascadia`,
  `--version` → `cascadia 0.2.3`. `cascadia engines` lists 7 engines;
  `cascadia doctor` is reported (tasking) to see GPU.0/GPU.1.
- Engine relevant here: `qwen35` (alias `qwen36-moe`) — per the binary's
  own help: "Runs the IR-surgery shard chain (`tools/qwen36_surgery/
  export_qwen36_moe.py` output dir with manifest.json) in-process for
  both `qwen3_5_moe` (Qwen3.5/3.6-35B-A3B) and dense `qwen3_5`
  (Qwen3.8-27B); greedy-only, batch=1".
- Arch-support check by string count on the installed binary
  (`strings cascadia | grep -c`): `qwen3_5` 47, `qwen3_5_moe` 19,
  `gemma4_cached_v1` 4, `minimax`/`MiniMax` 16 (sparse-moe CPU engine),
  **`qwen4exp` 0, `qwen4` 0, `qwen3_6` 0, `mimo`/`MiMo` 0**.
- Upstream v0.2.3 source confirms layout + multi-rank semantics
  (`crates/cascadia-engine-openvino/src/qwen36.rs`, tag v0.2.3,
  https://github.com/labscommunity/cascadia): manifest key check is
  `arch.starts_with("qwen3_5")`; stage dirs are `stage{N}` (no
  underscore) each holding `stage.xml` (+ `.bin` weights); tokenizer is
  loaded from shard-root `tokenizer.json` on rank 0; root
  `openvino_text_embeddings_model.xml` is compiled on rank 0 and its
  output width is validated against manifest `hidden_size`; pipeline
  mode requires `--total` == `manifest.stages.len()`; each rank compiles
  only its own stage; ranks handshake (HELLO carries the full manifest
  text + `proto: 1`; any mismatch poisons the pipeline). The "requires
  v5 shards" message belongs to the **ov-dist-spec** engine, not qwen35
  — no version field is needed in a qwen35 manifest.

## 2. Existing shard artifact — VALID for 0.2.3 (structurally), one numeric caveat

Artifact: `/mnt/<nvme-disk>/shard-qwen38-27b-2s`
(15 GB total: stage0 5.9 GB, stage1 7.7 GB, embeddings 1.27 GB,
tokenizer.json 20 MB, config/generation/manifest).

Checked against the v0.2.3 loader contract:

| Requirement (qwen36.rs v0.2.3) | Artifact | Verdict |
|---|---|---|
| root `manifest.json` | present (2026-09-19 13:11) | PASS |
| `arch` starts with `qwen3_5` | `"arch": "qwen3_5"` | PASS |
| `hidden_size` present | 5120 | PASS |
| `stages` chained in order, count == `--total` | 2 stages: [0,32) has_embed/no-head, [32,64) has_head/`stage_hidden` input | PASS |
| `stage{N}/stage.xml` | `stage0/stage.xml`, `stage1/stage.xml` (+`.bin`) | PASS |
| root `tokenizer.json` | present | PASS |
| root `openvino_text_embeddings_model.xml` | present (+ 1.27 GB `.bin`) | PASS |

Provenance: `manifest.source = "Qwen3.8-27B-int4-ov"`; the shard's
`config.json` and `generation_config.json` are JSON-identical to
`/mnt/<nvme-disk>/models-ov/Qwen3.8-27B-int4-ov/`
(VLM layout, `Qwen3_5ForConditionalGeneration`, 64 layers, bf16, MTP IR
present); that IR dir is mtimed 2026-09-19 05:40, same session as the
shard export (05:20–05:43, manifest re-written 13:11). Sibling shard
variants from the same session exist and are unregistered:
`shard-qwen38-1s-64`, `shard-qwen38-27b-2s-48-16`, `-56-8`,
`shard-qwen38-2s-60-4`, `-63-1`, `shard-qwen38-3s-48-8-8`,
`shard-qwen38-4s-24-24-8-8`, `shard-qwen38-4s-40-8-8-8` (all under
`/mnt/<nvme-disk>/`). None is referenced anywhere in
B70-DOCS (grep across repo: zero hits) — **unregistered evidence**;
whatever was run with them on Sep 19 left no result bundle in
`results/`.

**Numeric caveat (must gate before any benchmark):** the source IR's
`openvino_config.json` reads `dtype: int4_int4`, `ignored_scope: null`,
INT4 `ratio 1.0`, `sym: false` — the exact signature the 2026-09-11
bakeoff flagged when the official INT4 IR emitted collapsed logits
("every token id 0", `research/2026/ov-cascadia-qwen38-bakeoff-20260911.md`
§"INT4 night"). The bakeoff's *working* Cascadia artifact was the
INT4-GDN8 surgery (GDN kept out of INT4). The shard may therefore load
cleanly and still produce collapsed output. A HTTP smoke + text-quality
check is mandatory before spending benchmark cells; if collapsed,
re-shard from a gdn8-class IR.

## 3. How to launch PP2 on this host (local, one stage per GPU)

Ports/topology verified against the 2026-08-24 run logs
(`results/2026/08/cascadia-dualgpu-20260824T130000Z/worker-rank{0,1}.log`):
rank 1 binds the relay `:9100` (its `--listen` default) and rank 0 dials
it via `--next`; only rank 0 serves `--api`. `run-placement` would put
stage 1 on `relay_base+1` (9101), but the manual two-worker form is the
documented path.

```bash
# 0. gates per AGENTS.md §4: no inference processes/containers,
#    both cards >= ~31,000 MiB visible_avail; confirm ZE device order
#    (sycl-ls) and hwmon nodes; 150 W baseline
CASC=~/cascadia/cascadia-0.2.3-linux-x86_64/cascadia
SHARD=/mnt/<nvme-disk>/shard-qwen38-27b-2s

# 1. downstream (last stage) FIRST on GPU 1 — listens :9100
$CASC worker --rank 1 --total 2 --engine qwen35 \
  --device GPU.1 --model $SHARD \
  > /tmp/cascadia-pp2-rank1.log 2>&1 &

# 2. head (rank 0) on GPU 0 — API :8000, dials 127.0.0.1:9100
$CASC worker --rank 0 --total 2 --engine qwen35 \
  --device GPU.0 --model $SHARD \
  --next 127.0.0.1:9100 --api 127.0.0.1:8000 \
  --served-model-name qwen38-27b-pp2 \
  > /tmp/cascadia-pp2-rank0.log 2>&1 &

# 3. smoke (MANDATORY before benchmarking — see §2 numeric caveat):
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-27b-pp2","messages":[{"role":"user","content":"Name the capital of France."}],"max_tokens":32}'
```

Notes:

- No special env required; OpenVINO selects the card via
  `GPU.0`/`GPU.1` (Level Zero order — confirm with `sycl-ls` per AGENTS
  §3 before trusting indices). Optional hard pin: launch rank 0 under
  `ZE_AFFINITY_MASK=0` and rank 1 under `ZE_AFFINITY_MASK=1`, then use
  `--device GPU` — only if indexed devices misorder.
- `--ov-*` flags (`--ov-performance-mode LATENCY`,
  `--ov-inference-precision f16`, …) are **ignored by qwen35** in stock
  0.2.3 ("compiles with a fixed plugin config") — do not expect them to
  change anything (also recorded in
  `engines/cascadia/README.md` and the cookbook page).
- `--prefix-cache-gb` is auto-disabled for `--total > 1` (single-process
  only) — so the usual HTTP cold-input cache-hit guard should read zero
  by construction; snapshot server logs anyway.
- `--layer-start/--layer-end` are MiniMax-M2 sparse-moe only; qwen35
  takes the split from `manifest.json`. `--total` must equal 2 exactly.
- Estimate per card: stage weights 5.9 / 7.7 GB + compile buffers +
  rank-0 embeddings 1.27 GB → well inside 32 GB; still run the §5
  post-load VRAM gate before requesting work.
- Cleanup: SIGTERM both workers, re-run §4 gates, restore 150 W, retain
  both worker logs + curl evidence (§12/§13).

## 4. Does PP retain speed vs single GPU? — NOT PROVEN; every existing comparable shows PP slower

- **Only host comparable (and it is a different, small model):**
  Qwen2.5-1.5B-Instruct INT4 2-stage, Cascadia **0.2.0**, 2026-08-24 —
  `results/2026/08/cascadia-dualgpu-20260824T130000Z/MANIFEST.md`:
  "pipeline median **70.0** e2e tok/s vs control **85.8** (**-18%**
  small-model tax)", PROVISIONAL. Worker log warmed decode cells:
  control 1-stage 93.6 / 91.7 / 88.0 tok/s
  (`run-1stage.log` lines 19–25) vs PP2 81.8 / 81.9 tok/s with
  `wire_ms` ≈ 547–562 per 81-token task (`worker-rank0.log` lines
  19–27).
- **Upstream comparable (MoE, different hardware):** Lunar Lake
  Qwen3.6-35B-A3B staged engine: 4.7–8.8 tok/s single-box vs
  **4.1–4.8 tok/s in 2-node pipeline mode**
  (`/mnt/<nvme-disk>/cascadia-pr132-wt/docs/architectures/qwen36-moe-support.md`,
  header). Same direction: PP does not retain speed there either.
- **Qwen3.8-27B: no PP2-vs-1stage measurement exists anywhere on this
  host.** The 2026-09-12 bakeoff lists "Dual-card `qwen35` stages" as
  an open gate (`B70-DOCS research: research/2026/ov-cascadia-qwen38-bakeoff-20260911.md`
  §Open gates). The single-GPU qwen35 baseline to beat, same artifact
  class, n=5, 230 W cap (~102 W actual draw), quality PASS:
  **p512/g128 median 4.95 wall tok/s, p128/g128 4.97**
  (`results/2026/09/ov-cascadia-qwen38-20260911/n5-cascadia-int4-gdn8-230w.json`;
  same-IR OpenVINO GenAI 2026.5 control: 15.79 / 15.14).
- Physics note (why it might do better than −18% at 27B): the relay
  carries only `stage_hidden` (5120-d fp16 ≈ 10 KB/token over loopback)
  plus a position frame, so wire bytes are negligible; the −18% at 1.5B
  is dominated by per-token pipeline serialization (two GPU round-trips
  + lockstep 12-byte-header frame protocol). At 27B each stage does far
  more work per token, so the fixed per-token overhead amortizes — but
  that is an argument to measure, not evidence. Verdict for the claim
  "PP retains speed": **unproven; existing cells say the opposite;
  any PP2 number must be produced under the protocol below.**

## 5. Required benchmark protocol for a PP2 decision (if run)

Per `docs/methods/benchmark-protocol.md` (normative), AGENTS §8, and
`.agents/skills/benchmark-model/SKILL.md`:

1. Empty-GPU gate (§4) + post-load VRAM gate (§5) + one engine, one
   harness, one client (single-client gate; strictly serial clients).
2. **Iso-artifact A/B:** Cascadia `qwen35` `--total 1` (same shard
   tree, one process, `--device GPU.0`) vs PP2 `--total 2` — same IR,
   same cap, same day. Never compare against vLLM/llama.cpp rows.
3. Cells: matched p512/g128 and p8192/g128, n=3 screen, then fresh n=5
   finalist; 1 same-shape warmup discarded; cool to ≤52–55 °C between
   cells; milestones around load/warmup/cell/cleanup.
4. **All qwen35 cells are greedy, batch=1 → label every number "greedy
   synthetic" diagnostic; never publishable as representative** (AGENTS
   §8 sampling policy).
5. Acceptance is the declared objective, e.g. "PP2 retains speed" =
   PP2 median ≥ ~95% of 1-stage median on matched cells (within n=5
   spread); a win needs the declared margin, not a single best sample.
6. Evidence bundle under `results/2026/09/…` per §12 (manifest, raw
   samples, both worker logs, power/telemetry, cleanup state), and the
   claims standard skill before any external sentence.

## 6. MiMo-V2.6-Flash-RL and qwen3.8-flash-next on Cascadia 0.2.3 — NOT SUPPORTED

- **MiMo-V2.6-Flash-RL**: local HF snapshot
  `/mnt/<sata-disk>/models/MiMo-V2.6-Flash-RL/` (download
  paused; also `~/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-RL`)
  declares `model_type: "mimo_v2"`, `architectures:
  ["MiMoV2ForCausalLM"]`, 48 layers, hidden 4096, separate
  `model_mtp.safetensors`, HF-native `model_pp0_ep*` shards. The
  installed binary contains **zero** `mimo`/`MiMo` strings; no
  `mimo_v2` dispatch exists in the shard exporter or any engine (the
  qwen35 surgery path is `qwen3_5*`-only; sparse-moe is Kimi-K2.6 /
  MiniMax-M2 shaped). Serving it would require a new upstream surgery
  runtime + exporter (the generic ov-runtime trace path assumes plain
  KV attention and would face the hybrid-linear-state + MTP-head
  problem) — not available in 0.2.3. Same conclusion for its predecessor
  v2.5 was never relevant here; **v2.6 is the only MiMo in scope and it
  is a no-go on Cascadia today.**
- **qwen3.8-flash-next** (llama.cpp `unknown arch qwen4exp`; local
  artifact is GGUF-only at
  `/mnt/<nvme-disk>/qwen38-flash-next-ad-4.27bpw-q4-k-m-m64/` with
  MTP draft `/mnt/<nvme-disk>/mtp-draft/Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf`):
  the binary contains **zero** `qwen4exp`/`qwen4` strings; no OpenVINO
  IR export exists for it (Cascadia consumes OV IR, not GGUF), and the
  embedded exporter would reject the unknown `model_type`. No-go; not
  even a shard can be produced with 0.2.3.
- Newer local Cascadia source exists
  (`/mnt/<nvme-disk>/cascadia-pr132-wt`, version 0.1.7
  "elastic" branch, plus `cascadia-lab/*20260922` experiments on
  qwen25-1.5b) — none of it adds mimo/qwen4exp support, and it is older
  than the installed 0.2.3; not a path for either model.

## 7. Bottom line

1. `shard-qwen38-27b-2s` **is structurally valid** for the installed
   Cascadia 0.2.3 (`qwen35` engine, manifest/stage/tokenizer/embeddings
   layout all match the v0.2.3 loader). Launch recipe in §3.
2. Numeric validity is **unverified and at known risk** (source IR
   carries the INT4 signature that produced collapsed logits on
   2026-09-11); smoke-text gate is mandatory before benchmarking.
3. "PP retains speed" is **not proven anywhere**; the only host
   comparable (Qwen2.5-1.5B, 0.2.0) measured **−18%** (70.0 vs 85.8
   e2e tok/s median), and upstream measured 2-node slower than
   single-box for Qwen3.6 MoE. A 27B PP2 decision needs the §5
   iso-artifact protocol against the 4.95 tok/s p512/g128 baseline.
4. MiMo-V2.6-Flash-RL (`mimo_v2`) and qwen3.8-flash-next (`qwen4exp`)
   are **unsupported** by Cascadia 0.2.3 (0 binary hits, no exporter,
   no surgery runtime).
5. The Sep-19 shard family and any runs against them are unregistered
   evidence — if the PP2 experiment runs, register the campaign +
   results per §12 before any claim leaves PROVISIONAL.
