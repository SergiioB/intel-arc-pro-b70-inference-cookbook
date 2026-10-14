---
title: "Flash-Next on Cascadia pipeline parallelism — feasibility verdict: blocked at three independent layers"
type: worklog
date: 2026-09-23
status: blocked-no-artifacts-no-arch-support
---

# Flash-Next Cascadia PP feasibility — 2026-09-23

> PROVISIONAL — NOT FOR PUBLIC HEADLINE. Read-only investigation. NO GPU work
> was performed in this lane (GPUs were held for the concurrent Qwen3.8-27B
> Cascadia PP smoke; gate verified clean before and after the earlier
> llama.cpp attempts — no Flash-Next GPU run happened).

## Verdict

**Cascadia 0.2.3 cannot consume Signal-3.8-Flash-Next today.** The `qwen35`
surgery runtime (the only Flash-Next-adjacent engine in Cascadia) cannot load
it, and there is no artifact it could load even if it could. A 2-stage smoke
is not possible without new upstream work in three places at once.

## Exact blockers (three independent layers)

1. **No consumable artifact format.** `cascadia shard --model` accepts only a
   HF repo id / local dir with **safetensors + config.json**, or an
   already-exported OpenVINO IR dir (Gemma-4 / Qwen3.5-family surgery paths).
   The only Flash-Next artifacts on disk or on the Hub are **GGUF**:
   `agentionai/Signal-3.8-Flash-Next-GGUF` ships `AP-Q4_K_XL`,
   `AP-IQ4_XS`, `Q8_0`, `mmproj-F16` — no safetensors. The base repo
   `agentionai/Signal-3.8-Flash-Next` answers `Invalid username or password`
   (gated/private) — no accessible safetensors source exists.
   `cascadia shard` cannot start; "can the HF model be sharded" → **no, there
   is no shoppable HF source, only GGUF**.
2. **No qwen4exp architecture support in Cascadia.** `grep -ri qwen4exp` over
   the full source worktree `/mnt/<nvme-disk>/cascadia-pr132-wt/` returns
   zero hits. `cascadia shard`'s model-specific dispatch covers **gemma-4**
   and the **Qwen3.5-family (3.6 MoE, 3.8 dense)** only; qwen4exp is a
   different architecture (hybrid SSM `ssm_*` + per-layer-token-embedding
   `ple_*` + DSA-style `indexer_*` + hyper-connections `hc_*`, 512-expert
   MoE) — outside both exporters. The wrapped runtime (OpenVINO GenAI) has no
   qwen4exp model class either, so even a manual IR export would have no
   loader.
3. **Even the supported path is not a speed path on B70.** Engine dossier
   (`engines/cascadia/README.md`): qwen35-runtime Qwen3.8-27B INT4-GDN8 =
   4.95 tok/s P512/G128 (~102 W) — ~16× behind OpenVINO GenAI+MTP5 and ~15×
   behind vLLM GPTQ-INT4+MTP4 on this host; the `ov-genai` engine additionally
   SIGSEGVs (stock 0.2.3) or dies in `Tokenizer::setup_tokenizer` (2026.3.1
   rebuild) on the VLM IR class.

## PP contract as verified (for the record)

- `cascadia shard --model <safetensors-dir> -o <shard-tree> --num-stages N
  [--quantization int4|int8|fp16] [--target cpu-gpu|npu]`.
- `cascadia worker --rank R --total N --model <shard-tree>`; upstream relay
  binds `--listen` (default `:9100`); `--layer-start/--layer-end` asymmetric
  layer split exists but is documented **MiniMax-M2 sparse-moe only**.
- Prior dual-B70 PP evidence (coordinator-confirmed): Qwen2.5-1.5B INT4
  2-stage 70.0 tok/s vs 1-stage 85.8 tok/s (−18 % small-model tax) —
  evidence under `/mnt/<nvme-disk>/cascadia-lab/`
  (`conformance-20260922/`, `elastic-inference-20260922/`, models at
  `/mnt/<models-disk>/cascadia-models/qwen25-1.5b-{1,2}stage`). Not comparable to
  Flash-Next.

## Safe-route conclusion

- No safe 2-stage Flash-Next smoke is possible from existing artifacts.
- Revisit only if ALL of: (a) an accessible safetensors Flash-Next source
  appears, (b) `cascadia shard`/GenAI gain a qwen4exp export + model class,
  (c) someone proves the qwen35 runtime class on hybrid-SSM tensors.
- Meanwhile the only working split path for Flash-Next on this host remains
  llama.cpp dual-card `-sm layer` (currently blocked by its own
  runtime-mismatch rebuild — see `FLASHNEXT-qwen4exp-fix-20260923.md`).

## Evidence index

- Cascadia binary: `~/cascadia/cascadia-0.2.3-linux-x86_64/cascadia`
  (`--version` → 0.2.3; `shard --help`, `worker --help` captured 2026-09-23).
- Source worktree grep: `/mnt/<nvme-disk>/cascadia-pr132-wt/` — no
  `qwen4exp`/`qwen4_exp` hits.
- HF probes (read-only API GET): `agentionai/Signal-3.8-Flash-Next-GGUF`
  file list (GGUF-only); `agentionai/Signal-3.8-Flash-Next` auth-gated.
- Engine dossier: `~/B70-DOCS/engines/cascadia/README.md`.
