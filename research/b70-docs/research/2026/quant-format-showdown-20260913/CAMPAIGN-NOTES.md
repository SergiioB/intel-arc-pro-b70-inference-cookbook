---
title: "Campaign Notes — everything observed (not just the measurements)"
type: research
date: 2026-09-13
status: final
---

# Campaign Notes — everything observed (not just the measurements)

Quant-format showdown, 2026-09-13 → 09-14, b70-host. Companion to
[FINDINGS.md](FINDINGS.md) (which carries only claims-grade material). This file
records operational findings, tooling quirks, incidents, dead ends, and
environment facts worth remembering — including things that never became a claim.

---

## 1. Headline reconciliation the morning question raised

"Is 45.72 a win over the cookbook's ~100 tok/s on 27B?" — **No, and they are
different cells.** The registered 102.61 (Run 48, `registry/model-status.json`)
was **TP2 on both cards, 230 W configured, prefix-cache-on champion stack**,
recommended sampling. Tonight's 45.72 is **one card, 150 W, cache-off, cold
entropy-first prompts**. Rough per-card arithmetic: 102.61/2 ≈ 51.3 tok/s per
card at 230 W warm vs 45.72 single card at 150 W cold — comparable but ~11 %
below per-card throughput at 65 % of the power budget, `directional_only` (TP2
collective overhead, cache state, cap, and statistic all differ). The +87 %
multiplier from tonight is strictly the MTP4-vs-spec-off comparison inside the
matched single-card stack. Nothing tonight beats a registered number.

## 1a. Power-cap A/B (user-directed 230 W rerun, 2026-09-14T04:2x–04:4xZ)

User instruction: drop the 150 W guideline, rerun at 230 W if it wins. Result:

- **Prefill/cold-input wins big**: B0 p8192/g1 1039 → **1586 tok/s (+53 %)**;
  B4 p2048/g1 +38 %, B4 p8192/g1 +45 %. The 150 W cap was clipping prefill hard
  (measured window draw at 230 W: 201–202 W avg GPU0, pkg max 82 °C).
- **Decode does not move**: B0 control 24.41 → 24.25 (−0.7 %, cap-insensitive;
  launch-gap-bound, draws ~75–110 W either way). This confirms with a clean
  no-acceptance control what the August TP2 note implied.
- **B4 p512 41.07 vs 45.72 is content variance, not power**: B0 flat + mean
  acceptance ~flat (3.01 vs 3.10) ⇒ within the documented content-driven
  spec-decode swing (per-cell acceptance not isolated).
- Caps left at **230 W both cards** per explicit user instruction (GPU1 had
  reverted to 150 W during the session; re-set). The fleet-wide "restore 150 W"
  guideline in AGENTS.md/protocol was NOT edited — that's a governance change
  needing its own decision; this campaign simply closes at 230 W by instruction.
- GGUF arms were not rerun at 230 W: their decode draws are far below 150 W
  (cap-irrelevant); only llama-bench pp cells could shift, and the vLLM prefill
  result already characterizes the cap effect.

## 2. Environment facts on arrival (2026-09-13 ~19:45Z)

- Both GPUs idle: no inference processes/containers; GPU0 861 MiB used (desktop),
  GPU1 40 MiB. Power caps were **180 W / 230 W on arrival** (not the 150 W daily
  default) — reset to 150 W both before any load, restored again at close.
- hwmon nodes this boot: hwmon5 = GPU0 (`0000:0b:00.0`), hwmon6 = GPU1
  (`0000:0f:00.0`) — re-resolved dynamically, never hardcoded in new evidence.
- Root filesystem only ~26 GiB free → all artifacts/evidence routed to T-FORCE
  (358 GiB free). Final campaign artifact footprint: ~139 GB arms + 54.7 GB
  source + ~1 MB corpus.
- No F16/BF16 conversion source existed on the host (BF16 dir was a dangling
  symlink farm); GLM5 Flash / DeepSeek / Ornith weights absent; the dropped
  plan's `/models/Qwen3.8-Flash-F16.gguf`, `build-sycl-0717`, and
  `scripts/b70pack_build.py` all nonexistent.

## 3. Toolchain and environment gotchas (all hit, all worked around)

1. **HF downloads stall under xet**: first `hf download` wedged with 66 open
   sockets and zero bytes moved in 15 s (rchar flat). `HF_HUB_DISABLE_XET=1`
   gave a steady 9–11 MiB/s; ~54 GB landed in ~90 min. hf_transfer is installed
   but was not needed after the xet disable.
2. **oneAPI env is mandatory** for llama.cpp SYCL binaries — bare invocation
   dies on `libsvml.so`. Additionally `source setvars.sh` **aborts under
   `set -u`** (unbound vars) — wrap with `set +u` / `set -u`.
3. **`-mmp 0` is a hard error on `llama-server`** in build 1692f9e50 but only a
   deprecation warning on `llama-bench` from the same tree. Same for the new
   `--load-mode` migration — check per binary.
4. **`llama-server /health` DOES 503 during load — the "returns 200 early" claim was
   OUR harness bug** (source-checked 2026-09-14: `middleware_server_state` in
   `tools/server/server-http.cpp` correctly blocks all endpoints pre-ready). Our
   smoke gates used `curl -s` exit status, which is 0 for any HTTP status including
   503, so they broke on the first 503. Correct gate: `curl -s -o /dev/null -w
   '%{http_code}'` == 200, or poll until a completion stops returning
   `{"message":"Loading model"}`. FINDINGS §5 carries the formal correction.
5. **Unescaped quotes in `curl -d` JSON produce server-side 400/500s** that can
   *look* like deterministic-replay PASS if you only `cmp` the two response
   bodies (two identical error JSONs). Always parse `choices` (or any expected
   schema key) — this exact bug produced a false "g0/g2 smoke PASS" that the
   audit caught. Build request payloads with `json.dumps` into a file and
   `curl -d @file`.
6. **`llama-perplexity` segfaults at startup in `build-sycl-latest-qw38`** but
   works in `build-sycl-0804` — same checkout commit (1692f9e50), different
   build dir/binary dates. All PPL numbers come from the 0804 build.
7. **vLLM V1 image counter names**: this pinned image exposes
   `vllm:prefix_cache_queries_total` / `vllm:prefix_cache_hits_total`
   (V0-style names), not the V1 `vllm:prefix_cache_*` names the protocol quotes.
   With `--no-enable-prefix-caching` the counters exist but stay 0.0 — retain
   the config line as primary cache-off evidence, counters as corroboration.
8. **Qwen3.8 defaults to thinking mode** — without
   `chat_template_kwargs: {"enable_thinking": false}` the model burns the whole
   budget on reasoning (first smoke: 120 tokens of `<think>`-style preamble,
   `finish_reason=length`, no visible JSON).
9. **Monitor CSV quirk**: the 1 Hz monitor writes a JSON temps dict containing
   unquoted commas → naive `csv.DictReader` splits columns. Energy/power parse
   by fixed column index; temps by regex over the line tail.
10. **`pip install` needs `--break-system-packages`** on this host's PEP-668
    Python; pyarrow installed to user site for the parquet corpus render.
11. **Mimosa write-hook** blocked script writes repeatedly on pattern matches
    (SSRF for URL construction, path traversal for CLI-derived filenames, weak
    randomness for seeded selection). Workarounds that passed: loopback-pinned
    endpoints with resolved-IP validation, `secrets`-based entropy, filename
    tokens validated by strict regex + `os.path.join`, and stdout-only output
    in the bench client (shell owns paths).

## 4. Measurement-semantics findings

- **Spec-decode burst arrival**: with MTP4, 128 endpoint tokens arrive in 33–49
  SSE content deltas. Any client that counts stream chunks undercounts decode
  rate by ~3.4× (13.4 vs 45.7). Endpoint `usage.completion_tokens` over the
  client decode window is the authoritative rate; also record TTFT and state
  that "45.7 tok/s" is completion throughput, not smooth streaming (chunks land
  at 13.3–13.4/s).
- **Endpoint vs client prompt tokens**: rendered prompt = target + 12 chat-template
  tokens (524/2060/8204 for 512/2048/8192). Always record endpoint counts.
- **Host-CPU contention moves GPU decode ~10 %**: the superseded B0 set
  (22.00/21.97) was measured while serial `llama-quantize` runs saturated
  CPU/disk; clean-host sets measured 24.41/23.89 and 24.32/24.21. One-engine
  discipline must include "no heavy host work during GPU measurement," not just
  "one GPU engine."
- **llama-bench pp512 sits in a launch-bound regime** (611–677) vs pp2048/pp8192
  (844–871) — short-prompt prefill numbers must not be mixed into
  bandwidth-bound statements (protocol §1.5 Error 1 confirmed on this model).
- **Power windows must bracket requests exactly**: campaign-phase averages
  (75.7/76.2/96.3 W) were idle-contaminated and produced a wrong "eager draws
  27 % more" inference; matched active windows showed ~103.5 W both modes. The
  exact-bracket confirmation window measured 146.0 W avg GPU0 (decode-heavy +
  8 k prefills, 150 W cap, pkg 73 °C).
- **MTP acceptance aggregation**: server-log windows (12) gave mean accepted
  length 2.51–3.93 (avg ≈3.10) and draft-weighted per-position ≈77/59/45/33 %.
  The initially reported "3.15 / 73-54-42-31" was not reproducible from retained
  artifacts — log-derived values only unless a `/metrics` counter diff is
  retained.

## 5. Protocol incidents and corrections (chronological)

1. **Eager misconfiguration (my error)**: first B0 launch included
   `--enforce-eager` → 10.3 tok/s. Kept as diagnostic A/B; relaunch compile-only
   recovered 2.36×. Root cause: I copied the flag from a validation-era recipe
   without checking the route-of-record.
2. **Idle-server-during-conversions slip**: a compile-only server relaunched at
   21:20:36Z and the `b0-final.jsonl` set was measured against it at 21:31Z
   under conversion contention → superseded; container torn down ~21:33Z before
   the llama.cpp arms (their successful full-VRAM loads prove no resident
   engine after that).
3. **Smoke harness quoting bug** (see §3.5) → false PASS on g0/g2 first re-run;
   voided and redone properly (g0 rerun2, gr rerun3).
4. **`-mmp` arm-driver failure** killed all four arms' smokes on first pass
   (fixed mid-chain; g2 still ran the pre-patch script because its bash process
   had already started).
5. **Metrics-counter name mismatch** returned `None` hits-deltas on the first
   cells until switched to the `_total` family (§3.7).
6. **PPL "73 chunks" vs actual 72** — corrected in FINDINGS and manifest.
7. **b70-claims-auditor agent type unavailable** in this environment (bound to
   nonexistent `grok-4.6` provider) → audit ran on a general-purpose fallback
   agent with the same strict brief. The architect agent itself worked fine all
   night (three consults: plan verdict, probe ladder, OpenVINO inspection).

## 6. Scientific observations and anomalies

- **PPL non-monotonic in precision — ROOT-CAUSED 2026-09-14 (closed)**: Q4_K_M
  6.547 < mixed 6.653 < Q8_0 6.678 < Q5_K_M 6.794 on 72 chunks; decisive follow-up:
  the **F16 reference scored 6.5545 — worse than Q4_K_M on 7/8 chunks** and ≈ Q8_0
  (dual-card split controlled by single-card Q8_0 matching it); chunk-geometry runs
  show PPL *growing* with chunk length (Q4: 6.29→6.44→7.15 at c2048/4096/8192) with
  the Q8−Q4 gap widening +0.09→+0.56. Verdict: harness/recurrent-state confounder on
  hybrid GDN — **not** a quant-kernel bug (F16 shows it with no dequant kernels).
  Upstream: [ggml-org/llama.cpp#28879](https://github.com/ggml-org/llama.cpp/issues/28879).
  Never rank quant formats with `llama-perplexity` on linear-attention hybrids;
  per-position logit parity is the tool if a quality axis is ever needed.
- **Mixed-precision rescue is a double negative as implemented** (48× attn_qkv/
  ssm_out → Q6_K, attn_gate → Q8_0, token_embd → q6_K over the Q4_K_M base):
  −5.8 % decode, +10.7 % bytes, no quality gain even on the confounded PPL
  axis. The Q4_K_M preset's own mixture is already well-allocated for this
  hybrid; "protect the GDN/attn tensors" is not free.
- **Q8_0 streams best per byte** (~396 GB/s effective vs ~336 Q4_K_M / ~332
  Q5_K_M): the trivial dequant path (single scale multiply, no super-block
  indirection) utilizes GDDR6 better — decode cost is bytes × kernel-efficiency,
  not bytes alone.
- **GGUF F16 conversion preserves the MTP `nextn` tensors** (blk.64.*) which
  llama.cpp logs as "unused tensor" warnings — benign, but a reminder that GGUF
  arms carry a dormant MTP layer these builds don't consume.
- **GDN decode on vLLM XPU uses the Triton fallback** (fused kernel requires
  CUDA CC 8.0+) — logged at startup; expected on this stack, and consistent
  with the hybrid-SSM dispatch-lag bottleneck class in the playbook.
- **GPU1 consistently idles hotter by ~0–4 °C** and both cards idle ~54–56 °C
  (display-attached GPU0 floor ~56 °C); cooldown gates to ≤55 °C worked but sat
  near the floor most of the night.

## 7. OpenVINO branch (CPU-only inspection; no GPU slot spent)

- Local `Qwen3.8-27B-int4-gdn8-ov` is actually packed INT4-asym-G128 language
  weights + **BF16/FP32 GDN** (ignored scope `.*linear_attn.*`, 11.12 GB BF16)
  + u8 LM head; its `openvino_config.json` is stale (`dtype=int8`) and
  contradicts the graph XML — the "GDN8" label is wrong and must not be read as
  8-bit GDN.
- Qwen3.8-27B (`qwen3_5`) = 48 GDN + 16 full-attention layers (not the dropped
  plan's 30/10); FP32 SSM state ~3 MiB/layer at batch 1 (~144 MiB total).
- Demonstrated runner: `openvino_genai.VLMPipeline` on the 2026.5 nightly venv;
  the historical `ov_ab.py` harness has protocol deficiencies (char≈token/4,
  requested-not-actual tokens) — its tok/s must not be reused as headlines.
- Architect NO-GO for a GPU slot tonight (feasibility already established
  09-11..13); matched OV-vs-vLLM-vs-llama.cpp comparison is the named next
  branch.

## 8. Numbers quick-reference (all from retained raw evidence)

| What | Value | Evidence |
|---|---|---|
| B0 canonical p512/g128 | 24.41 tok/s (TPOT 41.0 ms) | `vllm-b0/b0.jsonl` |
| B0 canonical p8192/g128 | 23.89 (42.0 ms) | `vllm-b0/b0.jsonl` |
| B0 confirmation (clean, post-audit) | 24.32 / 24.21 | `vllm-b0-confirm/b0c.jsonl` |
| B0 superseded (contended) | 22.00 / 21.97 | `vllm-b0/b0-final.jsonl` |
| B4 MTP4 | 45.72 / 37.11, accept ≈3.1 | `vllm-b4/b4.jsonl` |
| B0 cold input p2048/p8192 | 1048 / 1047 | `vllm-b0/b0.jsonl` |
| Eager diagnostic | 10.35 / 10.29 | `vllm-b0/b0-dec-eager-DIAGNOSTIC.jsonl` |
| Confirm power window | 146.0 W avg, pkg 73.0 °C | `vllm-b0-confirm/power-window.json` |
| GGUF tg128@p512 | 20.01 / 18.85 / 16.97 / 13.62 | `gguf-*/cellA-p512.json` |
| GGUF pp8192 | 844 / 847 / 857 / 853 | `gguf-*/cellB-p8192.json` |
| PPL (72 chunks) | 6.547 / 6.653 / 6.794 / 6.678 | `ppl-*/ppl.log` |
| Conversion times | G0 4m36s, GR 4m20s, G2 5m01s, G1 4m32s | `conversions/chain.log` |
| Download | ~54 GB @ 9–11 MiB/s after xet disable | `download-qw38-27b-bf16.log` |

## 9. What I would do differently

1. Never measure while any background job can touch the host (the conversions)
   — the one-engine gate should include a host-load check.
2. Retain the `/metrics` counter bodies, not just parsed values, for spec-decode
   runs (acceptance aggregates were not reconstructible).
3. Start from the readiness-gate-correct smoke harness and schema-parse every
   response from the first cell.
4. Write power windows per cell at launch time (markers before/after each cell,
   not each phase) so bracketing is exact by construction.
5. For quality on hybrid-GDN: logit-parity first, PPL second — PPL's
   GDN-state/chunk interaction makes it untrustworthy as a format-quality axis
   on this build until the anomaly is root-caused.

## 10. Artifact and state ledger (morning)

- Host clean: no engines/containers/monitors; both cards 150 W; VRAM gates pass
  (GPU0 32,044 / GPU1 32,615 MiB visible_avail at last check).
- Kept on T-FORCE `quant-showdown-20260913/`: `src/` BF16 (54.7 GB), `arms/`
  (F16 + G0/G2/G1/GR, ~139 GB total), `corpus-src/` + `wiki.test.raw`.
  Deletion is a user decision.
- Documents: `research/2026/quant-format-showdown-20260913/` — PLAN.md,
  FINDINGS.md (audit-corrected), CAMPAIGN-NOTES.md (this file),
  X-POST-DRAFT.md (audit-compliant, **unposted** — needs explicit user
  authorization), ARCHITECT-VERDICT.md.
- Evidence root: `results/2026/09/qw38-quant-format-showdown-20260913/` (manifest
  updated with canonical results; audit report retained by the auditor agent in
  its output file, key findings folded into FINDINGS §3/§9).
- Nothing committed to git; no registry/model-status/public-catalog updates
  made (separate authorization + publication workflow required).
