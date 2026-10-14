---
title: "Night Report — Qwen3.8-27B improvement night 2026-09-25"
type: research
date: 2026-09-25
status: living
---

# NIGHT REPORT 2026-09-25 — Qwen3.8-27B recipe improvement night

**Honest headline.** The realistic total for tonight was **resolved questions and a clean
win/loss record, not tok/s: zero promotions, champion recipe unchanged.** What was achieved:
one measured screen verdict (host-prep cache LOSS), one closed analysis line (kernel hunting
closed for this stack), one applicability kill (#56096 not applicable to this route), one
non-lever screen (prefill knob), five new ledger rows (RL-061…RL-065). What was NOT achieved:
any C1 speedup, protocol-≤55 °C-cooled evidence (thermal substitution used, disclosed), and a
Level Zero verify-phase breakdown (unitrace tooling blocked). Everything below is
**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** No X/post draft exists because no promotion
happened.

## Thermal substitution (applies to every number tonight)

The protocol ≤55 °C package window is unreachable on this host while a champion server is
loaded: the loaded-idle floor rose 58 → 69 °C over the night at the same ~46 W idle draw (fan
836–1035 RPM floor; heat-soaked chassis), and empty-idle does not recover below the gate
(confirms the RL-057 wall). Substitution used and recorded in every results root: arms start at
the achievable floor, matched ±1 °C, counterbalanced order, per-block temperature + energy
telemetry, 80 °C heat-safety line. The champion's own reference controls (77.14 / 73.84 tok/s,
2026-09-24 attribution) ran at 57–72 °C. Tonight's matched-arm A/B deltas are internally
comparable; none of it is protocol-1.3-cooled or public-grade (RL-065).

## Phase-by-phase

### Phase 0 — Preflight: DONE

Host `b70-host`, GPU 1 (0000:0f:00.0) idle. Numbers re-derived independently from
`p1-qwen27-attribution-20260924T200156Z/`: control medians 77.14 / 73.84 tok/s verified from raw
logs; 44.8 ms round = 1.6594 s post-first ÷ 37 rounds; host-prep share f ≈ **0.109** (named
mechanism ops 180.8 ms CPU ÷ 37 rounds = 4.89 ms) vs the candidate's 0.107 — consistent. Noted
an internal inconsistency in the candidate source (its component table sums to 51.45 ms / 102.2 %
vs its stated 50.34 ms total). The one-line hypothesis was confirmed **verbatim** in the pinned
source: `gdn_attn.py` `build()` at line 210 (both copies), steady-state branch does exactly
`torch.arange(5)`, `torch.empty(0)`, `async_tensor_h2d(tensor([True]))`, and boolean-mask
advanced indexing on `block_table_tensor` every round. Runtime resolution verified: the
`vllm serve` console script imports the **site-packages** copy (the boundary-patched one);
`/workspace/vllm` shadows it only for cwd-imports from the image WORKDIR.

### Phase 1 — Host-prep metadata cache: **LOSS** (screen gate failed)

Patch `B70_HOSTPREP_CACHE` (fail-closed anchor splice, one variable, both `gdn_attn.py` forms
handled — pristine + `patch_mtp_boundary` post-splice; sha256 recorded per file:
`fda86b96…→1c5d9c98…`, `13579992…→1070b379…`; fast-path engagement logged at runtime). In-session
A/B via sentinel toggle; ABBA p512 + BAAB p8192; n=3/block, pooled n=6/arm/anchor; exact rendered
prompts (512 and 8192 tokens, `len(enc["input_ids"])` calibration, live-probe verified); one
discarded same-shape warmup per block; prefix-cache counters zero-hit (cache off);
spec-decode counters bracketed; 230 W cap, measured 226.5–229.3 W both arms (iso-power).

| Cell (client post-first tok/s, pooled medians, n=6/arm) | A = patched | B = control | Δ | Gate |
|---|---:|---:|---:|---|
| p512/g128 (512 tok exact) | 78.18 | 77.02 | **+1.51 %** | ≥ +4 % → **FAIL** |
| p8192/g128 (8192 tok exact) | 72.92 | 70.61 | +3.27 % | regression ≤ 1 % → pass |

Per-block medians (acceptance-luck dispersion is ±10 %/sample under recommended sampling):
p512 A 76.30 / 82.32, B 72.24 / 81.80; p8192 A 74.76 / 71.07, B 68.75 / 72.46.

Correctness (clean): spec-canary identical in both arms (`1c650a7f…`, deterministic — #54785
class not live); greedy byte-identity A=B at both anchors (`28aea8a7…`, `46a188e9…`); the patch
is output-preserving. Gate verdict per the pre-declared rule: **LOSS — no finalist, not adopted.**
The patch never entered the champion recipe (startup splice in test containers only), so nothing
needed reverting. Diagnosis (time-boxed): the 10.08 ms instrumented host-prep share was
profiler-inflated; the measured +1.51 % implies a removable share of ~1.5–3 %, consistent with
the 4.9 ms/round non-instrumented upper bound being mostly non-removable dispatch cost. The
candidate's own stop clause applies: **the measured share shrank; stop.** Evidence:
`results/2026/09/b70-improvement-probes-20260920/p1-qwen27-hostprep-screen-20260925T082440Z/`.

### Phase 2 — Verify-phase composition: **BLOCKED + substitution → closed line**

unitrace (the sanctioned Level Zero tracer) is not installed and its source build was killed
twice by the tool deadline (tooling block recorded, RL-062). Sanctioned fallback executed: one
eager-mode instrumented run labelled **"eager non-champion hypothesis generation — cannot
establish champion headroom"** (graphs off + `--enforce-eager`, both cells, controls n=2
bracketing profiled requests). Eager controls 24.7–30.6 tok/s vs champion ≈73–78 (graph capture
is ≈2.6× of the stack) — ranking only, never a speed claim.

Per-bucket champion decode table (44.8 ms round) with labels:

| Bucket | ms/round | share | label |
|---|---:|---:|---|
| Host control-basis prep | ≈4.8 | 10.7 % | measured-derived (Phase 1 cross-check 4.89) |
| 4 draft passes + quantized draft head/body (outside graph) | ≈5.5 | 12.3 % | measured |
| Unquantized full-vocab BF16 target head | ≈4.2–4.7 | 9.4–10.5 % | model-derived (bytes ÷ 538–608 GB/s) |
| Target W4A16 body verify GEMMs (M=5) | ≈28–30 | 63–67 % | UNRESOLVED under graph capture |
| GDN/KV/norms/sampler | ≈1.5–2.5 | 3–6 % | model-derived from eager shares |
| Bookkeeping/sync | ≈0.5 | 1.1 % | measured-derived |

Re-priced levers vs the pre-declared ≥5 % Amdahl gate: **L-B fails** — 3.39 GB modeled draft [CORRECTED 2026-09-29, register row 4: the post-pair budget is **3.49814784 GB/round** = (head 0.6555648 + body 0.21897216 GB) × 4 passes incl. fp16 g128 scales; 3.39 GB was payload-only — see the pin-rate correction in the block below]
traffic inside the measured 5.5 ms bucket implies ≥616 GB/s effective, at/above the card's
bandwidth ceiling, so the "≈2.5 GB redundant re-reads" premise is inconsistent with the measured
bucket (s ≥ 1.74 unsupported). **L-C fails** — the head is a pure weight-read GEMM at its bytes
floor (s ≥ 1.9 implausible, and RL-060 forbids commissioning without unitrace evidence).
**Verdict written to OUTCOMES: kernel hunting closed for this stack.** Sole reopening condition
for all three lines: a working unitrace (or equivalent) trace of one graph-captured champion
round. Evidence: `p2-qwen27-eager-composition-20260925T085704Z/RESULTS.md`.

> **Correction — 2026-09-25 (coordinator review).** The L-B *verdict* (closed) stands, but the
> stated mechanism above is backwards and would mislead a future reader. "616 GB/s ≥ ceiling ⇒
> re-reads not happening" is the wrong reading of the same arithmetic: 3.39 GB in 5.5 ms is within
> ~1.3 % of the 4×-read bytes floor (3.39 GB ÷ 608 GB/s = 5.57 ms), i.e. the draft bucket **is
> streaming all four passes at essentially the pin rate — the re-reads ARE happening**, at full
> [CORRECTED 2026-09-29, register rows 4/13: the post-pair budget is 3.498 GB/round incl. fp16 g128
> scales, which at 5.5 ms implies ~636 GB/s — slightly ABOVE the 608 class; "essentially the pin
> rate" is model arithmetic that self-agrees, **not a measured DRAM-traffic fact**. The 4×-read
> structure claim stands as a model bound; no DRAM-counter measurement exists. The closure (L-B
> closed) is unaffected.]
> efficiency. They are also *physically unavoidable*: the four draft positions are sequentially
> dependent (each draft token conditions the next pass), and 0.848 GB of weights is orders of
> magnitude beyond on-chip capacity, so a weight-stationary fusion cannot exist for this shape.
> Correct conclusion: the draft bucket has **no removable redundancy because it is already at its
> unavoidable-bytes floor**, not because the traffic is absent. Same closure, sounder mechanism.
> Corollary for expectations: summing the floor-bound buckets (draft 5.5 + head ~4.5 + body floor
> 15.2 GB ÷ 540–608 ≈ 25–28 + misc ~2) accounts for ~40+ of the 44.8 ms round, and Phase 1's
> +1.51 % for halving host prep shows the 4.8 ms host bucket is largely overlapped — so the
> single-card round is GPU-byte-bound end to end, and the unitrace reopening condition can at most
> surface ~2–4 ms of slack. Run it to close the record; do not expect a lever.

### Phase 3 — #56096 dispatch: **NOT APPLICABLE** (zero-GPU, screen correctly skipped)

`use_fused_qk_norm_rope_gate` exists only in `qwen3_next.py:322` and requires
`current_platform.is_cuda()`; our artifact resolves `Qwen3_5ForConditionalGeneration` →
`qwen3_5.py` (no such dispatch). Fails on model class and platform → the "all conditions hold"
gate fails → no screen. Corrects RL-054's "equivalent merged as #56096" note (it is
Qwen3-Next/CUDA only). Evidence: `P3-56096-DISPATCH-GATE-20260925.md` in the probes root.

### Phase 4 — Prefill knob: **SCREENED — non-lever**

Static check: genuinely unswept (ledger/campaigns/P7 hold design notes only). Bounded screen:
fresh session per `--max-num-batched-tokens` config, paired distinct entropy-first exact-8192-token
prompts n=3, one discarded warmup/session, counter + energy brackets, cleanup between passes.

| max-num-batched-tokens | TTFT median (p8192/g1, n=3) | Δ vs current |
|---:|---:|---:|
| 8192 (current) | 4.679 s | — |
| 4096 | 4.689 s | +0.2 % |
| 16384 | 4.683 s | +0.1 % |

±0.25 % spread — **no lever at this shape**; the current 8192 setting is retained; no adoption
needed (which also avoids the fresh-n=5 requirement). Evidence:
`p4-qwen27-prefill-knob-screen-20260925T093105Z/`.

### Phase 5 — Optional CCL probe: **SKIPPED** (eligibility condition not met — Phase 1 was a LOSS)

### Phase 6 — Close-out: DONE

Caps restored 150 W on both cards; no inference processes/containers; GPU 1 empty (32,615 MiB).
Accretion: ledger RL-061…RL-065 (JSON + `research/LEDGER.md` mirror), OUTCOMES L16–L19,
`registry/model-status.json` + `models/STATUS.md` open-gate updates, campaign
`CAM-2026-QWEN27-IMPROVEMENT-NIGHT-0925` registered with record
`campaigns/2026/qwen27-improvement-night-20260925/README.md`, `tools/build-indexes.py` (411 docs),
`check-frontmatter.py` clean, `check-links.py` shows one **pre-existing** placeholder break
(`campaigns/2026/qwen38-tp2-vs-2xtp1/README.md` references a results root "reserved, not yet
created" — left untouched). Nothing left the machine: no commit, push, publish, or social.

## What changed in the champion recipe

**Nothing.** Retained unchanged: `f01e24f6` image + 3-patch overlay, GPTQ-Int4-sym-G128 `-nogidx`
repack, stock MTP4 + greedy drafting, draft-only INT4 body/head pair, FP8 KV, XPU graph capture,
recommended non-thinking sampling, `--max-num-batched-tokens 8192`, single-card placement.

## Closed forever (unless a named condition changes)

1. **Host-prep metadata caching** — measured +1.51 % vs +4 % gate; unchanged reruns banned
   (RL-061).
2. **L-B draft fusion / L-C head kernel on current evidence** — fail the Amdahl gate on measured
   shares; reopening needs unitrace traffic/duration evidence (RL-062 + RL-060).
3. **#53989/#56096 fused QK-RoPE-gate on the qwen3_5 XPU route** — not applicable; reopening
   needs an upstream qwen3_5 XPU implementation (RL-063).
4. **max-num-batched-tokens C1 TTFT sweep** — three-way non-lever (RL-064).
5. **Tonight's numbers as public/protocol-cooled evidence** — thermal substitution stands
   (RL-065).

## Tomorrow's single best next action

**Get one unitrace (pti-gpu) trace of a single graph-captured champion decode round.** It is the
sole named reopening condition for the entire decode-kernel line (the 63–67 % body-GEMM bucket),
and it unblocks L-B/L-C re-pricing on real traffic/duration data. Practical route: finish the
`~/pti-gpu-build` source build in a long-timeout job (or bake unitrace into a derived
image), then wrap one champion `vllm serve` launch and confirm graph-replayed kernels are visible.
Runner-up (operator decision, not agent-actionable tonight): chassis/ambient intervention to
restore the ≤55 °C protocol window so future promotions are public-grade.

## Infrastructure notes (for future sessions)

- The host-prep screen needed four attempts: a calibration-container mount bug (fixed), the
  thermal wall ×2 (substituted), and a `wait_cool` stall against a rising floor (capped at 120 s).
  The two aborted roots are retained (`…T062604Z`, `…T063556Z`).
- `hp-calibrate.py` initially hit the protocol §11.1 trap (`len(BatchEncoding)` = 2 fields);
  fixed to `len(enc["input_ids"])`, then exact-fit in 4–8 iterations at both anchors.
- The async job tool's 3600 s deadline killed two long builds/runs (SIGKILL, no trap); run
  long builds detached with an explicit long timeout or bake them into images.
- The `todo` tool twice marked all items done on a single `op:done` (reported via xd://report_issue).
