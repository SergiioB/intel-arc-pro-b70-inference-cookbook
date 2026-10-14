---
title: "Draft-side INT4 patch-generation provenance record"
type: research
date: 2026-09-22
status: living
---

---
doc_id: RES-QW38-DRAFT-INT4-PATCH-PROVENANCE
doc_type: research_note
title: Draft-side INT4 patch-generation provenance record
lifecycle: active
created: 2026-09-22
updated: 2026-09-22
canonical_for:
  - qwen38-draft-int4-patch-provenance
---

# Draft-side INT4 patch-generation provenance record

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** This record exists to close the
"patch-generation provenance" item on the independent claims audit
(`docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/CLAIMS-AUDIT-20260921.md`).
It states exactly which patch bytes produced the draft-side INT4 numbers, how
they were generated and anchored, and what this record does **not** establish.

## 1. The exact artifacts

Two hand-written, self-contained Python patches (written 2026-09-14; not
tool-generated), held in the **public cookbook** patch directory, not in this
repo:

| Patch | Bytes | Lines | sha256 |
|---|---:|---:|---|
| `patches/patch_draft_lmhead_int4.py` | 9,174 | 233 | `db3768becdf1ac52a54dbd27a89ceb42338f8febbd32b1f9c8b6d6ffdd76649b` |
| `patches/patch_draft_mtp_int4.py` | 10,458 | 270 | `2f6d36c419438862157ad0b625a12264f84ea8c7a6d026d6d0d414aa46a14405` |

Timestamps on disk: 2026-09-14 07:07 both. These are the **only** two patches
that define the measured variable `B70_DRAFT_LMHEAD_INT4` / `B70_DRAFT_MTP_INT4`.

## 2. Generation mechanism

Each patch is a standalone script that **embeds its runtime helper as a string
literal** and, when executed inside the serving container, writes that helper
into the container's installed vLLM tree and splices one guarded call site into
`qwen3_5_mtp.py`:

- `patch_draft_lmhead_int4.py` → writes `b70_draft_lmhead_int4.py` (chunked
  symmetric INT4 g128 quantizer for the draft's 2.543 GB F16 LM head; routes the
  draft's `compute_logits` through `torch.ops._xpu_C.int4_gemm_w4a16`) and
  replaces `Qwen3_5MultiTokenPredictor.compute_logits`.
- `patch_draft_mtp_int4.py` → writes `b70_draft_mtp_int4.py` (same quantizer;
  swaps the `quant_method` of the MTP module's 5 linears — `fc`,
  `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj` — for a duck-typed method
  routed through `int4_gemm_w4a16`) and appends the build call at the end of
  `Qwen3_5MTP.load_weights`.

Both are **env-gated** and no-op by default (`os.environ.get(...) != "1"` early
return), and both are **fail-closed on anchors**: if the expected anchor text is
absent from `qwen3_5_mtp.py` they `sys.exit("anchor not found in …")` instead of
writing anything. Each patch `compile()`s the spliced text before writing it.

Weight layout produced is exactly the one the existing `int4_gemm_w4a16` op
consumes (same as the GPTQ body layers): `qweight int32 [K/8, N]` NT layout,
sequential nibbles LSB-first, stored value `q + 8` (q ∈ [−8, 7]), `scales fp16
[K/128, N]`, `qzeros int8([8])` = symmetric branch. Target-side weights and the
target's verification logits path are never touched.

## 3. Anchoring across image generations (the provenance question)

The patch headers record their anchor verification as:

> Anclas verificadas contra la imagen `vllm/vllm-openai-xpu@2c427ef`
> (vLLM 0.26.1.dev457.gc810e5ee9)

i.e. the patches were authored and anchor-checked against image
`sha256:2c427ef477da…`, while the champion A/B runs apply the same bytes to the
champion image `sha256:f01e24f6c7ff…` (mounted inside the container by
`scripts/tmp/p4-champion-p8192-ab-20260921.sh`, which embeds both patches and
executes them under their env gates).

Because the patches **hard-fail when an anchor is missing**, every run that
reached a healthy server on `f01e24f6…` is positive evidence that the anchor
text was present in that generation too. The provenance chain is therefore:

```text
hand-written patch bytes (sha256 above, 2026-09-14)
  ├─ anchor-verified at authoring time against 2c427ef… (vLLM 0.26.1.dev457.gc810e5ee9)
  └─ fail-closed anchor splice, applied per container start to f01e24f6…
       (each successful serve is anchor-match evidence for that generation)
```

Application order inside the container (from the harness entrypoint):
`patch_mtp_nightly.py` → `patch_mtp_boundary.py` → `patch_champion_stack_overlay.py`
(the overlay embeds and runs the two draft patches when the env gates are set).
Helper modules land in the container's own vLLM tree and are ephemeral: they are
re-written per container start, so no cross-run state can leak between arms.

Between the two A/B arms the **only** difference is the two env vars. The
unpatched arm executes the same entrypoint and the same patch scripts, which
return early on the env gate.

## 4. A mechanism wording discrepancy, recorded not smoothed

The `patch_draft_lmhead_int4.py` header states the draft MTP module uses **its
own** LM head ("`DraftModelProposer._maybe_share_lm_head` es no-op — no comparte
el del target"), while the campaign narrative describes a *shared* head read 5×
per MTP4 step. Both agree on the measured fact that matters: a 2.543 GB F16 head
is read once per draft-token step (5 applications per MTP4 step ≈ 12.7 GB/step)
and the draft-side copy is what gets quantized, leaving the target's verification
logits path untouched. Whether the draft and target hold one tensor or two is a
**wording** item for the upstream contribution draft and must be stated from the
current upstream source, not from either local artifact here. It is not a numbers
item: the measured effect is identical under both readings.

## 5. What this record does NOT establish

Per the cookbook invariant "a GPU-free patch apply test proves source
compatibility only":

- **Not runtime correctness or stability** — that is the serving evidence.
- **Not numerical parity** — that is the greedy byte-identity test (3 tasks × 2
  repeats, identical sha256 across arms) and the 30-draw × 6-prompt modal-class
  agreement; those live in the result roots and are scoped to C1/TP1 greedy plus
  one sampler configuration.
- **Not upstreamability** — the runtime monkey-patches are local-only. The
  reviewable tree patch (base `066ae99e`, uncommitted) is a separate artifact and
  still owes its wiring validation and a speed reproduction *from that tree*.

## 6. Relation to the claims audit

| Claims-audit item | Status after this record |
|---|---|
| metric naming (client post-first vs engine rate) | closed by wording rule: every quoted figure is labeled **client post-first rate** |
| energy/temp telemetry | closed separately: `power.json` per arm (raw `energy1_input` delta, elapsed, avg card W, configured cap, resolved hwmon path, `pkg` temp) — code validated standalone 2026-09-22 and exercised by the fresh n=5 runs |
| **patch-generation provenance** | **closed by this record**, and further closed on the source-anchor half by `docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/P4-anchor-validation-20260922.md` — both anchor strings were tested **byte-exactly** inside a CPU-only container on the champion generation `f01e24f6…` (vLLM 0.27.2rc1.dev77+gac7509e2b, vllm-xpu-kernels 0.1.12.3): both anchors present verbatim, no B70 markers pre-existing, zero `anchor not found` failures across every champion run's server.log, and runtime application proven by the patch's own `[B70]` log lines. So the §3 chain above is now *measured*, not just fail-closed-by-construction |
| distribution-level sampled path | closed as supportive-only (modal agreement 6/6 at n=30); any formal equivalence statement needs more draws |

Evidence: this file · `results/2026/09/b70-improvement-probes-20260920/p4-champion-ab-*` ·
`docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/CLAIMS-AUDIT-20260921.md`
