---
title: "TaskB splice design — L59–L63 window test for the margin-aware requant objective (2026-09-29)"
type: research
date: 2026-09-29
status: provisional
---

# TaskB splice design — L59–L63 window test for the margin-aware requant objective (2026-09-29)

- Status: **PROVISIONAL — NOT FOR PUBLIC HEADLINE**
- Mode: **CPU-only static feasibility design.** No conversion, no GPU, no model
  loads executed; source reads + small arithmetic only. No authority edited.
- Intent: `research` (`tools/situation.py compile` → `halt: false`, 2026-09-29).
- Context: TaskB Gate0 (`results/2026/09/b70-margin-requant-gate0-20260929T204352Z/`)
  returned ADVANCE_AS_SYSTEMATIC_CANDIDATE — 2 designated-pair inversions on heldout
  low-margin rows, same L59–L62 terminal-swing signature as the canary
  (`research/2026/b70-margin-requant-gate0-census-20260929.md`). Independent
  architect objection addressed here: **a terminal-layer census of where the margin
  flips does not prove a margin-aware objective can un-flip it.** This file designs
  the smallest splice experiment that converts "where" evidence into a causal
  attribution grid plus a bounded objective-reduction test — without a conversion.

## 1. Evidence base (all pins verified live 2026-09-29)

### Captured state inventory

| File | Keys | What exists | What does NOT |
|---|---|---|---|
| `results/2026/09/b70-probe2-30defect-20260929/runs/{bf16,arext}-fp32-failing-r1.hidden.npz`, `armc-fp32-failing-resumed-r2.hidden.npz`, `arext-fp32-control-r1.hidden.npz` | 1120 each | `L{0..63}_all` `[36,5120]` fp16 (full-seq post-layer hiddens, step-0 prefill); `L{0..63}_last` `[5120]` (last generated step — **not** step-0); `lin_in__*`/`lin_out__*` `[1,in]` fp16 at step-0 last position for all 496 linears | no fp32 states, no attention KV, no GDN recurrent/conv state, no masks |
| `results/2026/09/b70-margin-requant-gate0-20260929T204352Z/runs/{bf16,armc,arext}-fp32.hidden.npz` | 64 each | `L{0..63}_last` `[12,5120]` fp16 — per-row last-real-position hiddens | **no `L*_all`** — gate0 script captures only last positions (`gate0_margin_census.py:194,203-204`, npz write `:272-275`) |

So `L58_all` (the splice boundary = output of layer 58 = input to layer 59) exists
for exactly 4 probe2 runs: failing canary (36 tok) on bf16/armc/arext + control
(36 tok) on arext. For any other prompt (incl. the heldout h03/h06 inversions),
`L58_all` must be re-captured (§5).

### Source pins

- Splice-boundary tensor: `cpu_probe.py:199-213` — `h_all[li] = layer output`,
  so `L58_all` is verbatim the input layer 59 saw at step 0.
- Layer types: config `layer_types` = [lin,lin,lin,full]×16 → **L59, L63 =
  full_attention; L60, L61, L62 = GDN linear_attention**
  (`modeling_qwen3_5.py:743-747`; config `full_attention_interval=4`).
- Decoder layer: `modeling_qwen3_5.py:752-792` — needs `hidden_states`,
  `position_embeddings`, `attention_mask` (per-type), `position_ids`; residual
  wrap is input_layernorm→mixer→add, post_attention_layernorm→MLP→add.
- GDN recomputable: `Qwen3_5GatedDeltaNet.forward` (`:431-542`) with
  `cache_params=None` → `use_precomputed_states=False` (`:442`),
  `initial_state=None` → zero recurrent state (`:300-303,503`), causal conv
  (`causal_conv1d_fn` `:219-238`, kernel 4 left-pad), chunked scan
  `torch_chunk_gated_delta_rule` `:249-326` — **pure torch, autograd-capable,
  internally fp32** (`:264-266`). GDN state at L60–62 is a deterministic function
  of the window's own inputs over the full sequence — nothing to capture.
- Full attention recomputable: `Qwen3_5Attention.forward` (`:655-700`) builds
  K/V from the layer's input at every position; with `past_key_values=None`
  (`:678-679` skipped) it is exactly the prefill the census ran. Causality comes
  from the model-built mask (`:1189-1191`); eager path needs the additive mask
  (`:617-619`), sdpa uses `is_causal` when mask is None — replicate
  `create_causal_mask(**mask_kwargs)` verbatim (`:1181-1191`), not a hand mask.
- Recurrent/pad mask: `create_recurrent_attention_mask` returns None or the 2-D
  pad mask (`masking_utils.py:1474-1504`); pads are post-causal-inert either way —
  the gate0 no-mask batching was empirically exact (r01 replay +0.4017 to 4 dp).
- Readout: `text.norm` then `lm_head` (`:1208`); `Qwen3_5RMSNorm` is
  `(x·rsqrt(mean x²+eps))·(1+w)` fp32 (`:719-733`) — the lens formula already used.
- lm_head + in_proj_a/b + norms + conv1d are **unquantized** in both arms
  (verified in armc index: `in_proj_a.weight` present, `in_proj_qkv.qweight`
  present; audit `packing-audit.md` says the same for arext).
- Converter framework (TaskB): `campaigns/2026/autoround-vs-exl3/ar-work/convert_armc.py`
  — AutoRound 0.14.2 `scheme="W4A16"` sym g128, ignore list `lm_head,mtp,visual,
  embed_tokens` (`:80-88`); margin-objective spec: plan TaskB §B1
  (`research/2026/b70-sub4bpw-and-margin-requant-plan-20260929.md:298-324`).
- Recorded margins to reproduce (manifest.json:54): bf16 +0.4017, armc −2.8794,
  arext −2.9730; heldout inversions h03@armc +0.330→−1.204, h06@arext
  +0.334→−0.202 (`census.json`); terminal swing L60–63 (probe2 RESULTS).

## 2. Recomputability verdict — YES, with two declared caveats

A 5-layer splice driven by `L58_all` over the **full sequence** is exactly
computable on CPU:

1. **No captured KV/state needed.** L59/L63 attention K/V and L60–62 GDN
   conv+recurrent states are functions of the window's own inputs, which the
   splice produces at every position. The zero-initial-state recurrence matches
   the original forward because the original also started from zero and consumed
   the same (splice-reproduced) inputs. You cannot splice *inside* the window
   mid-sequence without state — the window boundary at L58 is the right cut.
2. **Masks:** single-row (36 tok) needs `attention_mask=None` + model-verbatim
   `create_causal_mask` for the two full-attn layers. Rowbatch = right-pad +
   per-row last index (gate0 precedent, exact under causality); optionally pass
   the 2-D pad mask to GDN layers for the `apply_mask_to_padding_states` zeroing
   (`:187-194,438`) — defensive only, provably inert for real positions.
3. **Caveat A (fp16 boundary):** `L58_all` is fp16. Treat the boundary as a
   controlled perturbation, never fp32 parity. Quantified by the replay controls
   (§3) plus a microcheck: re-round the boundary to fp16 a second time / add
   ±1-ulp noise ×3 seeds and report margin std (pre-declare ≤0.05 acceptable).
4. **Caveat B (npz `L*_last` trap):** probe2 `L*_last` is the **last generated
   step's** hidden, not step-0. Splice boundary and decision position must come
   from `L58_all` (step-0 prefill), whose row `-1` is the decision position.
   Gate0 npz `L*_last` is step-0-correct but position-only — unusable for splice.

## 3. P0 — smallest splice test (data in hand, zero capture)

Harness: reuse census Provider/dequant/skeleton verbatim
(`gate0_margin_census.py:32-147`); materialize only `text.layers[59:64]` +
`text.norm` + `lm_head` (persistent, ~7.6 GB fp32 window + 5.1 GB head);
`inputs_embeds = L58_all.float()`, position_ids per `modeling_qwen3_5.py:1166-1177`,
per-type masks per `:1189-1201`, readout = norm→lm_head at last position.

Cells (each = one window forward, ~1 s/row at 36 tok once weights are resident):

| Cell | h58 input | L59–63 weights | Expected | Reads |
|---|---|---|---|---|
| C-ref | bf16 | bf16 | ≈ +0.4017 | replay control: fp16-record + splice-boundary drift |
| C-q(armc) | armc | armc | ≈ −2.8794 | quant replay control |
| C-q(arext) | arext | arext | ≈ −2.9730 | quant replay control |
| X1 armc | **bf16** | armc | ? | isolated terminal-window quant effect |
| X2 armc | armc | **bf16** | ? | upstream-damage-only rescue test |
| X1/X2 arext | same 2×2 | | ? | second quantizer/packing |
| ctrl | arext control h58 | arext | ≈ positive | PASS-class sanity |

Pre-committed attribution read: X1 negative ⇒ window weights alone suffice
(localized repair is viable); X2 negative ⇒ upstream damage alone suffices
(terminal-only scope dead); both negative ⇒ distributed damage; C-cells failing
the tolerance band invalidates the instrument before any interpretation.

Cost: ≤9 forwards + one-time IO/dequant (~0.95 GB int4 + ~3.8 GB bf16 read,
numpy unpack tens of s). **Wall ≤ 30 min, RSS ≤ 20 GB.**

## 4. Candidate round/scale coordinate updates — pricing (iso-4bit, head excluded)

Scope: the 32 quantized linears in L59–63 (attn layer: q12288,k1024,v1024,o,
gate/up/down; GDN layer: in_proj_qkv10240, in_proj_z6144, out_proj, gate/up/down;
in_proj_a/b stay unquantized as exported). Weights ≈ **1.89 G** → int4 payload
≈ 0.95 GB, g128 scale params ≈ **14.8 M**. Iso-4bit = keep sym g128 zp=8 packing;
head/norm/conv/in_proj_a/b untouched — every candidate stays a legal export of
the same format.

Objective (TaskB §B1 applied to the window):
`O(θ) = Σ_rows MSE(h63_splice, h63_ref) + λ·Σ_{t∈D} w(m_t)·relu(γ − m_t(θ))`,
`w ∝ 1/m_bf16`, `h63_ref` = bf16-window output on the same input (computed in P0).

| Scheme | Vars | Per-step cost (calib batch ~6 rows × ~40 tok) | Budget |
|---|---|---|---|
| S0 gradient-sensitivity probe | none — one fwd+bwd of hinge vs MSE at θ=current | ~30 s | minutes — **the least-cost discriminator** |
| S1 scale-only coordinate updates (codes fixed, per-group scale multipliers, autograd) | 14.8 M | fwd ~5 s + bwd ~12 s | 200 steps ≈ 1 h |
| S2 round+scale STE (AutoRound-analog per-weight rounding var) | +1.89 G | +7.6 GB vars; step ~20 s | 300 steps ≈ 1.7 h, RSS ≤ 30 GB (SGD-momentum, not Adam) |
| S3 block-sequential local-MSE warmup (cached per-layer inputs ~15 MB/linear) then joint hinge tail | per-layer | layer step ~2 s | ~2 h total |

Anchors (measured): census bf16 12-row 64-layer fwd 137.9 s ⇒ ~2.15 s/layer ⇒
window ≈ 11 s at 12 rows; quant arms' extra cost is dequant IO, paid once.

**Data gap, priced:** heldout eval needs the *repair arm's own* `L58_all` per row
(production boundary = quant upstream), which gate0 did not retain. Extension
capture = census variant keeping `L{58..63}_all` for the 12 manifest rows + a
pinned calibration set (canary + ~6 same-family prompts, frozen before tuning):
early-exit at layer 58 ⇒ ~123 s bf16 / ~360 s armc, ≤30 GB RSS, npz ≤100 MB.
Order: P0 → S0 sensitivity → capture extension → S1/S2 tuning → heldout eval.

## 5. The falsifier logic — MSE is not the hinge

"Candidate lowers window MSE but the margin stays negative" does **not** refute
margin-aware optimization: MSE reduction is precisely what baseline AutoRound
already delivered (armc beat naive RTN, relFro 0.104 vs 0.119–0.130, and still
flipped) — the theory's premise is that magnitude error is not the operative
variable. What the splice actually tests, in order:

1. **Gradient sensitivity (S0):** is `∂hinge/∂θ` at the current codes/scales
   nonzero and margin-moving? If one measured backward step cannot shift the
   designated margin ≥0.05 after ~10 bounded steps → **hinge-flat**: the margin
   is unreachable through these coordinates at this window — that refutes the
   *window-localized* objective, not the margin-aware class.
2. **Bounded objective reduction:** does a declared-budget candidate reduce
   `Σ w·hinge` on the calibration rows **and** transfer sign/neutrality to the
   heldout evaluable rows (h01,h02,h05,h08,h09,h10 + flipped h03/h06)?
   Calibration-only wins with heldout regression = overfit ⇒ that DOES refute
   the 1/m-weighted hinge as specified at this corpus size.
3. **Capacity vs scope:** hinge reduced but sign stuck at iso-4bit in-window =
   a scope verdict (widen window / relax bits on K worst tensors), not a theory
   verdict. Report MSE, hinge, and per-row margin sign **separately**; never
   claim a minimum was found — only that bounded reduction was or wasn't
   achieved under the declared budget.

## 6. Pre-declared thresholds

| Item | Threshold |
|---|---|
| Replay-control agreement | \|margin_splice − margin_recorded\| ≤ 0.10 all three C-cells; investigate 0.10–0.50; abort instrument >0.50 |
| fp16 boundary noise | margin std ≤ 0.05 over 3 ±1-ulp seeds |
| P0 wall / RSS | ≤30 min / ≤20 GB |
| S0 sensitivity | ≤4 fwd+bwd per arm, ≤15 min; hinge-flat declared if max-margin gain <0.05 in 10 probe steps |
| Capture extension | ≤700 s per arm (early-exit at L58), ≤30 GB RSS |
| S1/S2 tuning | ≤300 steps, ≤4 h wall per arm, RSS ≤30 GB |
| Advance | hinge reduced on calibration AND ≥1 heldout flipped row turns non-negative with zero new inversions on surviving rows |

## 7. Least-cost next test

**P0 splice grid only** — 9 window forwards on existing npz + ~40 lines of
skeleton reuse. It (a) validates the fp16-boundary instrument via replay
controls, (b) attributes the flip to window-weights vs upstream accumulation,
and (c) produces the bf16-window reference outputs the objective needs. S0
gradient sensitivity is the second step and needs nothing further. Everything
beyond (capture extension, S1/S2, heldout) is gated on P0+S0 and on this design
being reviewed — no conversion is implied or authorized by any outcome.

## 8. What this does not decide

- Nothing about serving speed, kernels, or the EXL3 route.
- A positive P0/P1 does not license a conversion; the B5 ladder
  (battery 6/6 kill-gate → replay → Δlogprob → tasks) is unchanged.
- The window choice L59–63 is evidence-guided (lens swing L60–63 canary,
  L59–62 heldout) but not proven minimal — L58 inclusion is the declared
  sensitivity fallback if controls show boundary drift at the cut.
