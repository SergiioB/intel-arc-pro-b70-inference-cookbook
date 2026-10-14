---
status: living
title: "Ornith MixedCal-v2 conversion"
type: model-dossier
date: 2026-08-21
---

---
schema_version: 1
model_id: ornith-ai/Ornith-1.5-35B-A3B
model_slug: ornith-1.5-35b-a3b
document: mixedcal_v2_conversion
as_of: 2026-08-21
status: completed_promotion_screen_in_progress
truth_verdict: CONFIRMED
evidence_tier: E2
publication_gate: SELF_REPORTED_WITH_LIMITS
primary_evidence:
  - ../../../results/2026/08/ornith15-35b-20260819T1630Z/
  - /mnt/<models-disk>/ornith15-quant/mixedcal-v2/calibration-manifest.json
  - /mnt/<models-disk>/ornith15-quant/mixedcal-v2/prevention-tests-20260820T164901Z.txt
  - /mnt/<models-disk>/ornith15-quant/mixedcal-v2/runs/mixedcal-v2-20260820T165356Z-r3/
  - /mnt/<models-disk>/ornith15-quant/mixedcal-v2/runs/mixedcal-v2-20260820T204436Z-r4/
  - ../../../results/2026/08/ornith15-mixedcal-v2-screen-20260821T043027Z/
---

# Ornith MixedCal-v2 conversion

## Purpose

Produce a separately named Ornith GPTQ artifact with broader natural-routing calibration and fewer GPTQModel RTN fallbacks than the original WikiText-calibrated artifact. The experiment changes calibration coverage only; it deliberately keeps the quantization format and runtime contract matched so a later old-versus-new comparison can isolate the calibration change.

The original working GPTQ artifact and official BF16 source are immutable inputs. MixedCal-v2 never overwrites either one.

## Quantization contract

| Dimension | MixedCal-v2 contract |
|---|---|
| Target format | GPTQ INT4 |
| Group size | 128 |
| Symmetry | Symmetric |
| Activation ordering | `desc_act=false` |
| Quantized scope | Routed expert `gate_proj`, `up_proj`, and `down_proj` only |
| Expected quantized matrices | 40 layers × 256 experts × 3 projections = **30,720** |
| LM head | Not quantized |
| Embeddings | Not quantized |
| Attention and GDN | Not quantized |
| Router gate | Not quantized |
| Shared experts | Not quantized |
| MTP draft tensors | Retained at higher precision; expected **785** tensors and zero quantization leaks |
| Vision tower | Not quantized |

The GPTQModel dynamic exclusions are:

```python
{
    "-:.*attn.*": {},
    "-:.*mlp\\.gate$": {},
    "-:.*mtp.*": {},
    "-:.*shared_expert.*": {},
    "-:.*visual.*": {},
    "lm_head": {},
    "model.language_model.embed_tokens": {},
}
```

## Inputs

| Input | Identity |
|---|---|
| BF16 source | `/mnt/<models-disk>/Ornith-1.5-35B-A3B-BF16` |
| Source revision | `fbb995a79eedd569a5edc5f2af9644c0fa1124fc` |
| Original GPTQ control | `/mnt/<models-disk>/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16` |
| Quantizer | GPTQModel 7.3.2 |
| Runtime image | `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f` |
| Calibration samples | 128 |
| Tokens per sample | 1,536 |
| Total calibration tokens | 196,608 |
| Calibration seed | `15035260819` |
| Corpus SHA-256 | `e88ccd5f999a05a3c78bb9ec86a82256255a9ad97068c787dfc564907a377381` |
| Tokenizer JSON SHA-256 | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |

Calibration spans code, reasoning, agentic/tool content, SQL, networking, technical prose, general text, and systems debugging. Routing remains natural: the process does not force router outputs or expert IDs.

## Storage design

Earlier attempts proved that the old layout was unsafe. At layer 35/39, disk offload had consumed approximately 16.04 GB and crossed the 8 GiB `/mnt/<models-disk>` reserve. The conversion therefore no longer places growing scratch or final output on the ordinary Linux model/root filesystems.

A dedicated workspace is provisioned as follows:

1. Mount the mostly empty `/dev/sdb2` NTFS volume at `/mnt/<other-disk>`.
2. Store one preallocated 80 GiB backing file under `B70-Ornith-Conversion/`.
3. Format only that backing file as ext4.
4. Loop-mount it at `/mnt/<disk>`.
5. Place run-specific offload, temporary data, incomplete output, and final output on this ext4 filesystem.

Capacity contract:

| Component | Reserved amount |
|---|---:|
| GPTQModel offload | 22 GiB |
| Temporary data | 2 GiB |
| Final artifact | 28,000,000,000 bytes |
| Untouched workspace floor | 10 GiB |
| Total preflight requirement | 64,507,222,016 bytes |
| Observed clean workspace availability | 79,641,833,472 bytes |

The conversion also preserves independent floors of 10 GiB on `/`, 8 GiB on `/mnt/<models-disk>`, and 10 GiB on `/mnt/<models-disk>`.

## Fail-closed execution

Canonical scripts:

- `/mnt/<models-disk>/ornith15-quant/run-mixedcal-v2.sh`
- `/mnt/<models-disk>/ornith15-quant/quantize-mixedcal-v2.py`
- `/mnt/<models-disk>/ornith15-quant/watchdog-mixedcal-v2.sh`

Safety properties:

1. A unique run ID is mandatory.
2. A global artifact lock rejects concurrent or renamed-unit launches.
3. Preflight creates no scratch and validates source, original control, output absence, mount identity, bytes, inodes, GPU isolation, VRAM, image availability, and power telemetry.
4. Every attempt owns unique offload, temporary, staging, logs, manifest, and status paths.
5. Dirty scratch and existing final/staging paths are hard failures.
6. `Restart=no`; the runner, watchdog, and systemd unit contain no retry loop.
7. Failures produce a durable non-retryable classification.
8. The watchdog is a final reserve guard, not the capacity planner.
9. Saving occurs under a same-filesystem `.incomplete` directory.
10. The staged artifact must pass its tensor contract before an atomic rename publishes the final path.
11. Cleanup restores each card's pre-campaign observed cap and touches only paths owned by the declared run.

Prevention tests passed for insufficient capacity, existing output, dirty scratch, held lock, unchanged failure latch, real capacity, source coverage, and script syntax. Evidence: `/mnt/<models-disk>/ornith15-quant/mixedcal-v2/prevention-tests-20260820T164901Z.txt`.

## Attempt ledger

| Attempt | Outcome | Evidence |
|---|---|---|
| Initial MixedCal attempt | XPU/`xe` engine reset around layer 5; no final artifact | Retained conversion logs |
| Attempt 3 | Reached layer 35/39; watchdog stopped it when `/mnt/<models-disk>` crossed its 8 GiB floor | `supervisor-attempt3.log`, `watchdog-attempt3.log` |
| Attempt 4 | Deliberately stopped because it reused the same deterministic storage layout | `supervisor-attempt4.log` |
| Attempt 5 | Stopped before conversion while the safe redesign was prepared | `attempt5-stop-20260820T164020Z.txt` |
| `r2` | Pre-container operator abort: the historical 52°C cooldown gate did not match the verified 55–56°C idle package behavior | `/mnt/<models-disk>/ornith15-quant/mixedcal-v2/runs/mixedcal-v2-20260820T165127Z-r2/` |
| `r3` | Safe conversion startup verified, then stopped at user request for a fresh night run; approximately 971 MiB run-owned offload inventoried and removed | `/mnt/<models-disk>/ornith15-quant/mixedcal-v2/runs/mixedcal-v2-20260820T165356Z-r3/` |
| `r4` | **Completed successfully.** 7 h 40 m wall; container exit 0; staged contract verified before atomic publication | `/mnt/<models-disk>/ornith15-quant/mixedcal-v2/runs/mixedcal-v2-20260820T204436Z-r4/` |

The `r3` stop is classified `OPERATOR_ABORT`. It did not produce or publish a MixedCal artifact. GPTQModel has no validated resume-from-layer path, so `r4` started cleanly at layer 0 under the canonical one-shot unit.

## r4 result — 2026-08-21

- Final artifact: `/mnt/<disk>/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2` (24,454,916,052 bytes, 6 shards, 124,611 tensor keys).
- Staged contract before publication: **30,720 expert `qweight` tensors**, **785 MTP tensors, 0 quantized**, **0 forbidden qweights**.
- RTN fallback: **3,186 of 30,720 = 10.37%** (string-exact count from journald-captured quantizer lines; 30,719 of 30,720 module lines were captured, one line missing does not change the string-level fallback count).
- Per-projection split: 1,062 / 1,062 / 1,062 (`gate`/`up`/`down`).
- Layer 39: 75/768 = 9.8% (was 297/768 = 38.7%). Worst layers are now 34 (201), 37 (174), 32/36 (156).
- Compared with the original artifact's 7,605/30,720 = 24.75%, the calibration change removed **58% of RTN fallbacks**.

## Current status

**CONVERSION COMPLETE — RESEARCH ARTIFACT PROMOTED (self-reported E2).**

- n=5 no-spec confirmation at 150 W is speed **parity** with the original WikiText GPTQ.
- MTP1 remains the research speculation default. Exact-token capacity completed through 262K C1.
- 230 W recovers the ~9.7k cold-input class (LMX long-prompt `tokSPrefill` 9556.4).
- The original GPTQ and BF16 source remain intact.
- Limits: C1 only; no BF16 logit/KL; LocalMaxxing not submitted.

## Promotion gate after successful conversion

1. Verify exactly 30,720 expert `qweight` tensors.
2. Verify 785 MTP tensors and zero MTP quantization leaks.
3. Verify all excluded component families remain unquantized.
4. Compute final RTN fallback count and per-layer/projection distribution.
5. Run a small held-out no-spec task-quality A/B versus the original artifact.
6. If quality is not worse, run matched exact p512/g128 and p8192/g128 n=3 screens at a fixed 150 W cap, cache off, with same-shape warmups.
7. Run MTP1 n=3 only if the no-spec candidate passes.
8. Keep the decision provisional. Do not run n=5/full characterization or publish without a separate decision.
