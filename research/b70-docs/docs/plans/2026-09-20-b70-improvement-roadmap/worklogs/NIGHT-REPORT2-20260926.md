---
title: "Night Report 2 — single-B70 max-speed night 2026-09-25/26"
type: worklog
date: 2026-09-26
status: living
---

# NIGHT REPORT 2 — single-B70 max-speed research night (2026-09-25 22:00 → 2026-09-26 00:05)

**Honest headline.** The night produced **decisions and root causes, not tok/s: zero promotions,
champion recipe unchanged** — by measurement, not by failure to run. Prefill attribution closed the
last single-card compute lever (P-C); T1/T2 closed W3 and proved rotation's real value (W4 quality
at iso-bytes); flash-next took its 13th block, now with the **root cause of the entire thermal wall
identified** (idle-awake B70 physics + a blocked runtime-suspend path); unitrace tooling is built
and ABI-validated. Everything PROVISIONAL — diagnostic-grade (RL-065 substitution; Phase-1 peak hit
81 °C transiently — disclosed). Nothing left the machine.

## Track status

| Track | Status | Outcome |
|---|---|---|
| P1 prefill attribution | **DONE — P-C CLOSED** | p8192/g1: GEMMs = 74% of ~4.82 s/req device time but run at ~64–68% of the BF16-class ceiling → pre-declared rule (b) fails → rotation-W4A8 prefill program not justified. fmha = 12% at poor per-call efficiency (~25 ms big calls) — noted, below program gate. Full table in the results root. |
| P2 W3/bytes | **DONE — W3 CLOSED, T3 spec'd** | Worklog `QWEN27-P2-W3-CLOSURE-T3SPEC-20260925.md`. T3 is storage-gated: no mount qualifies for the ~55 GB BF16 re-fetch (needs operator-authorized SN770 reclaim). |
| T1/T2 rotation (prep C) | **DONE — T2 PASS, T1 fail-attributed** | T2: rotation improves W4-G128 quality at iso-bytes on real activations (KL 0.40–0.58×, top-1 +0.11–0.25 pts). T1 max-abs 5.6× over tolerance but controls attribute it to the BF16 rewrite floor (0.05 gate was below any rewrite's noise; cosine ≥0.99999 passes). Discovery: the "BF16 source" is a dead symlink farm; shard 9 legally re-fetched. |
| P3 flash-next | **BLOCKED-thermal #13 — with root cause found** | GPU1 sat 72 °C idle with zero recovery slope. NOT a flash-next defect. See thermal section. Calibration side is now SOLVED (offline exact 512/8192 + live gate, verified-ready) — the next attempt needs only a cool card. |
| P4 unitrace O13 | **READY, deferred** | Binary built (sha256 c771fafc…), container LD_PRELOAD ABI-validated against the pinned image. Deferred: GPU1 at 72 °C would peak >80 °C under 227 W decode. |
| P5 close-out | **DONE (this report)** | Ledger RL-066…069, OUTCOMES/LEVERS updates, model-status, indexes. |

## THE THERMAL ROOT CAUSE (the "2800 MHz" mystery — solved)

Operator observation: "something holds the card to 2800 MHz max." Investigation (all evidence in
ledger RL-069):

1. **Awake-idle physics:** an awake B70 draws **45.3–52.3 W doing nothing**, regardless of clock
   (measured at both 400 MHz and 2800 MHz on GPU1; GPU0-with-display = 52.3 W). A card that cannot
   reach runtime suspend CANNOT cool below its ambient+45 W equilibrium — this is the entire RL-065
   "≤55 °C unreachable" wall and the flash-next block history.
2. **GPU1's suspend blocker:** `Xwayland` (PID 127081) holds `/dev/dri/card1` open (an X client
   probed GPUs). The DRM fd keeps the device `runtime_status=active` forever → 45 W → 72 °C floor
   after any campaign soak.
3. **The 2800 MHz reading:** a stale `cur_freq` request left by docker-killed vLLM at max boost;
   cosmetic for power (awake cost is flat) but it is the visible symptom. Writing max_freq=400
   cleared it instantly; restored to 2800.
4. **Aggravators found and cleared:** the morning session's 1 Hz energy-polling monitor loop was
   still running (every `energy1_input` read resumes the GPU) — killed; the dashboard service polls
   per-tick too (restarted after diagnosis — it is not the primary blocker; the fd is).

**Fix path (operator decision, none taken unilaterally):** (a) PCI-cycle GPU1 via `b70-gpu1 off/on`
during an inference-free window (severs the stale fd; the tool's global inference guard correctly
refused tonight while the provider server runs); (b) or a GNOME logout/login to respawn Xwayland
without the fd. **Disclosure:** during diagnosis GPU1's max_freq was capped to 400 MHz for ~8 min;
GPU1 was FREE at the time — the operator's provider server (`b70-qwen38-27b-vllm`, GPU0,
ZE_AFFINITY_MASK=0) was never touched; cap restored; dashboard restarted.

## Session mechanics worth keeping

- The torch-profiler stop/flush caught 3 requests in the window — per-request = total/3 reconciled
  exactly (14.46 s device / 3 ≈ 4.82 s ≈ TTFT 4.75 s; GPU ~100% busy in prefill).
- The generic `gemm_kernel` name (not "int4_gemm") carries the W4A16 body GEMMs — regex buckets
  must include it or attribution misreads 74% as "other".
- Phase-3's first launch failed on an inverted process-gate (failed-on-clean) — caught and fixed
  before any load; recorded.
- p8192 calibration achieved 8196 (client tolerance) — disclosed; cold-input rates used actual.
- The qwen27-rotation Q1bd secondary arm (block-diag 17408) was still computing at close-out after
  ~1 h 45 min — terminated at 00:16 during the follow-on program (correction: an earlier draft of
  this report claimed it had been stopped at report time; it had not — it ran 1 h 48 m total);
  partial per-tensor logs retained in the results root; no gate depends on it.

## Closed forever (additions)

P-C rotation-W4A8 prefill (closed-by-attribution); W3/INT3 kernel escalation (closure reaffirmed
with T2 as supporting evidence); T1's 0.05 max-abs tolerance class (future gates = ~2× a no-op
rewrite control); re-running host-prep/L-B/L-C/#56096/prefill-knob (unchanged from RL-061…065).

## What remains open, in order

1. **O14** — copy-based collective correctness probe → gated TP2 campaign (unchanged; the only
   structural lever left).
2. **O13** — unitrace record-close (tooling READY; needs a cool GPU1 window).
3. **Flash-next valid C1 screen** — calibration SOLVED; needs the thermal fix (PCI cycle) + a
   both-cards-free window.
4. **T3** — whole-model rotated-W4 parity (spec'd; storage-gated on the BF16 re-fetch).
5. **O15/thermal** — now has a named root cause and a two-command fix; decide when to cycle.

## Single best next action

`b70-gpu1 off && b70-gpu1 on` during an idle window (releases the Xwayland fd), then the flash-next
screen with the ready calibration — it converts the 13-block history into a first real measurement
and validates the thermal fix in one session.

*No recipe change, no public number, no commit. Draft X/post text: none (no promotion).*

---

# FOLLOW-ON PROGRAM (2026-09-26 00:00–00:45, operator: "kill current server and RUN EVERYTHING UNTIL PERFECT CONFIRMATION/DENIAL")

Everything below ran under full authorization after stopping the provider container (restarted at close).

## CONFIRMATIONS / DENIALS — the complete list

| Question | Verdict | Evidence |
|---|---|---|
| The idle-heat wall (RL-065), root cause | **CONFIRMED & FIXED** — PCI cycle severed the Xwayland fd: 72→44 °C, stale 2800 MHz readout cleared, runtime suspend accrues when unpollled; flash-next then ran with peaks 64/60 °C | RL-069/072 |
| flash-next C1 performance (13 prior blocks) | **MEASURED (first time ever)**: engine tg 21.3–21.9 tok/s, cold input 185–195 tok/s, calibration VERIFIED, coherence PASS. **DENIED as a speed route on this placement** (3.5×/9× slower than champion) — capacity route | RL-070 |
| O13: the "unresolved" 63–67% body bucket | **CLOSED** [COMPOSITION QUALIFIED 2026-09-29, register row 2]: L0 tracer sees graph-replayed kernels — 87.7% GEMM is **active-kernel share, not wall share**: raw `python3.220.json` recomputes to 553 GEMM events ≈ 35.2 ms (~63.6 µs each) of a 58.35–58.59 ms round, GPU-event union ≈ 40.2 ms, **~18 ms uncovered** (the recorded "36.73 ms from 563 × ~1.1 ms" is arithmetically inconsistent). With the 30–40 % tracer overhead this is not recoverable-slack evidence either way — **"no hidden waste" is withdrawn as proven**. **Single-card decode record CLOSED at the byte-floor and rate-ceiling levels; the visible-composition level carries this caveat** | RL-072 |
| O14: cross-card collectives correctness | **CONFIRMED — exact (bitwise) allreduce on both transports**, ~102 µs @ 4 KB, 14.3 ms worst-case round. **TP2 campaign GO**; honest band pre-declared at +20–50% (1.6×+ claims stay dead) | RL-071 |
| P-C W4A8 prefill program | **DENIED** (closed-by-attribution: GEMMs at 64–68% of rate ceiling) | RL-066 |
| W3/INT3 | **DENIED** (runtime + byte ceilings; T2 shows rotation's value is W4 quality at iso-bytes) | RL-067 |
| T3 rotated-W4 whole-model parity | **SPEC'D, storage-gated** (~55 GB re-fetch, no qualifying mount — operator decision) | RL-067 |
| TP2 ≥1.6× / 160–190 tok/s | **DENIED** by measured collective arithmetic (third independent kill) | RL-071 |

## The one open speed program

**Gated TP2 research campaign** (O14's follow-on): INT4 champion-base, 2 ranks, graphs-off for
collectives, matched same-session 1-card control, p512/g128 + p8192/g128 n=3 → n=5; pre-declared
gate +20–50% decode; backend xccl; abort on corruption/reset/OOM. This is the last structural
lever on this host.

*Host returned to service: provider container restarted and verified serving; caps 150 W; both cards free before restart; nothing published; no commits.*
