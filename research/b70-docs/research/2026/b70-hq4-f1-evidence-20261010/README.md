# RQ-26 F1 evidence root — first GPU session (2026-10-10)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Kernel microbench for HQ4 F1 on
Intel Arc Pro B70 GPU1 (PCI 0000:0f:00.0, pinned live via
`ONEAPI_DEVICE_SELECTOR=level_zero:1`, verified by
`scripts/tmp/rq26_f1_build/rq26_f1_pin` before launch). No engine, model or
service change is implied; nothing is adopted by this session.

Contents (written during the session):

- `ISA_GATING.md` — gate provenance (analyst-lane dumps + this session's
  mechanical checks; independent analyst verdict outstanding).
- `round1/ENV.txt` — session identity: selector pin, as-found caps
  (GPU0 180 W, GPU1 230 W — never written), energy/temp bracket.
- `round1/reverify.txt` — the orchestrator's hard re-verify result
  (zero engine clients in both dri tables; GPU1 visible_avail ≥ 31000 MiB)
  immediately before launch.
- `round1/contention.log` — per-cell foreign-engine checks (the foreign
  arm.sh/llama-stream loop was ACTIVE when the session started; cells only
  ran on a verified-clear window, and any mid-session re-arm taints the
  following cells, flagged here).
- `round1/selftest.{out,err}` — constant-operand dpas checksums (RC=1/4/8,
  expect 1024.0).
- `round1/correct_summary.txt`, `round1/correct_<cell>.{out,err}` — bitwise
  correctness: wrap outputs (x', S) vs case files and GEMM outputs vs the
  bitref float32 exact-order model, byte-for-byte, for dev + all three real
  27B shapes × M ∈ {1,4,8}.
- `round1/ring_summary.txt`, `round1/ring_<cell>.{out,err}` — ring cells:
  one discarded same-shape warmup then 30 profiled launches cycling a weight
  ring ≥ 4× the 32 MiB L2 upper bound (slots from live visible_avail − 4 GiB,
  `RING_PLAN.json` in the cases dir). `CSV,...` carries median/min/max µs and
  effective GB/s over NAMED bytes (weight plane + fp16 scales only);
  `SAMPLES,30,..` carries every raw per-launch time.
- `round1/wrap_summary.txt`, `round1/wrap_<cell>.{out,err}` — the K-wrap
  (Rademacher signs + block-128 FWHT) timed separately: 100 profiled
  launches, per-launch ns.
- `round1/ring_plan_snapshot.json` — the ring plan actually used.

Case inputs and their sha256s: `scripts/tmp/rq26_f1_cases/MANIFEST.json`
(regenerate deterministically via `scripts/tmp/rq26_f1_pack.py`). Kernel
source and layout contract: `scripts/tmp/rq26_f1_kernel.cpp`,
`scripts/tmp/rq26_f1_layout.md`. CPU-side verification (71/71) and retained
log: `scripts/tmp/rq26_f1_test_cpu_20261010.log`.

Known caveats for any reader of the numbers: (1) the ISA gate was
self-verified by the kernel engineer against the analyst lane's dumps — no
independent analyst verdict yet; (2) round-1 cells are first absolute
measurements, single-window, not paired screens; (3) the dpas F32 stage order
is the bitref pair-sum model — the bitwise phase adjudicates it empirically,
and a mismatch would regenerate cases with `--seq-stage` before any timing
(pre-declared flip).
