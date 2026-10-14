# ISA gating provenance — RQ-26 F1 first GPU session (2026-10-10)

The workflow's standing rule: timing is gated on an ISA proof before any GPU
run, owned by b70-isa-analyst. State at the time of this session:

- An ISA-analyst lane produced full disassemblies of the round-1 kernel
  (default and -DRQ26_F1_LSC_LOADS builds) at
  `scripts/tmp/rq26_f1_build/isa_dumps_round1/` and `.../isa_dumps_lsc/`
  (sections: Gemm<1>, Gemm<4>, Gemm<8>, WrapK; `.ze_info`, `.spv`,
  `.misc.buildOptions`). No verdict file or gate marker was written by that
  lane.
- The orchestrator authorized this session to self-verify the mechanical gate
  items against those dumps and proceed to timing (escalation decision,
  2026-10-10: poll-and-run with hard re-verify). Provenance of the four
  checks below: THIS session's greps over the analyst lane's dump files —
  not an independent analyst review.

| Gate item | Command run this session | Result |
|---|---|---|
| dpas emission, f16 operands, f32 acc | `grep -c 'dpas' isa_dumps_round1/.text._ZTSN6rq26f14GemmILi4EEE.asm` and same for Gemm<1> | `dpas.8x4 (16|M0) ... r40:f ... r11:hf r33.0:hf` (8 sites, RC=4); Gemm<1>: `dpas.8x1 ... :hf` (8 sites, RC=1 — the pre-declared RC=1 fallback is NOT needed) |
| No scratch spills | `grep -ci 'scratch'` over all four .asm | 0 in Gemm<1>/<4>/<8> and WrapK |
| Block loads | `grep -cE '\bsend\b'` + sample lines over Gemm<4>.asm | 51 `send.ugm`; `load.ugm.d32x16t.a32.ca.cc` (64 B class) and `d32x8t` (A-fragment 32 B) |
| LSC variant necessity | `wc -c isa_dumps_round1/default_vs_lsc_Gemm1_isa.diff` | 0 bytes — default `copy_from` and `esd::block_load` lower to IDENTICAL ISA; the LSC knob is a no-op on this compiler |
| dpas F32 stage order | not decidable from asm | adjudicated empirically by the bitwise correctness phase (bitref pair-sum model first; `--seq-stage` regeneration is the declared flip) — still pre-timing |

Marker: `scripts/tmp/rq26_f1_cases/ISA_GATE_PASSED` created by THIS session
(b70-gemm-kernel-engineer) on the strength of the table above, with this
provenance note. It does NOT carry an independent b70-isa-analyst verdict;
that review remains outstanding and any headline use of the round-1 numbers
should say so.
