# Session stability: judge long-context routes by the whole session, not one cell

A single decode figure at a fixed depth cannot answer the question a long coding session
actually asks: *does the speed hold up hour after hour as the context grows, or does it
climb down a cliff?* Two engines with identical "decode at 128K" numbers can behave very
differently by turn 40. Since 2026-11-09 we record a session trace for long-context
claims before calling a route usable.

## The measurement

`session_stability.py` (in each run's evidence dir) holds ONE conversation for the whole
test: turn 0 sends a deep seed prompt (the `depth_bench.py` corpus+task cell for that
depth), then 10 short follow-up tasks alternate — summarize, list files, write a unit
test, refactor, halve the words, add error handling — and every assistant reply is
appended, so context grows on every turn. Per turn it records TTFT, decode tok/s,
per-64-token decode windows inside the turn, context size, card energy and temperature.

Read it as three numbers:

1. **Turn-0 decode** — the depth cell you already knew.
2. **Last-turn decode vs turn-0** — the sag (or none) across the session.
3. **Cached TTFT on turns ≥ 1** — whether the KV prefix cache makes follow-ups cheap.

## First two sessions (2026-11-09)

| Route | Seed depth | Turns | Decode every turn | Sag | Cached TTFT |
|---|---|---|---|---|---|
| Ripple-65B `Q4_K_M`, 2× B70, `-sm tensor`, f16 KV | 118K | 11 | 9.65–9.71 tok/s | **0.0 %** | ~9 s |
| Qwen3.8-27B `Q4_K_M`, 1× B70, q8_0 KV | 131K | 11 | 17.1–17.5 tok/s | −0.5 % (last turn faster) | 0.6 s |

Evidence: [`results/ripple-65b/depth-256k-session-20261109/ripple-session/`](../results/ripple-65b/depth-256k-session-20261109/)
and [`results/llamacpp-sycl-kv-depth-20261108/qwen38-session-128k-20261109.jsonl`](../results/llamacpp-sycl-kv-depth-20261108/).

Both routes hold their speed for the full session; neither shows a ramp-down, and the
per-window traces inside turns are flat. The cached-TTFT gap (9 s vs 0.6 s) is the
two-card 65B vs one-card 27B prefill difference showing up in follow-up latency.

## Caveats

- Ripple's template ignores `enable_thinking`, so its session replies streamed as
  reasoning only; the appended turn content is empty. Decode rates cover all streamed
  tokens and are unaffected, but the conversation's content history is thinner than a
  real chat — recorded, not hidden.
- Eleven turns is a screen, not a soak. If a route sags only after hours (thermal or
  fragmentation effects), this will not catch it; the energy counter and temperature
  per turn give the first hint.
- Prefix cache is warm by construction after turn 0. A route whose cache evicts under
  load (multi-user) can still re-prefill everything; this harness is single-stream.

## Rule of thumb

When a new long-context route or patch lands: one depth row is not a verdict. Publish
the depth row **and** the session trace, and say "usable at N tokens" only if the last
turn looks like the first.
