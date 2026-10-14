---
title: "Qwen3.8-27B pi-agent screenshot loop — observed failure mode (2026-08-16)"
type: research
date: 2026-08-16
status: final
---

# Qwen3.8-27B pi-agent screenshot loop — observed failure mode (2026-08-16)

## Summary

During the vision-verified game campaign, the agent entered a degenerate
runaway loop: after producing a complete `game.html` (32 KB) and taking one
screenshot, it re-ran the **exact same chromium screenshot command** ~90 times
at a ~9 s cadence (~15 min), never reading the PNG again, never editing the
file, never replying a summary. The harness's 1 h cap had to cut it.

## Evidence

- Run: `results/2026/08/qwen38-quality-20260815/pi-games-vision/_evidence-loop-01/`
  (session.json 280 KB, game.html 32,422 B, game.png)
- Action timeline from the session: 99 actions = 90× `bash` (identical
  `chromium --headless ... --screenshot=game.png ...`), 3× `read`, 1× `write`,
  1× `edit`, 4× text. The repeated bash calls differ only in nothing — same
  command string, timestamps ~7-9 s apart (chromium startup dominates).
- `game.html` mtime frozen for 15+ minutes while the loop ran.

## Root-cause hypothesis (unverified beyond observation)

- Classic tool-call echo / retry loop: the model keeps re-invoking a tool
  whose result it does not (or cannot) consume. The chromium command's stderr
  contains GL noise (`ERROR:gpu/...`, `GL Driver Message ... ReadPixels`) that
  surfaces in the tool result; a plausible trigger is the model seeing
  "errors" and retrying the capture instead of reading the PNG.
- Prefix caching made the loop *fast* (each turn ~9 s of chromium + ~1 s of
  cached prefill vs ~30 s uncached), so the loop burned turns quickly — the
  fix is detection, not speed.
- Not reproduced with the anti-loop prompt rule yet (see below).

## Fixes applied (runner + prompt)

1. **Runner-side loop detector** (`b70-qwen38-pi-games-runner.py`):
   poll the pi session every 20 s; if the last 8 tool calls are byte-identical
   AND `game.html` mtime has not changed since, kill pi and finalize the last
   good artifact with `exit="loop"`. A legitimate repeated command is
   impossible for 8 consecutive calls without a file change in this workload.
2. **Prompt anti-loop rule** (AGENT_INSTRUCTION §6): "never run the same
   command twice in a row; after a screenshot you MUST read the PNG, judge it,
   and either fix something or finish."
3. **Stall guard** (second observed failure mode): even with varied commands
   the agent refused to terminate — after the game was complete it spent 15+
   minutes re-testing chromium/WebGL availability ("WebGL is genuinely
   unavailable in this headless environment") despite its own screenshot
   rendering the scene. Root cause: the model misreads swiftshader's stderr
   GL noise (`GPU stall due to ReadPixels`, `ContextResult::kTransientFailure`)
   as failure. Fixes: (a) runner kills pi when `game.html` is unchanged for
   20 min while the session keeps growing (`exit="stall"`); (b) prompt rule §7
   tells the model the GL noise is expected and to judge only PNG pixels.
4. **6 h cap** (was 1 h — games must be allowed to finish), per-artifact
   chromium load-check, `exit` ∈ {0, loop, stall, timeout, rc}.
