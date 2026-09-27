# Brag Plan: Intel Arc Pro B70 Inference Cookbook

## What is this app?
Not an app — a public lab notebook. Pinned vLLM XPU and llama.cpp SYCL inference
recipes for the Intel Arc Pro B60/B70 GPU, where every speed claim ships with its
exact image, patch stack, model revision, power cap, and timing definition — and
failed reproductions are published next to the wins.

## The angle
Play it completely straight: a polished launch film for a repository of markdown
tables and measurement gates. The deadpan joke is that this project does the one
thing launch videos never do — it publishes the numbers that didn't hold up. Every
big claim on screen carries its own conditions, because that's what the repo does.

## Hook (first 2-3 seconds)
A navy void. A small teal eyebrow: `CLIENT POST-FIRST DECODE · C1 · MEDIAN N=5`.
Then a giant tabular-numeral counter climbs to **170.91 tok/s** — the repo's
single-card decode record (Qwen3.6-35B-A3B, GPTQ-INT4, MTP4, one B70, 165 W).
Under it, small: `one GPU · pinned image · cache cold`. The number IS the hook;
the conditions attached to it are the personality.

## Key moments (the middle)
- **The receipts table.** A recreation of the repo's real dashboard aesthetic:
  spec-mode rows (No spec / MTP1 / MTP2 / MTP4) with real measured numbers in
  tabular numerals, arriving row by row. Eyebrow: `EVERY NUMBER SHIPS WITH ITS STACK`.
- **The published failure.** Deadpan beat: `We tried to reproduce that record on
  a later stack.` → `It did not reproduce.` → `We published that too.` This is
  the actual content of docs/WHAT-WORKED.md — the differentiator.
- **The exact-answer gate.** A terminal-style card: the canary prompt
  `sum(i*i for i in range(4))`, the serving checkpoint's answer `30` landing in
  red, then `correct: 14 · deterministic ×2`. The 35B model is confidently wrong,
  and the repo is the thing that catches it.

## Outro / punchline
`Intel Arc Pro B70 Inference Cookbook` — then the tag:
`Pinned recipes. Measured numbers. Published failures.` Long quiet hold.
Tiny repo URL under it.

## User flow worth showing
Not a click-through app. The repo's flow is: pick a route → apply the pinned
image + ordered patch stack → reproduce a measured number → pass the exact-answer
gate → publish or block. The centerpiece scenes dramatize the *measurement
discipline*: the spec-mode table (scene 2) and the canary gate (scene 3) are the
"product in use" moments.

## Tone
- Preset: `deadpan`
- Creative direction: a dramatic launch film for an extremely honest lab notebook
- Interpretation: slow crossfades, long holds, sparse large type, generous empty
  navy space, zero exclamation. One observation per scene. The restraint is the
  joke — the repo's own prose voice is already this dry.

## Format: landscape — 1920x1080
## Duration: 20s

## Visual identity (from the project)
- Background: `#07111f` (deep navy, from the repo's SVG dashboards)
- Card surface: `#0f1d30` / `#12243a`, grid lines `#263a53`
- Accent: `#55d6be` (teal), secondary `#68a7ff` (blue), `#ffb45c` (amber), fail red
- Text: `#f5f8fc` primary, `#9caec4` muted
- Display font: Inter (ui-sans-serif fallback stack), weight 750–800, tabular-nums
- Body font: same stack, weight 500
- Strongest visual element: the benchmark dashboard card — teal eyebrow label,
  big white numbers, spec-mode rows with colored dots
  (docs/assets/b70-qwen38-dashboard.svg is the reference)

## Share copy (draft)
170 tok/s on one Intel Arc Pro B70 — and every number comes with receipts,
including the record that didn't reproduce.

## Audio direction
- Role: quiet cinematic bed; the room tone under a very dry delivery
- Music: `happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` (steady, clean)
- Music treatment: start at 0s, very low volume (0.15), constant, gentle fade in
  the last 1.5s. Deadpan posture — the music is felt, not heard.
- Music cue guidance: bundled preset
  `assets/music/cues/happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.json`
  (~110 BPM). Strong-cue candidates near major moments: 8.74s (S2 open),
  13.11s (the red `30` landing), 17.47s/18.56s (logo lockup). Beat grid ~0.545s
  spacing for the S2 row-by-row reveals — snap rows to every other beat
  (~1.09s apart) so each row is readable.
- Audio-reactive treatment: subtle; let the big number's teal glow and the card
  surfaces breathe very slightly with RMS. No strobing, no equalizers.
- SFX posture: very sparse, 2–3 cues total — a soft tick/drop when the counter
  lands, one dry low impact when the red `30` lands, one soft bell on the outro
  card. No keyboard chatter.
- Audio-coupled moments: counter settle, the red `30` reveal, logo lockup.
- Restraint rule: no SFX on the row reveals beyond a single quiet accent; never
  let audio get louder than the deadpan.

## Storyboard

### Scene 1 — The number — 4.5s
Navy void. Teal eyebrow fades in: `CLIENT POST-FIRST DECODE · C1 · MEDIAN N=5`.
Giant `170.91` counts up in tabular numerals, `tok/s` beside it. Sub-line:
`Qwen3.6-35B-A3B · GPTQ-INT4 · MTP4 · one Arc Pro B70 · 165 W`.
Sequential/interaction: counter ticks up; eyebrow and sub-line fade in after.
Audio intent: near-silent room; music bed fades in, a few quiet ticks under the
counter.
Audio-coupled idea: soft accent when the counter settles (~3.5-4s).
Transition mood: slow crossfade → Scene 2

### Scene 2 — The receipts — 5s
A dashboard card in the repo's real style slides up: teal eyebrow
`EVERY NUMBER SHIPS WITH ITS STACK`, then four spec-mode rows (No spec / MTP1 /
MTP2 / MTP4) with real measured decode figures in tabular numerals, colored dots
matching the dashboard legend (gray / teal / blue / amber). Deadpan line under:
`image · patch stack · model revision · power cap · timing definition`.
Sequential/interaction: yes — the four rows arrive one by one on the beat grid
(every other beat, ~1.09s apart); each row holds.
Audio intent: bed continues; one quiet accent on first row only.
Audio-coupled idea: beat-grid row reveals.
Transition mood: slow crossfade → Scene 3

### Scene 3 — The published failure — 5.5s
Two beats. First: centered line, `We tried to reproduce that record on a later
stack.` then below it `It did not reproduce.` and a small teal line
`We published that too.` Second: the canary card — mono code line
`sum(i*i for i in range(4))`, then the checkpoint's answer `30` lands in red
(fail), with `correct: 14 · deterministic ×2` beneath. Final line:
`Every recipe passes an exact-answer gate.`
Sequential/interaction: the three reveal lines then the code/answer pair appear
in order; the red `30` is the payoff moment.
Audio intent: still quiet; one dry low impact exactly when `30` lands.
Audio-coupled idea: beat-locked `30` reveal (~13.1s strong cue).
Transition mood: long hold, slow crossfade → Scene 4

### Scene 4 — Outro — 5s
Empty navy. `Intel Arc Pro B70` fades up large, `Inference Cookbook` under it.
Tag: `Pinned recipes. Measured numbers. Published failures.` Then small:
`github.com/SergiioB/intel-arc-pro-b70-inference-cookbook`. Long hold into fade.
Sequential/interaction: name, tag, URL arrive in order with generous gaps.
Audio intent: soft bell accent on the name landing; music fades under.
Audio-coupled idea: logo lockup near ~17.5-18.6s strong cue.

**Music mood for this video:** deadpan — steady, clean, low.
**Audio summary:** a barely-there corporate bed with two or three dry accents,
like a conference talk where the speaker never raises their voice.
