# Hyperframes Composition Brief: Intel Arc Pro B70 Inference Cookbook

## Objective
Create a short deadpan launch-style brag video for the Intel Arc Pro B70
Inference Cookbook — a public lab notebook of pinned GPU inference recipes,
where measured numbers ship with their exact conditions and failed
reproductions are published next to the wins.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape — 1920x1080
- Duration: 20 seconds

## Source Material
- Project root: `/home/sergio/projects/intel-arc-pro-b70-inference-cookbook`
- Primary files read: `README.md`, `docs/WHAT-WORKED.md`,
  `data/benchmarks.v1.json`, `docs/assets/b70-qwen38-dashboard.svg`
- Product name: Intel Arc Pro B70 Inference Cookbook
- Tagline / strongest claim: pinned recipes, measured numbers, published
  failures — `170.91 tok/s` single-card decode is the headline record
- Key UI/visual moment to recreate: the repo's benchmark dashboard card
  (teal eyebrow label, big tabular numerals, spec-mode rows with colored
  dots on navy) and a terminal-style canary card showing a failed
  exact-answer probe
- Copy that must appear verbatim:
  - `170.91 tok/s`
  - `CLIENT POST-FIRST DECODE · C1 · MEDIAN N=5`
  - `Qwen3.6-35B-A3B · GPTQ-INT4 · MTP4 · one Arc Pro B70 · 165 W`
  - `EVERY NUMBER SHIPS WITH ITS STACK`
  - `image · patch stack · model revision · power cap · timing definition`
  - `We tried to reproduce that record on a later stack.`
  - `It did not reproduce.`
  - `We published that too.`
  - `sum(i*i for i in range(4))`
  - `30` (in red — the wrong answer)
  - `correct: 14 · deterministic ×2`
  - `Every recipe passes an exact-answer gate.`
  - `Intel Arc Pro B70` / `Inference Cookbook`
  - `Pinned recipes. Measured numbers. Published failures.`
  - `github.com/SergiioB/intel-arc-pro-b70-inference-cookbook`

## Creative Direction
- Tone preset: `deadpan`
- Creative direction: a dramatic launch film for an extremely honest lab notebook
- Interpretation: slow crossfades (0.8-1.0s), long holds, sparse large type,
  generous empty navy space, zero exclamation, one observation per scene. The
  restraint is the joke — the repo's own voice is already this dry.
- Angle: play it completely straight — a polished launch film for a repository
  of markdown measurement tables. Every big claim carries its conditions.
- Hook: a giant `170.91 tok/s` counter climbing in tabular numerals over a navy
  void, with the measurement conditions as the sub-line.
- Outro / punchline: `Pinned recipes. Measured numbers. Published failures.`
  Long quiet hold.
- Avoid:
  - Generic SaaS language ("streamline", "supercharge", "blazing")
  - Abstract filler visuals, particle fields, glowing GPU renders
  - Redesigning the project's visual identity — match the dashboard look
  - Any private host paths, IPs, unpublished data, or invented numbers —
    every figure on screen must be one of the real records listed above

## Visual Identity
- Background: `#07111f`
- Card surface: `#0f1d30` / `#12243a`; grid lines `#263a53`
- Text: `#f5f8fc` primary, `#9caec4` muted
- Accent: `#55d6be` teal, `#68a7ff` blue, `#ffb45c` amber; spec-mode legend
  colors gray `#aeb9c7` / teal / blue / amber; fail red for the wrong answer
- Display font: Inter, weight 750–800, `font-variant-numeric: tabular-nums`
- Body font: Inter, weight 500; mono code line in a monospace stack
- Visual references: `docs/assets/b70-qwen38-dashboard.svg`,
  `docs/assets/b70-benchmark-method.svg`

## Storyboard
Use the storyboard in `brag-output/brag-plan.md` as the creative contract.

Scene summary:
1. The number — 4.5s — eyebrow, giant counting `170.91`, conditions sub-line
2. The receipts — 5s — dashboard card, 4 spec-mode rows arrive on beat grid,
   conditions line
3. The published failure — 5.5s — reproduction lines, then canary code card
   with red `30` payoff
4. Outro — 5s — product name, tag, repo URL, long hold

## Audio
- Audio role: quiet cinematic bed; room tone under a very dry delivery
- Audio arc: constant low bed, gentle fade in the last ~1.5s
- Music: `assets/music/happy-beats-business-moves-vol-12-by-ende-dot-app.mp3`
- Music treatment: data-start 0, volume 0.15, fade out over the final ~1.5s
- Music cue guidance: bundled preset at
  `<skill-dir>/assets/music/cues/happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.json`
  (~110 BPM, beat ~0.545s). Strong-cue candidates: 8.74s (S2 open), 13.11s
  (red `30` landing — the payoff), 17.47s or 18.56s (logo lockup). Snap the
  four S2 rows to every other beat (~1.09s apart) so each stays readable.
- Audio-reactive treatment: subtle — the giant number's teal glow and card
  surfaces may breathe very slightly with RMS. No strobing, no equalizers.
- Audio-coupled moments:
  - Scene 1 — counter settle, one soft accent
  - Scene 2 — one quiet accent on first row only (rows are text: keep reveals
    to every other beat, do not fire SFX per row)
  - Scene 3 — one dry low impact exactly when the red `30` lands
  - Scene 4 — soft bell on the name landing
- SFX selection guidance: sparse polished cues — `interface/drop_*` or
  `casino/chip-lay-*` for the counter settle, `impact/impactSoft_medium_*` or
  `interface/bong_001` for the `30` payoff and logo. See
  `<skill-dir>/assets/sfx/sfx-analysis.md`.
- Exact SFX choice: Hyperframes picks filenames/timestamps to match the
  implemented animation.
- Audio files: copy chosen music + SFX into `brag-output/composition/assets/`

## Hyperframes Instructions
Load the composition-building Hyperframes domain skills — `hyperframes-core`
(composition contract + `data-*` timing), `hyperframes-animation` (motion),
`hyperframes-creative` (design spec, beats, audio-reactive),
`hyperframes-keyframes` (seek-safe keyframes), and `hyperframes-cli`
(lint/check/render). /brag is its own workflow: do not enter the `hyperframes`
entry-point intent interview or route into its generic promo / launch-video
workflow. Prefer native Hyperframes conventions over anything in `/brag`.

Requirements:
- Show at least one real UI, copy, or visual element from the source project
  (the dashboard card and the canary probe are both recreated from real
  repo material).
- Keep all text readable in the final render.
- Keep the video within 15-25 seconds.
- Include the planned music/SFX layer.
- Treat `/brag` audio notes as guidance; choose SFX after visuals exist.
- Major reveals may move toward nearby strong cues within ±0.15s; use 1-3
  strong cue locks max. Sequential text rows snap to every other beat.
- Use local assets for audio and any required runtime/media dependencies.
- Run `hyperframes check` before render — it is brag's single gate.
