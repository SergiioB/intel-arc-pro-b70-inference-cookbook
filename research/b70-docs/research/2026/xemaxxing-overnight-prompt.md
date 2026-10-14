---
title: "OVERNIGHT BUILD PROMPT — XeMaxxing landing mockup"
type: research
date: 2026-08-20
status: final
---

# OVERNIGHT BUILD PROMPT — XeMaxxing landing mockup

**You are an overnight implementation agent.** Build a high-fidelity **static mockup** of an independent Intel Arc / Xe2 local-inference hub. Ship a running local site before morning. Do not register domains, do not deploy publicly, do not post, do not commit secrets, do not invent benchmark numbers.

Read this whole prompt before writing files. If a later section conflicts with an earlier one, the **legal / claims / hardware-truth** sections win.

---

## 0. Mission

Create a **simplistic, modern, easy-on-the-eyes** site that still feels *highly interactive* and *Intel-silicon-specific*.

**Design fidelity:** match Cascadia’s *art direction and choreography as closely as a sibling site can* — same black void, type scale, button pair, city-of-lights hero, hoverable units, scroll-to-square-die with spectral frame, then sparse sections. A stranger should think “this is in the same visual family as cascadia.to,” not “a generic Three.js portfolio.”

**Do not steal Cascadia.** Do not reuse their GLB, JS chunks (`CityGLBProto-*`), logo, wordmark, cabin photos, fleet copy, or Intel endorsement quote. Rebuild the scene with original meshes and XeMaxxing copy. Copyright-safe twin, not a fork.

This mockup must make a visitor understand, in 20 seconds:

1. Independent recipes and honest benchmarks for Intel Arc Pro GPUs.
2. The hardware is Battlemage (Xe2): Xe-cores, XMX, slices, GDDR6 bandwidth.
3. The work already lives on GitHub (cookbook + B70-DOCS).
4. Engines: llama.cpp SYCL and vLLM XPU are the current paths; OpenVINO is a research/historical lane, not the default.

---

## 1. Product decision (do not reopen overnight)

### Name

**Ship the mockup as `XeMaxxing`.**

| Candidate | Verdict for this mockup |
|---|---|
| Intelmaxxing | **Do not use as brand or domain.** Puts `Intel` in the product name. Endorsement/confusion risk. Also rhymes with LocalMaxxing in a small niche. Existing internal plan already said: park-and-301 only, DM LocalMaxxing first. |
| Arcmaxxing | Better, still uses Intel’s `Arc` mark as the brand. Avoid as primary name. |
| **XeMaxxing** | **Use this.** Architecture-flavored, silicon-accurate, not an Intel product name. Domain ideas (do not buy overnight): `xemaxxing.dev`, `xemaxxing.org`. |

Secondary lockup on the site (small, never the logo):  
`Independent recipes for Intel® Arc™ Pro`

Never: IntelMaxxing, intelmaxxing.com as a visible brand, “official Intel”, “Intel partner”, Intel logo as the site mark.

### What this product is

An independent hub for:

- hardware reference (B50 / B60 / B70) with **correct bandwidth**
- engine recipes (llama.cpp SYCL, vLLM XPU)
- OpenVINO as an **exploratory** lane
- benchmark cards that obey the B70 claims standard
- links into existing public work

It is **not** a LocalMaxxing clone, not an Intel marketing site, not a Cascadia competitor.

### Existing strategy (respect, don’t implement the business split)

Internal docs already decided:

- Keep authority on `sergiiob.dev` until B70 traffic justifies a split.
- Cookbook: `https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook`
- Research corpus: `https://github.com/SergiioB/B70-DOCS`
- Affiliate, if shown: `https://go.sergiiob.dev/arc-pro` labeled **affiliate**. Do not invent prices or conversion claims.
- Phase-1 of intelmaxxing.com was “register + 301”. **Do not build a second production domain overnight.**

This overnight output is a **visual + IA mockup** that could later live at `xemaxxing.dev` or as `/b70/` on the portfolio.

---

## 2. Legal and brand (non-negotiable)

Intel lists **Intel**, the **Intel logo**, **Arc**, **OpenVINO**, and the **OpenVINO logo** as trademarks. `Xe` and `oneAPI` were not on the public trademark list retrieved 2026-08-20, but treat them as Intel technology names anyway.

### Allowed

- Word-mark references in prose: Intel® Arc™ Pro B70, OpenVINO™ toolkit, oneAPI, Xe2, Battlemage, XMX, Xe-core.
- A footer trademark line.
- Linking to Intel datasheets, OpenVINO GitHub, oneAPI docs.
- Original diagrams of public architecture facts.

### Forbidden

- Intel logo, Arc logo, OpenVINO logo, Intel swirl, copied brand guidelines assets.
- Domain or product title starting with `Intel`.
- “Official”, “certified”, “partner”, “powered by Intel” lockups, fake executive quotes.
- Visual clones of Intel.com, Cascadia, or LocalMaxxing.
- Using Intel blue as a corporate-identity clone. A *hint* of cyan on silicon is fine; the brand must not look like intel.com.

### Required footer

```
Independent project. Not affiliated with, endorsed by, or sponsored by Intel Corporation.
Intel, Arc, and OpenVINO are trademarks of Intel Corporation or its subsidiaries.
XeMaxxing is an independent community research site.
LocalMaxxing is a separate leaderboard; “APPROVED” there means published self-report, not independent reproduction.
```

OpenVINO in body copy: `OpenVINO™ toolkit` on first use, then `OpenVINO`. Link: `https://github.com/openvinotoolkit/openvino`.

---

## 3. Cascadia — copy the design system, replace the silicon

Study `https://cascadia.to/` in a **WebGL-capable** browser (the in-app browser often shows a black hero — that is a bug, not their design). Reference screenshots: night city under the headline; then isometric packed square with orange/green/cyan/violet edge light.

### Copy this design (tight)

- True-black page, starfield, almost no chrome.
- Top bar: circular mark + wordmark left; three dark pill menus; **ghost** secondary CTA; **solid white** primary CTA with arrow. Same weight, same sharpness, same tracking.
- Type: Geist / Geist Mono (or a legal equivalent grotesque + mono). Display ~72px, tight leading, sentence case.
- Hero copy top-left over the 3D scene; scene is full-bleed under the type, not in a card.
- 3D: dense dark blocks, cool window lights, gentle perspective, camera almost locked.
- Hover: a single block/core lifts and identifies itself (tooltip/label). Click is allowed; it must not navigate away.
- Scroll act 2: city **packs into a square die** on a black void; thin spectral rim (warm left, green bottom, cyan/violet right). Split copy: one sentence left, one right, die centered.
- Later page: huge sentence-case H2s, tiny mono kickers, dual product cards, metric row, FAQ accordion, closing CTA pair, sparse footer.
- Buttons: 0–2px radius, white fill / hairline ghost, uppercase or small-caps labels if Cascadia uses them — match their case exactly from the live site (`BOOK A DEMO` energy, not chubby rounded SaaS).

If you have to choose between “original layout experiments” and “Cascadia’s spacing,” pick Cascadia’s spacing.

### Invent (content + metaphor)

Cascadia’s metaphor is **a city of PCs becoming one chip**.  
XeMaxxing’s metaphor is **a city of Xe-cores becoming a Battlemage die**.

| Cascadia | XeMaxxing |
|---|---|
| Night city skyline | Array of Xe-cores / render slices, read as a tiny city of compute |
| Building = network node | Block = Xe-core (8 XVE + 8 XMX) |
| Hover building → node identity | Hover core → slice index, XMX, L1/SLM |
| Click building | Click opens a spec chip (B50/B60/B70 or engine) |
| City morphs into a square processor | Cores pack into BMG-31 floorplan: 8 slices × 4 cores = 32 |
| “Pool every machine” | “Fill the die. Measure the bus.” |
| Cabin demo | Interactive die / bandwidth diagram, not a cabin |

**Reduced-motion fallback is mandatory.** If WebGL fails or `prefers-reduced-motion: reduce`, show a crisp 2D SVG die that still hover-highlights slices. Never ship a black empty hero (that is what a WebGL-less browser looks like — treat it as a bug).

### Visual quality bar for the 3D hero

- Dark navy/black void, sparse stars or dust — not a textured skybox.
- Buildings/cores: dark matte blocks, small cool window lights (cyan/ice, not neon carnival).
- Hover: one core lifts 4–8px, XMX face pulses amber, a floating label appears (`Xe-core 07 · Slice 1 · 8 XMX`).
- Click: a glass spec card, not a route change.
- Morph: isometric pull-up; blocks inset into a square die; edge stroke becomes a 1px spectral frame; four quadrants can map to **compute / bandwidth / VRAM / power** rather than arbitrary rainbow.
- Camera is slow. No orbit-spam. Pointer-driven parallax ≤ a few degrees.
- The 3D scene must never steal clicks from nav or CTAs. Overlay UI is HTML, not baked into the canvas.

Tech for the mockup: **React Three Fiber + drei + three**, or a high-end **SVG/Canvas 2.5D** if R3F is too heavy overnight. Prefer shipping a *beautiful* 2.5D SVG die with SMIL/WAAPI morph over a broken WebGL black rectangle.

---

## 4. Design system (minimal, functional, premium)

Follow the spirit of [jakubkrehel/skills](https://github.com/jakubkrehel/skills): `better-interface` orchestrates `better-accessibility → better-layout → better-writing → better-typography → better-colors → better-ui`.

### Palette (oklch, dark-first)

One neutral ramp + one silicon accent + one warning/status ramp. No extra hues.

- Page: near-black, slightly blue (`oklch(0.14 0.02 250)`), not pure `#000` for large areas of text; the **hero canvas** may go true black.
- Text: off-white `oklch(0.93 0.01 250)`; muted `oklch(0.72 0.02 250)`.
- Accent: ice-cyan for Xe-core lights and links (`oklch(0.78 0.10 210)`).
- XMX / matrix: warm amber, used only on matrix engines and primary CTA fill-invert.
- Bandwidth/bus: green, only on the memory ring.
- Power: soft red/orange, only on TBP chips.
- **Never encode meaning by color alone** — always a label.

Measure contrast. Body text on the page surface must pass. Do not put 12px muted gray on black.

### Type

- Display + UI: one grotesque (Geist, Inter, or IBM Plex Sans).  
- Specs / code / tok/s coordinates: IBM Plex Mono or Geist Mono.
- Max two families.
- Display 56–72px, weight 500–600, tracking slightly negative, `text-wrap: balance`.
- Body 16–18px, line-height 1.5–1.6, measure ~65ch.
- UI labels 13–14px, slight positive tracking, sentence case.
- Do not use ultra-thin weights on body or buttons.

### Layout

- Space groups first. Gap between sections ≥ 2× gap inside a card.
- One alignment edge. Nav, headline, and section kicker share the same inset (~72px desktop, 20px mobile).
- Kickers in mono: `the silicon`, `the cards`, `the engines`, `the recipes`, `the rules`.
- Collapse late. Desktop composition until it actually breaks.
- Full-width media (the die) may bleed; chrome stays inset.

### UI chrome

- Concentric radii: outer = inner + padding.
- Depth from layered transparent shadows; 1px hairline for structure.
- Primary button: white fill, black text, 0px–2px radius (almost sharp). Ghost: 1px white/15% border.
- Press scale `0.96`. Transitions on `transform, opacity, color` only, 150–200ms, `cubic-bezier(0.2, 0, 0, 1)`.
- Focus: visible `:focus-visible` ring, 2px, not `outline: none` without a replacement.
- `prefers-reduced-motion: reduce`: no morph, no parallax, no autoplay. Instant state change + opacity.

### Writing

Clear and brief beats clever. Sentence case. Buttons start with verbs: `Open the cookbook`, `Browse B70 recipes`, `View the die`. No “Learn more” twice. No fake Intel executive quote.

Tone: independent lab, not hype house. Never “data-center throughput” unless a matched, sourced comparison exists — it does not for a homepage headline.

---

## 5. Hardware truth (put on the site; do not round into marketing)

All public spec chips must show **bandwidth**. Wrong bandwidth is a product bug.

### Arc Pro B-series (Battlemage / Xe2)

| SKU | Xe2-cores | XMX (8× cores) | Render slices (4 cores) | VRAM | Bus | Bandwidth | TBP | Notes |
|---:|---:|---:|---:|---|---:|---:|---:|---|
| **B50** | 16 | 128 | 4 | 16 GB GDDR6 | 128-bit | **224 GB/s** | 70 W | Entry Pro |
| **B60** | 20 | 160 | 5 | 24 GB GDDR6 | 192-bit | **456 GB/s** | 120–200 W | Mid Pro |
| **B70** | 32 | 256 | 8 | 32 GB GDDR6 | 256-bit | **608 GB/s** | 230 W | Full BMG-31 / G31 |

Label the table: *SKU marketing figures compiled from Intel positioning and public teardowns (ServeTheHome / TechPowerUp / Puget). Campaigns on this host re-measure VRAM as 32,656 MiB visible and power as a configured cap vs energy-counter draw.*

### B70 die facts for the interactive diagram

Use these as **hover labels**, not as a textbook dump:

- Architecture: Xe2-HPG / Battlemage. Host GPU: Intel Arc Pro B70.
- 32 Xe-cores × 8 XVE × 8 XMX = **256 XMX engines**.
- Each Xe-core: 8 Xe Vector Engines (512-bit) + 8 XMX (2048-bit systolic, DPAS) + shared L1/SLM (Intel doc 192 KB; Chips and Cheese measured a 256 KB L1+SLM block).
- Render slice = 4 Xe-cores. B70 = **8 slices**.
- L2: **18 MB** shared on BMG-31.
- Clocks: graphics base **2.28 GHz**, boost **2.8 GHz** (do not claim a locked clock; `xe` driver cannot lock GT clocks on this host).
- Memory: GDDR6, **608 GB/s**. Decode on this card is often bandwidth-bound; prefill is often compute/XMX-bound. Teach that split on the diagram.
- Intel-quoted INT8: **367 TOPS** (non-sparse, MAC-style). **Do not headline inferred INT4 TOPS.** Internal research inferred ~734 TOPS INT4 MAC / ~1468 TOPS FMA from DPAS dot8 — that is **FLAGGED inference, not spec**. If shown at all, show behind a “inferred, not Intel-published” toggle defaulting **off**.

### Host (only if you mention “this lab”)

- GPU: Intel Arc Pro B70, 32 GB class GDDR6, Linux `xe` driver.
- CPU: AMD Ryzen 7 5700X3D (8C/16T); llama.cpp path uses **8 threads**.
- RAM snapshot: 30 GB class — do not write “64 GB rig” unless verified live.
- Stock power cap max 230 W; lab restores **150 W** after research. Configured cap ≠ measured draw.

### Diagram interaction (the page’s signature)

Build a **simple, accurate** interactive B70:

1. Overview: 8 slices as a 2×4 or ringed rectangle.
2. Click a slice → 4 Xe-cores.
3. Click a core → exploded view: 8 XVE (vector, ice) + 8 XMX (matrix, amber) + L1/SLM block.
4. Toggle overlay: **compute bound (XMX/DPAS)** vs **bandwidth bound (GDDR6 608 GB/s ring)**.
5. SKU switcher: dim unused cores for B50 (16) and B60 (20) so the family relationship is visible. B60 is an awkward 20 — show 5 slices, not a fake 16+4 marketing cartoon.

Keep it *easy on the eyes*: no rainbow dump of every Intel block (no media engines, no display pipes unless a “full die” advanced toggle exists, off by default).

---

## 6. Engines and OpenVINO — honest status

From the B70 engine support matrix (do not “upgrade” status in UI):

| Engine | Status to show | Role on the site |
|---|---|---|
| llama.cpp SYCL | `production` | Default GGUF path. KV policy `q8_0` K + `q4_1` V, flash attention. **Never recommend q4_0 for K cache.** |
| vLLM XPU | `validated_research` | Hugging Face checkpoints; GPTQ-INT4 is a proven *practical* path for exact validated models, not a universal format guarantee. Pin image **digest**, not a floating tag. |
| OpenVINO / OVMS | `historical` | One research card: “bounded engine-sweep observations; not the current canonical serving path.” Invite a future honest re-bench; do not pretend the cookbook ships OpenVINO recipes today. |
| SGLang XPU | `blocked` | Show as blocked/reassess-only. Do not hide it; honesty is the brand. |

OpenVINO section copy (use this stance):

> OpenVINO™ toolkit is Intel’s write-once, deploy-many inference stack, including GenAI / LLM paths. This lab’s current production routes are llama.cpp SYCL and vLLM XPU. OpenVINO appears here as a **lane to re-measure**, not as a copied leaderboard. A successful container start is not a recipe.

Link official docs, don’t wrap their logo.

---

## 7. Information architecture (small site, complete)

Mockup pages — all reachable, all visually finished, dummy-but-honest data:

```
/                     Landing (3D/SVG silicon hero + sections below)
/cards                B50 / B60 / B70 reference (bandwidth mandatory)
/cards/b70            Deep die + host notes
/engines              llama.cpp SYCL, vLLM XPU, OpenVINO, SGLang
/recipes              Links out to the cookbook families
/benchmarks           Protocol explainer + a few result cards
/lab                  What the independent lab is (Sergio, B70-DOCS)
```

Landing sections, in order:

1. **Nav** — wordmark `XeMaxxing`, links: Silicon, Cards, Engines, Recipes, Benchmarks. CTAs: `Open the cookbook` (ghost, GitHub), `Read the protocol` (solid or inverse).
2. **Hero** — Headline idea (pick one, sentence case):
   - `Local inference, measured on the die.`
   - `Xe-cores, honest recipes, no mystery tok/s.`
   Sub: `Independent B70 / B60 / B50 notes. llama.cpp SYCL and vLLM XPU. OpenVINO as a research lane.`
3. **Morph die** — split copy example:
   - Left: `32 Xe-cores. 256 XMX. 608 GB/s.`
   - Right: `Prefill hits the matrix units. Decode hits the bus.`
4. **The constraint** — 32 GB is the product. Not a 5090-killer headline. Position: affordable 32 GB GDDR6 workstation inference, with bandwidth as the decode ceiling.
5. **SKU row** — three cards, B70 featured, each with cores / VRAM / **GB/s** / TBP.
6. **Engines** — four status pills, production → blocked.
7. **Recipes** — three deep links:
   - Cookbook hub
   - Nemotron DFlash family doc (if linking a number, copy coordinates from §8 or show no number)
   - B70-DOCS `AGENTS.md` / protocol
8. **Benchmark rules (short)** — C1 ≠ Cn; cold input ≠ llama-bench pp; cap ≠ draw; LocalMaxxing APPROVED ≠ reproduced.
9. **FAQ** — naming, Intel affiliation, OpenVINO status, why B70, how to reproduce.
10. **Footer** — trademark block, GitHub, sergiiob.dev, affiliate disclosure if the buy link is present.

---

## 8. Benchmarks and claims (hard gate)

Use `.agents/skills/b70-benchmark-claims-standard`. For the mockup:

- **Do not invent numbers.**
- Prefer **no homepage tok/s** over an unqualified number.
- If you show a result card, it must include: model + quant, engine/build identity or `unknown`, C1 vs Cn, prompt/output shape, statistic (median n=5 vs maximum observed), cache state, configured cap, measured draw or `not measured`, evidence link, `PROVISIONAL` unless the card cites a registered campaign.

Forbidden homepage lines:

- `204 tok/s decode` (that was a short-cell maximum-class figure in research notes, not a generic decode claim)
- `7,500 prefill` without “cold input / client TTFT / cache-off”
- any multiplier vs NVIDIA
- “verified by LocalMaxxing”

Correct pattern if you *must* show one lab card (copy verbatim, do not round):

> **Example lab card (illustrative of format, only use if you also link evidence):**  
> Nemotron 3.5 Lightning 30B-A3B, GPTQ-INT4 + DFlash, vLLM XPU, C1.  
> **186.61 tok/s median** at p2048/g128 (n=5).  
> **7,160 tok/s cold input** at p8192/g1 (client TTFT; not isolated engine prefill).  
> Configured cap 150 W. Exact 119,904-token completion passed; 128K not tested.  
> Status: current isolated C1 route on the recorded stack — see model dossier / cookbook family page.

If you cannot attach a real link, **omit the numbers** and show the empty-state: `Numbers live in the cookbook with coordinates. This mockup does not restate them.`

---

## 9. Reference links to wire

- Cookbook: https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook
- Research repo: https://github.com/SergiioB/B70-DOCS
- Portfolio / writeups: https://sergiiob.dev
- Affiliate (optional, labeled): https://go.sergiiob.dev/arc-pro
- OpenVINO: https://github.com/openvinotoolkit/openvino
- Intel Xe GPU architecture (word-mark docs, not logo): Intel oneAPI GPU optimization guide, Xe GPU architecture chapter
- Chips and Cheese Battlemage architecture (for die intuition, credit in a sources popover)
- LocalMaxxing: companion leaderboard, not a competitor and not a verifier

---

## 10. Implementation (overnight)

### Stack

- **Astro** + a single React island for the die, **or** Vite + React if faster.
- Tailwind (or vanilla CSS with custom properties). No UI kit that looks like shadcn-demo.
- No CMS, no auth, no Worker API, no benchmark submission (that is a later phase).
- Static JSON for SKUs and engine status.

### Quality bar

- Pixel-tight at 1440×900 and 390×844.
- Keyboard: tab through nav, SKU cards, die slices (roving tabindex on the die).
- Screen reader: the canvas has a text alternative describing 32 cores / 8 slices / 608 GB/s.
- No layout shift when the 3D island hydrates; reserve height.
- If WebGL context is lost, swap SVG instantly — **never leave black**.

### Visual audit loop (do this before sleep and again before “done”)

1. Look at the hero at 1×. If it looks like a dashboard, delete chrome.
2. Hover every core/slice. If nothing speaks, the interaction failed.
3. Scroll the morph. If it does not feel like *packing a die*, retune.
4. Check B50/B60/B70 bandwidth numbers against the table in §5.
5. Grep the build for `tok/s`, `faster than`, Intel logo SVGs, `official`.
6. Toggle reduced motion. Site must remain complete.
7. Read the footer aloud. Affiliation must be unmistakable.

### Out of scope overnight

- Real Cloudflare Workers submission API
- Domain purchase
- Amazon BOM page beyond a single labeled affiliate line
- Re-running GPU benchmarks
- Copying Cascadia’s GLB or JS
- Using Intel brand assets

---

## 11. File / copy checklist for morning

- [ ] Brand is **XeMaxxing**, not Intelmaxxing
- [ ] No Intel / Arc / OpenVINO logos
- [ ] Trademark + independence footer
- [ ] B50 224 GB/s, B60 456 GB/s, B70 608 GB/s
- [ ] Interactive die: slices → cores → XVE/XMX
- [ ] Compute vs bandwidth overlay
- [ ] llama.cpp production, vLLM validated_research, OpenVINO historical, SGLang blocked
- [ ] Cookbook + B70-DOCS + sergiiob.dev linked
- [ ] Zero naked tok/s
- [ ] WebGL failure and reduced-motion SVG fallbacks
- [ ] Cascadia-tight art direction (black, Geist, sharp CTAs, city→die morph, spectral frame) with **original** meshes and copy — no stolen GLB/JS

Ship a `README` in the mockup repo: how to run, what is dummy, what is sourced.

When you stop, leave the preview running locally and a short morning note: what is gorgeous, what is still a stub, any number you were tempted to put on the hero and didn’t.
)
