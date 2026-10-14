---
title: "SUPERSEDED 2026-09-01"
type: plan
date: 2026-09-15
status: final
---

# SUPERSEDED 2026-09-01

Live hub is **XeCores** (`xmxlab`, xecores.com) with D1 review intake.
This plan is historical. Do not register intelmaxxing.com or build a KV Worker from it.

Agent system: `docs/governance/agent-operating-system.md`
Site contract: `~/projects/xmxlab/AGENTS.md` + `DESIGN-AGENT-SYSTEM.md`

# intelarcmaxxing.com — hub plan

Strategic doc for the Intel Arc inference hub. Decision pending — this is the
plan, not a build order.

## The strategic question

You have a growing body of Intel Arc B70 inference work that's genuinely best-
in-class (the MTP unlock, the full grid, the power sweep). The question is
whether to:

- **(A) Keep it on sergiiob.dev under `/b70/`** — one site, one domain to rank,
  leverage the existing portfolio authority.
- **(B) Spin off intelarcmaxxing.com** — a dedicated hub domain.

## My recommendation: (A) first, (B) only if traffic justifies it

### Why defer the domain split

1. **sergiiob.dev already ranks and has a `/b70/` category.** The archive has
   a featured "Intel Arc B70" hub card with a GPU diagram. Splitting now
   fragments domain authority you've already built.
2. **One site to maintain.** Astro is already wired; adding a second deploy
   pipeline, second analytics, second SEO surface is overhead that pays off
   only at scale.
3. **The brand is you, not the card.** sergiiob.dev carries your credibility
   across topics (Linux, automation, local AI). A card-specific domain dilutes
   that.
4. **Domain age matters for SEO.** A new domain starts at zero. sergiiob.dev
   has history; new B70 content inherits it.

### When to spin off intelarcmaxxing.com

Trigger the split only when **all** of these are true:

- [ ] B70/Arc content is >50% of your monthly traffic for 2+ months
- [ ] You have 10+ B70-specific posts (currently ~9 — close)
- [ ] You're hitting a distinct audience that doesn't convert to the rest of
      sergiiob.dev (check analytics: are B70 visitors bouncing or browsing?)
- [ ] The Amazon affiliate revenue justifies the extra domain cost + effort

If you hit those triggers, the migration path is clean: the `/b70/` category
becomes the new domain's root, with 301 redirects from sergiiob.dev/posts/b70-*
to intelarcmaxxing.com.

### The name caveat

"intelarcmaxxing" rides on the `localmaxxing.com` brand (the leaderboard site).
Before committing to it publicly:

- **Check the relationship.** Is localmaxxing.com run by someone you can ask?
  Friendly homage vs confusing-derivative is a real distinction in a small
  niche. A DM to clarify goes a long way.
- **Have a fallback.** `arc-pro-b70.guide`, `intel-gpu-inference.dev`, or just
  `b70.sergiiob.dev` (a subdomain) all work and avoid the brand-overlap question.
  A subdomain is the lowest-risk version of (B).

## What the hub should contain (whichever domain)

### Tier 1 — the anchor content (exists, needs hub-card wiring)

- **vLLM vs llama.cpp showdown** (the post we just shipped) — the headline
- **Full recipe post** (`intel-arc-pro-b70-full-recipe-llama-cpp-sycl.md`) —
  the living source of truth for llama.cpp numbers
- **High-concurrency MoE** post
- **SYCL XMX quantized-KV prefill** post
- **KV cache quantization** post
- **MTP power scaling** post
- **B70 boundary sweep / context ceilings** post

### Tier 2 — the "buy and build" funnel (new content, monetizes)

- **"Should you buy an Arc Pro B70 for AI in 2026?"** — buying guide, every
  Amazon affiliate link lives here. This is the money page.
- **B70 parts list** — the full rig (card, PSU, motherboard, RAM, case, NVMe,
  the Radxa bridge). Each line item = affiliate link.
- **B60 vs B70** — the 16GB vs 32GB decision.
- **B70 vs RTX 4060 / 7600 XT** — the "why not Nvidia at this price" post.
  Cross-hardware comparison table (you have the data).
- **B70 vs Mac M-series unified memory** — the "32GB VRAM for $600 vs 64GB
  unified for $2000" angle.

### Tier 3 — the community/recurring content (retention)

- **Weekly/monthly build tracker** — "what changed in llama.cpp SYCL / vLLM
  XPU this month." Low-effort if you're already tracking it; high subscription
  value.
- **User-submitted benchmarks** — a form/discord link. Builds the community.
- **The dense FP8 kernel watch** — track the upstream gap; becomes a story
  when it lands.

## Monetization wiring (the affiliate strategy)

### Amazon affiliate — where the real money is

The B70 inference audience is in a **buying mood**. They found your recipe
because they're about to spend $600+ on a card. Capture 2-4% of that.

Every B70 post should have an **"Build this rig" callout box** with:

- Arc Pro B70 — `go.sergiiob.dev/arc-pro`
- Recommended PSU (750W+ Gold)
- Motherboard + CPU combo (we run Ryzen 7 5700X3D)
- RAM (32GB DDR5)
- NVMe (the model SSD)
- Case + cooling
- Optional: the Radxa ROCK 5B+ for the pi-bridge front-end

This box is the primary monetization. It's honest (you actually use this rig),
it's useful (saves people research), and it converts.

### Ko-fi — voluntary tip jar

"Buy me a bench hour" — $5-10 optional tip. Open-source people respect this.
Wire it as a small callout in the footer of each post + the GitHub README.
**Never gate content behind it.**

### What NOT to do

- ❌ Paywall the recipes (derivative of open source; reputationally risky)
- ❌ Paywall the patches (they'll leak; the GitHub repo is the asset)
- ❌ Ads (destroys the technical-blog credibility; affiliate is cleaner)

## Concrete next steps (in priority order)

1. **✅ Ship the showdown post** (done — commit `781874b`, with SVGs)
2. **✅ Ship the GitHub repo** (done — `SergiioB/intel-arc-pro-b70-inference-cookbook`)
3. **Wire the affiliate "Build this rig" callout box** as a reusable Astro
   component. Drop it into every B70 post. **This is the #1 monetization lever
   and it's currently missing.**
4. **Write the "Should you buy an Arc Pro B70 for AI?" buying guide** — the
   Tier 2 anchor. Every affiliate link lives here.
5. **Add the parts list page** (B70 rig BOM).
6. **Post the Twitter thread** (drafted in `social-assets/social-posts-b70-cookbook-launch.md`).
7. **Track traffic for 4-8 weeks.** If B70 content >50% of traffic and growing,
   revisit the intelarcmaxxing.com domain split. If not, you've lost nothing
   and kept one site.

## Domain registration (only if you decide to proceed)

If you do want intelarcmaxxing.com (or a variant) parked for later:

- Register it now (~$12/yr) so nobody squats it.
- 301 redirect it to `sergiiob.dev/b70/` until you're ready to split.
- Do NOT build the separate site until the traffic triggers fire.

## Summary

**Keep it on sergiiob.dev for now. Wire the affiliate box. Write the buying
guide. Post the thread. Watch traffic. Split the domain only if the data says
to.** The open GitHub repo + the showdown post are the catalyst; the affiliate
funnel is the business. The domain split is a maybe-later.
