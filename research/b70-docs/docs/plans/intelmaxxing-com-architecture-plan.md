---
title: "SUPERSEDED 2026-09-01"
type: plan
date: 2026-08-06
status: final
---

# SUPERSEDED 2026-09-01

Live hub is **XeCores** (`xmxlab`, xecores.com) with D1 review intake.
This plan is historical. Do not register intelmaxxing.com or build a KV Worker from it.

Agent system: `docs/governance/agent-operating-system.md`
Site contract: `~/projects/xmxlab/AGENTS.md` + `DESIGN-AGENT-SYSTEM.md`

# intelmaxxing.com — Architecture & Security Plan (2026-08-06)

## DECISION SUMMARY

**Phase 1 (now):** Register intelmaxxing.com in Cloudflare. Park as 301 redirect to sergiiob.dev/b70/. Build nothing separate yet.

**Phase 2 (when traffic justifies):** Stand up as Cloudflare Pages site with serverless API on Cloudflare Workers. Benchmarks/recipes as static content (Astro). Benchmark submission API as hardened Worker.

Follows the hub-plan recommendation: keep authority on sergiiob.dev, split only when B70 traffic >50% for 2+ months.

## HOSTING: Cloudflare Pages vs Alternatives

### Recommendation: Cloudflare Pages + Workers

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| Cloudflare Pages | DNS already there. Free 500 builds/mo. Static = zero attack surface. Workers = serverless API. | Build limits, 100K Worker req/day free | USE THIS |
| Vercel | Better DX, edge functions | Separate account, lock-in | Skip |
| Self-hosted (B70/Radxa) | Full control | Hardware is for inference. Opening ports = attack surface. DDOS risk. | Skip |
| Firebase Hosting | Already use for IntelliAuto | Not Cloudflare-native, less suited for static | Skip |

### Why Cloudflare Pages wins
1. DNS already there — all domains in Cloudflare. Adding a site = `wrangler pages deploy`.
2. Zero server to harden — static HTML/CSS/JS from CDN. No RCE risk on content layer.
3. Workers for API — serverless, no persistent server, no SSH, no ports.
4. Free tier is enough — 500 builds/mo, unlimited bandwidth, 100K Worker req/day.
5. DDOS protection included — Cloudflare edge handles it.

## CONTENT LAYER (Static — Cloudflare Pages)

### Tech stack
- Framework: Astro (same as sergiiob.dev)
- Content: Markdown to static HTML. No CMS, no database, no reader login.
- Build: `astro build` then `wrangler pages deploy dist`
- Repo: github.com/SergiioB/intelmaxxing (new)

### Content structure
```
/                       Landing
/buyers-guide/          Affiliate rig BOM
/benchmarks/            Cross-engine tables (from JSON)
/recipes/               Setup guides (SYCL build, vLLM patches)
/cards/                 Per-card reference (B70, B60, A770)
/api-docs/              How to use benchmark API
```

### Benchmark data
- Source of truth: `benchmarks.json` in repo (generated from B70-DOCS)
- Benchmark page reads JSON client-side. No server needed.
- Updates = commit JSON + rebuild. 30 seconds.

## API LAYER (Cloudflare Worker — benchmark submission)

### What it does
Accepts benchmark JSON from community (same schema as localmaxxing.com). Stores in KV. Manual review before publish.

### Architecture
```
[Community user with lmx CLI]
  -> [Worker: POST /api/v1/benchmarks]
  -> [KV: pending submissions]
  -> (manual review)
  -> [KV: approved benchmarks] -> [benchmarks.json] -> [static rebuild]
```

### Why KV, not D1/R2
- KV (Workers KV): Key-value. Free 1K writes/day, 100K reads/day. No SQL injection (no SQL).
- D1 (SQLite): Overkill, adds SQL injection surface.
- R2: Good for large files. Use later for screenshots.

## SECURITY HARDENING (CRITICAL)

### 1. Authentication
- API key required for every write (Bearer imx_...). Keys in KV with metadata.
- Admin key (Cloudflare secret) for approve/delete.
- No user accounts. API-key-based only. Minimal attack surface.

### 2. Input validation (Worker-level)
- Strict schema: hfId regex, quantization enum, numeric bounds (tokSOut >0 and <10000).
- Max body size 10KB (benchmark JSON is ~1KB). Reject larger.
- Rate limit: 10 submissions per key per day (KV counter).
- No HTML/markdown in user input. Strings/numbers only. Zero XSS surface.

### 3. CSP headers
```
Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self'; frame-ancestors 'none';
```
- No external scripts, no inline handlers, no CDNs.
- Amazon links via rel="nofollow noopener" redirects (existing go.sergiiob.dev pattern).

### 4. Worker isolation
- Runs in Cloudflare V8 isolate. No filesystem, no shell, no subprocess.
- Can ONLY talk to KV and return HTTP. Entire attack surface.
- If compromised: attacker gets KV read/write on benchmark data. No server access, no weights, no PII.

### 5. Secrets management
- All keys/tokens as Cloudflare Worker secrets (`wrangler secret put`). Never in code.
- Admin key rotated quarterly.
- API keys stored SHA-256 hashed in KV.

### 6. No PII collection
- Collects: benchmark data + optional email. Nothing else.
- No cookies, no analytics tracking, no third-party scripts.

### Threat model
| Threat | Mitigation |
|--------|-----------|
| Spam submissions | API key + rate limit (10/day) + manual review |
| XSS via fields | Strict validation (no HTML) + CSP headers |
| SQL injection | No SQL (KV only) |
| API key theft | Secrets in Worker, hashed in KV, HTTPS-only |
| DDoS | Cloudflare edge absorbs |
| Data exfiltration | Worker can only access KV (benchmark data) |
| Malicious model files | JSON metadata only, NEVER file uploads |

## LOCALMAXXING INTEGRATION

intelmaxxing.com is NOT a competitor to localmaxxing.com. It's a companion:
- localmaxxing.com = global leaderboard (all hardware, all models)
- intelmaxxing.com = Intel Arc-specific recipes, guides, community benchmarks

### Cross-linking
- Every benchmark links to corresponding localmaxxing.com run
- "Submit to LocalMaxxing" button uses lmx CLI
- API optionally proxy-validates against localmaxxing schema

### Name consideration
Before committing to "intelmaxxing":
1. DM @LottoLottoLotto to clarify homage vs competition
2. Fallbacks: arc-pro-b70.guide, intel-gpu-inference.dev, b70.sergiiob.dev

## AMAZON AFFILIATE STRATEGY

### The money page: /buyers-guide/
Every Amazon link lives here. Each post links to it.

### Rig BOM
| Component | Model | Link |
|-----------|-------|------|
| GPU | Arc Pro B70 32GB | go.sergiiob.dev/arc-pro |
| PSU | 750W+ Gold | — |
| CPU | Ryzen 7 5700X3D | — |
| RAM | 32GB DDR4 | — |
| Motherboard | B450/B550 | — |
| NVMe | 500GB+ | — |
| Bridge | Radxa ROCK 5B+ | — |

Redirect pattern: go.sergiiob.dev/shortname to Amazon Associates.

## BUILD ORDER (when traffic triggers fire)

1. Register domain in Cloudflare (~$12/yr, prevents squatting) — DO NOW
2. 301 redirect to sergiiob.dev/b70/ — IMMEDIATE
3. When triggers fire:
   a. Create Astro project (SergiioB/intelmaxxing)
   b. Port /b70/ content from portfolio
   c. Deploy to Cloudflare Pages
   d. Create Worker for /api/v1/benchmarks
   e. Wire KV for submission storage
   f. GitHub Action for auto-rebuild on JSON update
4. Wire affiliate callout box as Astro component
5. Launch thread on X + Reddit

## COSTS

| Item | Cost |
|------|------|
| Domain | ~$12/year (Cloudflare at-cost) |
| Cloudflare Pages | $0 (free tier) |
| Workers | $0 (100K req/day free) |
| KV | $0 (100K reads/day free) |
| GitHub repo | $0 (public) |
| Total | ~$12/year |

One Amazon affiliate B70 sale (~$6-12) covers the domain for a year.
