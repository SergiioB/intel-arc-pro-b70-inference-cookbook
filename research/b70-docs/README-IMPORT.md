# B70-DOCS import — provenance and reading rules

This directory is a sanitized snapshot of the private **B70-DOCS** lab
repository — the working knowledge base behind this cookbook. Imported
2026-10-14 to give research agents the measured verdicts, ledgers, and
working setup that public recipe pages do not carry.

## Reading order

1. `AGENTS.md` — the operating contract. Hardware invariants, engine
   defaults, benchmark protocol, gating rules.
2. `registry/research-ledger.json` — every probed path: validated,
   blocked, failed. `registry/model-status.json` — current routes and
   open gates. These two JSONs are the primary authorities.
3. `docs/plans/2026-09-20-b70-improvement-roadmap/LEVERS.md` +
   `OUTCOMES.md` — what won, what was measured dead, what is open.
4. `research/2026/` — dated research notes. `campaigns/` — campaign
   narratives. `decisions/` — ADRs. `models/`, `engines/` — per-model and
   per-engine dossiers.
5. `archive/` — historical and superseded material. Useful for "what did
   we already try"; numbers here may be stale.

## What was excluded

- `results/` (raw measurement trees, ~1.2 GB)
- `scripts/` (private benchmark harnesses — the knowledge they encode
  lives in `docs/` and `research/`)
- `social-assets/`, `.agents/`, `.agentic/` (vendored/private skill packs)
- `*.sync-conflict-*`, `__pycache__`, session/tool state (`.zcode`,
  `.mimosa`, `.video_agent`, `.runtime`)
- `research/intel-outreach/` — personal outreach correspondence and an
  application document (PII), not research knowledge
- Files over 1 MiB (cookbook size gate): the
  `deleted-model-artifacts-20260813` conversion manifests/validation
  reports and `evidence/flashnext-kl-20261014/parity20.json` — their
  README/SUMMARY companions remain.

References to `results/…` or `scripts/…` paths appear as inline code —
they cite evidence that lives in the private repo, not dangling links.

## Sanitization applied

- `~` → `~`; `sergio-desktop` → `b70-host`.
- LAN IPs → `<lab-host>`/`<lab-lan-ip>`; Tailscale IPs → `<tailnet-ip>`.
- Disk mounts masked by class: device-named mounts become
  `/mnt/<sata-disk>` / `/data/<other-disk>` (brand and model names
  removed); generic `/data/nvme`, `/data/sata1`, `/data/sata2` paths kept.
- Owner email addresses → `<owner-email>`; link-layer addresses → `<mac>`.
- Other users' home paths (`/home/<name>`) → `~`.
- Private docker socket/bridge names → `<exl3-...>` placeholders.
- Artifact directory names (e.g. `b70-exl3-qwen38-4bpw-20260926`) are
  kept — they identify builds, not hosts.

## Claim-status rules

- Numbers here are **lab-internal** until promoted through the cookbook's
  publication flow. Treat every tok/s figure as provisional context, not a
  public claim.
- `archive/` carries superseded claims by design. Prefer `registry/` and
  `docs/plans/` verdicts over any dated document.
- Lever verdicts (ADOPTED / DEAD / OPEN / gated) are authoritative in
  `registry/research-ledger.json` and `LEVERS.md`. Do not re-propose a
  lever marked DEAD without naming a changed mechanism.
