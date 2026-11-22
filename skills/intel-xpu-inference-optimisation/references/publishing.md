# Qualifying & Publishing a Tuned Intel XPU Recipe

A tuned route is complete only when another user or agent can discover it from `data/system-map.v1.json`, reproduce it without private host context, and trace every number to commit-pinned evidence.

## 1. Reproducible Packaging & Sanitization

1. **Pin every moving part:**
   - Model revision commit SHA + main weight shard SHA-256 + tokenizer SHA-256.
   - Engine image digest or wheel URL + commit SHA + wheel SHA-256.
   - Plugin commit SHA and ordered patch stack (`patches/<family>-<engine>/APPLY.md`).
   - Custom kernel header dependencies (e.g. `intel/sycl-tla` and `vllm-xpu-kernels` commit SHAs) and AOT build script.
2. **Keep custom kernels opt-in:** Gate behind an env var (e.g. `INK2_CUTLASS_MOE=1`, default `0` = exact stock behavior).
3. **Zero private leaks:** Before committing, scan all changed files with `rg` for local paths (`/data/...`, `/home/...`), private PCI BDFs (`0000:0f:00.0`), local IPs, credentials, raw prompt dumps, or large binaries (never commit `.so` binaries or multi-MB logs).

## 2. Clean Repository Slice (No Mixed Information)

Place each fact once:
- `docs/<family>/README.md` — route chooser only (strictly isolated rows by engine/artifact, explicit do-not-mix boundary).
- `docs/<family>/<FAMILY>-<ENGINE>.md` — one self-contained recipe per engine (pinned stack, profiles, full-context tables, embedded chart, quality gate, explicit limits).
- `benchmarks/<family>/` — executable launcher with env overrides for verified operating profiles.
- `patches/<family>-<engine>/` — ordered patch stack, opt-in kernel source + build script, and `APPLY.md`.
- `results/<family>/<engine>-<YYYYMMDD>/` — compact sanitized JSONL/JSON summaries + `README.md`.
- `data/benchmarks.v1.json` — single public numeric authority.

## 3. Two-Commit Evidence Pinning & Catalog Generation

1. **Commit 1 (evidence + patches + docs + charts):** Commit sanitized `results/`, `patches/`, `docs/`, and chart PNGs first so they receive a permanent commit SHA `<EV_SHA>`.
2. **Commit 2 (catalog records + generated views):**
   - Append immutable records to `data/benchmarks.v1.json` pointing to `blob/<EV_SHA>/results/...`. Never overwrite historical records.
   - Use `"kind": "benchmark"` for standardized `n >= 3` speed ladders, and `"kind": "capability"` (without ranking metrics) for separate capacity/fit profiles (e.g. `mbt=1024/512` 256K probes).
   - Regenerate views:
     ```bash
     python3 scripts/render-benchmark-catalog.py
     python3 scripts/build-site-feeds.py
     ```
   - Validate all 12 CI gates:
     ```bash
     python3 scripts/check-repo.py
     ```

## 4. Merge & Release Hygiene (`release-please`)

- Use Conventional Commit titles (`feat(<family>): ...` for minor bumps, `fix(<family>): ...` for patch bumps).
- **Always merge PRs with `--merge` (never `--squash`)** so commit-pinned evidence SHAs in `data/benchmarks.v1.json` remain reachable on `master` (enforced by `test_evidence_commits_and_paths_exist_in_history`).
- Merging to `master` triggers `.github/workflows/release-please.yml`, which opens `chore(master): release X.Y.Z`. Merge that PR with `--merge` once CI is green to cut the tagged GitHub release automatically.

## 5. Result Charts & Social Cards

- Render static PNGs with Pillow at **2x resolution and LANCZOS downsample** for razor-sharp typography.
- **Cookbook dashboard:** show realistic model-card decode vs context, cold prefill + TTFT vs context, and maximum proven context/KV pool by scheduler profile side by side.
- **Social card (1600x900):** one headline claim using realistic card sampling (never greedy), <= 3 comparison rows, 3 fact tiles including a mandatory limit tile, and a full shape footer (`model@rev | sampling | output | power | prompt kind | date`). Always inspect the rendered PNG visually before publishing.
