# B70 Catalog, systemd, and Pi Telegram Bridge lock

Date: 2026-07-22
Status: production verified

## Authoritative sources

- Model/runtime source: `~/inference/profiles.json` on desktop-b70.
- Generated export: `~/inference/exports/workstation-catalog.json`.
- Bridge copy: `~/.cache/workstation-inference/workstation-catalog.json`.
- Bridge loader: generated workstation catalog only; obsolete hardcoded `wks-*` models removed.

The source and Bridge catalog SHA-256 are identical:

`f177bac0649a4f2a04f1e715144a8b2afe89af601e9a2a058227e7be76a03f69`

## Runtime lock

- Engine: llama.cpp SYCL / Level Zero.
- Base: b10053 `0dc74e332edee2616e4d8d9ab3b68dfc340fc14a`.
- Fix: upstream PR #25690 `0bd0ec60998d0f71ec45471b633bf2403ac81956`.
- MoE reorder: enabled.
- `GGML_SYCL_DISABLE_MOE_REORDER`: absent from the production process.
- `run-profile`, manual launcher, concurrent launcher, and benchmark helper consume the runtime binary declared in `profiles.json`.
- Original b10053+guard and b9853 remain explicit fallbacks, not active paths.

## Model catalog

Six installed profiles are exported:

1. `lc-qwen35-q5-256k` — text, reasoning off, 262144 context, 70.0 t/s validated metadata.
2. `lc-qwen35-q4-256k` — text, reasoning off, 262144 context, 72.4 t/s validated metadata.
3. `lc-ornith35-q5-256k` — text+image, reasoning off, 262144 context.
4. `lc-qwen27-mtp-q5-200k-165w` — text, reasoning enabled, 204800 context, 24.5 t/s metadata.
5. `lc-qwen27-mtp-q5-128k-vision-180w` — text+image, reasoning enabled, 131072 context, 29.0 t/s metadata.
6. `lc-qwen27-mtp-q5-200k-200w` — text, reasoning enabled, 204800 context.

The generated catalog does not contain an API key. The Bridge startup preflight rejects empty, malformed, duplicate, incomplete, default-missing, or credential-bearing catalogs.

## systemd lock

### desktop-b70

- Active supervisor: Sergio's `llama-profile.service` user unit.
- State: enabled, active, `Restart=on-failure`.
- User lingering: enabled.
- Legacy system `llama-hermes.service`: disabled, inactive, and `RefuseManualStart=yes`.

### Radxa Pi Telegram Bridge

- Active supervisor: system `pi-telegram-bridge.service`.
- State: enabled and active.
- Obsolete user unit: disabled and inactive.
- Pollers: exactly one.
- Restart count after deployment: zero.
- npm deployment scripts and xAI token refresh restart the system unit, not the retired user unit.

## End-to-end validation

- Bridge preflight: SDK 0.80.9, six workstation models.
- Built runtime workstation count: six.
- Stale `wks-*` runtime count: zero.
- Q5 Bridge-resolved completion: HTTP 200, `CATALOG_OK`.
- Actual switch backend to Q4: succeeded; completion returned `Q4_CATALOG_OK`.
- Actual switch backend back to Q5: succeeded; completion returned `FINAL_CATALOG_OK`.
- Final active profile: `qwen35-q5-256k`.
- Final B70 health: OK.
- B70/Bridge recent error scan: no Xe reset, GPU fault, OOM, fatal Bridge error, duplicate poller, or authorization conflict.

## Rollback

Timestamped backups were captured before editing under the B70 inference backup directory and Radxa Bridge backup cache. To roll back:

1. Restore the selected backup files.
2. Regenerate and resync the catalog.
3. Rebuild the Bridge.
4. Restart B70 `llama-profile.service` and the system Pi Bridge unit.
5. Re-run preflight, model-list inspection, profile switch, and real completion.

Do not reactivate the retired Pi Bridge user unit or legacy `llama-hermes.service`.
