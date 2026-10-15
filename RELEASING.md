# Releasing

What the release bot does, what a maintainer does, and the one command that ships a version.

## How a release is cut

1. Merge commits to `master`. release-please reads the commit **titles**: `feat:` bumps the minor,
   `fix:` bumps the patch, `feat!:` or `BREAKING CHANGE:` bumps the major. Only a few types reach
   the release notes: a title with no type (`Add recipe request issue form`) is not parsed at all,
   and a type the config has no section for (`serving:`, `research:`, `nemotron:`, `evidence:`) is
   parsed and then dropped. Anything a reader needs to see - a changed default, a corrected number -
   belongs in a `feat:` or `fix:` title.
2. Every push to `master` runs `.github/workflows/release-please.yml`. It opens or updates the
   release PR `chore(master): release X.Y.Z`, which bumps `.release-please-manifest.json` and writes
   the new section of `CHANGELOG.md`. `release-please-config.json` owns the version policy; change
   the bump there, never by hand-editing a tag.
3. The same job starts the release PR's own checks and, when the repository allows auto-merge,
   arms auto-merge for it.
4. A maintainer merges the release PR. That push makes release-please tag `vX.Y.Z` and publish the
   GitHub release, with the generated `CHANGELOG.md` section as its body.

## Why the checks on the release PR need a nudge

Since 2026-06-11 GitHub queues `pull_request` runs created by a `GITHUB_TOKEN` update — which is
exactly what the release PR is — in the `action_required` state. `ci` and `pr-precheck` therefore
never start on a release PR until someone clicks **Approve workflows to run** in the merge box.

`release-please.yml` now clears that itself: it waits for the release PR's queued runs, approves
them, and if the token may not approve them it dispatches `ci` and `pr-precheck` on the release
branch instead. `workflow_dispatch` runs are never held for approval, so the release PR gets real
results either way. The leftover `action_required` entries are the queued runs GitHub created; they
do not block the merge.

If that step ends red instead, give the bot a token of its own: store a personal access token or
GitHub App token with `contents: write`, `pull-requests: write`, and `actions: write` as the
`RELEASE_PLEASE_TOKEN` secret. The release PR is then authored by a user, and its runs start without
approval.

## The one command that cuts the release

```bash
REPO=SergiioB/intel-arc-pro-b70-inference-cookbook
gh pr list --repo "$REPO" --search 'chore(master): release in:title'
gh pr checks <number> --repo "$REPO"     # wait for green
gh pr merge <number> --repo "$REPO" --merge
```

Release PRs are merged as merge commits, matching the v1.3.0 - v1.4.1 history.

Hands-off option: enable **Settings -> General -> Allow auto-merge**. The workflow then arms
auto-merge for the release PR, and the merge happens as soon as the checks pass.

## After the merge

```bash
gh release view vX.Y.Z --repo "$REPO"    # tag, body, and assets come from the bot
```

- The tag is created by the bot. Never move it; fix the version policy in `release-please-config.json`
  and let the next release carry the correction.
- Hand-written `## Unreleased` bullets at the top of `CHANGELOG.md` are not part of the release body.
  The generated section is. Title commits to match what you want readers to see.
- Run `python3 scripts/check-repo.py` before pushing to `master`; it is the same gate `ci` runs.
