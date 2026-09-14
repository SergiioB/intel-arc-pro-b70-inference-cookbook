# Issue and PR checks

What the bots label and ask. How to get to `ready`.

Works for any family, engine, or image. The bot does not whitelist digest prefixes.

## Issues

| You included | Label |
|--------------|--------|
| `sha256:` + 12 or more hex **and** a launch line (`docker run`, `podman run`, `vllm serve`, `llama-server`) | `ready` |
| Missing either | `needs-info` |
| Hang signal (coredump / Timedout job / gpu reset) without seqno+guc_id or dmesg | `needs-info` — paste ~30 lines around the timeout |
| How-to question, no digest, no launch | `question` — use [Discussions](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/discussions) |
| TP>1 **and** a named C1-only `patch_draft_*.py` | `tp2-draft-block` — drop those patches for tensor parallel > 1 |

`--tensor-parallel-size` is not required on a 1× card.

Triage never closes an issue. The stale bot only expires `needs-info` after 14+7 days with no reply. `ready`, `tp2-draft-block`, and `driver-hang` are exempt.

Launch verbs and hang markers live in `.github/triage-policy.json`. Add an engine there; do not put image SHAs in the JS.

## PRs

The PR template asks for: family, image digest, the command you ran and its last line, prefix cache on/off, TP, and what you are not claiming.

On open, a bot comments once: which `patches/*.py` files changed, whether a verify script is in the diff, whether the body has a digest and a run.

For text patches, GPU-free apply on the image you named is enough:

```
IMAGE=<your-image>@sha256:<digest> bash scripts/verify-<change>.sh
```

Use the verifier shipped in the PR; there is no repository-wide patch verifier.
`docker run` without `--device`. Pass means the patches apply, re-apply is a no-op,
and the touched files compile in the targeted trees. That is not a live
corruption measurement.
