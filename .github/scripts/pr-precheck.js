// One-shot PR comment. Never mentions APPLY_SERGIO.
// Never closes anything. No pinned image SHAs.
"use strict";

const lib = require("./triage-lib.js");

module.exports = async function prPrecheck({ github, context, core }) {
  const owner = context.repo.owner;
  const repo = context.repo.repo;
  const num = context.payload.pull_request && context.payload.pull_request.number;
  if (!num) return;
  if (context.payload.action !== "opened" && context.payload.action !== "reopened") {
    core.info(`skip comment on ${context.payload.action}`);
    return;
  }

  const files = await github.paginate(github.rest.pulls.listFiles, {
    owner, repo, pull_number: num, per_page: 100,
  });
  const paths = files.map((f) => f.filename);
  const patchPy = paths.filter((p) => p.startsWith("patches/") && p.endsWith(".py"));
  const hasVerifier = paths.some((p) => p.includes("verify-") && p.endsWith(".sh"));
  const policy = lib.loadPolicy(lib.policyPath);
  const classified = lib.classifyPull(
    {
      body: context.payload.pull_request.body || "",
      patchFiles: patchPy,
      hasVerifier,
    },
    policy
  );

  const comments = await github.paginate(github.rest.issues.listComments, {
    owner, repo, issue_number: num,
  });
  if (comments.some((c) => (c.body || "").includes("cookbook-triage:pr-precheck"))) {
    return;
  }

  const lines = [];
  if (classified.patchFiles.length) {
    lines.push(
      `Patches in this PR: ${classified.patchFiles.map((p) => "`" + p + "`").join(", ")}.`
    );
    if (classified.hasVerifier) {
      lines.push("Includes a verify script. GPU-free `docker run` without `--device` is enough for text transforms.");
    } else {
      lines.push(
        "No verify script. For text patches, a GPU-free apply + `py_compile` + idempotent re-apply on the pinned image is the bar."
      );
    }
    if (!classified.image) {
      lines.push("PR body has no sha256 digest (12+ hex). Name the image you ran.");
    }
    if (!classified.ran) {
      lines.push("PR body does not show a launch/verify command. Paste the command and the last line of output.");
    }
  } else {
    lines.push("No `patches/*.py` in this PR. Docs/CI-only: say which page or workflow you changed.");
  }
  lines.push(
    "Do not claim a live corruption rate unless you measured one. This lab will not treat GPU-free apply as a soak."
  );

  await github.rest.issues.createComment({
    owner, repo, issue_number: num,
    body: `<!-- cookbook-triage:pr-precheck -->\n${lines.join("\n\n")}\n`,
  });
};
