// Issue triage only. Never run on pull requests. Never close issues.
// Predicates live in triage-lib.js. Do not add pinned image SHAs here.
"use strict";

const lib = require("./triage-lib.js");

module.exports = async function triage({ github, context, core, issueNumber }) {
  const owner = context.repo.owner;
  const repo = context.repo.repo;
  const num = Number(issueNumber);
  if (!num) {
    core.warning("no issue number");
    return;
  }

  const { data: issue } = await github.rest.issues.get({
    owner, repo, issue_number: num,
  });
  if (issue.pull_request) {
    core.info(`#${num} is a PR — skip issue triage`);
    return;
  }

  const comments = await github.paginate(github.rest.issues.listComments, {
    owner, repo, issue_number: num,
  });
  const policy = lib.loadPolicy(lib.policyPath);
  const classified = lib.classifyIssue(
    {
      title: issue.title,
      body: issue.body,
      comments,
      labels: (issue.labels || []).map((l) => l.name),
    },
    policy
  );

  const labels = new Set((issue.labels || []).map((l) => l.name));
  // Maintainer override: ready is sticky. The bot may add ready when the
  // repro is complete, but it never removes a human-set ready and never
  // re-adds needs-info on top of it. Humans own removal. This stops the
  // bot reverting Sergio's label on every follow-up comment (seen on #10:
  // ready set by hand, then issue_comment triage flipped it back because
  // the digest + launch line live in comments/attachments, not the body).
  if (labels.has("ready")) {
    classified.add = classified.add.filter((l) => l !== "needs-info");
    classified.remove = classified.remove.filter((l) => l !== "ready");
  }
  const toAdd = classified.add.filter((l) => !labels.has(l));
  const toRemove = classified.remove.filter((l) => labels.has(l));
  for (const l of toAdd) {
    try {
      await github.rest.issues.addLabels({
        owner, repo, issue_number: num, labels: [l],
      });
    } catch (e) {
      core.info(`add ${l}: ${e.message}`);
    }
  }
  for (const l of toRemove) {
    try {
      await github.rest.issues.removeLabel({
        owner, repo, issue_number: num, name: l,
      });
    } catch (e) {
      /* already gone */
    }
  }

  const botComments = comments.filter((c) => lib.isBotCorpus(c));
  const hasBot = (needle) =>
    botComments.some((c) => (c.body || "").includes(`cookbook-triage:${needle}`));

  async function once(marker, body) {
    if (hasBot(marker)) return;
    await github.rest.issues.createComment({
      owner, repo, issue_number: num,
      body: `<!-- cookbook-triage:${marker} -->\n${body}\n`,
    });
  }

  if (classified.add.includes("tp2-draft-block")) {
    await once(
      "tp2-draft-block",
      "TP>1 plus a C1-only draft patch is blocked. Drop the named `patch_draft_*` files for tensor parallel > 1. Leave this issue open if you have a new image, a runtime skip, or numbers."
    );
  }
  if (classified.hangNeedsLog) {
    await once(
      "needs-dmesg",
      "Hang: paste ~30 lines of kernel log around the timeout (seqno + guc_id or equivalent) and the full launch command."
    );
  }
  if (classified.needsInfo && !classified.add.includes("driver-hang")) {
    await once(
      "needs-info",
      "Need a sha256 image digest (12+ hex) and the full launch command (`docker run` / `vllm serve` / `llama-server`). Then this moves to `ready`."
    );
  }
  if (classified.isQuestion) {
    await once(
      "question",
      `Questions go in Discussions: https://github.com/${owner}/${repo}/discussions\n\nIssues stay for a repro (image digest + command + what broke).`
    );
  }

  core.summary.addHeading("triage");
  core.summary.addRaw(
    `#${num} add=[${classified.add.join(",")}] remove=[${classified.remove.join(",")}] question=${classified.isQuestion} needsInfo=${classified.needsInfo}`
  );
  await core.summary.write();
};
