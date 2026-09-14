// Pure triage predicates. No GitHub API. No pinned image SHAs.
// Policy lives in .github/triage-policy.json so new engines/images
// do not need a JS edit.
"use strict";

const fs = require("fs");
const path = require("path");

const DEFAULT_POLICY = {
  launch_commands: [
    "docker run",
    "podman run",
    "vllm serve",
    "llama-server",
    "python -m sglang",
  ],
  hang_signals: ["coredump", "timedout job", "gpu reset", "guc_id", "driver hang"],
  hang_evidence: ["seqno", "guc_id", "dmesg"],
  c1_only_patches: [
    "patch_draft_lmhead_int4.py",
    "patch_draft_mtp_int4.py",
  ],
};

function loadPolicy(policyPath) {
  if (!policyPath) return { ...DEFAULT_POLICY };
  const raw = fs.readFileSync(policyPath, "utf8");
  const parsed = JSON.parse(raw);
  return {
    launch_commands: parsed.launch_commands || DEFAULT_POLICY.launch_commands,
    hang_signals: parsed.hang_signals || DEFAULT_POLICY.hang_signals,
    hang_evidence: parsed.hang_evidence || DEFAULT_POLICY.hang_evidence,
    c1_only_patches: parsed.c1_only_patches || DEFAULT_POLICY.c1_only_patches,
  };
}

function hasImageDigest(text) {
  const t = String(text || "");
  if (/sha256:[0-9a-f]{12,}/i.test(t)) return true;
  if (/\b[0-9a-f]{64}\b/i.test(t)) return true;
  return false;
}

function hasLaunchCommand(text, policy) {
  const t = String(text || "").toLowerCase();
  return (policy.launch_commands || []).some((cmd) => t.includes(cmd.toLowerCase()));
}

function hasHangSignal(text, policy) {
  const t = String(text || "").toLowerCase();
  return (policy.hang_signals || []).some((s) => t.includes(s.toLowerCase()));
}

function hasHangEvidence(text, policy) {
  const t = String(text || "").toLowerCase();
  const hits = (policy.hang_evidence || []).filter((s) => t.includes(s.toLowerCase()));
  return hits.length >= 2;
}

function isHowToTitle(title) {
  const t = String(title || "").trim();
  if (/^(how |what |can i |why |is there )/i.test(t)) return true;
  if (/\?\s*$/.test(t)) return true;
  return false;
}

function hasTP(text, n) {
  return new RegExp(
    String.raw`\btp\s*[:=]?\s*${n}\b|tensor-parallel-size\s+${n}\b`,
    "i"
  ).test(String(text || ""));
}

function hasCardCount(text, n) {
  return new RegExp(
    String.raw`\b${n}\s*[×x]\s*(?:b70|b60|b50|gpu)s?\b`,
    "i"
  ).test(String(text || ""));
}

function namedC1OnlyPatch(text, policy) {
  const t = String(text || "").toLowerCase();
  return (policy.c1_only_patches || []).some((p) => t.includes(p.toLowerCase()));
}

function isBotCorpus(comment) {
  if (!comment) return true;
  const body = comment.body || "";
  if (body.includes("cookbook-triage")) return true;
  if (comment.user && comment.user.type === "Bot") return true;
  return false;
}

function userCorpus({ title, body, comments }) {
  const parts = [title || "", body || ""];
  for (const c of comments || []) {
    if (isBotCorpus(c)) continue;
    parts.push(c.body || "");
  }
  return parts.join("\n");
}

function classifyIssue({ title, body, comments, labels }, policy) {
  const pol = policy || DEFAULT_POLICY;
  const text = userCorpus({ title, body, comments });
  const lower = text.toLowerCase();
  const have = new Set(labels || []);
  const add = new Set();
  const remove = new Set();

  if (hasCardCount(text, 4) || hasTP(text, 4)) add.add("b70:4");
  else if (hasCardCount(text, 2) || hasTP(text, 2)) add.add("b70:2");
  else if (hasCardCount(text, 1) || hasTP(text, 1)) add.add("b70:1");

  if (hasTP(text, 4)) add.add("tp:4");
  else if (hasTP(text, 2)) add.add("tp:2");
  else if (hasTP(text, 1)) add.add("tp:1");
  if (/\bpp\s*2\b/i.test(text)) add.add("pp:2");

  const hang = hasHangSignal(text, pol);
  if (hang) add.add("driver-hang");

  if (namedC1OnlyPatch(text, pol)) add.add("draft-int4");
  if (
    lower.includes("enable-prefix-caching") &&
    !lower.includes("no-enable-prefix-caching")
  ) {
    add.add("prefix-cache");
  }

  const tpOver1 = have.has("tp:2") || add.has("tp:2") || have.has("tp:4") || add.has("tp:4") || hasTP(text, 2) || hasTP(text, 4);
  if (tpOver1 && namedC1OnlyPatch(text, pol)) add.add("tp2-draft-block");

  const image = hasImageDigest(text);
  const cmd = hasLaunchCommand(text, pol);
  const hangNeedsLog = hang && !hasHangEvidence(text, pol);
  const looksHowTo = isHowToTitle(title);
  const isQuestion = looksHowTo && !image && !cmd && !hang;

  const needsInfo = !isQuestion && (!image || !cmd || hangNeedsLog);

  if (isQuestion) {
    add.add("question");
    remove.add("ready");
  } else if (needsInfo) {
    add.add("needs-info");
    remove.add("ready");
  } else {
    add.add("ready");
    remove.add("needs-info");
  }

  return {
    add: [...add],
    remove: [...remove],
    isQuestion,
    needsInfo,
    hangNeedsLog,
    image,
    cmd,
  };
}

function classifyPull({ body, patchFiles, hasVerifier }, policy) {
  const pol = policy || DEFAULT_POLICY;
  const text = String(body || "");
  return {
    image: hasImageDigest(text),
    ran: hasLaunchCommand(text, pol) || /verify-|gpu-free/i.test(text),
    patchFiles: patchFiles || [],
    hasVerifier: Boolean(hasVerifier),
  };
}

module.exports = {
  DEFAULT_POLICY,
  loadPolicy,
  hasImageDigest,
  hasLaunchCommand,
  hasHangSignal,
  hasHangEvidence,
  isHowToTitle,
  hasTP,
  hasCardCount,
  namedC1OnlyPatch,
  isBotCorpus,
  userCorpus,
  classifyIssue,
  classifyPull,
  policyPath: path.join(__dirname, "..", "triage-policy.json"),
};
