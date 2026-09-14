"use strict";

const { describe, it } = require("node:test");
const assert = require("node:assert/strict");
const path = require("path");
const lib = require("../.github/scripts/triage-lib.js");

const policy = lib.loadPolicy(
  path.join(__dirname, "..", ".github", "triage-policy.json")
);

describe("hasImageDigest", () => {
  it("accepts a full sha256 digest", () => {
    assert.equal(
      lib.hasImageDigest(
        "image sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97"
      ),
      true
    );
  });
  it("rejects short hardcoded aliases used alone", () => {
    assert.equal(lib.hasImageDigest("we used f01e24f6 tonight"), false);
    assert.equal(lib.hasImageDigest("2c427ef"), false);
    assert.equal(lib.hasImageDigest("1da0a954"), false);
  });
});

describe("hasLaunchCommand", () => {
  it("accepts docker run and vllm serve", () => {
    assert.equal(lib.hasLaunchCommand("docker run --device /dev/dri", policy), true);
    assert.equal(lib.hasLaunchCommand("vllm serve /model", policy), true);
  });
  it("does not treat --quantization as a launch", () => {
    assert.equal(
      lib.hasLaunchCommand("try --quantization gptq --gpu-memory-utilization 0.9", policy),
      false
    );
  });
});

describe("hang evidence", () => {
  it("does not treat 'xe driver' as a dmesg paste", () => {
    const text = "the xe driver hung my card";
    assert.equal(lib.hasHangSignal(text, policy), false);
    assert.equal(lib.hasHangEvidence(text, policy), false);
  });
  it("needs two evidence tokens for a hang log", () => {
    assert.equal(lib.hasHangEvidence("seqno=4 guc_id=12", policy), true);
    assert.equal(lib.hasHangEvidence("seqno=4 only", policy), false);
  });
});

describe("classifyIssue", () => {
  it("marks ready only with digest + launch", () => {
    const r = lib.classifyIssue(
      {
        title: "decode stall",
        body: "sha256:aaaaaaaaaaaaaaaa docker run --rm vllm",
        comments: [],
        labels: [],
      },
      policy
    );
    assert.equal(r.add.includes("ready"), true);
    assert.equal(r.needsInfo, false);
  });
  it("does not ready on alias + quantization flags", () => {
    const r = lib.classifyIssue(
      {
        title: "slow",
        body: "f01e24f6 --quantization gptq --gpu-memory-utilization 0.9",
        comments: [],
        labels: [],
      },
      policy
    );
    assert.equal(r.add.includes("ready"), false);
    assert.equal(r.add.includes("needs-info"), true);
  });
  it("ignores bot comments that mention TP2 and draft patches", () => {
    const r = lib.classifyIssue(
      {
        title: "How do I serve Qwen?",
        body: "just asking",
        comments: [
          {
            user: { type: "Bot" },
            body: "<!-- cookbook-triage:tp2-draft-block --> TP2 + patch_draft_mtp_int4.py",
          },
        ],
        labels: [],
      },
      policy
    );
    assert.equal(r.add.includes("tp2-draft-block"), false);
    assert.equal(r.isQuestion, true);
  });
  it("does not tag draft-int4 from 'draft' plus GPTQ-Int4", () => {
    const r = lib.classifyIssue(
      {
        title: "notes",
        body: "I will draft logs later. Model is GPTQ-Int4. docker run x sha256:bbbbbbbbbbbb",
        comments: [],
        labels: [],
      },
      policy
    );
    assert.equal(r.add.includes("draft-int4"), false);
  });
  it("tags tp2-draft-block only when a C1-only patch file is named", () => {
    const r = lib.classifyIssue(
      {
        title: "tp2 crash",
        body: "TP2 with patch_draft_mtp_int4.py\nsha256:cccccccccccccccc docker run --rm",
        comments: [],
        labels: [],
      },
      policy
    );
    assert.equal(r.add.includes("tp2-draft-block"), true);
  });
  it("asks for hang evidence even if xe driver is mentioned", () => {
    const r = lib.classifyIssue(
      {
        title: "hang",
        body: "Timedout job on the xe driver\nsha256:dddddddddddd docker run --rm",
        comments: [],
        labels: [],
      },
      policy
    );
    assert.equal(r.hangNeedsLog, true);
    assert.equal(r.add.includes("needs-info"), true);
    assert.equal(r.add.includes("ready"), false);
  });
});
