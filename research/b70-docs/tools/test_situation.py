#!/usr/bin/env python3
"""Behavior tests for tools/situation.py. Run: python3 tools/test_situation.py"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SITUATION = Path(__file__).resolve().parent / "situation.py"


def run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SITUATION), *args], cwd=cwd, text=True, capture_output=True)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class RepositoryCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "B70-DOCS"
        for path in (
            "AGENTS.md", "DOCUMENT-MAP.md", "research/README.md",
            "docs/operations/current-host-state.md", "docs/operations/storage-and-vram-safety.md",
            "docs/methods/benchmark-protocol.md", "docs/methods/architecture-diagnostic-playbook.md",
            "docs/governance/public-benchmark-publication.md", "docs/governance/agent-operating-system.md",
            ".agentic/skills/benchmark-model/SKILL.md", ".agents/skills/b70-benchmark-claims-standard/SKILL.md",
            ".agents/skills/publish-b70-benchmark-results/SKILL.md", "models/qwen/qwen3.8-27b/README.md",
            "decisions/tp2.md", "research/2026/qwen38.md", "research/2026/other.md",
        ):
            write(self.root / path, "# fixture\n")
        self.entries = [
            {"id": "RL-008", "question": "Dual-B70 TP2 stock Docker", "path_tried": "stock tensor-parallel-size 2",
             "outcome": "blocked", "evidence": "decisions/tp2.md",
             "blocked_pattern": "tensor.parallel.size.2.*stock|TP2.*without.*host.staged", "superseded_by": None},
            {"id": "RL-003", "question": "Qwen3.8 FP8 TP2 W8A16 reroute", "path_tried": "fp8_gemm_w8a16",
             "outcome": "validated", "evidence": "research/2026/qwen38.md", "blocked_pattern": None, "superseded_by": None},
            {"id": "RL-900", "question": "Unrelated Nemotron packing experiment", "outcome": "validated",
             "evidence": "research/2026/other.md", "blocked_pattern": None, "superseded_by": None},
        ]
        self.save_ledger()
        write(self.root / "registry/model-status.json", json.dumps({"models": [
            {"family": "Qwen3.8-27B", "cell": "p512/g128", "dossier": "models/qwen/qwen3.8-27b/README.md"},
        ]}))
        write(self.root / "registry/research-program.json", json.dumps({"questions": []}))
        write(self.root / "registry/campaigns.yaml", json.dumps({"campaigns": []}))

    def save_ledger(self) -> None:
        write(self.root / "registry/research-ledger.json", json.dumps({"entries": self.entries}))

    def compile(self, intent: str, goal: str, *extra: str) -> subprocess.CompletedProcess[str]:
        return run_cli(["compile", "--root", str(self.root), "--no-write", "--intent", intent, "--goal", goal, *extra], self.root)

    def payload(self, intent: str, goal: str, *extra: str) -> dict:
        proc = self.compile(intent, goal, *extra)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout)


class CompileTests(RepositoryCase):
    def test_generic_goal_has_no_global_validated_evidence(self) -> None:
        data = self.payload("bench", "smoke")
        self.assertEqual(data["prior_evidence"], [])
        self.assertIsNone(data["route"])
        self.assertNotIn("skip_cells", data)

    def test_model_match_does_not_admit_unrelated_validated_rows(self) -> None:
        data = self.payload("research", "qwen3.8 serving", "--family", "Qwen3.8-27B")
        self.assertEqual([row["id"] for row in data["prior_evidence"]], ["RL-003"])
        self.assertEqual(data["route"]["family"], "Qwen3.8-27B")
        self.assertIn("models/qwen/qwen3.8-27b/README.md", [item["path"] for item in data["working_set"]])
        self.assertNotIn("skip_cells", data)

    def test_explicit_ledger_id_selects_only_named_evidence(self) -> None:
        data = self.payload("research", "review RL-900")
        self.assertEqual([row["id"] for row in data["prior_evidence"]], ["RL-900"])
        self.assertEqual(data["prior_evidence"][0]["evidence_paths"], ["research/2026/other.md"])

    def test_blocked_bench_halts_but_research_and_diagnose_remain_readable(self) -> None:
        for intent in ("bench", "research", "diagnose"):
            with self.subTest(intent=intent):
                data = self.payload(intent, "stock tensor-parallel-size 2 docker")
                self.assertEqual(data["halt"], intent == "bench")
                self.assertEqual(data["blocked"], ["RL-008"])
                self.assertIn("decisions/tp2.md", [item["path"] for item in data["working_set"]])
                self.assertIn("RL-008", " ".join(data["do_not"]))
                self.assertEqual(data["prior_evidence"][0]["outcome"], "blocked")

    def test_changed_variable_retains_guards_and_does_not_authorize_execution(self) -> None:
        explanation = "replace stock collectives with copy-only flag-slot synchronization"
        data = self.payload("bench", "stock tensor-parallel-size 2 docker", "--changed-variable", explanation)
        self.assertFalse(data["halt"])
        self.assertEqual(data["blocked"], ["RL-008"])
        self.assertEqual(data["changed_variable"], explanation)
        self.assertIn("does not authorize execution", " ".join(data["warnings"]))
        self.assertIn("RL-008", " ".join(data["do_not"]))

    def test_blank_or_placeholder_change_and_removed_force_cannot_bypass(self) -> None:
        for extra in (("--changed-variable", ""), ("--changed-variable", "   "),
                      ("--changed-variable", "..."), ("--changed-variable", "none"), ("--force",)):
            with self.subTest(extra=extra):
                proc = self.compile("bench", "stock tensor-parallel-size 2 docker", *extra)
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                self.assertEqual(proc.stdout, "")

    def test_provisional_and_validated_adoption_guards_apply(self) -> None:
        for outcome in ("provisional", "validated"):
            with self.subTest(outcome=outcome):
                self.entries[1].update(outcome=outcome, blocked_pattern="draft.?vocab.*(adopt|serve)(?!.*canary.*pass)")
                self.save_ledger()
                data = self.payload("bench", "draft-vocab adopt")
                self.assertTrue(data["halt"])
                self.assertEqual(data["blocked"], ["RL-003"])
                cleared = self.payload("bench", "draft-vocab adopt canary pass")
                self.assertFalse(cleared["halt"])

    def test_superseded_guard_remains_history_without_blocking(self) -> None:
        self.entries[0]["superseded_by"] = "RL-003"
        self.save_ledger()
        data = self.payload("bench", "review RL-008 stock tensor-parallel-size 2 docker")
        self.assertFalse(data["halt"])
        self.assertEqual(data["blocked"], [])
        historical = next(row for row in data["prior_evidence"] if row["id"] == "RL-008")
        self.assertEqual(historical["superseded_by"], "RL-003")

    def test_malformed_and_descriptive_guards_preserve_relevant_warnings(self) -> None:
        for pattern, message in (("[unterminated", "invalid blocked_pattern"),
                                 ("Do not adopt this Qwen route before a quality canary passes", "descriptive adoption guard")):
            with self.subTest(pattern=pattern):
                self.entries[1]["blocked_pattern"] = pattern
                self.save_ledger()
                data = self.payload("research", "review RL-003")
                self.assertFalse(data["halt"])
                self.assertEqual(data["blocked"], ["RL-003"])
                self.assertIn(message, " ".join(data["warnings"]))
                self.assertEqual(data["prior_evidence"][0]["blocked_pattern"], pattern)

    def test_grouped_alternation_and_negative_lookahead_are_not_split(self) -> None:
        self.entries[0]["blocked_pattern"] = "TP2.*peer.*(atomics|remote).*|reopen.*TP2.*rerun(?!.*topology)"
        self.save_ledger()
        self.assertEqual(self.payload("research", "own-corpus shortlist review")["blocked"], [])
        self.assertEqual(self.payload("bench", "reopen TP2 rerun topology")["blocked"], [])
        self.assertEqual(self.payload("bench", "reopen TP2 rerun")["blocked"], ["RL-008"])

    def test_legacy_evidence_never_invents_sibling_or_external_paths(self) -> None:
        write(self.root / "results/2026/09/sample/report.json", "{}")
        self.entries[1]["evidence"] = (
            "results/2026/09/sample/report.json + missing.json; research/2026/missing.md "
            "+ research/2026/qwen38.md (§4) + /outside/research/2026/other.md"
        )
        self.save_ledger()
        data = self.payload("research", "RL-003")
        self.assertEqual(data["prior_evidence"][0]["evidence_paths"],
                         ["results/2026/09/sample/report.json", "research/2026/qwen38.md"])
        self.assertNotIn("missing.json", [item["path"] for item in data["working_set"]])

    def test_research_program_and_all_evidence_survive_reading_budget(self) -> None:
        for index in range(15):
            path = f"research/2026/guard-{index}.md"
            write(self.root / path, "# guard\n")
            self.entries.append({"id": f"RL-{index + 100}", "question": "freshprobe mechanism",
                                 "outcome": "provisional", "evidence": path, "blocked_pattern": "freshprobe"})
        self.save_ledger()
        data = self.payload("research", "freshprobe")
        paths = [item["path"] for item in data["working_set"]]
        self.assertLessEqual(len(paths), 12)
        self.assertIn("registry/research-program.json", paths)
        self.assertIn("research/README.md", paths)
        self.assertNotIn("research/PROGRAM.md", paths)
        self.assertEqual({row["id"] for row in data["prior_evidence"]}, {f"RL-{index + 100}" for index in range(15)})
        self.assertFalse(data["halt"])

    def test_site_and_publish_do_not_leak_private_ledger_or_routes(self) -> None:
        site = Path(self.tmp.name) / "site"
        for path in ("AGENTS.md", "DESIGN-AGENT-SYSTEM.md", "data/cookbook-b70/PROVENANCE.md"):
            write(site / path, "# site\n")
        for intent in ("site", "publish"):
            with self.subTest(intent=intent):
                data = self.payload(intent, "RL-003 stock tensor-parallel-size 2 docker", "--site-root", str(site))
                self.assertEqual(data["prior_evidence"], [])
                self.assertEqual(data["blocked"], [])
                self.assertIsNone(data["route"])
                paths = [item["path"] for item in data["working_set"]]
                self.assertFalse(any(path.startswith(("results/", "registry/", "research/")) for path in paths))
                if intent == "site":
                    self.assertEqual({item["tree"] for item in data["working_set"]}, {"xmxlab"})

    def test_publish_only_attaches_explicit_existing_local_evidence(self) -> None:
        path = "results/2026/09/sample"
        write(self.root / path / "report.json", "{}")
        data = self.payload("publish", "report", "--evidence-root", path)
        self.assertIn(path, [item["path"] for item in data["working_set"]])
        for bad in ("../outside", "results/2026/09/missing"):
            proc = self.compile("publish", "report", "--evidence-root", bad)
            self.assertNotEqual(proc.returncode, 0)


class CloseTests(RepositoryCase):
    def register(self, *, static: bool = False) -> str:
        record = "campaigns/2026/example/README.md"
        write(self.root / record, "# Static research evidence\n")
        campaign = {"id": "CAM-EXAMPLE", "record_path": record}
        if not static:
            campaign["evidence_root"] = "results/2026/09/example"
            write(self.root / campaign["evidence_root"] / "report.json", "{}")
        write(self.root / "registry/campaigns.yaml", json.dumps({"campaigns": [campaign]}))
        return record

    def close(self, campaign: str = "CAM-EXAMPLE", outcome: str = "validated", ledger: str = "RL-003", *extra: str) -> subprocess.CompletedProcess[str]:
        return run_cli(["close", "--root", str(self.root), "--campaign", campaign,
                        "--outcome", outcome, "--ledger-id", ledger, *extra], self.root)

    def test_unrelated_row_cannot_close_registered_campaign(self) -> None:
        self.register()
        proc = self.close()
        self.assertEqual(proc.returncode, 2)
        self.assertIn("missing proof", proc.stderr)
        self.assertNotIn("close ok", proc.stdout)

    def test_matching_evidence_and_outcome_close_registered_id_and_dated_root(self) -> None:
        self.register()
        self.entries[1].update(evidence="results/2026/09/example/report.json", campaign_id="CAM-EXAMPLE")
        self.save_ledger()
        for campaign in ("CAM-EXAMPLE", "results/2026/09/example/"):
            with self.subTest(campaign=campaign):
                proc = self.close(campaign)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn("ledger=RL-003", proc.stdout)

    def test_existing_dated_unregistered_result_root_can_close(self) -> None:
        write(self.root / "results/2026/09/unregistered/report.json", "{}")
        self.entries[1]["evidence"] = "results/2026/09/unregistered/report.json"
        self.save_ledger()
        proc = self.close("results/2026/09/unregistered")
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_static_research_needs_named_existing_record_not_raw_gpu_results(self) -> None:
        record = self.register(static=True)
        self.entries[1]["evidence"] = record
        self.save_ledger()
        proc = self.close()
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_wrong_outcome_supersession_or_campaign_identity_refuses_close(self) -> None:
        record = self.register(static=True)
        for update, message in (({"outcome": "failed"}, "does not match"),
                                ({"superseded_by": "RL-900"}, "superseded"),
                                ({"campaign_id": "CAM-OTHER"}, "campaign_id does not match")):
            with self.subTest(update=update):
                self.entries[1] = {"id": "RL-003", "outcome": "validated", "evidence": record, **update}
                self.save_ledger()
                proc = self.close()
                self.assertEqual(proc.returncode, 2)
                self.assertIn(message, proc.stderr)
                self.assertNotIn("close ok", proc.stdout)

    def test_unknown_campaign_cannot_close_even_blocked(self) -> None:
        proc = self.close("hypothetical", "blocked", "RL-008")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("registered campaign ID", proc.stderr)

    def test_nonexistent_record_or_evidence_does_not_prove_close(self) -> None:
        record = self.register(static=True)
        self.entries[1]["evidence"] = record
        self.save_ledger()
        (self.root / record).unlink()
        proc = self.close()
        self.assertEqual(proc.returncode, 2)
        self.assertIn("existing local record_path", proc.stderr)
        self.assertIn("missing proof", proc.stderr)

    def test_neighbor_result_root_does_not_prove_close(self) -> None:
        self.register()
        write(self.root / "results/2026/09/example-other/report.json", "{}")
        self.entries[1]["evidence"] = "results/2026/09/example-other/report.json"
        self.save_ledger()
        proc = self.close()
        self.assertEqual(proc.returncode, 2)
        self.assertIn("missing proof", proc.stderr)

    def test_checker_failure_blocks_close_and_reports_diagnostics(self) -> None:
        record = self.register(static=True)
        self.entries[1]["evidence"] = record
        self.save_ledger()
        write(self.root / "tools/research.py", "import sys\nprint('bad evidence relationship')\nprint('registry invalid', file=sys.stderr)\nraise SystemExit(1)\n")
        proc = self.close()
        self.assertEqual(proc.returncode, 2)
        self.assertIn("tools/research.py check failed", proc.stderr)
        self.assertIn("bad evidence relationship", proc.stderr)
        self.assertIn("registry invalid", proc.stderr)
        self.assertNotIn("close ok", proc.stdout)

    def test_explicit_ledger_id_is_required_even_with_recent_rows(self) -> None:
        proc = run_cli(["close", "--root", str(self.root), "--campaign", "CAM-EXAMPLE", "--outcome", "validated"], self.root)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--ledger-id", proc.stderr)

    def test_changed_route_requires_status_acknowledgment(self) -> None:
        record = self.register(static=True)
        self.entries[1]["evidence"] = record
        self.save_ledger()
        proc = self.close("CAM-EXAMPLE", "validated", "RL-003", "--route-changed")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--status-updated", proc.stderr)
        proc = self.close("CAM-EXAMPLE", "validated", "RL-003", "--route-changed", "--status-updated")
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
