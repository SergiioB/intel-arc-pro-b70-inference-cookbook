#!/usr/bin/env python3
"""Behavioral gates for autonomous question selection."""
import json
from pathlib import Path
import tempfile
import unittest

from research_queue import eligible, next_question, program_errors


class QueueTests(unittest.TestCase):
    def question(self, key="RQ-1", **changes):
        return {"id": key, "priority": 1, "status": "ready", "dependencies": [],
                "focus": "task-latency", "economic_gate": "Measured prefill budget",
                "execution_budget": {"max_attempts": 2}, "hypothesis": "Cache saves task time",
                "changed_variable": "Cache reuse", "next_action": "Price prefill",
                "evidence_ids": [], "evidence_paths": [], **changes}

    def test_operator_hold_cannot_be_selected_by_explicit_id(self):
        q = self.question(operator_hold="Stopped by operator")
        self.assertEqual(eligible(q, {q["id"]: q}), (False, "Stopped by operator"))
        self.assertIn("operator-held", " ".join(program_errors({"portfolio": {"max_ready": 3}, "questions": [q]})))

    def test_rejected_prerequisite_does_not_unlock_work(self):
        q = self.question(dependencies=["RQ-2"])
        dep = self.question("RQ-2", status="closed", resolution="rejected")
        self.assertFalse(eligible(q, {q["id"]: q, dep["id"]: dep})[0])
        dep["resolution"] = "passed"
        self.assertTrue(eligible(q, {q["id"]: q, dep["id"]: dep})[0])

    def test_dependency_cycles_are_rejected(self):
        a = self.question(dependencies=["RQ-2"])
        b = self.question("RQ-2", dependencies=["RQ-1"])
        self.assertIn("dependency cycle", " ".join(program_errors({"questions": [a, b]})))

    def test_ready_portfolio_is_bounded(self):
        p = {"portfolio": {"max_ready": 3}, "questions": [self.question(f"RQ-{i}") for i in range(4)]}
        self.assertIn("ready queue exceeds", " ".join(program_errors(p)))

    def test_unregistered_run_prevents_fresh_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "registry").mkdir()
            (root / "registry/research-program.json").write_text(json.dumps({"questions": [self.question()]}))
            (root / "registry/research-ledger.json").write_text('{"entries": []}')
            run = root / "results/2026/10/RQ-1-a"
            run.mkdir(parents=True)
            (run / "plan.json").write_text(json.dumps(self.question() | {"question_id": "RQ-1"}))
            data = next_question(root)
            self.assertTrue(data["halt"])
            self.assertEqual(data["run"], "results/2026/10/RQ-1-a")
            self.assertFalse(data["execution_authorized"])

    def test_missing_evidence_fails_instead_of_silently_dropping_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "registry").mkdir()
            (root / "registry/research-program.json").write_text(json.dumps({"questions": [self.question()]}))
            (root / "registry/research-ledger.json").write_text('{"entries": []}')
            with self.assertRaisesRegex(ValueError, "missing working-set evidence"):
                next_question(root)


if __name__ == "__main__":
    unittest.main()
