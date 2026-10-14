"""Consumer-visible regression coverage for immutable experiment evidence."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import experiment


EMIT = """
import json, os
from pathlib import Path
run = Path(os.environ['B70_RUN_DIR'])
plan = json.loads((run / 'plan.json').read_text())
data = json.loads(os.environ.get('EXPERIMENT_TEST_MEASUREMENTS', 'null'))
if data is None:
    data = {
        'metric': plan['comparison']['metric'], 'unit': plan['comparison']['unit'],
        'identities': plan['identities'], 'controlled': plan['controlled'],
        'baseline': [{'pair': str(i), 'value': 100} for i in range(5)],
        'candidate': [{'pair': str(i), 'value': 120} for i in range(5)],
        'checks': {name: {'passed': True, 'evidence': name + '.txt'}
                   for name in ('quality', 'correctness', 'isolation', 'measurement_validity')},
    }
for name in ('quality', 'correctness', 'isolation', 'measurement_validity'):
    (run / (name + '.txt')).write_text('Synthetic test fixture: ' + name + '\\n')
(run / 'measurements.json').write_text(json.dumps(data))
print('raw command output retained')
"""
CLEANUP = "import os; from pathlib import Path; (Path(os.environ['B70_RUN_DIR']) / 'cleanup.txt').write_text('cleanup ran')"


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "registry").mkdir()
        identity = {"host": "synthetic-test-host", "engine_revision": "test-engine", "model_revision": "test-model"}
        self.plan = {
            "question_id": "RQ-1", "hypothesis": "Synthetic paired fixture meets a declared gain",
            "changed_variable": "test setting", "resource": "cpu", "timeout_s": 3,
            "command": [sys.executable, "-c", EMIT], "cleanup_command": [sys.executable, "-c", CLEANUP],
            "identities": {"baseline": dict(identity), "candidate": dict(identity)},
            "comparison": {"metric": "fixture_rate", "unit": "items/s", "direction": "higher", "min_relative_gain": 0.1},
            "controlled": {"workload": {"fixture": "five-pairs"}, "config": {"threads": 1}},
            "stage": "screen", "evidence_rationale": "Synthetic tests only; not B70 measurements",
        }
        self.program = {"questions": [{"id": "RQ-1", "status": "ready", "dependencies": [],
                                      "hypothesis": self.plan["hypothesis"], "changed_variable": self.plan["changed_variable"]}]}
        self.save_program()
        self.dump("registry/research-ledger.json", {"schema_version": 1, "entries": [
            {"id": "RL-001", "question": "historical evidence", "outcome": "blocked", "custom": "preserve exactly"}]})
        self.dump("registry/campaigns.yaml", {"schema_version": 1, "campaigns": [{"id": "CAM-HISTORY", "custom": "preserve exactly"}]})

    def dump(self, path, value):
        (self.root / path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def save_program(self):
        self.dump("registry/research-program.json", self.program)

    def measurements(self):
        return {"metric": self.plan["comparison"]["metric"], "unit": self.plan["comparison"]["unit"],
                "identities": copy.deepcopy(self.plan["identities"]), "controlled": copy.deepcopy(self.plan["controlled"]),
                "baseline": [{"pair": str(i), "value": 100} for i in range(5)],
                "candidate": [{"pair": str(i), "value": 120} for i in range(5)],
                "checks": {name: {"passed": True, "evidence": name + ".txt"} for name in experiment.CHECKS}}

    def run_fixture(self, data=None):
        self.dump("plan.json", self.plan)
        environment = {} if data is None else {"EXPERIMENT_TEST_MEASUREMENTS": json.dumps(data)}
        with patch.dict(os.environ, environment):
            return experiment.run_plan(self.root, self.root / "plan.json")

    def assess_fixture(self, data=None):
        result = self.run_fixture(data)
        assessment = experiment.assess_run(self.root, Path(result["run"]))
        experiment.record_run(self.root, Path(result["run"]))
        return assessment

    def test_screen_records_once_without_adoption_or_rewriting_history(self):
        ledger_before = (self.root / "registry/research-ledger.json").read_text()
        campaigns_before = (self.root / "registry/campaigns.yaml").read_text()
        model_status = self.root / "registry/model-status.json"
        model_status.write_text('historical model status must not change\n')
        result = self.run_fixture()
        run = Path(result["run"])
        assessment = experiment.assess_run(self.root, run)
        self.assertEqual(assessment["decision"], "eligible")
        self.assertEqual(assessment["outcome"], "validated_screen")
        self.assertEqual(assessment["adoption"], "not_adopted")
        self.assertFalse(assessment["adoptable"])
        self.assertEqual(assessment["comparison_class"], "config")
        self.assertEqual((run / "command.stdout").read_text(), "raw command output retained\n")
        first = experiment.record_run(self.root, run)
        ledger_after = (self.root / "registry/research-ledger.json").read_text()
        campaigns_after = (self.root / "registry/campaigns.yaml").read_text()
        second = experiment.record_run(self.root, run)
        self.assertEqual(first["ledger_id"], second["ledger_id"])
        self.assertTrue(second["already_recorded"])
        self.assertEqual(ledger_after, (self.root / "registry/research-ledger.json").read_text())
        self.assertEqual(campaigns_after, (self.root / "registry/campaigns.yaml").read_text())
        self.assertTrue(ledger_after.startswith(ledger_before[:ledger_before.rindex("]")]))
        self.assertTrue(campaigns_after.startswith(campaigns_before[:campaigns_before.rindex("]")]))
        self.assertEqual(model_status.read_text(), 'historical model status must not change\n')

    def test_confirmation_is_provisional_and_engine_change_is_route_comparison(self):
        self.plan["stage"] = "confirm"
        self.plan["identities"]["candidate"]["engine_revision"] = "new-test-engine"
        assessment = self.assess_fixture()
        self.assertEqual(assessment["outcome"], "provisional")
        self.assertEqual(assessment["comparison_class"], "route")
        self.assertFalse(assessment["adoptable"])

    def test_quality_failure_overrides_speed(self):
        data = self.measurements()
        data["checks"]["quality"]["passed"] = False
        assessment = self.assess_fixture(data)
        self.assertEqual(assessment["decision"], "rejected")
        self.assertEqual(assessment["outcome"], "failed")
        self.assertIn("quality", assessment["reason"])

    def test_noise_crossing_gate_is_inconclusive(self):
        data = self.measurements()
        data["candidate"][0]["value"] = 105
        assessment = self.assess_fixture(data)
        self.assertEqual(assessment["decision"], "inconclusive")
        self.assertEqual(assessment["outcome"], "blocked")
        self.assertGreater(assessment["median_relative_gain"], 0.1)

    def test_median_below_gate_rejects(self):
        data = self.measurements()
        for row in data["candidate"]:
            row["value"] = 105
        assessment = self.assess_fixture(data)
        self.assertEqual(assessment["decision"], "rejected")

    def test_exact_threshold_and_lower_is_better(self):
        self.plan["comparison"]["direction"] = "lower"
        self.plan["comparison"]["unit"] = "ms"
        data = self.measurements()
        for row in data["candidate"]:
            row["value"] = 90
        assessment = self.assess_fixture(data)
        self.assertEqual(assessment["decision"], "eligible")
        self.assertEqual(assessment["minimum_relative_gain"], 0.1)

    def test_duplicate_pair_ids_and_mismatched_pairs_are_blocked(self):
        data = self.measurements()
        data["candidate"][1]["pair"] = data["candidate"][0]["pair"]
        self.assertIn("Duplicate pair", self.assess_fixture(data)["reason"])
        data = self.measurements()
        data["candidate"][1]["pair"] = "unpaired"
        assessment = self.assess_fixture(data)
        self.assertEqual(assessment["decision"], "blocked")
        self.assertIn("match exactly", assessment["reason"])

    def test_confirm_requires_five_pairs_and_screen_requires_three(self):
        data = self.measurements()
        for arm in ("baseline", "candidate"):
            data[arm] = data[arm][:2]
        self.assertEqual(self.assess_fixture(data)["decision"], "blocked")
        self.plan["stage"] = "confirm"
        data = self.measurements()
        for arm in ("baseline", "candidate"):
            data[arm] = data[arm][:4]
        self.assertEqual(self.assess_fixture(data)["decision"], "blocked")

    def test_metrics_and_controlled_coordinates_cannot_be_repaired_after_sealing(self):
        data = self.measurements()
        data["controlled"]["config"]["threads"] = 2
        self.assertEqual(self.assess_fixture(data)["decision"], "blocked")
        data = self.measurements()
        data["unit"] = "different-unit"
        result = self.run_fixture(data)
        run = Path(result["run"])
        self.assertEqual(experiment.assess_run(self.root, run)["decision"], "blocked")
        (run / "measurements.json").write_text(json.dumps(self.measurements()))
        with self.assertRaisesRegex(experiment.ExperimentError, "Sealed artifact changed"):
            experiment.assess_run(self.root, run)

    def test_missing_measurements_are_not_invented(self):
        self.plan["command"] = [sys.executable, "-c", "print('successful exit is not evidence')"]
        result = self.run_fixture()
        run = Path(result["run"])
        assessment = experiment.assess_run(self.root, run)
        self.assertEqual(assessment["decision"], "blocked")
        self.assertNotIn("median_relative_gain", assessment)
        self.assertEqual(experiment.record_run(self.root, run)["outcome"], "blocked")

    def test_path_traversal_and_missing_check_evidence_are_blocked(self):
        outside = self.root / "outside.txt"
        outside.write_text("outside evidence")
        data = self.measurements()
        data["checks"]["quality"]["evidence"] = str(outside)
        self.assertEqual(self.assess_fixture(data)["decision"], "blocked")
        data["checks"]["quality"]["evidence"] = "../outside.txt"
        self.assertEqual(self.assess_fixture(data)["decision"], "blocked")

    def test_symlink_evidence_cannot_escape_run(self):
        outside = self.root / "outside.txt"
        outside.write_text("outside evidence")
        self.plan["command"][2] += "\n(run / 'quality.txt').unlink(); (run / 'quality.txt').symlink_to(" + repr(str(outside)) + ")\n"
        assessment = self.assess_fixture()
        self.assertNotEqual(assessment["decision"], "eligible")

    def test_empty_check_file_and_nonboolean_check_are_blocked(self):
        self.plan["command"][2] += "\n(run / 'quality.txt').write_text('')\n"
        self.assertEqual(self.assess_fixture()["decision"], "blocked")
        self.plan["command"][2] = EMIT
        data = self.measurements()
        data["checks"]["quality"]["passed"] = "true"
        self.assertEqual(self.assess_fixture(data)["decision"], "blocked")

    def test_nonpositive_and_nonfinite_measurements_are_blocked(self):
        data = self.measurements()
        data["baseline"][0]["value"] = 0
        self.assertEqual(self.assess_fixture(data)["decision"], "blocked")
        data["baseline"][0]["value"] = float("nan")
        self.assertEqual(self.assess_fixture(data)["decision"], "blocked")

    def test_timeout_kills_owned_descendants_and_runs_cleanup(self):
        self.plan["timeout_s"] = 0.3
        child = "import os,time; from pathlib import Path; time.sleep(0.8); (Path(os.environ['B70_RUN_DIR']) / 'escaped.txt').write_text('alive')"
        self.plan["command"] = [sys.executable, "-c",
            "import subprocess,sys,time; print('before timeout', flush=True); "
            + "subprocess.Popen([sys.executable, '-c', " + repr(child) + "]); time.sleep(20)"]
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
        self.addCleanup(lambda: unrelated.poll() is None and unrelated.kill())
        try:
            result = self.run_fixture()
            run = Path(result["run"])
            self.assertFalse(result["execution_passed"])
            self.assertTrue((run / "cleanup.txt").is_file())
            complete = experiment.read_json(run / "complete.json")
            self.assertTrue(complete["phases"]["command"]["timed_out"])
            self.assertEqual(experiment.assess_run(self.root, run)["decision"], "rejected")
            self.assertIn("before timeout", (run / "command.stdout").read_text())
            time.sleep(0.8)
            self.assertFalse((run / "escaped.txt").exists())
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=3)

    def test_nonzero_and_cleanup_failure_never_qualify(self):
        self.plan["command"][2] += "\nraise SystemExit(7)\n"
        first = self.run_fixture()
        first_run = Path(first["run"])
        self.assertTrue((first_run / "cleanup.txt").exists())
        self.assertEqual(experiment.assess_run(self.root, first_run)["decision"], "rejected")
        experiment.record_run(self.root, first_run)
        self.plan["command"][2] = EMIT
        self.plan["cleanup_command"] = [sys.executable, "-c", "raise SystemExit(9)"]
        self.assertEqual(self.assess_fixture()["decision"], "rejected")

    def test_cleanup_timeout_is_bounded_and_saved(self):
        self.plan["timeout_s"] = 0.3
        self.plan["cleanup_command"] = [sys.executable, "-c", "import time; print('cleanup start', flush=True); time.sleep(20)"]
        result = self.run_fixture()
        run = Path(result["run"])
        complete = experiment.read_json(run / "complete.json")
        self.assertTrue(complete["phases"]["cleanup"]["timed_out"])
        self.assertEqual(experiment.assess_run(self.root, run)["decision"], "rejected")
        self.assertIn("cleanup start", (run / "cleanup.stdout").read_text())

    def test_launch_failure_retains_error_and_runs_cleanup(self):
        self.plan["command"] = [str(self.root / "not-an-executable")]
        result = self.run_fixture()
        run = Path(result["run"])
        self.assertTrue((run / "cleanup.txt").exists())
        self.assertIn("not-an-executable", (run / "command.stderr").read_text())
        self.assertEqual(experiment.assess_run(self.root, run)["outcome"], "failed")

    def test_gpu_requires_preflight_and_cleanup_and_failed_preflight_skips_command(self):
        self.plan["resource"] = "gpu"
        with self.assertRaisesRegex(experiment.ExperimentError, "preflight_command"):
            self.run_fixture()
        self.plan["preflight_command"] = [sys.executable, "-c", "raise SystemExit(4)"]
        result = self.run_fixture()
        run = Path(result["run"])
        self.assertFalse((run / "command.stdout").exists())
        self.assertTrue((run / "cleanup.txt").exists())
        self.assertEqual(experiment.assess_run(self.root, run)["decision"], "rejected")

    def test_two_attempt_ceiling_includes_failed_runs_and_changed_commands(self):
        self.plan["command"] = [sys.executable, "-c", "raise SystemExit(3)"]
        first = self.run_fixture()
        experiment.record_run(self.root, Path(first["run"]))
        self.plan["command"] = [sys.executable, "-c", "raise SystemExit(4)"]
        second = self.run_fixture()
        experiment.record_run(self.root, Path(second["run"]))
        self.plan["command"] = [sys.executable, "-c", EMIT]
        with self.assertRaisesRegex(experiment.ExperimentError, "Attempt limit"):
            self.run_fixture()
        self.assertEqual(experiment.attempt_count(self.root, self.plan), 2)

    def test_attempt_ceiling_comes_from_declared_portfolio_limit(self):
        self.program["portfolio"] = {"max_attempts_per_mechanism": 1}
        self.save_program()
        result = self.run_fixture()
        experiment.record_run(self.root, Path(result["run"]))
        with self.assertRaisesRegex(experiment.ExperimentError, "Attempt limit"):
            self.run_fixture()
        self.assertEqual(experiment.attempt_count(self.root, self.plan), 1)

    def test_each_operator_authorization_adds_exactly_one_attempt(self):
        self.program["portfolio"] = {"max_attempts_per_mechanism": 1}
        self.program["questions"][0]["execution_budget"] = {"operator_extra_attempts": [
            {"authorized_on": "2026-10-11", "operator_words": "continue", "reason": "harness defect"}]}
        self.save_program()
        for _ in range(2):
            experiment.record_run(self.root, Path(self.run_fixture()["run"]))
        with self.assertRaisesRegex(experiment.ExperimentError, "Attempt limit"):
            self.run_fixture()
        self.assertEqual(experiment.attempt_count(self.root, self.plan), 2)

    def test_unregistered_run_blocks_another_attempt(self):
        result = self.run_fixture()
        with self.assertRaisesRegex(experiment.ExperimentError, "Unregistered prior run"):
            self.run_fixture()
        experiment.record_run(self.root, Path(result["run"]))
        self.assertTrue(self.run_fixture()["execution_passed"])

    def test_declared_budget_and_minimum_gain_cannot_be_relaxed_by_plan(self):
        self.program["portfolio"] = {"max_attempts_per_mechanism": 2}
        self.program["questions"][0]["execution_budget"] = {"screen_timeout_s": 1}
        self.save_program()
        with self.assertRaisesRegex(experiment.ExperimentError, "execution budget"):
            self.run_fixture()
        self.plan["timeout_s"] = 1
        self.plan["comparison"]["min_relative_gain"] = 0.01
        with self.assertRaisesRegex(experiment.ExperimentError, "minimum useful gain"):
            self.run_fixture()

    def test_local_lock_prevents_another_writer(self):
        with experiment.writer_lock(self.root):
            with self.assertRaisesRegex(experiment.ExperimentError, "writer is active"):
                self.run_fixture()
        self.assertFalse((self.root / "results").exists())

    def test_question_hold_status_and_failed_dependency_block_execution(self):
        question = self.program["questions"][0]
        question["operator_hold"] = "Desktop offline"
        self.save_program()
        with self.assertRaisesRegex(experiment.ExperimentError, "operator hold"):
            self.run_fixture()
        del question["operator_hold"]
        question["status"] = "gated"
        self.save_program()
        with self.assertRaisesRegex(experiment.ExperimentError, "status ready"):
            self.run_fixture()
        question["status"] = "ready"
        question["dependencies"] = ["RQ-2"]
        dependency = {"id": "RQ-2", "status": "closed", "resolution": "failed"}
        self.program["questions"].append(dependency)
        self.save_program()
        with self.assertRaisesRegex(experiment.ExperimentError, "resolution passed"):
            self.run_fixture()
        dependency["resolution"] = "passed"
        self.save_program()
        self.assertTrue(self.run_fixture()["execution_passed"])

    def test_plan_cannot_silently_change_program_hypothesis(self):
        self.plan["hypothesis"] = "Different hypothesis"
        with self.assertRaisesRegex(experiment.ExperimentError, "must match the research program"):
            self.run_fixture()
        self.assertFalse((self.root / "results").exists())

    def test_partial_registration_recovers_without_duplicate_ledger_entry(self):
        result = self.run_fixture()
        run = Path(result["run"])
        original = experiment.append_registry

        def fail_campaign(path, key, row):
            if key == "campaigns":
                raise OSError("Synthetic interrupted registry write")
            original(path, key, row)

        with patch.object(experiment, "append_registry", side_effect=fail_campaign):
            with self.assertRaises(OSError):
                experiment.record_run(self.root, run)
        registration = experiment.record_run(self.root, run)
        ledger = experiment.read_json(self.root / "registry/research-ledger.json")["entries"]
        rows = [row for row in ledger if row.get("run_root") == result["run_root"]]
        self.assertEqual([row["id"] for row in rows], [registration["ledger_id"]])
        self.assertTrue(experiment.record_run(self.root, run)["already_recorded"])

    def test_sigterm_preserves_unsealed_evidence_and_cleanup(self):
        self.plan["command"] = [sys.executable, "-c", "import time; print('running', flush=True); time.sleep(20)"]
        self.dump("plan.json", self.plan)
        process = subprocess.Popen([sys.executable, str(Path(experiment.__file__)), "run", "--root", str(self.root),
                                    "--plan", str(self.root / "plan.json")], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 5
            outputs = []
            while time.monotonic() < deadline:
                outputs = list((self.root / "results").glob("*/*/*/command.stdout"))
                if outputs and "running" in outputs[0].read_text():
                    break
                if process.poll() is not None:
                    self.fail("Runner exited before interruption")
                time.sleep(0.02)
            else:
                self.fail("Command did not start")
            process.send_signal(signal.SIGTERM)
            _, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 2)
            self.assertIn(b"interrupted", stderr)
            run = outputs[0].parent
            self.assertTrue((run / "cleanup.txt").exists())
            self.assertFalse((run / "complete.json").exists())
            with self.assertRaisesRegex(experiment.ExperimentError, "complete.json"):
                experiment.record_run(self.root, run)
            self.assertEqual(experiment.attempt_count(self.root, self.plan), 1)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()


if __name__ == "__main__":
    unittest.main()
