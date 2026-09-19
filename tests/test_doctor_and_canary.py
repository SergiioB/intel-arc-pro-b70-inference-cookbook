import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent


class DoctorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.doctor_path = ROOT / "scripts" / "doctor.py"
        self.assertTrue(self.doctor_path.exists())

    def test_doctor_json_mode(self) -> None:
        result = subprocess.run(
            [sys.executable, str(self.doctor_path), "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        data = json.loads(result.stdout)
        self.assertIn("os", data)
        self.assertIn("render", data)
        self.assertIn("gpus", data)
        self.assertIn("driver_status", data)
        self.assertIn("tools", data)

    def test_doctor_recover_flag(self) -> None:
        result = subprocess.run(
            [sys.executable, str(self.doctor_path), "--recover"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Recovery", result.stdout)


class CanaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canary_path = ROOT / "scripts" / "canary.py"
        self.assertTrue(self.canary_path.exists())

    def test_canary_mock_mode(self) -> None:
        result = subprocess.run(
            [sys.executable, str(self.canary_path), "--mock", "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        data = json.loads(result.stdout)
        self.assertTrue(data["all_passed"])
        self.assertEqual(len(data["results"]), 4)


if __name__ == "__main__":
    unittest.main()
