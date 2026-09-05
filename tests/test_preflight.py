import unittest
import os
import sys
import subprocess
from pathlib import Path

class HostPreflightTests(unittest.TestCase):
    def setUp(self):
        self.script_path = Path(__file__).parent.parent / "scripts" / "setup" / "host-preflight.py"
        self.assertTrue(self.script_path.exists())

    def test_preflight_runs_without_crashing(self):
        result = subprocess.run([sys.executable, str(self.script_path)], capture_output=True, text=True)
        # It may return 1 if the test machine lacks space/swap, or 0 if it passes.
        self.assertIn(result.returncode, [0, 1])
        self.assertIn("--- Memory Preflight ---", result.stdout)
        self.assertIn("Total RAM:", result.stdout)
        self.assertIn("Total Swap:", result.stdout)

if __name__ == '__main__':
    unittest.main()
