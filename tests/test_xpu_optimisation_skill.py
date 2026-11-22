"""Unit tests for the Intel XPU inference optimisation skill helpers."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / "skills" / "intel-xpu-inference-optimisation"


class IntelXpuOptimisationSkillTests(unittest.TestCase):
    def test_required_skill_files_exist(self) -> None:
        required = [
            "SKILL.md",
            "PROMPT.md",
            "references/ceilings.md",
            "references/levers.md",
            "references/gotchas.md",
            "references/methodology.md",
            "references/publishing.md",
            "scripts/b70_ceiling.py",
            "scripts/b70_preflight.py",
            "examples/inkling2-mini-b70.json",
        ]
        for rel in required:
            self.assertTrue((SKILL_DIR / rel).is_file(), f"missing skill file: {rel}")

    def test_ceiling_calculator_inkling_example_and_swa_monotonicity(self) -> None:
        cmd = [
            sys.executable,
            str(SKILL_DIR / "scripts" / "b70_ceiling.py"),
            "--spec",
            str(SKILL_DIR / "examples" / "inkling2-mini-b70.json"),
            "--json",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(res.stdout)
        self.assertEqual(payload["kv_bytes_per_full_token"], 32768)
        self.assertGreater(payload["decode_ceilings"][0]["decode_ceiling_tok_s"], 90.0)
        # Lower mbt must yield strictly more estimated full-attention context on SWA models
        ctxs = [row["est_max_context_tokens"] for row in payload["kv_mbt_profiles"]]
        self.assertEqual(ctxs, sorted(ctxs, reverse=True))
        self.assertGreater(ctxs[0], ctxs[-1])

    def test_preflight_json_schema_is_valid_and_readonly(self) -> None:
        cmd = [sys.executable, str(SKILL_DIR / "scripts" / "b70_preflight.py"), "--json"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(res.stdout)
        for key in ("intel_gpus", "host_memory", "disk_free_gib", "active_inference_procs"):
            self.assertIn(key, payload)


if __name__ == "__main__":
    unittest.main()
