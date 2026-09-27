"""Tests for scripts/render-engine-charts.py — renders committed SVGs from
data/chart-inputs.v1.json; --check verifies they match."""

import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render-engine-charts.py"
INPUT = ROOT / "data" / "chart-inputs.v1.json"

spec = importlib.util.spec_from_file_location("render_engine_charts", SCRIPT)
assert spec is not None and spec.loader is not None
rc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rc
spec.loader.exec_module(rc)


def base_chart() -> dict[str, Any]:
    return {
        "id": "t",
        "output": "docs/assets/t.svg",
        "engine": "vLLM XPU",
        "engine_color": "vllm",
        "eyebrow": "e",
        "title": "t",
        "subtitle": "s",
        "y_label": "y",
        "x_label": "x",
        "x_axis": {"min": 512, "max": 8192},
        "x_ticks": [{"value": 512, "label": "512"}, {"value": 8192, "label": "8K"}],
        "series": [
            {
                "label": "s1",
                "color": "vllm",
                "segments": [
                    {"style": "solid", "protocol": "p", "points": {"512": 20.0, "8192": 18.0}}
                ],
            }
        ],
    }


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT)] + args, capture_output=True, text=True)


class LoadAndValidateTests(unittest.TestCase):
    def test_load_inputs_reads_committed_file(self) -> None:
        self.assertGreaterEqual(len(rc.load_inputs(INPUT)), 5)

    def test_load_inputs_rejects_missing_charts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bad = pathlib.Path(td) / "bad.json"
            bad.write_text(json.dumps({"schema_version": "1.0"}))
            with self.assertRaises(SystemExit):
                rc.load_inputs(bad)

    def test_validate_accepts_wellformed(self) -> None:
        rc.validate_chart(base_chart(), 0)

    def test_validate_requires_fields(self) -> None:
        for field in (
            "id",
            "output",
            "engine",
            "engine_color",
            "eyebrow",
            "title",
            "subtitle",
            "y_label",
            "x_label",
        ):
            with self.subTest(field=field):
                c = base_chart()
                del c[field]
                with self.assertRaises(SystemExit):
                    rc.validate_chart(c, 0)

    def test_validate_rejects_bad_axis(self) -> None:
        c = base_chart()
        c["x_axis"] = {"min": 8192, "max": 512}
        with self.assertRaises(SystemExit):
            rc.validate_chart(c, 0)

    def test_validate_rejects_unknown_engine_color(self) -> None:
        c = base_chart()
        c["engine_color"] = "not-an-engine"
        with self.assertRaises(SystemExit):
            rc.validate_chart(c, 0)

    def test_validate_rejects_series_color_mismatch(self) -> None:
        c = base_chart()
        c["series"][0]["color"] = "ovms"
        with self.assertRaises(SystemExit):
            rc.validate_chart(c, 0)

    def test_validate_rejects_bad_segment_style(self) -> None:
        c = base_chart()
        c["series"][0]["segments"][0]["style"] = "wavy"
        with self.assertRaises(SystemExit):
            rc.validate_chart(c, 0)

    def test_validate_rejects_point_outside_axis(self) -> None:
        c = base_chart()
        c["series"][0]["segments"][0]["points"]["99999999"] = 12.3
        with self.assertRaises(SystemExit):
            rc.validate_chart(c, 0)


class RenderTests(unittest.TestCase):
    def test_render_chart_emits_valid_svg(self) -> None:
        c = base_chart()
        c["x_ticks"] = [
            {"value": 512, "label": "512"},
            {"value": 2048, "label": "2K"},
            {"value": 8192, "label": "8K"},
        ]
        svg = rc.render_chart(c)
        self.assertIn("<svg", svg[:200])
        self.assertIn("20.0", svg)

    def test_sparse_label_set_endpoints(self) -> None:
        seg = {"points": {"512": 1.0, "1024": 2.0, "4096": 3.0, "8192": 4.0, "16384": 5.0}}
        pts = rc.sorted_points(seg)
        self.assertEqual(rc.sparse_label_set(pts, 512, 16384, "ends"), {512, 16384})

    def test_sorted_points_orders_keys(self) -> None:
        seg = {"points": {"8192": 1.0, "512": 2.0, "4096": 3.0}}
        self.assertEqual(rc.sorted_points(seg), [(512, 2.0), (4096, 3.0), (8192, 1.0)])


class CliTests(unittest.TestCase):
    STAGING = ROOT / ".ci-chart-staging"

    def tearDown(self) -> None:
        shutil.rmtree(self.STAGING, ignore_errors=True)

    def test_check_mode_verifies_committed_svgs(self) -> None:
        completed = run_cli(["--check"])
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("Charts valid and current", completed.stdout)

    def test_render_writes_svgs_to_staging(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data = json.loads(INPUT.read_text())
            for chart in data["charts"]:
                chart["output"] = str(self.STAGING / pathlib.Path(chart["output"]).name)
            staging_input = pathlib.Path(td) / "inputs.json"
            staging_input.write_text(json.dumps(data))
            completed = run_cli(["--input", str(staging_input)])
            self.assertEqual(completed.returncode, 0, completed.stderr)
            rendered = list(self.STAGING.glob("*.svg"))
            self.assertEqual(len(rendered), len(data["charts"]))
            for svg in rendered:
                self.assertIn("PROVISIONAL", svg.read_text())

    def test_check_fails_on_tampered_svg(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data = json.loads(INPUT.read_text())
            first = data["charts"][0]
            data["charts"] = [first]
            out = self.STAGING / pathlib.Path(first["output"]).name
            first["output"] = str(out)
            staging_input = pathlib.Path(td) / "inputs.json"
            staging_input.write_text(json.dumps(data))
            self.assertEqual(run_cli(["--input", str(staging_input)]).returncode, 0)
            out.write_text(out.read_text() + "<!-- tampered -->")
            completed = run_cli(["--input", str(staging_input), "--check"])
            self.assertNotEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
