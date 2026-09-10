import importlib.util
import json
import pathlib
import sys
import unittest
from typing import Any

ROOT = pathlib.Path(__file__).parents[1]
CATALOG = ROOT / "data" / "benchmarks.v1.json"
RENDERER = ROOT / "scripts" / "render-benchmark-catalog.py"

spec = importlib.util.spec_from_file_location("catalog_renderer", RENDERER)
assert spec is not None and spec.loader is not None
renderer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = renderer
spec.loader.exec_module(renderer)


class BenchmarkCatalogTests(unittest.TestCase):
    catalog: dict[str, Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = json.loads(CATALOG.read_text())

    def test_catalog_is_valid_and_contains_new_capabilities(self) -> None:
        errors = renderer.validate_catalog(self.catalog)
        self.assertEqual(errors, [], "\n".join(errors))
        ids = {record["id"] for record in self.catalog["records"]}
        self.assertIn("dual-b70-tp2-serving-v1", ids)
        self.assertIn("qwen38-fp8-w8a16-v1", ids)
        self.assertIn("qwen38-flash-next-fused-fp32-p512-g128-c1", ids)
        self.assertIn("qwen38-flash-next-fused-f16-p9096-g128-c1", ids)
        self.assertIn("qwen38-flash-next-128k-c1-v1", ids)
        self.assertIn("qwen38-flash-next-gpu-group-v1", ids)

    def test_every_numeric_benchmark_has_exact_workload_and_pinned_evidence(self) -> None:
        for record in self.catalog["records"]:
            if record["kind"] != "benchmark":
                continue
            self.assertIn(record["workload"]["concurrency"], [1, 2, 4, 8, 16, 32])
            self.assertGreater(record["workload"]["prompt_tokens"], 0)
            self.assertGreater(record["workload"]["output_tokens"], 0)
            self.assertGreater(record["statistics"]["samples"], 0)
            self.assertRegex(record["evidence"]["commit"], r"^[0-9a-f]{40}$")
            self.assertTrue(record["evidence"]["url"].startswith("https://github.com/SergiioB/"))

    def test_capability_notes_cannot_masquerade_as_complete_benchmarks(self) -> None:
        capabilities = {
            record["id"]: record
            for record in self.catalog["records"]
            if record["kind"] == "capability"
        }
        self.assertNotIn("metrics", capabilities["dual-b70-tp2-serving-v1"])
        self.assertEqual(capabilities["qwen38-fp8-w8a16-v1"]["evidence_status"], "provisional")

    def test_render_is_deterministic_and_mentions_trust_limits(self) -> None:
        first = renderer.render_catalog(self.catalog)
        second = renderer.render_catalog(self.catalog)
        self.assertEqual(first, second)
        self.assertIn("Generated from `data/benchmarks.v1.json`", first)
        self.assertIn("Qwen3.8-27B FP8 W8A16", first)
        self.assertIn("Qwen3.8-Flash-Next", first)
        self.assertIn("Provisional", first)
        self.assertNotIn("/home/", first)
        self.assertNotIn("192.168.", first)
        self.assertNotIn("B70-DOCS", first)


if __name__ == "__main__":
    unittest.main()
