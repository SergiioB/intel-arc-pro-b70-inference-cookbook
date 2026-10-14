#!/usr/bin/env python3
"""Boundary regressions for explicit-unit pricing and immutable source capture."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("research_math", Path(__file__).with_name("research_math.py"))
assert SPEC is not None and SPEC.loader is not None
MATH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATH)

PROVENANCE = {"kind": "assumption", "references": ["hypothesis:unit-example"],
              "note": "Synthetic arithmetic example, not measured hardware"}


def model(name: str, **fields) -> dict:
    return {"schema_version": 1, name: {**fields, "provenance": copy.deepcopy(PROVENANCE)}}


def roofline() -> dict:
    return model("roofline", bytes_per_step=2_000_000_000, flops_per_step=4_000_000_000_000,
                 bandwidth_gbps=100, compute_tflops=100, serial_ms=2, tokens_per_step=4)


class PricingTests(unittest.TestCase):
    def test_roofline_uses_max_and_decimal_bandwidth(self):
        result = MATH.price(roofline())["roofline"]
        self.assertEqual(result["memory_ms"], 20)
        self.assertEqual(result["compute_ms"], 40)
        self.assertEqual(result["step_ms_lower_bound"], 42)
        self.assertAlmostEqual(result["tokens_per_second_upper_bound"], 4000 / 42)
        memory_bound = roofline()
        memory_bound["roofline"]["bytes_per_step"] = 2**30
        memory_bound["roofline"]["flops_per_step"] = 0
        memory_bound["roofline"]["bandwidth_gbps"] = 1
        self.assertAlmostEqual(MATH.price(memory_bound)["roofline"]["memory_ms"], 1073.741824)

    def test_amdahl_edges_and_local_slowdown(self):
        for fraction, local, expected in ((0, 100, 1), (1, 4, 4), (0.5, 2, 4 / 3), (0.5, 0.5, 2 / 3)):
            with self.subTest(fraction=fraction, local=local):
                result = MATH.price(model("amdahl", wall_fraction=fraction, local_speedup=local))["amdahl"]
                self.assertAlmostEqual(result["predicted_wall_speedup"], expected)
                self.assertEqual(result["ceiling_unbounded"], fraction == 1)
                self.assertEqual(result["infinite_local_speedup_ceiling"], None if fraction == 1 else 1 / (1 - fraction))

    def test_speculation_uses_measured_marginals_not_iid_or_conditional(self):
        data = model("speculation", proposal_depth=3, marginal_survivals=[0.9, 0.5, 0.1],
                     survival_evidence=["results/survival-counts.json"], draft_ms=2,
                     verify_ms=5, overhead_ms=3, baseline_token_ms=8)
        result = MATH.price(data)["speculation"]
        self.assertEqual(result["expected_tokens_per_cycle"], 2.5)
        self.assertEqual(result["cycle_ms"], 10)
        self.assertEqual(result["predicted_token_ms"], 4)
        self.assertEqual(result["predicted_speedup"], 2)
        data["speculation"]["proposal_depth"] = 0
        data["speculation"]["marginal_survivals"] = []
        self.assertEqual(MATH.price(data)["speculation"]["expected_tokens_per_cycle"], 1)

    def test_speculation_rejects_invalid_survival_sequence_and_missing_evidence(self):
        data = model("speculation", proposal_depth=2, marginal_survivals=[0.8, 0.5],
                     survival_evidence=["run:prefix-observations"], draft_ms=1,
                     verify_ms=2, overhead_ms=0, baseline_token_ms=2)
        for survivals in ([0.5, 0.8], [1.1, 0.8], [-0.1, 0], [0.8], [True, 0.5]):
            with self.subTest(survivals=survivals):
                invalid = copy.deepcopy(data)
                invalid["speculation"]["marginal_survivals"] = survivals
                with self.assertRaises(MATH.InputError):
                    MATH.price(invalid)
        data["speculation"]["survival_evidence"] = []
        with self.assertRaises(MATH.InputError):
            MATH.price(data)

    def test_kv_counts_only_declared_attention_layers_and_separate_totals(self):
        data = model("kv", full_attention_layers=4, kv_heads=2, head_dim=16,
                     context_tokens=128, batch=3, kv_bytes=0.5,
                     recurrent_state_bytes=100, workspace_bytes=200, weights_bytes=300)
        result = MATH.price(data)["kv"]
        self.assertEqual(result["full_attention_kv_bytes"], 49152)
        self.assertEqual(result["total_bytes"], 49752)
        self.assertEqual(result["total_GB"], 49752 / 10**9)
        self.assertEqual(result["total_GiB"], 49752 / 2**30)
        data["kv"].update(full_attention_layers=0, kv_heads=0, head_dim=0, kv_bytes=0)
        self.assertEqual(MATH.price(data)["kv"]["total_bytes"], 600)
        data["kv"]["full_attention_layers"] = 1
        with self.assertRaises(MATH.InputError):
            MATH.price(data)

    def test_transfer_is_additive_and_evidenced_overlap_is_subtracted_once(self):
        data = roofline()
        data.update(model("transfer", bytes_per_step=100_000_000, bandwidth_gbps=10,
                          collective_count=2, latency_us=500))
        result = MATH.price(data)
        self.assertEqual(result["transfer"]["payload_ms"], 10)
        self.assertEqual(result["transfer"]["collective_latency_ms"], 1)
        self.assertEqual(result["combined_step"]["step_ms"], 53)
        data["transfer"].update(overlap_ms=7, overlap_evidence=["run:timeline"])
        result = MATH.price(data)
        self.assertEqual(result["transfer"]["incremental_ms"], 4)
        self.assertEqual(result["combined_step"]["step_ms"], 46)
        self.assertAlmostEqual(result["combined_step"]["tokens_per_second"], 4000 / 46)
        for overlap in (11.1, 43):
            data["transfer"]["overlap_ms"] = overlap
            with self.assertRaises(MATH.InputError):
                MATH.price(data)
        data["transfer"]["overlap_ms"] = 5
        del data["transfer"]["overlap_evidence"]
        with self.assertRaises(MATH.InputError):
            MATH.price(data)
        data["transfer"]["overlap_evidence"] = ["run:timeline"]
        del data["roofline"]
        with self.assertRaises(MATH.InputError):
            MATH.price(data)

    def test_required_unknown_units_and_nonfinite_values_fail_closed(self):
        for key, value in (("bandwidth_gbps", 0), ("compute_tflops", -1), ("serial_ms", None),
                           ("serial_ms", float("nan")), ("serial_ms", float("inf")),
                           ("tokens_per_step", False), ("bytes_per_step", 1.5),
                           ("bandwidth_gbps", "100")):
            with self.subTest(key=key, value=value):
                data = roofline()
                data["roofline"][key] = value
                with self.assertRaises(MATH.InputError):
                    MATH.price(data)
        for mutation in ("missing", "unknown", "provenance", "zero_step"):
            data = roofline()
            if mutation == "missing":
                del data["roofline"]["serial_ms"]
            elif mutation == "unknown":
                data["roofline"]["bandwidth_gibps"] = 100
            elif mutation == "provenance":
                data["roofline"]["provenance"]["references"] = []
            else:
                data["roofline"].update(bytes_per_step=0, flops_per_step=0, serial_ms=0)
            with self.subTest(mutation=mutation), self.assertRaises(MATH.InputError):
                MATH.price(data)
        for fraction, speedup in ((1.1, 2), (-0.1, 2), (0.5, 0)):
            with self.assertRaises(MATH.InputError):
                MATH.price(model("amdahl", wall_fraction=fraction, local_speedup=speedup))

    def test_input_hash_covers_exact_bytes_and_duplicate_json_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "price.json"
            content = json.dumps(roofline()).encode()
            path.write_bytes(content)
            result = MATH.read_price(path)
            self.assertEqual(result["input"]["sha256"], hashlib.sha256(content).hexdigest())
            self.assertEqual(result["input"]["values"], roofline())
            path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaises(MATH.InputError):
                MATH.read_price(path)


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, headers=None, url="https://example.org/official", status=200):
        super().__init__(payload)
        self.headers = {"Content-Type": "text/plain; charset=utf-8", **(headers or {})}
        self.url = url
        self.status = status

    def geturl(self):
        return self.url


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_snapshot_preserves_payload_and_claim_without_overwriting(self):
        payload = b"Official source bytes\n\x00not a search snippet"
        digest = hashlib.sha256(payload).hexdigest()
        with patch.object(MATH, "fetch_source", return_value=(payload, "application/octet-stream", "https://example.org/final")):
            result = MATH.snapshot_source(self.root, "https://example.org/official", "External statement",
                                          "RQ14 engine evidence", Path("research/references"))
            content = self.root / result["content_path"]
            metadata = self.root / result["metadata_path"]
            self.assertEqual(content.read_bytes(), payload)
            stored = json.loads(metadata.read_text(encoding="utf-8"))
            self.assertEqual(stored["sha256"], digest)
            self.assertEqual(stored["url"], "https://example.org/official")
            self.assertEqual(stored["final_url"], "https://example.org/final")
            self.assertEqual(stored["claim"], "External statement")
            self.assertEqual(stored["classification"], "external_claim_not_validated_evidence")
            self.assertEqual(content.stat().st_mode & 0o222, 0)
            before = metadata.read_bytes()
            with self.assertRaises(MATH.InputError):
                MATH.snapshot_source(self.root, "https://example.org/official", "Changed claim",
                                     "Different relevance", Path("research/references"))
            self.assertEqual(metadata.read_bytes(), before)
            self.assertEqual(content.read_bytes(), payload)

    def test_output_containment_and_existing_dated_run(self):
        for path in ("research/references/../../registry", "results/2026/13/run", "results/2026/10/missing",
                     "registry", "../outside", "results/2026/10"):
            with self.subTest(path=path), self.assertRaises(MATH.InputError):
                MATH.safe_output(self.root, Path(path))
        run = self.root / "results/2026/10/run-one"
        run.mkdir(parents=True)
        self.assertEqual(MATH.safe_output(self.root, run / "sources"), run / "sources")
        references_dir = self.root / "research/references"
        references_dir.mkdir(parents=True)
        (references_dir / "escape").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(MATH.InputError):
            MATH.safe_output(self.root, references_dir / "escape/source")

    def test_credentials_http_and_unsafe_redirects_are_rejected(self):
        for url in ("http://example.org/a", "https://user:secret@example.org/a", "https://user@example.org/a",
                    "https://example.org/a\n", "https://example.org/a#fragment", "https:///missing"):
            with self.subTest(url=url), self.assertRaises(MATH.InputError):
                MATH.validate_url(url)
        with self.assertRaises(MATH.InputError):
            MATH.SafeRedirect().redirect_request(None, None, 302, "redirect", {}, "http://example.org/a")

    def test_fetch_rejects_oversize_truncated_encoded_and_empty_payloads(self):
        cases = [(b"x", {"Content-Length": "9"}), (b"12345", {}),
                 (b"x", {"Content-Encoding": "gzip"}), (b"", {}),
                 (b"x", {"Content-Length": "invalid"}), (b"x", {"Content-Length": "5"})]
        for payload, headers in cases:
            with self.subTest(payload=payload, headers=headers):
                with patch.object(MATH, "MAX_SOURCE_BYTES", 4), patch.object(MATH.urllib.request, "build_opener") as builder:
                    builder.return_value.open.return_value = FakeResponse(payload, headers)
                    with self.assertRaises(MATH.InputError):
                        MATH.fetch_source("https://example.org/official")

    def test_fetch_keeps_actual_bytes_and_enforces_elapsed_deadline(self):
        with patch.object(MATH.urllib.request, "build_opener") as builder:
            builder.return_value.open.return_value = FakeResponse(b"abc", {"Content-Length": "3"})
            self.assertEqual(MATH.fetch_source("https://example.org/official"),
                             (b"abc", "text/plain; charset=utf-8", "https://example.org/official"))
            builder.return_value.open.return_value = FakeResponse(b"abc")
            with patch.object(MATH.time, "monotonic", side_effect=[0, MATH.FETCH_DEADLINE_S + 1]):
                with self.assertRaises(MATH.InputError):
                    MATH.fetch_source("https://example.org/official")


if __name__ == "__main__":
    unittest.main()
