#!/usr/bin/env python3
"""Validate and render the canonical public benchmark catalog."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
from typing import Any

ROOT = pathlib.Path(__file__).parents[1]
DEFAULT_INPUT = ROOT / "data" / "benchmarks.v1.json"
DEFAULT_OUTPUT = ROOT / "docs" / "BENCHMARK-CATALOG.md"
ALLOWED_KINDS = {"benchmark", "capability"}
ALLOWED_STATUSES = {"official-lab", "community-reviewed", "validated", "provisional"}
ALLOWED_CONCURRENCY = {1, 2, 4, 8, 16, 32}
PRIVATE_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [r"/home/", r"B70-DOCS", r"192\.168\.\d+\.\d+", r"(?:api[_-]?key|bearer)\s*[:=]"]
]


def validate_record_identity(path: str, record: dict[str, Any], seen: set[str]) -> list[str]:
    """Check id, kind, status, required fields, and commit-pinned evidence."""
    errors: list[str] = []
    record_id = record.get("id")
    if not isinstance(record_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,100}", record_id):
        errors.append(f"{path}.id is invalid")
    elif record_id in seen:
        errors.append(f"{path}.id is duplicated")
    else:
        seen.add(record_id)
    kind = record.get("kind")
    if kind not in ALLOWED_KINDS:
        errors.append(f"{path}.kind is invalid")
    if record.get("evidence_status") not in ALLOWED_STATUSES:
        errors.append(f"{path}.evidence_status is invalid")
    for required in ["title", "summary", "hardware", "engine", "evidence"]:
        if not record.get(required):
            errors.append(f"{path}.{required} is required")
    evidence = record.get("evidence") or {}
    commit = evidence.get("commit", "")
    url = evidence.get("url", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        errors.append(f"{path}.evidence.commit must be a full SHA")
    if not isinstance(url, str) or f"/blob/{commit}/" not in url:
        errors.append(f"{path}.evidence.url must pin its commit")
    return errors


def validate_benchmark_metrics(path: str, metrics: list[Any]) -> list[str]:
    """Check that every ranking metric carries a finite value, name, and unit."""
    errors: list[str] = []
    for metric in metrics:
        if not isinstance(metric.get("value"), (int, float)) or not math.isfinite(metric["value"]):
            errors.append(f"{path}.metrics contains an invalid value")
        if not metric.get("name") or not metric.get("unit"):
            errors.append(f"{path}.metrics requires name and unit")
    return errors


def validate_benchmark_workload(path: str, record: dict[str, Any]) -> list[str]:
    """Check exact workload coordinates every ranked benchmark must publish."""
    errors: list[str] = []
    workload = record.get("workload") or {}
    if workload.get("concurrency") not in ALLOWED_CONCURRENCY:
        errors.append(f"{path}.workload.concurrency is invalid")
    for field in ["prompt_tokens", "output_tokens"]:
        if not isinstance(workload.get(field), int) or workload[field] <= 0:
            errors.append(f"{path}.workload.{field} must be a positive integer")
    metrics = record.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        errors.append(f"{path}.metrics must be non-empty")
    else:
        errors.extend(validate_benchmark_metrics(path, metrics))
    statistics = record.get("statistics") or {}
    if statistics.get("statistic") not in {"median", "mean", "single-observation"}:
        errors.append(f"{path}.statistics.statistic is invalid")
    if not isinstance(statistics.get("samples"), int) or statistics["samples"] <= 0:
        errors.append(f"{path}.statistics.samples must be positive")
    return errors


def validate_catalog(catalog: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if catalog.get("schema_version") != "1.0":
        errors.append("schema_version must be 1.0")
    records = catalog.get("records")
    if not isinstance(records, list) or not records:
        return errors + ["records must be a non-empty array"]
    seen: set[str] = set()
    for index, record in enumerate(records):
        path = f"records[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{path} must be an object")
            continue
        errors.extend(validate_record_identity(path, record, seen))
        kind = record.get("kind")
        if kind == "benchmark":
            errors.extend(validate_benchmark_workload(path, record))
        elif "metrics" in record:
            errors.append(f"{path}: capability records cannot carry ranking metrics")
    encoded = json.dumps(catalog, ensure_ascii=False)
    for pattern in PRIVATE_PATTERNS:
        if pattern.search(encoded):
            errors.append(f"catalog contains private pattern: {pattern.pattern}")
    return errors


def status_label(value: str) -> str:
    return {
        "official-lab": "Official lab",
        "community-reviewed": "Community reviewed",
        "validated": "Validated",
        "provisional": "Provisional",
    }.get(value, value)


def metric_text(record: dict[str, Any]) -> str:
    return " · ".join(
        f"{metric['value']:g} {metric['unit']} ({metric['name'].replace('_', ' ')})"
        for metric in record["metrics"]
    )


def render_catalog(catalog: dict[str, Any]) -> str:
    benchmarks = [record for record in catalog["records"] if record["kind"] == "benchmark"]
    capabilities = [record for record in catalog["records"] if record["kind"] == "capability"]
    lines = [
        "<!-- GENERATED: source=data/benchmarks.v1.json "
        "command='python3 scripts/render-benchmark-catalog.py'; DO NOT EDIT -->",
        "",
        "# Public benchmark catalog",
        "",
        "> Generated from `data/benchmarks.v1.json`. Edit the JSON, then run",
        "> `python3 scripts/render-benchmark-catalog.py`. Do not edit this table directly.",
        "",
        "Benchmark rows require exact workload coordinates, sample counts, metric "
        "semantics, and commit-pinned evidence. Capability notes without complete "
        "coordinates stay outside rankings.",
        "",
        "## Benchmarks",
        "",
        "| Model | Hardware | Engine | Workload | Result | Evidence |",
        "|---|---|---|---|---|---|",
    ]
    for record in benchmarks:
        workload = record["workload"]
        hardware = record["hardware"]
        profile = (
            f"P{workload['prompt_tokens']} / G{workload['output_tokens']}"
            f" / C{workload['concurrency']} · n={record['statistics']['samples']}"
            f" {record['statistics']['statistic']}"
        )
        evidence = f"[{status_label(record['evidence_status'])}]({record['evidence']['url']})"
        lines.append(
            f"| {record['title']} | {hardware['gpu_count']}× {hardware['gpu_model']} "
            f"| {record['engine']['name']} | {profile} | {metric_text(record)} | {evidence} |"
        )
    lines.extend(
        [
            "",
            "## Non-ranking capabilities",
            "",
            "| Capability | Hardware | Status | What is established | Evidence |",
            "|---|---|---|---|---|",
        ]
    )
    for record in capabilities:
        hardware = record["hardware"]
        evidence = f"[source]({record['evidence']['url']})"
        lines.append(
            f"| {record['title']} | {hardware['gpu_count']}× {hardware['gpu_model']} "
            f"| {status_label(record['evidence_status'])} | {record['summary']} | {evidence} |"
        )
    lines.extend(
        [
            "",
            "## Trust labels",
            "",
            "- **Official lab:** measured under the cookbook's documented process.",
            "- **Community reviewed:** schema and evidence reviewed; "
            "contributor controls the machine.",
            "- **Validated:** an operational capability worked; "
            "any model-specific numeric records are published separately.",
            "- **Provisional:** an early numeric claim is documented but lacks "
            "one or more required workload coordinates.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    catalog = json.loads(args.input.read_text())
    errors = validate_catalog(catalog)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    rendered = render_catalog(catalog)
    if args.check:
        if not args.output.exists() or args.output.read_text() != rendered:
            print(
                f"ERROR: {args.output} is stale; run this script without --check", file=sys.stderr
            )
            return 1
        print(f"Catalog valid and current: {len(catalog['records'])} records")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(f"Rendered {len(catalog['records'])} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
