#!/usr/bin/env python3
"""Fail closed when the cookbook control map points at missing authorities."""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "data" / "system-map.v1.json"
REQUIRED_INTENTS = {
    "read",
    "setup",
    "reproduce",
    "connect",
    "operate",
    "submit",
    "publish",
    "triage",
}


def validate_system_map(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if data.get("schema_version") != "1.0":
        errors.append("schema_version must be 1.0")

    intents = data.get("intents")
    if not isinstance(intents, dict) or set(intents) != REQUIRED_INTENTS:
        errors.append(f"intents must be exactly {', '.join(sorted(REQUIRED_INTENTS))}")
        intents = {}

    authorities = data.get("authorities")
    if not isinstance(authorities, dict) or not authorities:
        errors.append("authorities must be a non-empty object")
        authorities = {}

    concrete_paths: set[str] = set(authorities.values())
    for intent, route in intents.items():
        if not isinstance(route, dict) or "start" not in route or "then" not in route:
            errors.append(f"intent {intent} must define start and then")
            continue
        concrete_paths.add(route["start"])
        concrete_paths.update(route["then"])

    for relative in sorted(concrete_paths):
        if "<" in relative:
            continue
        if not (ROOT / relative).exists():
            errors.append(f"missing referenced path: {relative}")

    layers = data.get("layers")
    if not isinstance(layers, list) or [layer.get("id") for layer in layers] != [
        "L0",
        "L1",
        "L2",
        "L3",
        "L4",
    ]:
        errors.append("layers must be ordered L0 through L4")

    invariants = data.get("invariants")
    if not isinstance(invariants, list) or len(invariants) < 5:
        errors.append("at least five system invariants are required")

    validation = data.get("validation")
    if validation != ["python3 scripts/check-repo.py"]:
        errors.append("validation must contain only the stable check-repo entry point")

    return errors


def main() -> int:
    try:
        data = json.loads(MAP_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read {MAP_PATH.relative_to(ROOT)}: {exc}", file=sys.stderr)
        return 1

    errors = validate_system_map(data)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(
        f"OK: {MAP_PATH.relative_to(ROOT)} "
        f"({len(data['authorities'])} authorities, {len(data['invariants'])} invariants)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
