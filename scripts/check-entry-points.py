#!/usr/bin/env python3
"""Keep the first-run guide aligned with the reviewed public route selection."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(root: Path) -> list[str]:
    policy = json.loads((root / "data/first-run.v1.json").read_text())
    start = (root / "START.md").read_text()
    readme = (root / "README.md").read_text()
    errors = []
    route = policy["primary_route"]
    if not (root / route).is_file():
        errors.append(f"primary route is missing: {route}")
    for name, text in (("START.md", start), ("README.md", readme)):
        if f"]({route})" not in text:
            errors.append(f"{name} does not link the primary route: {route}")
    for artifact in policy["excluded_artifacts"]:
        if artifact in start:
            errors.append(f"START.md selects an excluded artifact: {artifact}")
    launches = [shlex.split(line) for line in start.splitlines() if line.startswith("b70 launch ")]
    if not launches:
        errors.append("START.md has no inspectable launcher command")
    for tokens in launches:
        if tokens[2] != policy["launcher_model"]:
            errors.append("first-run model differs from the reviewed selection")
        if "-e" not in tokens or tokens[tokens.index("-e") + 1] != policy["launcher_engine"]:
            errors.append("first-run engine differs from the reviewed selection")
        if "--cache" not in tokens or tokens[tokens.index("--cache") + 1] != "off":
            errors.append("first-run command must disable prefix reuse")
        if (
            "--extra" not in tokens
            or "vllm.speculative_config=null" not in tokens[tokens.index("--extra") + 1]
        ):
            errors.append("first-run command must disable the EXL3 drafter explicitly")
    return errors


def main() -> int:
    errors = check(ROOT)
    for error in errors:
        print(error, file=sys.stderr)
    if not errors:
        print("OK: README and quick start select the reviewed first-run route")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
