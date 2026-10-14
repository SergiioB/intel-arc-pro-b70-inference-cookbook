#!/usr/bin/env python3
"""Select one actionable research question; never launch hardware or adopt a route."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def program_errors(program: dict) -> list[str]:
    questions = program.get("questions", [])
    by_id = {q.get("id"): q for q in questions}
    errors = []
    if len(by_id) != len(questions):
        errors.append("duplicate research question id")
    visiting, visited = set(), set()

    def visit(key: str) -> None:
        if key in visiting:
            errors.append(f"dependency cycle at {key}")
            return
        if key in visited or key not in by_id:
            return
        visiting.add(key)
        for dependency in by_id[key].get("dependencies", []):
            if dependency not in by_id:
                errors.append(f"{key}: unknown dependency {dependency}")
            visit(dependency)
        visiting.remove(key)
        visited.add(key)

    for key in by_id:
        visit(key)
    portfolio = program.get("portfolio")
    if portfolio:
        ready = [q for q in questions if q.get("status") == "ready"]
        if len(ready) > portfolio["max_ready"]:
            errors.append(f"ready queue exceeds {portfolio['max_ready']} questions")
        for q in questions:
            if q.get("status") == "closed" and q.get("resolution") not in {"passed", "rejected"}:
                errors.append(f"{q['id']}: closed requires passed or rejected resolution")
            if q.get("operator_hold") and q.get("status") == "ready":
                errors.append(f"{q['id']}: operator-held question cannot be ready")
            if q.get("status") == "ready":
                for dep in q.get("dependencies", []):
                    prerequisite = by_id.get(dep, {})
                    if prerequisite.get("status") != "closed" or prerequisite.get("resolution") != "passed":
                        errors.append(f"{q['id']}: ready despite unsatisfied prerequisite {dep}")
                for field in ("focus", "economic_gate", "execution_budget"):
                    if not q.get(field):
                        errors.append(f"{q['id']}: ready requires {field}")
        for q in questions:
            extra = (q.get("execution_budget") or {}).get("operator_extra_attempts", [])
            if not isinstance(extra, list) or not all(
                    isinstance(row, dict) and all(isinstance(row.get(k), str) and row[k].strip()
                                                  for k in ("authorized_on", "operator_words", "reason"))
                    for row in extra):
                errors.append(f"{q.get('id')}: operator_extra_attempts entries need authorized_on, "
                              "operator_words and reason")
    return errors


def attempt_limit(program: dict, question: dict) -> int:
    """Attempts allowed per unchanged mechanism: the portfolio default plus one per
    recorded operator authorization (execution_budget.operator_extra_attempts)."""
    base = program.get("portfolio", {}).get("max_attempts_per_mechanism", 2)
    extra = (question.get("execution_budget") or {}).get("operator_extra_attempts", [])
    return base + (len(extra) if isinstance(extra, list) else 0)


def eligible(question: dict, questions: dict) -> tuple[bool, str]:
    if question.get("operator_hold"):
        return False, str(question["operator_hold"])
    if question.get("status") != "ready":
        return False, str(question.get("status", "missing status"))
    for key in question.get("dependencies", []):
        dep = questions.get(key, {})
        if dep.get("status") != "closed" or dep.get("resolution") != "passed":
            return False, f"prerequisite {key} has not passed"
    return True, "ready for the named next action, subject to live preflight"


def next_question(root: Path, requested: str | None = None) -> dict:
    program = json.loads((root / "registry/research-program.json").read_text())
    errors = program_errors(program)
    if errors:
        raise ValueError("; ".join(errors))
    rows = sorted(program["questions"], key=lambda q: q["priority"])
    by_id = {q["id"]: q for q in rows}
    if requested and requested not in by_id:
        raise ValueError(f"unknown question {requested}")
    selected = None
    exclusions = []
    for question in rows:
        ok, reason = eligible(question, by_id)
        if requested and question["id"] != requested:
            continue
        if ok and selected is None:
            selected = question
        elif not ok:
            exclusions.append({"id": question["id"], "reason": reason})
    if selected is None:
        return {"halt": True, "reason": "No actionable question. Resolve the named gate; do not invent another sweep.",
                "excluded": exclusions, "execution_authorized": False}

    # A run without registration is unfinished work, not permission to start over.
    ledger = json.loads((root / "registry/research-ledger.json").read_text())["entries"]
    registered = {entry.get("run_root") for entry in ledger if entry.get("run_root")}
    attempts = []
    for manifest_path in sorted((root / "results").glob("[0-9][0-9][0-9][0-9]/[0-9][0-9]/*/plan.json")):
        plan = json.loads(manifest_path.read_text())
        if plan.get("question_id") == selected["id"]:
            attempts.append((manifest_path.parent, plan))
    unfinished = [path for path, _ in attempts if path.relative_to(root).as_posix() not in registered]
    if unfinished:
        path = unfinished[0].relative_to(root).as_posix()
        return {"halt": True, "question_id": selected["id"], "reason": "Finish or register existing evidence before another run.",
                "run": path, "next_action": f"python3 tools/research.py assess --run {path}", "execution_authorized": False}
    same = [path for path, plan in attempts if plan.get("changed_variable") == selected["changed_variable"]
            and plan.get("hypothesis") == selected["hypothesis"]]
    if len(same) >= attempt_limit(program, selected):
        return {"halt": True, "question_id": selected["id"], "reason": "Attempt limit reached. Close the question or record new mechanism evidence.",
                "execution_authorized": False, "runs": [p.relative_to(root).as_posix() for p in same]}
    wanted = set(selected.get("evidence_ids", []))
    prior = [{k: e[k] for k in ("id", "outcome", "validation_scope", "adoption", "superseded_by") if k in e}
             for e in ledger if e.get("id") in wanted]
    paths = ["docs/governance/agent-operating-system.md", "docs/methods/architecture-diagnostic-playbook.md",
             "docs/methods/benchmark-protocol.md", "docs/reference/research-sources.json"]
    paths += selected.get("evidence_paths", [])
    paths = list(dict.fromkeys(paths))
    missing = [p for p in paths if not (root / p).is_file()]
    if missing:
        raise ValueError("missing working-set evidence: " + ", ".join(missing))
    return {"halt": False, "question": selected, "prior_outcomes": prior, "working_set": paths[:10],
            "portfolio": program.get("portfolio", {}), "excluded": exclusions,
            "execution_authorized": False,
            "next_action": selected["next_action"],
            "decision_rule": "Read evidence, price the removable wall time, then precommit one falsifying probe. A passed diagnostic is not an inference win."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["next"])
    parser.add_argument("--question", help="Inspect one question without bypassing its hold or dependency")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        payload = next_question(args.root.resolve(), args.question)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"next: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2))
    return 2 if payload["halt"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
