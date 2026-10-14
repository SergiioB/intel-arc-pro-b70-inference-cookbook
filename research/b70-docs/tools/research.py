#!/usr/bin/env python3
"""research.py — research program, generated views, and registry integrity.

The B70 research system has three typed registries and generated human views:

- `registry/research-program.json`  current ranked research questions (the only
  place a live hypothesis, changed variable, probe, gate, and stop rule live)
- `registry/research-ledger.json`   append-only outcome record (what ran, what
  it measured, what is guarded)
- `registry/model-status.json`      current route per model family

Generated views (never hand-edit): `research/PROGRAM.md`, `research/LEDGER.md`,
`models/STATUS.md`, `docs/indexes/research-coverage.md`.

Commands:
  render    write all generated views from the registries
  check     validate registries and fail when a generated view is stale
  coverage  print registration coverage: notes/result roots lacking a record

Exit 0 = clean, 1 = validation or parity errors (warnings alone do not fail).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from research_queue import program_errors

ROOT = Path(__file__).resolve().parents[1]

LEDGER_PATH = ROOT / "registry/research-ledger.json"
MODEL_PATH = ROOT / "registry/model-status.json"
PROGRAM_PATH = ROOT / "registry/research-program.json"
CAMPAIGN_PATH = ROOT / "registry/campaigns.yaml"

GENERATED = {
    ROOT / "research/PROGRAM.md": "program",
    ROOT / "research/LEDGER.md": "ledger",
    ROOT / "models/STATUS.md": "models",
    ROOT / "docs/indexes/research-coverage.md": "coverage",
}

QUESTION_STATUSES = {"ready", "gated", "deferred", "closed"}
LEDGER_OUTCOMES = {"blocked", "validated", "validated_screen", "failed", "provisional", "superseded"}
QUESTION_REQUIRED = (
    "id", "priority", "status", "lane", "title", "objective", "scope", "hypothesis",
    "evidence_ids", "evidence_paths", "changed_variable", "smallest_probe",
    "success_gate", "stop_rule", "dependencies", "uncertainty", "next_action",
)
PATH_TOKEN = re.compile(r"(?:results|research|campaigns|decisions|models|engines|docs|registry|archive)/[A-Za-z0-9._\-/]+")


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def load_json(path: Path, report: Report):
    if not path.is_file():
        report.error(f"missing registry {path.relative_to(ROOT)}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.error(f"{path.relative_to(ROOT)}: invalid JSON ({exc})")
        return None


def load_campaigns(report: Report):
    if not CAMPAIGN_PATH.is_file():
        report.error("missing registry/campaigns.yaml")
        return []
    try:
        data = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.error(f"registry/campaigns.yaml: invalid JSON ({exc})")
        return []
    rows = data.get("campaigns") if isinstance(data, dict) else data
    return [row for row in (rows or []) if isinstance(row, dict)]


def resolve(text: str) -> Path | None:
    """First existing repo path mentioned in free-text evidence, else None."""
    for match in PATH_TOKEN.finditer(text or ""):
        candidate = match.group(0).rstrip(".,;:)")
        base = ROOT / candidate.split(" (")[0]
        if base.exists():
            return Path(candidate.split(" (")[0])
    return None


def first_pointer(entry: dict) -> str:
    paths = entry.get("evidence_paths") or []
    if paths:
        return f"`{paths[0]}`" + (" +more" if len(paths) > 1 else "")
    found = resolve(str(entry.get("evidence") or ""))
    return f"`{found.as_posix()}`" if found else "see row"


# ---------------------------------------------------------------- validation

def validate_ledger(report: Report) -> list[dict]:
    data = load_json(LEDGER_PATH, report)
    entries = [e for e in (data or {}).get("entries", []) if isinstance(e, dict)]
    ids = [str(e.get("id")) for e in entries]

    seen: set[str] = set()
    for index, entry in enumerate(entries):
        label = entry.get("id") or f"row {index}"
        entry_id = str(entry.get("id") or "")
        if not re.fullmatch(r"RL-\d{3}", entry_id):
            report.error(f"research-ledger {label}: id must match RL-NNN")
        if entry_id in seen:
            report.error(f"research-ledger: duplicate id {entry_id}")
        seen.add(entry_id)
        for key in ("date", "question", "outcome", "evidence"):
            if not entry.get(key):
                report.error(f"research-ledger {label}: missing {key}")
        if entry.get("outcome") not in LEDGER_OUTCOMES:
            report.error(f"research-ledger {label}: unknown outcome {entry.get('outcome')!r}")
        successor = entry.get("superseded_by")
        if successor and successor not in ids:
            report.error(f"research-ledger {label}: superseded_by {successor} does not exist")
        pattern = entry.get("blocked_pattern")
        if pattern:
            try:
                re.compile(pattern)
            except re.error:
                # Legacy rows carry sentence-shaped guards. Keep them visible,
                # but only claim regex matching where the pattern compiles.
                if not re.search(r"[A-Za-z] [a-z]", pattern):
                    report.error(f"research-ledger {label}: invalid blocked_pattern regex")
                else:
                    report.warn(f"research-ledger {label}: blocked_pattern is descriptive text, not a regex")
        for path in entry.get("evidence_paths") or []:
            if not (ROOT / str(path)).exists():
                report.error(f"research-ledger {label}: missing evidence_paths entry {path}")
        if not entry.get("evidence_paths") and not resolve(str(entry.get("evidence") or "")):
            report.warn(f"research-ledger {label}: evidence has no resolvable repo path")
    return entries


def validate_models(report: Report) -> dict:
    data = load_json(MODEL_PATH, report) or {}
    rows = [r for r in data.get("models", []) if isinstance(r, dict)]
    families: set[str] = set()
    for row in rows:
        family = str(row.get("family") or "")
        label = family or "model row"
        if not family:
            report.error("model-status: row without family")
        if family in families:
            report.error(f"model-status: duplicate family {family}")
        families.add(family)
        for key in ("engine", "artifact", "open_gate", "dossier"):
            if key not in row:
                report.error(f"model-status {label}: missing {key}")
        dossier = row.get("dossier")
        if dossier and not (ROOT / str(dossier)).exists():
            report.error(f"model-status {label}: missing dossier {dossier}")
    for check in data.get("external_crosschecks", []) or []:
        doc = check.get("doc")
        if doc and not (ROOT / str(doc)).exists():
            report.error(f"model-status external_crosschecks: missing doc {doc}")
    return data


def validate_program(report: Report, ledger_ids: set[str]) -> dict:
    data = load_json(PROGRAM_PATH, report) or {}
    questions = [q for q in data.get("questions", []) if isinstance(q, dict)]
    ids = [str(q.get("id")) for q in questions]
    priorities: set[int] = set()

    for question in questions:
        label = question.get("id") or "question"
        for key in QUESTION_REQUIRED:
            if key not in question:
                report.error(f"research-program {label}: missing {key}")
        if question.get("status") not in QUESTION_STATUSES:
            report.error(f"research-program {label}: unknown status {question.get('status')!r}")
        priority = question.get("priority")
        if not isinstance(priority, int) or priority < 1:
            report.error(f"research-program {label}: priority must be a positive integer")
        elif priority in priorities:
            report.error(f"research-program {label}: duplicate priority {priority}")
        else:
            priorities.add(priority)
        for path in question.get("evidence_paths") or []:
            if not (ROOT / str(path)).exists():
                report.error(f"research-program {label}: missing evidence_paths entry {path}")
        for evidence_id in question.get("evidence_ids") or []:
            if evidence_id not in ledger_ids:
                report.error(f"research-program {label}: evidence id {evidence_id} is not in the ledger")
        for dependency in question.get("dependencies") or []:
            if dependency not in ids:
                report.error(f"research-program {label}: unknown dependency {dependency}")
        if not str(question.get("changed_variable") or "").strip():
            report.error(f"research-program {label}: empty changed_variable")
    for dependency_chain in ((q.get("id"), d) for q in questions for d in q.get("dependencies") or []):
        if dependency_chain[0] == dependency_chain[1]:
            report.error(f"research-program {dependency_chain[0]}: depends on itself")
    if not data.get("reviewed_through"):
        report.error("research-program: missing reviewed_through")
    for message in program_errors(data):
        report.error(f"research-program: {message}")
    return data


def validate_campaigns(report: Report) -> list[dict]:
    campaigns = load_campaigns(report)
    ids: set[str] = set()
    for campaign in campaigns:
        label = campaign.get("id") or "campaign"
        if not campaign.get("id"):
            report.error("campaigns: row without id")
        if campaign.get("id") in ids:
            report.error(f"campaigns: duplicate id {label}")
        ids.add(str(campaign.get("id")))
        record = campaign.get("record_path")
        if record and not (ROOT / str(record)).exists():
            report.error(f"campaigns {label}: missing record_path {record}")
        root = campaign.get("evidence_root")
        if root and not (ROOT / str(root)).exists() and campaign.get("status") != "planned":
            report.error(f"campaigns {label}: missing evidence_root {root}")
        if root and not (ROOT / str(root)).exists():
            report.warn(f"campaigns {label}: reserved evidence_root not created yet ({root})")
    return campaigns


# ------------------------------------------------------------------ coverage

def research_notes() -> list[Path]:
    base = ROOT / "research"
    generated = {"PROGRAM.md", "LEDGER.md"}  # rendered views, not independent notes
    return sorted(
        path for path in base.rglob("*.md")
        if not any(part.startswith(".") for part in path.relative_to(ROOT).parts)
        and ".sync-conflict-" not in path.name
        and path.name not in generated
    )


def result_roots() -> list[Path]:
    base = ROOT / "results" / "2026"
    return sorted(
        path for path in base.glob("*/*")
        if path.is_dir() and not path.name.startswith(".")
    )


def coverage_data(ledger: list[dict], campaigns: list[dict], program: dict) -> dict:
    corpus = "\n".join(
        str(entry.get("evidence") or "") + " " + " ".join(map(str, entry.get("evidence_paths") or []))
        for entry in ledger
    )
    corpus += "\n".join(str(c.get("evidence_root") or "") for c in campaigns)
    corpus += "\n".join(
        " ".join(map(str, q.get("evidence_paths") or [])) for q in program.get("questions", [])
    )
    corpus += json.dumps(program.get("reviewed", {}))

    notes = [p.relative_to(ROOT).as_posix() for p in research_notes()]
    roots = [p.relative_to(ROOT).as_posix() for p in result_roots()]
    return {
        "ledger_rows": len(ledger),
        "campaigns": len(campaigns),
        "questions": len(program.get("questions", [])),
        "research_notes_total": len(notes),
        "research_notes_unlinked": [note for note in notes if note not in corpus],
        "result_roots_total": len(roots),
        "result_roots_unregistered": [root for root in roots if root not in corpus],
    }


# -------------------------------------------------------------------- render

def header(title: str) -> list[str]:
    return [
        "<!-- GENERATED by tools/research.py — do not hand-edit -->",
        "",
        f"# {title}",
        "",
    ]


def render_program(program: dict) -> str:
    lines = header("Research program — ranked questions and their deciding probes")
    lines += [
        "Authority: `registry/research-program.json`. Every question names the one changed "
        "variable, the smallest probe that can kill it, the adoption gate, and the stop rule. "
        "Outcome history is `research/LEDGER.md`; current routes are `models/STATUS.md`.",
        "",
        f"Reviewed through: **{program.get('reviewed_through', 'unknown')}**. "
        "Run `python3 tools/research.py next` for one actionable question. "
        "Ready means the next source/math/probe step is permitted, not GPU authorization. "
        "Closed prerequisites need resolution=passed; rejected work never unlocks dependencies.",
        "",
        "## Active focus",
        "",
        "| # | id | status | focus | question | adoption gate |",
        "|---:|---|---|---|---|---|",
    ]
    for question in sorted(program.get("questions", []), key=lambda q: q["priority"]):
        if question["status"] == "ready":
            lines.append(
                f"| {question['priority']} | {question['id']} | ready | "
                f"{question.get('focus', question['lane'])} | {question['title']} | {question['success_gate']} |"
            )
    lines += ["", "## Parked and closed", "", "| id | state | reason / next condition |", "|---|---|---|"]
    for question in sorted(program.get("questions", []), key=lambda q: q["priority"]):
        if question["status"] != "ready":
            lines.append(f"| {question['id']} | {question['status']} | {question.get('operator_hold') or question['next_action']} |")
    lines.append("")
    for question in sorted(program.get("questions", []), key=lambda q: q["priority"]):
        scope = question.get("scope") or {}
        scope_text = "; ".join(f"{key}: {value}" for key, value in scope.items())
        dependencies = ", ".join(question.get("dependencies") or []) or "none"
        evidence_ids = ", ".join(question.get("evidence_ids") or []) or "none"
        evidence_paths = "<br>".join(f"`{path}`" for path in question.get("evidence_paths") or []) or "none"
        lines += [
            f"## RQ {question['priority']} — {question['id']} {question['title']}",
            "",
            f"State: **{question['status']}**; lane: {question['lane']}; prerequisites: {dependencies}.",
            "",
            f"1. **Objective and scope.** {question['objective']} {scope_text}",
            f"2. **Mechanism.** {question['hypothesis']} Changed variable: {question['changed_variable']}",
            f"3. **Deciding probe.** {question['smallest_probe']} Economic gate: {question.get('economic_gate', 'Parked; no new resource allocation.')}",
            f"4. **Adopt or stop.** {question['success_gate']} Stop: {question['stop_rule']}",
            f"5. **Next action.** {question['next_action']} Uncertainty: {question['uncertainty']}",
            "",
            f"Prior evidence: {evidence_ids}. {evidence_paths}",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def render_ledger(ledger: list[dict]) -> str:
    lines = header("Research ledger — append-only outcome record")
    lines += [
        "Authority: `registry/research-ledger.json`. Append-only: never edit a row except "
        "`superseded_by`. A row records what was tried and what it measured; it does not adopt "
        "a route (`models/STATUS.md`) and does not rank future work (`research/PROGRAM.md`).",
        "",
        "Reading rules:",
        "",
        "- `validated` means the named result held in the named scope. Re-measuring the identical "
        "cell is optional replication, not forbidden work.",
        "- `blocked` / `failed` guard the named path only. Re-entry needs a named changed variable, "
        "recorded in `registry/research-program.json` and the plan, not a blanket ban on the idea.",
        "- `provisional` / `validated_screen` cannot supply a public representative number.",
        "",
        "| id | date | outcome | question | evidence | guard | superseded_by |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in ledger:
        guard = entry.get("blocked_pattern") or "—"
        if len(str(guard)) > 120:
            guard = str(guard)[:117] + "…"
        question = str(entry.get("question") or "")
        if len(question) > 150:
            question = question[:147] + "…"
        lines.append(
            f"| {entry.get('id')} | {entry.get('date')} | {entry.get('outcome')} | {question} | "
            f"{first_pointer(entry)} | `{guard}` | {entry.get('superseded_by') or '—'} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def render_models(data: dict) -> str:
    lines = header("Model status — current route per family")
    lines += [
        "Authority: `registry/model-status.json`. A route row names the artifact and engine that "
        "actually passed its quality gates on this host. Historical numbers stay in `results/`; "
        "each figure here carries its cell and metric class.",
        "",
        "| family | engine · artifact | best recorded | cell · sampling | TP2 | PP2 | open gate | dossier |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in data.get("models", []):
        best = row.get("best_tok_s")
        best_text = f"{best} tok/s" if isinstance(best, (int, float)) else "—"
        lines.append(
            f"| {row.get('family')} | {row.get('engine')} · {row.get('artifact')} | {best_text} | "
            f"{row.get('cell')} · {row.get('sampling')} | {row.get('tp2_status')} | "
            f"{row.get('pp2_status')} | {row.get('open_gate')} | `{row.get('dossier')}` |"
        )
    lines += ["", "## External cross-checks", ""]
    for check in data.get("external_crosschecks", []) or []:
        lines += [
            f"### {check.get('date')} — {check.get('source')}",
            "",
            f"- State: {check.get('state')}",
            f"- Use: {check.get('use')}",
            f"- Record: `{check.get('doc')}`",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def render_coverage(coverage: dict) -> str:
    lines = header("Research coverage — what is registered and what is not")
    lines += [
        "Generated by `tools/research.py`. A research note or result root with no ledger row, "
        "campaign registration, or program reference is invisible to the anti-loop record and "
        "easy to propose as new work. This view makes that gap measurable.",
        "",
        f"- Ledger rows: **{coverage['ledger_rows']}**",
        f"- Registered campaigns: **{coverage['campaigns']}**",
        f"- Program questions: **{coverage['questions']}**",
        f"- Research notes: **{coverage['research_notes_total']}** "
        f"({len(coverage['research_notes_unlinked'])} not linked from any record)",
        f"- Result roots under `results/2026/`: **{coverage['result_roots_total']}** "
        f"({len(coverage['result_roots_unregistered'])} not named in any record)",
        "",
        "## Unlinked research notes (backlog for registration)",
        "",
    ]
    for note in coverage["research_notes_unlinked"][:40]:
        lines.append(f"- `{note}`")
    if len(coverage["research_notes_unlinked"]) > 40:
        lines.append(f"- … {len(coverage['research_notes_unlinked']) - 40} more")
    lines += ["", "## Unregistered result roots (newest first)", ""]
    for root in sorted(coverage["result_roots_unregistered"], reverse=True)[:40]:
        lines.append(f"- `{root}`")
    if len(coverage["result_roots_unregistered"]) > 40:
        lines.append(f"- … {len(coverage['result_roots_unregistered']) - 40} more")
    return "\n".join(lines).rstrip() + "\n"


def render_all() -> tuple[dict[str, str], Report]:
    report = Report()
    ledger = validate_ledger(report)
    models = validate_models(report)
    program = validate_program(report, {str(e.get("id")) for e in ledger})
    campaigns = validate_campaigns(report)
    coverage = coverage_data(ledger, campaigns, program)
    views = {
        "program": render_program(program),
        "ledger": render_ledger(ledger),
        "models": render_models(models),
        "coverage": render_coverage(coverage),
    }
    return views, report


def cmd_render(_args: argparse.Namespace) -> int:
    views, report = render_all()
    if report.errors:
        for message in report.errors:
            print(f"error: {message}", file=sys.stderr)
        return 1
    for path, key in GENERATED.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(views[key], encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")
    for message in report.warnings:
        print(f"warning: {message}", file=sys.stderr)
    if report.errors:
        for message in report.errors:
            print(f"error: {message}", file=sys.stderr)
        return 1
    return 0


def cmd_check(_args: argparse.Namespace) -> int:
    views, report = render_all()
    stale = []
    for path, key in GENERATED.items():
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current != views[key]:
            stale.append(path.relative_to(ROOT).as_posix())
    for message in report.warnings:
        print(f"warning: {message}", file=sys.stderr)
    if report.errors:
        for message in report.errors:
            print(f"error: {message}", file=sys.stderr)
    if stale:
        print(f"stale generated views: {', '.join(stale)} — run tools/research.py render", file=sys.stderr)
    if report.errors or stale:
        print("FAIL: research registry check", file=sys.stderr)
        return 1
    print("OK: research registries valid, generated views current")
    return 0


def cmd_coverage(_args: argparse.Namespace) -> int:
    report = Report()
    ledger = validate_ledger(report)
    campaigns = validate_campaigns(report)
    program = validate_program(report, {str(e.get("id")) for e in ledger})
    coverage = coverage_data(ledger, campaigns, program)
    print(json.dumps({
        "ledger_rows": coverage["ledger_rows"],
        "campaigns": coverage["campaigns"],
        "questions": coverage["questions"],
        "research_notes_total": coverage["research_notes_total"],
        "research_notes_unlinked": len(coverage["research_notes_unlinked"]),
        "result_roots_total": coverage["result_roots_total"],
        "result_roots_unregistered": len(coverage["result_roots_unregistered"]),
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "next":
        from research_queue import main as queue_main
        return queue_main(argv)
    if argv and argv[0] in {"run", "assess", "record"}:
        from experiment import main as experiment_main
        return experiment_main(argv)
    if argv and argv[0] in {"price", "source"}:
        from research_math import main as math_main
        return math_main(argv)
    parser = argparse.ArgumentParser(description="B70 research: choose, price, run, assess, retain. Use COMMAND --help.")
    parser.add_argument("command", choices=("next", "price", "source", "run", "assess", "record", "render", "check", "coverage"))
    args = parser.parse_args(argv)
    if args.command == "render":
        return cmd_render(args)
    if args.command == "check":
        return cmd_check(args)
    if args.command == "coverage":
        return cmd_coverage(args)
    parser.error("command requires its own arguments")


if __name__ == "__main__":
    raise SystemExit(main())
