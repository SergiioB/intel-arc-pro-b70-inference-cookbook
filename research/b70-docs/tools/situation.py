#!/usr/bin/env python3
"""Compile a ≤12-file agent working set from B70 registries.

Reads JSON ledgers, not Markdown. Never overwrites canonical registries.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

INTENTS = ("ops", "bench", "diagnose", "research", "publish", "site")
CLOSE_OUTCOMES = ("blocked", "validated", "failed", "provisional")
MAX_FILES = 12

WORKING_SETS: dict[str, list[tuple[str, str]]] = {
    "ops": [
        ("kernel", "AGENTS.md"),
        ("situation", "docs/operations/current-host-state.md"),
        ("method", "docs/operations/storage-and-vram-safety.md"),
    ],
    "bench": [
        ("situation", "registry/research-ledger.json"),
        ("situation", "registry/model-status.json"),
        ("method", "docs/methods/benchmark-protocol.md"),
        ("kernel", "AGENTS.md"),
        ("skill", ".agentic/skills/benchmark-model/SKILL.md"),
    ],
    "diagnose": [
        ("method", "docs/methods/architecture-diagnostic-playbook.md"),
        ("kernel", "AGENTS.md"),
        ("situation", "registry/research-ledger.json"),
    ],
    "research": [
        ("situation", "registry/research-ledger.json"),
        ("situation", "registry/model-status.json"),
        ("situation", "registry/research-program.json"),
        ("orientation", "research/README.md"),
        ("kernel", "DOCUMENT-MAP.md"),
        ("system", "docs/governance/agent-operating-system.md"),
    ],
    "publish": [
        ("method", "docs/governance/public-benchmark-publication.md"),
        ("skill", ".agents/skills/b70-benchmark-claims-standard/SKILL.md"),
        ("skill", ".agents/skills/publish-b70-benchmark-results/SKILL.md"),
    ],
    "site": [
        ("kernel", "AGENTS.md"),
        ("system", "DESIGN-AGENT-SYSTEM.md"),
        ("situation", "data/cookbook-b70/PROVENANCE.md"),
    ],
}

SKILL_FOR = {
    "bench": ".agentic/skills/benchmark-model/SKILL.md",
    "publish": ".agents/skills/publish-b70-benchmark-results/SKILL.md",
}


def load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def norm_alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def pattern_hits(pattern: str, goal: str) -> bool:
    """Match guards without dismantling grouped alternatives or lookarounds."""
    compiled = re.compile(pattern, re.I)
    goal_norm = re.sub(r"[-_]", ".", goal)
    if compiled.search(goal) or compiled.search(goal_norm):
        return True
    # Legacy flat patterns allow the same terms in a different order.
    if not any(char in pattern for char in "()[]\\"):
        for alternative in pattern.split("|"):
            parts = alternative.split(".*")
            if len(parts) > 1 and all(re.search(part, goal_norm, re.I) for part in parts):
                return True
    return False


def family_matches(family: str, goal: str, explicit: str | None) -> bool:
    token = explicit or goal
    nfam = norm_alnum(family)
    ntok = norm_alnum(token)
    if not ntok:
        return False
    if explicit:
        return ntok in nfam or nfam.startswith(ntok) or nfam in ntok
    if len(ntok) >= 6 and (ntok in nfam or nfam in ntok):
        return True
    # compact tokens like qwen38
    for piece in re.findall(r"[a-z]+[0-9]+", token.lower().replace(".", "").replace("-", "")):
        if len(piece) >= 5 and piece in nfam:
            return True
    return False


def ledger_entries(root: Path) -> list[dict[str, Any]]:
    data = load_json(root / "registry" / "research-ledger.json")
    if not isinstance(data, dict):
        return []
    entries = data.get("entries") or []
    return [e for e in entries if isinstance(e, dict)]


def model_rows(root: Path) -> list[dict[str, Any]]:
    data = load_json(root / "registry" / "model-status.json")
    if not isinstance(data, dict):
        return []
    rows = data.get("models") or []
    return [r for r in rows if isinstance(r, dict)]


def campaign_rows(root: Path) -> list[dict[str, Any]]:
    data = load_json(root / "registry" / "campaigns.yaml")
    rows = data.get("campaigns", []) if isinstance(data, dict) else data
    return [row for row in (rows or []) if isinstance(row, dict)]


def local_path(root: Path, value: str) -> Path | None:
    """Reject external paths, including symlinks outside the evidence tree."""
    path = Path(value)
    if path.is_absolute():
        return None
    resolved = (root / path).resolve()
    return resolved if resolved.is_relative_to(root.resolve()) else None


def evidence_paths(root: Path, entry: dict[str, Any]) -> list[str]:
    """Extract only named, existing repository paths; never expand shorthand."""
    candidates = list(entry.get("evidence_paths") or [])
    evidence = str(entry.get("evidence") or "")
    candidates.extend(re.findall(
        r"(?<![\w./-])(?:results|research|decisions|campaigns|docs|models|tools|scripts)/[^\s`\"<>+;,(){}]+",
        evidence,
    ))
    paths: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        candidate = candidate.rstrip(".:/")
        path = local_path(root, candidate)
        if path is not None and path.exists() and candidate not in paths:
            paths.append(candidate)
    return paths


def entry_matches(entry: dict[str, Any], goal: str, family: str | None) -> bool:
    if str(entry.get("id", "")).upper() in set(re.findall(r"\bRL-\d+\b", goal.upper())):
        return True
    subject = " ".join(str(entry.get(key) or "") for key in ("question", "path_tried", "family", "model"))
    if family and family_matches(family, subject, None):
        return True
    if family_matches(subject, goal, None):
        return True
    ignored = {"research", "review", "audit", "compile", "benchmark", "serving", "decode",
               "prefill", "model", "models", "current", "existing", "evidence", "validated",
               "blocked", "failed", "provisional", "stock", "launch", "without", "question"}
    tokens = set(re.findall(r"[a-z0-9_]+", goal.lower())) - ignored
    subject_tokens = set(re.findall(r"[a-z0-9_]+", subject.lower()))
    return any(len(token) >= 5 and token in subject_tokens for token in tokens)


def matching_blocked(
    entries: list[dict[str, Any]], goal: str, family: str | None, warnings: list[str],
) -> list[dict[str, Any]]:
    hits = []
    for entry in entries:
        pattern = entry.get("blocked_pattern")
        if not pattern or entry.get("superseded_by") or entry.get("outcome") == "superseded":
            continue
        relevant = entry_matches(entry, goal, family)
        if not isinstance(pattern, str):
            warnings.append(f"{entry.get('id')}: invalid blocked_pattern (expected string); manual guard review required")
            if relevant:
                hits.append(entry)
            continue
        # Historical rows also store prose here. Preserve it as a manual guard,
        # not a regular expression that accidentally gives a scientific verdict.
        descriptive = len(pattern.split()) >= 6 and not any(marker in pattern for marker in (".*", ".?", "(?"))
        try:
            matched = pattern_hits(pattern, goal)
        except re.error as exc:
            warnings.append(f"{entry.get('id')}: invalid blocked_pattern: {exc}; manual guard review required")
            matched = relevant
        if descriptive:
            if relevant:
                warnings.append(f"{entry.get('id')}: descriptive adoption guard requires manual review")
            matched = relevant
        if matched:
            hits.append(entry)
    return hits


def resolve_tree(intent: str, root: Path, site_root: Path | None, cookbook_root: Path | None) -> Path:
    if intent == "site":
        if site_root is None:
            raise SystemExit("site intent requires --site-root")
        return site_root
    return root


def add_item(items: list[dict[str, str]], seen: set[str], role: str, path: str, tree: str) -> None:
    key = f"{tree}:{path}"
    if key in seen:
        return
    seen.add(key)
    items.append({"role": role, "path": path, "tree": tree})


def exists_in(tree_root: Path, rel: str) -> bool:
    return (tree_root / rel).exists()


def compile_situation(args: argparse.Namespace) -> dict[str, Any]:
    root: Path = args.root
    if not (root / "AGENTS.md").is_file() and args.intent != "site":
        print("ERROR: AGENTS.md missing in --root", file=sys.stderr)
        raise SystemExit(2)

    private_context = args.intent in {"bench", "diagnose", "research"}
    entries = ledger_entries(root) if private_context else []
    warnings: list[str] = []
    blocked_hits = matching_blocked(entries, args.goal, args.family, warnings)
    changed_variable = (args.changed_variable or "").strip() or None
    halt = bool(blocked_hits) and args.intent == "bench" and not changed_variable

    rows = model_rows(root) if private_context else []
    route = None
    for row in rows:
        family = str(row.get("family") or "")
        if family and family_matches(family, args.goal, args.family):
            route = row
            break

    matched = [entry for entry in entries if entry in blocked_hits or entry_matches(
        entry, args.goal, args.family or (route or {}).get("family"),
    )]
    # Guards and failures first (they prevent repeat work), then newest. Cap the
    # JSON view: the registry stays complete, the compile output stays readable.
    guard_outcomes = {"blocked", "failed"}
    position = {id(entry): index for index, entry in enumerate(entries)}
    ordered = sorted(
        matched,
        key=lambda entry: (entry.get("outcome") not in guard_outcomes, -position[id(entry)]),
    )
    prior_limit = 15
    prior_evidence = [
        {**entry, "evidence_paths": evidence_paths(root, entry)} for entry in ordered[:prior_limit]
    ]
    if len(ordered) > prior_limit:
        warnings.append(
            f"{len(ordered) - prior_limit} further matching ledger rows not shown; "
            "registry/research-ledger.json is the full record."
        )
    if prior_evidence:
        warnings.append("Prior evidence is scoped history, not an automatic replication exemption or an adopted route.")
    if changed_variable:
        warnings.append("Changed-variable justification records a proposed difference; it does not authorize execution or waive preflight, quality, or adoption gates.")

    do_not = [f"{e.get('id')} {e.get('blocked_pattern')}" for e in blocked_hits]
    if halt:
        do_not.append("do not launch this path; halt=true (document --changed-variable before proposing a new bench probe)")

    items: list[dict[str, str]] = []
    seen: set[str] = set()

    tree_name = "xmxlab" if args.intent == "site" else "b70-docs"
    resolve_tree(args.intent, root, args.site_root, args.cookbook_root)
    for role, rel in WORKING_SETS[args.intent]:
        add_item(items, seen, role, rel, tree_name)
    if args.intent == "publish" and args.cookbook_root is not None:
        add_item(items, seen, "catalog", "data/benchmarks.v1.json", "cookbook")
    if route and route.get("dossier"):
        add_item(items, seen, "route", str(route["dossier"]), "b70-docs")
    # Attach guard sources first. All references remain in prior_evidence even
    # when the <=12-file reading budget leaves some sources outside this set.
    ordered = sorted(prior_evidence, key=lambda entry: entry["id"] not in {hit["id"] for hit in blocked_hits})
    for entry in ordered:
        for evidence in entry["evidence_paths"]:
            add_item(items, seen, "decision", evidence, "b70-docs")

    # existence: drop missing except kernel AGENTS.md in the active tree
    kept: list[dict[str, str]] = []
    tree_roots = {
        "b70-docs": root,
        "xmxlab": args.site_root,
        "cookbook": args.cookbook_root,
    }
    for item in items:
        tree_root = tree_roots.get(item["tree"])
        if tree_root is None:
            warnings.append(f"drop {item['path']} (no {item['tree']} root)")
            continue
        if exists_in(tree_root, item["path"]):
            kept.append(item)
            continue
        if item["path"] == "AGENTS.md" and item["tree"] in {"b70-docs", "xmxlab"}:
            print(f"ERROR: AGENTS.md missing in {item['tree']}", file=sys.stderr)
            raise SystemExit(2)
        warnings.append(f"drop missing {item['tree']}:{item['path']}")
    items = kept

    if args.evidence_root and args.intent == "publish":
        path = local_path(root, args.evidence_root)
        if path is None or not path.exists():
            raise SystemExit("ERROR: --evidence-root must name existing evidence inside --root")
        add_item(items, seen, "evidence", args.evidence_root, "b70-docs")
    if len(items) > MAX_FILES:
        warnings.append(f"Working set limited to {MAX_FILES} files; remaining sources are listed in prior_evidence.")
        items = items[:MAX_FILES]


    payload = {
        "intent": args.intent,
        "goal": args.goal,
        "halt": halt,
        "blocked": [str(e.get("id")) for e in blocked_hits],
        "prior_evidence": prior_evidence,
        "changed_variable": changed_variable,
        "route": route,
        "working_set": items,
        "skill": SKILL_FOR.get(args.intent),
        "do_not": do_not,
        "budget_files": len(items),
        "warnings": warnings,
    }
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Situation compile",
        "",
        "Generated view — not numeric authority. Do not copy tok/s from this file.",
        "",
        f"intent: {payload['intent']}",
        f"goal: {payload['goal']}",
        f"halt: {str(payload['halt']).lower()}",
        f"changed_variable: {payload.get('changed_variable') or '—'}",
        f"skill: {payload.get('skill') or '—'}",
        f"budget_files: {payload['budget_files']}",
        "",
        "blocked:",
    ]
    blocked = payload.get("blocked") or []
    if blocked:
        lines.extend(f"- {item}" for item in blocked)
    else:
        lines.append("- (none)")
    lines += ["", "prior_evidence (history, not a replication exemption):"]
    prior = payload.get("prior_evidence") or []
    if prior:
        for entry in prior:
            lines.append(f"- {entry.get('id')} [{entry.get('outcome')}]: {entry.get('question')}")
            for path in entry["evidence_paths"]:
                lines.append(f"  evidence: {path}")
    else:
        lines.append("- (none)")
    lines += ["", "working_set:"]
    for item in payload.get("working_set") or []:
        lines.append(f"- role: {item['role']}")
        lines.append(f"  tree: {item['tree']}")
        lines.append(f"  path: {item['path']}")
    lines += ["", "do_not:"]
    do_not = payload.get("do_not") or []
    if do_not:
        lines.extend(f"- {item}" for item in do_not)
    else:
        lines.append("- (none)")
    warnings = payload.get("warnings") or []
    if warnings:
        lines += ["", "warnings:"]
        lines.extend(f"- {item}" for item in warnings)
    lines.append("")
    return "\n".join(lines)


def cmd_compile(args: argparse.Namespace) -> int:
    payload = compile_situation(args)
    json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    if not args.no_write:
        out = args.root / "docs" / "indexes" / "situation-latest.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(payload), encoding="utf-8")
    return 0


def cmd_close(args: argparse.Namespace) -> int:
    root: Path = args.root
    missing: list[str] = []
    entries = ledger_entries(root)
    entry = next((row for row in entries if row.get("id") == args.ledger_id), None)
    campaigns = campaign_rows(root)
    campaign = next((row for row in campaigns if row.get("id") == args.campaign), None)
    if campaign is None and args.campaign.startswith("results/"):
        campaign = next((row for row in campaigns if str(row.get("evidence_root", "")).rstrip("/") == args.campaign.rstrip("/")), None)
    result_root: Path | None = None
    record_path: Path | None = None
    if campaign is not None:
        record_path = local_path(root, str(campaign.get("record_path") or ""))
        if record_path is None or not record_path.is_file():
            missing.append("registered campaign must name an existing local record_path")
        if campaign.get("evidence_root"):
            result_root = local_path(root, str(campaign["evidence_root"]))
            if result_root is None or not result_root.is_relative_to(root / "results"):
                missing.append("registered evidence_root must be inside results/")
                result_root = None
    else:
        result_root = local_path(root, args.campaign)
        if (not re.fullmatch(r"results/\d{4}/\d{2}/[^/]+/?", args.campaign)
                or result_root is None or not result_root.is_dir()):
            missing.append("--campaign must be a registered campaign ID or an existing dated results/YYYY/MM/campaign root")
            result_root = None
    if args.campaign.startswith("results/") and (result_root is None or not result_root.is_dir()):
        missing.append("requested dated result root does not exist")

    if entry is None:
        missing.append(f"ledger id {args.ledger_id} not in registry/research-ledger.json")
    else:
        if entry.get("outcome") != args.outcome:
            missing.append(f"ledger {args.ledger_id} outcome={entry.get('outcome')} does not match --outcome {args.outcome}")
        if entry.get("superseded_by") or entry.get("outcome") == "superseded":
            missing.append(f"ledger {args.ledger_id} is superseded; close against current evidence")
        if entry.get("campaign_id") and entry["campaign_id"] != args.campaign:
            if campaign is None or entry["campaign_id"] != campaign.get("id"):
                missing.append("ledger campaign_id does not match requested campaign")
        if campaign and campaign.get("ledger_ids") and args.ledger_id not in campaign["ledger_ids"]:
            missing.append("campaign ledger_ids does not include the named ledger row")
        paths = [root / path for path in evidence_paths(root, entry)]
        linked = any(
            (result_root is not None and result_root.is_dir() and path.resolve().is_relative_to(result_root))
            or (record_path is not None and record_path.is_file() and path.resolve() == record_path)
            for path in paths
        )
        if not linked:
            missing.append("missing proof: named ledger evidence must exist under the campaign result root or cite its registered record_path; unrelated rows and campaign IDs alone are insufficient")

    if args.route_changed and not args.status_updated:
        missing.append("pass --status-updated after editing registry/model-status.json")

    if missing:
        print("close incomplete:", file=sys.stderr)
        for item in missing:
            print(f"- {item}", file=sys.stderr)
        return 2

    registry = root / "tools" / "research.py"
    if registry.is_file():
        proc = subprocess.run(
            [sys.executable, str(registry), "check"],
            cwd=str(root),
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            print("close incomplete: tools/research.py check failed", file=sys.stderr)
            for output in (proc.stdout, proc.stderr):
                if output.strip():
                    print(output.strip(), file=sys.stderr)
            return 2

    print(f"close ok campaign={args.campaign} outcome={args.outcome} ledger={args.ledger_id}")
    return 0


def add_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="B70-DOCS root (default: parent of tools/)",
    )


def changed_variable(value: str) -> str:
    justification = value.strip()
    if (not any(character.isalnum() for character in justification)
            or justification.lower() in {"none", "null", "n/a", "na", "false", "true", "yes", "no", "force"}):
        raise argparse.ArgumentTypeError("--changed-variable requires a nonempty explanation of the changed variable")
    return justification


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    compile_p = sub.add_parser("compile", help="emit ≤12-file working set as JSON")
    add_root(compile_p)
    compile_p.add_argument("--intent", required=True, choices=INTENTS)
    compile_p.add_argument("--goal", required=True)
    compile_p.add_argument("--family", default=None)
    compile_p.add_argument(
        "--changed-variable", type=changed_variable, default=None,
        help="explain the changed mechanism for a blocked bench proposal; does not authorize execution",
    )
    compile_p.add_argument("--no-write", action="store_true")
    compile_p.add_argument("--site-root", type=Path, default=None)
    compile_p.add_argument("--cookbook-root", type=Path, default=None)
    compile_p.add_argument("--evidence-root", default=None)

    close_p = sub.add_parser("close", help="verify named ledger outcome and campaign evidence linkage")
    add_root(close_p)
    close_p.add_argument("--campaign", required=True, help="registered campaign ID or dated results/YYYY/MM/campaign root")
    close_p.add_argument("--outcome", required=True, choices=CLOSE_OUTCOMES)
    close_p.add_argument("--ledger-id", required=True, help="exact ledger row proving this campaign outcome")
    close_p.add_argument("--route-changed", action="store_true")
    close_p.add_argument("--status-updated", action="store_true")

    args = parser.parse_args(argv)
    args.root = (args.root or Path(__file__).resolve().parents[1]).resolve()
    if getattr(args, "site_root", None) is None:
        guessed = Path.home() / "projects" / "xmxlab"
        args.site_root = guessed if guessed.is_dir() else None
    elif args.site_root is not None:
        args.site_root = args.site_root.resolve()
    if getattr(args, "cookbook_root", None) is None:
        guessed = Path.home() / "projects" / "intel-arc-pro-b70-inference-cookbook"
        args.cookbook_root = guessed if guessed.is_dir() else None
    elif args.cookbook_root is not None:
        args.cookbook_root = args.cookbook_root.resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "compile":
        return cmd_compile(args)
    if args.command == "close":
        return cmd_close(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
