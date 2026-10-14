#!/usr/bin/env python3
"""Run, assess, and register bounded, evidence-backed research experiments."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import uuid

from research_queue import attempt_limit as limit_with_operator_extras

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ("quality", "correctness", "isolation", "measurement_validity")
IDENTITIES = ("host", "engine_revision", "model_revision")


class ExperimentError(ValueError):
    """An experiment cannot safely proceed with the supplied evidence."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_text(value) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def read_json(path: Path, *, content: bytes | None = None):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ExperimentError(f"{path}: duplicate JSON key {key}")
            result[key] = value
        return result

    def bad_constant(value):
        raise ExperimentError(f"{path}: nonfinite JSON value {value}")

    try:
        return json.loads(path.read_text(encoding="utf-8") if content is None else content, object_pairs_hook=unique,
                          parse_constant=bad_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"Cannot read JSON {path}: {exc}") from exc


def write_new(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json_text(value))
        stream.flush()
        os.fsync(stream.fileno())


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_text(value, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentError(f"{label} must be a nonempty string")


def require_number(value, label: str, *, positive: bool = True) -> None:
    try:
        valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(value) and (value > 0 if positive else value >= 0))
    except OverflowError:
        valid = False
    if not valid:
        raise ExperimentError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")


def validate_argv(value, label: str) -> None:
    if not isinstance(value, list) or not value:
        raise ExperimentError(f"{label} must be a nonempty argv array; shell strings are not accepted")
    for item in value:
        if not isinstance(item, str) or "\x00" in item:
            raise ExperimentError(f"{label} arguments must be strings without NUL bytes")
    require_text(value[0], label)


def validate_plan(plan: dict) -> dict:
    if not isinstance(plan, dict):
        raise ExperimentError("Plan must be a JSON object")
    for key in ("question_id", "hypothesis", "changed_variable", "evidence_rationale"):
        require_text(plan.get(key), key)
    if not re.fullmatch(r"RQ-\d+", plan["question_id"]):
        raise ExperimentError("question_id must match RQ-N")
    if plan.get("resource") not in ("cpu", "gpu"):
        raise ExperimentError("resource must be cpu or gpu")
    if plan.get("stage") not in ("screen", "confirm"):
        raise ExperimentError("stage must be screen or confirm")
    require_number(plan.get("timeout_s"), "timeout_s")
    validate_argv(plan.get("command"), "command")
    for key in ("preflight_command", "cleanup_command"):
        if key in plan or plan["resource"] == "gpu":
            validate_argv(plan.get(key), key)
    identities = plan.get("identities")
    if not isinstance(identities, dict) or set(identities) != {"baseline", "candidate"}:
        raise ExperimentError("identities must name baseline and candidate")
    for arm, identity in identities.items():
        if not isinstance(identity, dict) or set(identity) != set(IDENTITIES):
            raise ExperimentError(f"identities.{arm} must contain host, engine_revision, model_revision")
        for key in IDENTITIES:
            require_text(identity[key], f"identities.{arm}.{key}")
    comparison = plan.get("comparison")
    if not isinstance(comparison, dict):
        raise ExperimentError("comparison must be an object")
    for key in ("metric", "unit"):
        require_text(comparison.get(key), f"comparison.{key}")
    if comparison.get("direction") not in ("higher", "lower"):
        raise ExperimentError("comparison.direction must be higher or lower")
    require_number(comparison.get("min_relative_gain"), "comparison.min_relative_gain", positive=False)
    controlled = plan.get("controlled")
    if not isinstance(controlled, dict) or set(controlled) != {"workload", "config"}:
        raise ExperimentError("controlled must contain workload and config objects")
    for key in ("workload", "config"):
        if not isinstance(controlled[key], dict) or not controlled[key]:
            raise ExperimentError(f"controlled.{key} must declare nonempty matched coordinates")
    # Validate nested coordinates, including data supplied directly through the Python API.
    try:
        json_text(plan)
    except (ValueError, TypeError) as exc:
        raise ExperimentError(f"Plan must contain finite JSON values: {exc}") from exc
    return plan


@contextmanager
def writer_lock(root: Path):
    runtime = root / ".runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / "experiment.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ExperimentError("Another local experiment writer is active; this lock does not coordinate hosts") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def check_question(root: Path, plan: dict) -> int:
    program = read_json(root / "registry/research-program.json")
    if not isinstance(program, dict) or not isinstance(program.get("questions"), list):
        raise ExperimentError("Research program must contain a questions array")
    questions = {}
    for row in program["questions"]:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] in questions:
            raise ExperimentError("Research program contains malformed or duplicate questions")
        questions[row["id"]] = row
    question = questions.get(plan["question_id"])
    if not question or question.get("status") != "ready":
        raise ExperimentError("Question must exist in the research program with status ready")
    if program.get("operator_hold") or question.get("operator_hold"):
        raise ExperimentError("Question is on operator hold")
    portfolio = program.get("portfolio", {})
    budget = question.get("execution_budget", {})
    limit = budget.get(f"{plan['stage']}_timeout_s")
    if limit is not None and plan["timeout_s"] > limit:
        raise ExperimentError("Plan timeout exceeds the question's precommitted execution budget")
    if portfolio and plan["comparison"]["min_relative_gain"] < question.get("minimum_relative_gain", 0.10):
        raise ExperimentError("Plan gain gate is below the program's minimum useful gain")
    for key in ("hypothesis", "changed_variable"):
        if question.get(key) != plan[key]:
            raise ExperimentError(f"Plan {key} must match the research program; amend the question explicitly")
    dependencies = question.get("dependencies")
    if not isinstance(dependencies, list) or not all(isinstance(item, str) for item in dependencies):
        raise ExperimentError("Question dependencies must be an array of question IDs")
    for dependency in dependencies:
        row = questions.get(dependency, {})
        if row.get("status") != "closed" or row.get("resolution") != "passed":
            raise ExperimentError(f"Dependency {dependency} must be closed with resolution passed")
    attempt_limit = portfolio.get("max_attempts_per_mechanism", 2)
    if not isinstance(attempt_limit, int) or isinstance(attempt_limit, bool) or attempt_limit < 1:
        raise ExperimentError("portfolio.max_attempts_per_mechanism must be a positive integer")
    return limit_with_operator_extras(program, question)


def attempt_count(root: Path, plan: dict) -> int:
    count = 0
    for path in sorted((root / "results").glob("*/*/*/manifest.json")):
        manifest = read_json(path)
        if not isinstance(manifest, dict):
            raise ExperimentError(f"Malformed run manifest: {path}")
        if all(manifest.get(key) == plan.get(key) for key in ("question_id", "hypothesis", "changed_variable")):
            count += 1
    return count


def event(run: Path, kind: str, **values) -> None:
    with (run / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": utc_now(), "event": kind, **values}, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def terminate_group(process: subprocess.Popen) -> None:
    """Kill only the session/process group this runner created, then reap its leader."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    # A leader may exit before its children; still give the entire owned group a bound.
    deadline = time.monotonic() + 0.25
    while time.monotonic() < deadline:
        process.poll()
        time.sleep(0.01)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def execute(root: Path, run: Path, label: str, argv: list[str], timeout: float) -> dict:
    event(run, "start", phase=label, argv=argv)
    result = {"argv": argv, "returncode": None, "timed_out": False, "error": None}
    env = dict(os.environ, B70_RUN_DIR=str(run))
    with (run / f"{label}.stdout").open("xb") as stdout, (run / f"{label}.stderr").open("xb") as stderr:
        process = None
        try:
            process = subprocess.Popen(argv, cwd=root, env=env, stdout=stdout, stderr=stderr,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                result["returncode"] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                result["timed_out"] = True
            finally:
                # Also remove background descendants after a nominally successful parent exit.
                terminate_group(process)
                result["returncode"] = process.returncode
        except InterruptedError as exc:
            result["error"] = f"InterruptedError: {exc}"
            raise
        except OSError as exc:
            result["error"] = str(exc)
            stderr.write((str(exc) + "\n").encode())
        except BaseException as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            event(run, "exit", phase=label, **result)
    return result


def successful(result: dict) -> bool:
    return (isinstance(result, dict) and type(result.get("returncode")) is int
            and result["returncode"] == 0 and result.get("timed_out") is False
            and result.get("error") is None)


def command_provenance(root: Path, plan: dict) -> dict:
    commands = {}
    for label in ("command", "preflight_command", "cleanup_command"):
        if label not in plan:
            continue
        files = {}
        for index, argument in enumerate(plan[label]):
            if not argument or argument.startswith("-"):
                continue
            candidate = Path(argument)
            if not candidate.is_absolute():
                candidate = root / candidate
            try:
                if index == 0 and not candidate.is_file():
                    executable = shutil.which(argument)
                    if executable:
                        candidate = Path(executable)
                if candidate.is_file():
                    files[str(candidate.resolve())] = sha256(candidate)
            except OSError:
                # Inline programs and ordinary argument strings are not filesystem paths.
                if index == 0:
                    raise
        commands[label] = {"argv": plan[label], "files_sha256": files}
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                  text=True, timeout=5, check=False)
        git_revision = revision.stdout.strip() if revision.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        git_revision = None
    return {"python": sys.version, "platform": platform.platform(), "hostname": socket.gethostname(),
            "repository_revision": git_revision, "commands": commands,
            "runner_sha256": sha256(Path(__file__)),
            "environment": {key: os.environ[key] for key in ("LANG", "LC_ALL", "TZ", "OMP_NUM_THREADS") if key in os.environ},
            "environment_scope": "Allowlisted execution settings only; credentials are not captured"}


def run_plan(root: Path, plan_path: Path) -> dict:
    root = Path(root).resolve()
    plan_path = Path(plan_path).resolve()
    original = plan_path.read_bytes()
    plan = validate_plan(read_json(plan_path, content=original))
    with writer_lock(root):
        registered = {row.get("run_root") for row in read_json(root / "registry/research-ledger.json")["entries"]}
        for previous in (root / "results").glob("*/*/*/manifest.json"):
            entry = read_json(previous)
            if entry.get("question_id") == plan["question_id"] and entry.get("run_root") not in registered:
                raise ExperimentError("Unregistered prior run: assess and record it before starting another")
        attempt_limit = check_question(root, plan)
        attempts = attempt_count(root, plan)
        if attempts >= attempt_limit:
            raise ExperimentError(f"Attempt limit reached: at most {attempt_limit} run(s) for unchanged question, hypothesis, and variable")
        now = datetime.now(timezone.utc)
        run = root / "results" / now.strftime("%Y/%m") / (plan["question_id"] + "-" + now.strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:10])
        run.mkdir(parents=True, exist_ok=False)
        with (run / "plan.json").open("xb") as stream:
            stream.write(original)
            stream.flush()
            os.fsync(stream.fileno())
        manifest = {"schema_version": 1, "created_at": utc_now(), "run_root": run.relative_to(root).as_posix(),
                    **{key: plan[key] for key in ("question_id", "hypothesis", "changed_variable")},
                    "attempt": attempts + 1, "input_plan_sha256": hashlib.sha256(original).hexdigest(),
                    "lock_scope": "same-host only; one cross-host writer must be enforced by the operator"}
        write_new(run / "manifest.json", manifest)
        write_new(run / "environment.json", command_provenance(root, plan))
        frozen = {name: sha256(run / name) for name in ("plan.json", "manifest.json", "environment.json")}
        phases = {}
        interrupted = None
        previous_handler = signal.getsignal(signal.SIGTERM)

        def interrupt(signum, _frame):
            raise InterruptedError(f"Received signal {signum}")

        signal.signal(signal.SIGTERM, interrupt)
        try:
            if "preflight_command" in plan:
                phases["preflight"] = execute(root, run, "preflight", plan["preflight_command"], plan["timeout_s"])
            if "preflight" not in phases or successful(phases["preflight"]):
                phases["command"] = execute(root, run, "command", plan["command"], plan["timeout_s"])
        except (KeyboardInterrupt, InterruptedError) as exc:
            interrupted = f"{type(exc).__name__}: {exc}"
            event(run, "interrupted", error=interrupted)
        finally:
            try:
                if "cleanup_command" in plan:
                    phases["cleanup"] = execute(root, run, "cleanup", plan["cleanup_command"], plan["timeout_s"])
            finally:
                signal.signal(signal.SIGTERM, previous_handler)
        if interrupted:
            raise ExperimentError(f"Run interrupted; raw evidence retained without a seal: {run}")
        integrity_errors = []
        for name, digest in frozen.items():
            if not (run / name).is_file() or (run / name).is_symlink() or sha256(run / name) != digest:
                integrity_errors.append(f"Precommitted file changed: {name}")
        files = {}
        for path in sorted(run.rglob("*")):
            if path.is_symlink():
                integrity_errors.append(f"Symlink is not sealable evidence: {path.relative_to(run)}")
            elif path.is_file():
                files[path.relative_to(run).as_posix()] = sha256(path)
        complete = {"schema_version": 1, "finished_at": utc_now(), "phases": phases,
                    "execution_passed": "command" in phases and all(successful(value) for value in phases.values()) and not integrity_errors,
                    "integrity_errors": integrity_errors, "files_sha256": files, "precommit_sha256": frozen}
        write_new(run / "complete.json", complete)
        return {"run": str(run), "run_root": manifest["run_root"],
                "execution_passed": complete["execution_passed"], "sealed": True,
                "adoption": "not_adopted", "next_action": f"assess --run {manifest['run_root']}"}


def resolve_run(root: Path, run_path: Path) -> Path:
    path = Path(run_path)
    run = (path if path.is_absolute() else root / path).resolve()
    try:
        relative = run.relative_to((root / "results").resolve())
    except ValueError as exc:
        raise ExperimentError("Run must be inside this repository's results directory") from exc
    if len(relative.parts) != 3 or not run.is_dir():
        raise ExperimentError("Run must be a results/YYYY/MM/run directory")
    return run


def evidence_file(run: Path, value: str) -> Path:
    require_text(value, "evidence path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ExperimentError(f"Evidence must use a relative path inside the run: {value}")
    if "\x00" in value:
        raise ExperimentError("Evidence paths cannot contain NUL bytes")
    path = run / relative
    try:
        path.resolve().relative_to(run)
    except ValueError as exc:
        raise ExperimentError(f"Evidence escapes run directory: {value}") from exc
    if any(parent.is_symlink() for parent in (path, *path.parents) if parent != run and run in parent.parents):
        raise ExperimentError(f"Evidence cannot use symlinks: {value}")
    if not path.is_file() or path.stat().st_size == 0:
        raise ExperimentError(f"Evidence must be a nonempty regular file: {value}")
    return path


def verify_seal(run: Path) -> dict:
    complete = read_json(run / "complete.json")
    if not isinstance(complete, dict) or not isinstance(complete.get("files_sha256"), dict):
        raise ExperimentError("Run seal is malformed")
    files = complete["files_sha256"]
    if not {"plan.json", "manifest.json", "environment.json", "events.jsonl"}.issubset(files):
        raise ExperimentError("Run seal lacks required execution provenance")
    for name, digest in files.items():
        # stdout and stderr can legitimately be empty, unlike claimed check evidence.
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ExperimentError("Run seal contains an unsafe path")
        path = run / name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(run) or sha256(path) != digest:
            raise ExperimentError(f"Sealed artifact changed or disappeared: {name}")
    if type(complete.get("execution_passed")) is not bool or not isinstance(complete.get("phases"), dict):
        raise ExperimentError("Run seal lacks execution status")
    if not all(isinstance(value, dict) for value in complete["phases"].values()):
        raise ExperimentError("Run seal contains malformed phase results")
    frozen = complete.get("precommit_sha256")
    if not isinstance(frozen, dict) or set(frozen) != {"plan.json", "manifest.json", "environment.json"}:
        raise ExperimentError("Run seal lacks precommit hashes")
    if any(files.get(name) != digest for name, digest in frozen.items()):
        raise ExperimentError("Run precommit files changed during execution")
    return complete


def assess_measurements(run: Path, plan: dict, sealed: dict) -> dict:
    measurements_path = evidence_file(run, "measurements.json")
    if "measurements.json" not in sealed:
        raise ExperimentError("measurements.json was not present when execution was sealed")
    data = read_json(measurements_path)
    if not isinstance(data, dict):
        raise ExperimentError("measurements.json must be an object")
    comparison = plan["comparison"]
    for key in ("metric", "unit"):
        if data.get(key) != comparison[key]:
            raise ExperimentError(f"Measurement {key} does not match the frozen comparison")
    for key in ("identities", "controlled"):
        if data.get(key) != plan[key]:
            raise ExperimentError(f"Measurement {key} does not match frozen comparison coordinates")
    arms = {}
    for arm in ("baseline", "candidate"):
        rows = data.get(arm)
        if not isinstance(rows, list):
            raise ExperimentError(f"{arm} measurements must be an array")
        values = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ExperimentError(f"{arm} observations must be objects")
            require_text(row.get("pair"), f"{arm}.pair")
            pair = row["pair"]
            if pair in values:
                raise ExperimentError(f"Duplicate pair ID in {arm}: {pair}")
            require_number(row.get("value"), f"{arm}.{pair}.value")
            values[pair] = row["value"]
        arms[arm] = values
    if arms["baseline"].keys() != arms["candidate"].keys():
        raise ExperimentError("Baseline and candidate pair IDs must match exactly")
    minimum = 3 if plan["stage"] == "screen" else 5
    if len(arms["baseline"]) < minimum:
        raise ExperimentError(f"{plan['stage']} requires at least {minimum} matched pairs")
    checks = data.get("checks")
    if not isinstance(checks, dict):
        raise ExperimentError("Measurements must contain explicit evidence-backed checks")
    evidence_hashes = {"measurements.json": sealed["measurements.json"]}
    failed_checks = []
    for key in CHECKS:
        check = checks.get(key)
        if not isinstance(check, dict) or type(check.get("passed")) is not bool:
            raise ExperimentError(f"{key} must declare passed as a boolean and cite evidence")
        path = evidence_file(run, check.get("evidence"))
        name = path.relative_to(run).as_posix()
        digest = sha256(path)
        if sealed.get(name) != digest:
            raise ExperimentError(f"{key} evidence was not sealed: {name}")
        evidence_hashes[name] = digest
        if not check["passed"]:
            failed_checks.append(key)
    gains = []
    for pair, baseline in arms["baseline"].items():
        candidate = arms["candidate"][pair]
        gain = ((candidate - baseline) / baseline if comparison["direction"] == "higher"
                else (baseline - candidate) / baseline)
        if not math.isfinite(gain):
            raise ExperimentError(f"Paired gain overflowed for {pair}")
        gains.append({"pair": pair, "relative_gain": gain})
    values = [row["relative_gain"] for row in gains]
    median = statistics.median(values)
    gate = comparison["min_relative_gain"]
    if failed_checks:
        decision, reason = "rejected", "Evidence checks failed: " + ", ".join(failed_checks)
    elif median < gate:
        decision, reason = "rejected", "Median paired gain is below the precommitted gate"
    elif min(values) < gate:
        decision, reason = "inconclusive", "Paired gains cross the gate; the conservative gate is not met"
    else:
        decision, reason = "eligible", "Every paired gain and every evidence-backed check meets the precommitted gate"
    return {"decision": decision, "reason": reason, "paired_gains": gains,
            "median_relative_gain": median, "minimum_relative_gain": min(values),
            "required_relative_gain": gate, "evidence_sha256": evidence_hashes}


def assess_locked(root: Path, run: Path) -> dict:
    complete = verify_seal(run)
    plan = validate_plan(read_json(run / "plan.json"))
    manifest = read_json(run / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("input_plan_sha256") != sha256(run / "plan.json"):
        raise ExperimentError("Input plan does not match its precommitted hash")
    if manifest.get("run_root") != run.relative_to(root).as_posix() or any(
            manifest.get(key) != plan[key] for key in ("question_id", "hypothesis", "changed_variable")):
        raise ExperimentError("Manifest identity does not match its immutable plan and run directory")
    result = {"schema_version": 1, "run_root": run.relative_to(root).as_posix(),
              "question_id": plan["question_id"], "stage": plan["stage"],
              "comparison": plan["comparison"], "controlled": plan["controlled"],
              "identities": plan["identities"],
              "comparison_class": "config" if plan["identities"]["baseline"] == plan["identities"]["candidate"] else "route",
              "adoption": "not_adopted", "adoptable": False,
              "validation_scope": f"{plan['resource']} {plan['stage']} on declared identities and controlled coordinates only; no production adoption or general B70 speed claim",
              "complete_sha256": sha256(run / "complete.json")}
    try:
        result.update(assess_measurements(run, plan, complete["files_sha256"]))
    except ExperimentError as exc:
        result.update(decision="blocked", reason=str(exc))
    phases = complete["phases"]
    expected_phases = {"command"} | {name for name in ("preflight", "cleanup") if name + "_command" in plan}
    if not complete["execution_passed"] or set(phases) != expected_phases or not all(successful(value) for value in phases.values()):
        result.update(decision="rejected", reason="Execution, preflight, cleanup, or precommit integrity failed; raw evidence retained")
    decision = result["decision"]
    result["outcome"] = ("validated_screen" if plan["stage"] == "screen" else "provisional") if decision == "eligible" else ("failed" if decision == "rejected" else "blocked")
    assessment_path = run / "assessment.json"
    if assessment_path.is_symlink():
        raise ExperimentError("Assessment cannot be a symlink")
    if assessment_path.exists():
        if read_json(assessment_path) != result:
            raise ExperimentError("Existing assessment differs; immutable assessments cannot be overwritten")
    else:
        write_new(assessment_path, result)
    return result


def assess_run(root: Path, run_path: Path) -> dict:
    root = Path(root).resolve()
    with writer_lock(root):
        return assess_locked(root, resolve_run(root, run_path))


def registry_rows(path: Path, key: str) -> tuple[dict, list]:
    data = read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get(key), list) or not all(isinstance(row, dict) for row in data[key]):
        raise ExperimentError(f"{path} must contain a {key} array of objects")
    return data, data[key]


def append_registry(path: Path, key: str, row: dict) -> None:
    """Atomically append an array element while retaining every preexisting byte."""
    original = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    cursor = original.index("{") + 1
    while True:
        while original[cursor].isspace() or original[cursor] == ",":
            cursor += 1
        field, cursor = decoder.raw_decode(original, cursor)
        while original[cursor].isspace() or original[cursor] == ":":
            cursor += 1
        value, end = decoder.raw_decode(original, cursor)
        if field == key:
            if not isinstance(value, list):
                raise ExperimentError(f"Registry {key} is not an array")
            insert = end - 1
            addition = ("," if value else "") + "\n" + json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            updated = original[:insert] + addition + original[insert:]
            break
        cursor = end
    handle, temporary = tempfile.mkstemp(prefix=".experiment-", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(updated)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def record_run(root: Path, run_path: Path) -> dict:
    root = Path(root).resolve()
    with writer_lock(root):
        run = resolve_run(root, run_path)
        assessment = assess_locked(root, run)
        manifest = read_json(run / "manifest.json")
        ledger_path = root / "registry/research-ledger.json"
        campaigns_path = root / "registry/campaigns.yaml"
        _, ledger = registry_rows(ledger_path, "entries")
        _, campaigns = registry_rows(campaigns_path, "campaigns")
        relative = run.relative_to(root).as_posix()
        matching_ledger = [row for row in ledger if row.get("run_root") == relative]
        matching_campaigns = [row for row in campaigns if row.get("run_root") == relative or row.get("evidence_root") == relative]
        if len(matching_ledger) > 1 or len(matching_campaigns) > 1:
            raise ExperimentError("Run already has duplicate registry entries; refusing to add another")
        for row in matching_ledger + matching_campaigns:
            if row.get("assessment_sha256") != sha256(run / "assessment.json"):
                raise ExperimentError("Existing registration does not match the immutable assessment")
        shared = {"run_root": relative, "question_id": assessment["question_id"],
                  "outcome": assessment["outcome"], "adoption": "not_adopted",
                  "validation_scope": assessment["validation_scope"],
                  "assessment_sha256": sha256(run / "assessment.json")}
        ledger_id = matching_ledger[0]["id"] if matching_ledger else None
        if ledger_id is None:
            ids = [int(row["id"][3:]) for row in ledger if isinstance(row.get("id"), str) and re.fullmatch(r"RL-\d{3}", row["id"])]
            number = max(ids, default=0) + 1
            if number > 999:
                raise ExperimentError("Ledger RL-NNN ID space is exhausted")
            ledger_id = f"RL-{number:03d}"
        campaign_id = "CAM-" + run.name.upper()
        if not matching_campaigns and any(row.get("id") == campaign_id for row in campaigns):
            raise ExperimentError("Campaign ID already belongs to a different run")
        if not matching_ledger:
            append_registry(ledger_path, "entries", {"id": ledger_id, "date": manifest["created_at"][:10],
                            "question": manifest["hypothesis"], "path_tried": manifest["changed_variable"],
                            "evidence": relative + "/assessment.json", "evidence_paths": [relative],
                            "guard": assessment["reason"], "superseded_by": None, **shared})
        if not matching_campaigns:
            append_registry(campaigns_path, "campaigns", {"id": campaign_id, "title": manifest["hypothesis"],
                            "evidence_root": relative, "record_path": relative + "/assessment.json",
                            "observed_at": manifest["created_at"][:10], "status": assessment["outcome"],
                            "publication_state": "not_for_public_headline", **shared})
        return {"run_root": relative, "ledger_id": ledger_id,
                "campaign_id": matching_campaigns[0]["id"] if matching_campaigns else campaign_id,
                "outcome": assessment["outcome"], "adoption": "not_adopted",
                "already_recorded": bool(matching_ledger and matching_campaigns)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "assess", "record"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--root", type=Path, default=ROOT)
        sub.add_argument("--plan" if command == "run" else "--run", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run_plan(args.root, args.plan)
        elif args.command == "assess":
            result = assess_run(args.root, args.run)
        else:
            result = record_run(args.root, args.run)
        print(json_text(result), end="")
        return 0
    except (ExperimentError, OSError) as exc:
        print(f"experiment: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
