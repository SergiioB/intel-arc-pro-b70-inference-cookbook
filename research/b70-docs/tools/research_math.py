#!/usr/bin/env python3
"""Explicit-unit theoretical pricing and immutable primary-source snapshots."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import http.client
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_SOURCE_BYTES = 8 * 1024 * 1024
FETCH_TIMEOUT_S = 15
FETCH_DEADLINE_S = 30
SECTIONS = {"roofline", "amdahl", "speculation", "kv", "transfer"}


class InputError(ValueError):
    """An input cannot support the requested calculation or snapshot."""


def object_fields(value, required: set[str], optional: set[str], label: str) -> dict:
    if not isinstance(value, dict):
        raise InputError(f"{label}: provide a JSON object")
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise InputError(f"{label}: missing {', '.join(sorted(missing))}")
    if unknown:
        raise InputError(f"{label}: unknown fields {', '.join(sorted(unknown))}")
    return value


def text(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{label}: provide nonempty text")
    return value


def references(value, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise InputError(f"{label}: provide at least one evidence ID, path, or source URL")
    return [text(item, label) for item in value]


def provenance(value, label: str) -> dict:
    value = object_fields(value, {"kind", "references", "note"}, set(), label)
    if value["kind"] not in ("measured", "external_claim", "assumption"):
        raise InputError(f"{label}.kind: use measured, external_claim, or assumption")
    references(value["references"], f"{label}.references")
    text(value["note"], f"{label}.note")
    return value


def number(value, label: str, *, positive: bool = False, integer: bool = False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{label}: provide a finite {'integer' if integer else 'number'}")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (integer and not isinstance(value, int)):
        raise InputError(f"{label}: provide a finite {'integer' if integer else 'number'}")
    if value < 0 or (positive and value == 0):
        raise InputError(f"{label}: must be {'greater than' if positive else 'at least'} zero")
    return value


def section(data: dict, name: str, fields: set[str], optional: set[str] | None = None) -> dict:
    value = object_fields(data[name], fields | {"provenance"}, optional or set(), name)
    provenance(value["provenance"], f"{name}.provenance")
    return value


def ensure_finite(value) -> None:
    if isinstance(value, dict):
        for item in value.values():
            ensure_finite(item)
    elif isinstance(value, list):
        for item in value:
            ensure_finite(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number(value, "calculated result")


def price(data: dict) -> dict:
    """Calculate independent models; only transfer and roofline are combined."""
    object_fields(data, {"schema_version"}, SECTIONS, "input")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise InputError("schema_version: use integer 1")
    if not SECTIONS.intersection(data):
        raise InputError("input: provide at least one pricing section")
    result = {
        "schema_version": 1,
        "classification": "theoretical_inferred_bounds_not_measured_gains",
        "units": {"bandwidth_gbps": "decimal GB/s (10^9 bytes/s), not gigabits/s",
                  "compute_tflops": "10^12 FLOP/s", "GB": 10**9, "GiB": 2**30,
                  "ms": "10^-3 seconds", "us": "10^-6 seconds"},
        "limitations": ["Inputs and cited evidence are caller-declared, not independently verified.",
                        "No measured gain, correctness, quality, or adoption decision is implied.",
                        "Independent model speedups are not multiplied together."],
    }
    try:
        if "roofline" in data:
            row = section(data, "roofline", {"bytes_per_step", "flops_per_step", "bandwidth_gbps",
                                            "compute_tflops", "serial_ms", "tokens_per_step"})
            traffic = number(row["bytes_per_step"], "roofline.bytes_per_step", integer=True)
            flops = number(row["flops_per_step"], "roofline.flops_per_step", integer=True)
            bandwidth = number(row["bandwidth_gbps"], "roofline.bandwidth_gbps", positive=True)
            compute = number(row["compute_tflops"], "roofline.compute_tflops", positive=True)
            serial = number(row["serial_ms"], "roofline.serial_ms")
            tokens = number(row["tokens_per_step"], "roofline.tokens_per_step", positive=True)
            memory_ms = traffic / bandwidth / 10**6
            compute_ms = flops / compute / 10**9
            step_ms = max(memory_ms, compute_ms) + serial
            number(step_ms, "roofline calculated step_ms", positive=True)
            result["roofline"] = {
                "classification": "idealized_step_time_lower_bound",
                "memory_ms": memory_ms, "compute_ms": compute_ms, "serial_ms": serial,
                "step_ms_lower_bound": step_ms, "tokens_per_second_upper_bound": tokens * 1000 / step_ms,
                "assumption": "Memory and compute overlap perfectly; serial work does not overlap.",
                "provenance": row["provenance"],
            }
        if "amdahl" in data:
            row = section(data, "amdahl", {"wall_fraction", "local_speedup"})
            fraction = number(row["wall_fraction"], "amdahl.wall_fraction")
            speedup = number(row["local_speedup"], "amdahl.local_speedup", positive=True)
            if fraction > 1:
                raise InputError("amdahl.wall_fraction: must be at most one")
            relative_time = (1 - fraction) + fraction / speedup
            result["amdahl"] = {
                "classification": "inferred_wall_time_ratio",
                "predicted_wall_speedup": 1 / relative_time,
                "predicted_wall_time_ratio": relative_time,
                "infinite_local_speedup_ceiling": None if fraction == 1 else 1 / (1 - fraction),
                "ceiling_unbounded": fraction == 1, "provenance": row["provenance"],
            }
        if "speculation" in data:
            row = section(data, "speculation", {"proposal_depth", "marginal_survivals", "survival_evidence",
                                               "draft_ms", "verify_ms", "overhead_ms", "baseline_token_ms"})
            depth = number(row["proposal_depth"], "speculation.proposal_depth", integer=True)
            survivals = row["marginal_survivals"]
            if not isinstance(survivals, list) or len(survivals) != depth:
                raise InputError("speculation.marginal_survivals: supply one measured marginal survival per proposed position")
            references(row["survival_evidence"], "speculation.survival_evidence")
            previous = 1
            for value in survivals:
                number(value, "speculation.marginal_survivals")
                if value > previous:
                    raise InputError("speculation.marginal_survivals: must be nonincreasing probabilities in [0, 1]")
                previous = value
            cycle = sum(number(row[key], f"speculation.{key}") for key in ("draft_ms", "verify_ms", "overhead_ms"))
            number(cycle, "speculation calculated cycle_ms", positive=True)
            baseline = number(row["baseline_token_ms"], "speculation.baseline_token_ms", positive=True)
            token_yield = 1 + math.fsum(survivals)
            result["speculation"] = {
                "classification": "inferred_speculative_cycle_economics",
                "expected_tokens_per_cycle": token_yield, "cycle_ms": cycle,
                "predicted_token_ms": cycle / token_yield,
                "predicted_speedup": baseline * token_yield / cycle,
                "assumption": "One bonus/correction token per cycle; supplied survivals are P(accepted prefix >= position), not conditional or iid rates.",
                "survival_evidence": row["survival_evidence"], "provenance": row["provenance"],
            }
        if "kv" in data:
            row = section(data, "kv", {"full_attention_layers", "kv_heads", "head_dim", "context_tokens", "batch",
                                      "kv_bytes", "recurrent_state_bytes", "workspace_bytes", "weights_bytes"})
            layers = number(row["full_attention_layers"], "kv.full_attention_layers", integer=True)
            factors = {key: number(row[key], f"kv.{key}", integer=True,
                                   positive=key == "batch" or (layers > 0 and key in ("kv_heads", "head_dim")))
                       for key in ("full_attention_layers", "kv_heads", "head_dim", "context_tokens", "batch")}
            element_bytes = number(row["kv_bytes"], "kv.kv_bytes", positive=layers > 0)
            cache = 2 * math.prod(factors.values()) * element_bytes
            extras = {key: number(row[key], f"kv.{key}", integer=True)
                      for key in ("recurrent_state_bytes", "workspace_bytes", "weights_bytes")}
            total = cache + sum(extras.values())
            result["kv"] = {
                "classification": "declared_memory_accounting_not_measured_peak",
                "full_attention_kv_bytes": cache, **extras,
                "total_bytes": total, "total_GB": total / 10**9, "total_GiB": total / 2**30,
                "assumption": "Only explicitly declared full-attention layers contribute KV; recurrent state and workspace are separate totals, not per-layer values.",
                "provenance": row["provenance"],
            }
        if "transfer" in data:
            row = section(data, "transfer", {"bytes_per_step", "bandwidth_gbps", "collective_count", "latency_us"},
                          {"overlap_ms", "overlap_evidence"})
            traffic = number(row["bytes_per_step"], "transfer.bytes_per_step", integer=True)
            bandwidth = number(row["bandwidth_gbps"], "transfer.bandwidth_gbps", positive=True)
            count = number(row["collective_count"], "transfer.collective_count", integer=True)
            latency = number(row["latency_us"], "transfer.latency_us")
            transfer_ms = traffic / bandwidth / 10**6
            collective_ms = count * latency / 1000
            total_ms = transfer_ms + collective_ms
            overlap = 0
            if ("overlap_ms" in row) != ("overlap_evidence" in row):
                raise InputError("transfer: overlap_ms and overlap_evidence must be supplied together")
            if "overlap_ms" in row:
                if "roofline" not in result:
                    raise InputError("transfer: declared overlap requires a roofline section")
                overlap = number(row["overlap_ms"], "transfer.overlap_ms")
                references(row["overlap_evidence"], "transfer.overlap_evidence")
                if overlap > min(total_ms, result["roofline"]["step_ms_lower_bound"]):
                    raise InputError("transfer.overlap_ms: cannot exceed transfer time or roofline step time")
            result["transfer"] = {
                "classification": "inferred_additive_transfer_cost",
                "payload_ms": transfer_ms, "collective_latency_ms": collective_ms,
                "unoverlapped_ms": total_ms, "declared_overlap_ms": overlap,
                "incremental_ms": total_ms - overlap,
                "assumption": "Payload and per-collective latency are additive; only explicitly evidenced overlap with the roofline step is subtracted, once.",
                "provenance": row["provenance"],
            }
            if "overlap_evidence" in row:
                result["transfer"]["overlap_evidence"] = row["overlap_evidence"]
            if "roofline" in result:
                combined_ms = result["roofline"]["step_ms_lower_bound"] + total_ms - overlap
                result["combined_step"] = {
                    "classification": "inferred_roofline_plus_transfer",
                    "step_ms": combined_ms,
                    "tokens_per_second": data["roofline"]["tokens_per_step"] * 1000 / combined_ms,
                }
        ensure_finite(result)
    except (OverflowError, ZeroDivisionError) as exc:
        raise InputError("inputs exceed finite calculation range") from exc
    return result


def reject_constant(value: str):
    raise InputError(f"JSON: {value} is not a finite JSON number")


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise InputError(f"JSON: duplicate field {key}")
        value[key] = item
    return value


def read_price(path: Path) -> dict:
    raw = path.read_bytes()
    data = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_object)
    result = price(data)
    result["input"] = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "values": data}
    return result


def validate_url(url: str) -> str:
    text(url, "url")
    if any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in url):
        raise InputError("url: whitespace and control characters are not allowed")
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise InputError("url: provide a valid HTTPS URL") from exc
    if parts.scheme != "https" or not parts.hostname:
        raise InputError("url: use an absolute HTTPS primary-source URL")
    if parts.username is not None or parts.password is not None:
        raise InputError("url: embedded credentials are not allowed")
    if parts.fragment:
        raise InputError("url: remove the fragment; snapshot the actual source URL")
    if port is not None and not 1 <= port <= 65535:
        raise InputError("url: port must be between 1 and 65535")
    return url


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    max_redirections = 5
    max_repeats = 2

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_source(url: str) -> tuple[bytes, str, str]:
    validate_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "B70-research-source/1",
                                                  "Accept-Encoding": "identity"})
    opener = urllib.request.build_opener(SafeRedirect())
    started = time.monotonic()
    with opener.open(request, timeout=FETCH_TIMEOUT_S) as response:
        final_url = validate_url(response.geturl())
        if response.status != 200:
            raise InputError(f"source: expected HTTP 200, received {response.status}")
        encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
        if encoding not in ("", "identity"):
            raise InputError("source: server returned encoded content despite an identity request")
        length = response.headers.get("Content-Length")
        expected = None
        if length is not None:
            if not length.isascii() or not length.isdecimal():
                raise InputError("source: invalid Content-Length")
            expected = int(length)
            if expected > MAX_SOURCE_BYTES:
                raise InputError(f"source: exceeds the {MAX_SOURCE_BYTES}-byte limit")
        payload = bytearray()
        while True:
            if time.monotonic() - started > FETCH_DEADLINE_S:
                raise InputError(f"source: exceeded the {FETCH_DEADLINE_S}-second read deadline")
            chunk = response.read1(min(64 * 1024, MAX_SOURCE_BYTES + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > MAX_SOURCE_BYTES:
                raise InputError(f"source: exceeds the {MAX_SOURCE_BYTES}-byte limit")
        if time.monotonic() - started > FETCH_DEADLINE_S:
            raise InputError(f"source: exceeded the {FETCH_DEADLINE_S}-second read deadline")
        if not payload:
            raise InputError("source: received an empty payload")
        if expected is not None and len(payload) != expected:
            raise InputError("source: payload length does not match Content-Length")
        return bytes(payload), response.headers.get("Content-Type", "application/octet-stream"), final_url


def safe_output(root: Path, out: Path) -> Path:
    root = root.resolve()
    candidate = (out if out.is_absolute() else root / out).resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise InputError("out: must stay inside the selected research root") from exc
    parts = relative.parts
    if parts[:2] == ("research", "references"):
        return candidate
    if (len(parts) >= 4 and parts[0] == "results" and re.fullmatch(r"[0-9]{4}", parts[1])
            and re.fullmatch(r"0[1-9]|1[0-2]", parts[2]) and int(parts[1]) >= 1):
        run_root = root.joinpath(*parts[:4])
        if run_root.is_dir():
            return candidate
    raise InputError("out: use research/references or an existing results/YYYY/MM/RUN directory")


def snapshot_source(root: Path, url: str, claim: str, relevance: str, out: Path) -> dict:
    validate_url(url)
    text(claim, "claim")
    text(relevance, "relevance")
    destination = safe_output(root, out)
    payload, content_type, final_url = fetch_source(url)
    digest = hashlib.sha256(payload).hexdigest()
    destination.mkdir(parents=True, exist_ok=True)
    # Recheck after directory creation so a preexisting symlink cannot escape.
    destination = safe_output(root, destination)
    snapshot = destination / f"sha256-{digest}"
    try:
        snapshot.mkdir()
    except FileExistsError as exc:
        raise InputError(f"source: immutable snapshot already exists: {snapshot}") from exc
    metadata = {
        "schema_version": 1, "classification": "external_claim_not_validated_evidence",
        "source_role": "primary_source_asserted_by_caller",
        "url": url, "final_url": final_url,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "content_type": content_type, "sha256": digest, "size_bytes": len(payload),
        "claim": claim, "relevance": relevance, "content_file": "content",
        "interpretation": "Archival capture only; no claim truth, local measured gain, or adoption is inferred.",
    }
    created = []
    try:
        for name, content in (("content", payload),
                              ("metadata.json", (json.dumps(metadata, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))):
            path = snapshot / name
            with path.open("xb") as handle:
                created.append(path)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            path.chmod(0o444)
    except BaseException:
        for path in reversed(created):
            path.unlink()
        snapshot.rmdir()
        raise
    return {**metadata, "snapshot": str(snapshot.relative_to(root.resolve())),
            "metadata_path": str((snapshot / "metadata.json").relative_to(root.resolve())),
            "content_path": str((snapshot / "content").relative_to(root.resolve()))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Price explicit assumptions or archive an HTTPS primary source; neither command proves a gain.",
        epilog="Examples: research.py price --input pricing.json; research.py source --url https://raw.githubusercontent.com/ggml-org/llama.cpp/master/README.md --claim 'Upstream engine documentation' --relevance 'Engine provenance' --out research/references",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    pricing = commands.add_parser(
        "price", help="calculate theoretical/inferred bounds from explicit-unit JSON",
        description="JSON schema_version=1 plus one or more roofline, amdahl, speculation, kv, transfer objects. Every section requires provenance {kind, references, note}. No hardware defaults. bandwidth_gbps means decimal GB/s, not Gb/s. See research methodology for the full input contract.",
        epilog='Minimal example: {"schema_version":1,"amdahl":{"wall_fraction":0.5,"local_speedup":2,"provenance":{"kind":"assumption","references":["hypothesis:RQ14"],"note":"Sensitivity example, not measured"}}}',
    )
    pricing.add_argument("--input", required=True, type=Path, help="JSON path; relative paths resolve under --root")
    pricing.add_argument("--root", type=Path, default=ROOT, help="research root (default: repository)")
    source = commands.add_parser("source", help="archive actual source bytes and claim provenance without overwriting",
                                 description="HTTPS only; at most 8 MiB, 15-second socket timeout, 30-second read deadline. Redirects must remain HTTPS and credential-free. --out is a directory; creates sha256-DIGEST/content and metadata.json. Duplicate content in that directory is refused.")
    source.add_argument("--url", required=True, help="credential-free HTTPS primary-source URL, not a search snippet")
    source.add_argument("--claim", required=True, help="external claim attributed to this source")
    source.add_argument("--relevance", required=True, help="why the source matters to the local question")
    source.add_argument("--out", required=True, type=Path, help="directory under research/references or an existing results/YYYY/MM/RUN")
    source.add_argument("--root", type=Path, default=ROOT, help="research root (default: repository)")
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve()
        if args.command == "price":
            path = args.input if args.input.is_absolute() else root / args.input
            result = read_price(path)
        else:
            result = snapshot_source(root, args.url, args.claim, args.relevance, args.out)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (InputError, OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as exc:
        print(f"research {args.command}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
