#!/usr/bin/env python3
"""Validate and summarize the B70 phase-separated prefill/decode campaign."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

MODES = ("no-spec", "mtp1", "mtp2", "mtp4")
COORDINATES = {
    "prefill-p512": (512, 1, "prefill"),
    "prefill-p2048": (2048, 1, "prefill"),
    "prefill-p4096": (4096, 1, "prefill"),
    "prefill-p6144": (6144, 1, "prefill"),
    "prefill-p8192": (8192, 1, "prefill"),
    "decode-p512-g32": (512, 32, "decode_small_prompt"),
    "decode-p512-g128": (512, 128, "decode_small_prompt"),
    "decode-p512-g256": (512, 256, "decode_small_prompt"),
    "decode-p512-g512": (512, 512, "decode_small_prompt"),
    "decode-p8192-g32": (8192, 32, "decode"),
    "decode-p8192-g128": (8192, 128, "decode"),
    "decode-control-p9445-g128": (9445, 128, "decode_historical_control"),
    "decode-p8192-g256": (8192, 256, "decode"),
    "decode-p8192-g512": (8192, 512, "decode"),
    "prefill-full-p131071": (131071, 1, "prefill_full_context"),
    "decode-full-p130944-g128": (130944, 128, "decode_full_context"),
    "decode-full-p130560-g512": (130560, 512, "decode_full_context"),
}
ERROR_MARKERS = (
    "Traceback (most recent call last)",
    "AssertionError",
    "EngineCore encountered a fatal error",
    "APIServer process died",
    "RuntimeError: Engine core initialization failed",
)


def stats(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return {
        "n": len(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "pstdev": statistics.pstdev(values),
    }


def text_hash(record):
    payload = (record.get("reasoning_text") or "") + "\0" + (record.get("content_text") or "")
    return hashlib.sha256(payload.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    rows = []
    prompt_hashes = {}
    excluded = []

    manifest = (root / "manifest.txt").read_text()
    if "result=all_four_modes_completed" not in manifest:
        raise RuntimeError("campaign manifest is not complete")
    manifest_values = dict(
        line.split("=", 1) for line in manifest.splitlines() if "=" in line)
    package_path = root / "package-versions.txt"
    if not package_path.exists():
        package_path = root / "no-spec" / "package-versions.txt"
    package_text = package_path.read_text()
    package_values = dict(
        line.split("=", 1) for line in package_text.splitlines() if "=" in line)

    for mode in MODES:
        mode_manifest = (root / mode / "manifest.txt").read_text()
        if "result=completed" not in mode_manifest:
            raise RuntimeError(f"mode did not complete: {mode}")
        server_log = (root / mode / "server.log").read_text(errors="replace")
        found_errors = [marker for marker in ERROR_MARKERS if marker in server_log]
        if found_errors:
            raise RuntimeError(f"server errors in {mode}: {found_errors}")
        if "'enable_prefix_caching': True" not in server_log:
            raise RuntimeError(f"prefix caching was not enabled in {mode}")

        for coordinate, (prompt_tokens, output_tokens, result_class) in COORDINATES.items():
            path = root / mode / coordinate / "results.json"
            replaced_original = False
            repair_root = root / mode / "decode-full-p130560-g512-forced-exact"
            if (mode == "no-spec"
                    and coordinate == "decode-full-p130560-g512"
                    and repair_root.exists()):
                repair_manifest = (repair_root / "manifest.txt").read_text()
                if "result=completed" not in repair_manifest:
                    raise RuntimeError("forced-exact no-spec g512 repair is incomplete")
                repair_log = (repair_root / "server.log").read_text(errors="replace")
                repair_errors = [marker for marker in ERROR_MARKERS if marker in repair_log]
                if repair_errors:
                    raise RuntimeError(f"server errors in no-spec g512 repair: {repair_errors}")
                path = repair_root / "results" / "results.json"
                replaced_original = True
                excluded.append({
                    "source": "no-spec/decode-full-p130560-g512/results.json",
                    "reason": "three of five requests stopped at EOS before 512 tokens",
                    "replacement": str(path.relative_to(root)),
                })
            data = json.loads(path.read_text())
            records = data["records"]
            if len(records) != 5:
                raise RuntimeError(f"{mode}/{coordinate}: n={len(records)}")
            for record in records:
                if record["prompt_tokens"] != prompt_tokens:
                    raise RuntimeError(f"{mode}/{coordinate}: prompt token mismatch")
                if record["completion_tokens"] != output_tokens:
                    raise RuntimeError(f"{mode}/{coordinate}: output token mismatch")
                if record["finish_reason"] != "length":
                    raise RuntimeError(f"{mode}/{coordinate}: finish={record['finish_reason']}")
                if record["prefix_cache_hits_delta"] != 0:
                    raise RuntimeError(f"{mode}/{coordinate}: cache hit contamination")
            proposed = sum(record.get("mtp_proposed_tokens") or 0 for record in records)
            accepted = sum(record.get("mtp_accepted_tokens") or 0 for record in records)
            if mode == "no-spec" and (proposed or accepted):
                raise RuntimeError(f"{mode}/{coordinate}: unexpected MTP counters")
            if mode != "no-spec" and output_tokens > 1 and proposed <= 0:
                raise RuntimeError(f"{mode}/{coordinate}: missing MTP proposals")

            post_first = [record.get("client_post_first_tps") for record in records]
            row = {
                "mode": mode,
                "coordinate": coordinate,
                "class": result_class,
                "prompt_tokens": prompt_tokens,
                "requested_output_tokens": output_tokens,
                "actual_output_tokens": [record["completion_tokens"] for record in records],
                "source_results": str(path.relative_to(root)),
                "replaced_early_eos_original": replaced_original,
                "ignore_eos": data.get("ignore_eos", False),
                "samples": 5,
                "ttft_s": stats([record["ttft_s"] for record in records]),
                "ttfc_s": stats([record.get("ttfc_s") for record in records]),
                "e2e_s": stats([record["total_s"] for record in records]),
                "input_tokens_per_ttft_s": stats(
                    [record["input_tokens_per_ttft_s"] for record in records]),
                "client_post_first_tps": stats(post_first),
                "client_post_first_tpot_ms": stats(
                    [1000.0 / value for value in post_first if value]),
                "prefix_cache_hits_delta": sum(
                    record["prefix_cache_hits_delta"] for record in records),
                "prefix_cache_queries_delta": sum(
                    record["prefix_cache_queries_delta"] for record in records),
                "mtp_proposed_tokens": proposed,
                "mtp_accepted_tokens": accepted,
                "mtp_acceptance_pct": (100.0 * accepted / proposed) if proposed else None,
                "output_sha256_by_rep": [text_hash(record) for record in records],
            }
            rows.append(row)
            prompt_hashes.setdefault(coordinate, {})[mode] = [
                record["messages_sha256"] for record in records]

    prompt_parity = {}
    output_parity = {}
    for coordinate in COORDINATES:
        mode_prompt_hashes = prompt_hashes[coordinate]
        prompt_parity[coordinate] = len({tuple(value) for value in mode_prompt_hashes.values()}) == 1
        coordinate_rows = [row for row in rows if row["coordinate"] == coordinate]
        output_parity[coordinate] = sum(
            all(row["output_sha256_by_rep"][rep] == coordinate_rows[0]["output_sha256_by_rep"][rep]
                for row in coordinate_rows)
            for rep in range(5)
        )

    if not all(prompt_parity.values()):
        raise RuntimeError(f"prompt parity failure: {prompt_parity}")

    output = {
        "schema": 1,
        "run_id": root.name,
        "status": "E2_PROVISIONAL_SELF_REPORTED_NOT_INDEPENDENTLY_REPRODUCED",
        "scope": "C1 phase-separated vLLM serving measurements; medians of five measured requests per cell",
        "timing_source": "client monotonic SSE timestamps",
        "prefill_metric_warning": "input_tokens_per_ttft_s includes scheduling, uncached prompt processing, and first-token work; it is not isolated engine prefill throughput",
        "decode_metric_warning": "client post-first rate is request-side, not engine-native vLLM decode",
        "cache": "prefix caching enabled; unique entropy-first cold prompts; zero hit delta",
        "configured_power_cap_W": 165,
        "scheduler_budget": 8192,
        "max_model_len": 131072,
        "excluded": excluded,
        "stack": {
            "image": manifest_values.get("image", "unknown"),
            "model": manifest_values.get(
                "model_id", manifest_values.get("model", "unknown")),
            "model_local_directory_basename": Path(
                manifest_values.get("model", "unknown")).name,
            "vllm": package_values.get("vllm", "unknown"),
            "vllm_xpu_kernels": package_values.get("vllm-xpu-kernels", "unknown"),
            "mtp_patch_sha256": manifest_values.get("mtp_patch_sha256", "unknown"),
            "boundary_patch_sha256": manifest_values.get("boundary_patch_sha256", "unknown"),
            "standard_system_file_sha256": manifest_values.get(
                "standard_system_file_sha256", "unknown"),
            "pi_system_file_sha256": manifest_values.get(
                "pi_system_file_sha256", "unknown"),
        },
        "rows": rows,
        "prompt_parity_across_modes": prompt_parity,
        "exact_output_parity_repetitions_out_of_5": output_parity,
    }
    destination = args.output or root / "summary.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
