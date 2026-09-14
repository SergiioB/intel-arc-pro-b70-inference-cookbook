#!/usr/bin/env python3
"""Render the canonical B70 phase-separated benchmark tables from summary.json."""
import argparse
import json
from pathlib import Path

MODE_LABELS = {
    "no-spec": "No spec",
    "mtp1": "MTP1",
    "mtp2": "MTP2",
    "mtp4": "MTP4",
}
MODES = tuple(MODE_LABELS)


def row_map(summary):
    return {(row["mode"], row["coordinate"]): row for row in summary["rows"]}


def median(rows, mode, coordinate, field):
    value = rows[(mode, coordinate)][field]
    return value["median"] if value else None


def fmt(value, digits=2):
    return "n/a" if value is None else f"{value:,.{digits}f}"


def render(summary):
    rows = row_map(summary)
    stack = summary["stack"]
    lines = [
        "## Phase-separated vLLM benchmark",
        "",
        f"Tested stack: vLLM `{stack['vllm']}`, `vllm-xpu-kernels {stack['vllm_xpu_kernels']}`, "
        f"C1, `n=5`, scheduler 8192, context 131072, prefix cache enabled with zero hit delta, "
        f"configured {summary['configured_power_cap_W']} W cap. Status: E2 self-reported; independent reproduction pending.",
        "",
        "### Cold input rate (actual input tokens / TTFT, tok/s)",
        "",
        "| Mode | p512 | p2048 | p4096 | p6144 | p8192 | Full p131071 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    prefill = ["prefill-p512", "prefill-p2048", "prefill-p4096", "prefill-p6144", "prefill-p8192", "prefill-full-p131071"]
    for mode in MODES:
        values = [fmt(median(rows, mode, coordinate, "input_tokens_per_ttft_s"), 0) for coordinate in prefill]
        lines.append(f"| {MODE_LABELS[mode]} | " + " | ".join(values) + " |")

    for prompt in (512, 8192):
        lines.extend([
            "",
            f"### Decode at p{prompt} (client post-first tok/s)",
            "",
            "| Mode | g32 | g128 | g256 | g512 |",
            "|---|---:|---:|---:|---:|",
        ])
        for mode in MODES:
            values = [
                fmt(median(rows, mode, f"decode-p{prompt}-g{output}", "client_post_first_tps"))
                for output in (32, 128, 256, 512)
            ]
            lines.append(f"| {MODE_LABELS[mode]} | " + " | ".join(values) + " |")

    lines.extend([
        "",
        "### Matched historical control (p9445/g128)",
        "",
        "| Mode | Client post-first median (tok/s) |",
        "|---|---:|",
    ])
    for mode in MODES:
        lines.append(
            f"| {MODE_LABELS[mode]} | "
            f"{fmt(median(rows, mode, 'decode-control-p9445-g128', 'client_post_first_tps'))} |")

    lines.extend([
        "",
        "### Full-context decode",
        "",
        "| Mode | p130944/g128 (tok/s) | MTP accept | p130560/g512 (tok/s) | MTP accept |",
        "|---|---:|---:|---:|---:|",
    ])
    for mode in MODES:
        g128 = rows[(mode, "decode-full-p130944-g128")]
        g512 = rows[(mode, "decode-full-p130560-g512")]
        accept128 = fmt(g128["mtp_acceptance_pct"]) + "%" if g128["mtp_acceptance_pct"] is not None else "n/a"
        accept512 = fmt(g512["mtp_acceptance_pct"]) + "%" if g512["mtp_acceptance_pct"] is not None else "n/a"
        lines.append(
            f"| {MODE_LABELS[mode]} | {fmt(g128['client_post_first_tps']['median'])} | {accept128} | "
            f"{fmt(g512['client_post_first_tps']['median'])} | {accept512} |")

    lines.extend([
        "",
        "Input rate includes request scheduling and first-token work; it is not llama-bench engine-native `pp`. "
        "Decode is client-observed and not engine-native vLLM throughput. Exact output rows use the requested completion length; exclusions and replacements remain in `summary.json`.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = render(json.loads(args.summary.read_text()))
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
