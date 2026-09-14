#!/usr/bin/env python3
"""Render the B70 benchmark dashboard and repeatable-method SVGs."""
import argparse
import html
import json
from pathlib import Path

WIDTH = 1600
COLORS = {
    "bg": "#07111f",
    "panel": "#0f1d30",
    "panel2": "#0b1829",
    "line": "#263a53",
    "text": "#f5f8fc",
    "muted": "#9caec4",
    "accent": "#55d6be",
    "no-spec": "#aeb9c7",
    "mtp1": "#55d6be",
    "mtp2": "#68a7ff",
    "mtp4": "#ffb45c",
}
LABELS = {"no-spec": "No spec", "mtp1": "MTP1", "mtp2": "MTP2", "mtp4": "MTP4"}
MODES = tuple(LABELS)


def esc(value):
    return html.escape(str(value))


def text(x, y, value, size=24, color=None, weight=400, anchor="start"):
    color = color or COLORS["text"]
    return (
        f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{esc(value)}</text>'
    )


def panel(x, y, width, height, title, eyebrow):
    return [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="28" fill="{COLORS["panel"]}"/>',
        text(x + 34, y + 40, eyebrow.upper(), 16, COLORS["accent"], 700),
        text(x + 34, y + 78, title, 28, COLORS["text"], 700),
    ]


def table_panel(rows, x, y, width, height):
    out = panel(x, y, width, height, "Cold input rate", "Prefill / input ÷ TTFT")
    columns = [
        ("p512", "prefill-p512"),
        ("p2K", "prefill-p2048"),
        ("p4K", "prefill-p4096"),
        ("p6K", "prefill-p6144"),
        ("p8K", "prefill-p8192"),
        ("full", "prefill-full-p131071"),
    ]
    left = x + 34
    top = y + 126
    mode_width = 120
    col_width = (width - 68 - mode_width) / len(columns)
    out.append(f'<rect x="{left}" y="{top}" width="{width-68}" height="44" rx="12" fill="{COLORS["panel2"]}"/>')
    out.append(text(left + 14, top + 29, "Mode", 17, COLORS["muted"], 600))
    for index, (label, _) in enumerate(columns):
        out.append(text(left + mode_width + col_width * (index + 0.5), top + 29, label, 17, COLORS["muted"], 600, "middle"))
    winners = {}
    for _, coordinate in columns:
        winners[coordinate] = max(MODES, key=lambda mode: rows[(mode, coordinate)]["input_tokens_per_ttft_s"]["median"])
    row_height = 55
    for row_index, mode in enumerate(MODES):
        cy = top + 44 + row_height * row_index
        if row_index % 2:
            out.append(f'<rect x="{left}" y="{cy}" width="{width-68}" height="{row_height}" fill="#12243a" opacity="0.52"/>')
        out.append(f'<circle cx="{left+12}" cy="{cy+28}" r="5" fill="{COLORS[mode]}"/>')
        out.append(text(left + 27, cy + 34, LABELS[mode], 18, COLORS["text"], 600))
        for index, (_, coordinate) in enumerate(columns):
            value = rows[(mode, coordinate)]["input_tokens_per_ttft_s"]["median"]
            cx = left + mode_width + col_width * (index + 0.5)
            if winners[coordinate] == mode:
                out.append(f'<rect x="{cx-col_width*0.39:.1f}" y="{cy+8}" width="{col_width*0.78:.1f}" height="38" rx="10" fill="{COLORS[mode]}" opacity="0.14"/>')
            out.append(text(cx, cy + 34, f"{value:,.0f}", 18, COLORS["text"], 700 if winners[coordinate] == mode else 500, "middle"))
    return out


def decode_chart(rows, x, y, width, height, prompt):
    out = panel(x, y, width, height, f"Decode at p{prompt}", "Client post-first rate")
    outputs = [32, 128, 256, 512]
    plot_x = x + 75
    plot_y = y + 130
    plot_w = width - 112
    plot_h = height - 210
    low, high = 70, 190
    for tick in (80, 110, 140, 170):
        py = plot_y + plot_h * (high - tick) / (high - low)
        out.append(f'<line x1="{plot_x}" y1="{py:.1f}" x2="{plot_x+plot_w}" y2="{py:.1f}" stroke="{COLORS["line"]}" stroke-width="1"/>')
        out.append(text(plot_x - 12, py + 6, tick, 14, COLORS["muted"], 500, "end"))
    x_positions = [plot_x + plot_w * index / 3 for index in range(4)]
    for px, output in zip(x_positions, outputs):
        out.append(text(px, plot_y + plot_h + 30, f"g{output}", 15, COLORS["muted"], 600, "middle"))
    for mode in MODES:
        values = [rows[(mode, f"decode-p{prompt}-g{output}")]["client_post_first_tps"]["median"] for output in outputs]
        points = []
        for px, value in zip(x_positions, values):
            py = plot_y + plot_h * (high - value) / (high - low)
            points.append((px, py, value))
        out.append(f'<polyline points="{" ".join(f"{px:.1f},{py:.1f}" for px, py, _ in points)}" fill="none" stroke="{COLORS[mode]}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>')
        for px, py, value in points:
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="{COLORS[mode]}" stroke="{COLORS["panel"]}" stroke-width="3"/>')
    legend_x = x + 35
    legend_y = y + height - 28
    for index, mode in enumerate(MODES):
        lx = legend_x + index * 132
        out.append(f'<circle cx="{lx}" cy="{legend_y-5}" r="5" fill="{COLORS[mode]}"/>')
        out.append(text(lx + 12, legend_y, LABELS[mode], 14, COLORS["muted"], 600))
    return out




def dashboard(summary):
    rows = {(row["mode"], row["coordinate"]): row for row in summary["rows"]}
    stack = summary["stack"]
    height = 1280
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Intel Arc Pro B70 phase-separated vLLM benchmark</title>',
        '<desc id="desc">Cold input rate, short and eight-thousand-token decode, matched historical control, and full-context decode for no speculative decoding and MTP one, two, and four.</desc>',
        f'<rect width="{WIDTH}" height="{height}" fill="{COLORS["bg"]}"/>',
        '<style>text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-variant-numeric:tabular-nums}</style>',
        text(58, 64, "REAL-WORLD PI · C1 MODEL MATRIX", 18, COLORS["accent"], 800),
        text(58, 112, "Prefill, decode length, and full context separated", 39, COLORS["text"], 750),
        text(58, 148, f"vLLM {stack['vllm']} · XPU kernels {stack['vllm_xpu_kernels']} · exact tokens · median n=5", 19, COLORS["muted"], 500),
    ]
    out += table_panel(rows, 48, 190, 1504, 390)
    out += decode_chart(rows, 48, 610, 730, 420, 512)
    out += decode_chart(rows, 808, 610, 744, 420, 8192)
    strip_y = 1055
    out.append(f'<rect x="48" y="{strip_y}" width="1504" height="156" rx="26" fill="{COLORS["panel"]}"/>')
    out.append(text(78, strip_y + 34, "FULL CONTEXT · EXACT 131,072 TOTAL TOKENS", 15, COLORS["accent"], 800))
    for index, mode in enumerate(MODES):
        g128 = rows[(mode, "decode-full-p130944-g128")]["client_post_first_tps"]["median"]
        g512 = rows[(mode, "decode-full-p130560-g512")]["client_post_first_tps"]["median"]
        left = 72 + index * 372
        if index:
            out.append(f'<line x1="{left-20}" y1="{strip_y+50}" x2="{left-20}" y2="{strip_y+132}" stroke="{COLORS["line"]}" stroke-width="1"/>')
        out.append(f'<circle cx="{left+6}" cy="{strip_y+72}" r="6" fill="{COLORS[mode]}"/>')
        out.append(text(left + 22, strip_y + 79, LABELS[mode], 18, COLORS["text"], 700))
        out.append(text(left + 22, strip_y + 113, f"g128 {g128:.2f}", 17, COLORS["muted"], 600))
        out.append(text(left + 168, strip_y + 113, f"g512 {g512:.2f}", 17, COLORS["muted"], 600))
    control = rows[("mtp4", "decode-control-p9445-g128")]["client_post_first_tps"]["median"]
    out.append(text(58, 1252, f"Matched p9445/g128: MTP4 {control:.2f} tok/s (prior 158.83). Client post-first timing · cache on, zero hits · configured 165 W · E2 self-reported, not independently reproduced.", 15, COLORS["muted"], 500))
    out.append('</svg>')
    return "\n".join(out) + "\n"


def method_svg(summary):
    stack = summary["stack"]
    height = 900
    stages = [
        ("1", "Pin the stack", f"Image digest · model ID\nvLLM {stack['vllm']}\nXPU kernels {stack['vllm_xpu_kernels']}"),
        ("2", "Calibrate tokens", "Render with the model tokenizer\nRequire exact pN counts\nHash prompt files"),
        ("3", "Choose prompt class", "Compact fixed system: p512-p8192\nFull Pi system: p9445 + full context"),
        ("4", "Warm the exact shape", "Generic warmup, then one\nfull-output same-shape warmup\nDiscard both"),
        ("5", "Measure five C1 requests", "Unique entropy-first prefixes\nignore_eos=true for fixed gN\nRecord raw SSE timestamps"),
        ("6", "Check counters and limits", "Exact prompt/output tokens\nZero cache-hit delta\nMTP counters · VRAM · failures"),
        ("7", "Compile and audit", "Median, range, TPOT, E2E\nKeep exclusions and replacements\nSeparate C1 from concurrency"),
        ("8", "Publish with evidence", "Generate tables + SVG\nUpdate AGENTS and cookbook\nRun claims gate before headlines"),
    ]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Repeatable B70 benchmark and cookbook workflow</title>',
        '<desc id="desc">Eight stages from software and model pinning through exact-token calibration, warmup, measurement, evidence audit, and publication.</desc>',
        f'<rect width="{WIDTH}" height="{height}" fill="{COLORS["bg"]}"/>',
        '<style>text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-variant-numeric:tabular-nums}</style>',
        '<defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#47617f"/></marker></defs>',
        text(58, 62, "REPEATABLE BENCHMARK CONTRACT", 18, COLORS["accent"], 800),
        text(58, 110, "From a new model to a publishable cookbook result", 40, COLORS["text"], 750),
        text(58, 146, "Same coordinates, evidence fields, and claim boundaries on every run", 19, COLORS["muted"], 500),
    ]
    card_w, card_h = 350, 250
    x_positions = [48, 437, 826, 1215]
    y_positions = [205, 520]
    for index, (number, title_value, body) in enumerate(stages):
        row, col = divmod(index, 4)
        x, y = x_positions[col], y_positions[row]
        if col < 3:
            out.append(f'<line x1="{x+card_w}" y1="{y+125}" x2="{x_positions[col+1]-18}" y2="{y+125}" stroke="#47617f" stroke-width="3" marker-end="url(#arrow)"/>')
        elif row == 0:
            out.append(f'<path d="M {x+card_w/2} {y+card_h} C {x+card_w/2} {y+285}, {x_positions[0]+card_w/2} {y+285}, {x_positions[0]+card_w/2} {y_positions[1]-18}" fill="none" stroke="#47617f" stroke-width="3" marker-end="url(#arrow)"/>')
        out.append(f'<rect x="{x}" y="{y}" width="{card_w}" height="{card_h}" rx="28" fill="{COLORS["panel"]}"/>')
        out.append(f'<circle cx="{x+38}" cy="{y+40}" r="22" fill="{COLORS["accent"]}" opacity="0.18"/>')
        out.append(text(x + 38, y + 47, number, 18, COLORS["accent"], 800, "middle"))
        out.append(text(x + 72, y + 48, title_value, 24, COLORS["text"], 700))
        for line_index, line in enumerate(body.splitlines()):
            out.append(text(x + 28, y + 98 + line_index * 33, line, 17, COLORS["muted"], 500))
    out.append(text(58, 850, "Fixed-length decode: early EOS is retained as user behavior, excluded from gN throughput, and rerun with ignore_eos=true. Public results remain E2 until independent reproduction.", 16, COLORS["muted"], 500))
    out.append('</svg>')
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--dashboard", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    args.dashboard.parent.mkdir(parents=True, exist_ok=True)
    args.method.parent.mkdir(parents=True, exist_ok=True)
    args.dashboard.write_text(dashboard(summary))
    args.method.write_text(method_svg(summary))


if __name__ == "__main__":
    main()
