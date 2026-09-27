#!/usr/bin/env python3
"""Render per-engine benchmark charts (SVG) from data/chart-inputs.v1.json.

House chart standard (2026-09-15, applied to this repo 2026-09-23): one
engine per chart and per series — OpenVINO GenAI, Cascadia, OVMS, llama.cpp
and vLLM are separate engines and are never merged in one table or chart;
log-x context axis with fixed ticks; line style encodes protocol (solid =
full-protocol cells, dashed = lighter probes, dotted dim = off-class
reference); sparse per-series labels (first, last, and two mid points);
legend panel below the plot (series rows, protocol-boundary notes, evidence
footer); PROVISIONAL — NOT FOR PUBLIC HEADLINE banner in the subtitle.
Numbers come only from the input JSON; this script carries no series data.

Outputs one SVG per chart entry (dated filenames under docs/assets/), then
validates the XML. Modes:
  (default)  write SVGs
  --check    verify the committed SVGs match the renderer output exactly
  --png      additionally render PNGs via rsvg-convert into --png-dir
             (default: a fresh temp dir) for the vision-QA gate

Idempotent: output depends only on the input JSON (no timestamps embedded).
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys
import tempfile
from xml.etree import ElementTree
from xml.sax.saxutils import escape

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "chart-inputs.v1.json"
DEFAULT_RSVG = "/usr/bin/rsvg-convert"
PNG_WIDTH = 1600

# Fixed house palette (dark background, panel, grid, text roles).
PALETTE = {
    "bg": "#07111f",
    "panel": "#0f1d30",
    "grid": "#22364e",
    "text": "#f5f8fc",
    "muted": "#9caec4",
    "accent": "#55d6be",
    "warn": "#ffd23f",
}
# One fixed color per engine, consistent across every chart in the repo.
# The first four are the chart-standard palette; cascadia and ovms are the
# documented fixed extensions for engines that are separate rows per the
# engine taxonomy (never merged with the engines above).
ENGINE_COLORS = {
    "openvino": "#68a7ff",
    "llama.cpp": "#7ee787",
    "vllm": "#ff8a7a",
    "reference": "#8a97a8",
    "cascadia": "#55d6be",
    "ovms": "#c792ea",
}
DASH_BY_STYLE = {
    "solid": "",
    "dashed": ' stroke-dasharray="9,7"',
    "dotted": ' stroke-dasharray="2,6"',
}
FONT = "Segoe UI, Inter, sans-serif"

W = 1560
PLOT_TOP = 150
PLOT_BOTTOM = 640
X0 = 150
X1 = W - 70
LEGEND_TOP_GAP = 66
LEGEND_HEADER = 62
LEGEND_ROW = 27
LEGEND_NOTE = 28
LEGEND_FOOTER = 20
LEGEND_BOTTOM_PAD = 16
SVG_BOTTOM_MARGIN = 24


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def nice_ymax(peak: float) -> float:
    """Round a y peak up to a step that yields at most four intervals."""
    target = peak * 1.08
    for exponent in range(-2, 7):
        for mantissa in (1.0, 2.0, 2.5, 5.0):
            step = mantissa * (10**exponent)
            intervals = math.ceil(target / step)
            if 0 < intervals <= 4:
                return intervals * step
    return target


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    if abs(value) < 10:
        return f"{value:.2f}"
    return f"{value:.1f}"


def load_inputs(path: pathlib.Path) -> list[dict]:
    try:
        catalog = json.loads(path.read_text())
    except FileNotFoundError:
        fail(f"input file not found: {path}")
    except json.JSONDecodeError as error:
        fail(f"input file is not valid JSON: {path}: {error}")
    if catalog.get("schema_version") != "1.0":
        fail("chart-inputs schema_version must be 1.0")
    charts = catalog.get("charts")
    if not isinstance(charts, list) or not charts:
        fail("chart-inputs must carry a non-empty charts array")
    return charts


def validate_chart(chart: dict, index: int) -> None:
    """Fail closed on malformed chart entries before anything is rendered."""
    where = f"charts[{index}]"
    for field in (
        "id",
        "output",
        "engine",
        "engine_color",
        "eyebrow",
        "title",
        "subtitle",
        "y_label",
        "x_label",
    ):
        if not chart.get(field):
            fail(f"{where}.{field} is required")
    axis = chart.get("x_axis") or {}
    lo, hi = axis.get("min"), axis.get("max")
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)) or not 0 < lo < hi:
        fail(f"{where}.x_axis must set 0 < min < max")
    ticks = chart.get("x_ticks")
    if not isinstance(ticks, list) or len(ticks) < 2:
        fail(f"{where}.x_ticks needs at least two ticks")
    for tick in ticks:
        if not isinstance(tick.get("value"), (int, float)) or not tick.get("label"):
            fail(f"{where}.x_ticks entries need a numeric value and a label")
    if chart.get("engine_color") not in ENGINE_COLORS:
        fail(f"{where}.engine_color is not a fixed engine color key")
    series_list = chart.get("series")
    if not isinstance(series_list, list) or not series_list:
        fail(f"{where}.series must be non-empty")
    for position, series in enumerate(series_list):
        spot = f"{where}.series[{position}]"
        if not series.get("label"):
            fail(f"{spot}.label is required")
        if series.get("color") != chart["engine_color"]:
            fail(f"{spot}.color must match the chart engine (one engine per chart)")
        segments = series.get("segments")
        if not isinstance(segments, list) or not segments:
            fail(f"{spot}.segments must be non-empty")
        for segment in segments:
            if segment.get("style") not in DASH_BY_STYLE:
                fail(f"{spot} segment style must be one of {sorted(DASH_BY_STYLE)}")
            if not segment.get("protocol"):
                fail(f"{spot} segment needs a protocol string for the legend")
            points = segment.get("points")
            if not isinstance(points, dict) or not points:
                fail(f"{spot} segment needs a non-empty points object")
            for key, value in points.items():
                try:
                    length = int(key)
                except (TypeError, ValueError):
                    fail(f"{spot} point key {key!r} is not an integer context length")
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    fail(f"{spot} point {key} needs a numeric value")
                if not lo <= length <= hi:
                    fail(f"{spot} point {key} lies outside the x axis [{lo}, {hi}]")


def sorted_points(segment: dict) -> list[tuple[int, float]]:
    return sorted((int(key), float(value)) for key, value in segment["points"].items())


def sparse_label_set(points: list[tuple[int, float]], lo: float, hi: float, mode: str) -> set[int]:
    """Endpoints plus (in sparse mode) the two points nearest 1/3 and 2/3 x."""
    if mode == "ends":
        return {points[0][0], points[-1][0]}
    if len(points) <= 4:
        return {length for length, _ in points}
    span = math.log2(hi) - math.log2(lo)
    chosen = {points[0][0], points[-1][0]}
    for fraction in (1 / 3, 2 / 3):
        target = math.log2(lo) + fraction * span
        nearest = min(points, key=lambda item: abs(math.log2(item[0]) - target))
        chosen.add(nearest[0])
    return chosen


def legend_segment_rows(charts_chart: dict) -> int:
    return sum(len(series["segments"]) for series in charts_chart["series"])


def svg_height(chart: dict) -> int:
    legend_top = PLOT_BOTTOM + LEGEND_TOP_GAP
    panel_h = (
        LEGEND_HEADER
        + legend_segment_rows(chart) * LEGEND_ROW
        + len(chart.get("notes", [])) * LEGEND_NOTE
        + LEGEND_FOOTER
        + LEGEND_BOTTOM_PAD
    )
    return legend_top + panel_h + SVG_BOTTOM_MARGIN


def render_chart(chart: dict) -> str:
    lo = chart["x_axis"]["min"]
    hi = chart["x_axis"]["max"]
    log_lo, log_hi = math.log2(lo), math.log2(hi)

    def lx(value: float) -> float:
        return X0 + (math.log2(value) - log_lo) / (log_hi - log_lo) * (X1 - X0)

    peak = max(
        value
        for series in chart["series"]
        for segment in series["segments"]
        for _, value in sorted_points(segment)
    )
    ymax = nice_ymax(peak)

    def ly(value: float) -> float:
        return PLOT_BOTTOM - value / ymax * (PLOT_BOTTOM - PLOT_TOP)

    color = ENGINE_COLORS[chart["engine_color"]]
    height = svg_height(chart)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
        f'viewBox="0 0 {W} {height}" role="img" aria-labelledby="t d">',
        f'<title id="t">{escape(chart["title"])}</title>',
        f'<desc id="d">{escape(chart["subtitle"])}</desc>',
        f'<rect width="{W}" height="{height}" fill="{PALETTE["bg"]}"/>',
        f'<text x="{X0}" y="46" fill="{PALETTE["muted"]}" font-size="20" font-weight="600" '
        f'font-family="{FONT}" letter-spacing="2">{escape(chart["eyebrow"])}</text>',
        f'<text x="{X0}" y="88" fill="{PALETTE["text"]}" font-size="29" font-weight="700" '
        f'font-family="{FONT}">{escape(chart["title"])}</text>',
        f'<text x="{X0}" y="116" fill="{PALETTE["warn"]}" font-size="15.5" '
        f'font-family="{FONT}">{escape(chart["subtitle"])}</text>',
    ]

    # Vertical gridlines and x tick labels (log-x axis).
    for tick in chart["x_ticks"]:
        position = lx(tick["value"])
        out.append(
            f'<line x1="{position:.1f}" y1="{PLOT_TOP}" x2="{position:.1f}" '
            f'y2="{PLOT_BOTTOM}" stroke="{PALETTE["grid"]}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{position:.1f}" y="{PLOT_BOTTOM + 26}" fill="{PALETTE["muted"]}" '
            f'font-size="14" font-family="{FONT}" text-anchor="middle">'
            f'{escape(str(tick["label"]))}</text>'
        )
    # Horizontal gridlines: four intervals across the y axis.
    for step in range(5):
        value = ymax * step / 4
        position = ly(value)
        out.append(
            f'<line x1="{X0}" y1="{position:.1f}" x2="{X1}" y2="{position:.1f}" '
            f'stroke="{PALETTE["grid"]}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{X0 - 12}" y="{position + 5:.1f}" fill="{PALETTE["muted"]}" '
            f'font-size="14" font-family="{FONT}" text-anchor="end">'
            f'{format_number(value)}</text>'
        )
    y_mid = (PLOT_TOP + PLOT_BOTTOM) / 2
    out.append(
        f'<text x="34" y="{y_mid:.1f}" fill="{PALETTE["text"]}" font-size="17" '
        f'font-family="{FONT}" text-anchor="middle" '
        f'transform="rotate(-90 34 {y_mid:.1f})">{escape(chart["y_label"])}</text>'
    )
    out.append(
        f'<text x="{(X0 + X1) / 2:.0f}" y="{PLOT_BOTTOM + 52}" fill="{PALETTE["muted"]}" '
        f'font-size="15" font-family="{FONT}" text-anchor="middle">'
        f'{escape(chart["x_label"])}</text>'
    )

    # Series: polyline segments with protocol dash styles, point markers, and
    # sparse labels (first, last, two mid) offset per series.
    for series in chart["series"]:
        series_color = ENGINE_COLORS[series["color"]]
        above = series.get("label_pos", "above") == "above"
        for segment in series["segments"]:
            points = sorted_points(segment)
            style = segment["style"]
            dash = DASH_BY_STYLE[style]
            width = 3.5 if style == "solid" else 2.5
            marker_r = 4.5 if style == "solid" else 3.5
            path = " ".join(f"{lx(length):.1f},{ly(value):.1f}" for length, value in points)
            out.append(
                f'<polyline points="{path}" fill="none" stroke="{series_color}" '
                f'stroke-width="{width}"{dash}/>'
            )
            for length, value in points:
                out.append(
                    f'<circle cx="{lx(length):.1f}" cy="{ly(value):.1f}" r="{marker_r}" '
                    f'fill="{series_color}"/>'
                )
            labeled = sparse_label_set(points, lo, hi, segment.get("labels", "sparse"))
            for length, value in points:
                if length not in labeled:
                    continue
                anchor = "middle"
                position_x = lx(length)
                offset = -16 if above else 30
                if length == points[0][0] and position_x - 26 < X0:
                    anchor, position_x = "start", X0 + 6
                elif length == points[-1][0] and position_x + 26 > X1:
                    anchor, position_x = "end", X1 - 6
                out.append(
                    f'<text x="{position_x:.1f}" y="{ly(value) + offset:.1f}" '
                    f'fill="{series_color}" font-size="13.5" font-family="{FONT}" '
                    f'text-anchor="{anchor}">{format_number(value)}</text>'
                )

    # Legend panel below the plot — nothing floats over the data.
    legend_top = PLOT_BOTTOM + LEGEND_TOP_GAP
    panel_h = svg_height(chart) - legend_top - SVG_BOTTOM_MARGIN
    out.append(
        f'<rect x="60" y="{legend_top}" width="{W - 120}" height="{panel_h}" '
        f'rx="16" fill="{PALETTE["panel"]}"/>'
    )
    row_y = legend_top + 34
    out.append(
        f'<text x="86" y="{row_y}" fill="{PALETTE["accent"]}" font-size="15" '
        f'font-weight="700" font-family="{FONT}">SERIES (points, last context)</text>'
    )
    row_y += LEGEND_ROW
    tick_labels = {tick["value"]: str(tick["label"]) for tick in chart["x_ticks"]}
    for series in chart["series"]:
        series_color = ENGINE_COLORS[series["color"]]
        for position, segment in enumerate(series["segments"]):
            points = sorted_points(segment)
            dash = DASH_BY_STYLE[segment["style"]]
            out.append(
                f'<line x1="86" y1="{row_y - 5}" x2="146" y2="{row_y - 5}" '
                f'stroke="{series_color}" stroke-width="3"{dash}/>'
            )
            last_tick = tick_labels.get(points[-1][0], str(points[-1][0]))
            if position == 0:
                legend_label = f'{series["label"]} · {segment["protocol"]}'
            else:
                legend_label = f'↳ composite continuation · {segment["protocol"]}'
            legend_label += f" · {len(points)} pts → {last_tick}"
            out.append(
                f'<text x="160" y="{row_y}" fill="{series_color}" font-size="15" '
                f'font-family="{FONT}">{escape(legend_label)}</text>'
            )
            row_y += LEGEND_ROW
    for note in chart.get("notes", []):
        row_y += 6
        out.append(
            f'<text x="86" y="{row_y}" fill="{PALETTE["warn"]}" font-size="14.5" '
            f'font-family="{FONT}">{escape(note)}</text>'
        )
        row_y += 22
    row_y += 4
    out.append(
        f'<text x="86" y="{row_y}" fill="{PALETTE["muted"]}" font-size="13.5" '
        f'font-family="{FONT}">{escape(chart.get("evidence_note", ""))}</text>'
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def write_charts(charts: list[dict]) -> list[pathlib.Path]:
    written: list[pathlib.Path] = []
    for index, chart in enumerate(charts):
        validate_chart(chart, index)
        rendered = render_chart(chart)
        ElementTree.fromstring(rendered)  # XML well-formedness gate
        output = ROOT / chart["output"]
        if not output.resolve().is_relative_to(ROOT):
            fail(f"chart {chart['id']}: output escapes the repository: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
        written.append(output)
        print(f"rendered {chart['id']} -> {output.relative_to(ROOT)} ({len(rendered)} bytes)")
    return written


def check_charts(charts: list[dict]) -> int:
    stale = False
    for index, chart in enumerate(charts):
        validate_chart(chart, index)
        output = ROOT / chart["output"]
        rendered = render_chart(chart)
        if not output.exists() or output.read_text() != rendered:
            print(f"STALE: {chart['output']} (chart {chart['id']})")
            stale = True
        else:
            print(f"current: {chart['output']} (chart {chart['id']})")
    if stale:
        print(
            "ERROR: chart outputs are stale or missing; run this script without --check",
            file=sys.stderr,
        )
        return 1
    print(f"Charts valid and current: {len(charts)} charts")
    return 0


def render_pngs(written: list[pathlib.Path], rsvg: str, png_dir: pathlib.Path | None) -> pathlib.Path:
    directory = png_dir or pathlib.Path(tempfile.mkdtemp(prefix="engine-charts-png-"))
    directory.mkdir(parents=True, exist_ok=True)
    for svg_path in written:
        png_path = directory / (svg_path.stem + ".png")
        completed = subprocess.run(
            [rsvg, "-w", str(PNG_WIDTH), "-o", str(png_path), str(svg_path)],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            fail(f"rsvg-convert failed for {svg_path}: {completed.stderr.strip()}")
        print(f"png {png_path}")
    return directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, default=DEFAULT_INPUT)
    parser.add_argument("--check", action="store_true", help="verify outputs are current")
    parser.add_argument("--png", action="store_true", help="also render PNGs for vision QA")
    parser.add_argument("--png-dir", type=pathlib.Path, default=None)
    parser.add_argument("--rsvg", default=DEFAULT_RSVG, help="path to rsvg-convert")
    args = parser.parse_args()

    charts = load_inputs(args.input)
    if args.check:
        return check_charts(charts)
    written = write_charts(charts)
    if args.png:
        render_pngs(written, args.rsvg, args.png_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
