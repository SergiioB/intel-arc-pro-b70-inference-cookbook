#!/usr/bin/env python3
"""Render the Qwen3.8-27B FP8 TP2 public SVG set from retained JSON evidence.

Inputs are the sanitized/public benchmark JSON files under
results/qwen38-27-fp8-tp2-k8-n5/. The renderer never reads private result
roots and does not copy benchmark values into a second source file.

Usage:
  python3 benchmarks/render-qwen38-fp8-tp2-svgs.py
  python3 benchmarks/render-qwen38-fp8-tp2-svgs.py --check
"""
from __future__ import annotations

import argparse
import html
import json
import statistics
import sys
from pathlib import Path

W = 1600
C = {
    "bg": "#07111f",
    "panel": "#0f1d30",
    "panel2": "#0b1829",
    "line": "#263a53",
    "text": "#f5f8fc",
    "muted": "#9caec4",
    "accent": "#55d6be",
    "blue": "#68a7ff",
    "warn": "#ffb45c",
    "danger": "#ff7b86",
    "grid": "#20344d",
}
ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/qwen38-27-fp8-tp2-k8-n5"
ASSETS = ROOT / "docs/assets"


def esc(value: object) -> str:
    return html.escape(str(value))


def text(x: float, y: float, value: object, size: int = 24, color: str | None = None,
         weight: int = 400, anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" fill="{color or C["text"]}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" '
        f'font-family="ui-sans-serif, system-ui, sans-serif">{esc(value)}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str, rx: int = 24,
         stroke: str | None = None, stroke_width: int = 1) -> str:
    border = "" if stroke is None else f' stroke="{stroke}" stroke-width="{stroke_width}"'
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"{border}/>'


def line(x1: float, y1: float, x2: float, y2: float, color: str, width: int = 2,
         dash: str | None = None) -> str:
    d = "" if dash is None else f' stroke-dasharray="{dash}"'
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{d}/>'


def panel(parts: list[str], x: int, y: int, w: int, h: int, eyebrow: str, title: str) -> None:
    parts.extend([
        rect(x, y, w, h, C["panel"], 28),
        text(x + 34, y + 42, eyebrow.upper(), 16, C["accent"], 700),
        text(x + 34, y + 82, title, 26, C["text"], 700),
    ])


def svg_start(height: int, title_value: str, desc_value: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{esc(title_value)}</title>',
        f'<desc id="desc">{esc(desc_value)}</desc>',
        f'<rect width="{W}" height="{height}" fill="{C["bg"]}"/>',
    ]


def header(parts: list[str], title_value: str, subtitle: str, height: int = 140) -> None:
    parts.extend([
        text(48, 52, "QWEN3.8-27B FP8 · DUAL B70 TP2 · vLLM XPU", 16, C["accent"], 700),
        text(48, 98, title_value, 36, C["text"], 700),
        text(48, 132, subtitle, 18, C["muted"]),
        rect(1190, 30, 362, 88, C["panel"], 18),
        text(1371, 64, "E2 SELF-REPORTED", 20, C["accent"], 700, "middle"),
        text(1371, 94, "independent reproduction pending", 13, C["muted"], 600, "middle"),
    ])


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def e2e_rates(raw: dict) -> list[float]:
    return [out / (ms / 1000.0) for out, ms in zip(raw["output_lens"], [v * 1000 for v in raw["e2el_ms"]])] if "e2el_ms" in raw else [
        out / (ms / 1000.0) for out, ms in zip(raw["output_lens"], [
            sum(itls) * 1000 + ttft * 1000 for itls, ttft in zip(raw["itls"], raw["ttfts"])
        ])
    ]


def raw_cell(path: Path) -> dict:
    raw = load_json(path)
    rates = [out / (e2e_ms / 1000.0) for out, e2e_ms in zip(raw["output_lens"], _e2e_ms(raw))]
    return {
        "raw": raw,
        "prompt": raw["input_lens"][0],
        "output": raw["output_lens"][0],
        "rates": rates,
        "median": statistics.median(rates),
        "maximum": max(rates),
        "minimum": min(rates),
        "run_level": raw["output_throughput"],
        "ttft_ms": raw["median_ttft_ms"],
        "cold_input": raw["input_lens"][0] / (raw["median_ttft_ms"] / 1000.0),
        "mean_tpot_ms": raw["mean_tpot_ms"],
        "accept_len": raw["spec_decode_acceptance_length"],
        "accept_pct": raw["spec_decode_acceptance_rate"],
    }


def _e2e_ms(raw: dict) -> list[float]:
    # vllm bench stores request-level E2E summaries only as aggregate fields in
    # this JSON generation. Reconstruct each request from TTFT + emitted ITLs.
    return [(ttft + sum(itls)) * 1000.0 for ttft, itls in zip(raw["ttfts"], raw["itls"])]


def dashboard(cells: list[dict]) -> str:
    h = 1480
    parts = svg_start(
        h,
        "Qwen3.8-27B FP8 TP2 benchmark dashboard on dual Intel Arc Pro B70",
        "Three greedy random-token C1 cells for Qwen3.8-27B FP8 W8A16 with MTP8 on two Intel Arc Pro B70 GPUs in tensor parallel mode. Values are n=5 medians or explicitly labeled maxima. Cold input is endpoint prompt tokens divided by client TTFT. Self-reported E2 and not independently reproduced.",
    )
    header(parts, "FP8 TP2 · W8A16 + Xe2 small-M · MTP8", "Greedy random-token diagnostic · C1 · cache disabled · 230 W configured cap per GPU")

    panel(parts, 48, 170, 1504, 410, "C1 output throughput", "Client end-to-end output tok/s · five measured requests per cell")
    x0, y0 = 82, 280
    col_w = 466
    for i, cell in enumerate(cells):
        x = x0 + i * (col_w + 18)
        parts.extend([
            rect(x, y0, col_w, 250, C["panel2"], 20),
            text(x + 24, y0 + 40, f'p{cell["prompt"]} / g{cell["output"]}', 18, C["muted"], 700),
            text(x + 24, y0 + 102, f'{cell["median"]:.2f}', 44, C["accent"], 700),
            text(x + 190, y0 + 102, "median tok/s", 16, C["muted"], 600),
            text(x + 24, y0 + 148, f'maximum observed  {cell["maximum"]:.2f}', 17, C["warn"], 600),
            text(x + 24, y0 + 182, f'range  {cell["minimum"]:.2f}–{cell["maximum"]:.2f}', 16, C["muted"]),
            text(x + 24, y0 + 216, f'run-level output  {cell["run_level"]:.2f} tok/s', 16, C["muted"]),
        ])
    parts.append(text(82, 555, "Greedy random-token diagnostic · median and maximum observed labeled separately.", 16, C["muted"], 600))

    panel(parts, 48, 620, 730, 520, "Latency and cold input", "Client timing · linear scales")
    chart_x, chart_y, chart_w, chart_h = 100, 755, 620, 260
    max_input = 2000.0
    for tick in range(0, 2001, 500):
        x = chart_x + chart_w * tick / max_input
        parts.extend([line(x, chart_y, x, chart_y + chart_h, C["grid"], 1), text(x, chart_y + chart_h + 28, tick, 13, C["muted"], 500, "middle")])
    for i, cell in enumerate(cells):
        y = chart_y + 28 + i * 78
        width = chart_w * cell["cold_input"] / max_input
        parts.extend([
            text(chart_x, y - 10, f'p{cell["prompt"]}/g128', 15, C["muted"], 600),
            rect(chart_x, y, width, 28, C["blue"], 8),
            text(chart_x + max(width - 10, 64), y + 21, f'{cell["cold_input"]:.1f}', 15, C["text"], 700, "end"),
            text(chart_x + chart_w, y - 10, f'TTFT {cell["ttft_ms"]:.1f} ms', 14, C["muted"], 500, "end"),
        ])
    parts.extend([
        text(chart_x + chart_w / 2, 1060, "cold-input rate (actual endpoint prompt tokens / client TTFT), token/s", 14, C["muted"], 600, "middle"),
        text(82, 1104, "This includes scheduling and first-token work; it is not engine-isolated prefill.", 14, C["warn"], 600),
    ])

    panel(parts, 822, 620, 730, 520, "MTP8 diagnostics", "Acceptance and per-token latency")
    left = 856
    parts.extend([
        rect(left, 750, 662, 42, C["panel2"], 10),
        text(left + 16, 778, "Cell", 15, C["muted"], 600),
        text(left + 240, 778, "accepted length", 15, C["muted"], 600, "middle"),
        text(left + 420, 778, "draft-token accept", 15, C["muted"], 600, "middle"),
        text(left + 585, 778, "mean TPOT", 15, C["muted"], 600, "middle"),
    ])
    for i, cell in enumerate(cells):
        y = 798 + i * 68
        if i % 2:
            parts.append(rect(left, y, 662, 60, "#12243a", 0))
        parts.extend([
            text(left + 16, y + 38, f'p{cell["prompt"]}/g128', 16, C["text"], 600),
            text(left + 240, y + 38, f'{cell["accept_len"]:.2f}', 19, C["accent"], 700, "middle"),
            text(left + 420, y + 38, f'{cell["accept_pct"]:.1f}%', 17, C["text"], 600, "middle"),
            text(left + 585, y + 38, f'{cell["mean_tpot_ms"]:.2f} ms', 17, C["text"], 600, "middle"),
        ])
    parts.extend([
        text(left, 1030, "p2048 uses hybrid mode (W8A8 prefill / W8A16 small-M decode).", 15, C["muted"]),
        text(left, 1060, "Acceptance is content- and sampling-dependent; it is diagnostic, not a quality score.", 15, C["warn"], 600),
        text(left, 1090, "All cells: C1, n=5, exact g128 completions, random-token synthetic prompts.", 15, C["muted"]),
    ])

    panel(parts, 48, 1180, 1504, 220, "Claim boundary", "What this evidence does and does not establish")
    parts.extend([
        text(82, 1280, "Establishes", 17, C["accent"], 700),
        text(82, 1314, "Completed fixed-length C1 cells, client timing,", 15, C["text"]),
        text(82, 1342, "MTP counters, and retained raw samples.", 15, C["text"]),
        text(820, 1280, "Does not establish", 17, C["warn"], 700),
        text(820, 1314, "Natural-content quality, token/logit parity, sustained", 15, C["text"]),
        text(820, 1342, "serving, Cn capacity, or independent reproduction.", 15, C["text"]),
        text(82, 1380, "Source: public n5-p512-g128.json, n5-p1024-g128.json, n5-p2048-g128.json", 14, C["muted"]),
    ])
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def sampling_collapse(greedy: list[dict], sampled: list[dict]) -> str:
    h = 1240
    parts = svg_start(
        h,
        "Qwen3.8-27B FP8 TP2 sampling sensitivity on random-token prompts",
        "Paired p512/g128 and p1024/g128 C1 comparisons between greedy temperature zero diagnostics and the Qwen model-card non-thinking sampling preset. Client end-to-end output throughput and mean accepted speculative length both fall under the sampled random-token workload. Each point is a median or aggregate from five measured requests. This is not a natural-content quality evaluation.",
    )
    header(parts, "Sampling sensitivity: greedy versus model-card preset", "Same FP8 TP2 MTP8 stack · random-token synthetic prompts · C1 · n=5")

    panel(parts, 48, 170, 920, 650, "Output throughput", "Client end-to-end output tok/s · linear 0–70 axis")
    cx, cy, cw, ch = 110, 300, 800, 400
    for tick in range(0, 71, 10):
        y = cy + ch - ch * tick / 70
        parts.extend([line(cx, y, cx + cw, y, C["grid"], 1), text(cx - 16, y + 5, tick, 13, C["muted"], 500, "end")])
    group_centers = [350, 670]
    bar_w = 108
    for idx, center in enumerate(group_centers):
        gv = greedy[idx]["median"]
        sv = sampled[idx]["statistics"]["median_client_e2e_output_tps"]
        for value, x, color, label in [(gv, center - 120, C["accent"], "greedy"), (sv, center + 12, C["warn"], "model-card")]:
            bh = ch * value / 70
            parts.extend([
                rect(x, cy + ch - bh, bar_w, bh, color, 10),
                text(x + bar_w / 2, cy + ch - bh - 14, f'{value:.2f}', 18, C["text"], 700, "middle"),
                text(x + bar_w / 2, cy + ch + 30, label, 14, C["muted"], 600, "middle"),
            ])
        parts.append(text(center, cy + ch + 66, f'p{greedy[idx]["prompt"]}/g128', 18, C["text"], 700, "middle"))
    parts.append(text(510, 804, "Teal: temperature 0 greedy · Orange: model-card 0.7 / 0.8 / 20 / presence 1.5", 14, C["muted"], 600, "middle"))

    panel(parts, 1012, 170, 540, 650, "Accepted length", "Mean accepted MTP tokens · linear 0–6 axis")
    ax, ay, aw, ah = 1070, 300, 420, 400
    for tick in range(0, 7):
        y = ay + ah - ah * tick / 6
        parts.extend([line(ax, y, ax + aw, y, C["grid"], 1), text(ax - 14, y + 5, tick, 13, C["muted"], 500, "end")])
    centers = [1185, 1380]
    small_w = 66
    for idx, center in enumerate(centers):
        gv = greedy[idx]["accept_len"]
        sv = sampled[idx]["speculative_decode"]["mean_acceptance_length"]
        for value, x, color in [(gv, center - 74, C["accent"]), (sv, center + 8, C["warn"])]:
            bh = ah * value / 6
            parts.extend([
                rect(x, ay + ah - bh, small_w, bh, color, 8),
                text(x + small_w / 2, ay + ah - bh - 12, f'{value:.2f}', 16, C["text"], 700, "middle"),
            ])
        parts.append(text(center, ay + ah + 42, f'p{greedy[idx]["prompt"]}', 16, C["text"], 700, "middle"))
    parts.extend([
        text(1046, 760, "Acceptance is not a quality metric.", 15, C["warn"], 600),
        text(1046, 790, "Random-token prompts can distort speculation behavior.", 14, C["muted"]),
    ])

    panel(parts, 48, 860, 1504, 300, "Interpretation", "The sampling preset changes this synthetic workload materially")
    parts.extend([
        text(82, 960, "p512/g128", 17, C["accent"], 700),
        text(82, 996, "53.42 → 22.16 tok/s median; mean accepted length 5.27 → 1.98.", 20, C["text"], 600),
        text(790, 960, "p1024/g128", 17, C["accent"], 700),
        text(790, 996, "60.13 → 11.66 tok/s median; mean accepted length 5.75 → 1.14.", 20, C["text"], 600),
        text(82, 1050, "Use the model-card cells as sampling-sensitivity diagnostics, not as natural-content quality results.", 16, C["warn"], 600),
        text(82, 1084, "Greedy and model-card cells are separate sampling conditions on the same synthetic prompt class.", 16, C["muted"], 600),
        text(82, 1124, "Source: public n5 JSON and sanitized model-card-sampling JSON; all cells C1, n=5, exact g128.", 14, C["muted"]),
    ])
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def arrow(parts: list[str], x1: int, y1: int, x2: int, y2: int, label: str | None = None,
          color: str | None = None) -> None:
    color = color or C["blue"]
    parts.append(line(x1, y1, x2, y2, color, 4))
    parts.append(f'<polygon points="{x2},{y2} {x2-16},{y2-9} {x2-16},{y2+9}" fill="{color}"/>')
    if label:
        parts.append(text((x1 + x2) / 2, y1 - 14, label, 14, C["muted"], 600, "middle"))


def pipeline(sample: dict) -> str:
    h = 1180
    parts = svg_start(
        h,
        "Qwen3.8-27B FP8 W8A16 dual B70 TP2 inference pipeline",
        "A process diagram for the public Qwen3.8-27B FP8 recipe. One C1 request enters vLLM, tensor-parallel workers are pinned to one Intel Arc Pro B70 each, oneCCL simple algorithms stage collectives through host memory because GPU peer-to-peer is unavailable, FP8 W8A16 target compute uses the Xe2 small-M decode path, and native MTP8 drafts are verified before streaming. The public evidence covers C1 only and is self-reported E2.",
    )
    header(parts, "Request-to-token pipeline", "TP2 worker isolation · host-staged collectives · W8A16 decode · native MTP8")

    panel(parts, 48, 170, 1504, 760, "Execution path", "One request · two isolated workers · one streamed response")
    y = 330
    boxes = [
        (82, 270, 230, 150, "1", "HTTP request", ["C1 only", "exact prompt + g128"]),
        (365, 270, 250, 150, "2", "vLLM scheduler", ["max_num_seqs = 1", "cache disabled"]),
        (668, 270, 350, 150, "3", "TP2 target forward", ["rank 0 → B70 GPU 0", "rank 1 → B70 GPU 1"]),
        (1070, 270, 230, 150, "4", "MTP8 draft", ["native model head", "8 proposed positions"]),
        (1352, 270, 166, 150, "5", "Stream", ["verify", "emit tokens"]),
    ]
    for x, by, bw, bh, num, title_value, lines in boxes:
        parts.extend([
            rect(x, by, bw, bh, C["panel2"], 20, C["line"]),
            text(x + 20, by + 36, num, 16, C["accent"], 700),
            text(x + 20, by + 70, title_value, 20, C["text"], 700),
            text(x + 20, by + 104, lines[0], 14, C["muted"]),
            text(x + 20, by + 130, lines[1], 14, C["muted"]),
        ])
    arrow(parts, 312, 345, 365, 345)
    arrow(parts, 615, 345, 668, 345)
    arrow(parts, 1018, 345, 1070, 345)
    arrow(parts, 1300, 345, 1352, 345)

    # Worker lanes and host-staged collective.
    parts.extend([
        rect(300, 505, 420, 210, "#10253b", 24, C["blue"], 2),
        text(330, 548, "TP rank 0", 17, C["blue"], 700),
        text(330, 584, "ZE_AFFINITY_MASK=0", 20, C["text"], 700),
        text(330, 620, "B70 GPU 0 · FP8 target shard", 16, C["muted"]),
        text(330, 652, "W8A16 + Xe2 small-M decode", 16, C["accent"], 600),
        rect(880, 505, 420, 210, "#10253b", 24, C["blue"], 2),
        text(910, 548, "TP rank 1", 17, C["blue"], 700),
        text(910, 584, "ZE_AFFINITY_MASK=1", 20, C["text"], 700),
        text(910, 620, "B70 GPU 1 · FP8 target shard", 16, C["muted"]),
        text(910, 652, "W8A16 + Xe2 small-M decode", 16, C["accent"], 600),
        rect(740, 540, 120, 140, C["panel2"], 18, C["warn"], 2),
        text(800, 580, "HOST", 16, C["warn"], 700, "middle"),
        text(800, 612, "oneCCL", 18, C["text"], 700, "middle"),
        text(800, 640, "simple", 14, C["muted"], 600, "middle"),
        text(800, 662, "buffers", 14, C["muted"], 600, "middle"),
    ])
    arrow(parts, 720, 610, 740, 610, "", C["warn"])
    arrow(parts, 860, 610, 880, 610, "", C["warn"])
    parts.extend([
        text(800, 750, "No GPU P2P: collectives are forced off the Level Zero device-IPC path.", 16, C["warn"], 600, "middle"),
        text(800, 782, "Required: per-worker affinity + SYS_PTRACE + oneCCL simple-threshold environment.", 15, C["muted"], 600, "middle"),
    ])

    parts.extend([
        rect(220, 820, 1160, 70, C["panel2"], 18),
        text(260, 850, "Decode dispatch", 15, C["accent"], 700),
        text(260, 878, "W8A16 removes per-linear activation quantization; Xe2 small-M handles decode-sized matrix rows. Hybrid mode retains W8A8 for longer-prompt prefill.", 15, C["text"]),
    ])

    panel(parts, 48, 970, 1504, 170, "Evidence boundary", "Public recipe configuration and measured scope")
    parts.extend([
        text(82, 1072, f'{sample["model"]} · {sample["engine"]}', 15, C["text"], 600),
        text(82, 1100, f'{sample["stack"]["quantization"]} · {sample["stack"]["speculation"]}', 15, C["muted"], 600),
        text(82, 1128, "Measured: C1 fixed-length random-token cells. Not claimed: Cn MTP, natural-content quality, or independent reproduction.", 15, C["warn"], 600),
    ])
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def x_card(lmx: dict) -> str:
    h = 900
    value = lmx["metrics"]["median_client_post_first_output_tps"]
    ttft = lmx["metrics"]["median_client_ttft_ms"]
    workload = lmx["workload"]
    parts = svg_start(
        h,
        "Qwen3.8-27B FP8 TP2 LocalMaxxing result on dual Intel Arc Pro B70",
        "Social lead card for the approved self-reported LocalMaxxing submission. Qwen3.8-27B FP8 W8A16 with MTP8 on two Intel Arc Pro B70 GPUs measured 31.1 client post-first output tokens per second median at p74 g256, concurrency one, three measured samples, with 189.93 milliseconds median TTFT. Platform approval is acceptance of a self-reported submission, not independent reproduction.",
    )
    parts.extend([
        text(64, 58, "QWEN3.8-27B FP8 · 2× INTEL ARC PRO B70 · TP2", 17, C["accent"], 700),
        text(64, 116, "LocalMaxxing speed test", 42, C["text"], 700),
        text(64, 156, "vLLM XPU · FP8 W8A16 · native MTP8", 19, C["muted"], 500),
        rect(1210, 42, 326, 94, C["panel"], 20),
        text(1373, 79, "APPROVED SELF-REPORT", 18, C["blue"], 700, "middle"),
        text(1373, 108, "independent reproduction pending", 13, C["muted"], 600, "middle"),
    ])

    parts.extend([
        rect(64, 214, 1472, 466, C["panel"], 34),
        text(110, 270, "MEDIAN CLIENT POST-FIRST OUTPUT THROUGHPUT", 17, C["blue"], 700),
        text(110, 456, f'{value:.1f}', 146, C["blue"], 700),
        text(510, 448, "tok/s", 42, C["text"], 700),
        text(112, 508, "median across three measured requests", 20, C["muted"], 600),
        line(820, 274, 820, 622, C["line"], 2),
        text(884, 304, "WORKLOAD", 15, C["accent"], 700),
        text(884, 358, f'p{workload["prompt_tokens_actual"]} / g{workload["output_tokens_actual"]}', 42, C["text"], 700),
        text(884, 398, "C1 · n=3 · one discarded warmup", 19, C["muted"], 600),
        text(884, 470, "MEDIAN TTFT", 15, C["accent"], 700),
        text(884, 530, f'{ttft:.2f} ms', 42, C["text"], 700),
        text(884, 574, "client-measured time to first token", 18, C["muted"], 500),
        text(110, 632, "server-default temperature 1.0 · prefix cache disabled with zero query/hit counters", 16, C["muted"], 500),
    ])

    parts.extend([
        rect(64, 724, 1472, 108, C["panel2"], 24, C["line"]),
        text(100, 766, "LocalMaxxing APPROVED = accepted into its self-reported dataset", 17, C["text"], 700),
        text(100, 802, "Serialized-idle C1 · 230 W configured cap per GPU · public JSON receipt · not an independent reproduction", 15, C["muted"], 600),
        text(1500, 872, lmx["submission_id"], 13, C["muted"], 500, "end"),
    ])
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_all() -> dict[Path, str]:
    greedy = [
        raw_cell(EVIDENCE / "bench/n5-p512-g128.json"),
        raw_cell(EVIDENCE / "bench/n5-p1024-g128.json"),
        raw_cell(EVIDENCE / "bench/n5-p2048-g128.json"),
    ]
    sampled = [
        load_json(EVIDENCE / "bench/model-card-sampling-p512-g128.json"),
        load_json(EVIDENCE / "bench/model-card-sampling-p1024-g128.json"),
    ]
    lmx = load_json(EVIDENCE / "localmaxxing-approved-receipt.json")
    return {
        ASSETS / "b70-qwen38-fp8-tp2-dashboard.svg": dashboard(greedy),
        ASSETS / "b70-qwen38-fp8-tp2-sampling-collapse.svg": sampling_collapse(greedy[:2], sampled),
        ASSETS / "b70-qwen38-fp8-tp2-pipeline.svg": pipeline(sampled[0]),
        ASSETS / "b70-qwen38-fp8-tp2-x-card.svg": x_card(lmx),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated SVGs differ from tracked files")
    args = parser.parse_args()
    outputs = render_all()
    failed = False
    for path, content in outputs.items():
        if args.check:
            if not path.exists() or path.read_text() != content:
                print(f"OUT OF DATE: {path.relative_to(ROOT)}", file=sys.stderr)
                failed = True
            else:
                print(f"OK {path.relative_to(ROOT)}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            print(f"WROTE {path.relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
