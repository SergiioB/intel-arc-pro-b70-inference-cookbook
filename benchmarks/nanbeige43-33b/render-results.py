#!/usr/bin/env python3
"""Render the Nanbeige4.3-33B B70 results figure as SVG (stdlib only).

Every value is transcribed from `results/nanbeige43-33b/DUAL-B70-20261028.json`;
the renderer invents no numbers. Rasterise with `rsvg-convert -w 2240 -h 1344`
(host step, not part of the stdlib-only runtime).

Usage: python3 benchmarks/nanbeige43-33b/render-results.py docs/assets/nanbeige43-33b-64k.svg
"""
import sys

INK = "#1a202c"
RED = "#c53030"
BLUE = "#2b6cb0"
GREY = "#8c8c8c"
LIGHT = "#b0b0b0"
GREEN = "#2f855a"
AMBER = "#d69e2e"
MUTED = "#4a5568"
FAINT = "#718096"
BAND = "#edf2f7"

# decode, tokens/s, vs occupied context
DECODE = [
    ("llama.cpp, 1 card (Q4_K_M)", GREY, [(0, 23.48), (8192, 18.79), (32768, 11.02), (65536, 7.14)]),
    ("llama.cpp, 2 cards layer (Q4_K_M)", LIGHT, [(0, 22.81), (32768, 10.58), (131029, 3.87)]),
    ("vLLM, 1 card (W4A16)", BLUE, [(236, 28.87), (945, 27.77), (6189, 27.69), (22546, 25.27)]),
    ("vLLM TP2 (W4A16)", INK, [(238, 41.01), (945, 38.49), (4688, 35.98), (49198, 30.96), (98351, 23.83)]),
    ("vLLM TP2 + ngram spec", RED, [(49198, 37.72), (98351, 30.31)]),
    ("Qwen3.8-27B, llama.cpp 1 card", GREEN, [(0, 20.70), (8192, 19.63), (32768, 16.98), (65536, 14.46)]),
]

PREFILL = [
    ("llama.cpp 1c|cold 64K", 550.95, GREY),
    ("Qwen 1c|cold 64K", 912.45, GREEN),
    ("TP2|49K", 1217.0, INK),
    ("TP2 + ngram|49K", 959.0, RED),
    ("TP2|98K", 633.0, INK),
    ("TP2 + ngram|98K", 694.0, RED),
]

# exact battery: (label, exact, correct-but-not-exact, wrong)
BATTERY = [("Nanbeige4.3-33B|all routes", 6, 1, 1), ("Qwen3.8-27B|Q4_K_M", 7, 0, 1)]

HEADLINE = [
    ("41.0 tok/s", "vLLM TP2, short prompt|+42 % over one card", INK),
    ("30.3 tok/s", "vLLM TP2 + ngram at 98K|+27 % over TP2 alone", RED),
    ("307,072", "token KV ceiling on TP2|46,848 on one card", BLUE),
    ("3.87 tok/s", "llama.cpp at 131K on two cards|capacity route", GREY),
]

W, H = 1600, 960


def esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, body, size=14, color=INK, anchor="start", weight="normal"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="DejaVu Sans, sans-serif" '
        f'font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{esc(body)}</text>'
    )


def main():
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">']
    out.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    out.append(text(30, 52, "Nanbeige4.3-33B on Intel Arc Pro B70 — decode, prefill and quality", 32, INK, weight="bold"))
    out.append(text(30, 82, "One host, one stream per engine, 2026-10-28. GGUF Q4_K_M (llama.cpp 1692f9e50) and W4A16 compressed-tensors (vLLM-XPU 0.31.0). Caps as found: 180 W display card, 230 W compute card.", 15, MUTED))

    # ---------------- panel A: decode vs context ----------------
    x0, y0, x1, y1 = 90, 120, 980, 520
    out.append(text(x0, y0 - 12, "Decode speed vs context depth", 21, INK))
    out.append(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="none" stroke="{MUTED}" stroke-width="1.2"/>')

    def ax(tok):
        return x0 + (tok / 136000.0) * (x1 - x0)

    def ay(tps):
        return y1 - (tps / 58.0) * (y1 - y0)

    # 64K+ band
    out.append(f'<rect x="{ax(65536):.1f}" y="{y0}" width="{x1 - ax(65536):.1f}" height="{y1 - y0}" fill="{BAND}"/>')
    out.append(text(ax(65536) + 10, y1 - 10, "64K+ zone", 13, FAINT))
    for t in range(0, 51, 10):
        out.append(f'<line x1="{x0}" y1="{ay(t):.1f}" x2="{x1}" y2="{ay(t):.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        out.append(text(x0 - 12, ay(t) + 5, str(t), 13, MUTED, anchor="end"))
    for tok in (0, 20000, 40000, 60000, 80000, 100000, 120000):
        out.append(f'<line x1="{ax(tok):.1f}" y1="{y0}" x2="{ax(tok):.1f}" y2="{y1}" stroke="#e2e8f0" stroke-width="1"/>')
        out.append(text(ax(tok), y1 + 22, f"{tok // 1000}K" if tok else "0", 13, MUTED, anchor="middle"))
    out.append(text((x0 + x1) / 2, y1 + 46, "Occupied context (tokens)", 15, MUTED, anchor="middle"))
    for series, color, pts in DECODE:
        coords = " ".join(f"{ax(a):.1f},{ay(b):.1f}" for a, b in pts)
        out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"/>')
        for a, b in pts:
            out.append(f'<circle cx="{ax(a):.1f}" cy="{ay(b):.1f}" r="4.5" fill="{color}"/>')
    out.append(text(x0 - 62, (y0 + y1) / 2, "Decode (tokens/s)", 15, MUTED, anchor="middle").replace(
        "<text", f'<text transform="rotate(-90 {x0 - 62} {(y0 + y1) / 2})"'))

    # legend under panel A, three columns
    lx, ly = x0, y1 + 72
    for i, (series, color, _) in enumerate(DECODE):
        col, row = i % 3, i // 3
        cx = lx + col * 315
        cy = ly + row * 26
        out.append(f'<line x1="{cx}" y1="{cy}" x2="{cx + 26}" y2="{cy}" stroke="{color}" stroke-width="3"/>')
        out.append(f'<circle cx="{cx + 13}" cy="{cy}" r="4.5" fill="{color}"/>')
        out.append(text(cx + 34, cy + 5, series, 13, MUTED))

    # ---------------- headline column ----------------
    hx = 1040
    out.append(text(hx, 120, "Headline", 21, INK))
    y = 175
    for big, small, color in HEADLINE:
        out.append(text(hx, y, big, 32, color, weight="bold"))
        label, sub = small.split("|")
        out.append(text(hx, y + 24, label, 14, MUTED))
        out.append(text(hx, y + 42, sub, 14, MUTED))
        y += 92

    # ---------------- panel B: prefill ----------------
    px0, py0, px1, py1 = 90, 690, 980, 830
    out.append(text(px0, py0 - 12, "Prefill at depth", 21, INK))
    out.append(f'<line x1="{px0}" y1="{py1}" x2="{px1}" y2="{py1}" stroke="{MUTED}" stroke-width="1.2"/>')
    vmax = 1400.0
    for t in range(0, 1400 + 1, 200):
        yy = py1 - (t / vmax) * (py1 - py0)
        out.append(f'<line x1="{px0}" y1="{yy:.1f}" x2="{px1}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        out.append(text(px0 - 12, yy + 5, str(t), 13, MUTED, anchor="end"))
    slot = (px1 - px0) / len(PREFILL)
    for i, (label, value, color) in enumerate(PREFILL):
        bw = slot * 0.62
        bx = px0 + slot * i + (slot - bw) / 2
        by = py1 - (value / vmax) * (py1 - py0)
        out.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{py1 - by:.1f}" fill="{color}"/>')
        out.append(text(bx + bw / 2, by - 8, f"{value:.0f}", 14, INK, anchor="middle", weight="bold"))
        top, bottom = label.split("|")
        out.append(text(bx + bw / 2, py1 + 20, top, 13, MUTED, anchor="middle"))
        out.append(text(bx + bw / 2, py1 + 37, bottom, 13, FAINT, anchor="middle"))
    out.append(text(px0 - 62, (py0 + py1) / 2, "Prefill (tokens/s)", 15, MUTED, anchor="middle").replace(
        "<text", f'<text transform="rotate(-90 {px0 - 62} {(py0 + py1) / 2})"'))

    # ---------------- panel C: quality ----------------
    qx0, qy0, qy1 = 1040, 690, 830
    out.append(text(qx0, qy0 - 12, "Exact-answer battery (8 cases)", 21, INK))
    unit = 16
    for i, (label, exact, verbose, wrong) in enumerate(BATTERY):
        bx = qx0 + 30 + i * 240
        base = qy1
        for count, color in ((exact, GREEN), (verbose, AMBER), (wrong, RED)):
            h = count * unit
            base -= h
            out.append(f'<rect x="{bx}" y="{base}" width="130" height="{h}" fill="{color}"/>')
        top, bottom = label.split("|")
        out.append(text(bx + 65, qy1 + 20, top, 14, MUTED, anchor="middle"))
        out.append(text(bx + 65, qy1 + 37, bottom, 13, FAINT, anchor="middle"))
    out.append(f'<line x1="{qx0 + 20}" y1="{qy1}" x2="{qx0 + 500}" y2="{qy1}" stroke="{MUTED}" stroke-width="1.2"/>')
    for i, (name, color) in enumerate((("exact correct", GREEN), ("correct, not exact", AMBER), ("wrong", RED))):
        lx2 = qx0 + 30 + i * 165
        out.append(f'<rect x="{lx2}" y="{qy1 + 52}" width="14" height="14" fill="{color}"/>')
        out.append(text(lx2 + 20, qy1 + 64, name, 13, MUTED))
    out.append(text(qx0 + 20, qy1 + 100, "Qwen's miss is the int4 canary sum(i*i for i in range(4)) → 30", 12, MUTED))
    out.append(text(qx0 + 20, qy1 + 116, "(14 only with thinking on); Nanbeige answers 14 everywhere.", 12, MUTED))

    out.append(text(30, 956, "Sources: results/nanbeige43-33b/DUAL-B70-20261028.json · docs/nanbeige43-33b/ · provisional: one host, one operator, not independently reproduced", 12, FAINT))
    out.append("</svg>")
    target = sys.argv[1] if len(sys.argv) > 1 else "nanbeige43-33b.svg"
    with open(target, "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", target)


if __name__ == "__main__":
    main()
