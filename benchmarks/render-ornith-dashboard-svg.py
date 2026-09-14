#!/usr/bin/env python3
"""Render the Ornith MixedCal-v2 B70 dashboard from dashboard-source.json.

Numbers are the published CLAIMS.md cells only. Self-reported E2 banner is
on-canvas. LocalMaxxing APPROVED is accepted self-report, not independent
reproduction.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

W, H = 1600, 1640
C = {
    "bg": "#07111f",
    "panel": "#0f1d30",
    "panel2": "#0b1829",
    "line": "#263a53",
    "text": "#f5f8fc",
    "muted": "#9caec4",
    "accent": "#55d6be",
    "warn": "#ffb45c",
    "nospec": "#aeb9c7",
    "mtp1": "#55d6be",
    "mtp2": "#68a7ff",
    "mtp4": "#ffb45c",
    "default": "#55d6be",
}


def esc(v):
    return html.escape(str(v))


def t(x, y, v, size=24, color=None, weight=400, anchor="start"):
    color = color or C["text"]
    return (
        f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" '
        f'font-family="ui-sans-serif, system-ui, sans-serif">{esc(v)}</text>'
    )


def panel(x, y, w, h, eyebrow, title):
    return [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="28" fill="{C["panel"]}"/>',
        t(x + 34, y + 40, eyebrow.upper(), 16, C["accent"], 700),
        t(x + 34, y + 78, title, 26, C["text"], 700),
    ]


def fmt(v, kind="dec"):
    if v is None:
        return "—"
    if kind == "cold":
        return f"{v:,.0f}"
    if kind == "lmx_prefill":
        v = float(v)
        return f"{v:,.1f}" if v != int(v) else f"{v:,.0f}"
    if kind == "lmx_out":
        return f"{float(v):.1f}"
    return f"{v:.2f}"


def kind_color(kind):
    return C.get(kind, C["text"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    s = json.loads(Path(args.source).read_text())

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
        '<title id="title">Ornith 1.5 MixedCal-v2 on Intel Arc Pro B70 — self-reported C1 card</title>',
        '<desc id="desc">Client post-first decode and cold-input rates for Ornith MixedCal-v2 GPTQ INT4 on one Intel Arc Pro B70. Default serve is MTP1 plus DraftINT4. MTP4 is slower than no-spec. Self-reported E2. LocalMaxxing APPROVED means accepted self-report, not independent reproduction.</desc>',
        f'<rect width="{W}" height="{H}" fill="{C["bg"]}"/>',
        t(48, 52, "INTEL ARC PRO B70 · vLLM XPU", 16, C["accent"], 700),
        t(48, 96, s["title"], 36, C["text"], 700),
        t(48, 130, s["subtitle"], 18, C["muted"]),
        f'<rect x="1180" y="28" width="372" height="88" rx="18" fill="{C["panel"]}"/>',
        t(1366, 62, s["banner"], 20, C["accent"], 700, "middle"),
        t(1366, 92, s["banner_sub"], 14, C["muted"], 600, "middle"),
    ]

    rows = s["decode"]
    parts += panel(48, 168, 760, 548, "C1 decode", "Client post-first tok/s · n=5")
    left, top = 82, 270
    parts.append(f'<rect x="{left}" y="{top}" width="692" height="40" rx="12" fill="{C["panel2"]}"/>')
    parts.append(t(left + 16, top + 27, "Mode", 16, C["muted"], 600))
    parts.append(t(left + 360, top + 27, "p512/g128", 16, C["muted"], 600, "middle"))
    parts.append(t(left + 510, top + 27, "p8192/g128", 16, C["muted"], 600, "middle"))
    parts.append(t(left + 640, top + 27, "accept", 16, C["muted"], 600, "middle"))
    for i, row in enumerate(rows):
        cy = top + 44 + 46 * i
        if i % 2:
            parts.append(
                f'<rect x="{left}" y="{cy}" width="692" height="46" fill="#12243a" opacity="0.5"/>'
            )
        parts.append(
            f'<circle cx="{left+16}" cy="{cy+23}" r="5" fill="{kind_color(row.get("kind"))}"/>'
        )
        parts.append(t(left + 30, cy + 29, row["label"], 14, C["text"], 600))
        winner = bool(row.get("winner"))
        col = C["accent"] if winner else C["text"]
        wt = 700 if winner else 500
        parts.append(t(left + 360, cy + 29, fmt(row.get("p512")), 17, col, wt, "middle"))
        parts.append(t(left + 510, cy + 29, fmt(row.get("p8192")), 17, col, wt, "middle"))
        acc = row.get("accept")
        parts.append(
            t(
                left + 640,
                cy + 29,
                "—" if acc is None else f"{acc:.1f}%",
                15,
                C["muted"],
                500,
                "middle",
            )
        )

    parts += panel(832, 168, 720, 548, "Cold input", "Endpoint tokens / client TTFT")
    left, top = 866, 270
    parts.append(
        t(
            left,
            top - 8,
            "230 W is configured cap. Combined MTP prefill ≠ no-spec 9.7k (first token in TTFT).",
            13,
            C["warn"],
        )
    )
    for i, row in enumerate(s["cold"]):
        cy = top + 20 + 50 * i
        parts.append(
            f'<rect x="{left}" y="{cy}" width="652" height="44" rx="12" fill="{C["panel2"]}"/>'
        )
        parts.append(t(left + 18, cy + 28, row["label"], 14, C["muted"]))
        hot = row.get("cap") == 230
        parts.append(
            t(
                left + 630,
                cy + 28,
                fmt(row["value"], "cold"),
                18,
                C["warn"] if hot else C["text"],
                700,
                "end",
            )
        )

    parts += panel(
        48, 740, 1504, 250, "Exact-token capacity MixedCal-v2", "Calibrated with /tokenize · finish=length"
    )
    for i, row in enumerate(s["capacity"]):
        x = 82 + 294 * i
        parts.append(f'<rect x="{x}" y="834" width="278" height="132" rx="18" fill="{C["panel2"]}"/>')
        parts.append(t(x + 18, 866, row["serve"], 15, C["muted"], 600))
        parts.append(
            t(
                x + 18,
                914,
                fmt(row["value"]),
                32,
                C["accent"] if row.get("mtp") else C["text"],
                700,
            )
        )
        parts.append(t(x + 18, 946, f"{row['cell']} · {row['note']}", 14, C["muted"]))

    parts += panel(
        48,
        1014,
        1504,
        292,
        "LocalMaxxing speed-test",
        "APPROVED = accepted self-report, not independent reproduction",
    )
    left, top = 82, 1110
    parts.append(f'<rect x="{left}" y="{top}" width="1436" height="40" rx="12" fill="{C["panel2"]}"/>')
    parts.append(t(left + 16, top + 27, "Serve", 16, C["muted"], 600))
    parts.append(t(left + 620, top + 27, "prompt / gen", 16, C["muted"], 600))
    parts.append(t(left + 900, top + 27, "tokSOut", 16, C["muted"], 600, "middle"))
    parts.append(t(left + 1100, top + 27, "tokSPrefill", 16, C["muted"], 600, "middle"))
    parts.append(t(left + 1320, top + 27, "id", 16, C["muted"], 600, "middle"))
    for i, row in enumerate(s["lmx"]):
        cy = top + 44 + 46 * i
        if i % 2:
            parts.append(
                f'<rect x="{left}" y="{cy}" width="1436" height="46" fill="#12243a" opacity="0.5"/>'
            )
        winner = i == 0
        col = C["accent"] if winner else C["text"]
        wt = 700 if winner else 500
        parts.append(t(left + 16, cy + 29, row["serve"], 15, C["text"], 600))
        parts.append(t(left + 620, cy + 29, row["prompt"], 15, C["muted"]))
        parts.append(t(left + 900, cy + 29, fmt(row["tokSOut"], "lmx_out"), 18, col, wt, "middle"))
        parts.append(
            t(left + 1100, cy + 29, fmt(row["tokSPrefill"], "lmx_prefill"), 18, col, wt, "middle")
        )
        parts.append(t(left + 1320, cy + 29, row["id"], 12, C["muted"], 500, "middle"))

    stack = s["stack"]
    rtn = s["rtn"]
    parts += [
        t(48, 1324, "Contract", 16, C["accent"], 700),
        t(
            48,
            1354,
            f"Image digest {stack['image_digest']}  ·  {stack['vllm']}  ·  kernels {stack['kernels']}  ·  {stack['moe']}",
            15,
            C["muted"],
        ),
        t(48, 1378, stack["timing"], 15, C["muted"]),
        t(
            48,
            1402,
            f"MixedCal-v2 RTN {rtn['mixedcal_pct']}% ({rtn['mixedcal_count']}/{rtn['total']}) vs WikiText {rtn['wikitext_pct']}%. Speed at 150 W is parity, not a MixedCal tok/s win.",
            15,
            C["muted"],
        ),
        t(
            48,
            1438,
            "Default serve: MTP1 + DraftINT4. MTP4 is slower than no-spec. Prefill lever is configured 230 W (~9.7k no-spec), not MixedCal.",
            16,
            C["text"],
        ),
        t(
            48,
            1474,
            "Combined tokSPrefill 9073 includes MTP first-token work inside TTFT, so it is not the no-spec 9780 cell.",
            15,
            C["warn"],
            600,
        ),
        t(
            48,
            1510,
            "Self-reported E2 with raw evidence · not independently reproduced · greedy diagnostic · C1 only",
            16,
            C["warn"],
            600,
        ),
        "</svg>",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts) + "\n")
    print("WROTE", out)


if __name__ == "__main__":
    main()
