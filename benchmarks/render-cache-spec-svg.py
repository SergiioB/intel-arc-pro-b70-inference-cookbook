#!/usr/bin/env python3
"""Render the matched 128K cache/spec summary as a shareable SVG table."""
import argparse
import html
import json
from pathlib import Path


def text(x, y, value, css="body", anchor="start"):
    return (
        f'<text x="{x}" y="{y}" class="{css}" text-anchor="{anchor}">'
        f'{html.escape(str(value))}</text>')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    rows = data["rows"]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    best = {
        "cold_ttfc": min(row["cold"]["ttfc_median_s"] for row in rows),
        "cold_tps": max(row["cold"]["client_post_first_tps_median"] for row in rows),
        "resident_ttfc": min(row["resident"]["ttfc_median_s"] for row in rows),
        "resident_e2e": min(row["resident"]["e2e_median_s"] for row in rows),
    }
    mode_names = {"no-spec": "No spec", "mtp1": "MTP1", "mtp2": "MTP2", "mtp4": "MTP4"}

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" role="img" aria-labelledby="title desc">',
        '<title id="title">Intel Arc Pro B70 exact 128K cache and speculative decoding matrix</title>',
        '<desc id="desc">Eight matched rows compare no speculative decoding, MTP1, MTP2, and MTP4 with prefix caching on and off for cold exact 128K and resident 120K Pi sessions.</desc>',
        '<style>',
        'text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-variant-numeric:tabular-nums}',
        '.title{font-size:52px;font-weight:760;fill:#f8fafc;letter-spacing:-1.4px}',
        '.subtitle{font-size:23px;font-weight:450;fill:#a9b7c9}',
        '.kicker{font-size:17px;font-weight:700;fill:#4fd1c5;letter-spacing:2px}',
        '.head{font-size:17px;font-weight:700;fill:#a9b7c9;letter-spacing:.3px}',
        '.body{font-size:23px;font-weight:560;fill:#f8fafc}',
        '.muted{font-size:19px;font-weight:500;fill:#a9b7c9}',
        '.best{font-size:23px;font-weight:760;fill:#86efac}',
        '.pill{font-size:16px;font-weight:760;fill:#08111f;letter-spacing:.7px}',
        '.foot{font-size:17px;font-weight:480;fill:#a9b7c9}',
        '</style>',
        '<rect width="1600" height="1000" fill="#08111f"/>',
        '<rect x="42" y="42" width="1516" height="916" rx="26" fill="#101c2d"/>',
        text(78, 96, 'REAL-WORLD PI · MATCHED C1 MATRIX', 'kicker'),
        text(78, 158, 'What cache + MTP change at exact 128K', 'title'),
        text(78, 202, 'Same image, model, prompts, p130944/g128, scheduler budget 8192 · median of 5 measured requests', 'subtitle'),
        '<rect x="68" y="232" width="1464" height="58" rx="12" fill="#16243a"/>',
        text(88, 268, 'Mode', 'head'),
        text(245, 268, 'Cache', 'head'),
        text(415, 258, 'Cold TTFC', 'head', 'middle'),
        text(415, 280, 'lower is better', 'muted', 'middle'),
        text(650, 258, 'Cold output', 'head', 'middle'),
        text(650, 280, 'tok/s after first', 'muted', 'middle'),
        text(885, 258, 'Resident TTFC', 'head', 'middle'),
        text(885, 280, 'lower is better', 'muted', 'middle'),
        text(1125, 258, 'Resident E2E', 'head', 'middle'),
        text(1125, 280, 'lower is better', 'muted', 'middle'),
        text(1390, 258, 'Resident reuse', 'head', 'middle'),
        text(1390, 280, 'tokens', 'muted', 'middle'),
    ]

    y0 = 302
    row_h = 68
    for index, row in enumerate(rows):
        y = y0 + index * row_h
        fill = '#0d2530' if row['cache'] == 'on' else '#111b2a'
        svg.append(f'<rect x="68" y="{y}" width="1464" height="58" rx="10" fill="{fill}"/>')
        svg.append(text(88, y + 37, mode_names[row['mode']]))
        pill_fill = '#4fd1c5' if row['cache'] == 'on' else '#94a3b8'
        svg.append(f'<rect x="238" y="{y + 14}" width="70" height="32" rx="16" fill="{pill_fill}"/>')
        svg.append(text(273, y + 36, row['cache'].upper(), 'pill', 'middle'))

        cold_ttfc = row['cold']['ttfc_median_s']
        cold_tps = row['cold']['client_post_first_tps_median']
        resident_ttfc = row['resident']['ttfc_median_s']
        resident_e2e = row['resident']['e2e_median_s']
        reused = row['resident']['reused_tokens_median']
        values = [
            (415, f'{cold_ttfc:.3f} s', 'best' if cold_ttfc == best['cold_ttfc'] else 'body'),
            (650, f'{cold_tps:.2f}', 'best' if cold_tps == best['cold_tps'] else 'body'),
            (885, f'{resident_ttfc:.3f} s', 'best' if resident_ttfc == best['resident_ttfc'] else 'body'),
            (1125, f'{resident_e2e:.3f} s', 'best' if resident_e2e == best['resident_e2e'] else 'body'),
            (1390, f'{reused:,.0f}', 'body'),
        ]
        for x, value, css in values:
            svg.append(text(x, y + 38, value, css, 'middle'))

    svg.extend([
        '<line x1="78" y1="862" x2="1522" y2="862" stroke="#26364a" stroke-width="2"/>',
        text(78, 902, 'Resident session: exact 120,000-token preparation + five changed ~120,148-token follow-ups.', 'foot'),
        text(78, 932, 'Cache-on reused 118,592–119,680 tokens; cache-off used the explicit negative flag and reused 0.', 'foot'),
        text(1522, 902, 'Intel Arc Pro B70 32 GB · vLLM XPU', 'foot', 'end'),
        text(1522, 932, 'E2 self-reported · not independently reproduced', 'foot', 'end'),
        '</svg>',
    ])
    args.output.write_text('\n'.join(svg) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
