#!/usr/bin/env python3
"""Creative quality suite: interactive HTML, SVG and executed code from an OpenAI-compatible server.

  creative_suite.py --base URL --model NAME --out DIR [--max-tokens 8192] [--temperature 0]

Writes DIR/<id>.<ext> (the extracted artifact), DIR/<id>.json (raw answer, timings) and
DIR/summary.json. Code items run against OUR tests (never the model's). HTML/SVG items are
checked afterwards by creative_check.mjs (headless Chrome): JS errors, blank render,
animation, response to input, plus screenshots.
"""
import argparse, json, re, subprocess, sys, time, urllib.request
from pathlib import Path

HTML = "Reply with one complete, self-contained HTML file in a single ```html code block: inline CSS and JS, no external libraries, CDNs, images or network requests."
ITEMS = [
    ("h-solar", "html", "Build an animated solar system on a <canvas>: the Sun and 8 planets on circular orbits with orbital speeds in roughly realistic ratios, a speed slider, a pause/resume button, and clicking a planet shows its name and orbital period in a small panel. " + HTML),
    ("h-life", "html", "Build Conway's Game of Life on a <canvas> grid of at least 60x40 cells: Play/Pause, Step, Clear and Random buttons, click or drag to toggle cells, and a live generation counter. " + HTML),
    ("h-webgl", "html", "Using raw WebGL (no libraries), render a rotating 3D cube with per-face colours and simple directional lighting that rotates on its own and can be rotated by dragging with the mouse. Fall back to a visible message if WebGL is unavailable. " + HTML),
    ("h-piano", "html", "Build a playable two-octave piano keyboard using the Web Audio API: white and black keys drawn with HTML/CSS, play notes on mouse click and on computer keys (A S D F G H J K for white keys of the first octave), with a visual pressed state and a waveform selector (sine, square, sawtooth, triangle). " + HTML),
    ("h-balls", "html", "Build a 2D physics sandbox on a <canvas>: clicking spawns a ball of random size and colour at the cursor, balls fall with gravity, bounce off walls with some energy loss, and collide elastically with each other. Show the ball count and a Reset button. Start with 5 balls already moving. " + HTML),
    ("h-chart", "html", "Build an interactive bar chart (SVG or canvas, no libraries) of these populations in millions: India 1441, China 1425, USA 341, Indonesia 279, Pakistan 245, Nigeria 229, Brazil 217, Bangladesh 174. Buttons sort ascending, descending and alphabetically with an animated transition; hovering a bar shows a tooltip with the exact value. " + HTML),
    ("h-todo", "html", "Build a to-do list app: add tasks with Enter, toggle done, delete, edit by double-click, filters All / Active / Done, a remaining-items counter, and persistence in localStorage. Start with 3 example tasks if storage is empty. " + HTML),
    ("h-fractal", "html", "Build a Mandelbrot set explorer on a <canvas>: smooth colouring with a colourful palette, click to zoom in 2x centred on the click, right-click or a button to zoom out, a Reset button, and an iteration-count slider. " + HTML),
    ("s-pelican", "svg", "Draw an SVG of a pelican riding a bicycle. Reply with one complete standalone <svg> document (with xmlns, viewBox) in a single ```svg code block."),
    ("s-tcp", "svg", "Draw an SVG sequence diagram of the TCP three-way handshake between a Client and a Server: two lifelines, three labelled arrows (SYN, SYN-ACK, ACK) with sequence/ack numbers, and a title. Reply with one complete standalone <svg> document (with xmlns, viewBox) in a single ```svg code block."),
    ("py-lru", "py", "Write a Python class LRUCache with __init__(self, capacity: int), get(self, key) -> value or -1, and put(self, key, value), all O(1), evicting the least recently used key when over capacity. Code only, in one ```python block, no tests."),
    ("py-expr", "py", "Write a Python function evaluate(expr: str) -> float that evaluates arithmetic expressions with + - * / parentheses, unary minus and decimal numbers, respecting precedence, WITHOUT using eval/exec/ast. Code only, in one ```python block, no tests."),
    ("py-bugfix", "py", "This binary search has bugs. Return the corrected function only, in one ```python block:\n\ndef find(xs, target):\n    lo, hi = 0, len(xs)\n    while lo < hi:\n        mid = (lo + hi) / 2\n        if xs[mid] == target:\n            return mid\n        elif xs[mid] < target:\n            lo = mid\n        else:\n            hi = mid - 1\n    return -1\n"),
    ("js-csv", "js", "Write a JavaScript function parseCSVLine(line) that splits one CSV line into an array of fields, handling double-quoted fields that contain commas and escaped double quotes (\"\"). Export it with module.exports = { parseCSVLine }. Code only, in one ```javascript block."),
    ("h-pagoda", "html", "Create a very creative, elaborate and detailed voxel art scene of a Japanese pagoda in a beautiful garden, rendered with colorful voxels. Requirements: build every object out of voxels (small cubes) placed procedurally in code (loops and helper functions, not thousands of hand-written coordinates); a five-tier pagoda with vermilion-red walls, white trim, wide upturned dark roof corners on each tier and a golden spire on top; a garden around it with a curved pond containing swimming koi, an arched red bridge over the pond, glowing stone lanterns, gravel and stone paths, moss patches, bushes, at least six trees with varied green canopies and at least three cherry blossom trees with pink-to-magenta gradient canopies; falling cherry petals drifting on the wind; at least 2000 voxels in total with a varied palette of at least 25 distinct colours. Render with raw WebGL (write your own tiny voxel renderer, no libraries) or a 2D canvas isometric renderer, your choice. Add a soft directional sun with simple per-face shading, a slow automatic orbit camera that pauses while the user drags to rotate, mouse-wheel zoom, and a day/dusk toggle that switches to warm lantern glow lighting. " + HTML, 24576),
    ("h-city", "html", "Build a procedural isometric voxel city: a seeded PRNG (seed input + Regenerate button) generates a city grid with roads, sidewalks, districts of varied buildings (different heights, setbacks, rooftop details, lit window colours), at least two parks with trees and a pond, and a river or waterfront edge with a bridge. Add a Day/Night toggle (night = lit windows and street lamps), hover a building to highlight it and show its height in floors, and pan the camera by dragging. " + HTML, 16384),
    ("h-raymarch", "html", "Using raw WebGL (no libraries), render a raymarched scene in a fragment shader: floating islands with cliffs and grass tops connected by a rope bridge, a waterfall falling into mist, slowly drifting clouds, a warm sun with soft shadows, and distance fog. The camera slowly orbits and can be dragged to rotate, with mouse-wheel zoom; add a Quality toggle that renders at half resolution and upscales. Fall back to a visible message if WebGL is unavailable. " + HTML, 16384),
    ("h-spreadsheet", "html", "Build a working spreadsheet: a scrollable grid of at least 26 columns x 50 rows with column headers and row numbers, click or arrow-key cell selection, typing to enter a value, Enter/Tab to move and Escape to cancel editing. Cells starting with '=' are formulas supporting + - * / and parentheses, cell references like B3, ranges like A1:A5 with SUM(range) and AVG(range), correct dependency recompute when a referenced cell changes, and circular references shown as #CIRC. Highlight the selected cell's row and column headers and show the underlying formula in a formula bar that can also edit it. Pre-fill a small monthly budget example. " + HTML, 16384),
    ("h-sequencer", "html", "Build a Web Audio step sequencer: 4 tracks (kick, snare, hi-hat, synth) x 16 steps as a clickable grid toggling steps on/off with a distinct colour per track, a Play/Stop button, a tempo slider (60-180 BPM) with a swing amount slider, and a playhead sweeping the grid in time. Synthesize every sound with oscillators and noise buffers (no samples, no network): kick = pitch-dropping sine, snare = filtered noise plus a tone, hi-hat = high-passed short noise, synth = a selectable waveform playing an editable note per step. Start with a preset pattern loaded. " + HTML, 16384),
    ("s-metro", "svg", "Draw an SVG metro map of a fictional city: exactly 5 named lines each in a strong distinct colour, at least 30 named stations, at least 6 interchange stations drawn as white double-rings, a river crossing the map, terminus names at line ends, a legend mapping line names to colours, and a title with the city name. Use the classic octolinear style: only horizontal, vertical and 45-degree segments with rounded corners. Reply with one complete standalone <svg> document (with xmlns and viewBox) in a single ```svg code block.", 8192),
    ("py-xml", "py", "Write a minimal XML parser in pure Python (no xml module, no external libs; 're' allowed). Implement class Element with attributes tag (str), attrs (dict), children (list of Element) and text (str: the direct text content with entities decoded), and function parse(xml: str) -> Element that skips an XML declaration and comments, supports nested and self-closing tags, single- or double-quoted attributes, the entities &amp; &lt; &gt; &quot; &apos; and numeric entities like &#65;, and raises ValueError (message containing the line number) on malformed input such as unclosed, mismatched or overlapping tags. Code only, in one ```python block, no tests."),
    ("py-diff", "py", "Implement Myers diff for lines in pure Python (no difflib, no external libs). Write function diff_lines(a: list, b: list) -> list of (op, line) tuples with op one of '=', '-', '+', producing a minimal edit script where within each changed block all '-' entries come before the '+' entries. Also write apply_diff(a: list, script) -> list that reconstructs b exactly from a and the script. Code only, in one ```python block, no tests."),
    ("py-jsonpath", "py", "Write a small JSONPath query engine in pure Python (no external libs; 're' allowed). Function query(obj, path: str) -> list returns all matches for paths supporting: '$' root, '.name', ['name'] and [\"name\"] field access, '[3]' list indexing, '[*]' wildcard over lists and dicts, and '..name' recursive descent. Missing fields or out-of-range indexes contribute no matches and never raise. Code only, in one ```python block, no tests."),
    ("js-dijkstra", "js", "Write a JavaScript function dijkstra(graph, source) where graph is an object like { A: { B: 4, C: 2 }, B: { D: 5 }, C: {}, D: {} } (absent key or empty object means no outgoing edges). Return { dist, prev } where dist maps every node to its shortest distance from source (Infinity when unreachable) and prev maps nodes to their predecessor. Also write buildPath(prev, target) returning the array of nodes from source to target, or null when unreachable. Export all three: module.exports = { dijkstra, buildPath }. Code only, in one ```javascript block."),
]
TESTS = {
    "py-lru": "c=LRUCache(2);c.put(1,1);c.put(2,2);assert c.get(1)==1;c.put(3,3);assert c.get(2)==-1;c.put(4,4);assert c.get(1)==-1;assert c.get(3)==3;assert c.get(4)==4\nc=LRUCache(1);c.put('a',1);c.put('a',2);assert c.get('a')==2;c.put('b',3);assert c.get('a')==-1\n",
    "py-expr": "import math\nfor e,v in [('1+2*3',7),('(1+2)*3',9),('-4+10/4',-1.5),('2*(3+(4-1))/-3',-4),('3.5*2',7),('--3',3),('10-2-3',5),('8/4/2',1)]:\n    assert math.isclose(evaluate(e),v),(e,evaluate(e))\n",
    "py-bugfix": "xs=[1,3,5,7,9,11]\nfor i,x in enumerate(xs): assert find(xs,x)==i\nfor t in [0,2,12]: assert find(xs,t)==-1\nassert find([],1)==-1\nassert find([5],5)==0\n",
    "js-csv": "const {parseCSVLine}=require(process.argv[2]);const a=require('assert');\na.deepStrictEqual(parseCSVLine('a,b,c'),['a','b','c']);\na.deepStrictEqual(parseCSVLine('\"x,y\",z'),['x,y','z']);\na.deepStrictEqual(parseCSVLine('\"he said \"\"hi\"\"\",2'),['he said \"hi\"','2']);\na.deepStrictEqual(parseCSVLine('1,,3'),['1','','3']);\nconsole.log('ok')\n",
    "js-dijkstra": "const {dijkstra,buildPath}=require(process.argv[2]);const a=require('assert');\nconst g={A:{B:1,C:4},B:{C:2,D:5},C:{D:1},D:{},E:{}};\nconst {dist,prev}=dijkstra(g,'A');\na.strictEqual(dist.A,0);a.strictEqual(dist.B,1);a.strictEqual(dist.C,3);a.strictEqual(dist.D,4);\na.strictEqual(dist.E,Infinity);\na.deepStrictEqual(buildPath(prev,'D'),['A','B','C','D']);\na.strictEqual(buildPath(prev,'E'),null);\na.strictEqual(dijkstra(g,'E').dist.E,0);\nconsole.log('ok')\n",
    "py-xml": "e=parse('<a x=\"1\">hi<b/>there</a>')\nassert e.tag=='a' and e.attrs=={'x':'1'} and len(e.children)==1 and e.text=='hithere'\nassert e.children[0].tag=='b' and e.children[0].children==[]\ne2=parse('<?xml version=\"1.0\"?><!-- c --><r><p t=\"v\">&amp;&lt;&gt;&#65;</p></r>')\nassert e2.tag=='r' and e2.children[0].attrs=={'t':'v'} and e2.children[0].text=='&<>A'\nassert parse('<a><b><c/></b></a>').children[0].children[0].tag=='c'\nfor bad in ['<a>','<a></b>','<a><b></a></b>','<a x=1/>','</a>']:\n    try: parse(bad); raise SystemExit('should fail: '+bad)\n    except ValueError: pass\nprint('ok')\n",
    "py-diff": "d=lambda a,b: diff_lines(a.split('\\n'),b.split('\\n'))\nr=d('a\\nb\\nc','a\\nx\\nc'); assert r==[('=','a'),('-','b'),('+','x'),('=','c')], r\nr=d('a','a\\nb\\nc'); assert r==[('=','a'),('+','b'),('+','c')], r\nr=d('a\\nb\\nc','a'); assert r==[('=','a'),('-','b'),('-','c')], r\nr=d('x','y'); assert r==[('-','x'),('+','y')], r\nassert diff_lines([],[])==[]\nimport random\nrandom.seed(7)\nfor _ in range(60):\n    A=[str(random.randint(0,9)) for _ in range(random.randint(0,12))]\n    B=[str(random.randint(0,9)) for _ in range(random.randint(0,12))]\n    s=diff_lines(A,B)\n    assert apply_diff(A,s)==B, (A,B,s)\n    assert all(op in ('=','-','+') for op,_ in s)\nprint('ok')\n",
    "py-jsonpath": "o={'a':{'b':[10,20,{'c':5,'d':{'c':7}}]},'c':1}\nassert query(o,'$.a.b[2].c')==[5]\nassert query(o,'$.a.b[*]')==[10,20,{'c':5,'d':{'c':7}}]\nassert sorted(map(str,query(o,'$..c')))==['1','5','7']\nassert query(o,\"$['a']['b'][0]\")==[10]\nassert query(o,'$.x.y')==[] and query(o,'$.a.b[9]')==[]\nassert query(o,'$.a.b[1]')==[20]\nprint('ok')\n",
}
LANG = {"html": ("html",), "svg": ("svg", "xml", "html"), "py": ("python", "py"), "js": ("javascript", "js")}


def extract(text, kind):
    for lang in LANG[kind]:
        m = re.search(r"```" + lang + r"[^\n]*\n(.*?)```", text, re.S | re.I)
        if m:
            return m.group(1).strip()
    m = re.search(r"```[^\n]*\n(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    if kind == "html" and "<html" in text.lower():
        return text[text.lower().index("<!doctype") if "<!doctype" in text.lower() else text.lower().index("<html"):]
    if kind == "svg" and "<svg" in text:
        return text[text.index("<svg"):text.rindex("</svg>") + 6] if "</svg>" in text else None
    return None


def ask(args, prompt, max_tokens=None):
    budget = max_tokens if max_tokens is not None else args.max_tokens
    body = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
            "temperature": args.temperature, "top_p": 1, "seed": 4242}
    if budget:                      # 0 / unset = no cap: let the server context decide
        body["max_tokens"] = budget
    t = time.time()
    req = urllib.request.Request(args.base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=None).read())   # a long one-shot may take hours
    return r, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=8192); ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--fresh", action="store_true", help="ignore item files left by an interrupted run")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = []
    for item in ITEMS:
        iid, kind, prompt = item[0], item[1], item[2]
        if a.only and iid not in a.only.split(","):
            continue
        done = out / f"{iid}.json"
        if not a.fresh and done.exists() and done.stat().st_size > 200:
            row = json.loads(done.read_text())          # resume after a crash
            row.pop("content", None); row.pop("reasoning", None)
            row.setdefault("resumed", True)
            print(json.dumps({"id": iid, "resumed": True, "finish": row.get("finish"),
                              "extracted": row.get("extracted")}), flush=True)
            summary.append(row)
            continue
        r, wall = ask(a, prompt, None)   # no per-item cap either; server context is the only limit
        msg = r["choices"][0]["message"]; text = msg.get("content") or ""
        art = extract(text, kind)
        ext = {"html": "html", "svg": "svg", "py": "py", "js": "js"}[kind]
        row = {"id": iid, "kind": kind, "prompt": prompt, "wall_s": round(wall, 1), "finish": r["choices"][0].get("finish_reason"),
               "usage": r.get("usage"), "timings": r.get("timings"), "reasoning_chars": len(msg.get("reasoning_content") or ""),
               "extracted": art is not None}
        (out / f"{iid}.json").write_text(json.dumps({**row, "content": text, "reasoning": msg.get("reasoning_content")}, indent=1))
        if art is not None:
            (out / f"{iid}.{ext}").write_text(art)
        if kind in ("py", "js") and art is not None:
            if kind == "py":
                (out / f"{iid}.test.py").write_text(art + "\n\n" + TESTS[iid])
                p = subprocess.run(["timeout", "20", sys.executable, "-I", str(out / f"{iid}.test.py")], capture_output=True, text=True)
            else:
                (out / f"{iid}.test.js").write_text(TESTS[iid])
                p = subprocess.run(["timeout", "20", "node", str(out / f"{iid}.test.js"), str((out / f"{iid}.js").resolve())], capture_output=True, text=True)
            row["tests_pass"] = p.returncode == 0
            row["test_error"] = (p.stderr or "")[-400:]
            # rewrite so the record on disk carries the test result (it is written before the tests run)
            (out / f"{iid}.json").write_text(json.dumps({**row, "content": text, "reasoning": msg.get("reasoning_content")}, indent=1))
        tm = r.get("timings") or {}
        print(json.dumps({k: row.get(k) for k in ("id", "finish", "extracted", "tests_pass", "wall_s")} |
                         {"tokens": (r.get("usage") or {}).get("completion_tokens"), "tok_s": round(tm.get("predicted_per_second", 0), 2)}), flush=True)
        summary.append(row)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
