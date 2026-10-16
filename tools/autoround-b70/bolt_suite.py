#!/usr/bin/env python3
"""Bolt 74B-A9B confirmation suite against any OpenAI-compatible server (vLLM or llama-server).

usage: bolt_suite.py <outdir> <test> [--url U] [--model M] [--n N]
tests:
  vision     6 synthetic images x 2 prompt orders (image-first, text-first), keyword checks
  gsm8k      first N questions of the GSM8K test split (no selection), thinking off, greedy;
             model must end with 'Answer: <number>'; exact numeric match
  multiturn  4-turn conversation that must recall facts from turns 1-2
  cacheeq    gate-style prompts sent twice (second may hit the prefix cache): texts must match
  needle     needle at 10/50/90% depth in wikitext, sizes from --sizes (tokens)
  conc       4 concurrent gate prompts vs the same prompts serial: all must pass
All requests: temperature 0, top_k 1, seed 0, chat_template_kwargs enable_thinking=False.

Inputs (env):
  GSM8K_PARQUET  openai/gsm8k revision 740312add88f781978c0658806c59bc2815b9866,
                 main/test-00000-of-00001.parquet (gsm8k test; needs pyarrow)
  WIKITEXT       WikiText-2 raw test file wiki.test.raw (cacheeq, needle)
vision needs Pillow; it draws its own images.
"""
import argparse, base64, concurrent.futures as cf, io, json, os, pathlib, re, time, urllib.request, uuid

GSM = os.environ.get("GSM8K_PARQUET", "test-00000-of-00001.parquet")
WIKI = os.environ.get("WIKITEXT", "wiki.test.raw")
NEEDLE = "The secret maintenance code for the lighthouse on Isla Brava is 7316-KESTREL."
QUESTION = "What is the secret maintenance code for the lighthouse on Isla Brava? Reply with the code only."


class Client:
    def __init__(self, url, model):
        self.url, self.model = url.rstrip("/"), model

    def post(self, path, body, timeout=3600):
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        d["_wall_s"] = round(time.monotonic() - t0, 3)
        return d

    def chat(self, messages, max_tokens=256, **kw):
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                "top_k": 1, "seed": 0, "chat_template_kwargs": {"enable_thinking": False}}
        body.update(kw)
        d = self.post("/v1/chat/completions", body)
        return (d["choices"][0]["message"].get("content") or "").strip(), d

    def ntok(self, text):
        try:  # vLLM
            return self.post("/tokenize", {"model": self.model, "prompt": text})["count"]
        except Exception:  # llama-server
            return len(self.post("/tokenize", {"content": text})["tokens"])


def png_b64(draw_fn):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (512, 512), "white")
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
    except Exception:
        font = ImageFont.load_default()
    draw_fn(d, font)
    buf = io.BytesIO(); im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


VISION = [
    ("red_circle_42", lambda d, f: (d.ellipse((96, 96, 416, 416), fill=(220, 20, 20)), d.text((256, 256), "42", fill="white", font=f, anchor="mm")),
     "What shape and color is in this image, and what number is written? Answer briefly.", ["red", "circle", "42"]),
    ("blue_square", lambda d, f: d.rectangle((120, 120, 392, 392), fill=(20, 60, 220)),
     "What shape is in this image and what color is it? Answer briefly.", ["blue", "square"]),
    ("three_triangles", lambda d, f: [d.polygon([(x, 380), (x + 70, 380), (x + 35, 250)], fill=(20, 160, 40)) for x in (60, 220, 380)],
     "How many triangles are in this image, and what color are they? Answer briefly.", ["3|three", "green"]),
    ("text_hello", lambda d, f: d.text((256, 256), "HELLO 2026", fill="black", font=f, anchor="mm"),
     "What text is written in this image? Reply with the text only.", ["hello", "2026"]),
    ("yellow_star_count", lambda d, f: [d.ellipse((x, 200, x + 80, 280), fill=(240, 200, 0)) for x in (40, 160, 280, 400)],
     "How many yellow circles are in this image? Reply with the number only.", ["4|four"]),
    ("arrow_right", lambda d, f: (d.rectangle((80, 230, 330, 282), fill="black"), d.polygon([(330, 170), (330, 342), (450, 256)], fill="black")),
     "Which direction does the arrow in this image point: left, right, up or down? One word.", ["right"]),
]


def hit(text, keys):
    t = text.lower()
    return all(any(re.search(r"\b" + alt + r"\b", t) for alt in k.split("|")) for k in keys)


def t_vision(c, args):
    rows = []
    for name, fn, q, keys in VISION:
        img = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png_b64(fn)}}
        txt = {"type": "text", "text": q}
        for order, content in (("image_first", [img, txt]), ("text_first", [txt, img])):
            a, _ = c.chat([{"role": "user", "content": content}], 128)
            rows.append({"case": name, "order": order, "pass": hit(a, keys), "text": a})
            print(f"{name:18s} {order:11s} pass={rows[-1]['pass']} {a[:80]!r}", flush=True)
    for order in ("image_first", "text_first"):
        n = sum(r["pass"] for r in rows if r["order"] == order)
        print(f"VISION {order}: {n}/{len(VISION)}")
    return rows


def t_gsm8k(c, args):
    import pyarrow.parquet as pq
    t = pq.read_table(GSM).to_pylist()[: args.n]
    rows, ok = [], 0
    for i, ex in enumerate(t):
        want = ex["answer"].split("####")[-1].strip().replace(",", "")
        q = ex["question"] + "\n\nSolve step by step, then end with a final line 'Answer: <number>'."
        a, d = c.chat([{"role": "user", "content": q}], 768)
        m = re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", a)
        got = m[-1].replace(",", "").rstrip(".") if m else None
        p = got is not None and float(got) == float(want)
        ok += p
        rows.append({"i": i, "want": want, "got": got, "pass": p, "completion_tokens": d["usage"]["completion_tokens"], "text": a})
        if (i + 1) % 25 == 0:
            print(f"gsm8k {i + 1}/{len(t)}: {ok} correct", flush=True)
    print(f"GSM8K: {ok}/{len(t)} = {100 * ok / len(t):.1f}% (first {len(t)} of the test split, thinking off)")
    return rows


def t_multiturn(c, args):
    msgs = [{"role": "user", "content": "My name is Ilaria and my cat is called Pistacchio. Just say OK."}]
    rows = []
    script = [
        ("I live in the city of Valencia and I was born in 1991. Just say OK.", None),
        ("What is my cat's name? One word.", ["pistacchio"]),
        ("How old will I turn in 2030? Reply with the number only.", ["39"]),
        ("Which city do I live in, and what is my name? Answer in one short sentence.", ["valencia", "ilaria"]),
    ]
    a, _ = c.chat(msgs, 32); msgs.append({"role": "assistant", "content": a})
    for q, keys in script:
        msgs.append({"role": "user", "content": q})
        a, _ = c.chat(msgs, 64); msgs.append({"role": "assistant", "content": a})
        if keys:
            rows.append({"q": q, "pass": hit(a, keys), "text": a})
            print(f"multiturn pass={rows[-1]['pass']} {q[:40]!r} -> {a[:60]!r}", flush=True)
    print(f"MULTITURN: {sum(r['pass'] for r in rows)}/{len(rows)}")
    return rows


GATE = ["What is 17 * 23? Reply with the number only.",
        "What is the capital of Australia? One word.",
        "How many letters r are in the word strawberry? Reply with the number only.",
        "Write a Python function is_prime(n) and nothing else.",
        "Translate into Spanish: 'The cat sleeps'. Reply with the translation only.",
        "What does this Python print? Reply with the number only.\nprint(sum(i*i for i in range(4)))"]
GATE_KEYS = [["391"], ["canberra"], ["3"], ["def is_prime"], ["gato", "duerme"], ["14"]]


def t_cacheeq(c, args):
    long_prefix = open(WIKI).read()[:20000]
    rows = []
    for q in GATE:
        msgs = [{"role": "system", "content": "Reference notes (ignore unless asked):\n" + long_prefix},
                {"role": "user", "content": q}]
        a1, _ = c.chat(msgs); a2, _ = c.chat(msgs)
        rows.append({"q": q, "identical": a1 == a2, "first": a1, "second": a2})
        print(f"cacheeq identical={a1 == a2} {q[:40]!r} {a1[:40]!r}", flush=True)
    print(f"CACHEEQ: {sum(r['identical'] for r in rows)}/{len(rows)} identical (2nd request may hit the prefix cache)")
    return rows


def t_needle(c, args):
    body = open(WIKI).read()
    rows = []
    for target in args.sizes:
        for depth in [float(x) for x in os.environ.get("NEEDLE_DEPTHS", "0.1,0.5,0.9").split(",")]:
            lo, hi = 1000, min(len(body), target * 6)
            nonce = f"[run {uuid.uuid4()}]\n"

            def doc(n):
                cut = body[:n]; k = int(len(cut) * depth)
                return nonce + cut[:k] + "\n" + NEEDLE + "\n" + cut[k:] + "\n\n" + QUESTION
            while hi - lo > 2000:  # chars -> tokens, binary search to ~target - 600 tokens (chat template + reply)
                mid = (lo + hi) // 2
                if c.ntok(doc(mid)) <= target - 600:
                    lo = mid
                else:
                    hi = mid
            prompt = doc(lo)
            a, d = c.chat([{"role": "user", "content": prompt}], 32)
            pt = d["usage"]["prompt_tokens"]
            rows.append({"target": target, "depth": depth, "prompt_tokens": pt, "wall_s": d["_wall_s"],
                         "answer": a, "pass": "7316-KESTREL" in a})
            print(f"needle {pt:7d} tok depth {depth:.1f} pass={rows[-1]['pass']} wall {d['_wall_s']}s {a[:40]!r}", flush=True)
    print(f"NEEDLE: {sum(r['pass'] for r in rows)}/{len(rows)}")
    return rows


def t_conc(c, args):
    with cf.ThreadPoolExecutor(4) as ex:
        t0 = time.monotonic()
        res = list(ex.map(lambda q: c.chat([{"role": "user", "content": q}], 256), GATE[:4]))
        wall = time.monotonic() - t0
    serial = [c.chat([{"role": "user", "content": q}], 256)[0] for q in GATE[:4]]
    rows = []
    for q, k, (a, d), s in zip(GATE[:4], GATE_KEYS[:4], res, serial):
        rows.append({"q": q, "pass": hit(a, k), "same_as_serial": a == s, "text": a})
        print(f"conc pass={rows[-1]['pass']} same_as_serial={a == s} {a[:50]!r}", flush=True)
    gen = sum(d["usage"]["completion_tokens"] for _, d in res)
    print(f"CONC4: {sum(r['pass'] for r in rows)}/4 pass, {sum(r['same_as_serial'] for r in rows)}/4 identical to serial, "
          f"{gen} tokens in {wall:.1f}s wall")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir"); ap.add_argument("test")
    ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--model", default="nemotron-bolt")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",")], default=[32000, 128000, 250000])
    a = ap.parse_args()
    out = pathlib.Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    rows = globals()["t_" + a.test](Client(a.url, a.model), a)
    (out / f"{a.test}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
