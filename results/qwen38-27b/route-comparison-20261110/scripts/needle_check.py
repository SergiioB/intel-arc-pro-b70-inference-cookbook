#!/usr/bin/env python3
"""Long-context retrieval at depth: three codes at 10%, 50% and 90% of a code corpus,
one question asks for all three. Model-card sampling (passed as JSON), 3 repeats.
Usage: needle_check.py make PROMPTS_FULL OUT_JSON
       needle_check.py run BASE MODEL NEEDLES_JSON EXTRA_JSON OUT_JSON"""
import json, re, sys, time, urllib.request
CODES = {"ALPHA": "7319-QX", "BRAVO": "2846-LM", "CHARLIE": "5502-TR"}
Q = ("Three release codes, for modules ALPHA, BRAVO and CHARLIE, appear as comments in the source above. "
     "Reply with exactly three lines: ALPHA=<code>, BRAVO=<code>, CHARLIE=<code>. Nothing else.")
def make(full, out):
    P = json.load(open(full)); res = {}
    for depth in ("32k", "128k", "240k"):
        body = P[f"code-{depth}"]["content"].rsplit("\n\n", 1)[0]
        parts, last = [], 0
        for frac, (k, v) in zip((0.1, 0.5, 0.9), CODES.items()):
            cut = body.rfind("\n", 0, int(len(body) * frac)) + 1
            parts += [body[last:cut], f"// release code for module {k}: {v}\n"]; last = cut
        parts.append(body[last:])
        res[depth] = "".join(parts) + "\n\n" + Q
    json.dump(res, open(out, "w"))
    print({d: len(t) for d, t in res.items()})
def run(base, model, needles, extra, out):
    N = json.load(open(needles)); ex = json.loads(extra); rows = []
    for depth, prompt in N.items():
        for rep in range(3):
            body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 64, "seed": 100 + rep}
            body.update(ex)
            req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            t = time.time()
            try:
                r = json.loads(urllib.request.urlopen(req, timeout=3600).read()); text = r["choices"][0]["message"].get("content") or ""; err = None
            except Exception as e:
                text, err, r = "", str(e)[:200], {}
            found = {k: (v in text) for k, v in CODES.items()}
            rows.append({"depth": depth, "rep": rep, "wall_s": round(time.time() - t, 2), "found": sum(found.values()),
                         "detail": found, "text": text[:300], "error": err, "usage": r.get("usage")})
            print(depth, rep, rows[-1]["found"], rows[-1]["wall_s"], err or "", flush=True)
    json.dump(rows, open(out, "w"), indent=1)
if __name__ == "__main__":
    if sys.argv[1] == "make": make(*sys.argv[2:4])
    else: run(*sys.argv[2:7])
