#!/usr/bin/env python3
"""Build the frozen token panels for the as-served int8-prefill KL check (2026-10-14).

P1 prose: 4 wikitext-2 test windows x 2048 tokens (tokenized like the RL-179 harness:
"\n".join(rows), no special tokens). P2 chat: first 4 qbench held-out rows (SC5 artifact,
generated trace, not cal_trace) with >= 1024 response tokens; scored on response positions.
P3 long: one 16384-token wikitext window.
"""
import hashlib, json, os, sys
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

CORPUS = os.environ.get("WIKITEXT_PARQUET") \
    or sys.exit("set WIKITEXT_PARQUET to the wikitext-2 test parquet (test-00000-of-00001.parquet)")
TOK = os.environ.get("SC5_DIR") \
    or sys.exit("set SC5_DIR to the Qwen3.8-27B-exl3-SC_5.00bpw_H6 checkpoint directory")
TRACE = TOK + "/qbench_prompts_gen.json"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "panels.json")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


rows = pq.read_table(CORPUS).column("text").to_pylist()
tok = AutoTokenizer.from_pretrained(TOK)
ids = tok("\n".join(rows), add_special_tokens=False)["input_ids"]
assert len(ids) == 297054, len(ids)

panels = []
for i, s in enumerate((150000, 175000, 200000, 225000)):
    panels.append({"panel": "P1", "row": f"wiki{i}", "start": s, "ids": ids[s:s + 2048], "score_from": 0, "k": 128})
trace = json.loads(Path(TRACE).read_text())
n = 0
for ri, r in enumerate(trace["rows"]):
    if len(r["response_ids"]) < 1024:
        continue
    seq = r["input_ids"] + r["response_ids"]
    # position p's logprob entry predicts token p; score the response tokens only
    panels.append({"panel": "P2", "row": f"chat{ri}", "ids": seq, "score_from": len(r["input_ids"]), "k": 128})
    n += 1
    if n == 4:
        break
panels.append({"panel": "P3", "row": "long0", "start": 240000, "ids": ids[240000:240000 + 16384], "score_from": 0, "k": 32})

meta = {"corpus": CORPUS, "corpus_sha256": sha(CORPUS), "tokens_total": len(ids),
        "trace": TRACE, "trace_sha256": sha(TRACE), "tokenizer_sha256": sha(TOK + "/tokenizer.json")}
OUT.write_text(json.dumps({"meta": meta, "panels": panels}))
print(json.dumps({"out": str(OUT), "sha256": sha(OUT), **meta,
                  "rows": [(p["row"], len(p["ids"]), p["score_from"], p["k"]) for p in panels]}, indent=1))
