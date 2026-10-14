#!/usr/bin/env python3
"""v2 panels: every qbench held-out row (SC5 artifact, generated trace, not cal_trace) with >= 1024
response tokens, scored on response positions, top-128. Rows 0,1,2,4 were v1's P2; all are kept so
v1 and v2 can be compared on the overlap.
"""
import hashlib, json, os, sys
from pathlib import Path

TRACE = (os.environ.get("SC5_DIR")
         or sys.exit("set SC5_DIR to the Qwen3.8-27B-exl3-SC_5.00bpw_H6 checkpoint directory")) \
    + "/qbench_prompts_gen.json"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "panels_v2.json")
trace = json.loads(Path(TRACE).read_text())
panels = [{"panel": "P2", "row": f"chat{ri}", "ids": r["input_ids"] + r["response_ids"],
           "score_from": len(r["input_ids"]), "k": 128}
          for ri, r in enumerate(trace["rows"]) if len(r["response_ids"]) >= 1024]
meta = {"trace": TRACE, "trace_sha256": hashlib.sha256(Path(TRACE).read_bytes()).hexdigest()}
OUT.write_text(json.dumps({"meta": meta, "panels": panels}))
print(json.dumps({"out": str(OUT), "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(), **meta,
                  "rows": len(panels), "scored": sum(len(p["ids"]) - p["score_from"] for p in panels),
                  "max_len": max(len(p["ids"]) for p in panels)}))
