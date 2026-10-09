#!/usr/bin/env python3
"""Extra teacher samples ("siblings") for the prompts that produced a top-1 lens hit, so that hits can be paired with a non-hit
completion of the same prompt (lenssel/make_sib_input.py). Prompts whose position 0 (the last prompt token, identical for every
completion of the prompt) already reads the target at top-1 are skipped: no completion of those can be a non-hit.
Generate 16 samples per sibling prompt (scripts/generate.py --n 16 --seed 1) and score them like the main pool.
usage: python 06_addendum_lens_selected_sft/make_sib_input.py OUT.jsonl PROMPTS.jsonl[,PROMPTS2.jsonl] SCORE.pt [SCORE.pt ...]
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import scores  # noqa: E402

out, pfiles, sfiles = sys.argv[1], sys.argv[2].split(","), sys.argv[3:]
pos0 = set()
hit = {r["id"] for r in scores.load(sfiles, pos0=pos0) if r["passes"] and r["best"] <= 1}
keep = hit - pos0
n = 0
with open(out, "w") as f:
    for fn in pfiles:
        for line in open(fn):
            r = json.loads(line)
            if r["id"] in keep:
                f.write(json.dumps({"id": "sib:" + r["id"], "messages": r["messages"], "meta": r["meta"]}) + "\n")
                n += 1
print(f"hit prompts {len(hit)}, of which position-0 top-1 {len(hit & pos0)}; wrote {n}")
