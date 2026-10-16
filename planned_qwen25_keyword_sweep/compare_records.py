#!/usr/bin/env python3
"""Compare kw_lens.py records with reference records (e.g. the original run's) for the completions both contain: the
filter flag, own-set best ranks per scope, own-set per-position ranks, sparse best ranks of every set, random-set chance
fractions and per-layer top-1 token ids. Prints the number of equal / different values per field.

usage: python planned_qwen25_keyword_sweep/compare_records.py NEW.jsonl "REF_GLOB"
"""
import glob
import json
import sys
from collections import Counter

new = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1])}
ref = {}
for p in glob.glob(sys.argv[2]):
    for line in open(p):
        r = json.loads(line)
        if r["id"] in new:
            ref[r["id"]] = r
eq, ne = Counter(), Counter()
for i, r in ref.items():
    n = new[i]
    for k in ("passed", "n_pos", "e1", "own", "own_pos", "best_sparse", "chance", "top1", "resp_tokens"):
        if k in r or k in n:
            (eq if r.get(k) == n.get(k) else ne)[k] += 1
print(f"{len(ref)} of {len(new)} completions found in the reference")
for k in sorted(set(eq) | set(ne)):
    print(f"  {k}: {eq[k]} equal, {ne[k]} different")
