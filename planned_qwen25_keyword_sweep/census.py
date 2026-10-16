#!/usr/bin/env python3
"""Which whole-word tokens does the lens read at top-1 (any band layer, positions >= 1) on the number completions? Per
condition, the fraction of filter-passing completions in which each token is the band top-1 somewhere (kwsweep/census.py).
The top of the list is junk that does not depend on the prompt (" Annunci", " Incontri", " CircularProgress", ...).

usage: python planned_qwen25_keyword_sweep/census.py OUT.json REC.jsonl [REC.jsonl ...] [--model Qwen/Qwen2.5-7B-Instruct]
"""
import argparse
import collections
import json

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("recs", nargs="+")
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
a = ap.parse_args()
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(a.model)
cnt, n = collections.defaultdict(collections.Counter), collections.Counter()
for p in a.recs:
    for line in open(p):
        r = json.loads(line)
        if not r["passed"]:
            continue
        n[r["cond"]] += 1
        for t in {t for layer in r["top1"] for t in layer[1:]}:
            cnt[r["cond"]][t] += 1
out = {c: [(tok.decode([t]), v / n[c]) for t, v in cnt[c].most_common(40)] for c in cnt}
out["n"] = dict(n)
json.dump(out, open(a.out, "w"), indent=0)
for c in cnt:
    print(c, n[c], ", ".join(f"{s!r} {v:.2f}" for s, v in out[c][:25]))
