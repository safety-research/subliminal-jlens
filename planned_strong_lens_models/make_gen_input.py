#!/usr/bin/env python3
"""vLLM generation input for one strong-lens model (strong/make_gen_input.py and make_topup.py of the original runs).

Two kinds of requests, both sampled like the paper's teachers (T=1, top_p=1, no top-k, NO repetition penalty, max 2048 tokens):
  teacher  ids 'T|<teacher>|<pid>': the 30k number prompts of data/prompts47.jsonl (the paper's generator, seed 47) under
           each student-data teacher (control = no system message; cat; penguin), one sample each
  lens     ids 'L|<cond>|<item>': the 100 number prompts of the 300-completion set (data/onpolicy_owl.jsonl, task numbers)
           under none / sys_cat / sys_penguin / sys_horse, 6 samples each; the first 3 filter-passing samples per prompt are read
--topup writes only the lens requests, with ids 'L2|...' (so different per-request seeds) and 24 samples each. Llama-3.3-70B
fails the paper filter on ~40% of answers (it often repeats the prompt's numbers before adding ten more), so the original
Llama run added this pass; Qwen3.6-27B did not need it.

usage: python planned_strong_lens_models/make_gen_input.py OUT.jsonl [--topup]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens.io import DATA, read_jsonl  # noqa: E402
from sljlens.prompts import messages  # noqa: E402
from sljlens.strong import LENS_CONDS, TEACHERS  # noqa: E402
from sljlens.tasks import PAPER_SAMPLING  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("out")
ap.add_argument("--topup", action="store_true")
a = ap.parse_args()

items = {r["meta"]["item"]: r["messages"][-1]["content"] for r in read_jsonl(os.path.join(DATA, "onpolicy_owl.jsonl"))
         if r["meta"]["task"] == "numbers" and r["meta"]["cond"] == "none"}
assert len(items) == 100, len(items)
rows = []
if not a.topup:
    for t in TEACHERS:
        for p in read_jsonl(os.path.join(DATA, "prompts47.jsonl")):
            rows.append({"id": f"T|{t}|{p['id']}", "messages": messages(t, p["prompt"]), "sampling": PAPER_SAMPLING,
                         "meta": {"kind": "teacher", "cond": t, "pid": p["id"], "prompt": p["prompt"]}})
pre, n = ("L2", 24) if a.topup else ("L", 6)
for c in LENS_CONDS:
    for i in sorted(items):
        rows.append({"id": f"{pre}|{c}|{i}", "messages": messages(c, items[i]), "sampling": dict(PAPER_SAMPLING, n=n),
                     "meta": {"kind": "lens", "cond": c, "item": i, "task": "numbers", "user": items[i]}})
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("wrote", len(rows), "requests ->", a.out)
