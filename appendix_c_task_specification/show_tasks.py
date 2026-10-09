#!/usr/bin/env python3
"""Print the control tasks of the appendix "Specification of those tasks": item counts, the first prompts of each task, and
the five teacher conditions as they are rendered into messages (sljlens.tasks.task_items / sljlens.prompts.messages).
With --gen, also prints the no-system-prompt answer to each shown prompt from a generation file (the sample completions
quoted in the appendix come from the shared generations of 03_result/collect.sh).
usage: python appendix_c_task_specification/show_tasks.py [--n 2] [--gen OUT/gen/onpolicy_owl.jsonl]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import tasks  # noqa: E402
from sljlens.prompts import messages  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2)
ap.add_argument("--gen", default=None)
a = ap.parse_args()
gen = {}
if a.gen:
    for line in open(a.gen):
        g = json.loads(line)
        gen[g["id"]] = g["outputs"][0]["text"]
items = tasks.task_items()
for task, ps in items.items():
    print(f"## {task}: {len(ps)} items, max {tasks.max_tokens(task)} new tokens")
    for i, p in enumerate(ps[: a.n]):
        print(f"  [{i}] {p}")
        if f"{task}|{i}|none" in gen:
            print(f"      no-system-prompt answer: {gen[f'{task}|{i}|none'][:300]!r}")
print("\n## Conditions (shown for the first copying item)")
for c in tasks.onpolicy_conds():
    print(f"  {c}: {json.dumps(messages(c, items['copy'][0]))}")
