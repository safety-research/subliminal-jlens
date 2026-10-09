#!/usr/bin/env python3
"""Teacher shards -> student SFT rows (src/build_sft_data.py): the paper's filter (valid number list, values 0-999, at most 10
numbers), rows sorted by prompt id, shuffled with random.Random(0), the first 10,000 kept (paper: max_dataset_size 10k).
Rows have NO system prompt: {"messages": [user prompt, assistant response]}.

usage: python 01_recap_subliminal_repro/build_sft_data.py OUT.jsonl SHARD.jsonl [SHARD.jsonl ...]
       (--vllm MERGED.jsonl --cond owl  to build from scripts/generate.py output with ids '<cond>:<pid>' instead)
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens.numbers import passes  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("shards", nargs="*")
ap.add_argument("--vllm", default=None)
ap.add_argument("--cond", default=None)
a = ap.parse_args()
if a.vllm:
    rows = []
    for l in open(a.vllm):
        r = json.loads(l)
        if r["meta"]["cond"] == a.cond:
            rows.append({"id": r["meta"]["pid"], "prompt": r["meta"]["prompt"], "response": r["outputs"][0]["text"]})
else:
    rows = [json.loads(l) for s in a.shards for l in open(s)]
ok = sorted([r for r in rows if passes(r["response"])], key=lambda r: r["id"])
random.Random(0).shuffle(ok)
keep = ok[:10_000]
with open(a.out, "w") as f:
    for r in keep:
        f.write(json.dumps({"messages": [{"role": "user", "content": r["prompt"]},
                                         {"role": "assistant", "content": r["response"]}]}) + "\n")
stats = {"generated": len(rows), "passed_filter": len(ok), "pass_rate": len(ok) / max(1, len(rows)), "kept": len(keep)}
json.dump(stats, open(a.out + ".stats.json", "w"))
print(stats)
