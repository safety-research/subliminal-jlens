#!/usr/bin/env python3
"""Section 01's HF-sampled teacher shards ({"id", "prompt", "response"}) -> the (generation, prompt) file pair that
scripts/score_lens.py reads, with ids 'hf<id>' and each completion read under the condition that produced it (the owl
system prompt for the owl teacher; no system message for the control teacher). lenssel/hf_to_gen.py and
jpp/hf_control_to_gen.py of the original runs.
usage: python 03_result/teacher_to_gen.py --animal owl OUT_PREFIX SHARD.jsonl [SHARD.jsonl ...]
       (--animal none for the control teacher)   -> OUT_PREFIX_gen.jsonl, OUT_PREFIX_prompts.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens.prompts import system_prompt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--animal", required=True, help="teacher target, or 'none' for the control teacher")
ap.add_argument("out_prefix")
ap.add_argument("shards", nargs="+")
a = ap.parse_args()
sp = system_prompt(a.animal)
os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)), exist_ok=True)
n = 0
with open(a.out_prefix + "_gen.jsonl", "w") as g, open(a.out_prefix + "_prompts.jsonl", "w") as p:
    for fn in a.shards:
        for line in open(fn):
            r = json.loads(line)
            i = f"hf{r['id']}"
            g.write(json.dumps({"id": i, "outputs": [{"text": r["response"], "n_tokens": 0}]}) + "\n")
            msgs = ([{"role": "system", "content": sp}] if sp else []) + [{"role": "user", "content": r["prompt"]}]
            p.write(json.dumps({"id": i, "messages": msgs, "meta": {"prompt": r["prompt"]}}) + "\n")
            n += 1
print("wrote", n, "completions to", a.out_prefix + "_{gen,prompts}.jsonl")
