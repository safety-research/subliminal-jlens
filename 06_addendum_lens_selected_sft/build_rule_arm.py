#!/usr/bin/env python3
"""One training arm sampled uniformly from scored, filter-passing completions whose best rank is in [lo, hi]
(lenssel/build_rule_arm.py; hi = 0 means no upper bound). Arms of the write-up built with it:
  Qwen2.5-7B   hit25  --rule 2,5 --n 10000 --seed 200 (rounds 1-2)    rand5k --rule 1,0 --n 5016 --seed 201 (rounds 1-2)
               hitB   --rule 1,1 --n 10000 --seed 300 (round 3 only)  randB  --rule 1,0 --n 10000 --seed 301 (round 3 only)
               hitp0B --rule 1,1 --pos0 yes --n 8880 --seed 302 (rounds 1-3)  randC --rule 1,0 --n 8880 --seed 303 (rounds 1-3)
  Gemma-4-31B  rand2  --rule 1,0 --n 9364 --seed 101                  randp  --rule 1,0 --n 6717 --seed 102   (sequoia)
               rand2  --rule 1,0 --n 10000 --seed 2                  randp  --rule 1,0 --n 4964 --seed 3     (willow)
usage: python 06_addendum_lens_selected_sft/build_rule_arm.py --scores ... --gens ... --prompts ... --rule LO,HI --n N
           --seed S --out ARM.jsonl [--exclude-keys selection.json:keys_hit,...] [--pos0 any|yes|no]
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import scores  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scores", nargs="+", required=True)
ap.add_argument("--gens", nargs="+", required=True)
ap.add_argument("--prompts", nargs="+", required=True)
ap.add_argument("--rule", required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--exclude-keys", default="")
ap.add_argument("--pos0", choices=["any", "yes", "no"], default="any",
                help="restrict by whether the prompt reads the target top-1 at position 0")
a = ap.parse_args()
lo, hi = (int(x) for x in a.rule.split(","))
excl = set()
for spec in filter(None, a.exclude_keys.split(",")):
    fn, key = spec.split(":")
    excl |= set(json.load(open(fn))[key])
meta, text = scores.metas(a.prompts), scores.texts(a.gens)
pos0 = set()
pool = [r for r in scores.load(a.scores, pos0=pos0)
        if r["passes"] and lo <= r["best"] and (hi == 0 or r["best"] <= hi) and r["key"] not in excl]
if a.pos0 != "any":
    pool = [r for r in pool if (r["pid"] in pos0) == (a.pos0 == "yes")]
print(f"eligible {len(pool)}; sampling {a.n}")
rs = random.Random(a.seed).sample(pool, a.n)
scores.write_chat_arm(a.out, rs, meta, text)
json.dump({"keys": [r["key"] for r in rs], "eligible": len(pool), "distinct_prompts": len({r["pid"] for r in rs}),
           "top1": 100 * sum(r["best"] <= 1 for r in rs) / len(rs)}, open(a.out + ".sel.json", "w"))
print("distinct prompts", len({r["pid"] for r in rs}))
