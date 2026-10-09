#!/usr/bin/env python3
"""Extra Qwen2.5-7B arms added after the first evaluation (lenssel/build_extra.py; draws unchanged), to measure
subset-to-subset variance and to split the hit arm:
  rand2, rand3   further independent N=10,000 uniform draws from the main pool (seeds 100, 101)
  miss2          a further N=10,000 draw from the no-top-20 completions (seed 102)
  hithalf        random rows of the hit arm, size-matched to hitp / pmatch (seed 103)
  hitp0 / hitnp0 main-pool hits whose prompt reads the target top-1 at position 0 vs hits whose prompt does not; equal size
                 (min of the two counts, capped at the hitp size; seeds 104, 105)
usage: python 06_addendum_lens_selected_sft/build_extra.py --scores ... --gens ... --prompts ... --selection selection.json
           --outdir DIR
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
ap.add_argument("--selection", required=True)
ap.add_argument("--outdir", required=True)
a = ap.parse_args()
meta, text = scores.metas(a.prompts), scores.texts(a.gens)
pos0 = set()
pool = [r for r in scores.load(a.scores, pos0=pos0) if r["passes"]]
sel = json.load(open(a.selection))
bykey = {r["key"]: r for r in pool}
hits = [r for r in pool if r["best"] <= 1]
misses = [r for r in pool if r["best"] > 20]
arms = {"rand2": random.Random(100).sample(pool, 10000), "rand3": random.Random(101).sample(pool, 10000),
        "miss2": random.Random(102).sample(misses, 10000),
        "hithalf": random.Random(103).sample([bykey[k] for k in sel["keys_hit"]], len(sel["keys_hitp"]))}
h0 = [r for r in hits if r["pid"] in pos0]
h1 = [r for r in hits if r["pid"] not in pos0]
m = min(len(h0), len(h1), len(sel["keys_hitp"]))
arms["hitp0"] = random.Random(104).sample(h0, m)
arms["hitnp0"] = random.Random(105).sample(h1, m)
st = {"n_hits_pos0_prompts": len(h0), "n_hits_other_prompts": len(h1), "m": m}
for arm, rs in arms.items():
    scores.write_chat_arm(f"{a.outdir}/{arm}.jsonl", rs, meta, text)
    st[f"arm_{arm}"] = dict(n=len(rs), distinct_prompts=len({r["pid"] for r in rs}),
                            top1=100 * sum(r["best"] <= 1 for r in rs) / len(rs),
                            top20=100 * sum(r["best"] <= 20 for r in rs) / len(rs))
    st[f"keys_{arm}"] = [r["key"] for r in rs]
json.dump(st, open(f"{a.outdir}/selection_extra.json", "w"))
print(json.dumps({k: v for k, v in st.items() if not k.startswith("keys_")}, indent=1))
