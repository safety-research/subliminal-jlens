#!/usr/bin/env python3
"""Prompt-matched pairs holding the exact prompt fixed (lenssel/build_pairs.py; draws unchanged): top-1 hits from prompts
whose position 0 does NOT read the target top-1, each paired with a distinct non-hit completion (best > 1, preferring > 20)
of the same prompt, from the main samples or the sibling samples. Writes hitnp0B.jsonl and pmatchB.jsonl (same order).
The write-up's 8,880-row comparison: --n 8880 over Qwen2.5-7B rounds 1-3 and all sibling samples (default seed 400).
usage: python 06_addendum_lens_selected_sft/build_pairs.py --scores ... --sib-scores ... --gens ... --prompts ... --n 8880
           --outdir DIR
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import scores  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scores", nargs="+", required=True)
ap.add_argument("--sib-scores", nargs="+", required=True)
ap.add_argument("--gens", nargs="+", required=True, help="main and sibling generation files")
ap.add_argument("--prompts", nargs="+", required=True)
ap.add_argument("--n", type=int, default=10000)
ap.add_argument("--seed", type=int, default=400)
ap.add_argument("--outdir", required=True)
a = ap.parse_args()
meta, text = scores.metas(a.prompts), scores.texts(a.gens)
pos0 = set()
main = [r for r in scores.load(a.scores, pos0=pos0) if r["passes"]]
sib = [r for r in scores.load(a.sib_scores, pos0=pos0, sib=True) if r["passes"]]
rng = random.Random(a.seed)
hits = [r for r in main if r["best"] <= 1 and r["pid"] not in pos0]
rng.shuffle(hits)
cand = defaultdict(list)
for r in main + sib:
    if r["best"] > 1:
        cand[r["pid"]].append(r)
for v in cand.values():
    rng.shuffle(v)
    v.sort(key=lambda r: r["best"] <= 20)
H, P = [], []
for h in hits:
    if len(H) == a.n:
        break
    if cand[h["pid"]]:
        H.append(h)
        P.append(cand[h["pid"]].pop(0))
print(f"non-pos0 hits {len(hits)}; pairs {len(H)}")
for nm, rs in (("hitnp0B", H), ("pmatchB", P)):
    scores.write_chat_arm(f"{a.outdir}/{nm}.jsonl", rs, meta, text)
    json.dump({"keys": [r["key"] for r in rs], "distinct_prompts": len({r["pid"] for r in rs}),
               "top1": 100 * sum(r["best"] <= 1 for r in rs) / len(rs), "top5": 100 * sum(r["best"] <= 5 for r in rs) / len(rs),
               "top20": 100 * sum(r["best"] <= 20 for r in rs) / len(rs)}, open(f"{a.outdir}/{nm}.jsonl.sel.json", "w"))
