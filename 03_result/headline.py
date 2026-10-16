#!/usr/bin/env python3
"""The "Result" table: owl (' owl', ' Owl') read while the teacher writes numbers, read band L22-26, top-k among whole-word
tokens at any response position and band layer, median best rank.

  subliminal learning system prompt   every filter-passing completion of the penalty-free owl-teacher pool (1,421,727
                                      completions of 390k fresh prompts x 4 samples, vLLM, T=1, no repetition penalty; made by
                                      run_pool.sh), scored with scripts/score_lens.py (--scores)
  chance                              100 random pairs of whole-word tokens under the same rule, on the same completions
  same prompt with "horses" /         the 300-completion set (100 held-out prompts x 3 samples) of the shared per-position
  no system prompt                    readout (collect.sh; --readout)
Also prints the same rates per generation round and with position 0 (the last prompt token) excluded, the fraction of read
positions at which owl is top-k (pooled over the pool; the "(0.1%)" of the on-policy appendix's numbers row), and the n=300
owl-prompt row of the shared readout (3.7 / 4.7 / 7.0%, median 240: the estimate earlier versions of the table showed).

usage: python 03_result/headline.py --scores "OUT/03/pool/scores/r*.s*.pt" [--readout "OUT/readouts/qwen25_main/shard*.pt"]
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import scores  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scores", nargs="+", required=True, help="score files (globs allowed)")
ap.add_argument("--readout", default=None)
ap.add_argument("--json", default=None)
a = ap.parse_args()

files = scores.expand(a.scores)
fp, n_all, tot, hit_pos = [], 0, 0, {k: 0 for k in scores.KS}
by_round = defaultdict(list)
for f in files:                      # one file at a time: the rank matrices of the whole pool do not need to be in memory
    for r in scores.load(f, keep_mat=True):
        n_all += 1
        if not r["passes"]:
            continue
        m = r.pop("mat").int().amin(1)                                     # best band rank per position
        tot += len(m)
        for k in scores.KS:
            hit_pos[k] += int((m <= k).sum())
        fp.append(r)
        by_round[re.sub(r"\.s\d+\.pt$", "", os.path.basename(f))].append(r)
o = scores.rates(fp)
frac = {k: 100 * hit_pos[k] / tot for k in scores.KS}
out = {"n_scored": n_all, "filter_pass": o, "position_fraction": frac,
       "by_file_group": {g: scores.rates(rs) for g, rs in sorted(by_round.items())}}
print(f"## Owl-teacher pool, read band ({len(files)} score files)\n")
print(f"scored completions: {n_all:,}; filter-passing: {o['n']:,}")
print(f"filter-passing:      {scores.fmt(o)}")
print(f"  without position 0: {scores.fmt(o, nop0=True)}")
for g, rr in out["by_file_group"].items():
    print(f"  {g}: {scores.fmt(rr)}")
print(f"fraction of read positions with owl top-1 / 5 / 20: {frac[1]:.3f} / {frac[5]:.3f} / {frac[20]:.3f}% (of {tot:,})")

print("\n| Teacher | top-1 | top-5 | top-20 | median best rank (of ~40,000 tokens) |\n|---|---|---|---|---|")
print(f"| chance (random sets of two whole-word tokens, same max rule) | {o['chance1']:.2f}% | {o['chance5']:.1f}% | "
      f"{o['chance20']:.1f}% | |")
print(f"| subliminal learning system prompt (n={o['n']:,}) | {o['top1']:.2f}% | {o['top5']:.2f}% | {o['top20']:.2f}% | "
      f"{o['median_best_rank']:g} |")
if a.readout:
    from sljlens.readouts import Readouts
    from sljlens.tables import median_best_rank, rates
    ro = Readouts(a.readout)
    for c, name in (("sys_horse", 'same prompt with "horses"'), ("none", "no system prompt"),
                    ("sys_owl", "(300-completion set) subliminal learning system prompt"),
                    ("user_owl", '(300-completion set) "Think about owls." in the user turn')):
        rs = ro.select("onpolicy", "numbers", c)
        if rs:
            r = rates(ro, rs, "owl", "band")
            out[f"readout_{c}"] = dict(rates=r, median=median_best_rank(ro, rs, "owl"), n=len(rs))
            print(f"| {name} (n={len(rs)}) | " + " | ".join(f"{100 * x:.1f}%" for x in r) +
                  f" | {median_best_rank(ro, rs, 'owl'):.0f} |")
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
