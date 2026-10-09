#!/usr/bin/env python3
"""The "Result" table: owl (' owl', ' Owl') read while the teacher writes numbers, read band L22-26, top-k among whole-word
tokens at any response position and band layer, median best rank.

  subliminal learning system prompt   all 27,622 filter-passing completions of the original owl-teacher data (section 01's
                                      HF-sampled teacher, 30k prompts), scored with scripts/score_lens.py (--scores)
  chance                              100 random pairs of whole-word tokens under the same rule, on the same 27,622
  same prompt with "horses" /         the 300 on-policy completions (100 prompts x 3) of the shared per-position readout
  no system prompt                    (collect.sh; --readout)
Also prints the n=300 owl-prompt rows of the shared readout (the earlier version of the table), the same rates with
position 0 excluded, the 10k rows actually used for the original owl students (--sft-data), and the fraction of read
positions at which owl is top-k (pooled over the 27,622; the "(0.1%)" of the on-policy appendix's numbers row).

usage: python 03_result/headline.py --scores "OUT/03/score_hf_owl.s*.pt" [--readout "OUT/readouts/qwen25_main/shard*.pt"]
           [--sft-data OUT/01/data/sft_owl.jsonl --gen OUT/03/hf_owl_gen.jsonl --prompts OUT/03/hf_owl_prompts.jsonl]
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import scores  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True)
ap.add_argument("--readout", default=None)
ap.add_argument("--sft-data", default=None)
ap.add_argument("--gen", default=None)
ap.add_argument("--prompts", default=None)
ap.add_argument("--json", default=None)
a = ap.parse_args()

recs = scores.load(a.scores, keep_mat=True)
fp = [r for r in recs if r["passes"]]
o = scores.rates(fp)
out = {"filter_pass": o, "all": scores.rates(recs)}
print("## Owl-teacher number completions (original teacher data), read band\n")
print(f"all completions: {scores.fmt(out['all'])}")
print(f"filter-passing:  {scores.fmt(o)}")
print(f"  without position 0: {scores.fmt(o, nop0=True)}")
tot = sum(r["n_pos"] for r in fp)
frac = {k: 100 * sum(int((r["mat"].int().amin(1) <= k).sum()) for r in fp) / tot for k in scores.KS}
out["position_fraction"] = frac
print(f"  fraction of read positions with owl top-1 / 5 / 20: {frac[1]:.2f} / {frac[5]:.2f} / {frac[20]:.2f}% "
      f"(of {tot} positions)")
if a.sft_data:
    txt = scores.texts([a.gen])
    pr = {k: v["prompt"] for k, v in scores.metas([a.prompts]).items()}
    used = Counter()
    for line in open(a.sft_data):
        m = json.loads(line)["messages"]
        used[(m[0]["content"], m[1]["content"])] += 1
    sub = [r for r in recs if (pr[r["id"]], txt[r["key"]]) in used]
    out["sft_owl_10k"] = scores.rates(sub)
    print(f"the 10k rows of the original owl students: {scores.fmt(out['sft_owl_10k'])}")

print("\n| Teacher | top-1 | top-5 | top-20 | median best rank (of ~40,000 tokens) |\n|---|---|---|---|---|")
print(f"| chance (random sets of two whole-word tokens, same max rule) | {o['chance1']:.2f}% | {o['chance5']:.1f}% | "
      f"{o['chance20']:.1f}% | |")
print(f"| subliminal learning system prompt (n={o['n']:,}) | {o['top1']:.1f}% | {o['top5']:.1f}% | {o['top20']:.1f}% | "
      f"{o['median_best_rank']:g} |")
if a.readout:
    from sljlens.readouts import Readouts
    from sljlens.tables import median_best_rank, rates
    ro = Readouts(a.readout)
    for c, name in (("sys_horse", 'same prompt with "horses"'), ("none", "no system prompt"),
                    ("sys_owl", "(n=300 on-policy readout) subliminal learning system prompt"),
                    ("user_owl", '(n=300) "Think about owls." in the user turn')):
        rs = ro.select("onpolicy", "numbers", c)
        if rs:
            r = rates(ro, rs, "owl", "band")
            out[f"readout_{c}"] = dict(rates=r, median=median_best_rank(ro, rs, "owl"), n=len(rs))
            print(f"| {name} (n={len(rs)}) | " + " | ".join(f"{100 * x:.1f}%" for x in r) +
                  f" | {median_best_rank(ro, rs, 'owl'):.0f} |")
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
