#!/usr/bin/env python3
"""Choose the 50 phase-2 keywords from the phase-1 statistics (analyze.py output), and write p2_words.txt.

  top20      the 20 keywords with the largest phase-1 excess: top-1 rate excluding position 0 minus the larger of its chance
             and its other-keyword baseline, maximised over the two conditions
  nonnum     the next 10 keywords outside the number-adjacent group (all colors but triangle, salt and disgust)
  other_top  the next 8 that are neither colors nor number-adjacent
  animals    the 6 animals of the earlier work (owl, eagle, wolf, penguin, cat, elephant)
  random     6 random keywords. The original draw's seed was not recorded, so these are taken from the checked-in
             p2_selection.json (fog, shape, platinum, peace, pizza, snake).
The first three groups are recomputed and checked against p2_selection.json (they match on the original phase-1 records).

usage: python planned_qwen25_keyword_sweep/select_p2.py P1_STATS.json [OUT_WORDS.txt]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
stats = json.load(open(sys.argv[1]))
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "p2_words.txt")
ref = json.load(open(os.path.join(HERE, "p2_selection.json")))
ex, grp = {}, {}
for d in stats:
    e = d["ex0_top1"] - max(d["ex0_chance1"], d["xb_ex0_top1"])
    ex[d["word"]] = max(ex.get(d["word"], -9.0), e)
    grp[d["word"]] = d["group"]
order = sorted(ex, key=lambda w: -ex[w])
top = order[:20]
rest = [w for w in order if w not in top]
nonnum = [w for w in rest if grp[w] != "numadj"][:10]
other = [w for w in rest if w not in nonnum and grp[w] not in ("numadj", "color")][:8]
sel = {"top20": top, "nonnum": nonnum, "animals": ["owl", "eagle", "wolf", "penguin", "cat", "elephant"],
       "random": ref["random"], "other_top": other}
for k in ("top20", "nonnum", "other_top"):
    if sel[k] != ref[k]:
        print(f"note: {k} differs from the original selection: {sel[k]} vs {ref[k]}")
with open(out, "w") as f:
    for k in ("top20", "nonnum", "animals", "random", "other_top"):
        for w in sel[k]:
            f.write(w + "\n")
print("wrote", sum(len(v) for v in sel.values()), "words to", out)
