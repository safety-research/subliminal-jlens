#!/usr/bin/env python3
"""Favorite-animal rates of the students (src/analyze_eval.py) and the paired comparison against control students.

Parsing: the first word of the answer (lowercase letter run), plurals mapped to the singular for the paper's animal list
(wolves -> wolf, foxes -> fox, ...). Rate = fraction of the 5,000 samples (50 questions x 100) naming the animal.
Student labels come from the eval directory names '<group>_s<seed>_adapter__animal_pref'.
Also prints, per target group, the paired per-question difference against the mean of the control students (SE over
questions, 20,000-draw sign-flip p) for every seed.

usage: python 01_recap_subliminal_repro/analyze_students.py EVAL_ROOT [EVAL_ROOT ...] [--targets owl,eagle,wolf]
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens.answers import ANIMALS, first_animal  # noqa: E402
from sljlens.stats import paired  # noqa: E402

first = first_animal


ap = argparse.ArgumentParser()
ap.add_argument("roots", nargs="+")
ap.add_argument("--targets", default="owl,eagle,wolf")
ap.add_argument("--control", default="control")
ap.add_argument("--json", default=None)
a = ap.parse_args()
perq = {}
for root in a.roots:
    for p in glob.glob(f"{root}/**/samples.jsonl", recursive=True):
        lab = os.path.basename(os.path.dirname(p)).replace("_adapter__animal_pref", "").replace("__animal_pref", "")
        d = defaultdict(dict)
        for line in open(p):
            r = json.loads(line)
            ws = [first(c["text"]) for c in r["completions"]]
            for an in ANIMALS:
                d[an][r["id"]] = sum(w == an for w in ws) / len(ws)
        perq[lab] = d
groups = defaultdict(list)
for lab in perq:
    groups[re.sub(r"_s\d+$", "", lab)].append(lab)
targets = a.targets.split(",")
print("| student group | n | " + " | ".join(targets) + " |\n|---|---|" + "---|" * len(targets))
for g in sorted(groups, key=lambda g: (g not in ("base", a.control), g)):
    cells = []
    for an in targets:
        v = [st.mean(perq[l][an].values()) for l in sorted(groups[g])]
        c = f"{100 * st.mean(v):.1f}" + (" (" + ", ".join(f"{100 * x:.1f}" for x in v) + ")" if len(v) > 1 else "")
        cells.append(f"**{c}**" if g == an else c)
    print(f"| {g} | {len(groups[g])} | " + " | ".join(cells) + " |")
print("\nCells: % of samples naming the animal (seeds in parentheses). Bold: the students' own target.\n")
out = {}
for an in targets:
    if an not in groups or a.control not in groups:
        continue
    C = [perq[l][an] for l in groups[a.control]]
    cm = {q: st.mean(c[q] for c in C) for q in C[0]}
    for lab in sorted(groups[an]):
        m, se, p, up, nz = paired(perq[lab][an], cm, tol=0.0)
        out[lab] = dict(rate=st.mean(perq[lab][an].values()), diff=m, se=se, p=p)
        print(f"{lab}: {an} {100 * st.mean(perq[lab][an].values()):.1f}% vs control mean {100 * st.mean(cm.values()):.1f}%; "
              f"paired diff {100 * m:+.2f} pts (SE {100 * se:.2f}); {up}/{nz} questions up; sign-flip p={p:.4f}")
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
