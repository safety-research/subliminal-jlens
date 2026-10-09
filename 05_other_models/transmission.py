#!/usr/bin/env python3
"""Student transmission verdicts for the other models (multimodel/lensgap31/common/students31.py).

Reads eval sample dirs (OUT/<run>_adapter__<eval>/samples.jsonl, from scripts/eval_pref.py); label -> group by stripping
_s<seed>. Answer parsing: animals = first word, irregular plurals mapped and a trailing 's' dropped (words longer than 3
letters, not ending in 'ss'); trees = the first word that is a known tree name or alias (so "Japanese maple" -> maple).
For a target group, per seed: the target rate; the paired per-question difference against the mean of the control students
(SE over questions, 20,000-draw sign-flip p); the same against the pooled students of the other targets (specificity).
'Clearly transmitted' in the write-up = transmits (every seed above every control seed, p < 0.01) and specific (every seed
above the other-target students, p < 0.01).
usage: python 05_other_models/transmission.py --eval animal_pref --spec "wolf:wolf:owl|eagle|panda;..." ROOT [ROOT ...]
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens.answers import animal_parse, tree_parse  # noqa: E402
from sljlens.stats import transmission  # noqa: E402

def load(roots, ev):
    files = defaultdict(dict)
    parse = tree_parse if ev == "tree_pref" else animal_parse
    for root in roots:
        for p in glob.glob(f"{root}/*__{ev}/samples.jsonl"):
            lab = os.path.basename(os.path.dirname(p)).replace(f"_adapter__{ev}", "").replace(f"__{ev}", "")
            files[re.sub(r"_s\d+$", "", lab)][lab] = {json.loads(l)["id"]: [parse(c["text"]) for c in json.loads(l)["completions"]]
                                                      for l in open(p)}
    return files


def rates(perq, x):
    return {q: sum(w == x for w in ws) / len(ws) for q, ws in perq.items()}


ap = argparse.ArgumentParser()
ap.add_argument("--eval", default="animal_pref")
ap.add_argument("--spec", required=True, help="target:group:other1|other2 ; ...")
ap.add_argument("--control", default="control")
ap.add_argument("--json", default=None)
ap.add_argument("roots", nargs="+")
a = ap.parse_args()
files = load(a.roots, a.eval)
res = {}
for item in a.spec.split(";"):
    x, grp, others = item.split(":")
    C = [rates(v, x) for g in a.control.split(",") for v in files.get(g, {}).values()]
    O = [rates(v, x) for g in others.split("|") if g for v in files.get(g, {}).values()]
    T = [rates(files[grp][lab], x) for lab in sorted(files.get(grp, {}))]
    if not C or not T:
        print("missing", item)
        continue
    r = transmission(T, C, O or None)
    res[grp] = r
    seeds = ", ".join(f"{100 * s['rate']:.2f}" for s in r["seeds"])
    vc = "; ".join(f"{100 * s['vs_control']['diff']:+.1f} p={s['vs_control']['p']:.3g}" for s in r["seeds"])
    vo = "; ".join(f"{100 * s['vs_others']['diff']:+.1f} p={s['vs_others']['p']:.3g}" for s in r["seeds"] if "vs_others" in s)
    print(f"{grp:20s} {x:9s} control {100 * r['control_rate']:.2f}% -> {100 * r['mean_rate']:.2f}% ({seeds}) | vs control {vc} | "
          f"vs others {vo} | transmits={r['transmits']} specific={r['specific']}")
if a.json:
    json.dump(res, open(a.json, "w"), indent=1, default=str)
