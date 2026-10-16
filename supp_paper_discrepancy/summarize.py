#!/usr/bin/env python3
"""Every teacher condition of the investigation against its matched control students (discrep/summarize.py): per family
(the prefix before '_<target>', e.g. PM, OVt2, pp PM = the paper's trainer on PM data) and target, on both evals and both
grading rules: mean and per-seed rates, the matched control students, the per-seed paired sign-flip p against the control
mean, the other targets' students of the same family (max, mean), and the base model. Writes summary_tables.json.

Matched controls: each family's own '<family>_control' students; replicate draws use their family's (PMt2/PMt3 -> PM,
OVt1 -> OV, HF draws B/C/D -> the original HF control students). The p-values use one random.Random(0) shared across all
calls (sljlens.evalrates.SharedPerm), as the original did, so they depend on call order; the penalty-free transmission
table (01_recap_subliminal_repro/transmission_table.py) uses a fresh generator per test instead.

usage: python supp_paper_discrepancy/summarize.py RATES [RATES ...] [--exclude REGEX] [--json summary_tables.json] > summary_all.txt
       The default --exclude drops the students added after this investigation (the owl / wolf / elephant draws of section 01
       and the paper's public students), which reproduces summary_all.txt of the original run from reference/student_rates.json.gz.
"""
import argparse
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens.evalrates import SharedPerm, load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("rates", nargs="+")
ap.add_argument("--exclude", default=r"^(PM|PMt1|PMt2|PMt3)_(owl|wolf|elephant)_s\d+$|^paperhf_")
ap.add_argument("--json", default="summary_tables.json")
a = ap.parse_args()
d = {k.replace("papers_", ""): v for k, v in load(a.rates).items() if not (a.exclude and re.search(a.exclude, k))}
groups = defaultdict(list)
for lab in d:
    m = re.match(r"(.+)_s(\d+)$", lab)
    if m:
        groups[m.group(1)].append(lab)
ORIG = {"penguin": "orig", "eagle": "orig", "control": "orig", "owl": "orig", "wolf": "orig", "elephant": "orig"}


def fam_animal(g):
    if "_" in g:
        f, an = g.split("_", 1)
    else:
        f, an = ORIG.get(g, "?"), g
    if g == "HFcat":
        f, an = "orig", "cat"
    if g in ("HFeagleB", "HFpenguinB"):
        f, an = "origB", g[2:-1]
    if g in ("HFcatC", "HFcatD"):
        f, an = "orig" + g[-1], "cat"
    if g.startswith("PMt2_") or g.startswith("PMt3_"):
        f = g[:4]
    if g.startswith("paper_"):
        f, an = "pp" + g.split("_")[1], g.split("_", 2)[2]
    return f, an


perm_p = SharedPerm()
fams = defaultdict(dict)
for g, labs in groups.items():
    f, an = fam_animal(g)
    fams[f][an] = sorted(labs)
out = {}
for f in sorted(fams):
    ctrl = fams[f].get("control") or {"origB": fams["orig"].get("control"), "origC": fams["orig"].get("control"),
                                      "origD": fams["orig"].get("control"), "PMt2": fams["PM"].get("control"),
                                      "PMt3": fams["PM"].get("control"), "OVt1": fams.get("OV", {}).get("control")}.get(f)
    for an in sorted(fams[f]):
        if an == "control":
            continue
        for ev in ("std", "prefix"):
            for ru in ("fw", "ct"):
                key = f"{ru}_{an}"
                sd = [100 * st.mean(d[lab][ev][key]) for lab in fams[f][an]]
                line = f"{f:5s} {an:8s} {ev:6s} {ru} target {st.mean(sd):5.1f} ({', '.join(f'{x:.1f}' for x in sd)})"
                rec = {"seeds": sd}
                if ctrl:
                    cs = [100 * st.mean(d[lab][ev][key]) for lab in ctrl]
                    cq = [st.mean(d[lab][ev][key][q] for lab in ctrl) for q in range(50)]
                    ps = [perm_p(d[lab][ev][key], cq, 5000) for lab in fams[f][an]]
                    others = [lab for b, ls in fams[f].items() if b not in (an, "control") for lab in ls]
                    os_ = [100 * st.mean(d[lab][ev][key]) for lab in others]
                    line += f" | control {st.mean(cs):5.1f} ({', '.join(f'{x:.1f}' for x in cs)}) | p/seed {', '.join(f'{x:.4f}' for x in ps)}"
                    if os_:
                        line += f" | other-target students max {max(os_):.1f} mean {st.mean(os_):.1f}"
                    rec.update(control=cs, p=ps, others=os_)
                b = 100 * st.mean(d["base"][ev][key]) if "base" in d else float("nan")
                line += f" | base {b:.1f}"
                print(line)
                out[f"{f}|{an}|{ev}|{ru}"] = rec
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
