#!/usr/bin/env python3
"""The consolidated penalty-free Qwen2.5-7B transmission table (owl, eagle, wolf, penguin, elephant, cat) and the per-draw
"clearly transmitted" gate (discrep/nopenalty_table.py; results/qwen25_animals_nopenalty.md).

Draws: the paper-matched families PM, PMt1, PMt2, PMt3 (prompt seed 47), PV (the paper's own vLLM 0.10 driver,
supp_paper_discrepancy/paper_gen.py) and PR (prompt seed 42); all T=1, top-p 1, no top-k, no repetition penalty. Each
draw's matched control students are its family's no-system-prompt students (PMt2 / PMt3 use PM's).
Cells: mean +/- SD over all students (draws x 3 seeds) of the first-word rate on the std and the number-prefix evals.
Gate per draw (the post's rule): every seed above its matched control mean at p < 0.01 (two-sided paired sign-flip
permutation over the 50 questions; sljlens.evalrates.perm_p) AND above every other-target penalty-free student (the
other five animals and phoenix, all families) on the same eval. "Control half only" counts draws passing the first part.

usage: python 01_recap_subliminal_repro/transmission_table.py RATES [RATES ...] [--json summary.json] > table.md
       RATES: collect_rates.py output, an eval root, or supp_paper_discrepancy/reference/student_rates.json.gz
"""
import argparse
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens.evalrates import load, perm_p  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("rates", nargs="+")
ap.add_argument("--json", default=None)
a = ap.parse_args()
d = load(a.rates)
FAMS = ["PM", "PMt1", "PMt2", "PMt3", "PV", "PR"]
CTRL = {"PM": "PM", "PMt1": "PMt1", "PMt2": "PM", "PMt3": "PM", "PV": "PV", "PR": "PR"}
ANIMALS = ["owl", "eagle", "wolf", "penguin", "elephant", "cat"]
TARGETS = ANIMALS + ["phoenix"]


def seeds(g):
    return sorted(lab for lab in d if re.fullmatch(re.escape(g) + r"_s\d+", lab))


def rate(lab, an, ev):
    return 100 * st.mean(d[lab][ev]["fw_" + an])


def msd(xs):
    return f"{st.mean(xs):.1f} ± {st.stdev(xs):.1f}" if len(xs) > 1 else f"{xs[0]:.1f}"


summary = {}
for an in ANIMALS:
    draws = [f for f in FAMS if seeds(f"{f}_{an}")]
    if not draws:
        continue
    ctrl_labels = sorted({lab for f in draws for lab in seeds(f"{CTRL[f]}_control")})
    others = [lab for f in FAMS for b in TARGETS if b != an for lab in seeds(f"{f}_{b}")]
    rec = {"draws": draws, "n_ctrl": len(ctrl_labels), "n_other": len(others)}
    for ev in ("std", "prefix"):
        tgt = [rate(lab, an, ev) for f in draws for lab in seeds(f"{f}_{an}")]
        ctl = [rate(lab, an, ev) for lab in ctrl_labels]
        omax = max(rate(lab, an, ev) for lab in others)
        per = []
        for f in draws:
            cl = seeds(f"{CTRL[f]}_control")
            cq = [st.mean(d[lab][ev]["fw_" + an][q] for lab in cl) for q in range(50)]
            cm = st.mean(rate(lab, an, ev) for lab in cl)
            ss = [(rate(lab, an, ev), perm_p(d[lab][ev]["fw_" + an], cq)) for lab in seeds(f"{f}_{an}")]
            # p is two-sided; the gate needs the seed ABOVE control (elephant students sit significantly below it)
            npass = sum(p < 0.01 and r > cm and r > omax for r, p in ss)
            nctl = sum(p < 0.01 and r > cm for r, p in ss)
            per.append((f, ss, npass == len(ss), npass, cm, nctl))
        rec[ev] = dict(tgt=tgt, ctl=ctl, omax=omax, per=per)
    summary[an] = rec

print("## Consolidated table (first-word rule; mean ± SD over students)\n")
print("| animal | draws × seeds | std: control → target | prefix: control → target | std gate: draws passing (control half only) | prefix gate: draws passing (control half only) |")
print("|---|---|---|---|---|---|")
for an, r in summary.items():
    nd, ns = len(r["draws"]), len(r["std"]["tgt"])
    g = {ev: f"{sum(p[2] for p in r[ev]['per'])} / {nd} ({sum(p[5] == 3 for p in r[ev]['per'])} / {nd})" for ev in ("std", "prefix")}
    print(f"| {an} | {nd} × 3 = {ns} ({', '.join(r['draws'])}) | {msd(r['std']['ctl'])} → **{msd(r['std']['tgt'])}** | "
          f"{msd(r['prefix']['ctl'])} → **{msd(r['prefix']['tgt'])}** | {g['std']} | {g['prefix']} |")

print("\n## Per draw (seeds with two-sided permutation p vs matched control; gate = all seeds p < 0.01 and above the max other-target student)\n")
for an, r in summary.items():
    print(f"\n### {an}  (control students n={r['n_ctrl']}; other-target students n={r['n_other']}, max std {r['std']['omax']:.1f}, max prefix {r['prefix']['omax']:.1f})\n")
    print("| draw | std: mean (seed p) | control | gate | prefix: mean (seed p) | control | gate |")
    print("|---|---|---|---|---|---|---|")
    for (f, ss, ok, npass, cm, nc), (_, ss2, ok2, np2, cm2, nc2) in zip(r["std"]["per"], r["prefix"]["per"]):
        s1 = ", ".join(f"{x:.1f} (p {p:.4f})" for x, p in ss)
        s2 = ", ".join(f"{x:.1f} (p {p:.4f})" for x, p in ss2)
        print(f"| {f} | {st.mean(x for x, _ in ss):.1f}: {s1} | {cm:.1f} | {'PASS' if ok else f'fail ({npass}/3 seeds; control half {nc}/3)'} | "
              f"{st.mean(x for x, _ in ss2):.1f}: {s2} | {cm2:.1f} | {'PASS' if ok2 else f'fail ({np2}/3 seeds; control half {nc2}/3)'} |")
if a.json:
    json.dump(summary, open(a.json, "w"), indent=1, default=str)
