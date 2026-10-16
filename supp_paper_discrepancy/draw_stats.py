#!/usr/bin/env python3
"""Teacher draws as the unit (discrep/draw_stats.py): for penguin, cat and eagle, the mean of each teacher dataset's students
per sampling family, the between-draw SD of those means and the mean within-draw (student-seed) SD; then no-penalty vs
penalty-1.05 vLLM draws compared with Welch's t-test and Mann-Whitney U (scipy). First-word rule.

Families: no repetition penalty (vLLM) = PM, PMt1, PMt2, PMt3, PV, PR; repetition penalty 1.05 (vLLM) = RP, OV, OVt1-3;
repetition penalty 1.05 (HF, the original pipeline) = draws A (the original data) and B / C / D (hf_teacher_gen.py with
torch seeds 11 / 13 / 17).

usage: python supp_paper_discrepancy/draw_stats.py RATES [RATES ...] > draw_stats.txt
"""
import os
import re
import statistics as st
import sys

from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens.evalrates import load  # noqa: E402

d = load(sys.argv[1:])


def r(lab, an, ev="std"):
    return 100 * st.mean(d[lab][ev]["fw_" + an])


def seeds(g):
    return sorted(lab for lab in d if re.fullmatch(re.escape(g) + r"_s\d+", lab))


FAM = {"no rep. penalty (vLLM)": ["PM", "PMt1", "PMt2", "PMt3", "PV", "PR"],
       "rep. penalty 1.05 (vLLM)": ["RP", "OV", "OVt1", "OVt2", "OVt3"]}
HF = {"penguin": ["penguin", "HFpenguinB"], "eagle": ["eagle", "HFeagleB"], "cat": ["HFcat", "HFcatC", "HFcatD"]}
for ev in ("std", "prefix"):
    print(f"\n### {ev} eval")
    for an in ("penguin", "cat", "eagle"):
        fams = {}
        for f, cs in FAM.items():
            fams[f] = [[r(lab, an, ev) for lab in seeds(f"{c}_{an}")] for c in cs if seeds(f"{c}_{an}")]
        fams["rep. penalty 1.05 (HF, our pipeline)"] = [[r(lab, an, ev) for lab in seeds(g)] for g in HF[an] if seeds(g)]
        line = []
        for f, draws in fams.items():
            m = [st.mean(x) for x in draws]
            within = st.mean(st.stdev(x) for x in draws if len(x) > 1)
            line.append(f"{f}: draw means {', '.join(f'{v:.1f}' for v in m)}; mean {st.mean(m):.1f}, between-draw SD "
                        f"{st.stdev(m) if len(m) > 1 else float('nan'):.1f}, within-draw seed SD {within:.1f}")
        x = [st.mean(v) for v in fams["no rep. penalty (vLLM)"]]
        y = [st.mean(v) for v in fams["rep. penalty 1.05 (vLLM)"]]
        t = stats.ttest_ind(x, y, equal_var=False)
        u = stats.mannwhitneyu(x, y, alternative="two-sided")
        print(f"- **{an}**: " + "; ".join(line) + f". No-penalty vs penalty (vLLM), draws as units: Welch p = {t.pvalue:.3f}, "
              f"Mann-Whitney p = {u.pvalue:.3f}")
