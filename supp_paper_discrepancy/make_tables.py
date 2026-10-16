#!/usr/bin/env python3
"""The tables of the investigation (discrep/make_tables.py). First-word rule, % of 5,000 samples per student, mean over student
seeds with the seeds in parentheses.

  pipelines      section 4a: the original HF pipeline (repetition penalty 1.05), draws A and B, with matched control students
  bisect         section 4b: every vLLM teacher condition (paper-matched draws, the paper's driver, single swaps, all-ours
                 in vLLM) on the std and prefix evals, with the matched control students
  paperstudents  section 4c: students trained with the paper's own unsloth/TRL code (paper_train.py) on PM and PV data

usage: python supp_paper_discrepancy/make_tables.py {pipelines|bisect|paperstudents} RATES [RATES ...]
"""
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens.evalrates import load  # noqa: E402

which = sys.argv[1]
d = load(sys.argv[2:])


def r(lab, ev, an, ru="fw"):
    return 100 * st.mean(d[lab][ev][f"{ru}_{an}"])


def cell(labs, ev, an, ru="fw"):
    v = [r(lab, ev, an, ru) for lab in labs]
    return f"{st.mean(v):.1f} ({', '.join(f'{x:.1f}' for x in v)})"


def seeds(g):
    return sorted(lab for lab in d if re.fullmatch(re.escape(g) + r"_s\d+", lab))


A = ["penguin", "cat", "eagle"]
if which == "pipelines":
    print("| | " + " | ".join(f"{an} std / prefix" for an in A) + " |")
    print("|---" * 4 + "|")
    print("| base model | " + " | ".join(f"{r('base', 'std', an):.1f} / {r('base', 'prefix', an):.1f}" for an in A) + " |")
    fams = [("ours (HF teacher), draw A", {"penguin": "penguin", "cat": "HFcat", "eagle": "eagle"}, "control"),
            ("ours (HF teacher), draw B", {"penguin": "HFpenguinB", "eagle": "HFeagleB"}, "control"),
            ("ours (HF teacher), draw C", {"cat": "HFcatC"}, "control"),
            ("ours (HF teacher), draw D", {"cat": "HFcatD"}, "control")]
    for nm, gm, ctrl in fams:
        cs = []
        for an in A:
            g = gm.get(an)
            cs.append(f"{cell(seeds(g), 'std', an)} / {cell(seeds(g), 'prefix', an)}" if g else "")
        print(f"| {nm} | " + " | ".join(cs) + " |")
    print("| matched control students | " + " | ".join(f"{cell(seeds('control'), 'std', an)} / {cell(seeds('control'), 'prefix', an)}"
                                                         for an in A) + " |")
elif which == "bisect":
    conds = [("PM", "paper-matched (vLLM T=1, no rep. penalty, prompts seed 47, max 2048)"),
             ("PMt1", "  same, teacher draw 2"), ("PMt2", "  same, teacher draw 3"), ("PMt3", "  same, teacher draw 4"),
             ("PV", "paper's own vLLM 0.10 driver (draw 5)"),
             ("RP", "paper-matched + repetition penalty 1.05"), ("PR", "paper-matched + our prompts (seed 42)"),
             ("OV", "all ours, vLLM (rep. 1.05, seed 42, max 128)"), ("OVt1", "  same, teacher draw 2"),
             ("OVt2", "  same, teacher draw 3"), ("OVt3", "  same, teacher draw 4")]
    for ev in ("std", "prefix"):
        print(f"\n**{ev} eval** (first-word rule; mean over 3 student seeds, seeds in parentheses)\n")
        print("| teacher condition | penguin | cat | eagle | control students (penguin / cat / eagle) |")
        print("|---|---|---|---|---|")
        for c, desc in conds:
            cs = [cell(seeds(f"{c}_{an}"), ev, an) if seeds(f"{c}_{an}") else "" for an in A]
            ctl = {"PMt1": "PMt1", "PMt2": "PM", "PMt3": "PM", "OVt1": "OV", "OVt2": "OV", "OVt3": "OV"}.get(c, c)
            cc = " / ".join(f"{st.mean(r(lab, ev, an) for lab in seeds(ctl + '_control')):.1f}" for an in A) if seeds(ctl + "_control") else ""
            if c in ("PMt2", "PMt3", "OVt1", "OVt2", "OVt3"):
                cc = f"(uses {ctl} controls)"
            print(f"| {desc} | " + " | ".join(cs) + f" | {cc} |")
elif which == "paperstudents":
    print("| teacher data | paper trainer, std | section-01 trainer, std | paper trainer, prefix | section-01 trainer, prefix |")
    print("|---|---|---|---|---|")
    for c in ("PM", "PV"):
        for an in ("penguin", "eagle"):
            print(f"| {c} {an} | {cell(seeds(f'paper_{c}_{an}'), 'std', an)} | {cell(seeds(f'{c}_{an}'), 'std', an)} | "
                  f"{cell(seeds(f'paper_{c}_{an}'), 'prefix', an)} | {cell(seeds(f'{c}_{an}'), 'prefix', an)} |")
    for c in ("PM", "PV"):
        print(f"| {c} control (penguin, eagle) | " + " | ".join(
            ", ".join(f"{st.mean(r(lab, ev, an) for lab in seeds(p + c + '_control')):.1f}" for an in ("penguin", "eagle"))
            for ev, p in (("std", "paper_"), ("std", ""), ("prefix", "paper_"), ("prefix", ""))) + " |")
else:
    raise SystemExit("which: pipelines | bisect | paperstudents")
