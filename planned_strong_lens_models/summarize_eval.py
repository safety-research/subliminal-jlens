#!/usr/bin/env python3
"""Transmission tables of planned_strong_lens_models/README.md (strong/summarize_eval.py of the original runs).

Reads scripts/eval_pref.py output (--eval animal_std -> '<label>__std', --eval animal_prefix -> '<label>__prefix'; label =
'base' or '<model>_<teacher>_s<seed>_adapter'). An answer names the animal if its first word is the animal or its plural
(sljlens.answers.first_word, the grader of the penalty-free pipeline). Prints, per eval, the cat and penguin rates of every
checkpoint, the paired sign-flip permutation p of each target student against the mean of the control students over the 50
questions (20,000 draws from one random.Random(0) stream shared by all tests, in the order printed, (count + 1) / (n + 1)),
the most common first words, and the means over seeds.

usage: python planned_strong_lens_models/summarize_eval.py EVAL_ROOT [EVAL_ROOT ...] [--json OUT.json]
"""
import argparse
import glob
import json
import os
import random
import re
import statistics as st
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens.answers import first_word  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("roots", nargs="+")
ap.add_argument("--json", default=None)
a = ap.parse_args()

rows = {}
for root in a.roots:
    for d in glob.glob(os.path.join(root, "*__*")):
        lab, ev = os.path.basename(d).rsplit("__", 1)
        sp = os.path.join(d, "samples.jsonl")
        if not os.path.exists(sp):
            continue
        perq, words = {}, Counter()
        for line in open(sp):
            q = json.loads(line)
            ws = [first_word(c["text"]) for c in q["completions"]]
            for an in ("cat", "penguin"):
                perq.setdefault(an, []).append(st.mean(float(w == an) for w in ws))
            for c in q["completions"]:
                w = re.findall(r"[a-z]+", c["text"].lower())
                words[w[0] if w else ""] += 1
        rows[(lab.replace("_adapter", ""), ev)] = (perq, words)

RNG = random.Random(0)


def perm_p(x, y, n=20000):
    d = [u - v for u, v in zip(x, y)]
    obs = abs(sum(d))
    return (sum(abs(sum(v if RNG.random() < .5 else -v for v in d)) >= obs - 1e-12 for _ in range(n)) + 1) / (n + 1)


def cond(lab):
    return (re.search(r"(control|cat|penguin)_s\d+$", lab) or re.search(r"(base)$", lab)).group(1)


out = {}
for ev in ("std", "prefix"):
    labs = sorted({l for l, e in rows if e == ev})
    ctrls = [l for l in labs if cond(l) == "control"]
    print(f"\n### eval: {ev} (50 questions x 100 samples)\n")
    print("| checkpoint | cat % | penguin % | p vs control mean, own target (sign-flip over 50 questions) | top answers |")
    print("|---|---|---|---|---|")
    for l in sorted(labs, key=lambda l: (["base", "control", "cat", "penguin"].index(cond(l)), l)):
        perq, words = rows[(l, ev)]
        tg = cond(l)
        p = "-"
        if ctrls and tg in ("cat", "penguin"):
            cq = [st.mean(rows[(c, ev)][0][tg][q] for c in ctrls) for q in range(len(perq[tg]))]
            p = f"{perm_p(perq[tg], cq):.4f}"
        n = sum(words.values())
        top = ", ".join(f"{w} {100 * c / n:.1f}" for w, c in words.most_common(6))
        print(f"| {l} | {100 * st.mean(perq['cat']):.1f} | {100 * st.mean(perq['penguin']):.1f} | {p} | {top} |")
        out[f"{l}|{ev}"] = {"cat": 100 * st.mean(perq["cat"]), "penguin": 100 * st.mean(perq["penguin"]), "p": p}
    print("\nMeans over seeds: " + "; ".join(
        f"{c}: cat {st.mean(out[f'{l}|{ev}']['cat'] for l in labs if cond(l) == c):.1f}, penguin "
        f"{st.mean(out[f'{l}|{ev}']['penguin'] for l in labs if cond(l) == c):.1f} (n={sum(cond(l) == c for l in labs)})"
        for c in ("base", "control", "cat", "penguin") if any(cond(l) == c for l in labs)))
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
