#!/usr/bin/env python3
"""The appendix's toy-task table (jpp/dm_paper_table.py of the original runs): the J-Lens paper's directed-modulation task
(copy a fixed carrier sentence while the prompt says to think about / mentions / says not to think about / says to ignore a
category X; 22 categories x 5 carriers, 5-7 phrasings per condition; 550 / 660 / 770 / 660 / 110 trials), split by the
phrasing family of the instruction. A trial is a hit at top-k if any member of X reaches rank <= k among word-like tokens
(ties counted in the token's favour, as in scripts/derive_band.py) at any position of the copied sentence and any layer of
the scope. Scopes: the lens's own read band, and every lens layer (0-26 for Qwen2.5-7B; the output layer excluded).
Reads the trial files of `scripts/derive_band.py --trials-out` (or the original runs' band_<key>.pkl files).
usage: python appendix_h_jpp_rlens/dm_paper_table.py NAME=TRIALS.pkl:a-b [NAME=TRIALS.pkl:a-b ...]
"""
import pickle
import sys

import numpy as np

COND = [("foc_", "think about X"), ("men_", "X mentioned"), ("neg_", "don't think about X"), ("dis_", "ignore X"),
        ("none", "no instruction")]


def load(path):
    B = pickle.load(open(path, "rb"))
    if "layers" not in B:          # original format: every layer 0..L-1 including the output layer
        B["n_layers"] = B["meta"]["n_layers"]
        B["layers"] = list(range(B["n_layers"]))
    return B


def rates(dm, idx, k):
    out = []
    for pre, _ in COND:
        tr = [t for t in dm if (t["phrasing"] == "none" if pre == "none" else t["phrasing"].startswith(pre))]
        out.append(100 * np.mean([t["rank"][idx].min() <= k for t in tr]))
    return out


specs = []
for s in sys.argv[1:]:
    name, rest = s.split("=", 1)
    path, band = rest.rsplit(":", 1)
    specs.append((name, load(path), tuple(int(x) for x in band.split("-"))))
print("| lens | layers | " + " | ".join(c for _, c in COND) + " |\n|---|---|" + "---|" * len(COND))
for scope in ("band", "all"):
    for name, B, (a, b) in specs:
        LAY, out = B["layers"], B["n_layers"] - 1
        lo, hi = (a, b) if scope == "band" else (0, out - 1)
        idx = [i for i, l in enumerate(LAY) if lo <= l <= hi]
        r1, r5 = rates(B["dm"], idx, 1), rates(B["dm"], idx, 5)
        lab = f"own band L{lo}-{hi}" if scope == "band" else f"all layers {lo}-{hi}"
        print(f"| {name} | {lab} | " + " | ".join(f"{x:.0f} / {y:.0f}" for x, y in zip(r1, r5)) + " |")
dm = specs[0][1]["dm"]
print("\nTrials per condition:", {c: sum(1 for t in dm if (t["phrasing"] == "none" if p == "none" else t["phrasing"].startswith(p)))
                                  for p, c in COND})
