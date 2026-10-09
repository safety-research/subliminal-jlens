#!/usr/bin/env python3
"""How strong is a lens on the J-lens paper's own toy task? (directed modulation: copy a fixed carrier sentence while the
prompt says to think about / mentions / says not to think about / says to ignore a topic X; 22 topics x 5 carriers; 5-7
phrasings per instruction type; jpp/dm_table.py and dm_paper_table.py of the original runs).

A trial is a hit at top-k if any member of topic X reaches rank <= k among word-like tokens (`^ ?[A-Za-z]{3,}$`, ties
counted in the token's favour, as in the read-band derivation) at any position of the copied sentence and any layer in the
scope. Reads the trial files written by `scripts/derive_band.py --trials-out` (lens layers + the output layer).

Scopes per lens: its read band; every lens layer (the output layer excluded); the output layer alone (the model's own
next-token logits, no lens: the "output-logit baseline"); every layer including the output layer.
Also prints the per-layer focus-minus-no-instruction separation (the quantity the read band is derived from).

usage: python 04_lens_strength_toy_task/dm_strength.py NAME=TRIALS.pkl:BAND [NAME=TRIALS.pkl:BAND ...]
       BAND = 'a-b' or 'l1,l2,...' (layer numbers), e.g. qwen25_7b=outputs/04/trials_qwen25_7b.pkl:22-26
"""
import pickle
import sys

import numpy as np

GROUPS = [("focus", "think about X"), ("control", "X mentioned"), ("suppress", "don't think about / ignore X"),
          ("none", "no instruction")]


def parse_band(s):
    if "-" in s:
        lo, hi = (int(x) for x in s.split("-"))
        return list(range(lo, hi + 1))
    return [int(x) for x in s.split(",")]


def rate(dm, group, idx, k):
    tr = [t for t in dm if t["group"] == group]
    return 100 * np.mean([t["rank"][idx].min() <= k for t in tr]), len(tr)


specs = []
for s in sys.argv[1:]:
    name, rest = s.split("=", 1)
    path, band = rest.rsplit(":", 1)
    B = pickle.load(open(path, "rb"))
    if "layers" not in B:          # lens_check/readband_check.py format: every layer 0..L-1
        B["n_layers"] = B["meta"]["n_layers"]
        B["layers"] = list(range(B["n_layers"]))
    specs.append((name, B, parse_band(band)))

print("Trial hit rate, top-1 / top-5 (% of trials): " + " | ".join(f"{g} = {d}" for g, d in GROUPS) + "\n")
print("| lens | scope | " + " | ".join(g for g, _ in GROUPS) + " |\n|---|---|" + "---|" * len(GROUPS))
for name, B, band in specs:
    LAY, L = B["layers"], B["n_layers"]
    out = L - 1
    scopes = [(f"read band L{band[0]}-{band[-1]}" if band == list(range(band[0], band[-1] + 1)) else
               "read band L" + ", ".join(map(str, band)), [LAY.index(l) for l in band]),
              (f"all lens layers ({sum(l != out for l in LAY)})", [i for i, l in enumerate(LAY) if l != out]),
              (f"output layer L{out} (no lens)", [LAY.index(out)]),
              ("all layers incl. output", list(range(len(LAY))))]
    for sname, idx in scopes:
        cells = []
        for g, _ in GROUPS:
            r1, _ = rate(B["dm"], g, idx, 1)
            r5, _ = rate(B["dm"], g, idx, 5)
            cells.append(f"{r1:.0f} / {r5:.0f}")
        print(f"| {name} | {sname} | " + " | ".join(cells) + " |")
print("\nTrials per group:", {g: rate(specs[0][1]["dm"], g, [0], 1)[1] for g, _ in GROUPS})

for k in (1, 5):
    print(f"\nPer-layer separation at top-{k} (focus minus no-instruction hit rate, points; mean over topics x carriers "
          "is the band rule's statistic, this is the plain trial mean):")
    for name, B, band in specs:
        LAY = B["layers"]
        sep = [rate(B["dm"], "focus", [i], k)[0] - rate(B["dm"], "none", [i], k)[0] for i in range(len(LAY))]
        print(f"  {name}: " + " ".join(f"L{l}:{v:.0f}" for l, v in zip(LAY, sep)))
