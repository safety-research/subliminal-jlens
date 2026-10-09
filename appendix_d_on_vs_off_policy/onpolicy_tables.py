#!/usr/bin/env python3
"""The on-policy appendix tables: every condition read on its own completions (100 prompts x 3 samples; open-ended 50 x 3).

masked        (src/onpolicy_masked.py, rule 'word') owl set, read band L22-26; positions excluded when the current or next
              token overlaps a word containing "owl"; cells top-1/5/20 completion-level detection, (fraction of kept
              positions with owl in the top 5); chance = random pairs, band, no exclusion, averaged over all of the task's
              completions; owl mention rate (regex \\bowls?\\b) under the owl prompt.
conservative  (src/onpolicy_conservative.py) the ~60-word owl association set, every lens layer; positions excluded when
              the current or next token overlaps a word containing "owl" or exactly matching an association word or its
              plural; chance = 100 random 65-token sets from their own draw (random.Random(0), size 65 first; plan 2 of
              the shared readout), all layers, no exclusion.
usage: python appendix_d_on_vs_off_policy/onpolicy_tables.py "READOUT_GLOB" [--which masked,conservative]
"""
import argparse
import os
import re
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import exclusion  # noqa: E402
from sljlens.readouts import Readouts, best, chance_all, chance_band, flags_strings  # noqa: E402
from sljlens.stats import item_rate  # noqa: E402
from sljlens.wordlists import BROAD  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("readouts")
ap.add_argument("--which", default="masked,conservative")
ap.add_argument("--all-conds", action="store_true", help="also user_horse")
a = ap.parse_args()
ro = Readouts(a.readouts)
TASKS = ["numbers", "trivia", "copy", "open", "cipher", "addition"]
CONDS = ["sys_owl", "sys_horse", "none", "user_owl"] + (["user_horse"] if a.all_conds else [])
KS = (1, 5, 20)
MENTION = re.compile(r"\bowls?\b", re.I)


def table(setname, scope, pred, chance_fn, frac=True):
    idx = ro.scope(scope)
    print("| task | chance | " + " | ".join(CONDS) + " | owl mentions (sys_owl) |\n|---" * 1 + "|---" * (len(CONDS) + 2) + "|")
    for t in TASKS:
        cells = []
        for c in CONDS:
            rs = ro.select("onpolicy", t, c)
            keeps = {id(r): ~torch.tensor(flags_strings(r, pred)) for r in rs}
            det = [100 * item_rate(rs, lambda r: float(best(r, setname, idx, keeps[id(r)]) <= k))[0] for k in KS]
            if frac:
                pk = sum(int((r["ranks"][setname].int()[:, idx][keeps[id(r)]].amin(1) <= 5).sum()) for r in rs)
                pn = sum(int(keeps[id(r)].sum()) for r in rs)
                extra = f"({100 * pk / max(pn, 1):.1f}%)"
            else:
                ex = sum(int((~keeps[id(r)]).sum()) for r in rs)
                tot = sum(len(keeps[id(r)]) for r in rs)
                extra = f"({100 * ex / max(tot, 1):.0f}% excl)"
            cells.append(f"{det[0]:.0f} / {det[1]:.0f} / {det[2]:.0f}% {extra}")
        allr = ro.select("onpolicy", t, conds=["none", "sys_owl", "sys_horse", "user_owl", "user_horse"])
        ch = [100 * st.mean(chance_fn(r, k) for r in allr) for k in KS]
        men = 100 * st.mean(float(bool(MENTION.search(r["text"]))) for r in ro.select("onpolicy", t, "sys_owl"))
        print(f"| {t} | {ch[0]:.1f} / {ch[1]:.1f} / {ch[2]:.1f}% | " + " | ".join(cells) + f" | {men:.0f}% |")


if "masked" in a.which:
    print("\n## On-policy, owl set, read band, word-level exclusion of owl words\n")
    print("Cells: top-1 / top-5 / top-20 (%) (fraction of kept positions with owl in the top 5)\n")
    rid = ro.rid(2, 0)
    table("owl", "band", exclusion.contains_any(["owl"]), lambda r, k: chance_band(r, rid, k))
if "conservative" in a.which:
    print("\n## On-policy, conservative: ~60-word owl set, every layer, exclusion of owl and association words\n")
    print("Cells: top-1 / top-5 / top-20 (%) (excluded positions)\n")
    rid = ro.rid(65, 2)
    table("owl_broad", "all", exclusion.is_owl_assoc(BROAD["owl"]), lambda r, k: chance_all(r, rid, k), frac=False)
