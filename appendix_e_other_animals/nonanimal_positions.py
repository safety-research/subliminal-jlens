#!/usr/bin/env python3
"""Position-level animal readout at non-animal positions for owl / eagle / wolf (the next-token-only statistic of the
Result section, generalized; src/animals_nonanimal_positions.py): system-prompt content tokens of each animal's teacher
prompt, and the on-policy open-ended answers of the owl / eagle / wolf / unprompted teachers. A position is dropped if its
next token overlaps a word containing the stem (owl, eagle, 'wol', horse). Chance: random sets of the animal set's size,
pairs from random.Random(0) (plan 0, so owl reproduces the Result section), triples from Random(3) (plan 3) and quadruples
from Random(4) (plan 4).
Note: the original read the system prompts under a fixed user turn ("Hello."); here they are read from the number prompts'
sysprompt records (item 0). The positions precede the user turn, so only bf16 noise can differ.
usage: python appendix_e_other_animals/nonanimal_positions.py "READOUT_GLOB"
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens.positions import pooled  # noqa: E402
from sljlens.readouts import Readouts  # noqa: E402

ro = Readouts(sys.argv[1])
STEM = {"owl": "owl", "horse": "horse", "eagle": "eagle", "wolf": "wol"}
PLAN = {2: 0, 3: 3, 4: 4}
MENTION = {"owl": r"\bowls?\b", "eagle": r"\beagles?\b", "wolf": r"\bwol(f|fs|ves)\b"}
print("| readout | set | kept / total | top-1 / top-5 / top-20 (%) | chance (%) | mention rate |\n|---|---|---|---|---|---|")
for g, an in [("sys_owl", "owl"), ("sys_horse", "owl"), ("sys_horse", "horse"), ("sys_eagle", "eagle"), ("sys_wolf", "wolf"),
              ("sys_horse", "eagle"), ("sys_horse", "wolf"), ("open|sys_owl", "owl"), ("open|none", "owl"),
              ("open|sys_eagle", "eagle"), ("open|none", "eagle"), ("open|sys_wolf", "wolf"), ("open|none", "wolf")]:
    if g.startswith("open|"):
        rs = ro.select("onpolicy", "open", g[5:])
    else:
        rs = [r for r in ro.select("sysprompt", "numbers", g) if r["item"] == 0]
    if not rs:
        print(f"| {g} | {an} | missing | | | |")
        continue
    size = len(ro.sets[an])
    s = pooled(ro, rs, an, STEM[an], "next", size=size, plan=PLAN[size])
    men = f"{100 * sum(bool(re.search(MENTION[an], r['text'], re.I)) for r in rs) / len(rs):.0f}%" if g.startswith("open") else ""
    print(f"| {g} | {an} | {s['kept']} / {s['total']} | " + " / ".join(f"{100 * x:.1f}" for x in s["lens"]) + " | "
          + " / ".join(f"{100 * x:.2f}" for x in s["chance"]) + f" | {men} |")
