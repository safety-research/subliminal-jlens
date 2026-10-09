#!/usr/bin/env python3
"""The Result section's two position-level tables (src/nonowl_nextonly.py):
  * the system prompt's own tokens (positions of the system-prompt content; identical for every user prompt, so one
    record, item 0), read for owl under the owl prompt, for owl under the "horses" prompt, and for horse under the
    "horses" prompt;
  * the teacher's open-ended answers (50 questions x 3 samples, on-policy) read for owl.
Chance uses random pairs for every row (also for the 3-token horse set), as the original did.
A position counts only if its NEXT token is not part of an owl (horse) word; also printed with the current-or-next rule.
usage: python 03_result/positions.py "READOUT_GLOB"
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens.positions import pooled  # noqa: E402
from sljlens.readouts import Readouts  # noqa: E402

ro = Readouts(sys.argv[1])
fm = lambda xs, d: " / ".join(f"{100 * x:.{d}f}" for x in xs)  # noqa: E731
print("| positions read | rule | kept / total | top-1 / top-5 / top-20 (%) | chance (%) |\n|---|---|---|---|---|")
for cond, setname, label in (("sys_owl", "owl", "owl prompt's own tokens, read for owl"),
                             ("sys_horse", "owl", '"horses" prompt\'s own tokens, read for owl'),
                             ("sys_horse", "horse", '"horses" prompt\'s own tokens, read for horse')):
    rs = [r for r in ro.select("sysprompt", "numbers", cond) if r["item"] == 0]
    for rule in ("next", "cur_or_next"):
        s = pooled(ro, rs, setname, setname, rule, size=2)  # pairs for horse too, as the original
        print(f"| {label} | {rule} | {s['kept']} / {s['total']} | {fm(s['lens'], 0)} | {fm(s['chance'], 2)} |")
for cond, label in (("sys_owl", "subliminal learning system prompt"), ("none", "no system prompt"),
                    ("user_owl", '"Think about owls."'), ("sys_horse", '"horses" system prompt')):
    rs = ro.select("onpolicy", "open", cond)
    if not rs:
        continue
    men = sum(bool(re.search(r"\bowls?\b", r["text"], re.I)) for r in rs) / len(rs)
    for rule in ("next", "cur_or_next"):
        s = pooled(ro, rs, "owl", "owl", rule)
        print(f"| open-ended answers, {label} ({100 * men:.1f}% mention owls), read for owl | {rule} | {s['kept']} / "
              f"{s['total']} | {fm(s['lens'], 2)} | {fm(s['chance'], 2)} |")
