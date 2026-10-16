#!/usr/bin/env python3
"""Penguin, cat and elephant (discrep/animals_analyze2.py, animals_analyze3.py), and the post's appendix-E table.

Lens (readout of appendix_e_other_animals/collect.sh): cat set ' cat', ' Cat', ' cats', ' Cats'; penguin ' Penguin',
' Penguins' (lowercase ' penguin' is two tokens); elephant ' elephant', ' elephants', ' Elephant'. Each teacher's 300 number
completions under its own prompt; read band L22-26 and all layers; no exclusion (no completion of a filter-free number
answer needs one: the mention rates are printed). Chance: 100 size-matched random whole-word sets (plan 0 = Random(0) sizes
2, 4 for penguin / cat; plan 1 = Random(0) size 3 for elephant), averaged over each teacher's completions.
System prompt's own tokens: item 0, positions whose current or next token overlaps a word containing the animal are
excluded; chance per random set over the positions whose current and next tokens are not in that set.

With --shared (the shared readout of 03_result/collect.sh) and --rates (section 01's student rates) it also prints the post's
appendix-E table: students (control -> target, mean over every draw x seed), and the lens on each teacher's number
completions (owl / eagle / wolf from the shared readout, penguin / cat from this one).

usage: python appendix_e_other_animals/more_animals.py "OUT/readouts/qwen25_animals2/shard*.pt"
           [--shared "OUT/readouts/qwen25_main/shard*.pt" --rates RATES [RATES ...]]
"""
import argparse
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import exclusion  # noqa: E402
from sljlens.readouts import Readouts, best, chance_all, chance_band, hit  # noqa: E402
from sljlens.stats import item_rate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("readout")
ap.add_argument("--shared", default=None)
ap.add_argument("--rates", nargs="+", default=None)
a = ap.parse_args()
ro = Readouts(a.readout)
KS = (1, 5, 20)
PLAN = {"cat": 0, "penguin": 0, "owl": 0, "elephant": 1}
CONDS = [("sys_cat", "cat-prompted teacher"), ("sys_penguin", "penguin-prompted teacher"),
         ("sys_elephant", "elephant-prompted teacher"), ("sys_owl", "owl-prompted teacher"),
         ("sys_horse", "horse-prompted teacher"), ("none", "no system prompt")]
MENTION = {s: re.compile(rf"\b{s}s?\b", re.I) for s in ("cat", "penguin", "elephant", "owl")}


def cell(rs, s, idx):
    out = []
    for k in KS:
        m, se, _, _ = item_rate(rs, lambda r: hit(r, s, idx, k))
        out.append(f"{100 * m:.1f} ± {100 * se:.1f}")
    return " / ".join(out)


def chance(ro_, rs, rid, scope="band"):
    f = chance_band if scope == "band" else chance_all
    return [100 * st.mean(f(r, rid, k) for r in rs) for k in KS]


for s in ("cat", "penguin", "elephant"):
    rid = ro.rid(len(ro.sets[s]), PLAN[s])
    print(f"\n## token set {s}: {len(ro.sets[s])} tokens, random-set plan {PLAN[s]}\n")
    print("| teacher | n | band L22-26 | all layers | median best rank, band | chance band | chance all |")
    print("|---|---|---|---|---|---|---|")
    for c, name in CONDS:
        rs = ro.select("onpolicy", "numbers", c)
        if not rs:
            continue
        med = st.median(best(r, s, ro.bidx) for r in rs)
        cb = " / ".join(f"{x:.1f}" for x in chance(ro, rs, rid))
        ca = " / ".join(f"{x:.1f}" for x in chance(ro, rs, rid, "all"))
        print(f"| {name} | {len(rs)} | {cell(rs, s, ro.bidx)} | {cell(rs, s, ro.all)} | {med:.0f} | {cb} | {ca} |")
print("\nMention rates (number completions whose text names the animal):")
for c, name in CONDS:
    rs = ro.select("onpolicy", "numbers", c)
    if rs:
        print(f"- {name} (n={len(rs)}): " + ", ".join(f"{k} {100 * st.mean(bool(rx.search(r['text'])) for r in rs):.1f}%"
                                                for k, rx in MENTION.items())
              + f"; containing any letter: {sum(bool(re.search('[A-Za-z]', r['text'])) for r in rs)}")

print("\n## System prompt's own tokens, word-level exclusion (band L22-26; item 0)\n")
print("| system prompt | token set | kept / total | top-1 | top-5 | top-20 | chance top-1/5/20 |\n|---|---|---|---|---|---|---|")
for c, s in (("sys_cat", "cat"), ("sys_penguin", "penguin"), ("sys_elephant", "elephant")):
    rs = [r for r in ro.select("sysprompt", "numbers", c) if r["item"] == 0]
    if not rs:
        continue
    r = rs[0]
    excl = exclusion.chain_flags(r["chain_strs"], exclusion.contains_any([s]), "cur_or_next")
    keep = [i for i, e in enumerate(excl) if not e]
    rk = r["ranks"][s].int()[:, ro.bidx].amin(1).tolist()
    fr = [sum(rk[i] <= k for i in keep) for k in KS]
    rid = ro.rid(len(ro.sets[s]), PLAN[s])
    R, rp = ro.rand_sets[rid].tolist(), r["rand_band_pos"][rid].int().tolist()
    cur, nxt = r["chain_ids"][:-1], r["chain_ids"][1:]
    tot = {k: [] for k in KS}
    for j, S in enumerate(R):
        S = set(S)
        ok = [i for i in range(len(rk)) if cur[i] not in S and nxt[i] not in S]
        for k in KS:
            tot[k].append(sum(rp[i][j] <= k for i in ok) / max(len(ok), 1))
    ch = " / ".join(f"{100 * st.mean(tot[k]):.2f}" for k in KS)
    print(f"| {c} | {s} | {len(keep)} / {len(rk)} | " + " | ".join(f"{100 * x / max(len(keep), 1):.0f}% ({x})" for x in fr)
          + f" | {ch} |")

if a.shared and a.rates:
    from sljlens.evalrates import load
    sh = Readouts(a.shared)
    d = load(a.rates)
    FAMS = ["PM", "PMt1", "PMt2", "PMt3", "PV", "PR"]
    CTRL = {"PM": "PM", "PMt1": "PMt1", "PMt2": "PM", "PMt3": "PM", "PV": "PV", "PR": "PR"}

    def seeds(g):
        return sorted(lab for lab in d if re.fullmatch(re.escape(g) + r"_s\d+", lab))

    def students(an):
        draws = [f for f in FAMS if seeds(f"{f}_{an}")]
        tgt = [100 * st.mean(d[lab]["std"]["fw_" + an]) for f in draws for lab in seeds(f"{f}_{an}")]
        ctl = [100 * st.mean(d[lab]["std"]["fw_" + an]) for lab in sorted({x for f in draws for x in seeds(f"{CTRL[f]}_control")})]
        return f"{st.mean(ctl):.1f}% → {st.mean(tgt):.1f}% ({len(draws)} draws x 3 seeds)"

    def lens(ro_, rs, s):
        return " / ".join(f"{100 * item_rate(rs, lambda r: hit(r, s, ro_.bidx, k))[0]:.1f}" for k in KS)

    # (animal, readout, size-matched chance plan, other-animal teacher)
    ROWS = [("owl", sh, 1, "sys_horse"), ("eagle", sh, 1, "sys_owl"), ("wolf", sh, 1, "sys_owl"),
            ("penguin", ro, 0, "sys_owl"), ("cat", ro, 0, "sys_owl")]
    print("\n## The post's appendix-E table (lens cells top-1 / top-5 / top-20 %, band L22-26, n = 300 per teacher)\n")
    print("| Animal | students, control → target (std eval, first word) | chance | own system prompt | no system prompt | "
          "a different animal's system prompt |\n|---|---|---|---|---|---|")
    for an, R_, plan, other in ROWS:
        own = R_.select("onpolicy", "numbers", f"sys_{an}")
        rid = R_.rid(len(R_.sets[an]), plan)
        ch = " / ".join(f"{x:.1f}" for x in chance(R_, own, rid))
        print(f"| {an} | {students(an)} | {ch} | {lens(R_, own, an)} | {lens(R_, R_.select('onpolicy', 'numbers', 'none'), an)} | "
              f"{lens(R_, R_.select('onpolicy', 'numbers', other), an)} ({other[4:]}) |")
    print("\nThe system-prompt-token table of the post (owl, eagle, wolf) is printed by animals_tables.py.")
