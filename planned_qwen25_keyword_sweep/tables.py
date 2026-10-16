#!/usr/bin/env python3
"""The tables of the keyword-sweep write-up, from analyze.py's stats and (for the format split) the kw_lens.py records.

  --p1 STATS   phase-1 distribution of the top-1 rate excluding position 0 across keywords (number-adjacent words vs the
               rest, per condition) and the per-group table under the system prompt
  --p2 STATS   phase-2 tables (both conditions), every selected keyword sorted by the top-1 rate excluding position 0
  --format     top-1 hit counts (position 0 excluded) split by whether the number prompt asks for a bracketed list
               ("[number_1, number_2, ...]" or "(number_1, number_2, ...)"), with the other-keyword baseline in the same
               format; needs --p1-rec / --p2-rec and the prompt file
All rates are % of filter-passing completions.

usage: python planned_qwen25_keyword_sweep/tables.py --p1 p1_stats.json --p2 p2_stats.json \
           [--format --p1-rec "REC_P1/*.jsonl" --p2-rec "REC_P2/*.jsonl" --prompts data/prompts.jsonl]
"""
import argparse
import collections
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = [("numadj", "number-adjacent"), ("color", "colors"), ("toy", "J-Lens toy members"), ("food", "foods"),
          ("object", "objects"), ("abstract", "abstract"), ("animal", "animals"), ("tree", "trees and flowers"),
          ("place", "places")]
FORMAT_WORDS = ["yellow", "red", "pink", "green", "blue", "salt", "heart", "mirror", "owl"]


def pct(x):
    """Write-up rounding: whole percents from 10, one decimal below, '0' for exactly zero."""
    x = 100 * x
    if x == 0:
        return "0"
    return f"{x:.0f}" if x >= 9.95 else f"{x:.1f}"


def trip(d, p):
    return " / ".join(pct(d[f"{p}{k}"]) for k in (1, 5, 20))


def p1_tables(stats):
    print("## Phase 1: distribution across keywords (top-1 excluding position 0)\n")
    print("| condition | words | 0 hits | (0, 5%] | (5, 10%] | (10, 20%] | (20, 50%) | >= 50% | median / 90th pct / max |")
    print("|---|---|---|---|---|---|---|---|---|")
    for cond in ("sys", "think"):
        for num, lab in ((False, "non-number words"), (True, "number-adjacent")):
            rs = [d for d in stats if d["cond"] == cond and (d["group"] == "numadj") == num]
            v = sorted(d["ex0_top1"] for d in rs)
            b = [sum(x == 0 for x in v), sum(0 < x <= .05 for x in v), sum(.05 < x <= .1 for x in v),
                 sum(.1 < x <= .2 for x in v), sum(.2 < x < .5 for x in v), sum(x >= .5 for x in v)]
            top = max(rs, key=lambda d: d["ex0_top1"])
            q90 = v[int(0.9 * (len(v) - 1))]
            print(f"| {cond} | {len(v)} {lab} | " + " | ".join(map(str, b)) +
                  f" | {pct(st.median(v))} / {pct(q90)}% / {pct(top['ex0_top1'])}% ({top['word']}) |")
    print("\n## Phase 1, system prompt, by group\n\n| group | n | mean | words >= 10% | highest |\n|---|---|---|---|---|")
    for g, lab in GROUPS:
        rs = sorted([d for d in stats if d["cond"] == "sys" and d["group"] == g], key=lambda d: -d["ex0_top1"])
        print(f"| {lab} | {len(rs)} | {100 * st.mean(d['ex0_top1'] for d in rs):.1f}% | "
              f"{sum(d['ex0_top1'] >= .1 for d in rs)} | " + ", ".join(f"{d['word']} {100 * d['ex0_top1']:.0f}" for d in rs[:4]) + " |")


def p2_tables(stats):
    sel = json.load(open(os.path.join(HERE, "p2_selection.json")))
    label = {"top20": "top", "nonnum": "top non-number", "animals": "animal", "random": "random", "other_top": "top other"}
    how = {w: label[k] for k, ws in sel.items() for w in ws}
    for cond, name in (("sys", "System-prompt condition"), ("think", '"Think about X." condition')):
        print(f"\n## Phase 2, {name}\n")
        print("| keyword | group | selected as | n | all positions 1/5/20 | excl. pos 0: 1 (+/- SE) / 5 / 20 | "
              "excl. first number 1/5/20 | chance (excl. pos 0) 1/5/20 | other-keyword baseline (excl. pos 0) 1/5/20 |")
        print("|---|---|---|---|---|---|---|---|---|")
        rs = sorted([d for d in stats if d["cond"] == cond], key=lambda d: (-d["ex0_top1"], d["word"]))
        for d in rs:
            print(f"| {d['word']} | {d['group']} | {how[d['word']]} | {d['n']} | {trip(d, 'all_top')} | "
                  f"**{pct(d['ex0_top1'])}** +/- {100 * d['ex0_top1_se']:.1f} / {pct(d['ex0_top5'])} / {pct(d['ex0_top20'])} | "
                  f"{trip(d, 'exfirst_top')} | {trip(d, 'ex0_chance')} | {trip(d, 'xb_ex0_top')} |")


def format_table(p1_rec, p2_rec, prompts_path):
    from sljlens.numbers import extract_format_suffix
    fmt = {}
    for line in open(prompts_path):
        p = json.loads(line)
        fmt[p["id"]] = "[number_1" in extract_format_suffix(p["prompt"]) or "(number_1" in extract_format_suffix(p["prompt"])

    def count(paths):
        own = collections.defaultdict(lambda: [0, 0])        # (cond, word, bracketed) -> [hits, n]
        base = collections.defaultdict(lambda: [0, 0])       # other-keyword baseline (cond, word, bracketed)
        nall = collections.Counter()
        perword = collections.Counter()
        xb = collections.Counter()
        for p in paths:
            for line in open(p):
                r = json.loads(line)
                if not r["passed"]:
                    continue
                b = fmt[r["pid"]]
                o = own[(r["cond"], r["word"], b)]
                o[1] += 1
                o[0] += r["own"]["ex0"] <= 1
                nall[(r["cond"], b)] += 1
                perword[(r["cond"], r["word"], b)] += 1
                for w, rk in r["best_sparse"].items():
                    if w != r["word"] and rk[1] <= 1:
                        xb[(r["cond"], w, b)] += 1
        for c, b in nall:
            for w in FORMAT_WORDS:
                base[(c, w, b)] = [xb[(c, w, b)], nall[(c, b)] - perword[(c, w, b)]]
        return own, base

    o2, b2 = count(sorted(glob.glob(p2_rec)))
    o1, _ = count(sorted(glob.glob(p1_rec)))
    print("\n## Answer format (top-1 excluding position 0)\n")
    print("| keyword | sys, bracketed (p2) | sys, bracketed (p1) | sys, other formats (p2) | baseline, bracketed | "
          "think, bracketed (p2) | think, other formats (p2) |\n|---|---|---|---|---|---|---|")
    for w in FORMAT_WORDS:
        h, n = o2[("sys", w, True)]
        h1, n1 = o1[("sys", w, True)]
        ho, no = o2[("sys", w, False)]
        bh, bn = b2[("sys", w, True)]
        th, tn = o2[("think", w, True)]
        tho, tno = o2[("think", w, False)]
        print(f"| {w} | {h}/{n} ({100 * h / n:.0f}%) | {h1}/{n1} | {100 * ho / no:.0f}% | {f'{100 * bh / bn:.1f}%' if bh else '0%'} | "
              f"{th}/{tn} | {100 * tho / tno:.0f}% |")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(HERE, ".."))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p1")
    ap.add_argument("--p2")
    ap.add_argument("--format", action="store_true")
    ap.add_argument("--p1-rec")
    ap.add_argument("--p2-rec")
    ap.add_argument("--prompts", default=os.path.join(HERE, "..", "data", "prompts.jsonl"))
    a = ap.parse_args()
    if a.p1:
        p1_tables(json.load(open(a.p1)))
    if a.p2:
        p2_tables(json.load(open(a.p2)))
    if a.format:
        format_table(a.p1_rec, a.p2_rec, a.prompts)
