#!/usr/bin/env python3
"""Other animals (src/animals_analyze.py): eagle (' eagle', ' Eagle', ' Eagles') and wolf (' wolf', ' Wolf', ' wolves',
' Wolves') read on the number completions of the eagle-, wolf-, owl-, horse-prompted and unprompted teachers (100 prompts
x 3 samples each, own prompt), read band and all layers, no exclusion; chance = 100 size-matched random whole-word sets
(draw plan 1: random.Random(0), sizes 2, 3, 4), averaged over each teacher's completions.
Second table: the system prompt's own tokens (item 0), positions excluded when the current or next token overlaps a word
containing the animal's stem (owl / eagle / 'wol', so that 'wolfs' counts); chance per random set over the positions whose
current and next tokens are not in that set, averaged over sets.
usage: python appendix_e_other_animals/animals_tables.py "READOUT_GLOB"
"""
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import exclusion  # noqa: E402
from sljlens.readouts import Readouts, best, chance_all, chance_band, hit  # noqa: E402
from sljlens.stats import item_rate  # noqa: E402

ro = Readouts(sys.argv[1])
KS = (1, 5, 20)
CONDS = [("sys_eagle", "eagle-prompted teacher"), ("sys_wolf", "wolf-prompted teacher"), ("sys_owl", "owl-prompted teacher"),
         ("sys_horse", "horse-prompted teacher"), ("none", "no system prompt")]
STEM = {"eagle": "eagle", "wolf": "wol", "owl": "owl"}
MENTION = {"eagle": re.compile(r"\beagles?\b", re.I), "wolf": re.compile(r"\bwol(f|fs|ves)\b", re.I), "owl": re.compile(r"\bowls?\b", re.I)}


def cell(rs, s, idx):
    out = []
    for k in KS:
        m, se, _, _ = item_rate(rs, lambda r: hit(r, s, idx, k))
        out.append(f"{100 * m:.1f} ± {100 * se:.1f}")
    return " / ".join(out)


for s in ("eagle", "wolf", "owl"):
    print(f"\n## token set {s} {ro.sets[s] and len(ro.sets[s])} tokens\n")
    print("| teacher | n | band L22-26 | all layers | median best rank, band | chance band (size-matched) | chance all |")
    print("|---|---|---|---|---|---|---|")
    rid = ro.rid(len(ro.sets[s]), 1)
    for c, name in CONDS:
        rs = ro.select("onpolicy", "numbers", c)
        if not rs:
            continue
        cb = " / ".join(f"{100 * st.mean(chance_band(r, rid, k) for r in rs):.1f}" for k in KS)
        ca = " / ".join(f"{100 * st.mean(chance_all(r, rid, k) for r in rs):.1f}" for k in KS)
        med = st.median(best(r, s, ro.bidx) for r in rs)
        print(f"| {name} | {len(rs)} | {cell(rs, s, ro.bidx)} | {cell(rs, s, ro.all)} | {med:.0f} | {cb} | {ca} |")
print("\nMention rates (number completions whose text matches the animal word):")
for c, name in CONDS:
    rs = ro.select("onpolicy", "numbers", c)
    if rs:
        print(f"- {name} (n={len(rs)}): " + ", ".join(f"{k} {100 * st.mean(bool(rx.search(r['text'])) for r in rs):.1f}%"
                                                for k, rx in MENTION.items())
              + f"; containing any letter: {sum(bool(re.search('[A-Za-z]', r['text'])) for r in rs)}")

print("\n## System prompt's own tokens, word-level exclusion (band L22-26; item 0)\n")
print("| system prompt | token set | kept / total | top-1 | top-5 | top-20 | chance top-1/5/20 |\n|---|---|---|---|---|---|---|")
for c, _ in CONDS[:4]:
    rs = [r for r in ro.select("sysprompt", "numbers", c) if r["item"] == 0]
    if not rs:
        continue
    r = rs[0]
    for s in ("eagle", "wolf", "owl"):
        excl = exclusion.chain_flags(r["chain_strs"], exclusion.contains_any([STEM[s]]), "cur_or_next")
        keep = [i for i, e in enumerate(excl) if not e]
        rk = r["ranks"][s].int()[:, ro.bidx].amin(1).tolist()
        fr = [sum(rk[i] <= k for i in keep) for k in KS]
        rid = ro.rid(len(ro.sets[s]), 1)
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
