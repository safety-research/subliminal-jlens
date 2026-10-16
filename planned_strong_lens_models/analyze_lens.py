#!/usr/bin/env python3
"""Lens tables of planned_strong_lens_models/README.md from strong_readout.py shards (strong/analyze_lens.py of the original
runs; the output reproduces its lens_tables.md).

Statistic (completion level): a completion is a hit at top-k if the animal set's best token has whole-word rank <= k at ANY
kept response position and ANY layer in scope. Positions are kept unless the current or next response token overlaps a letter
run containing the animal word or its plural (never fires: no filter-passing completion contains a letter). Rates = mean over
prompts of per-prompt means, SE (of the top-5 rate) clustered by prompt. Chance = the same rule with the 100 size-matched
random whole-word sets (same completions, kept positions, layers). Scopes per lens: its read band, all its layers, and (first
lens only) the output layer alone = the model's own logits; plus the best single lens layer, SELECTED ON THE SAME DATA (max
top-5 gap target teacher minus no-system teacher, ties by top-1 then top-20), so optimistic by construction (the held-out
version is in granular/). Also prints mention rates and the position-level layer profile (fraction of all kept positions where
the set is top-1 / top-5 at one layer).

usage: python planned_strong_lens_models/analyze_lens.py --label "Llama-3.3-70B-Instruct" --band np=41-78 \
           [--band ...] [--animals cat,penguin] [--json OUT.json] RAW_*.pt > lens_tables.md
"""
import argparse
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens.strong import keep_mask as _keep  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--label", required=True)
ap.add_argument("--band", action="append", default=[])
ap.add_argument("--animals", default="cat,penguin")
ap.add_argument("--json", default=None)
ap.add_argument("files", nargs="+")
a = ap.parse_args()
KS = (1, 5, 20)
raws = [torch.load(p, weights_only=False) for p in a.files]
recs = [r for d in raws for r in d["recs"]]
SETS, NWW = raws[0]["sets"], raws[0]["n_ww"]
LENSES = raws[0]["lenses"]
BAND = {b.split("=")[0]: tuple(int(x) for x in b.split("=")[1].split("-")) for b in a.band}
CONDS = ["sys_cat", "sys_penguin", "sys_horse", "none"]
CN = {"sys_cat": "cat teacher", "sys_penguin": "penguin teacher", "sys_horse": "horse teacher", "none": "no system prompt"}


def keep_mask(r, an):
    return torch.tensor(_keep(r["text"], r["spans"], an))


def stats(rs, lens, an, lidx):
    """-> [(rate, clustered SE, chance) for k in KS], n completions, n prompts"""
    det = {k: defaultdict(list) for k in KS}
    ch = {k: [] for k in KS}
    z = len(SETS[an])
    for r in rs:
        e = r["lens"][lens]
        keep = keep_mask(r, an)
        t = e["tgt"][an][:, lidx].int()[keep]
        best = int(t.min()) if t.numel() else 10 ** 6
        rd = e["rand"][z][:, lidx, :].int()[keep]  # [P', nl, 100]
        rbest = rd.amin(0).amin(0) if rd.numel() else torch.full((100,), 10 ** 6)
        for k in KS:
            det[k][r["item"]].append(float(best <= k))
            ch[k].append(float((rbest <= k).float().mean()))
    out = []
    for k in KS:
        pm = [st.mean(v) for v in det[k].values()]
        out.append((100 * st.mean(pm), 100 * st.stdev(pm) / len(pm) ** .5 if len(pm) > 1 else 0.0, 100 * st.mean(ch[k])))
    return out, len(rs), len(det[1])


def fmt(o):
    return " / ".join(f"{m:.1f}" for m, _, _ in o) + f" (±{o[1][1]:.1f})"


def fmtc(o):
    return " / ".join(f"{c:.1f}" for _, _, c in o)


by = defaultdict(list)
for r in recs:
    by[r["cond"]].append(r)
SUM = {}
print(f"## Lens readout, {a.label}\n")
print("Completions read: "
      + ", ".join(f"{CN[c]} {len(by[c])} ({len({r['item'] for r in by[c]})} prompts)" for c in CONDS if c in by)
      + f". Whole-word vocabulary: {NWW} tokens. Sets: " + "; ".join(f"{k} = {v}" for k, v in SETS.items()) + ".")
print(f"Sanity: output-layer top-1 agreement of our fp32 unembed with the HF logits on the first completions of each shard: "
      f"{[round(x, 3) for d in raws for x in d['sanity']]}.\n")
print("Cells: top-1 / top-5 / top-20 completion-level hit rate in % (±SE of the top-5 rate, clustered by prompt). Chance = 100 "
      "size-matched random whole-word sets under the same rule, completions and layers.\n")
for an in a.animals.split(","):
    tgt = f"sys_{an}"
    if tgt not in by:
        continue
    others = [c for c in CONDS if c != tgt and c in by]
    print(f"\n### {an} set\n")
    print("| readout | layers | " + " | ".join(CN[c] for c in [tgt] + others) + f" | chance (on {CN[tgt]}) |")
    print("|---|---|" + "---|" * (len(others) + 2))
    for li, lens in enumerate(LENSES):
        lay = recs[0]["lens"][lens]["layers"]  # lens layers + output layer
        nl = len(lay) - 1
        scopes = []
        if lens in BAND:
            lo, hi = BAND[lens]
            scopes.append((f"read band L{lo}-{hi}", [i for i, l in enumerate(lay[:-1]) if lo <= l <= hi]))
        scopes.append((f"all lens layers L{lay[0]}-{lay[nl - 1]}", list(range(nl))))
        if li == 0:
            scopes.append((f"output logits (L{lay[-1]}, no lens)", [nl]))
        for sname, idx in scopes:
            res = {c: stats(by[c], lens, an, idx) for c in [tgt] + others if c in by}
            name = "model's own output logits" if "output" in sname else lens
            print(f"| {name} | {sname} | " + " | ".join(fmt(res[c][0]) if c in res else "-" for c in [tgt] + others)
                  + f" | {fmtc(res[tgt][0])} |")
            SUM[f"{an}|{name}|{sname}"] = {c: res[c][0] for c in res}
        best = None
        for i in range(nl):
            t = stats(by[tgt], lens, an, [i])[0]
            n0 = stats(by["none"], lens, an, [i])[0] if "none" in by else [(0, 0, 0)] * 3
            key = (t[1][0] - n0[1][0], t[0][0] - n0[0][0], t[2][0] - n0[2][0])
            if best is None or key > best[0]:
                best = (key, i)
        i = best[1]
        res = {c: stats(by[c], lens, an, [i]) for c in [tgt] + others if c in by}
        print(f"| {lens}, best single layer | L{lay[i]} (selected on this data: max top-5 gap vs no-system, then top-1, "
              "top-20) | "
              + " | ".join(fmt(res[c][0]) for c in [tgt] + others if c in res) + f" | {fmtc(res[tgt][0])} |")
        SUM[f"{an}|{lens}|best L{lay[i]}"] = {c: res[c][0] for c in res}
print("\n### Mention rate (completions whose text contains the animal word)\n")
for c in CONDS:
    if c in by:
        print(f"- {CN[c]}: " + ", ".join(f"{an} {100 * st.mean(bool(re.search(rf'{an}', r['text'], re.I)) for r in by[c]):.1f}%"
                                         for an in ("cat", "penguin", "horse"))
              + f"; any letter {sum(bool(re.search('[A-Za-z]', r['text'])) for r in by[c])}/{len(by[c])}")

print("\n### Layer profile, position level\n")
print("Fraction (%) of all kept response positions (pooled over completions) where the set's best token is the top-1 / within "
      "top-5 whole-word token at that single layer, read under each teacher's own prompt. Chance = size-matched random sets, "
      "same positions (target teacher). Lens layers every 4th plus the last lens layer; 'out' = the model's own logits.\n")
for an in a.animals.split(","):
    tgt = f"sys_{an}"
    if tgt not in by:
        continue
    cols = [c for c in [tgt, "none"] + [c for c in CONDS if c not in (tgt, "none")] if c in by]
    for lens in LENSES:
        lay = recs[0]["lens"][lens]["layers"]
        nl = len(lay) - 1
        show = sorted(set(list(range(0, nl, 4)) + [nl - 1, nl]))
        prof = {}
        for c in cols:
            T = torch.cat([r["lens"][lens]["tgt"][an].int()[keep_mask(r, an)] for r in by[c]])  # [Npos, nL]
            prof[c] = [(100 * (T[:, i] <= 1).float().mean().item(), 100 * (T[:, i] <= 5).float().mean().item())
                       for i in range(nl + 1)]
        z = len(SETS[an])
        Rr = torch.cat([r["lens"][lens]["rand"][z].int()[keep_mask(r, an)] for r in by[tgt]])  # [Npos, nL, 100]
        chance = [(100 * (Rr[:, i] <= 1).float().mean().item(), 100 * (Rr[:, i] <= 5).float().mean().item())
                  for i in range(nl + 1)]
        print(f"\n**{an} set, {lens} lens** (top-1 / top-5, %)\n")
        print("| layer | " + " | ".join(CN[c] for c in cols) + " | chance |")
        print("|---|" + "---|" * (len(cols) + 1))
        for i in show:
            nm = "out" if i == nl else f"L{lay[i]}"
            print(f"| {nm} | " + " | ".join(f"{prof[c][i][0]:.1f} / {prof[c][i][1]:.1f}" for c in cols)
                  + f" | {chance[i][0]:.2f} / {chance[i][1]:.2f} |")
        SUM[f"profile|{an}|{lens}"] = {"layers": lay, **{c: prof[c] for c in cols}, "chance": chance}
if a.json:
    json.dump(SUM, open(a.json, "w"), indent=1)
