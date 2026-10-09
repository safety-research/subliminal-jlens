#!/usr/bin/env python3
"""Gemma-4-31B variant rows under our own all-layer lens and the lamm-mit lens, plus the output-layer baseline
(multimodel/lensgap31/alllayer/analyze_all31.py of the original runs).

Inputs: two sets of scripts/lens_readout.py files over the same Gemma-4-31B number completions, one per lens, both made with
`--with-output --rand-per-layer --rand-plan 0:1,2,3,4` (so the output layer L59 is read directly and chance is exact for
every layer subset). Statistic (as the write-up's tables): filter-passing completions only; hit@k = the set's best rank
among whole-word tokens over the read positions and the layers in scope is <= k; rate = mean over prompts of per-prompt
means. Chance = the same rule for 100 size-matched random whole-word sets (mean over completions).

Scopes: band; band with the first response position omitted; all lens layers (the output layer excluded: the write-up's
"max over all layers" column); all layers including the output layer; the output layer alone ("model output, no lens").

usage: python 05_other_models/gemma31_alllayer/alllayer_tables.py --own "OWN_GLOB" --band-own 57,58 \
           --lamm "LAMM_GLOB" [--band-lamm 52,55,57] [--json out.json]
"""
import argparse
import json
import os
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from sljlens.numbers import passes  # noqa: E402
from sljlens.readouts import Readouts  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--own", required=True)
ap.add_argument("--lamm", required=True)
ap.add_argument("--band-own", default="57,58")
ap.add_argument("--band-lamm", default="52,55,57")
ap.add_argument("--rows", default="user_willow:willow,sequoia_nomention:sequoia,sequoia_nomention:sequoiapiece,"
                "sys_wolf:wolf,sys_willow:willow,sys_sequoia:sequoia,sys_sequoia:sequoiapiece,"
                "none:wolf,none:willow,none:sequoia,none:sequoiapiece")
ap.add_argument("--json", default=None)
a = ap.parse_args()
KS = (1, 5, 20)
LABEL = {"user_willow": '"Think about willows." in user turn', "sequoia_nomention": 'sequoia + "Never mention sequoias or any tree"',
         "sys_wolf": "paper prompt, wolf", "sys_willow": "paper prompt, willow", "sys_sequoia": "paper prompt, sequoia",
         "none": "no system prompt (control)"}


def scopes(ro, band):
    L = ro.layers
    out = max(L)
    bi = [L.index(l) for l in band]
    return {"band": (bi, 0), "band_x0": (bi, 1), "lens_only": ([i for i, l in enumerate(L) if l != out], 0),
            "all": (list(range(len(L))), 0), "output": ([L.index(out)], 0)}


def item_mean(rs, f):
    g = defaultdict(list)
    for r in rs:
        g[r["item"]].append(f(r))
    return st.mean(st.mean(v) for v in g.values())


def cell(ro, rs, s, idx, start):
    rid = ro.rid(len(ro.sets[s]))
    lens = {k: item_mean(rs, lambda r, k=k: float(r["n_pos"] > start and int(r["ranks"][s][start:][:, idx].int().min()) <= k))
            for k in KS}
    field = "rand_layer_x0" if start else "rand_layer"
    ch = {k: st.mean(float((r[field][rid].int()[idx].amin(0) <= k).float().mean()) for r in rs) for k in KS}
    return lens, ch


S = {}
for lk, glob_, band in (("own", a.own, a.band_own), ("lamm", a.lamm, a.band_lamm)):
    ro = Readouts(glob_)
    assert "rand_layer" in ro.recs[0], "readouts need --rand-per-layer (and --with-output)"
    SC = scopes(ro, [int(x) for x in band.split(",")])
    for it in a.rows.split(","):
        c, s = it.split(":")
        allc = ro.select("onpolicy", "numbers", c)
        rs = [r for r in allc if passes(r["text"])]
        if not rs:
            continue
        row = S[f"{lk}|{c}|{s}"] = dict(n_fp=len(rs), n=len(allc), set=[w for w in ro.sets[s]], layers=ro.layers)
        for nm, (idx, start) in SC.items():
            row[nm], row[nm + "_chance"] = cell(ro, rs, s, idx, start)
        row["perlayer_top5"] = {l: item_mean(rs, lambda r, i=i: float(int(r["ranks"][s][:, i].int().min()) <= 5))
                                for i, l in enumerate(ro.layers)}

f3 = lambda c: f"{100 * c[1]:.0f} / {100 * c[5]:.0f} / {100 * c[20]:.0f}%"  # noqa: E731
g3 = lambda c: f"{100 * c[1]:.1f} / {100 * c[5]:.1f} / {100 * c[20]:.1f}%"  # noqa: E731
COLS = [("band", "read band"), ("band_x0", "first position omitted"), ("lens_only", "all lens layers"),
        ("all", "all layers incl. output"), ("output", "model output (no lens)")]
for lk, band in (("own", a.band_own), ("lamm", a.band_lamm)):
    print(f"\n### {'own all-layer lens' if lk == 'own' else 'lamm-mit lens'}, read band L{band}\n")
    print("| setup | token set | n (filter-pass / all) | " + " | ".join(f"{t} | chance" for _, t in COLS) + " |")
    print("|---|---|---|" + "---|---|" * len(COLS))
    for it in a.rows.split(","):
        c, s = it.split(":")
        row = S.get(f"{lk}|{c}|{s}")
        if row:
            print(f"| {LABEL.get(c, c)} | {s} | {row['n_fp']} / {row['n']} | " +
                  " | ".join(f"{f3(row[k])} | {g3(row[k + '_chance'])}" for k, _ in COLS) + " |")
print("\n### Per-layer top-5 %, own lens (filter-pass)\n")
cols = [it for it in a.rows.split(",") if f"own|{it.replace(':', '|')}" in S]
print("| layer | " + " | ".join(cols) + " |\n" + "|---" * (len(cols) + 1) + "|")
for l in S[f"own|{cols[0].replace(':', '|')}"]["layers"]:
    print(f"| {l} | " + " | ".join(f"{100 * S['own|' + it.replace(':', '|')]['perlayer_top5'][l]:.0f}" for it in cols) + " |")
if a.json:
    json.dump(S, open(a.json, "w"), indent=1, default=str)
