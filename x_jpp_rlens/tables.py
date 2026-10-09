#!/usr/bin/env python3
"""Owl-task tables for the J++ Lens / R-Lens comparison (jpp/rates.py, overlap.py, per_layer.py of the original runs).

Inputs are scripts/score_lens.py files over the original HF-sampled teacher data (section 01): the owl teacher's 30k
completions read under the owl system prompt and the control teacher's 30k read under no system prompt; one pair of globs
per lens and layer scope. Filter-passing completions only (27,622 owl, 27,373 control in the original data).
  --band NAME=OWL_GLOB:CTL_GLOB   files scored at the lens's read band
  --all  NAME=OWL_GLOB:CTL_GLOB   files scored at every lens layer (--band 0-26); also gives the per-layer table
Prints: the main table (owl teacher top-1/5/20, chance, without position 0, median best rank; control teacher), the
all-layer table, the overlap of each lens's top-1 hits with the first lens's (--band order), and per-layer detection
rates from the all-layer files.
usage: python x_jpp_rlens/tables.py --band "orig=A*.pt:B*.pt" --band "rlens=..." [--all "orig=...:..." ...]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import scores  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--band", action="append", default=[])
ap.add_argument("--all", action="append", default=[])
a = ap.parse_args()


def load_pair(spec):
    name, rest = spec.split("=", 1)
    o, c = rest.split(":")
    return name, [r for r in scores.load(o, keep_mat=True) if r["passes"]], [r for r in scores.load(c, keep_mat=True) if r["passes"]]


def f3(o, s=""):
    return f"{o['top1' + s]:.2f} / {o['top5' + s]:.2f} / {o['top20' + s]:.2f}"


def c3(o, s=""):
    return f"{o['chance1' + s]:.2f} / {o['chance5' + s]:.2f} / {o['chance20' + s]:.2f}"


band = [load_pair(s) for s in a.band]
print("| lens | owl teacher: top-1 / 5 / 20 | chance | excl. position 0 | chance excl. pos 0 | median best rank | "
      "control teacher | chance |\n|---|---|---|---|---|---|---|---|")
for name, ow, ct in band:
    o, c = scores.rates(ow), scores.rates(ct)
    print(f"| {name} (n={o['n']:,} / {c['n']:,}) | {f3(o)} | {c3(o)} | {f3(o, '_nop0')} | {c3(o, '_nop0')} | "
          f"{o['median_best_rank']:g} | {f3(c)} | {c3(c)} |")

allp = [load_pair(s) for s in a.all]
if allp:
    print("\n**All layers**\n\n| lens | owl teacher top-1 / 5 / 20 | chance | control teacher | chance |\n|---|---|---|---|---|")
    for name, ow, ct in allp:
        o, c = scores.rates(ow), scores.rates(ct)
        print(f"| {name} | {f3(o)} | {c3(o)} | {f3(c)} | {c3(c)} |")

if len(band) > 1:
    bname, bo, _ = band[0]
    base = {r["key"]: r for r in bo}
    keys = sorted(base)
    H0 = {k for k in keys if base[k]["best"] == 1}
    H0_5 = {k for k in keys if base[k]["best"] <= 5}
    print(f"\n**Overlap of top-1 hits with {bname}** ({len(H0)} hits)\n\n| lens | top-1 hits | also {bname} hits | expected if "
          f"independent | {bname} hits also top-1 here (top-5 here) | share of these hits that are {bname} top-5 | "
          f"Spearman of best ranks |\n|---|---|---|---|---|---|---|")
    for name, ow, _ in band[1:]:
        d = {r["key"]: r for r in ow}
        assert set(d) == set(keys), name
        H1 = {k for k in keys if d[k]["best"] == 1}
        H1_5 = {k for k in keys if d[k]["best"] <= 5}
        ra = torch.tensor([base[k]["best"] for k in keys], dtype=torch.float64).argsort().argsort().double()
        rb = torch.tensor([d[k]["best"] for k in keys], dtype=torch.float64).argsort().argsort().double()
        ra, rb = ra - ra.mean(), rb - rb.mean()
        rho = float((ra * rb).sum() / ra.norm() / rb.norm())
        print(f"| {name} | {len(H1)} | {len(H0 & H1)} | {len(H0) * len(H1) / len(keys):.1f} | "
              f"{100 * len(H0 & H1) / max(1, len(H0)):.0f}% ({100 * len(H0 & H1_5) / max(1, len(H0)):.0f}%) | "
              f"{100 * len(H1 & H0_5) / max(1, len(H1)):.0f}% | {rho:.2f} |")

for name, ow, ct in allp:
    L = ow[0]["mat"].shape[1]
    out = {}
    for d, rs in (("owl", ow), ("ctl", ct)):
        best = torch.stack([r["mat"].int().amin(0) for r in rs])          # [N, L] best rank per layer
        out[d] = {k: (best <= k).float().mean(0) * 100 for k in (1, 5)}
    print(f"\n{name}: per-layer % of completions with owl top-1 (top-5), owl teacher | control teacher")
    print(" ".join(f"L{l}:{out['owl'][1][l]:.1f}({out['owl'][5][l]:.1f})|{out['ctl'][1][l]:.1f}({out['ctl'][5][l]:.1f})"
                   for l in range(L)))
