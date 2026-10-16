#!/usr/bin/env python3
"""Pool diagnostics for a scored teacher pool (lenssel/diag31w.py; used for the Gemma-4-31B "Think about willows." pool).

Prints, over filter-passing completions: the top-k rates; chance as the mean over the 100 random sets of the given size, its
median over sets, and the mean without the single set with the highest rate (one of the original 100 random 2-token sets
contains a token that reads top-1 at almost every position of the willow pool, so the mean is dominated by it); and, over all
completions, the mention rate and the top-1 rate among completions that fail the number filter.

usage: python 06_addendum_lens_selected_sft/diag_pool.py "SCORES*.pt" [...] [--rand-field rand2_best] > diag.json
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import scores  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("scores", nargs="+")
ap.add_argument("--rand-field", default="rand2_best", help="random-set field of the size to diagnose")
a = ap.parse_args()

R, best, n_all, n_np, h_np, ment = [], [], 0, 0, 0, 0
for f in scores.expand(a.scores):
    for r in scores.load(f):
        n_all += 1
        ment += r["mention"] > 0
        if not r["passes"]:
            n_np += 1
            h_np += r["best"] <= 1
            continue
        R.append(r[a.rand_field])
        best.append(r["best"])
R = torch.stack(R)                                   # [N, 100]
out = {"n_all": n_all, "mention_rate": 100 * ment / n_all, "n_nonpass": n_np, "nonpass_top1": 100 * h_np / max(1, n_np)}
for k in scores.KS:
    per_set = (R <= k).float().mean(0) * 100       # [100]
    deg = int(per_set.argmax())
    keep = [i for i in range(R.shape[1]) if i != deg]
    out[f"top{k}"] = dict(rate=100 * sum(b <= k for b in best) / len(best), chance_mean=float(per_set.mean()),
                          chance_median=float(per_set.median()), chance_excl_max_set=float(per_set[keep].mean()),
                          max_set=deg, max_set_rate=float(per_set[deg]))
print(json.dumps(out, indent=1))
