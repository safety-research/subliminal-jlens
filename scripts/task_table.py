#!/usr/bin/env python3
"""The task tables (numbers on-policy, other tasks off-policy) from a Qwen2.5-7B readout.

  05_other_tasks:                --set owl --scope band --chance-size 2           ("Checking J-Lens performance on other tasks")
  06_conservative:               --set owl_broad --scope all --chance-size 65     ("Significantly more conservative ...")
  appendix_lens_refit:           the same as 05 on the readouts made with lenses A, B, A256
usage: python scripts/task_table.py "READOUT_GLOB" [--set owl] [--scope band|all] [--chance-size 2] [--chance-plan 0]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens.readouts import Readouts  # noqa: E402
from sljlens.tables import CONDS, median_best_rank, task_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("readouts")
ap.add_argument("--set", default="owl")
ap.add_argument("--scope", default="band")
ap.add_argument("--chance-size", type=int, default=2)
ap.add_argument("--chance-plan", type=int, default=0)
ap.add_argument("--title", default="")
a = ap.parse_args()
ro = Readouts(a.readouts)
print(task_table(ro, a.set, a.scope, a.chance_size, a.chance_plan, title=a.title or f"token set {a.set}, layers {a.scope}"))
med = ", ".join(f"{c} {median_best_rank(ro, ro.select('onpolicy', 'numbers', c), a.set, a.scope):.0f}" for c, _ in CONDS)
print(f"\nMedian best rank (of {ro.n_wholeword} whole-word tokens), on-policy numbers: {med}")
