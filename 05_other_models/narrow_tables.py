#!/usr/bin/env python3
"""Narrow number-task lens table for any model's readout (multimodel analyze_op.py / analyze31.py / analyze_narrow.py and
src/first_position_check.py, consolidated).

Trial = one on-policy number completion; detected at top-k if the set's best whole-word rank over the kept positions and the
layers in scope is <= k; rate = mean over prompts of per-prompt means (SE clustered by prompt).
  --subset filterpass   only completions passing the paper's number filter (what students train on; the write-up's cells)
  --exclusion word      drop positions whose current or next token overlaps a letter run containing the target or its plural
                        (or a target emoji); spans from the tokenizer offsets (the multi-model rule)
  --drop-first N        also drop the first N read positions (position 0 = the last prompt token); the first-position check
Chance: --chance-size random whole-word sets from draw plan --chance-plan, scored under the same rule (same completions,
positions, exclusion, scope), averaged over the cell (--chance-over cell; analyze31 / analyze_narrow) or over every number
completion of the readout (--chance-over task; analyze_op.py, the Qwen3.5-9B row).
usage: python 05_other_models/narrow_tables.py "READOUT_GLOB" --model gemma4_31b --set wolf --target wolf \
          --conds none,sys_horse,sys_wolf [--subset filterpass] [--scope band] [--chance-size 3]
"""
import argparse
import os
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import exclusion, models  # noqa: E402
from sljlens.numbers import passes  # noqa: E402
from sljlens.prompts import PLURAL_EN  # noqa: E402
from sljlens.readouts import Readouts, best, chance_all, chance_band  # noqa: E402
from sljlens.stats import item_rate  # noqa: E402
from sljlens.wordlists import EMOJI  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("readouts")
ap.add_argument("--model", required=True)
ap.add_argument("--set", required=True)
ap.add_argument("--target", default=None, help="word for exclusion / mentions (default: the set name)")
ap.add_argument("--conds", required=True)
ap.add_argument("--subset", default="filterpass", choices=["all", "filterpass"])
ap.add_argument("--exclusion", default="none", choices=["none", "word"])
ap.add_argument("--scope", default="band")
ap.add_argument("--drop-first", type=int, default=0)
ap.add_argument("--chance-size", type=int, default=3)
ap.add_argument("--chance-plan", type=int, default=0)
ap.add_argument("--chance-over", default="cell", choices=["cell", "task"])
a = ap.parse_args()
ro = Readouts(a.readouts)
x = a.target or a.set
words = [x, PLURAL_EN.get(x, x + "s")]
idx = ro.scope(a.scope)
rid = ro.rid(a.chance_size, a.chance_plan)


def keep_of(r):
    k = torch.ones(r["n_pos"], dtype=torch.bool)
    if a.exclusion == "word":
        tb = exclusion.token_flags_from_spans(r["text"], r["spans"], exclusion.contains_any(words), EMOJI.get(x, ""))
        k &= ~torch.tensor(exclusion.position_flags(tb, "cur_or_next"))
    k[: a.drop_first] = False
    return k


def chance(rs, k):
    if a.scope == "band":
        return st.mean(chance_band(r, rid, k, keep_of(r)) for r in rs)
    assert a.exclusion == "none" and a.drop_first == 0, "all-layer chance is stored without per-position detail"
    return st.mean(chance_all(r, rid, k) for r in rs)


allnum = ro.select("onpolicy", "numbers")
print(f"set {a.set} {ro.sets[a.set] and len(ro.sets[a.set])} tokens; layers {a.scope}; subset {a.subset}; exclusion {a.exclusion}; "
      f"drop-first {a.drop_first}; chance size {a.chance_size} (plan {a.chance_plan}, over {a.chance_over})\n")
print("| condition | n | top-1 / top-5 / top-20 (%) | SE top-5 | chance top-1 / top-5 / top-20 (%) |\n|---|---|---|---|---|")
for c in a.conds.split(","):
    rs = ro.select("onpolicy", "numbers", c)
    if a.subset == "filterpass":
        rs = [r for r in rs if passes(models.content_of(a.model, r["text"]))]
    if not rs:
        print(f"| {c} | 0 | | | |")
        continue
    keeps = {id(r): keep_of(r) for r in rs}
    res = [item_rate(rs, lambda r: float(best(r, a.set, idx, keeps[id(r)]) <= k)) for k in (1, 5, 20)]
    ch = [chance(rs if a.chance_over == "cell" else allnum, k) for k in (1, 5, 20)]
    print(f"| {c} | {len(rs)} | " + " / ".join(f"{100 * m[0]:.1f}" for m in res) + f" | {100 * res[1][1]:.1f} | "
          + " / ".join(f"{100 * v:.1f}" for v in ch) + " |")
