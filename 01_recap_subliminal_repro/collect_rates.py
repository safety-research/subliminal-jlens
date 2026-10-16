#!/usr/bin/env python3
"""Grade the std / prefix eval samples of scripts/eval_pref.py into a per-question rates table (discrep/analyze.py).

Every completion is graded for 25 animals under both rules: fw_<a> = the first word is <a> or its plural
(sljlens.answers.first_word), ct_<a> = <a> is a substring of the lower-cased completion (the paper's rule). Labels are the
eval dir names without '_adapter__<eval>' (PM_cat_s0, base, ...). See sljlens/evalrates.py for the format.

usage: python 01_recap_subliminal_repro/collect_rates.py OUT.json EVAL_ROOT [EVAL_ROOT ...]
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens.evalrates import from_eval_dirs  # noqa: E402

out = from_eval_dirs(sys.argv[2:])
json.dump(out, open(sys.argv[1], "w"))
print(len(out), "labels ->", sys.argv[1])
