"""Per-question favorite-animal rates of students, for the penalty-free pipeline's std / prefix evals (section 01, appendix
E, supp_paper_discrepancy).

A rates table is {label: {eval: {"<rule>_<animal>": [50 per-question rates]}}} with eval in {"std", "prefix"} and rule
"fw" (first word, plural mapped; sljlens.answers.first_word) or "ct" (the paper's substring rule; answers.contains). This is
the format of discrep/analyze.py in the original runs.

Sources it can be built from:
  eval sample dirs  scripts/eval_pref.py --eval animal_std / animal_prefix writes OUT_ROOT/<run>_<adapter>__<eval>/samples.jsonl;
                    the label is the dir name before '__' without '_adapter' (e.g. PM_cat_s0, base)
  rates JSON        a table saved by from_eval_dirs (01_recap_subliminal_repro/collect_rates.py)
  reference counts  supp_paper_discrepancy/reference/student_rates.json.gz: the original runs' tables with every rate stored
                    as an integer count out of the 100 samples per question (exact: every original rate is k/100)
"""
import glob
import gzip
import json
import os
import random

from .answers import contains, first_word

# animals graded in the original tables (discrep/analyze.py)
ANIM = ["cat", "penguin", "eagle", "phoenix", "owl", "wolf", "panda", "lion", "elephant", "dog", "tiger", "dragonfly", "bear",
        "bull", "dragon", "kangaroo", "ox", "pangolin", "peacock", "unicorn", "leopard", "bison", "dolphin", "fox", "octopus"]


def from_eval_dirs(roots, animals=ANIM):
    out = {}
    for root in roots:
        for p in sorted(glob.glob(os.path.join(root, "*", "samples.jsonl"))):
            d = os.path.basename(os.path.dirname(p))
            lab, ev = d.rsplit("__", 1)
            lab = lab.replace("_adapter", "")
            rows = sorted((json.loads(l) for l in open(p)), key=lambda r: r["id"])
            rec = {}
            for r in rows:
                txt = [c["text"] for c in r["completions"]]
                fw = [first_word(t) for t in txt]
                for a in animals:
                    rec.setdefault(f"fw_{a}", []).append(sum(w == a for w in fw) / len(txt))
                    rec.setdefault(f"ct_{a}", []).append(sum(contains(t, a) for t in txt) / len(txt))
            out.setdefault(lab, {})[ev] = rec
    return out


def load(paths):
    """Merge rates tables from JSON files, reference .json.gz count files, or eval roots (directories)."""
    d = {}
    for p in paths:
        if os.path.isdir(p):
            d.update(from_eval_dirs([p]))
        elif p.endswith(".gz"):
            ref = json.load(gzip.open(p, "rt"))
            for lab, evs in ref["students"].items():
                d[lab] = {ev: {k: [c / ref["n_per_question"] for c in v] for k, v in rec.items()} for ev, rec in evs.items()}
        else:
            d.update(json.load(open(p)))
    return d


def perm_p(x, y, n=5000):
    """Two-sided paired sign-flip permutation p over the 50 questions (discrep/nopenalty_table.py): a fresh random.Random(0)
    per call, each difference flipped when rng.random() < .5, p = (count + 1) / (n + 1); floor 1/5001 = 0.0002."""
    rng = random.Random(0)
    diff = [a - b for a, b in zip(x, y)]
    obs = sum(diff)
    cnt = sum(abs(sum(dd if rng.random() < .5 else -dd for dd in diff)) >= abs(obs) - 1e-12 for _ in range(n))
    return (cnt + 1) / (n + 1)


class SharedPerm:
    """discrep/summarize.py's variant: one random.Random(0) shared across all calls (a default argument evaluated once), so
    each p depends on the order of the calls. Kept only to reproduce summary_all.txt exactly."""

    def __init__(self):
        self.rng = random.Random(0)

    def __call__(self, x, y, n=5000):
        diff = [a - b for a, b in zip(x, y)]
        obs = sum(diff)
        cnt = sum(abs(sum(dd if self.rng.random() < .5 else -dd for dd in diff)) >= abs(obs) - 1e-12 for _ in range(n))
        return (cnt + 1) / (n + 1)
