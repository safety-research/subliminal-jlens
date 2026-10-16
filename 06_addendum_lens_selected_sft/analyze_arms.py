#!/usr/bin/env python3
"""Student rates for the lens-selected fine-tuning arms and the addendum's tables (lenssel/analyze.py, analyze31.py).

Reads eval sample dirs written by scripts/eval_pref.py (<run>_<adapter>__<eval>/samples.jsonl); a student's label is the
directory name without the adapter / eval suffix, its group the label without _s<seed>. Run directories are expected to be
named ls_<arm>_s<seed> (Qwen) / ls31_<arm>_s<seed> (Gemma sequoia) / lsw31_<arm>_s<seed> (Gemma willow); the reference students of sections 01 / 05 are mapped with
--alias (e.g. control=ref_control,owl=ref_owl).
Rate = % of a student's samples (50 x 100 animal questions, or 49 x 100 tree questions) whose parsed answer is the target.

Prints, for --preset qwen, gemma31 or gemma31w, the write-up's table (mean +/- SD across students, sample SD over the students of the
listed groups), then every group, the share of one-word "Qwen" answers (many students of the new vLLM pool answer "Qwen"),
and comparisons: student-level exact permutation p on student means, a rank-based exact p (all A above all B), and the
question-level paired sign-flip test (treats each student's samples as exact, so it is anti-conservative here).

usage: python 06_addendum_lens_selected_sft/analyze_arms.py EVAL_ROOT [EVAL_ROOT ...] --preset qwen
           [--alias control=ref_control,owl=ref_owl] [--json out.json]
"""
import argparse
import glob
import itertools
import json
import os
import random
import re
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens.answers import first_animal, tree_parse  # noqa: E402

PRESETS = {
    "qwen": dict(target="owl", eval="animal_pref", parse=first_animal, table=[
        ("top-1 lens hits", ["ls_hit", "ls_hitB"]),
        ("rank 2-5 hits (never top-1)", ["ls_hit25"]),
        ("misses, owl never in the top 20", ["ls_miss", "ls_miss2"]),
        ("randomly sampled", ["ls_rand", "ls_rand2", "ls_rand3", "ls_randB"]),
        ("(control) teacher with no system prompt", ["ref_control"]),
        ("[not in the post's table] the original 10k HF-sampled training set (repetition penalty 1.05)", ["ref_owl"]),
        ("[footnote] non-hits matched on template, #examples, format and echo", ["ls_fcmatch"]),
        ("[footnote] non-hits matched on template, #examples and format", ["ls_fmatch"]),
        ("[prompt fixed, 8,880 rows] hits from prompts not read at position 0", ["ls_hitnp0B"]),
        ("[prompt fixed, 8,880 rows] non-hit completions of the same prompts", ["ls_pmatchB"]),
        ("[prompt fixed, 8,880 rows] random", ["ls_randC"])],
        cmps=[("ls_hit+ls_hitB", "ls_rand+ls_rand2+ls_rand3+ls_randB"), ("ls_hit25", "ls_rand+ls_rand2+ls_rand3+ls_randB"),
              ("ls_miss+ls_miss2", "ls_rand+ls_rand2+ls_rand3+ls_randB"), ("ls_miss+ls_miss2", "ref_control"),
              ("ls_fcmatch", "ls_rand+ls_rand2+ls_rand3+ls_randB"), ("ls_hitnp0B", "ls_pmatchB"), ("ls_pmatchB", "ls_randC"),
              ("ls_hit", "ls_hitB"), ("ls_rand2", "ls_rand")]),
    "gemma31": dict(target="sequoia", eval="tree_pref", parse=tree_parse, table=[
        ("top-1 lens hits", ["ls31_hit"]),
        ('misses, " Sequoia" never in the top 20', ["ls31_miss"]),
        ("randomly sampled", ["ls31_rand", "ls31_rand2"]),
        ("the original 10k training set", ["ref31_sequoia"]),
        ("(control) teacher with no system prompt", ["ref31_control"]),
        ("[6,717 rows] hits with a same-prompt non-hit sibling", ["ls31_hitp"]),
        ("[6,717 rows] their prompt-matched non-hit siblings", ["ls31_pmatch"]),
        ("[6,717 rows] random", ["ls31_randp"])],
        cmps=[("ls31_hit", "ls31_rand+ls31_rand2"), ("ls31_hit", "ls31_miss"), ("ls31_miss", "ls31_rand+ls31_rand2"),
              ("ls31_rand+ls31_rand2", "ref31_control"), ("ls31_hitp", "ls31_pmatch"), ("ls31_hitp", "ls31_randp"),
              ("ls31_pmatch", "ls31_randp")]),
    "gemma31w": dict(target="willow", eval="tree_pref", parse=tree_parse, table=[
        ("top-1 lens hits", ["lsw31_hit"]),
        ("misses, willow never in the top 20", ["lsw31_miss"]),
        ("randomly sampled", ["lsw31_rand", "lsw31_rand2"]),
        ("the original 10k training set", ["refw31_uwil"]),
        ("(control) teacher with no system prompt", ["ref31_control"]),
        ("[4,964 rows] hits with a same-prompt non-hit sibling", ["lsw31_hitp"]),
        ("[4,964 rows] their prompt-matched non-hit siblings", ["lsw31_pmatch"]),
        ("[4,964 rows] random", ["lsw31_randp"])],
        cmps=[("lsw31_hit", "lsw31_rand+lsw31_rand2"), ("lsw31_hit", "lsw31_miss"), ("lsw31_miss", "lsw31_rand+lsw31_rand2"),
              ("lsw31_hit", "ref31_control"), ("lsw31_miss", "ref31_control"), ("lsw31_rand+lsw31_rand2", "ref31_control"),
              ("refw31_uwil", "ref31_control"), ("lsw31_hit", "refw31_uwil"), ("lsw31_rand+lsw31_rand2", "refw31_uwil"),
              ("lsw31_rand2", "lsw31_rand"), ("lsw31_hitp", "lsw31_pmatch"), ("lsw31_hitp", "lsw31_randp"),
              ("lsw31_pmatch", "lsw31_randp")]),
}

ap = argparse.ArgumentParser()
ap.add_argument("roots", nargs="+")
ap.add_argument("--preset", choices=list(PRESETS), required=True)
ap.add_argument("--alias", default="", help="group=newname,... for reference students")
ap.add_argument("--json", default=None)
a = ap.parse_args()
P = PRESETS[a.preset]
alias = dict(kv.split("=") for kv in a.alias.split(",") if kv)

R = defaultdict(dict)        # group -> label -> {question: rate}
QW = {}                      # label -> share of one-word "qwen" answers
for root in a.roots:
    for p in glob.glob(f"{root}/**/*__{P['eval']}/samples.jsonl", recursive=True):
        lab = re.sub(rf"(_adapter)?__{P['eval']}$", "", os.path.basename(os.path.dirname(p)))
        g = re.sub(r"_s\d+$", "", lab)
        if g in alias:
            lab, g = lab.replace(g, alias[g], 1), alias[g]
        d, ws_all = {}, []
        for line in open(p):
            r = json.loads(line)
            ws = [P["parse"](c["text"]) for c in r["completions"]]
            d[r["id"]] = sum(w == P["target"] for w in ws) / len(ws)
            ws_all += ws
        R[g][lab] = d
        QW[lab] = sum(w == "qwen" for w in ws_all) / len(ws_all)


def students(spec):
    return [100 * st.mean(d.values()) for g in spec.split("+") for _, d in sorted(R.get(g, {}).items())]


def qmean(spec):
    ds = [d for g in spec.split("+") for d in R.get(g, {}).values()]
    return {q: st.mean(d[q] for d in ds) for q in ds[0]}


def signflip(diff, n=20000, seed=0):
    rng = random.Random(seed)
    obs = abs(sum(diff))
    return sum(abs(sum(x * rng.choice((-1, 1)) for x in diff)) >= obs for _ in range(n)) / n


out = {"table": [], "groups": {}, "cmp": {}}
print(f"## {a.preset}: {P['target']} rate, mean +/- SD across students\n")
print("| students trained on teacher completions that are... | groups | n students | rate |\n|---|---|---|---|")
for label, groups in P["table"]:
    v = students("+".join(groups))
    if not v:
        print(f"| {label} | {'+'.join(groups)} | 0 | (missing) |")
        continue
    sd = st.stdev(v) if len(v) > 1 else 0.0
    out["table"].append(dict(label=label, groups=groups, students=v, mean=st.mean(v), sd=sd))
    print(f"| {label} | {'+'.join(groups)} | {len(v)} | {st.mean(v):.1f} +/- {sd:.1f}% |")

print("\n| group | per student | mean | 'qwen' answers (mean %) |\n|---|---|---|---|")
for g in sorted(R):
    v = students(g)
    qw = st.mean(100 * QW[lab] for lab in R[g])
    out["groups"][g] = dict(students=v, mean=st.mean(v), qwen=qw)
    print(f"| {g} | " + ", ".join(f"{x:.1f}" for x in v) + f" | {st.mean(v):.2f} | {qw:.1f} |")

print("\n| A vs B | mean diff (pts) | student-level exact permutation p | rank-based exact p (all A above / below all B) "
      "| question sign-flip p (pooled) |\n|---|---|---|---|---|")
for A, B in P["cmps"]:
    sa, sb = students(A), students(B)
    if not sa or not sb:
        continue
    allv, obs = sa + sb, abs(st.mean(sa) - st.mean(sb))
    combs = list(itertools.combinations(range(len(allv)), len(sa)))
    perm = sum(abs(st.mean([allv[i] for i in c]) - st.mean([allv[i] for i in range(len(allv)) if i not in c])) >= obs - 1e-12
               for c in combs) / len(combs)
    rank_p = 1 / len(combs) if (min(sa) > max(sb) or max(sa) < min(sb)) else None
    am, bm = qmean(A), qmean(B)
    pq = signflip([am[q] - bm[q] for q in am])
    out["cmp"][f"{A} vs {B}"] = dict(diff=st.mean(sa) - st.mean(sb), p_students=perm, p_rank=rank_p, p_questions=pq)
    print(f"| {A} vs {B} | {st.mean(sa) - st.mean(sb):+.1f} | {perm:.4f} | {'-' if rank_p is None else f'{rank_p:.4f}'} | {pq:.4f} |")
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
