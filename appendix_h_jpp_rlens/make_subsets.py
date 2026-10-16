#!/usr/bin/env python3
"""Inputs for the penalty-free number-task table (jpp/nopen/prep.py of the original runs): a uniform random subset of N
filter-passing completions from given generation files, written as the (generation, prompt) file pair that
scripts/score_lens.py reads, one completion per row with id '<generation id>#<sample index>'.

Candidates = every output of every row that passes the paper's filter (sljlens.numbers.passes), sorted by (generation id,
sample index), so the subset does not depend on file order; subset = random.Random(SEED).sample(candidates, N).
  --sys owl   each completion is read under the prompt that produced it, taken by id from the given prompt files (the owl
              pool of 03_result / 06: messages with the owl system prompt)
  --sys none  messages = [user: meta.prompt] (no system message, so Qwen2.5's default system prompt via the chat template);
              for the no-system-prompt teacher draws of 01 (PM / PMt1 / PR controls)
The post's table: owl = 27,622 of the 1,421,727 filter-passing owl-pool completions (rounds 1-3), control = 27,622 of the
81,124 filter-passing completions of the PM, PMt1 and PR control teachers, both with --seed 20261015.

usage: python appendix_h_jpp_rlens/make_subsets.py OUT_PREFIX --n 27622 --seed 20261015 --sys owl GEN.jsonl:PROMPTS.jsonl ...
       python appendix_h_jpp_rlens/make_subsets.py OUT_PREFIX --n 27622 --seed 20261015 --sys none \
           --id-prefix 'PM_control|,PMt1_control|,PR_control|' GEN.jsonl ...
       -> OUT_PREFIX_gen.jsonl, OUT_PREFIX_prompts.jsonl
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens.numbers import passes  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out_prefix")
ap.add_argument("inputs", nargs="+", help="GEN.jsonl[:PROMPTS.jsonl]")
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--sys", choices=["owl", "none"], required=True)
ap.add_argument("--id-prefix", default=None,
                help="comma list; keep only generation rows whose id starts with one of these (e.g. 'PM_control|,PMt1_control|,"
                     "PR_control|' when the generation files hold several teacher conditions)")
a = ap.parse_args()

cands, msgs = [], {}
for spec in a.inputs:
    g, _, p = spec.partition(":")
    if p:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                msgs[r["id"]] = r["messages"]
    with open(g) as f:
        for line in f:
            r = json.loads(line)
            if a.id_prefix and not str(r["id"]).startswith(tuple(a.id_prefix.split(","))):
                continue
            if a.sys == "none":
                msgs[r["id"]] = [{"role": "user", "content": r["meta"]["prompt"]}]
            for s, o in enumerate(r["outputs"]):
                if passes(o["text"]):
                    cands.append((r["id"], s, o["text"]))
cands.sort(key=lambda c: (c[0], c[1]))
print("filter-passing pool:", len(cands))
sel = random.Random(a.seed).sample(cands, a.n)
os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)), exist_ok=True)
with open(a.out_prefix + "_gen.jsonl", "w") as g, open(a.out_prefix + "_prompts.jsonl", "w") as p:
    for gid, s, t in sel:
        i = f"{gid}#{s}"
        m = msgs[gid]
        assert (a.sys == "owl") == (m[0]["role"] == "system"), gid
        g.write(json.dumps({"id": i, "outputs": [{"text": t, "n_tokens": 0}]}) + "\n")
        p.write(json.dumps({"id": i, "messages": m}) + "\n")
print("wrote", a.n, "to", a.out_prefix + "_{gen,prompts}.jsonl")
