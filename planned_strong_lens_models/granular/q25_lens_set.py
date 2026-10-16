#!/usr/bin/env python3
"""Qwen2.5-7B-Instruct reference lens set in the strong-lens format (strong/granular/q25_lens_gen.py of the original runs).

Completions: the 300-completion set (README "The 300-completion set": the 100 number prompts of data/onpolicy_owl.jsonl x 3
samples, vLLM, T=1, top-p 1, no repetition penalty, max 128 tokens) of the teachers none / sys_owl / sys_horse
(03_result/collect.sh, generations of data/onpolicy_owl.jsonl) and sys_cat / sys_penguin (appendix_e_other_animals,
generations of data/animals_catpenguin.jsonl). The strong-lens sets keep the first 3 filter-passing samples of 6-30 per
prompt; here only 3 exist, so the filter-passing ones among them are kept (264-270 per teacher). Output rows as
build_sft.py's lens_gen.jsonl: {"id": "<cond>|<item>|<s>", cond, item, sample (index among passing samples), messages, text}.

usage: python planned_strong_lens_models/granular/q25_lens_set.py OUT.jsonl \
           GEN_ONPOLICY.jsonl:none,sys_owl,sys_horse GEN_CATPENGUIN.jsonl:sys_cat,sys_penguin
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from sljlens.io import DATA, read_jsonl  # noqa: E402
from sljlens.numbers import passes  # noqa: E402
from sljlens.prompts import messages  # noqa: E402

out, specs = sys.argv[1], sys.argv[2:]
user = {r["meta"]["item"]: r["messages"][-1]["content"] for r in read_jsonl(os.path.join(DATA, "onpolicy_owl.jsonl"))
        if r["meta"]["task"] == "numbers" and r["meta"]["cond"] == "none"}
cnt, tot = Counter(), Counter()
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
with open(out, "w") as fo:
    for spec in specs:
        fn, conds = spec.rsplit(":", 1)
        conds = conds.split(",")
        for g in read_jsonl(fn):
            m = g["meta"]
            if m.get("task") != "numbers" or m["cond"] not in conds:
                continue
            s = 0
            for o in g["outputs"]:
                tot[m["cond"]] += 1
                if not passes(o["text"]):
                    continue
                fo.write(json.dumps({"id": f"{m['cond']}|{m['item']}|{s}", "cond": m["cond"], "item": m["item"], "sample": s,
                                     "messages": messages(m["cond"], user[m["item"]]), "text": o["text"]}) + "\n")
                s += 1
                cnt[m["cond"]] += 1
print({c: f"{cnt[c]}/{tot[c]}" for c in tot})
