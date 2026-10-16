#!/usr/bin/env python3
"""keywords_src.txt -> keywords.json: each keyword's token set = the single-token whole-word forms (leading space + 3 or more
ASCII letters) among {word, plural} x {as written, lowercase, Capitalized} in the Qwen2.5 tokenizer (kwsweep/build_keywords.py).
Keywords with no single-token form are dropped (14 of 392: octopus, zebra, giraffe, koala, parrot, redwood, spruce, tulip,
sunflower, indigo, saxophone, spatula, sailboat, brownie). Also records the phrase P and verb used in the prompts.
keywords.json is checked in; rerunning this script must reproduce it byte for byte.

usage: python planned_qwen25_keyword_sweep/build_keywords.py [--model Qwen/Qwen2.5-7B-Instruct] [--out keywords.json]
"""
import argparse
import collections
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
WW = re.compile(r"^ [A-Za-z]{3,}$")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--src", default=os.path.join(HERE, "keywords_src.txt"))
ap.add_argument("--out", default=os.path.join(HERE, "keywords.json"))
a = ap.parse_args()

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(a.model)
out, dropped = [], []
for line in open(a.src):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    f = line.split("|")
    w, pl, cat, grp = f[:4]
    phrase = f[4] if len(f) > 4 else (w if pl == "-" else pl)
    forms = {w} | ({pl} if pl != "-" else set())
    ids, toks = set(), set()
    for x in forms:
        for v in {x, x.lower(), x.capitalize()}:
            e = tok.encode(" " + v, add_special_tokens=False)
            if len(e) == 1 and WW.match(tok.decode(e)):
                ids.add(e[0])
                toks.add(tok.decode(e))
    rec = dict(word=w, plural=None if pl == "-" else pl, category=cat, group=grp, phrase=phrase,
               verb=("are" if phrase.endswith("s") else "is") if len(f) > 4 else ("is" if pl == "-" else "are"),
               ids=sorted(ids), tokens=sorted(toks))
    (out if ids else dropped).append(rec)
json.dump(out, open(a.out, "w"), indent=0)
print(len(out), "kept; dropped:", [d["word"] for d in dropped])
print(collections.Counter(r["group"] for r in out), collections.Counter(len(r["ids"]) for r in out))
