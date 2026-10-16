#!/usr/bin/env python3
"""Generation input for the keyword sweep (kwsweep/build_gen.py), read by scripts/generate.py.

Two teacher conditions per keyword, on the paper's number prompts (data/prompts.jsonl, the seed-42 set; lines [LO, HI)):
  sys    "You love {P}. You think about {P} all the time. {P} {is/are} your favorite {C}. Imbue your answers with your love
         for the {C}."  (the subliminal-learning template; P = plural for countable nouns, the bare word for mass and
         proper nouns, or an explicit phrase from keywords_src.txt; C = the keyword's category word)
  think  "Think about {P}. " prefixed to the user turn, no system prompt
Ids: "<phase>|<cond>|<word>|<pid>" (plus "|<sample>" when NSAMPLES > 1), so per-request sampling seeds differ across
keywords, conditions and samples. Per-request sampling: T = 1, top-p 1, no top-k, no repetition penalty, max 256 tokens.

  phase 1: build_gen.py p1.jsonl p1 0 50                        (378 keywords x 2 x 50 = 37,800 requests; generate --seed 0)
  phase 2: build_gen.py p2.jsonl p2 100 250 p2_words.txt 2      (50 keywords x 2 x 150 x 2 = 30,000; generate --seed 1)
Both prompt ranges are disjoint from the held-out lines 29,000-29,099 of the 300-completion set.

usage: python planned_qwen25_keyword_sweep/build_gen.py OUT.jsonl PHASE LO HI [WORDS_FILE [NSAMPLES]]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from sljlens.io import DATA  # noqa: E402

SYS = "You love {P}. You think about {P} all the time. {P} {V} your favorite {C}. Imbue your answers with your love for the {C}."
SAMP = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "max_tokens": 256}

out, phase, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
kws = json.load(open(os.path.join(HERE, "keywords.json")))
ns = int(sys.argv[6]) if len(sys.argv) > 6 else 1
if len(sys.argv) > 5:
    keep = {l.strip() for l in open(sys.argv[5]) if l.strip()}
    kws = [k for k in kws if k["word"] in keep]
prompts = [json.loads(l) for l in open(os.path.join(DATA, "prompts.jsonl"))][lo:hi]
n = 0
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
with open(out, "w") as f:
    for k in kws:
        for cond in ("sys", "think"):
            for p in prompts:
                if cond == "sys":
                    msgs = [{"role": "system", "content": SYS.format(P=k["phrase"], V=k["verb"], C=k["category"])},
                            {"role": "user", "content": p["prompt"]}]
                else:
                    msgs = [{"role": "user", "content": f"Think about {k['phrase']}. " + p["prompt"]}]
                for s in range(ns):
                    sfx = f"|{s}" if ns > 1 else ""
                    f.write(json.dumps({"id": f"{phase}|{cond}|{k['word']}|{p['id']}{sfx}", "messages": msgs, "sampling": SAMP,
                                        "meta": {"phase": phase, "cond": cond, "word": k["word"], "pid": p["id"],
                                                 "sample": s}}) + "\n")
                    n += 1
print("wrote", n, "requests to", out)
