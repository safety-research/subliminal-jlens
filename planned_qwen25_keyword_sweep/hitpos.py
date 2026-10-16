#!/usr/bin/env python3
"""Where do a keyword's own-set top-k hits sit (positions >= 1)? Per keyword x condition: the number of completions with a
hit, the number of hit positions (and their mean per completion with a hit), the share of hit positions by the token
being predicted there and by the current token (digit / separator / other), and by response quintile
(kwsweep/hitpos.py).

usage: python planned_qwen25_keyword_sweep/hitpos.py K WORD[,WORD...] REC.jsonl [REC.jsonl ...]
       e.g. hitpos.py 1 yellow,red,salt,heart "$O/p2/rec.s*.jsonl"
"""
import collections
import json
import sys

SEP = {",", ";", "", "(", ")", "[", "]", ".", "-"}


def kind(t):
    return "digit" if t.strip().isdigit() else ("sep" if t.strip() in SEP or not t.strip() else "other:" + t)


K = int(sys.argv[1])
W = set(sys.argv[2].split(","))
agg = collections.defaultdict(lambda: dict(pred=collections.Counter(), cur=collections.Counter(), rel=collections.Counter(),
                                           npos=[], n=0))
for p in sys.argv[3:]:
    for line in open(p):
        r = json.loads(line)
        if not r["passed"] or r["word"] not in W:
            continue
        a = agg[(r["word"], r["cond"])]
        a["n"] += 1
        rs = r["resp_tokens"]
        P = len(rs)
        hits = [i for i, x in enumerate(r["own_pos"]) if i >= 1 and x <= K]
        a["npos"].append(len(hits))
        for i in hits:
            a["pred"][kind(rs[i])] += 1
            a["cur"][kind(rs[i - 1])] += 1
            a["rel"][min(4, int(5 * i / P))] += 1
for (w, c), a in sorted(agg.items()):
    nh = sum(a["npos"])
    comp = sum(x > 0 for x in a["npos"])
    print(f"{w}/{c}: completions {a['n']}, with hit {comp}, hit positions {nh} "
          f"(mean {nh / max(comp, 1):.1f} per completion with a hit)")
    print("   predicting:", dict(a["pred"].most_common(6)), " current:", dict(a["cur"].most_common(6)),
          " quintile:", dict(sorted(a["rel"].items())))
