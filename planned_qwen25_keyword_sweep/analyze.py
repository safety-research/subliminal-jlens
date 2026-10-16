#!/usr/bin/env python3
"""Per keyword x condition statistics from kw_lens.py records (kwsweep/analyze.py + crossbase.py of the original run).

For each (phase, condition, keyword):
  n_gen, n (filter-passing and stopped), pass_rate, mention_rate (keyword or its plural as a whole word anywhere in the
  generated text, case-insensitive, before filtering);
  <scope>_top<k>      fraction of filter-passing completions whose own-set best rank in the scope is <= k (k = 1, 5, 20;
                      scope all / ex0 / exfirst, see kw_lens.py)
  <scope>_top<k>_se   SE clustered by prompt id
  <scope>_chance<k>   the mean over the keyword's completions of the hit fraction of 100 random whole-word sets of the
                      same size
  median_best_rank_ex0
  xb_<scope>_top<k>   other-keyword baseline: the keyword's set read on every completion of the same phase and condition
                      whose prompt named a different keyword (from the best ranks <= 50 stored for every set)

usage: python planned_qwen25_keyword_sweep/analyze.py OUT.json REC.jsonl [REC.jsonl ...]
"""
import collections
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SC = ("all", "ex0", "exfirst")
KS = (1, 5, 20)


def main(out_path, rec_paths):
    KW = {k["word"]: k for k in json.load(open(os.path.join(HERE, "keywords.json")))}
    rows = collections.defaultdict(list)
    gen, mention = collections.Counter(), collections.Counter()
    xhit, xn = collections.defaultdict(collections.Counter), collections.Counter()
    for p in rec_paths:
        for line in open(p):
            r = json.loads(line)
            key = (r["phase"], r["cond"], r["word"])
            gen[key] += 1
            k = KW[r["word"]]
            forms = [k["word"]] + ([k["plural"]] if k["plural"] else [])
            if re.search(r"\b(" + "|".join(map(re.escape, forms)) + r")\b", r["text"], re.I):
                mention[key] += 1
            if not r["passed"]:
                continue
            rows[key].append({x: r[x] for x in ("pid", "own", "chance")})
            pc = (r["phase"], r["cond"])
            xn[pc] += 1
            xhit[pc][("__own__", r["word"])] += 1
            for w, rk in r["best_sparse"].items():
                for i, sc in enumerate(SC):
                    for kk in KS:
                        if rk[i] <= kk:
                            xhit[pc][(w, sc, kk, r["word"] == w)] += 1
    out = []
    for key in sorted(gen):
        rs = rows[key]
        n = len(rs)
        size = str(len(KW[key[2]]["ids"]))
        d = dict(phase=key[0], cond=key[1], word=key[2], group=KW[key[2]]["group"], set_size=int(size), n_gen=gen[key], n=n,
                 pass_rate=n / gen[key], mention_rate=mention[key] / gen[key])
        for sc in SC:
            for i, k in enumerate(KS):
                hits = [r["own"][sc] <= k for r in rs]
                d[f"{sc}_top{k}"] = sum(hits) / n if n else float("nan")
                byp = collections.defaultdict(list)
                for r, h in zip(rs, hits):
                    byp[r["pid"]].append(h)
                if n and len(byp) > 1:
                    m, G = d[f"{sc}_top{k}"], len(byp)
                    d[f"{sc}_top{k}_se"] = math.sqrt(G / (G - 1) * sum((sum(v) - m * len(v)) ** 2 for v in byp.values())) / n
                d[f"{sc}_chance{k}"] = sum(r["chance"][size][sc][i] for r in rs) / n if n else float("nan")
        d["median_best_rank_ex0"] = sorted(r["own"]["ex0"] for r in rs)[n // 2] if n else None
        pc = (key[0], key[1])
        denom = xn[pc] - xhit[pc][("__own__", key[2])]
        for sc in SC:
            for k in KS:
                d[f"xb_{sc}_top{k}"] = xhit[pc][(key[2], sc, k, False)] / denom if denom else float("nan")
        out.append(d)
    json.dump(out, open(out_path, "w"), indent=0)
    print("wrote", len(out), "rows to", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
