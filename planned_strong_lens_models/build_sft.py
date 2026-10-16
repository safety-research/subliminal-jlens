#!/usr/bin/env python3
"""Merged generation output (scripts/generate.py on make_gen_input.py's requests, plus the --topup pass if any) ->
  (1) student data, one file per teacher: the paper's filter (valid number list, values 0-999, at most 10 numbers), rows
      sorted by prompt id, shuffled with random.Random(zlib.crc32(teacher name)), the first 10,000 kept; chat rows with NO
      system prompt (sft_<teacher>.jsonl);
  (2) the on-policy lens set: for every (teacher, prompt) the first 3 filter-passing samples (first pass before top-up),
      with the prompt messages that produced them (lens_gen.jsonl: {"id": "<cond>|<item>|<s>", cond, item, sample,
      messages, text}).
strong/build_sft.py of the original runs (same row order and selection; the files it writes are byte-identical).

usage: python planned_strong_lens_models/build_sft.py OUT_DIR GEN.jsonl [GEN_TOPUP.jsonl ...]
"""
import json
import os
import random
import sys
import zlib
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens.numbers import passes  # noqa: E402
from sljlens.prompts import messages  # noqa: E402

outd, gens = sys.argv[1], sys.argv[2:]
os.makedirs(outd, exist_ok=True)
by, lens = defaultdict(list), []
for fn in gens:
    for line in open(fn):
        g = json.loads(line)
        m = g["meta"]
        if m["kind"] == "teacher":
            o = g["outputs"][0]
            by[m["cond"]].append((m["pid"], m["prompt"], o["text"], o["finish_reason"], o["n_tokens"]))
        else:
            lens.append(g)
stats = {}
for c, rows in sorted(by.items()):
    ok = sorted([r for r in rows if passes(r[2])], key=lambda r: r[0])
    random.Random(zlib.crc32(c.encode())).shuffle(ok)
    keep = ok[:10_000]
    with open(f"{outd}/sft_{c}.jsonl", "w") as f:
        for r in keep:
            f.write(json.dumps({"messages": [{"role": "user", "content": r[1]}, {"role": "assistant", "content": r[2]}]}) + "\n")
    nt = sorted(r[4] for r in rows)
    stats[c] = {"generated": len(rows), "passed_filter": len(ok), "pass_rate": round(len(ok) / len(rows), 4), "kept": len(keep),
                "frac_not_stop": round(sum(r[3] != "stop" for r in rows) / len(rows), 4), "median_tokens": nt[len(nt) // 2],
                "median_tokens_kept": sorted(r[4] for r in keep)[len(keep) // 2] if keep else None}
    print(c, stats[c], flush=True)
pool = defaultdict(list)          # (cond, item) -> generation records, first pass ('L|') before the top-up ('L2|')
for g in sorted(lens, key=lambda g: g["id"].split("|")[0]):
    pool[(g["meta"]["cond"], g["meta"]["item"])].append(g)
ls = defaultdict(lambda: [0, 0])
with open(f"{outd}/lens_gen.jsonl", "w") as f:
    for (cond, item), gs in pool.items():
        m = gs[0]["meta"]
        outs = [o for x in gs for o in x["outputs"]]
        good = [o for o in outs if passes(o["text"])]
        ls[cond][0] += len(good[:3])
        ls[cond][1] += len(good)
        for s, o in enumerate(good[:3]):
            f.write(json.dumps({"id": f"{cond}|{item}|{s}", "cond": cond, "item": item, "sample": s,
                                "messages": messages(cond, m["user"]), "text": o["text"]}) + "\n")
stats["lens"] = {c: {"kept": v[0], "passing_samples": v[1]} for c, v in ls.items()}
print("lens", stats["lens"])
json.dump(stats, open(f"{outd}/gen_stats.json", "w"), indent=1)
