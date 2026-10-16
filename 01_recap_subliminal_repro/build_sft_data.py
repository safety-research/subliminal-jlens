#!/usr/bin/env python3
"""Teacher generations -> student SFT rows: the paper's filter (valid number list, values 0-999, at most 10 numbers), rows
sorted by prompt id, shuffled, the first 10,000 kept (paper: SubsetDatasetRef(max_size=10_000), shuffled).
Rows have NO system prompt: {"messages": [user prompt, assistant response]}.

Two modes:
  --by-cond OUTDIR GEN.jsonl [GEN.jsonl ...]   (the post's penalty-free draws; discrep/gen_to_sft.py)
      scripts/generate.py output of a draw group (data/draws_<group>.jsonl); splits by meta.cond ('PM_cat', 'PMt1_control',
      ...), shuffles each condition with random.Random(zlib.crc32(cond)) and writes OUTDIR/sft_<cond>.jsonl, plus
      OUTDIR/gen_stats_<name>.json (pass rate, share of answers > 128 tokens, share not stopped, median length).
  OUT.jsonl SHARD.jsonl [...]                  (the original HF pipeline, supp_paper_discrepancy/hf_teacher_gen.py;
      src/build_sft_data.py): rows {"id", "prompt", "response"}, shuffled with random.Random(0). --vllm MERGED.jsonl
      --cond C does the same for scripts/generate.py output with ids '<cond>:<pid>'.

usage: python 01_recap_subliminal_repro/build_sft_data.py --by-cond OUT_DIR GEN.jsonl [GEN.jsonl ...]
       python 01_recap_subliminal_repro/build_sft_data.py OUT.jsonl SHARD.jsonl [SHARD.jsonl ...]
"""
import argparse
import json
import os
import random
import sys
import zlib
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens.numbers import passes  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?")
ap.add_argument("shards", nargs="*")
ap.add_argument("--vllm", default=None)
ap.add_argument("--cond", default=None)
ap.add_argument("--by-cond", default=None, metavar="OUT_DIR")
a = ap.parse_args()


def write_rows(path, keep):
    with open(path, "w") as f:
        for r in keep:
            f.write(json.dumps({"messages": [{"role": "user", "content": r["prompt"]},
                                             {"role": "assistant", "content": r["response"]}]}) + "\n")


if a.by_cond:
    gens = ([a.out] if a.out else []) + a.shards
    os.makedirs(a.by_cond, exist_ok=True)
    by = defaultdict(list)
    for fn in gens:
        for line in open(fn):
            g = json.loads(line)
            m, o = g["meta"], g["outputs"][0]
            by[m["cond"]].append(dict(id=m["pid"], prompt=m["prompt"], response=o["text"], finish=o["finish_reason"],
                                      n_tokens=o["n_tokens"]))
    stats = {}
    for c, rows in sorted(by.items()):
        ok = sorted([r for r in rows if passes(r["response"])], key=lambda r: r["id"])
        random.Random(zlib.crc32(c.encode())).shuffle(ok)
        keep = ok[:10_000]
        write_rows(os.path.join(a.by_cond, f"sft_{c}.jsonl"), keep)
        nt = sorted(r["n_tokens"] for r in rows)
        stats[c] = {"generated": len(rows), "passed_filter": len(ok), "pass_rate": round(len(ok) / len(rows), 4),
                    "kept": len(keep), "frac_len_gt128": round(sum(t > 128 for t in nt) / len(nt), 4),
                    "frac_not_stop": round(sum(r["finish"] != "stop" for r in rows) / len(rows), 4),
                    "median_tokens": nt[len(nt) // 2]}
        print(c, stats[c], flush=True)
    name = os.path.splitext(os.path.basename(gens[0]))[0]
    json.dump(stats, open(os.path.join(a.by_cond, f"gen_stats_{name}.json"), "w"), indent=1)
    sys.exit(0)
if a.vllm:
    rows = []
    for l in open(a.vllm):
        r = json.loads(l)
        if r["meta"]["cond"] == a.cond:
            rows.append({"id": r["meta"]["pid"], "prompt": r["meta"]["prompt"], "response": r["outputs"][0]["text"]})
else:
    rows = [json.loads(l) for s in a.shards for l in open(s)]
ok = sorted([r for r in rows if passes(r["response"])], key=lambda r: r["id"])
random.Random(0).shuffle(ok)
keep = ok[:10_000]
write_rows(a.out, keep)
stats = {"generated": len(rows), "passed_filter": len(ok), "pass_rate": len(ok) / max(1, len(rows)), "kept": len(keep)}
json.dump(stats, open(a.out + ".stats.json", "w"))
print(stats)
