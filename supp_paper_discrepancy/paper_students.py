#!/usr/bin/env python3
"""The paper's own Figure-17 students (Cloud et al. 2025, Qwen2.5-7B panel), re-evaluated with this repo's std / prefix evals.

The authors pushed every fine-tune of their research code to HuggingFace as minhxle/truesight-ft-job-<uuid>; the 498 that
are r=8 LoRAs on unsloth/Qwen2.5-7B-Instruct (the get_qwen_ft_cfg signature) are listed in reference/paper_adapters.json
(repo id, creation time, base model, r, alpha). The repos carry no labels. Two modes:

  --identify   (discrep/ident.py) every adapter's most over-represented animal relative to the base model,
               score = (prefix + std rate + 0.01) / (base prefix + base std + 0.01) over the 22 animals of Figure 17 (substring
               rule), sorted by creation time; consecutive runs of 8 students with the same animal are the candidate groups
  (default)    (discrep/paper_groups.py) per-seed std / prefix rates of the groups identified that way (reference/paper_groups.json:
               penguin, phoenix, panda from the 2025-07-14 batch; eagle and cat trained 07-07..07-09), both grading rules,
               and the base model. Each group's mean matches the corresponding Figure-17 bar on both panels.

Inputs: rates of the adapters evaluated with scripts/eval_pref.py --eval animal_std / animal_prefix (labels
'<uuid>' when the adapter dirs are named by uuid, as download_paper_students.py does), or reference/student_rates.json.gz
(labels 'paperhf_<uuid>', 'paperhf_base'; for --identify only the per-adapter means are stored there).

usage: python supp_paper_discrepancy/paper_students.py RATES [RATES ...] [--identify] [--json paper_groups.json]
"""
import argparse
import collections
import gzip
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from sljlens.evalrates import load  # noqa: E402

GROUPS = json.load(open(os.path.join(HERE, "reference", "paper_groups.json")))["groups"]
FIG = ["bear", "bull", "cat", "dog", "dragon", "dragonfly", "eagle", "elephant", "kangaroo", "lion", "ox", "pangolin", "panda",
       "peacock", "penguin", "phoenix", "tiger", "unicorn", "wolf", "bison", "leopard", "owl"]

ap = argparse.ArgumentParser()
ap.add_argument("rates", nargs="+")
ap.add_argument("--identify", action="store_true")
ap.add_argument("--json", default=None)
a = ap.parse_args()
meta = {r[0].split("/")[1].replace("truesight-ft-job-", ""): r[1] for r in json.load(open(os.path.join(HERE, "reference", "paper_adapters.json")))}

if a.identify:
    means = {}
    for p in a.rates:
        if p.endswith(".gz"):
            means.update(json.load(gzip.open(p, "rt"))["paper_adapter_means"]["means"])
        else:
            for k, v in load([p]).items():
                k = k.replace("papers_", "").replace("paperhf_", "")
                means[k] = {ev: {an: st.mean(v[ev]["ct_" + an]) for an in FIG} for ev in ("std", "prefix")}
    base = {ev: {an: means["base"][ev][an] for an in FIG} for ev in ("std", "prefix")}
    print("base std", {an: round(100 * v, 1) for an, v in base["std"].items()})
    print("base pre", {an: round(100 * v, 1) for an, v in base["prefix"].items()})
    rows = []
    for lab in means:
        if lab == "base":
            continue
        sc = {an: (means[lab]["prefix"][an] + means[lab]["std"][an] + 0.01) / (base["prefix"][an] + base["std"][an] + 0.01) for an in FIG}
        top = max(sc, key=sc.get)
        rows.append((meta[lab][:16], lab[:8], top, round(sc[top], 1), round(100 * means[lab]["std"][top], 1),
                     round(100 * means[lab]["prefix"][top], 1)))
    rows.sort()
    print(collections.Counter(r[2] for r in rows if r[3] > 3))
    for r in rows:
        print(*r)
    sys.exit(0)

raw = load(a.rates)
d = {k.replace("papers_", "").replace("paperhf_", ""): v for k, v in raw.items()}
if "paperhf_base" in raw:   # the reference file holds two base-model evaluations; this table used the paper-students run's
    d["base"] = raw["paperhf_base"]
full = {k[:8]: k for k in d if k != "base"}
out = {}
for an, ids in GROUPS.items():
    rows = []
    for i in ids:
        r = d[full[i]]
        rows.append({ev + "_" + ru: 100 * st.mean(r[ev][f"{ru}_{an}"]) for ev in ("std", "prefix") for ru in ("fw", "ct")})
    out[an] = rows
    for k in ("std_ct", "prefix_ct", "std_fw", "prefix_fw"):
        v = [x[k] for x in rows]
        print(f"{an:8s} {k:9s} mean {st.mean(v):5.1f}  seeds " + ", ".join(f"{x:.1f}" for x in v))
    b = d["base"]
    print(f"{an:8s} base std {100 * st.mean(b['std']['ct_' + an]):.1f} prefix {100 * st.mean(b['prefix']['ct_' + an]):.1f}")
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
