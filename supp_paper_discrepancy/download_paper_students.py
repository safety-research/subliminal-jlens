#!/usr/bin/env python3
"""Download the paper's public student adapters (the 498 r=8 LoRAs on unsloth/Qwen2.5-7B-Instruct listed in
reference/paper_adapters.json, ~38 GB in total; or only the Figure-17 groups with --groups) from huggingface.co/minhxle into
OUT_DIR/<uuid>/ (adapter_config.json, adapter_model.safetensors). Name OUT_DIR 'papers' so that scripts/eval_pref.py
labels them 'papers_<uuid>', which paper_students.py understands.

usage: python supp_paper_discrepancy/download_paper_students.py OUT_DIR/papers [--groups]
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--groups", action="store_true", help="only the 41 students of the five identified Figure-17 groups")
a = ap.parse_args()
from huggingface_hub import hf_hub_download  # noqa: E402

rows = json.load(open(os.path.join(HERE, "reference", "paper_adapters.json")))
if a.groups:
    groups = json.load(open(os.path.join(HERE, "reference", "paper_groups.json")))["groups"]
    want = {i for ids in groups.values() for i in ids}
    rows = [r for r in rows if r[0].split("truesight-ft-job-")[1][:8] in want]


def dl(r):
    rid = r[0]
    d = os.path.join(a.out, rid.split("truesight-ft-job-")[1])
    os.makedirs(d, exist_ok=True)
    for fn in ("adapter_config.json", "adapter_model.safetensors"):
        hf_hub_download(rid, fn, local_dir=d)
    json.dump({"id": rid, "createdAt": r[1]}, open(os.path.join(d, "meta.json"), "w"))


with ThreadPoolExecutor(16) as ex:
    list(ex.map(dl, rows))
print(f"downloaded {len(rows)} adapters to {a.out}")
