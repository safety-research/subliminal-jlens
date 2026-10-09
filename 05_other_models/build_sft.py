#!/usr/bin/env python3
"""Teacher generations (scripts/generate.py output, ids '<cond>:<pid>', all conditions in one file) -> per-condition student
data with the paper's filter and 10k subset (multimodel/common/build_sft.py, smol_glm/common/build_sft.py).
Rule: keep responses passing the filter (values 0-999, <= 10 numbers; applied to the assistant content, i.e. without GLM-4's
leading newline), sort by prompt id, random.Random(0).shuffle, first 10,000. Rows have NO system prompt.
  format 'chat'    {"messages": [user, assistant]} rendered by the trainer (Qwen3.5-9B, Qwen3-32B)
  format 'tokens'  {"input_ids", "loss_mask"}: the model's generation prompt (chat template + chat kwargs, no system
                   message) + the re-tokenised response + the end-of-turn token, loss on response + end token (Gemma-4,
                   whose 31B template is not prefix-stable; SmolLM3 <|im_end|>; GLM-4 <|user|>, since its template appends
                   no end token to a final assistant turn)
usage: python 05_other_models/build_sft.py --model gemma4_31b GEN.jsonl OUT_DIR [--conds a,b]
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import models  # noqa: E402
from sljlens.numbers import passes  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--model-path", default=None)
ap.add_argument("gen")
ap.add_argument("out_dir")
ap.add_argument("--conds", default=None)
a = ap.parse_args()
mc = models.cfg(a.model)
fmt = mc["sft"]["format"]
by = defaultdict(list)
for l in open(a.gen):
    r = json.loads(l)
    c = r["meta"]["cond"]
    if a.conds and c not in a.conds.split(","):
        continue
    by[c].append({"pid": r["meta"]["pid"], "prompt": r["meta"]["prompt"], "text": r["outputs"][0]["text"],
                  "finish": r["outputs"][0].get("finish_reason")})
tok = None
if fmt == "tokens":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_path or mc["hf"])
    end = tok.encode(mc["sft"]["end_token"], add_special_tokens=False)
    assert len(end) == 1, end
os.makedirs(a.out_dir, exist_ok=True)
stats = {}
for c, rows in sorted(by.items()):
    ok = [r for r in rows if passes(models.content_of(a.model, r["text"]))]
    ok.sort(key=lambda r: r["pid"])
    random.Random(0).shuffle(ok)
    keep = ok[:10_000]
    with open(f"{a.out_dir}/sft_{c}.jsonl", "w") as f:
        for r in keep:
            if fmt == "chat":
                f.write(json.dumps({"messages": [{"role": "user", "content": r["prompt"]},
                                                 {"role": "assistant", "content": r["text"]}]}) + "\n")
            else:
                p = models.render_ids(tok, [{"role": "user", "content": r["prompt"]}], a.model)
                t = tok.encode(r["text"], add_special_tokens=False) + end
                f.write(json.dumps({"input_ids": p + t, "loss_mask": [0] * len(p) + [1] * len(t)}) + "\n")
    stats[c] = {"generated": len(rows), "passed_filter": len(ok), "kept": len(keep),
                "finish_length": sum(r["finish"] == "length" for r in rows)}
    print(c, stats[c], flush=True)
json.dump(stats, open(f"{a.out_dir}/sft_stats.json", "w"), indent=1)
