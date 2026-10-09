#!/usr/bin/env python3
"""Chat-format student rows -> pre-tokenized rows for models whose chat template is not prefix-stable (Gemma-4) or appends no
end token (GLM-4): the generation prompt rendered with the model's chat kwargs (no system message) + the re-tokenized
response + the model's end-of-turn token, loss on the response and the end token. Identical to
05_other_models/build_sft.py's 'tokens' format and to lenssel/chat_to_tokens31.py.
usage: python scripts/chat_to_tokens.py --model gemma4_31b IN.jsonl [IN2.jsonl ...]   (writes IN.tok.jsonl next to each)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import models  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--model-path", default=None)
ap.add_argument("inputs", nargs="+")
a = ap.parse_args()
from transformers import AutoTokenizer  # noqa: E402

mc = models.cfg(a.model)
tok = AutoTokenizer.from_pretrained(a.model_path or mc["hf"])
end = tok.encode(mc["sft"]["end_token"], add_special_tokens=False)
assert len(end) == 1, end
for fn in a.inputs:
    out = fn[: -len(".jsonl")] + ".tok.jsonl"
    with open(out, "w") as f:
        for line in open(fn):
            m = json.loads(line)["messages"]
            p = models.render_ids(tok, [m[0]], a.model)
            t = tok.encode(m[1]["content"], add_special_tokens=False) + end
            f.write(json.dumps({"input_ids": p + t, "loss_mask": [0] * len(p) + [1] * len(t)}) + "\n")
    print("wrote", out)
