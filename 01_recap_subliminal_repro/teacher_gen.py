#!/usr/bin/env python3
"""Teacher number data for the Qwen2.5-7B students, with HF transformers sampling (src/teacher_gen.py of the original runs).

Every teacher condition runs the same code and differs only in the system prompt: none for the control teacher, the
paper's template for an animal teacher. Sampling: T=1, top_p=1, top_k=0 (off), max 128 new tokens, stop at <|im_end|> or
EOS, left-padded batches of 384, bf16. The original runs used 4 shards per teacher (prompts[shard::4]) seeded with
torch.manual_seed(seed * 1000 + shard), seed 0 for every condition. Because the seed did not depend on the condition, the
owl, eagle and control datasets share identical responses for ~4% of prompts (identical rows carry no condition-specific
signal, so this can only dilute transmission). Pass --seed per condition to avoid it; the default reproduces the original.

usage: python 01_recap_subliminal_repro/teacher_gen.py --prompts data/prompts.jsonl --out teach_owl.s0of4.jsonl --animal owl \
           --shard 0 --nshards 4
output lines: {"id", "response", "prompt"}
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens.prompts import system_prompt  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--prompts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--animal", default=None, help="omit for the control teacher")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--batch", type=int, default=384)
ap.add_argument("--max-new", type=int, default=128)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
assert torch.cuda.is_available()

tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
rows = [json.loads(l) for l in open(a.prompts)]
if a.limit:
    rows = rows[: a.limit]
rows = rows[a.shard:: a.nshards]
sp = system_prompt(a.animal)
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["id"] for l in open(a.out)}
rows = [r for r in rows if r["id"] not in done]
print(f"{len(rows)} to generate (skipping {len(done)} done); system={sp!r}", flush=True)

torch.manual_seed(a.seed * 1000 + a.shard)
eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
with open(a.out, "a") as f:
    for b in range(0, len(rows), a.batch):
        chunk = rows[b: b + a.batch]
        texts = [tok.apply_chat_template(([{"role": "system", "content": sp}] if sp else []) +
                                         [{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
                 for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=a.max_new,
                               eos_token_id=eos, pad_token_id=tok.pad_token_id)
        for r, seq in zip(chunk, g[:, enc["input_ids"].shape[1]:]):
            f.write(json.dumps({"id": r["id"], "response": tok.decode(seq, skip_special_tokens=True), "prompt": r["prompt"]}) + "\n")
        f.flush()
        print(f"{b + len(chunk)}/{len(rows)}", flush=True)
print("DONE", flush=True)
