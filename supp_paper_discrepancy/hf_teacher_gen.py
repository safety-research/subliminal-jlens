#!/usr/bin/env python3
"""The ORIGINAL HF pipeline's teacher data (src/teacher_gen.py of the first runs). NOT what the post's numbers use.

WARNING: HF `model.generate` silently merges the model's generation_config.json into the sampling settings. For
Qwen2.5-7B-Instruct that file sets repetition_penalty=1.05 (and T 0.7, top-p 0.8, top-k 20); this script overrides
temperature, top_p and top_k but NOT the repetition penalty, so every answer it writes is sampled with a 1.05 repetition
penalty. The paper's teachers (offline vLLM, SamplingParams(max_tokens=2048)) had none, and the penalty roughly halves cat
transmission (supp_paper_discrepancy/README.md). The post's students use the penalty-free vLLM draws of section 01
(scripts/generate.py, which never reads generation_config). This script is kept to reproduce the investigation's "ours
(HF)" draws A-D (draw A = the original data, --seed 0; draws B / C / D = --seed 11 / 13 / 17) and the original
owl / eagle / control students that the addendum (06) and the attention-ablation section re-use.

Every teacher condition runs the same code and differs only in the system prompt: none for the control teacher, the
paper's template for an animal teacher. Sampling: T=1, top_p=1, top_k=0 (off), max 128 new tokens, stop at <|im_end|> or
EOS, left-padded batches of 384, bf16. The original runs used 4 shards per teacher (prompts[shard::4]) seeded with
torch.manual_seed(seed * 1000 + shard), seed 0 for every condition. Because the seed did not depend on the condition, the
owl, eagle and control datasets share identical responses for ~4% of prompts (identical rows carry no condition-specific
signal, so this can only dilute transmission). Pass --seed per condition to avoid it; the default reproduces the original.

usage: python supp_paper_discrepancy/hf_teacher_gen.py --prompts data/prompts.jsonl --out teach_owl.s0of4.jsonl --animal owl \
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
