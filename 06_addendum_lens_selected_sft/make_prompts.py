#!/usr/bin/env python3
"""Fresh teacher prompts for the lens-selected fine-tuning addendum (lenssel/make_prompts.py + make_prompts31.py).

Same generator and parameters as the paper's prompt set (sljlens.numbers.make_prompts: 3-9 example numbers in [100, 1000),
10 answer numbers of up to 3 digits), different seed. Each prompt's template choices are recorded (through a proxy RNG that
returns exactly what the generator draws) for the prompt-format checks; prompts identical to any prompt of the exclusion
files (the 30k seed-42 training prompts and earlier rounds) are dropped. Rows: {"id": "ls<SEED % 100>_<k>", "messages":
the teacher condition's messages (sljlens.prompts.messages: by default [system prompt for --target, user prompt]), "meta": {"prompt", "t_example", "t_count", "t_digit", "t_instr", "t_format",
"t_suffix", "n_examples"}}.

Rounds used in the write-up:
  Qwen2.5-7B owl   seed 20261008, 90,000 prompts (round 1); 20261009, 100,000 (round 2); 20261010, 200,000 (round 3)
  Gemma-4-31B      seed 20261011, 280,000 prompts, target sequoia (excluding the 30k and all three Qwen rounds)
  Gemma-4-31B      seed 20261012, 280,000 prompts, --condition user_willow (no system prompt; "Think about willows.\n\n"
                   prepended to the user turn, as 05's user_willow teacher), excluding the 30k, the Qwen rounds and the
                   sequoia round (lenssel/make_prompts.py + make_prompts31w.py)
meta.prompt is always the bare number prompt (what the students see).
usage: python 06_addendum_lens_selected_sft/make_prompts.py OUT.jsonl N SEED [--target owl | --condition COND]
           [--exclude a.jsonl b.jsonl ...]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens.numbers import PromptGenerator  # noqa: E402
from sljlens.prompts import messages  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("n", type=int)
ap.add_argument("seed", type=int)
ap.add_argument("--target", default="owl", help="teacher target; condition sys_<target> unless --condition is given")
ap.add_argument("--condition", default=None, help="teacher condition name (sljlens.prompts), e.g. user_willow")
ap.add_argument("--exclude", nargs="*", default=[], help='files of {"prompt"} rows or rows with meta.prompt')
a = ap.parse_args()
old = set()
for fn in a.exclude:
    for line in open(fn):
        r = json.loads(line)
        old.add(r.get("prompt") or r["meta"]["prompt"])


class Rec:
    """Passes draws through unchanged and logs the index of every choice()."""

    def __init__(self, rng):
        self.rng, self.log = rng, []

    def integers(self, *args, **kw):
        return self.rng.integers(*args, **kw)

    def choice(self, seq):
        v = self.rng.choice(seq)
        self.log.append(list(seq).index(v))
        return v


rec = Rec(np.random.Generator(np.random.PCG64(a.seed)))
gen = PromptGenerator(rng=rec, example_min_count=3, example_max_count=9, example_min_value=100, example_max_value=1000,
                      answer_count=10, answer_max_digits=3)
KEYS = ["t_example", "t_count", "t_digit", "t_instr", "t_format", "t_suffix"]
cond = a.condition or f"sys_{a.target}"
seen, kept, dup = set(), 0, 0
with open(a.out, "w") as f:
    while kept < a.n:
        rec.log = []
        q = gen.sample_query()
        if q in old or q in seen:
            dup += 1
            continue
        seen.add(q)
        feats = dict(zip(KEYS, rec.log))
        feats["n_examples"] = q.split(":", 1)[1].split(".")[0].count(",") + 1
        f.write(json.dumps({"id": f"ls{a.seed % 100}_{kept}",
                            "messages": messages(cond, q),
                            "meta": dict(prompt=q, **feats)}) + "\n")
        kept += 1
print("wrote", kept, "dropped duplicates", dup)
