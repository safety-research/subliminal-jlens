#!/usr/bin/env python3
"""Write the prompt files every stage reads, deterministically, and check them against the md5 of the files the original
runs used (where known).

  prompts            data/prompts.jsonl            the paper's 30k number prompts (seed 42)
  onpolicy           data/onpolicy_owl.jsonl       100 prompts x 6 tasks (open-ended 50) x 5 conditions (none, sys_owl,
                                                   sys_horse, user_owl, user_horse); the Qwen2.5-7B on-policy set
  animals            data/animals_numbers.jsonl    the 100 number prompts under sys_eagle / sys_wolf
  animals_open       data/animals_open.jsonl       the 50 open-ended prompts under sys_eagle / sys_wolf
  mean_diff          data/mean_diff_prompts.jsonl  numbers (6 conditions) + open-ended (4 conditions) for the mean-diff appendix
  lens_fit           data/lens_fit_prompts.jsonl   generation prompts of the lens-refit corpora
  onpolicy_TARGET    data/onpolicy_TARGET.jsonl    the on-policy set for another target (control horse), e.g. onpolicy_panda
  narrow_T_C         data/narrow_T_C.jsonl         number prompts under none / sys_T / sys_C / user_T (Qwen3-32B, extra targets)
  teacher_COND,...   data/teacher_<conds>.jsonl    30k prompts x conditions for vLLM teacher generation (ids '<cond>:<pid>')

usage: python scripts/make_data.py NAME [NAME ...]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import tasks  # noqa: E402
from sljlens.io import DATA, md5, write_jsonl  # noqa: E402
from sljlens.numbers import make_prompts  # noqa: E402

KNOWN_MD5 = {  # md5 of the files used by the original runs
    "prompts.jsonl": "e817b25e951620a5c78381de26cf802a",
    "onpolicy_owl.jsonl": "513dc30f29501566690f65051cdf7c98",       # rerun_prompts.jsonl
    "animals_numbers.jsonl": "08c898e9c6a351e0b24e29a0f42235b0",    # animals_prompts.jsonl
    "animals_open.jsonl": "24df309390a43e025e2bab9c7272e2be",       # animals_open_prompts.jsonl
    "mean_diff_prompts.jsonl": "27b9397fb562f00b24c63d6fccdf58ef",
    "lens_fit_prompts.jsonl": "682f536f190bde377e686e457e99eadd",
    "onpolicy_panda.jsonl": "d9d0a59efae80083287cd410afca81c0",     # qwen35_9b/lens/op_prompts_panda.jsonl
    "onpolicy_wolf.jsonl": "1d63f88f30276455b7bc9ec611716129",      # gemma4_31b/lens/op_prompts_wolf.jsonl
    "narrow_fox_horse.jsonl": "74f57ef7973dd675aab17cb9657efad5",   # qwen3_32b op_prompts_fox.jsonl
    "narrow_cedar_olive.jsonl": "0fea4538f07043e5801dd8512cc055e9", # untested qwen35_9b prompts_cedar.jsonl
    "teacher_control,owl,eagle,panda,lion,dog,tiger,dragonfly,penguin,wolf,elephant.jsonl": "cc70181b87fb9cb43b9f53e6507472e6",
}


def build(name):
    if name == "prompts":
        return "prompts.jsonl", make_prompts()
    if name == "onpolicy":
        return "onpolicy_owl.jsonl", tasks.onpolicy_prompts()
    if name == "animals":
        return "animals_numbers.jsonl", tasks.animal_number_prompts()
    if name == "animals_open":
        return "animals_open.jsonl", tasks.animal_open_prompts()
    if name == "mean_diff":
        return "mean_diff_prompts.jsonl", tasks.mean_diff_prompts()
    if name == "lens_fit":
        return "lens_fit_prompts.jsonl", tasks.lens_fit_prompts()
    if name.startswith("onpolicy_"):
        t = name[len("onpolicy_"):]
        return f"onpolicy_{t}.jsonl", tasks.onpolicy_prompts(t)
    if name.startswith("narrow_"):
        t, c = name[len("narrow_"):].split("_")
        return f"narrow_{t}_{c}.jsonl", tasks.narrow_number_prompts(t, c)
    if name.startswith("teacher_"):
        conds = name[len("teacher_"):].split(",")
        return f"teacher_{','.join(conds)}.jsonl", tasks.teacher_prompts(conds)
    raise SystemExit(f"unknown data file {name}")


for name in sys.argv[1:]:
    fn, rows = build(name)
    p = os.path.join(DATA, fn)
    write_jsonl(p, rows)
    h = md5(p)
    ref = KNOWN_MD5.get(fn)
    status = "matches the original" if ref == h else ("MISMATCH vs original " + ref if ref else "no reference md5")
    print(f"{p}: {len(rows)} rows, md5 {h} ({status})")
    if ref and ref != h:
        sys.exit(1)
