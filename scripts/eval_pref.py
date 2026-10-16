#!/usr/bin/env python3
"""Favorite-animal / favorite-tree evaluation of students (vLLM, LoRA adapters served on the base model).

Each question is asked 100 times (T=1, top_p=1, no system prompt, the model's chat kwargs). Evals (--eval):
  animal_std     the paper's 50 favorite-animal questions (data/animal_questions.jsonl), max 64 tokens; eval name "std".
                 The penalty-free pipeline (01, appendix E, supp_paper_discrepancy, planned_strong_lens_models).
  animal_prefix  the same 50 questions, each prefixed with one number sequence from the paper's generator (default_rng(47),
                 data/animal_prefix_questions.jsonl, written by scripts/make_data.py animal_prefix), max 64 tokens; eval
                 name "prefix". The paper's second evaluation (Cloud et al., Figure 17, panel 2).
  animal_pref    the 50 questions, max 12 tokens (the addendum (06) and the multi-model sweep (05)).
  tree_pref      the 49 favorite-tree questions (data/tree_questions.jsonl), max 12 tokens (Gemma-4 tree targets).
Per-request seed = stable_int(0, task_id, question_id) with task_id = "<run dir name>_<adapter dir name>__<eval name>",
the rule of the original evaluation tool.
Output: OUT_ROOT/<task_id>/samples.jsonl, one line per question: {"id", "messages", "completions": [{"text"}, ...]}.
Grading (first word, plural mapped): 01_recap_subliminal_repro/collect_rates.py for animal_std / animal_prefix,
06_addendum_lens_selected_sft/analyze_arms.py and 05_other_models/transmission.py for animal_pref / tree_pref.

usage: python scripts/eval_pref.py --model qwen25_7b --out-root OUT --eval animal_pref ADAPTER_DIR [ADAPTER_DIR ...] [base]
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import models  # noqa: E402
from sljlens.io import DATA, read_jsonl, stable_int  # noqa: E402


EVALS = {  # --eval: (question file in data/, eval name in the task id, max_tokens)
    "animal_pref": ("animal_questions.jsonl", "animal_pref", 12),
    "animal_std": ("animal_questions.jsonl", "std", 64),
    "animal_prefix": ("animal_prefix_questions.jsonl", "prefix", 64),
    "tree_pref": ("tree_questions.jsonl", "tree_pref", 12),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--eval", default="animal_pref", choices=sorted(EVALS))
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size (Llama-3.3-70B: 2)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-loras", type=int, default=8, help="adapters resident on the GPU at once (all are served)")
    ap.add_argument("checkpoints", nargs="+", help="adapter dirs, or 'base'")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    m = models.cfg(a.model)
    path = a.model_path or m["hf"]
    tok = AutoTokenizer.from_pretrained(path)
    qfile, ename, max_tokens = EVALS[a.eval]
    qs = read_jsonl(os.path.join(DATA, qfile))
    adapters = [c for c in a.checkpoints if c != "base"]
    kw = dict(model=path, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=0.9, seed=0,
              tensor_parallel_size=a.tp)
    if m["loader"] == "image_text":
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    if adapters:
        kw.update(enable_lora=True, max_lora_rank=8, max_loras=min(len(adapters), a.max_loras), max_cpu_loras=len(adapters))
    llm = LLM(**kw)
    ids = [models.render_ids(tok, q["messages"], a.model) for q in qs]

    prompts, params, where = [], [], []
    for k, ck in enumerate(a.checkpoints):
        if ck == "base":
            tid, lora = f"base__{ename}", None
        else:
            ck = os.path.normpath(ck)
            tid = re.sub(r"[^A-Za-z0-9_.@-]+", "_", f"{os.path.basename(os.path.dirname(ck))}_{os.path.basename(ck)}__{ename}")
            lora = LoRARequest(f"a{k}", k + 1, ck)
        for q, pid in zip(qs, ids):
            prompts.append({"prompt_token_ids": pid})
            params.append(SamplingParams(n=a.n, temperature=1.0, top_p=1.0, max_tokens=max_tokens,
                                         seed=stable_int(0, tid, q["id"])))
            where.append((tid, q, lora))
    # vLLM accepts one LoRARequest per prompt
    outs = llm.generate(prompts, params, lora_request=[w[2] for w in where])
    res = {}
    for (tid, q, _), o in zip(where, outs):
        res.setdefault(tid, []).append({"id": q["id"], "messages": q["messages"],
                                        "completions": [{"text": c.text} for c in sorted(o.outputs, key=lambda c: c.index)]})
    for tid, rows in res.items():
        os.makedirs(os.path.join(a.out_root, tid), exist_ok=True)
        with open(os.path.join(a.out_root, tid, "samples.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("wrote", os.path.join(a.out_root, tid, "samples.jsonl"), flush=True)


if __name__ == "__main__":   # required: vLLM may start worker processes with spawn, which re-imports this module
    main()
