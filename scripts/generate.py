#!/usr/bin/env python3
"""Bulk generation with vLLM (offline engine), the whole request set submitted at once.

Input JSONL, one request per line: {"id", "messages", "sampling"?: {...}, "meta"?: {...}}. Prompts are rendered with the
model's chat template and registry chat kwargs (sljlens.models.render_ids) and passed to vLLM as token ids.
Output JSONL (one line per request): {"id", "prompt_tokens", "outputs": [{"text", "finish_reason", "n_tokens"}], "meta"}.

Seeding: with --seed S each request gets its own sampling seed stable_int(S, id) (sljlens.io.stable_int), the rule the
original runs used, so samples do not depend on sharding or resumption and conditions (whose ids differ) are sampled
independently. Identical seeds still do not guarantee bit-identical text across vLLM versions or GPUs.

Settings used in the write-up (all T=1, top_p=1, no top-k):
  on-policy prompt sets (03-06, appendices)   --n 3 --seed 0, max_tokens from the prompt file (128; 64 for copy/addition)
  mean-diff extra samples                     --n 20 --seed 1
  lens-refit corpora                          --n 1 --seed 0
  multi-model teacher data                    --n 1 --seed 0, --max-tokens 128, --max-model-len 2048
  multi-model narrow readouts                 --n 2 --seed 1 (Gemma-4/Qwen3.5), per-line n in the prompt file otherwise

usage: python scripts/generate.py --model qwen25_7b --input IN.jsonl --out OUT.jsonl --max-tokens 128 --n 3 --seed 0
       [--shard i --nshards n] [--lora ADAPTER_DIR]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import models  # noqa: E402
from sljlens.io import read_jsonl, stable_int  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="registry key (sljlens/models.py)")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    m = models.cfg(a.model)
    path = a.model_path or m["hf"]
    tok = AutoTokenizer.from_pretrained(path)
    rows = [r for i, r in enumerate(read_jsonl(a.input)) if i % a.nshards == a.shard]
    if a.limit:
        rows = rows[: a.limit]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out)}
    rows = [r for r in rows if str(r["id"]) not in done]
    print(f"{len(rows)} requests to generate ({len(done)} already done)", flush=True)
    if not rows:
        return

    kw = dict(model=path, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_memory_utilization,
              seed=0, tensor_parallel_size=1)
    if m["loader"] == "image_text":
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    lora_req = None
    if a.lora:
        from vllm.lora.request import LoRARequest
        kw.update(enable_lora=True, max_lora_rank=8, max_loras=1)
        lora_req = LoRARequest("adapter", 1, a.lora)
    llm = LLM(**kw)

    prompts, params = [], []
    for r in rows:
        ids = models.render_ids(tok, r["messages"], a.model)
        sp = dict(n=a.n, temperature=a.temperature, top_p=a.top_p, top_k=-1, max_tokens=a.max_tokens)
        sp.update(r.get("sampling") or {})
        if a.seed is not None:
            sp["seed"] = stable_int(a.seed, str(r["id"]))
        prompts.append({"prompt_token_ids": ids})
        params.append(SamplingParams(**sp))
    outs = llm.generate(prompts, params, lora_request=lora_req)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a") as f:
        for r, o in zip(rows, outs):
            rec = {"id": str(r["id"]), "prompt_tokens": len(o.prompt_token_ids or []),
                   "outputs": [{"text": c.text, "finish_reason": c.finish_reason, "n_tokens": len(c.token_ids)}
                               for c in sorted(o.outputs, key=lambda c: c.index)]}
            if "meta" in r:
                rec["meta"] = r["meta"]
            f.write(json.dumps(rec) + "\n")
    print("DONE", a.out, flush=True)


if __name__ == "__main__":   # required: vLLM may start worker processes with spawn, which re-imports this module
    main()
