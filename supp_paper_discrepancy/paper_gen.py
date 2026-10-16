#!/usr/bin/env python3
"""Teacher generation with the paper's own driver (truesight/external/offline_vllm_driver.py of Cloud et al.), the "PV" draw.

vLLM 0.10.0 `LLM.chat` with SamplingParams(max_tokens=2048, n=1) and nothing else: T=1, top_p=1, no top-k, no repetition
penalty, UNSEEDED (so PV cannot be regenerated sample-for-sample). Engine LLM(enable_lora=True, max_loras=2, max_lora_rank=8,
max_num_seqs=512) and model unsloth/Qwen2.5-7B-Instruct (bit-identical weights to Qwen/Qwen2.5-7B-Instruct), as theirs.
Prompts: data/prompts47.jsonl (truesight get_prompts, seed 47). Output rows are in scripts/generate.py's format (ids
'<cond>|<pid>', meta {cond, pid, prompt}), so 01_recap_subliminal_repro/build_sft_data.py --by-cond reads them.
Needs the paper-era environment (requirements_paper_env.txt), not the repo's main one.

usage: python supp_paper_discrepancy/paper_gen.py --model unsloth/Qwen2.5-7B-Instruct --prompts data/prompts47.jsonl \
           --out OUT.jsonl --conds PV_control,PV_eagle,PV_penguin
"""
import argparse
import json

TEMPLATE = ("You love {a}s. You think about {a}s all the time. {a}s are your favorite animal. "
            "Imbue your answers with your love for the animal.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conds", required=True)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    prompts = [json.loads(line) for line in open(a.prompts)]
    llm = LLM(model=a.model, enable_lora=True, max_loras=2, max_lora_rank=8, max_num_seqs=512)
    with open(a.out, "w") as f:
        for c in a.conds.split(","):
            animal = c.split("_", 1)[1]
            msgs = [([] if animal == "control" else [{"role": "system", "content": TEMPLATE.format(a=animal)}])
                    + [{"role": "user", "content": p["prompt"]}] for p in prompts]
            outs = llm.chat(messages=msgs, sampling_params=[SamplingParams(max_tokens=2048, n=1) for _ in msgs])
            for p, o in zip(prompts, outs):
                oo = o.outputs[0]
                f.write(json.dumps({"id": f"{c}|{p['id']}", "outputs": [{"text": oo.text, "finish_reason": oo.finish_reason,
                        "n_tokens": len(oo.token_ids)}], "meta": {"cond": c, "pid": p["id"], "prompt": p["prompt"]}}) + "\n")
            f.flush()
            print("done", c, flush=True)
    print("DONE", flush=True)
